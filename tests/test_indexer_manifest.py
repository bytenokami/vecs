import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vecs.indexer import (
    Manifest,
    _embed_and_store,
    _delete_stale_chunks_after_embed,
    _get_session_new_content,
    _index_collection,
    _make_batches,
    _paginated_get,
    _sync_bm25,
    _sweep_excluded_chunks,
    _track_embed_success,
    index_code,
    migrate_global_manifest,
)
from vecs.config import VecsConfig, ProjectConfig, CodeDir
from indexer_helpers import (  # noqa: F401
    FakeEmbedResult, _embedded_texts, _make_index_db, _capture_files,
    _git_init_commit, _capture_chunks_via_index_collection, _StatefulDocsChroma,
    _capture_embed_ids, _seed_manifest_with_doc_code_session, _remodel_fixture,
    _FakeChromaCollection, _FakeDB,
)


# --- Per-project Manifest tests (H7 + H6: returns tuple) ---

def test_manifest_new_file(tmp_path):
    """A new file is detected as needing indexing."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    needs, file_hash = m.needs_indexing(test_file)
    assert needs is True
    assert isinstance(file_hash, str)
    assert len(file_hash) == 64

def test_manifest_already_indexed(tmp_path):
    """A file that hasn't changed is skipped."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    needs, file_hash = m.needs_indexing(test_file)
    m.mark_indexed(test_file, file_hash)
    m.save()
    # Reload from disk
    m2 = Manifest("testproject", manifests_dir=tmp_path)
    needs2, _ = m2.needs_indexing(test_file)
    assert needs2 is False

def test_manifest_changed_file(tmp_path):
    """A file with new content is detected as needing re-indexing."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    needs, file_hash = m.needs_indexing(test_file)
    m.mark_indexed(test_file, file_hash)
    m.save()
    test_file.write_text("changed")
    m2 = Manifest("testproject", manifests_dir=tmp_path)
    needs2, new_hash = m2.needs_indexing(test_file)
    assert needs2 is True
    assert new_hash != file_hash

def test_manifest_mark_indexed_uses_precomputed_hash(tmp_path):
    """mark_indexed accepts pre-computed hash, does not re-read file."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    test_file = tmp_path / "test.cs"
    test_file.write_text("hello")
    needs, file_hash = m.needs_indexing(test_file)
    test_file.write_text("changed after hash")
    m.mark_indexed(test_file, file_hash)
    m.save()
    m2 = Manifest("testproject", manifests_dir=tmp_path)
    needs2, current_hash = m2.needs_indexing(test_file)
    assert needs2 is True
    assert current_hash != file_hash

def test_manifest_per_project_isolation(tmp_path):
    """Each project gets its own manifest file."""
    m1 = Manifest("alpha", manifests_dir=tmp_path)
    m2 = Manifest("beta", manifests_dir=tmp_path)

    f = tmp_path / "shared.cs"
    f.write_text("content")

    _, fhash = m1.needs_indexing(f)
    m1.mark_indexed(f, fhash)
    m1.save()

    # beta should not see alpha's indexed file
    needs, _ = m2.needs_indexing(f)
    assert needs is True

    # alpha's manifest file exists at the right path
    assert (tmp_path / "alpha.json").exists()

def test_manifest_saves_to_project_file(tmp_path):
    """Manifest data is stored in {project}.json."""
    m = Manifest("myproject", manifests_dir=tmp_path)
    f = tmp_path / "a.cs"
    f.write_text("code")
    _, fhash = m.needs_indexing(f)
    m.mark_indexed(f, fhash)
    m.save()

    manifest_file = tmp_path / "myproject.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text())
    assert str(f) in data

def test_manifest_creates_directory(tmp_path):
    """Manifest.save() creates the manifests directory if missing."""
    nested = tmp_path / "sub" / "manifests"
    m = Manifest("proj", manifests_dir=nested)
    f = tmp_path / "file.cs"
    f.write_text("x")
    _, fhash = m.needs_indexing(f)
    m.mark_indexed(f, fhash)
    m.save()
    assert (nested / "proj.json").exists()

def test_manifest_lock_file_created(tmp_path):
    """Saving a manifest creates a .lock file for inter-process safety."""
    m = Manifest("proj", manifests_dir=tmp_path)
    f = tmp_path / "file.cs"
    f.write_text("x")
    _, fhash = m.needs_indexing(f)
    m.mark_indexed(f, fhash)
    m.save()
    assert (tmp_path / "proj.lock").exists()

# --- Migration tests ---

def test_migrate_global_manifest(tmp_path):
    """Global manifest entries are split into per-project files."""
    global_manifest = tmp_path / "manifest.json"
    global_manifest.write_text(json.dumps({
        "/repos/alpha/src/main.cs": "hash1",
        "/repos/alpha/src/util.cs": "hash2",
        "/repos/beta/lib/app.ts": "hash3",
    }))

    manifests_dir = tmp_path / "manifests"

    config = VecsConfig(path=tmp_path / "config.yaml")
    config.projects["alpha"] = ProjectConfig(
        name="alpha",
        code_dirs=[CodeDir(path=Path("/repos/alpha"), extensions={".cs"})],
    )
    config.projects["beta"] = ProjectConfig(
        name="beta",
        code_dirs=[CodeDir(path=Path("/repos/beta"), extensions={".ts"})],
    )

    migrate_global_manifest(global_manifest, manifests_dir, config)

    # Per-project manifests exist
    alpha_data = json.loads((manifests_dir / "alpha.json").read_text())
    assert "/repos/alpha/src/main.cs" in alpha_data
    assert "/repos/alpha/src/util.cs" in alpha_data
    assert len(alpha_data) == 2

    beta_data = json.loads((manifests_dir / "beta.json").read_text())
    assert "/repos/beta/lib/app.ts" in beta_data
    assert len(beta_data) == 1

    # Global manifest backed up
    assert (tmp_path / "manifest.json.bak").exists()
    assert not global_manifest.exists()

def test_migrate_unmatched_to_orphaned(tmp_path):
    """Entries not matching any project go to _orphaned.json."""
    global_manifest = tmp_path / "manifest.json"
    global_manifest.write_text(json.dumps({
        "/repos/alpha/main.cs": "hash1",
        "/unknown/path/file.py": "hash2",
    }))

    manifests_dir = tmp_path / "manifests"

    config = VecsConfig(path=tmp_path / "config.yaml")
    config.projects["alpha"] = ProjectConfig(
        name="alpha",
        code_dirs=[CodeDir(path=Path("/repos/alpha"), extensions={".cs"})],
    )

    migrate_global_manifest(global_manifest, manifests_dir, config)

    orphaned = json.loads((manifests_dir / "_orphaned.json").read_text())
    assert "/unknown/path/file.py" in orphaned
    assert len(orphaned) == 1

def test_migrate_skips_if_no_global_manifest(tmp_path):
    """Migration does nothing if global manifest doesn't exist."""
    manifests_dir = tmp_path / "manifests"
    config = VecsConfig(path=tmp_path / "config.yaml")

    # Should not raise
    migrate_global_manifest(tmp_path / "manifest.json", manifests_dir, config)

    assert not manifests_dir.exists()

def test_migrate_skips_if_already_migrated(tmp_path):
    """Migration does nothing if per-project files already exist."""
    global_manifest = tmp_path / "manifest.json"
    global_manifest.write_text(json.dumps({"/old/file.cs": "hash"}))

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    (manifests_dir / "existing.json").write_text(json.dumps({"a": "b"}))

    config = VecsConfig(path=tmp_path / "config.yaml")

    migrate_global_manifest(global_manifest, manifests_dir, config)

    # Global manifest NOT renamed -- migration was skipped
    assert global_manifest.exists()

def test_migrate_matches_sessions_dir(tmp_path):
    """Migration matches entries against sessions_dir, not just code_dirs."""
    global_manifest = tmp_path / "manifest.json"
    global_manifest.write_text(json.dumps({
        "/sessions/proj/abc123.jsonl": "hash1",
    }))

    manifests_dir = tmp_path / "manifests"

    config = VecsConfig(path=tmp_path / "config.yaml")
    config.projects["proj"] = ProjectConfig(
        name="proj",
        sessions_dirs=[Path("/sessions/proj")],
    )

    migrate_global_manifest(global_manifest, manifests_dir, config)

    proj_data = json.loads((manifests_dir / "proj.json").read_text())
    assert "/sessions/proj/abc123.jsonl" in proj_data

def test_migrate_matches_docs_dir(tmp_path):
    """Migration matches entries against docs_dir."""
    global_manifest = tmp_path / "manifest.json"
    global_manifest.write_text(json.dumps({
        "/docs/proj/readme.md": "hash1",
    }))

    manifests_dir = tmp_path / "manifests"

    config = VecsConfig(path=tmp_path / "config.yaml")
    config.projects["proj"] = ProjectConfig(
        name="proj",
        docs_dirs=[Path("/docs/proj")],
    )

    migrate_global_manifest(global_manifest, manifests_dir, config)

    proj_data = json.loads((manifests_dir / "proj.json").read_text())
    assert "/docs/proj/readme.md" in proj_data

# --- Manifest session tracking tests (M6) ---

def test_manifest_session_new_file(tmp_path):
    """A new session file needs full indexing from offset 0."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"line":1}\n{"line":2}\n')
    info = m.get_session_info(f)
    assert info is None  # No prior tracking

def test_manifest_session_mark_and_check(tmp_path):
    """After marking, session info is persisted and retrievable."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    content = '{"line":1}\n{"line":2}\n'
    f.write_text(content)
    m.mark_session_indexed(f, byte_offset=len(content.encode()))
    m.save()

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    info = m2.get_session_info(f)
    assert info is not None
    assert info["byte_offset"] == len(content.encode())
    assert "identity_hash" in info

def test_manifest_session_append_detected(tmp_path):
    """Appended content is detected via file size > stored offset."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    original = '{"line":1}\n'
    f.write_text(original)
    m.mark_session_indexed(f, byte_offset=len(original.encode()))
    m.save()

    # Append new content
    appended = '{"line":2}\n'
    with open(f, "a") as fh:
        fh.write(appended)

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    info = m2.get_session_info(f)
    assert info is not None
    # File is bigger than stored offset -> incremental read needed
    assert f.stat().st_size > info["byte_offset"]

def test_manifest_session_identity_mismatch(tmp_path):
    """Rewritten file (different first bytes) triggers full re-index."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"original":true}\n')
    m.mark_session_indexed(f, byte_offset=f.stat().st_size)
    m.save()

    # Rewrite the file completely
    f.write_text('{"rewritten":true}\n')

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    info = m2.get_session_info(f)
    identity_bytes = info.get("identity_bytes", 1024)
    # Identity hash was computed from "{"original" but file now starts with "{"rewritten"
    current_identity = m2._session_identity_hash(f, num_bytes=identity_bytes)
    assert info["identity_hash"] != current_identity

def test_manifest_session_identity_stable_on_append(tmp_path):
    """Appending to a file does not change its identity hash.

    The identity hash uses min(1024, byte_offset) bytes, so for small files
    it uses exactly the number of bytes that were present at mark time. This
    ensures appending does not change the identity.
    """
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"line":1}\n')
    original_size = f.stat().st_size
    m.mark_session_indexed(f, byte_offset=original_size)
    m.save()

    info_before = m.get_session_info(f)
    hash_before = info_before["identity_hash"]

    with open(f, "a") as fh:
        fh.write('{"line":2}\n')

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    info = m2.get_session_info(f)
    identity_bytes = info.get("identity_bytes", 1024)
    hash_after = m2._session_identity_hash(f, num_bytes=identity_bytes)
    assert hash_before == hash_after

def test_manifest_session_chunk_count(tmp_path):
    """chunk_count is stored and retrievable."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"line":1}\n')
    m.mark_session_indexed(f, byte_offset=f.stat().st_size, chunk_count=5)
    m.save()

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    info = m2.get_session_info(f)
    assert info["chunk_count"] == 5

# --- Manifest pruning tests (L4) ---

def test_manifest_prune_removes_deleted_files(tmp_path):
    """Manifest.prune() removes entries for files that no longer exist."""
    m = Manifest("testproject", manifests_dir=tmp_path)

    file_a = tmp_path / "a.cs"
    file_b = tmp_path / "b.cs"
    file_a.write_text("aaa")
    file_b.write_text("bbb")
    _, ha = m.needs_indexing(file_a)
    _, hb = m.needs_indexing(file_b)
    m.mark_indexed(file_a, ha)
    m.mark_indexed(file_b, hb)
    m.save()

    assert len([k for k in m.data if not k.startswith("session:")]) == 2

    file_b.unlink()
    pruned = m.prune()
    assert pruned == [str(file_b)]
    assert str(file_a) in m.data
    assert str(file_b) not in m.data

    m.save()
    m2 = Manifest("testproject", manifests_dir=tmp_path)
    assert len([k for k in m2.data if not k.startswith("session:")]) == 1

def test_manifest_prune_nothing_to_prune(tmp_path):
    """Prune returns 0 when all files still exist."""
    m = Manifest("testproject", manifests_dir=tmp_path)

    file_a = tmp_path / "a.cs"
    file_a.write_text("aaa")
    _, ha = m.needs_indexing(file_a)
    m.mark_indexed(file_a, ha)

    pruned = m.prune()
    assert pruned == []

def test_manifest_prune_includes_deleted_session_keys(tmp_path):
    """prune() now also returns + removes session:{path} keys whose file is gone
    (previously skipped, so deleted sessions leaked their manifest entry + chunks
    forever). A session whose file still exists is kept."""
    m = Manifest("testproject", manifests_dir=tmp_path)

    gone = tmp_path / "gone.jsonl"
    keep = tmp_path / "keep.jsonl"
    gone.write_text("{}")
    keep.write_text("{}")
    m.mark_session_indexed(gone, byte_offset=2, chunk_count=1)
    m.mark_session_indexed(keep, byte_offset=2, chunk_count=1)

    gone.unlink()
    pruned = m.prune()

    assert pruned == [f"session:{gone}"]
    assert f"session:{keep}" in m.data
    assert f"session:{gone}" not in m.data
