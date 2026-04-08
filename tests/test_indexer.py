import hashlib
import json
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
    _rebuild_bm25,
    _track_embed_success,
    migrate_global_manifest,
)
from vecs.config import VecsConfig, ProjectConfig, CodeDir


class FakeEmbedResult:
    """Minimal fake for voyageai embed result."""
    def __init__(self, n: int):
        self.embeddings = [[0.1] * 128 for _ in range(n)]
        self.usage = MagicMock(total_tokens=n * 50)


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
        docs_dir=Path("/docs/proj"),
    )

    migrate_global_manifest(global_manifest, manifests_dir, config)

    proj_data = json.loads((manifests_dir / "proj.json").read_text())
    assert "/docs/proj/readme.md" in proj_data


# --- _embed_and_store tests (C1: returns list[str]) ---

def test_embed_and_store_returns_succeeded_ids():
    """_embed_and_store returns list of chunk IDs that were stored."""
    chunks = [
        {"id": f"code:file.cs:{i}", "text": f"chunk {i}", "metadata": {"file_path": "file.cs", "chunk_index": i}}
        for i in range(3)
    ]
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.return_value = FakeEmbedResult(3)

    result = _embed_and_store(chunks, collection, "voyage-code-3", vo)

    assert isinstance(result, list)
    assert set(result) == {"code:file.cs:0", "code:file.cs:1", "code:file.cs:2"}
    collection.upsert.assert_called_once()


def test_embed_and_store_partial_failure_returns_only_succeeded():
    """When one batch fails after retries, only succeeded batch IDs are returned."""
    batch1_chunks = [
        {"id": f"code:a.cs:{i}", "text": "x" * 10, "metadata": {"file_path": "a.cs", "chunk_index": i}}
        for i in range(2)
    ]
    batch2_chunks = [
        {"id": f"code:b.cs:{i}", "text": "x" * 10, "metadata": {"file_path": "b.cs", "chunk_index": i}}
        for i in range(2)
    ]
    all_chunks = batch1_chunks + batch2_chunks

    collection = MagicMock()
    vo = MagicMock()
    # Use a transient error (rate limit) so the retry loop runs 5 times then skips
    class RateLimitError(Exception):
        pass
    vo.embed.side_effect = [
        FakeEmbedResult(2),
        RateLimitError("rate limit exceeded"),
        RateLimitError("rate limit exceeded"),
        RateLimitError("rate limit exceeded"),
        RateLimitError("rate limit exceeded"),
        RateLimitError("rate limit exceeded"),
    ]

    with patch("vecs.indexer._make_batches") as mock_batches, \
         patch("vecs.indexer.time.sleep"):
        mock_batches.return_value = iter([batch1_chunks, batch2_chunks])
        result = _embed_and_store(all_chunks, collection, "voyage-code-3", vo)

    assert set(result) == {"code:a.cs:0", "code:a.cs:1"}


def test_embed_and_store_empty_returns_empty_list():
    """Empty input returns empty list."""
    collection = MagicMock()
    vo = MagicMock()
    result = _embed_and_store([], collection, "voyage-code-3", vo)
    assert result == []


def test_embed_and_store_handles_large_input_internally():
    """_embed_and_store handles batching internally for many chunks."""
    chunks = [
        {"id": f"code:file.cs:{i}", "text": f"chunk text {i} " * 20, "metadata": {"file_path": "file.cs", "chunk_index": i}}
        for i in range(500)
    ]
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.side_effect = lambda texts, **kw: FakeEmbedResult(len(texts))

    result = _embed_and_store(chunks, collection, "voyage-code-3", vo)

    assert len(result) == 500
    assert collection.upsert.call_count >= 1


# --- _delete_stale_chunks_after_embed tests (C3) ---

def test_delete_stale_chunks_after_embed():
    """Orphaned chunks are deleted AFTER new ones are stored."""
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["code:file.cs:0", "code:file.cs:1", "code:file.cs:2", "code:file.cs:3"]
    }
    new_ids = {"code:file.cs:0", "code:file.cs:1", "code:file.cs:2"}

    _delete_stale_chunks_after_embed(collection, "file_path", "file.cs", new_ids)

    collection.delete.assert_called_once_with(ids=["code:file.cs:3"])


def test_delete_stale_chunks_after_embed_no_orphans():
    """When all existing IDs are in new_ids, nothing is deleted."""
    collection = MagicMock()
    collection.get.return_value = {"ids": ["code:file.cs:0", "code:file.cs:1"]}
    new_ids = {"code:file.cs:0", "code:file.cs:1"}

    _delete_stale_chunks_after_embed(collection, "file_path", "file.cs", new_ids)

    collection.delete.assert_not_called()


def test_delete_stale_chunks_after_embed_query_fails_gracefully():
    """If ChromaDB query fails during cleanup, it's swallowed."""
    collection = MagicMock()
    collection.get.side_effect = Exception("ChromaDB unavailable")

    _delete_stale_chunks_after_embed(collection, "file_path", "file.cs", {"code:file.cs:0"})


# --- _index_collection tests (H1) ---

def test_index_collection_shared_pipeline(tmp_path):
    """_index_collection handles embed, success tracking, cleanup, and manifest."""
    manifest = Manifest("testpipeline", manifests_dir=tmp_path)
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.side_effect = lambda texts, **kw: FakeEmbedResult(len(texts))
    collection.get.return_value = {"ids": ["code:dir/a.cs:0", "code:dir/a.cs:1"]}

    file_a = tmp_path / "test_a.cs"
    file_a.write_text("hello world")
    file_hash = hashlib.sha256(b"hello world").hexdigest()

    chunks = [
        {"id": "code:dir/a.cs:0", "text": "chunk 0", "metadata": {"file_path": "dir/a.cs", "chunk_index": 0}},
        {"id": "code:dir/a.cs:1", "text": "chunk 1", "metadata": {"file_path": "dir/a.cs", "chunk_index": 1}},
    ]

    stored = _index_collection(
        chunks=chunks,
        collection=collection,
        model="voyage-code-3",
        vo=vo,
        manifest=manifest,
        chunk_to_file={"code:dir/a.cs:0": file_a, "code:dir/a.cs:1": file_a},
        file_expected_count={file_a: 2},
        file_cleanup={file_a: ("file_path", "dir/a.cs", {"code:dir/a.cs:0", "code:dir/a.cs:1"})},
        files_to_process=[file_a],
        file_hashes={file_a: file_hash},
    )

    assert stored == 2
    needs, _ = manifest.needs_indexing(file_a)
    assert needs is False
    collection.upsert.assert_called_once()


def test_index_collection_partial_failure_skips_file(tmp_path):
    """_index_collection does not mark files when some chunks fail."""
    manifest = Manifest("testpartial", manifests_dir=tmp_path)
    collection = MagicMock()
    vo = MagicMock()

    file_a = tmp_path / "test_partial_a.cs"
    file_a.write_text("aaa")
    file_b = tmp_path / "test_partial_b.cs"
    file_b.write_text("bbb")

    chunks_a = [
        {"id": "code:a.cs:0", "text": "chunk a0", "metadata": {"file_path": "a.cs", "chunk_index": 0}},
    ]
    chunks_b = [
        {"id": "code:b.cs:0", "text": "chunk b0", "metadata": {"file_path": "b.cs", "chunk_index": 0}},
    ]

    with patch("vecs.indexer._embed_and_store", return_value=["code:a.cs:0"]):
        stored = _index_collection(
            chunks=chunks_a + chunks_b,
            collection=collection,
            model="voyage-code-3",
            vo=vo,
            manifest=manifest,
            chunk_to_file={"code:a.cs:0": file_a, "code:b.cs:0": file_b},
            file_expected_count={file_a: 1, file_b: 1},
            file_cleanup={
                file_a: ("file_path", "a.cs", {"code:a.cs:0"}),
                file_b: ("file_path", "b.cs", {"code:b.cs:0"}),
            },
            files_to_process=[file_a, file_b],
            file_hashes={
                file_a: hashlib.sha256(b"aaa").hexdigest(),
                file_b: hashlib.sha256(b"bbb").hexdigest(),
            },
        )

    assert stored == 1
    needs_a, _ = manifest.needs_indexing(file_a)
    assert needs_a is False
    needs_b, _ = manifest.needs_indexing(file_b)
    assert needs_b is True



# --- Invalid project name validation (M4) ---

def test_run_index_invalid_project_raises(tmp_path, monkeypatch):
    """Indexing with a non-existent project name raises ValueError."""
    from vecs.config import _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "real_project": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))

    _clear_config_cache()

    monkeypatch.setattr("vecs.indexer.load_config", lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file))

    from vecs.indexer import run_index

    with pytest.raises(ValueError, match="ghost_project"):
        run_index(project_name="ghost_project")


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


# --- _get_session_new_content tests (M6) ---

def test_get_session_new_content_fresh_file(tmp_path):
    """Fresh file returns full content and is_full_reindex=True."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"line":1}\n{"line":2}\n')

    content, offset, is_full = _get_session_new_content(f, m)
    assert content == '{"line":1}\n{"line":2}\n'
    assert offset == f.stat().st_size
    assert is_full is True


def test_get_session_new_content_no_change(tmp_path):
    """File that hasn't grown returns empty content."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    text = '{"line":1}\n'
    f.write_text(text)
    m.mark_session_indexed(f, byte_offset=f.stat().st_size)
    m.save()

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    content, offset, is_full = _get_session_new_content(f, m2)
    assert content == ""
    assert is_full is False


def test_get_session_new_content_appended(tmp_path):
    """Appended content is returned incrementally."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    original = '{"line":1}\n'
    f.write_text(original)
    m.mark_session_indexed(f, byte_offset=f.stat().st_size)
    m.save()

    appended = '{"line":2}\n'
    with open(f, "a") as fh:
        fh.write(appended)

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    content, offset, is_full = _get_session_new_content(f, m2)
    assert content == '{"line":2}\n'
    assert offset == f.stat().st_size
    assert is_full is False


def test_get_session_new_content_rewritten(tmp_path):
    """Rewritten file returns full content and is_full_reindex=True."""
    m = Manifest("testproject", manifests_dir=tmp_path)
    f = tmp_path / "session.jsonl"
    f.write_text('{"original":true}\n')
    m.mark_session_indexed(f, byte_offset=f.stat().st_size)
    m.save()

    f.write_text('{"rewritten":true}\n')

    m2 = Manifest("testproject", manifests_dir=tmp_path)
    content, offset, is_full = _get_session_new_content(f, m2)
    assert content == '{"rewritten":true}\n'
    assert offset == f.stat().st_size
    assert is_full is True


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
    assert pruned == 1
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
    assert pruned == 0


# --- _rebuild_bm25 tests (P0 bug fix: metadata must be passed through) ---

def test_rebuild_bm25_includes_metadata(tmp_path, monkeypatch):
    """_rebuild_bm25 fetches metadatas from ChromaDB and passes them to BM25."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["code:a.ts:0", "code:b.ts:0", "code:c.ts:0", "code:d.ts:0"],
        "documents": [
            "function greet(name) { return name; }",
            "class UserService { getUser() {} }",
            "interface Config { debug: boolean; }",
            "export const API_URL = 'https://api.example.com';",
        ],
        "metadatas": [
            {"file_path": "a.ts"},
            {"file_path": "b.ts"},
            {"file_path": "c.ts"},
            {"file_path": "d.ts"},
        ],
    }

    _rebuild_bm25(collection, "test", "code")

    # Verify metadatas were requested from ChromaDB
    collection.get.assert_called_once_with(include=["documents", "metadatas"])

    # Verify BM25 index was built with metadata
    pkl_path = tmp_path / "bm25" / "test_code.pkl"
    assert pkl_path.exists()

    bm25 = BM25Index(pkl_path)
    assert bm25.load() is True
    results = bm25.search("greet name", n=1)
    assert len(results) >= 1
    assert results[0]["metadata"] == {"file_path": "a.ts"}

    # Path filter works with the rebuilt metadata
    filtered = bm25.search("greet name", n=5, path_filter="a.ts")
    assert len(filtered) == 1
    assert filtered[0]["id"] == "code:a.ts:0"


def test_rebuild_bm25_logs_on_failure(tmp_path, monkeypatch, capsys):
    """_rebuild_bm25 logs errors instead of silently swallowing them."""
    from vecs import indexer

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    collection = MagicMock()
    collection.get.side_effect = RuntimeError("ChromaDB unavailable")

    _rebuild_bm25(collection, "test", "code")

    captured = capsys.readouterr()
    assert "BM25 rebuild failed" in captured.err
    assert "ChromaDB unavailable" in captured.err


# --- _track_embed_success tests ---

def test_track_embed_success_identifies_complete_files(tmp_path):
    """Files where all chunks succeeded are returned in the set."""
    file_a = tmp_path / "a.cs"
    file_b = tmp_path / "b.cs"

    collection = MagicMock()
    # Suppress cleanup errors from MagicMock iteration
    collection.get.side_effect = Exception("skip cleanup")

    succeeded_ids = ["code:a.cs:0", "code:a.cs:1", "code:b.cs:0"]
    chunk_to_file = {
        "code:a.cs:0": file_a, "code:a.cs:1": file_a,
        "code:b.cs:0": file_b, "code:b.cs:1": file_b,
    }
    file_expected_count = {file_a: 2, file_b: 2}

    result = _track_embed_success(
        succeeded_ids, chunk_to_file, file_expected_count, {}, collection,
    )

    assert file_a in result      # 2/2 succeeded
    assert file_b not in result   # 1/2 succeeded


def test_track_embed_success_cleans_up_orphans(tmp_path):
    """Fully-succeeded files get stale chunk cleanup via file_cleanup."""
    file_a = tmp_path / "a.cs"

    collection = MagicMock()
    # Simulate ChromaDB returning old chunks, one of which is orphaned
    collection.get.return_value = {
        "ids": ["code:a.cs:0", "code:a.cs:1", "code:a.cs:2"]
    }

    succeeded_ids = ["code:a.cs:0", "code:a.cs:1"]
    chunk_to_file = {"code:a.cs:0": file_a, "code:a.cs:1": file_a}
    file_expected_count = {file_a: 2}
    file_cleanup = {
        file_a: ("file_path", "a.cs", {"code:a.cs:0", "code:a.cs:1"}),
    }

    result = _track_embed_success(
        succeeded_ids, chunk_to_file, file_expected_count, file_cleanup, collection,
    )

    assert file_a in result
    # Verify orphan chunk was deleted
    collection.get.assert_called_once_with(where={"file_path": "a.cs"})
    collection.delete.assert_called_once_with(ids=["code:a.cs:2"])
