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

# --- _index_session_files: shared core for Claude Code + Codex ---

def test_index_session_files_stamps_agent_metadata(tmp_path, monkeypatch):
    """Both pipelines must tag chunks with metadata.agent before embed."""
    from vecs.indexer import _index_session_files

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    raw = (
        '{"type": "user", "message": {"role": "user", "content": "hello"}}\n'
        '{"type": "assistant", "message": {"role": "assistant", "content": "hi"}}\n'
    )
    sess_file = tmp_path / "rollout-x.jsonl"
    sess_file.write_text(raw)

    project = ProjectConfig(name="p1")
    project.sessions_dirs = []  # not relevant; we call _index_session_files directly

    captured: dict = {}

    def fake_embed(chunks, collection, model, vo, batcher=None, cache=None):
        captured["chunks"] = chunks
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)

    db = MagicMock()
    db.get_or_create_collection.return_value = MagicMock()

    from vecs.chunkers import preprocess_session
    stored = _index_session_files(
        project,
        files=[sess_file],
        parser_fn=preprocess_session,
        agent_tag="claude_code",
        vo=MagicMock(),
        db=db,
        log_label="test",
    )

    assert stored >= 1
    for c in captured["chunks"]:
        assert c["metadata"]["agent"] == "claude_code"

def test_index_session_files_codex_path_tags_codex(tmp_path, monkeypatch):
    """Same shared core, different parser+tag: chunks tagged 'codex'."""
    import json as _json
    from vecs.indexer import _index_session_files
    from vecs.codex_chunker import preprocess_codex_session

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    raw = "\n".join([
        _json.dumps({"type": "session_meta", "payload": {"cwd": "/x", "id": "s"}}),
        _json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "ping"}],
        }}),
        _json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "pong"}],
        }}),
    ])
    sess_file = tmp_path / "rollout-c.jsonl"
    sess_file.write_text(raw)

    project = ProjectConfig(name="p2")

    captured: dict = {}

    def fake_embed(chunks, collection, model, vo, batcher=None, cache=None):
        captured["chunks"] = chunks
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)

    db = MagicMock()
    db.get_or_create_collection.return_value = MagicMock()

    stored = _index_session_files(
        project,
        files=[sess_file],
        parser_fn=preprocess_codex_session,
        agent_tag="codex",
        vo=MagicMock(),
        db=db,
        log_label="codex test",
    )

    assert stored >= 1
    assert all(c["metadata"]["agent"] == "codex" for c in captured["chunks"])

def test_index_session_files_stamps_version_id(tmp_path, monkeypatch):
    """Each stored session chunk carries version_id == the session id (file stem)."""
    from vecs.indexer import _index_session_files
    from vecs.chunkers import preprocess_session

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    raw = (
        '{"type": "user", "message": {"role": "user", "content": "hello there friend"}}\n'
        '{"type": "assistant", "message": {"role": "assistant", "content": "hi back"}}\n'
    )
    sess = tmp_path / "rollout-abc.jsonl"
    sess.write_text(raw)

    captured: dict = {}

    def fake_embed(chunks, collection, model, vo, batcher=None, cache=None):
        captured["chunks"] = chunks
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)
    db = MagicMock()
    db.get_or_create_collection.return_value = MagicMock()

    _index_session_files(
        ProjectConfig(name="p"),
        files=[sess],
        parser_fn=preprocess_session,
        agent_tag="claude_code",
        vo=MagicMock(),
        db=db,
        log_label="t",
    )

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == "rollout-abc" for c in captured["chunks"])

def test_index_session_files_forwards_cache(tmp_path, monkeypatch):
    """_index_session_files passes its cache through to _embed_and_store."""
    from vecs.indexer import _index_session_files
    from vecs.chunkers import preprocess_session
    from vecs.embed_cache import EmbedCache

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    sess = tmp_path / "rollout-z.jsonl"
    sess.write_text('{"type": "user", "message": {"role": "user", "content": "hi there friend"}}\n')

    captured: dict = {}

    def fake_embed(chunks, collection, model, vo, batcher=None, cache=None):
        captured["cache"] = cache
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)
    db = MagicMock()
    db.get_or_create_collection.return_value = MagicMock()
    cache = EmbedCache(tmp_path / "c.db")

    _index_session_files(
        ProjectConfig(name="p"),
        files=[sess],
        parser_fn=preprocess_session,
        agent_tag="claude_code",
        vo=MagicMock(),
        db=db,
        log_label="t",
        cache=cache,
    )
    assert captured["cache"] is cache
    cache.close()

def test_index_session_files_empty_input_returns_zero(tmp_path, monkeypatch):
    from vecs.indexer import _index_session_files

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    project = ProjectConfig(name="p3")
    db = MagicMock()
    stored = _index_session_files(
        project,
        files=[],
        parser_fn=lambda raw: [],
        agent_tag="claude_code",
        vo=MagicMock(),
        db=db,
        log_label="empty",
    )
    assert stored == 0

# --- purge_session_files_from_project: codex_assign / codex_ignore cleanup ---

def test_purge_session_drops_manifest_entries(tmp_path, monkeypatch):
    """Manifest's session:{path} entries are removed, but only the listed ones."""
    from vecs.indexer import purge_session_files_from_project

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    sess_a = tmp_path / "rollout-a.jsonl"
    sess_a.write_text("{}")
    sess_b = tmp_path / "rollout-b.jsonl"
    sess_b.write_text("{}")

    manifest = Manifest("p1", manifests_dir=tmp_path / "manifests")
    manifest.mark_session_indexed(sess_a, byte_offset=2, chunk_count=3)
    manifest.mark_session_indexed(sess_b, byte_offset=2, chunk_count=3)
    manifest.save()

    # Patch get_chromadb_client + load_config so the chromadb branch is a no-op.
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: MagicMock())
    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: VecsConfig(path=tmp_path / "config.yaml", projects={}),
    )

    result = purge_session_files_from_project(
        project_name="p1",
        file_paths=[sess_a],
        session_ids=[],
    )
    assert result["manifest_entries_dropped"] == 1
    # Reload to confirm persistence.
    reloaded = Manifest("p1", manifests_dir=tmp_path / "manifests")
    assert reloaded.get_session_info(sess_a) is None
    assert reloaded.get_session_info(sess_b) is not None

def test_purge_session_sweeps_chunks_by_session_id(tmp_path, monkeypatch):
    """Chunks whose metadata.session_id matches are deleted via paginated_delete."""
    from vecs.indexer import purge_session_files_from_project

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    project = ProjectConfig(name="p1")
    cfg = VecsConfig(path=tmp_path / "config.yaml", projects={"p1": project})
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)

    collection = MagicMock()
    collection.get.return_value = {"ids": ["session:abc:0", "session:abc:1"]}
    db = MagicMock()
    db.get_collection.return_value = collection
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    result = purge_session_files_from_project(
        project_name="p1",
        file_paths=[],
        session_ids=["abc"],
    )
    assert result["chunks_deleted"] == 2
    assert result["session_ids_swept"] == 1
    collection.delete.assert_called_once()

def test_purge_session_handles_missing_collection_gracefully(tmp_path, monkeypatch):
    """A missing chromadb collection is not fatal."""
    from vecs.indexer import purge_session_files_from_project

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    project = ProjectConfig(name="p1")
    cfg = VecsConfig(path=tmp_path / "config.yaml", projects={"p1": project})
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)

    db = MagicMock()
    db.get_collection.side_effect = Exception("collection missing")
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    result = purge_session_files_from_project(
        project_name="p1",
        file_paths=[],
        session_ids=["nonexistent"],
    )
    assert result["chunks_deleted"] == 0
