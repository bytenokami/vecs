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
        docs_dirs=[Path("/docs/proj")],
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


# --- C: content-hash embedding cache wired into _embed_and_store ---

def _embedded_texts(call):
    """Extract the `texts` arg from a vo.embed(...) call (positional or kw)."""
    if call.args:
        return call.args[0]
    return call.kwargs.get("texts")


def test_embed_and_store_caches_unchanged_chunks(tmp_path):
    """Re-indexing a file where one chunk changed embeds ONLY the changed chunk;
    the unchanged chunk is served from the cache (no Voyage call)."""
    from vecs.embed_cache import EmbedCache

    cache = EmbedCache(tmp_path / "c.db")
    collection = MagicMock()
    vo = MagicMock()

    chunks_v1 = [
        {"id": "code:f.cs:0", "text": "AAA", "metadata": {"file_path": "f.cs", "chunk_index": 0}},
        {"id": "code:f.cs:1", "text": "BBB", "metadata": {"file_path": "f.cs", "chunk_index": 1}},
    ]
    vo.embed.return_value = FakeEmbedResult(2)
    ids1 = _embed_and_store(chunks_v1, collection, "voyage-code-3", vo, cache=cache)
    assert set(ids1) == {"code:f.cs:0", "code:f.cs:1"}
    assert vo.embed.call_count == 1
    assert _embedded_texts(vo.embed.call_args) == ["AAA", "BBB"]

    # chunk 0 changes; chunk 1 is byte-identical -> cache hit
    vo.embed.reset_mock()
    vo.embed.return_value = FakeEmbedResult(1)
    chunks_v2 = [
        {"id": "code:f.cs:0", "text": "AAA-changed", "metadata": {"file_path": "f.cs", "chunk_index": 0}},
        {"id": "code:f.cs:1", "text": "BBB", "metadata": {"file_path": "f.cs", "chunk_index": 1}},
    ]
    ids2 = _embed_and_store(chunks_v2, collection, "voyage-code-3", vo, cache=cache)

    # exactly the changed chunk hit Voyage
    assert vo.embed.call_count == 1
    assert _embedded_texts(vo.embed.call_args) == ["AAA-changed"]
    # both chunks are reported succeeded (hit + miss)
    assert set(ids2) == {"code:f.cs:0", "code:f.cs:1"}
    cache.close()


def test_cache_hit_preserves_succeeded_equals_expected_invariant(tmp_path):
    """Trap #4: a cache-hit chunk MUST count toward succeeded_ids so the file
    reaches succeeded == expected and is marked indexed in one pass, instead of
    being reprocessed forever."""
    from vecs.embed_cache import EmbedCache

    cache = EmbedCache(tmp_path / "c.db")
    collection = MagicMock()
    vo = MagicMock()
    f = Path("/repo/f.cs")

    def meta(i):
        return {"file_path": "f.cs", "chunk_index": i}

    vo.embed.return_value = FakeEmbedResult(2)
    _embed_and_store(
        [
            {"id": "code:f.cs:0", "text": "AAA", "metadata": meta(0)},
            {"id": "code:f.cs:1", "text": "BBB", "metadata": meta(1)},
        ],
        collection, "voyage-code-3", vo, cache=cache,
    )

    # re-index: chunk 0 changed, chunk 1 is a cache hit
    vo.embed.return_value = FakeEmbedResult(1)
    succeeded = _embed_and_store(
        [
            {"id": "code:f.cs:0", "text": "AAA2", "metadata": meta(0)},
            {"id": "code:f.cs:1", "text": "BBB", "metadata": meta(1)},
        ],
        collection, "voyage-code-3", vo, cache=cache,
    )

    chunk_to_file = {"code:f.cs:0": f, "code:f.cs:1": f}
    fully = _track_embed_success(succeeded, chunk_to_file, {f: 2}, {}, collection)
    assert f in fully  # one pass marks it indexed despite a cache-hit chunk
    cache.close()


def test_embed_and_store_cache_keys_on_embedded_text_not_full_text(tmp_path, monkeypatch):
    """Truncation safety (Phase-4 finding): an oversized chunk is truncated by
    _make_batches before embedding, so the cache must key on the EMBEDDED
    (post-truncation) text. Keying on the full text would false-hit on a later
    run and pair the full document with a truncated-text vector."""
    from vecs.embed_cache import EmbedCache

    cache = EmbedCache(tmp_path / "c.db")
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.return_value = FakeEmbedResult(1)

    full = "X" * 1000
    truncated = "X" * 10
    chunk = {"id": "code:f.cs:0", "text": full, "metadata": {"file_path": "f.cs", "chunk_index": 0}}

    # Simulate _make_batches truncating the oversized chunk before embed.
    monkeypatch.setattr(
        "vecs.indexer._make_batches",
        lambda chunks, batcher=None: iter([[{**chunk, "text": truncated}]]) if chunks else iter([]),
    )
    _embed_and_store([chunk], collection, "voyage-code-3", vo, cache=cache)

    # The embedded (truncated) text is cached; the full text is NOT, so a later
    # run re-embeds rather than serving a wrong-doc vector.
    assert cache.get("voyage-code-3", [EmbedCache.content_hash(truncated)])
    assert cache.get("voyage-code-3", [EmbedCache.content_hash(full)]) == {}
    cache.close()


def test_embed_and_store_survives_cache_get_and_put_errors():
    """A cache whose get/put raise (e.g. 'database is locked' under overlapping
    reindex) MUST NOT abort indexing: chunks still embed and all ids return, so
    the caller can mark the manifest and avoid a reprocess-forever loop."""
    class BrokenCache:
        def get(self, model, hashes):
            raise Exception("database is locked")

        def put(self, model, items):
            raise Exception("database is locked")

    chunks = [
        {"id": "code:f.cs:0", "text": "AAA", "metadata": {"file_path": "f.cs", "chunk_index": 0}},
        {"id": "code:f.cs:1", "text": "BBB", "metadata": {"file_path": "f.cs", "chunk_index": 1}},
    ]
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.return_value = FakeEmbedResult(2)

    result = _embed_and_store(chunks, collection, "voyage-code-3", vo, cache=BrokenCache())
    assert set(result) == {"code:f.cs:0", "code:f.cs:1"}
    assert vo.embed.call_count == 1  # cache read failed -> all chunks embedded


def test_index_single_doc_stamps_mtime_version_id(tmp_path, monkeypatch):
    """index_single_doc (vecs add-document / MCP add_document) must stamp the
    same mtime version_id as index_docs on the shared -docs collection."""
    from vecs.indexer import index_single_doc
    from vecs.config import VecsConfig

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "note.md"
    md.write_text("# Note\n\nBody text long enough to chunk into something useful.\n")
    expected = str(md.stat().st_mtime)

    cfg = VecsConfig(path=tmp_path / "config.yaml")
    cfg.projects["p"] = ProjectConfig(name="p", docs_dirs=[docs])
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    captured = _capture_chunks_via_index_collection(monkeypatch)
    index_single_doc("p", md)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == expected for c in captured["chunks"])


def test_embed_and_store_no_cache_unchanged_behavior():
    """With cache=None (default), behavior is identical to pre-cache: all embed."""
    chunks = [
        {"id": f"code:f.cs:{i}", "text": f"chunk {i}", "metadata": {"file_path": "f.cs", "chunk_index": i}}
        for i in range(3)
    ]
    collection = MagicMock()
    vo = MagicMock()
    vo.embed.return_value = FakeEmbedResult(3)
    result = _embed_and_store(chunks, collection, "voyage-code-3", vo)
    assert set(result) == {"code:f.cs:0", "code:f.cs:1", "code:f.cs:2"}
    assert vo.embed.call_count == 1


def test_embed_and_store_treats_voyage_timeout_as_transient():
    """voyageai.error.Timeout (class name 'Timeout') must be retried, not raised.

    Regression: substring check `"TimeoutError" in "Timeout"` is False, so prior
    code raised on Voyage timeouts and aborted the run before manifest.save(),
    causing the next cron tick to redo the same work — a doom loop.
    """
    import voyageai
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
    vo.embed.side_effect = [
        FakeEmbedResult(2),
        voyageai.error.Timeout("Request timed out"),
        voyageai.error.Timeout("Request timed out"),
        voyageai.error.Timeout("Request timed out"),
        voyageai.error.Timeout("Request timed out"),
        voyageai.error.Timeout("Request timed out"),
    ]

    with patch("vecs.indexer._make_batches") as mock_batches, \
         patch("vecs.indexer.time.sleep"):
        mock_batches.return_value = iter([batch1_chunks, batch2_chunks])
        result = _embed_and_store(all_chunks, collection, "voyage-code-3", vo)

    assert set(result) == {"code:a.cs:0", "code:a.cs:1"}


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


# --- _sync_bm25 tests ---

def test_sync_bm25_inserts_new_chunks(tmp_path, monkeypatch):
    """_sync_bm25 inserts chunks that exist in chromadb but not yet in BM25."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a", "b", "c"],
        "documents": ["alpha text", "beta text", "gamma text"],
        "metadatas": [{"file_path": "a.ts"}, {"file_path": "b.ts"}, {"file_path": "c.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    db = tmp_path / "bm25" / "test_code.db"
    assert db.exists()
    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == {"a", "b", "c"}
    bm25.close()


def test_sync_bm25_deletes_removed_chunks(tmp_path, monkeypatch):
    """_sync_bm25 removes BM25 docs that are no longer in chromadb."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    db = tmp_path / "bm25" / "test_code.db"
    pre = BM25Index(db)
    pre.upsert([
        {"id": "a", "text": "alpha", "metadata": {"file_path": "a.ts"}},
        {"id": "b", "text": "beta", "metadata": {"file_path": "b.ts"}},
        {"id": "c", "text": "gamma", "metadata": {"file_path": "c.ts"}},
    ])
    pre.close()

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a"],
        "documents": ["alpha text"],
        "metadatas": [{"file_path": "a.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == {"a"}
    bm25.close()


def test_sync_bm25_updates_changed_chunks(tmp_path, monkeypatch):
    """_sync_bm25 updates content for existing chunk ids when the chromadb text changes."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    db = tmp_path / "bm25" / "test_code.db"
    pre = BM25Index(db)
    pre.upsert([{"id": "a", "text": "old text alpha", "metadata": {"file_path": "a.ts"}}])
    pre.close()

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a"],
        "documents": ["new text completely different"],
        "metadatas": [{"file_path": "a.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    bm25 = BM25Index(db)
    bm25.load()
    # Old-unique term "alpha" should no longer match document "a"
    old_results = bm25.search("alpha", n=5)
    assert all(r["id"] != "a" for r in old_results)
    # New text should match
    new_results = bm25.search("completely different", n=5)
    assert any(r["id"] == "a" for r in new_results)
    bm25.close()


def test_sync_bm25_logs_on_failure(tmp_path, monkeypatch, capsys):
    """_sync_bm25 logs errors instead of swallowing them."""
    from vecs import indexer

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    collection = MagicMock()
    collection.get.side_effect = RuntimeError("ChromaDB unavailable")

    indexer._sync_bm25(collection, "test", "code")

    captured = capsys.readouterr()
    assert "BM25 sync failed" in captured.err
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


# --- exclude_dirs filtering on index_code ---


def _make_index_db(tmp_path):
    """Build a chromadb-like mock that records upserted ids per file_path metadata."""
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    db = MagicMock()
    db.get_or_create_collection.return_value = collection
    return db, collection


def _capture_files(monkeypatch):
    """Patch _index_collection to capture files_to_process and skip embed work."""
    captured: dict = {}

    def fake_index_collection(*, files_to_process, file_hashes, manifest, **kw):
        captured["files"] = list(files_to_process)
        captured["file_hashes"] = dict(file_hashes)
        # Mark all files as indexed so manifest persists -- mirrors successful path
        for f in files_to_process:
            manifest.mark_indexed(f, file_hashes[f])
        manifest.save()
        return len(files_to_process)

    monkeypatch.setattr("vecs.indexer._index_collection", fake_index_collection)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    return captured


def test_index_code_exclude_dirs_drops_files(tmp_path, monkeypatch):
    """Files under exclude_dirs are not indexed."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Scripts" / "keep.cs"
    keep.write_text("public class K {}")
    drop = code_root / "Library" / "drop.cs"
    drop.write_text("public class D {}")

    project = ProjectConfig(
        name="exclproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert keep in captured["files"]
    assert drop not in captured["files"]


def test_index_code_exclude_wins_over_include(tmp_path, monkeypatch):
    """When include_dirs and exclude_dirs overlap, exclude wins."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts" / "Editor").mkdir(parents=True)
    (code_root / "Scripts" / "Runtime").mkdir(parents=True)
    editor_file = code_root / "Scripts" / "Editor" / "e.cs"
    editor_file.write_text("// editor")
    runtime_file = code_root / "Scripts" / "Runtime" / "r.cs"
    runtime_file.write_text("// runtime")

    project = ProjectConfig(
        name="overlapproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            include_dirs=["Scripts"],
            exclude_dirs=["Scripts/Editor"],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert runtime_file in captured["files"]
    assert editor_file not in captured["files"]


def test_index_code_empty_exclude_dirs_is_noop(tmp_path, monkeypatch):
    """Empty exclude_dirs walks the full tree just like before."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "UI").mkdir(parents=True)
    a = code_root / "Scripts" / "a.cs"
    a.write_text("// a")
    b = code_root / "UI" / "b.cs"
    b.write_text("// b")

    project = ProjectConfig(
        name="noexclproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=[],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert a in captured["files"]
    assert b in captured["files"]


def test_index_code_prunes_manifest_for_now_excluded_files(tmp_path, monkeypatch):
    """Files previously indexed but now under exclude_dirs are pruned from the manifest."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Scripts" / "keep.cs"
    keep.write_text("// keep")
    stale = code_root / "Library" / "stale.cs"
    stale.write_text("// stale")

    # Pre-populate the manifest as if both files were previously indexed
    manifest = Manifest("pruneproj", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(keep, hashlib.sha256(keep.read_bytes()).hexdigest())
    manifest.mark_indexed(stale, hashlib.sha256(stale.read_bytes()).hexdigest())
    manifest.save()

    project = ProjectConfig(
        name="pruneproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    reloaded = Manifest("pruneproj", manifests_dir=tmp_path / "manifests")
    assert str(keep) in reloaded.data
    assert str(stale) not in reloaded.data


# --- _paginated_get tests (Problem 1: SQLite IN-clause limit) ---


def test_paginated_get_passes_limit_and_offset():
    """_paginated_get always passes a bounded limit and a starting offset.

    Without limit, chromadb constructs an internal SQL query whose IN-clause
    can exceed SQLITE_MAX_VARIABLE_NUMBER (32766 on modern SQLite).
    """
    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    list(_paginated_get(collection, batch_size=5000, include=["documents"]))

    assert collection.get.called
    first_kwargs = collection.get.call_args_list[0].kwargs
    assert first_kwargs.get("limit") == 5000
    assert first_kwargs.get("offset") == 0
    # Defensive: limit must stay well under SQLite's IN-clause cap
    assert first_kwargs["limit"] <= 5000


def test_paginated_get_walks_multiple_pages():
    """_paginated_get advances offset until the collection is exhausted."""
    collection = MagicMock()

    def fake_get(*, limit, offset, **kw):
        # Simulate 12 total ids spread across pages of 5
        all_ids = [f"id_{i}" for i in range(12)]
        page_ids = all_ids[offset:offset + limit]
        return {
            "ids": page_ids,
            "documents": [f"doc_{i.split('_')[1]}" for i in page_ids],
            "metadatas": [{"i": i} for i in page_ids],
        }

    collection.get.side_effect = fake_get

    pages = list(_paginated_get(collection, batch_size=5, include=["documents", "metadatas"]))

    # Three pages: 5, 5, 2
    assert [len(p["ids"]) for p in pages] == [5, 5, 2]
    flat = [i for p in pages for i in p["ids"]]
    assert flat == [f"id_{i}" for i in range(12)]


def test_paginated_get_stops_on_empty_page():
    """_paginated_get does not infinite-loop when the collection returns no ids."""
    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    pages = list(_paginated_get(collection, batch_size=5000))

    # One probe call, no iterations yielded
    assert pages == []
    assert collection.get.call_count == 1


def test_sync_bm25_paginates_through_all_chunks(tmp_path, monkeypatch):
    """_sync_bm25 walks all chromadb pages, not just the first."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    total_chunks = 12
    all_ids = [f"code:f{i}.ts:0" for i in range(total_chunks)]
    all_docs = [f"function f{i}() {{ return {i}; }}" for i in range(total_chunks)]
    all_metas = [{"file_path": f"f{i}.ts"} for i in range(total_chunks)]

    def fake_get(*, limit, offset, include):
        return {
            "ids": all_ids[offset:offset + limit],
            "documents": all_docs[offset:offset + limit],
            "metadatas": all_metas[offset:offset + limit],
        }

    collection = MagicMock()
    collection.get.side_effect = fake_get

    # Force small batch size so pagination is actually exercised
    original_paginated_get = indexer._paginated_get
    monkeypatch.setattr(
        indexer,
        "_paginated_get",
        lambda col, **kw: original_paginated_get(col, batch_size=5, **{k: v for k, v in kw.items() if k != 'batch_size'}),
    )

    indexer._sync_bm25(collection, "test", "code")

    db = tmp_path / "bm25" / "test_code.db"
    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == set(all_ids)
    bm25.close()

    assert collection.get.call_count >= 2, (
        f"expected pagination to make multiple get() calls, got {collection.get.call_count}"
    )


# --- _sweep_excluded_chunks tests (Problem 2: orphan chunks for excluded files) ---


def test_sweep_excluded_chunks_deletes_orphans_under_excluded_subdirs(tmp_path):
    """Chunks whose file_path is under an excluded subdir are deleted."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    collection = MagicMock()
    # Single page is fine for this case
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Library/PackageCache/foo.cs:0",
            "code:client-uk/Assets/Scripts/Player.cs:0",
            "code:client-uk/Library/Cached.cs:1",
        ],
        "documents": ["", "", ""],
        "metadatas": [
            {"file_path": "client-uk/Library/PackageCache/foo.cs"},
            {"file_path": "client-uk/Assets/Scripts/Player.cs"},
            {"file_path": "client-uk/Library/Cached.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 2
    collection.delete.assert_called_once()
    deleted_ids = collection.delete.call_args.kwargs.get("ids") or collection.delete.call_args.args[0]
    assert set(deleted_ids) == {
        "code:client-uk/Library/PackageCache/foo.cs:0",
        "code:client-uk/Library/Cached.cs:1",
    }


def test_sweep_excluded_chunks_skips_chunks_outside_excluded_subdirs(tmp_path):
    """Chunks whose file_path is NOT under an excluded subdir stay put."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Assets/Scripts/A.cs:0",
            "code:client-uk/Assets/Scripts/B.cs:0",
        ],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "client-uk/Assets/Scripts/A.cs"},
            {"file_path": "client-uk/Assets/Scripts/B.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 0
    collection.delete.assert_not_called()


def test_sweep_excluded_chunks_noop_with_empty_exclude_dirs(tmp_path):
    """When exclude_dirs is empty, the helper does nothing -- no get, no delete."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=[],
    )

    collection = MagicMock()

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 0
    collection.get.assert_not_called()
    collection.delete.assert_not_called()


def test_sweep_excluded_chunks_paginates_delete_for_huge_orphan_sets(tmp_path):
    """When orphan count exceeds the SQLite IN-clause cap, delete must be batched.

    Same SQL-variable bug that hit collection.get() rebirths on collection.delete()
    if we hand it a single list of 10K+ ids. Must chunk into batches <= 5000.
    """
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    n_orphans = 12000
    ids = [f"code:client-uk/Library/file{i}.cs:0" for i in range(n_orphans)]
    metadatas = [{"file_path": f"client-uk/Library/file{i}.cs"} for i in range(n_orphans)]

    collection = MagicMock()
    collection.get.side_effect = [
        {"ids": ids, "documents": [""] * n_orphans, "metadatas": metadatas},
        {"ids": [], "documents": [], "metadatas": []},
    ]

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == n_orphans
    assert collection.delete.call_count >= 2, (
        f"delete called {collection.delete.call_count} times for {n_orphans} orphans "
        "— must batch to avoid SQL variable limit"
    )
    all_deleted: list[str] = []
    for call in collection.delete.call_args_list:
        batch = call.kwargs.get("ids") or call.args[0]
        assert len(batch) <= 5000, f"batch size {len(batch)} exceeds safe SQLite limit"
        all_deleted.extend(batch)
    assert set(all_deleted) == set(ids)


def test_sweep_excluded_chunks_does_not_match_path_prefix_collisions(tmp_path):
    """Excluding 'Lib' must not match 'Library/' -- match on path component boundary."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Lib"],
    )

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Library/foo.cs:0",  # NOT under "Lib/"
            "code:client-uk/Lib/bar.cs:0",      # IS  under "Lib/"
        ],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "client-uk/Library/foo.cs"},
            {"file_path": "client-uk/Lib/bar.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 1
    collection.delete.assert_called_once()
    deleted_ids = collection.delete.call_args.kwargs.get("ids") or collection.delete.call_args.args[0]
    assert deleted_ids == ["code:client-uk/Lib/bar.cs:0"]


def test_index_code_sweeps_orphan_chunks_for_excluded_dirs(tmp_path, monkeypatch):
    """index_code sweeps chromadb for orphan chunks under exclude_dirs.

    Regression: prune_out_of_scope only removes manifest-tracked entries. If a
    file was upserted to chromadb but the run died before manifest.save(),
    chunks remain. After exclude_dirs is configured, those chunks must be
    deleted regardless of manifest state.
    """
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "client-uk"
    (code_root / "Assets").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Assets" / "keep.cs"
    keep.write_text("public class K {}")

    project = ProjectConfig(
        name="sweepproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    # Simulate orphan chunks living in chromadb under Library/
    orphan_ids = [
        "code:client-uk/Library/PackageCache/orphan_a.cs:0",
        "code:client-uk/Library/orphan_b.cs:0",
    ]

    collection = MagicMock()

    def fake_get(*args, **kwargs):
        # First call: paginate ALL chunks for the sweep
        # Subsequent calls (from _delete_stale_chunks_after_embed via _track_embed_success
        # or other paths) get an empty result -- not under test here.
        if "where" in kwargs:
            return {"ids": []}
        offset = kwargs.get("offset", 0)
        if offset == 0:
            return {
                "ids": orphan_ids + ["code:client-uk/Assets/keep.cs:0"],
                "documents": ["", "", ""],
                "metadatas": [
                    {"file_path": "client-uk/Library/PackageCache/orphan_a.cs"},
                    {"file_path": "client-uk/Library/orphan_b.cs"},
                    {"file_path": "client-uk/Assets/keep.cs"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    _capture_files(monkeypatch)
    index_code(project, vo=MagicMock(), db=db)

    # Find the sweep-driven delete call (by orphan ids), tolerating other delete calls
    sweep_calls = [
        c for c in collection.delete.call_args_list
        if set(c.kwargs.get("ids") or (c.args[0] if c.args else [])) == set(orphan_ids)
    ]
    assert len(sweep_calls) == 1, (
        f"Expected exactly one sweep delete with orphan ids, "
        f"got delete calls: {collection.delete.call_args_list}"
    )


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


# --- C: version_id stamping (git SHA for code, mtime for docs, session id) ---


def _git_init_commit(repo: Path, relfile: str, content: str) -> str:
    """Init a git repo with one committed file; return the HEAD sha."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    fp = repo / relfile
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
        cwd=repo, check=True,
    )
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _capture_chunks_via_index_collection(monkeypatch):
    """Patch _index_collection to capture the chunks passed in (code/docs path)."""
    captured: dict = {}

    def fake_index_collection(*, chunks, manifest, files_to_process, file_hashes, **kw):
        captured["chunks"] = chunks
        for f in files_to_process:
            manifest.mark_indexed(f, file_hashes[f])
        manifest.save()
        return len(chunks)

    monkeypatch.setattr("vecs.indexer._index_collection", fake_index_collection)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    return captured


def test_git_sha_returns_head_for_repo(tmp_path):
    from vecs.indexer import _git_sha
    repo = tmp_path / "r"
    sha = _git_init_commit(repo, "a.cs", "class A{}")
    assert _git_sha(repo) == sha
    sub = repo / "sub"
    sub.mkdir()
    assert _git_sha(sub) == sha  # resolves from a subdir too


def test_git_sha_none_for_non_git_dir(tmp_path):
    from vecs.indexer import _git_sha
    d = tmp_path / "nogit"
    d.mkdir()
    assert _git_sha(d) is None


def test_index_code_stamps_git_sha_version_id(tmp_path, monkeypatch):
    """Each stored code chunk carries version_id == the repo HEAD sha."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    repo = tmp_path / "repo"
    sha = _git_init_commit(repo, "Main.cs", "public class Main { void M() { int x = 1; } }")

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=repo, extensions={".cs"})])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == sha for c in captured["chunks"])


def test_index_docs_stamps_mtime_version_id(tmp_path, monkeypatch):
    """Each stored docs chunk carries version_id == the file mtime."""
    from vecs.indexer import index_docs
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "readme.md"
    md.write_text("# Title\n\nSome documentation body text that is long enough to chunk.\n")
    expected = str(md.stat().st_mtime)

    project = ProjectConfig(name="p", docs_dirs=[docs])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == expected for c in captured["chunks"])


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


# --- C: production wiring of the EmbedCache through run_index / indexers ---

def test_index_code_uses_embed_cache_end_to_end(tmp_path, monkeypatch):
    """A cache passed to index_code is threaded down to _embed_and_store: a
    second index of unchanged files makes zero Voyage calls."""
    from vecs.embed_cache import EmbedCache
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    repo = tmp_path / "repo"
    _git_init_commit(repo, "A.cs", "public class A { void M() { int x = 1; } }")
    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=repo, extensions={".cs"})])

    cache = EmbedCache(tmp_path / "c.db")
    db, _collection = _make_index_db(tmp_path)
    vo = MagicMock()
    vo.embed.side_effect = lambda texts, model, input_type: FakeEmbedResult(len(texts))

    n1 = index_code(project, vo=vo, db=db, cache=cache)
    assert n1 >= 1
    assert vo.embed.call_count >= 1

    # Force a re-index (drop the manifest) but keep the cache warm.
    (tmp_path / "manifests" / "p.json").unlink()
    vo.embed.reset_mock()
    n2 = index_code(project, vo=vo, db=db, cache=cache)
    assert n2 >= 1                    # same chunks re-stored
    assert vo.embed.call_count == 0   # all served from cache -> wiring intact
    cache.close()


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


def test_run_index_threads_one_cache_to_all_indexers(tmp_path, monkeypatch):
    """run_index constructs a single EmbedCache and hands it to every indexer."""
    from vecs.config import _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {"p": {"code_dirs": [{"path": str(tmp_path / "code"), "extensions": [".cs"]}]}},
        "codex_disabled": True,
    }))
    (tmp_path / "code").mkdir()
    _clear_config_cache()

    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file),
    )
    monkeypatch.setattr("vecs.indexer.migrate_global_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    # Fresh project: collections are empty, so the B2 pre-pass finds nothing to
    # re-embed. count() must be a real int (the pre-pass compares it > 0).
    db = MagicMock()
    db.get_collection.return_value.count.return_value = 0
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)
    monkeypatch.setattr("vecs.indexer.VECS_DIR", tmp_path)
    monkeypatch.setattr("vecs.indexer.CHROMADB_DIR", tmp_path / "chromadb")
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    seen: dict = {}

    def cap(name):
        def f(*a, cache=None, **kw):
            seen[name] = cache
            return 0
        return f

    monkeypatch.setattr("vecs.indexer.index_code", cap("code"))
    monkeypatch.setattr("vecs.indexer.index_sessions", cap("sessions"))
    monkeypatch.setattr("vecs.indexer.index_docs", cap("docs"))

    from vecs.indexer import run_index
    run_index()

    assert seen["code"] is not None
    assert seen["code"] is seen["sessions"]
    assert seen["code"] is seen["docs"]


# --- B2: model-change re-embed trigger (run_index pre/post pass) -----------

def _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f):
    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(doc_f, "hdoc")
    manifest.mark_indexed(code_f, "hcode")
    manifest.mark_session_indexed(sess_f, byte_offset=2, chunk_count=1)
    manifest.save()


def _remodel_fixture(tmp_path):
    """Build (project, doc_f, code_f, sess_f) sharing one manifest namespace."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    doc_f = docs_dir / "a.md"
    doc_f.write_text("doc body")
    code_f = code_dir / "m.py"
    code_f.write_text("print(1)")
    sess_f = tmp_path / "sess.jsonl"
    sess_f.write_text("{}")
    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code_dir, extensions={".py"})],
        docs_dirs=[docs_dir],
    )
    return project, doc_f, code_f, sess_f


def test_remodel_clears_docs_and_session_keys_leaves_code_on_model_change(tmp_path, monkeypatch):
    """Model change (recorded != configured) + non-empty collection: the pre-pass
    drops docs file-keys and session: keys so the next pass re-embeds them, but
    leaves code file-keys (code has no re-embed trigger)."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")
    monkeypatch.setattr("vecs.indexer.SESSIONS_MODEL", "voyage-4")

    project, doc_f, code_f, sess_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")      # stale
    cache.set_collection_model("p-sessions", "voyage-3")  # stale

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5  # non-empty

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) not in reloaded.data          # docs cleared -> re-embed
    assert f"session:{sess_f}" not in reloaded.data  # sessions cleared
    assert str(code_f) in reloaded.data              # code untouched
    cache.close()


def test_remodel_clear_scope_matches_index_docs_scan_not_extra_docs_dirs(tmp_path, monkeypatch):
    """The clear pass must only drop docs keys for files index_docs will re-embed.

    index_docs scans project.docs_dir (== docs_dirs[0]) only; multi-source docs
    is deferred to F. So a model change must NOT clear keys for files under
    docs_dirs[1..] -- clearing a key index_docs won't re-scan would strand that
    file's old-model vectors in a collection whose marker just advanced. Clear
    and re-embed scopes must agree."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")
    monkeypatch.setattr("vecs.indexer.SESSIONS_MODEL", "voyage-4")

    docs0 = tmp_path / "docs0"
    docs0.mkdir()
    docs1 = tmp_path / "docs1"
    docs1.mkdir()
    f0 = docs0 / "a.md"
    f0.write_text("x")
    f1 = docs1 / "b.md"
    f1.write_text("y")

    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(f0, "h0")
    manifest.mark_indexed(f1, "h1")
    manifest.save()

    project = ProjectConfig(name="p", docs_dirs=[docs0, docs1])

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")  # stale

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(f0) not in reloaded.data, "docs_dir[0] file (index_docs re-scans it) must be cleared"
    assert str(f1) in reloaded.data, "docs_dirs[1..] file (index_docs won't re-scan it yet) must NOT be cleared"
    cache.close()


def test_remodel_noop_when_model_unchanged(tmp_path, monkeypatch):
    """Recorded == configured: nothing cleared even though collection is non-empty."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")
    monkeypatch.setattr("vecs.indexer.SESSIONS_MODEL", "voyage-4")

    project, doc_f, code_f, sess_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-4")      # matches
    cache.set_collection_model("p-sessions", "voyage-4")  # matches
    cache.close()
    cache = EmbedCache(tmp_path / "embed_cache.db")

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    assert f"session:{sess_f}" in reloaded.data
    assert str(code_f) in reloaded.data
    cache.close()


def test_remodel_noop_when_collection_empty(tmp_path, monkeypatch):
    """Recorded differs but collection is empty (count==0): no re-embed needed."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")
    monkeypatch.setattr("vecs.indexer.SESSIONS_MODEL", "voyage-4")

    project, doc_f, code_f, sess_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")      # stale...
    cache.set_collection_model("p-sessions", "voyage-3")

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 0  # ...but empty

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    assert f"session:{sess_f}" in reloaded.data
    cache.close()


def test_remodel_treats_missing_collection_as_empty(tmp_path, monkeypatch):
    """A collection that does not exist yet (get_collection raises) is empty -> no clear."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")
    monkeypatch.setattr("vecs.indexer.SESSIONS_MODEL", "voyage-4")

    project, doc_f, code_f, sess_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")
    cache.set_collection_model("p-sessions", "voyage-3")

    db = MagicMock()
    db.get_collection.side_effect = Exception("collection not found")

    _remodel_clear(project, db, cache)  # must not raise

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    assert f"session:{sess_f}" in reloaded.data
    cache.close()


def test_run_index_remodel_migrates_docs_manifest_and_records_marker(tmp_path, monkeypatch):
    """End-to-end migration: an existing voyage-3 docs collection (recorded
    voyage-3, non-empty) under configured voyage-4 has its docs manifest entry
    cleared by the pre-pass, and the post-pass records voyage-4 as the new
    marker so the next run is a no-op."""
    from vecs.config import _clear_config_cache
    from vecs.embed_cache import EmbedCache

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_f = docs_dir / "a.md"
    doc_f.write_text("doc body")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {"p": {
            "code_dirs": [{"path": str(tmp_path / "code"), "extensions": [".py"]}],
            "docs_dirs": [str(docs_dir)],
        }},
        "codex_disabled": True,
    }))
    (tmp_path / "code").mkdir()
    _clear_config_cache()

    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file),
    )
    monkeypatch.setattr("vecs.indexer.migrate_global_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    monkeypatch.setattr("vecs.indexer.VECS_DIR", tmp_path)
    monkeypatch.setattr("vecs.indexer.CHROMADB_DIR", tmp_path / "chromadb")
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    # Existing voyage-3 docs collection: non-empty, recorded under the old model.
    db = MagicMock()
    db.get_collection.return_value.count.return_value = 12
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    # Pre-seed: a docs manifest entry + the old marker (simulates pre-B2 state).
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    m.mark_indexed(doc_f, "h")
    m.save()
    seed = EmbedCache(tmp_path / "embed_cache.db")
    seed.set_collection_model("p-docs", "voyage-3")
    seed.close()

    # Indexers are no-ops: we are testing the orchestration pre/post pass only.
    monkeypatch.setattr("vecs.indexer.index_code", lambda *a, **kw: 0)
    monkeypatch.setattr("vecs.indexer.index_sessions", lambda *a, **kw: 0)
    monkeypatch.setattr("vecs.indexer.index_docs", lambda *a, **kw: 0)
    monkeypatch.setattr("vecs.indexer.index_codex_sessions", lambda *a, **kw: 0)

    from vecs.indexer import run_index
    run_index()

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) not in reloaded.data, "pre-pass should clear the stale docs entry"

    chk = EmbedCache(tmp_path / "embed_cache.db")
    assert chk.get_collection_model("p-docs") == "voyage-4", "post-pass should record new marker"
    # _remodel_record stamps BOTH docs and sessions; assert sessions too so a
    # dropped session set_collection_model line cannot pass unnoticed.
    assert chk.get_collection_model("p-sessions") == "voyage-4", "post-pass should record sessions marker"
    chk.close()


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
