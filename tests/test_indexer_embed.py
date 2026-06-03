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
