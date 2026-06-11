import os
from unittest.mock import MagicMock

import pytest

from vecs.searcher import format_results, deduplicate_results, reciprocal_rank_fusion


def test_format_results_empty():
    results = format_results({"ids": [[]], "documents": [[]], "metadatas": [[]]})
    assert results == []


def test_format_results_with_data():
    results = format_results(
        {
            "ids": [["id1", "id2"]],
            "documents": [["doc one", "doc two"]],
            "metadatas": [[{"file_path": "a.cs"}, {"session_id": "abc"}]],
            "distances": [[0.1, 0.5]],
        }
    )
    assert len(results) == 2
    assert results[0]["text"] == "doc one"
    assert results[0]["metadata"]["file_path"] == "a.cs"
    assert results[0]["distance"] == 0.1


def test_deduplicate_removes_overlapping():
    """Results with >55% line overlap are deduplicated."""
    shared = "\n".join(f"line {i}" for i in range(80))
    results = [
        {"id": "a", "text": shared + "\nextra_a1\nextra_a2", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": shared + "\nextra_b1", "metadata": {}, "distance": 0.2},
        {"id": "c", "text": "totally different content\nnothing in common", "metadata": {}, "distance": 0.3},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "a"
    assert deduped[1]["id"] == "c"


def test_deduplicate_keeps_unique():
    """Non-overlapping results are all kept."""
    results = [
        {"id": "a", "text": "alpha\nbeta\ngamma", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "delta\nepsilon\nzeta", "metadata": {}, "distance": 0.2},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 2


def test_rrf_merges_rankings():
    """RRF combines two rankings into a merged ranking."""
    vector_results = [
        {"id": "a", "text": "alpha", "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "beta", "metadata": {}, "distance": 0.3},
    ]
    bm25_results = [
        {"id": "b", "text": "beta", "score": 5.0},
        {"id": "c", "text": "gamma", "score": 3.0},
    ]
    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    ids = [r["id"] for r in merged]
    # "b" appears in both, should rank high
    assert "b" in ids
    assert "a" in ids
    assert "c" in ids


def test_rrf_bm25_results_have_metadata():
    """BM25 results with metadata are preserved through RRF."""
    vector_results = [
        {"id": "a", "text": "alpha", "metadata": {"file_path": "a.ts"}, "distance": 0.1},
    ]
    bm25_results = [
        {"id": "b", "text": "beta", "score": 5.0, "metadata": {"file_path": "b.ts"}},
    ]
    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    b_result = next(r for r in merged if r["id"] == "b")
    assert b_result["metadata"] == {"file_path": "b.ts"}


def test_deduplicate_default_threshold_is_055():
    """Default threshold is 0.55 -- pairs with 0.556 Jaccard are deduped."""
    shared = [f"line {i}" for i in range(55)]
    only_a = [f"unique_a_{i}" for i in range(22)]
    only_b = [f"unique_b_{i}" for i in range(22)]
    results = [
        {"id": "a", "text": "\n".join(shared + only_a), "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "\n".join(shared + only_b), "metadata": {}, "distance": 0.2},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 1
    assert deduped[0]["id"] == "a"


def test_deduplicate_below_055_kept():
    """Pairs below 0.55 Jaccard are NOT deduped."""
    shared = [f"line {i}" for i in range(50)]
    only_a = [f"unique_a_{i}" for i in range(25)]
    only_b = [f"unique_b_{i}" for i in range(25)]
    results = [
        {"id": "a", "text": "\n".join(shared + only_a), "metadata": {}, "distance": 0.1},
        {"id": "b", "text": "\n".join(shared + only_b), "metadata": {}, "distance": 0.2},
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 2


def test_rrf_default_weights():
    """Default weights: vector=1.0, bm25=0.6. Vector-only hit ranks higher than BM25-only."""
    vector_results = [
        {"id": "v_only", "text": "vector only", "metadata": {}, "distance": 0.1},
    ]
    bm25_results = [
        {"id": "b_only", "text": "bm25 only", "score": 5.0},
    ]
    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    ids = [r["id"] for r in merged]
    assert ids[0] == "v_only"
    assert ids[1] == "b_only"


def test_rrf_custom_weights():
    """Custom weights change the ranking."""
    vector_results = [
        {"id": "v_only", "text": "vector only", "metadata": {}, "distance": 0.1},
    ]
    bm25_results = [
        {"id": "b_only", "text": "bm25 only", "score": 5.0},
    ]
    merged = reciprocal_rank_fusion(
        vector_results, bm25_results, k=60,
        w_vector=0.3, w_bm25=1.0,
    )
    ids = [r["id"] for r in merged]
    assert ids[0] == "b_only"
    assert ids[1] == "v_only"


def test_rrf_shared_result_accumulates_weighted_scores():
    """A result in both lists accumulates weighted scores from both."""
    vector_results = [
        {"id": "shared", "text": "shared doc", "metadata": {}, "distance": 0.1},
        {"id": "v_only", "text": "vector only", "metadata": {}, "distance": 0.2},
    ]
    bm25_results = [
        {"id": "shared", "text": "shared doc", "score": 5.0},
        {"id": "b_only", "text": "bm25 only", "score": 3.0},
    ]
    merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    ids = [r["id"] for r in merged]
    assert ids[0] == "shared"


def test_deduplicate_returns_fewer_when_heavy_overlap():
    """Verify dedup can reduce results below n_results (precondition for L3)."""
    base = "\n".join(f"line {i}" for i in range(100))
    results = [
        {"id": f"r{i}", "text": base + f"\nunique_{i}", "metadata": {}, "distance": 0.1 * i}
        for i in range(5)
    ]
    deduped = deduplicate_results(results)
    assert len(deduped) == 1


# --- Inc 1.5b: -docs searched even without a configured docs_dir -------------


def test_search_queries_docs_collection_without_docs_dir(monkeypatch):
    """A project with NO docs_dir but a populated -docs collection (F routed its
    in-repo .md there) must still be searched -- the -docs target must not be
    gated on proj.docs_dir, mirroring code/sessions' skip-on-miss."""
    from vecs import searcher
    from vecs.config import VecsConfig, ProjectConfig

    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["bloomly"] = ProjectConfig(name="bloomly")  # docs_dirs empty -> docs_dir None
    assert cfg.projects["bloomly"].docs_dir is None
    monkeypatch.setattr(searcher, "load_config", lambda: cfg)
    monkeypatch.setattr(searcher, "get_provider", lambda config=None, name=None: MagicMock())
    monkeypatch.setattr(searcher, "_cached_embed", lambda vo, q, m: [0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)  # skip BM25
    # 1.5c interlock now consults the embed-cache markers; stub them to "no
    # marker" (fail-open) so this 1.5b test still exercises the vector path.
    monkeypatch.setattr(
        searcher, "_collection_markers", lambda cols: {c: None for c in cols}
    )

    queried: list[str] = []

    def fake_get_collection(name):
        queried.append(name)
        if name == "bloomly-docs":
            col = MagicMock()
            col.query.return_value = {
                "ids": [["bloomly-docs:bloomly/README.md:0"]],
                "documents": [["doc body"]],
                "metadatas": [[{"file_path": "bloomly/README.md"}]],
                "distances": [[0.12]],
            }
            return col
        raise Exception("empty/absent collection")  # code/sessions skip-on-miss

    db = MagicMock()
    db.get_collection.side_effect = fake_get_collection
    monkeypatch.setattr(searcher, "get_chromadb_client", lambda: db)

    results = searcher.search("anything", collection_name="docs", project="bloomly")

    assert "bloomly-docs" in queried  # the -docs target was attempted
    assert any(
        (r.get("metadata") or {}).get("file_path") == "bloomly/README.md" for r in results
    )


# --- Inc 1-B: post-re-embed quality check (acceptance line 18) --------------
# "known query -> expected-source pairs return the expected source post-re-embed
# (not merely non-empty)." This needs a LIVE Voyage embed + a store actually
# re-embedded under voyage-4, so it is gated exactly like
# tests/test_prose_drift.py::test_integration_real_anthropic and DEFAULT-SKIPPED.
# Run it AFTER the migrating reindex on the machine that owns the live store:
#     VECS_TEST_REAL_LLM=1 uv run pytest -q tests/test_searcher.py -k reembed_quality
# The pairs target the `vecs` repo's own docs; edit them if your docs_dir differs.
REEMBED_EVAL_SET = [
    # (query, project, collection, expected file_path substring)
    ("content-addressable embedding cache keyed by content hash", "vecs", "docs", "kb-foundations"),
    ("prose staleness detector stage 2 semantic recall", "vecs", "docs", "prose-staleness-detector"),
    ("hybrid search reciprocal rank fusion across collections", "vecs", "docs", "vecs"),
]


@pytest.mark.skipif(
    os.environ.get("VECS_TEST_REAL_LLM") != "1"
    or not os.environ.get("VOYAGE_API_KEY"),
    reason="Set VECS_TEST_REAL_LLM=1 and VOYAGE_API_KEY to run the live post-re-embed quality check.",
)
def test_reembed_quality_expected_sources_returned():
    """Each eval query must surface its EXPECTED docs source in the top results
    once docs are re-embedded under voyage-4 (B). Catches the silent ranking
    degradation a non-empty smoke test misses (querying a voyage-3 store with
    voyage-4 query vectors would still return results -- just wrong ones)."""
    from vecs.searcher import search
    from vecs.config import load_config

    config = load_config()
    checked = 0
    for query, project, collection, expected in REEMBED_EVAL_SET:
        if project not in config.projects:
            continue
        proj = config.projects[project]
        if collection == "docs" and not proj.docs_dir:
            continue
        results = search(query, collection_name=collection, n_results=5, project=project)
        sources = [(r.get("metadata") or {}).get("file_path", "") for r in results]
        assert any(expected in s for s in sources), (
            f"expected a source containing {expected!r} in top-5 for query "
            f"{query!r}; got {sources!r}"
        )
        checked += 1
    if checked == 0:
        pytest.skip("No eval pairs matched a configured project/collection; adjust REEMBED_EVAL_SET.")


# --- Inc 1.5c: query-time model-flip interlock -------------------------------
# When a collection's recorded embed-model marker differs from the configured
# model, scoring new-model query vectors against old-model stored vectors
# silently degrades ranking. The searcher must skip vector scoring for that
# collection and fall back to BM25 (model-agnostic) until a reindex re-embeds it.


def _single_project(searcher, monkeypatch):
    """One project (bloomly), with voyage/embed stubbed. Returns the config."""
    from vecs.config import VecsConfig, ProjectConfig

    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["bloomly"] = ProjectConfig(name="bloomly")
    monkeypatch.setattr(searcher, "load_config", lambda: cfg)
    monkeypatch.setattr(searcher, "get_provider", lambda config=None, name=None: MagicMock())
    monkeypatch.setattr(searcher, "_cached_embed", lambda vo, q, m: [0.1, 0.2, 0.3, 0.4])
    return cfg


def _vector_query_recorder(monkeypatch, searcher):
    """Wire a chroma stub whose .query appends the collection name to a list."""
    queried: list[str] = []

    def fake_get_collection(name):
        col = MagicMock()

        def _query(*a, **k):
            queried.append(name)
            return {
                "ids": [[f"{name}:0"]],
                "documents": [["vector hit"]],
                "metadatas": [[{"file_path": "bloomly/x"}]],
                "distances": [[0.1]],
            }

        col.query.side_effect = _query
        return col

    db = MagicMock()
    db.get_collection.side_effect = fake_get_collection
    monkeypatch.setattr(searcher, "get_chromadb_client", lambda: db)
    return queried


def test_model_flip_interlock_skips_vector_falls_back_to_bm25(monkeypatch):
    """Marker != configured model -> the collection's vectors are NOT queried,
    and BM25 supplies the fallback hit for it."""
    from vecs import searcher

    _single_project(searcher, monkeypatch)
    monkeypatch.setattr(
        searcher,
        "_collection_markers",
        lambda cols: {c: ("voyage-3-OLD" if c.endswith("-docs") else None) for c in cols},
    )
    vector_queried = _vector_query_recorder(monkeypatch, searcher)

    def fake_get_bm25(path):
        if path.name.endswith("_docs.db"):
            bm = MagicMock()
            bm.search.return_value = [
                {
                    "id": "bloomly-docs:bloomly/README.md:0",
                    "text": "bm25 docs hit",
                    "metadata": {"file_path": "bloomly/README.md"},
                    "score": 5.0,
                }
            ]
            return bm
        return None

    monkeypatch.setattr(searcher, "get_bm25", fake_get_bm25)

    results = searcher.search("anything", collection_name="docs", project="bloomly")

    assert "bloomly-docs" not in vector_queried  # vectors skipped (interlock)
    assert any(
        (r.get("metadata") or {}).get("file_path") == "bloomly/README.md"
        for r in results
    )  # BM25 fallback surfaced its content


def test_model_flip_interlock_inactive_when_marker_matches(monkeypatch):
    """Marker == configured model -> vectors ARE queried (no interlock)."""
    from vecs import searcher
    from vecs.config import DOCS_MODEL

    _single_project(searcher, monkeypatch)
    monkeypatch.setattr(
        searcher,
        "_collection_markers",
        lambda cols: {c: (DOCS_MODEL if c.endswith("-docs") else None) for c in cols},
    )
    vector_queried = _vector_query_recorder(monkeypatch, searcher)
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)

    searcher.search("anything", collection_name="docs", project="bloomly")

    assert "bloomly-docs" in vector_queried


def test_model_marker_none_fails_open_vectors_queried(monkeypatch):
    """No recorded marker (e.g. code collections, never marked) must NOT disable
    vector search -- we cannot assert a mismatch, so fail open."""
    from vecs import searcher

    _single_project(searcher, monkeypatch)
    monkeypatch.setattr(
        searcher, "_collection_markers", lambda cols: {c: None for c in cols}
    )
    vector_queried = _vector_query_recorder(monkeypatch, searcher)
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)

    searcher.search("anything", collection_name="code", project="bloomly")

    assert "bloomly-code" in vector_queried


def test_model_flip_interlock_is_per_collection_not_all_or_nothing(monkeypatch):
    """collection=None over one project: the -docs marker is OLD (mismatch) and
    the -code marker is None (fail-open). The interlock must drop ONLY -docs
    from the vector path while -code is still vector-queried -- per-collection,
    NOT all-or-nothing (an all-or-nothing collapse would nuke code retrieval,
    the exact catastrophe the fail-open design pin guards against)."""
    from vecs import searcher

    _single_project(searcher, monkeypatch)
    monkeypatch.setattr(
        searcher,
        "_collection_markers",
        lambda cols: {c: ("voyage-3-OLD" if c.endswith("-docs") else None) for c in cols},
    )
    vector_queried = _vector_query_recorder(monkeypatch, searcher)
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)

    searcher.search("anything", project="bloomly")  # collection_name=None -> both

    assert "bloomly-code" in vector_queried  # None marker -> fail-open, kept
    assert "bloomly-docs" not in vector_queried  # OLD marker -> dropped


# --- Inc 1.5c: the real _collection_markers (fail-open contract) -------------
# The interlock tests above stub _collection_markers; these exercise the REAL
# function, pinning the load-bearing fail-open-on-error branch that guarantees a
# marker-read failure can never disable vector search (else it nukes all code
# retrieval). Without these, a refactor narrowing the try/except would leave the
# stubbed interlock tests green while production search crashes query-wide.


def test_collection_markers_reads_real_cache(tmp_path, monkeypatch):
    """Reads recorded markers from a live EmbedCache; an unmarked collection
    returns None."""
    from vecs import searcher
    from vecs.embed_cache import EmbedCache as RealCache

    db = tmp_path / "cache.db"
    seed = RealCache(db_path=db)
    seed.set_collection_model("vecs-docs", "voyage-4")
    seed.close()

    monkeypatch.setattr(searcher, "EmbedCache", lambda: RealCache(db_path=db))
    markers = searcher._collection_markers(["vecs-docs", "vecs-code"])

    assert markers["vecs-docs"] == "voyage-4"
    assert markers["vecs-code"] is None  # never marked -> None (fail-open)


def test_collection_markers_fail_open_on_cache_error(monkeypatch):
    """A cache-open/read error yields None for EVERY collection and never
    raises -- the load-bearing safety branch behind the interlock."""
    from vecs import searcher

    class _Boom:
        def __init__(self, *a, **k):
            raise OSError("cache unavailable")

    monkeypatch.setattr(searcher, "EmbedCache", _Boom)
    markers = searcher._collection_markers(["x-code", "x-docs"])

    assert markers == {"x-code": None, "x-docs": None}


def test_cached_embed_uses_provider_query_input_type():
    from unittest.mock import MagicMock
    from vecs.embed_provider import VoyageProvider
    from vecs.searcher import _cached_embed, _clear_caches

    _clear_caches()
    vo = MagicMock()
    result = MagicMock()
    result.embeddings = [[0.2] * 4]
    vo.embed.return_value = result
    provider = VoyageProvider(client=vo)
    emb = _cached_embed(provider, "q", "voyage-code-3")
    assert emb == [0.2] * 4
    assert vo.embed.call_args.kwargs["input_type"] == "query"
    # second call: cache hit, no new provider call
    _cached_embed(provider, "q", "voyage-code-3")
    assert vo.embed.call_count == 1


def test_model_flip_interlock_drops_mismatched_code_collection(monkeypatch):
    """A -code collection marked under a different model is dropped from the
    vector path (BM25-only). The interlock itself needs zero change for this —
    the test pins that the NEW code markers (L1.4) engage it."""
    from vecs import searcher

    _single_project(searcher, monkeypatch)
    monkeypatch.setattr(
        searcher,
        "_collection_markers",
        lambda cols: {
            c: ("voyage-code-OLD" if c.endswith("-code") else None) for c in cols
        },
    )
    vector_queried = _vector_query_recorder(monkeypatch, searcher)
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)

    searcher.search("anything", project="bloomly")

    assert "bloomly-code" not in vector_queried  # mismatched marker -> dropped
    assert "bloomly-docs" in vector_queried      # None marker -> fail-open, kept


def test_search_collections_accepts_explicit_targets_and_bm25_paths(tmp_path, monkeypatch):
    """The A/B harness entry point: same pipeline as search(), but targets,
    provider and bm25 paths are injected instead of derived from config."""
    from unittest.mock import MagicMock
    from vecs.embed_provider import VoyageProvider
    from vecs.searcher import search_collections, _clear_caches

    _clear_caches()
    vo = MagicMock()
    emb_result = MagicMock()
    emb_result.embeddings = [[0.3] * 4]
    vo.embed.return_value = emb_result
    provider = VoyageProvider(client=vo)

    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["hello world"]],
        "metadatas": [[{"file_path": "src/x.py"}]],
        "distances": [[0.1]],
    }
    db = MagicMock()
    db.get_collection.return_value = collection
    monkeypatch.setattr("vecs.searcher.get_chromadb_client", lambda: db)

    out = search_collections(
        "q",
        targets=[("shadow-code-qwen", "qwen3-embedding-0.6b", "vecs")],
        provider=provider,
        n_results=3,
        bm25_paths={"shadow-code-qwen": tmp_path / "absent.db"},
        check_markers=False,
    )
    assert out and out[0]["id"] == "c1"
    assert out[0]["collection"] == "shadow-code-qwen"
    assert vo.embed.call_args.kwargs["model"] == "qwen3-embedding-0.6b"
