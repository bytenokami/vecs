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
    monkeypatch.setattr(searcher, "get_voyage_client", lambda: MagicMock())
    monkeypatch.setattr(searcher, "_cached_embed", lambda vo, q, m: [0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(searcher, "get_bm25", lambda path: None)  # skip BM25

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
