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
