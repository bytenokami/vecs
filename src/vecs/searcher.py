from __future__ import annotations

import sys

from cachetools import TTLCache

from vecs.bm25_index import get_bm25
from vecs.clients import get_chromadb_client
from vecs.embed_provider import get_provider
from vecs.config import (
    CODE_MODEL,
    DOCS_MODEL,
    VECS_DIR,
    load_config,
)
from vecs.embed_cache import EmbedCache

# Cache embeddings by (query, model) for 5 minutes
_embedding_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def _clear_caches() -> None:
    """Clear all caches."""
    _embedding_cache.clear()


def _cached_embed(provider, query: str, model: str) -> list[float]:
    """Embed a query through the provider, using cache when available."""
    key = (query, model)
    if key in _embedding_cache:
        return _embedding_cache[key]
    embedding = provider.embed([query], model=model, input_type="query").embeddings[0]
    _embedding_cache[key] = embedding
    return embedding


def _warn(msg: str) -> None:
    """Emit a non-fatal searcher warning to stderr (mirrors indexer._log)."""
    print(f"vecs.searcher: {msg}", file=sys.stderr)


def _collection_markers(collections: list[str]) -> dict[str, str | None]:
    """Recorded embed model per collection (None = unknown), read in ONE cache
    lifetime.

    None means 'no marker' — e.g. code collections (never marked; only the docs
    re-embed path records one) or a pre-marker store. The model-flip interlock
    (1.5c) treats None as FAIL-OPEN: it cannot assert a mismatch, so vector
    scoring proceeds. Any error opening/reading the cache yields None for EVERY
    collection — a marker-read failure must never silently disable vector search
    (it would nuke all code retrieval). One EmbedCache open per search, mirroring
    the indexer's one-cache-per-operation lifecycle rather than reopening it per
    collection.
    """
    try:
        cache = EmbedCache()
        try:
            return {c: cache.get_collection_model(c) for c in collections}
        finally:
            cache.close()
    except Exception:
        return {c: None for c in collections}


def format_results(raw: dict) -> list[dict]:
    """Format ChromaDB query results into a clean list."""
    if not raw["ids"] or not raw["ids"][0]:
        return []

    results = []
    ids = raw["ids"][0]
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    distances = raw.get("distances", [[None] * len(ids)])[0]

    for id_, doc, meta, dist in zip(ids, docs, metas, distances):
        results.append(
            {
                "id": id_,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return results


def deduplicate_results(results: list[dict], threshold: float = 0.55) -> list[dict]:
    """Remove results with high line overlap.

    For each pair, if Jaccard similarity of their line sets exceeds threshold,
    the lower-ranked (higher distance) result is dropped.
    """
    if len(results) <= 1:
        return results

    keep = []
    for r in results:
        r_lines = set(r["text"].split("\n"))
        is_dup = False
        for kept in keep:
            k_lines = set(kept["text"].split("\n"))
            intersection = len(r_lines & k_lines)
            union = len(r_lines | k_lines)
            if union > 0 and intersection / union > threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(r)
    return keep


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    w_vector: float = 1.0,
    w_bm25: float = 0.6,
) -> list[dict]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = w * (1 / (k + rank)) across both result lists.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, r in enumerate(vector_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + w_vector * (1 / (k + rank + 1))
        if rid not in doc_map:
            doc_map[rid] = r

    for rank, r in enumerate(bm25_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + w_bm25 * (1 / (k + rank + 1))
        if rid not in doc_map:
            doc_map[rid] = {
                "id": rid,
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "distance": None,
            }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[rid] for rid, _ in ranked]


def search(
    query: str,
    collection_name: str | None = None,
    n_results: int = 5,
    path_filter: str | None = None,
    project: str | None = None,
) -> list[dict]:
    """Search across one or both collections, optionally filtered by path.

    Args:
        query: Search query string.
        collection_name: "code", "docs", or None (all).
        n_results: Number of results to return.
        path_filter: Filter results to paths containing this substring.
        project: Search a specific project (default: all).
    """
    db = get_chromadb_client()
    config = load_config()
    provider = get_provider(config)

    projects = (
        {project: config.projects[project]}
        if project and project in config.projects
        else config.projects
    )

    targets = []
    for proj_name, proj in projects.items():
        if collection_name is None or collection_name == "code":
            targets.append((proj.code_collection, CODE_MODEL, proj_name))
        if collection_name is None or collection_name == "docs":
            # Always attempt -docs (skip-on-miss at get_collection below, like
            # code). F populates -docs from in-repo .md even for projects with no
            # configured docs_dir, so gating on proj.docs_dir would leave those
            # collections permanently unsearched.
            targets.append((proj.docs_collection, DOCS_MODEL, proj_name))

    # 1.5c model-flip interlock: drop from the VECTOR path any target whose
    # recorded embed model differs from the configured model — its stored
    # vectors live in a different space, so scoring new-model query vectors
    # against them silently corrupts ranking. The collection still participates
    # in BM25 below (model-agnostic), the safe fallback until a reindex
    # re-embeds it. A None marker (unknown) is fail-open: the vector path keeps
    # it. Markers read ONCE here (one cache open), not per fetch-multiplier retry
    # and not per target.
    vector_targets = []
    markers = _collection_markers([t[0] for t in targets])
    for tgt in targets:
        col_name, model, _ = tgt
        marker = markers.get(col_name)
        if marker is not None and marker != model:
            _warn(
                f"model-flip interlock: '{col_name}' embedded under {marker!r} "
                f"but configured model is {model!r}; skipping vector scoring, "
                f"BM25-only for this collection until reindex"
            )
            continue
        vector_targets.append(tgt)

    # Fetch with escalating multiplier: try 2x first, then 3x if dedup eats too many
    for fetch_multiplier in (2, 3):
        fetch_n = n_results * fetch_multiplier

        all_results = []
        for col_name, model, proj_name in vector_targets:
            try:
                collection = db.get_collection(col_name)
            except Exception:
                continue

            embedding = _cached_embed(provider, query, model)

            where = None
            if path_filter:
                where = {"file_path": {"$contains": path_filter}}

            try:
                raw = collection.query(
                    query_embeddings=[embedding],
                    n_results=fetch_n,
                    include=["documents", "metadatas", "distances"],
                    where=where,
                )
            except Exception:
                # tolerate a where-clause failure on a collection that lacks
                # file_path metadata rather than aborting the whole search
                if path_filter:
                    continue
                raise

            results = format_results(raw)
            for r in results:
                r["collection"] = col_name
                r["project"] = proj_name
            all_results.extend(results)

        # BM25 keyword search (cached, graceful degradation on failure)
        bm25_results = []
        bm25_dir = VECS_DIR / "bm25"
        for col_name, model, proj_name in targets:
            suffix = "code" if col_name.endswith("-code") else "docs"
            bm25_path = bm25_dir / f"{proj_name}_{suffix}.db"
            bm25 = get_bm25(bm25_path)
            if bm25 is not None:
                hits = bm25.search(query, n=fetch_n, path_filter=path_filter)
                for h in hits:
                    h["collection"] = col_name
                    h["project"] = proj_name
                bm25_results.extend(hits)

        if bm25_results:
            all_results = reciprocal_rank_fusion(all_results, bm25_results)
        else:
            all_results.sort(key=lambda r: r.get("distance") or float("inf"))

        all_results = deduplicate_results(all_results)

        if len(all_results) >= n_results or fetch_multiplier == 3:
            break

    return all_results[:n_results]
