"""A/B benchmark: FTS5 BM25 (current) vs rank_bm25 (legacy) on a real corpus.

Usage:
    .venv/bin/python scripts/bench_bm25.py [--project PROJECT] [--suffix code|sessions|docs]

Reads chunks from the project's chromadb collection, builds both indexes
in tmpdirs, runs a fixed query set, and prints:

  - top-K overlap (Jaccard of result sets at K=5 and K=10)
  - per-query latency (median, p95)
  - bulk build / single-doc upsert latency

Requires `rank_bm25` to be installed temporarily for the comparison:
    uv pip install rank_bm25
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecs.bm25_index import BM25Index, _tokenize  # noqa: E402
from vecs.clients import get_chromadb_client  # noqa: E402
from vecs.config import load_config  # noqa: E402
from vecs.indexer import _paginated_get  # noqa: E402

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank_bm25 not installed; run: uv pip install rank_bm25", file=sys.stderr)
    sys.exit(2)


CANONICAL_QUERIES = [
    "BM25 index",
    "rebuild bm25",
    "tokenize camelCase",
    "VECS_DIR",
    "reciprocal rank fusion",
    "ChromaDB collection",
    "voyage embed",
    "HTTPSConnection",
    "test_indexer",
    "_sync_bm25",
]


def fetch_chunks(collection_name: str) -> list[dict]:
    db = get_chromadb_client()
    col = db.get_collection(collection_name)
    out = []
    for page in _paginated_get(col, include=["documents", "metadatas"]):
        for id_, text, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            out.append({"id": id_, "text": text, "metadata": meta or {}})
    return out


def build_legacy(docs: list[dict]) -> tuple[BM25Okapi, list[str], list[dict]]:
    ids = [d["id"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]
    tokenized = [_tokenize(d["text"]) for d in docs]
    return BM25Okapi(tokenized), ids, metas


def search_legacy(b25, ids, metas, query: str, n: int) -> list[str]:
    scores = b25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [ids[i] for i, s in ranked if s > 0]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def time_call(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--suffix", default="code", choices=["code", "sessions", "docs"])
    args = ap.parse_args()

    cfg = load_config()
    if args.project:
        proj = cfg.projects[args.project]
    else:
        proj = next(iter(cfg.projects.values()))

    if args.suffix == "code":
        col_name = proj.code_collection
    elif args.suffix == "sessions":
        col_name = proj.sessions_collection
    else:
        col_name = proj.docs_collection

    print(f"Loading chunks from {col_name}…", file=sys.stderr)
    docs = fetch_chunks(col_name)
    print(f"  {len(docs)} chunks", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        # Build legacy
        print("\n# Build phase", file=sys.stderr)
        (legacy, l_ids, l_metas), legacy_build_ms = time_call(build_legacy, docs)
        print(f"  legacy (rank_bm25 build): {legacy_build_ms:.1f} ms")

        # Build FTS5
        fts_path = Path(tmp) / "fts.db"
        idx = BM25Index(fts_path)
        _, fts_build_ms = time_call(idx.build, docs)
        print(f"  fts5   (build full):      {fts_build_ms:.1f} ms")

        # Single-doc upsert latency (fts5 only — legacy has no incremental API)
        sample = docs[len(docs) // 2]
        _, upsert_ms = time_call(idx.upsert, [sample])
        print(f"  fts5   (single upsert):   {upsert_ms:.2f} ms")
        print(f"  fts5   (single delete):   ", end="")
        _, delete_ms = time_call(idx.delete, [sample["id"]])
        print(f"{delete_ms:.2f} ms")
        idx.upsert([sample])  # restore for query phase

        # Query phase
        print("\n# Query phase (top-K overlap and latency)", file=sys.stderr)
        print(f"\n{'query':<28} {'k=5 jacc':<10} {'k=10 jacc':<10} {'leg ms':<8} {'fts ms':<8}")
        leg_lat, fts_lat = [], []
        overlap_5, overlap_10 = [], []
        for q in CANONICAL_QUERIES:
            l5, l_ms_5 = time_call(search_legacy, legacy, l_ids, l_metas, q, 5)
            l10, l_ms_10 = time_call(search_legacy, legacy, l_ids, l_metas, q, 10)
            f5_results, f_ms_5 = time_call(idx.search, q, 5)
            f10_results, f_ms_10 = time_call(idx.search, q, 10)
            f5 = [r["id"] for r in f5_results]
            f10 = [r["id"] for r in f10_results]
            j5, j10 = jaccard(l5, f5), jaccard(l10, f10)
            overlap_5.append(j5); overlap_10.append(j10)
            leg_lat.append(l_ms_5); fts_lat.append(f_ms_5)
            print(f"{q:<28} {j5:<10.2f} {j10:<10.2f} {l_ms_5:<8.2f} {f_ms_5:<8.2f}")

        print(f"\n# Summary")
        print(f"  mean top-5 overlap:  {statistics.mean(overlap_5):.2f}")
        print(f"  mean top-10 overlap: {statistics.mean(overlap_10):.2f}")
        print(f"  legacy median ms:    {statistics.median(leg_lat):.2f}")
        print(f"  fts5   median ms:    {statistics.median(fts_lat):.2f}")
        idx.close()


if __name__ == "__main__":
    main()
