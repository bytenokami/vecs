"""L2 shadow A/B (local-embed-ab): build qwen shadow collections from the LIVE
store's chunk texts and score both arms on a golden set.

Why stored texts (not a disk tree): the live collections ARE the voyage arm's
snapshot; re-embedding the same stored documents under qwen gives both arms an
identical corpus by construction (no staleness skew, no Voyage spend, work
content never leaves the machine). Shadow chunks reuse the SAME ids/metadata,
so the shared BM25 sidecars join either arm's vector results identically.

Usage (work mac, vecs[local] synced):
  uv run python scripts/ab_local_embed.py build-shadow \
      --collections livly-code,livly-docs --model qwen3-embedding-4b@mrl1024 \
      --suffix qwen4b --batch 32
  uv run python scripts/ab_local_embed.py run \
      --golden ~/.vecs/evalsets/livly.yaml --project livly \
      --model qwen3-embedding-4b@mrl1024 --suffix qwen4b --out ab-report.json

The report is aggregate-only (means/deltas/CIs per metric and query class) —
safe to export off the work mac.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _paginated(col, batch=500):
    offset = 0
    while True:
        got = col.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            return
        yield ids, got.get("documents") or [], got.get("metadatas") or []
        offset += len(ids)


def build_shadow(args) -> None:
    from vecs.clients import get_chromadb_client
    from vecs.embed_provider import get_provider

    db = get_chromadb_client()
    provider = get_provider(name="qwen-local")
    for name in args.collections.split(","):
        src = db.get_collection(name)
        shadow_name = f"{name}-{args.suffix}"
        shadow = db.get_or_create_collection(shadow_name)
        done = set()
        for ids, _docs, _metas in _paginated(shadow):
            done.update(ids)
        total = src.count()
        print(f"{shadow_name}: {len(done)}/{total} already present")
        t0 = time.monotonic()
        processed = 0
        for ids, docs, metas in _paginated(src):
            todo = [(i, d, m) for i, d, m in zip(ids, docs, metas) if i not in done and d]
            for j in range(0, len(todo), args.batch):
                window = todo[j : j + args.batch]
                out = provider.embed(
                    [d for _i, d, _m in window], model=args.model, input_type="document"
                )
                shadow.upsert(
                    ids=[i for i, _d, _m in window],
                    embeddings=out.embeddings,
                    documents=[d for _i, d, _m in window],
                    metadatas=[m for _i, _d, m in window],
                )
                processed += len(window)
                if processed % 500 < args.batch:
                    rate = processed / (time.monotonic() - t0)
                    eta_min = (total - len(done) - processed) / rate / 60 if rate else 0
                    print(f"  {shadow_name}: +{processed} ({rate:.1f} chunks/s, ~{eta_min:.0f} min left)")
        print(f"{shadow_name}: done, count={shadow.count()} (src {total})")


def _bm25_paths_for(project: str, suffix: str | None) -> dict:
    from vecs.config import VECS_DIR

    base = VECS_DIR / "bm25"
    out = {}
    for kind in ("code", "docs"):
        col = f"{project}-{kind}" + (f"-{suffix}" if suffix else "")
        out[col] = base / f"{project}_{kind}.db"
    return out


def run_ab(args) -> None:
    from vecs.eval_harness import ab_report, load_eval_set, run_arm
    from vecs.searcher import search, search_collections
    from vecs.embed_provider import get_provider

    golden = load_eval_set(Path(args.golden).expanduser())
    print(f"golden set: {len(golden)} cases")

    def arm_voyage(query, collection_name=None, n_results=10, project=None):
        return search(query, collection_name=collection_name,
                      n_results=n_results, project=project)

    provider = get_provider(name="qwen-local")
    bm25_paths = _bm25_paths_for(args.project, args.suffix)

    def arm_qwen(query, collection_name=None, n_results=10, project=None):
        targets = [(
            f"{args.project}-{collection_name}-{args.suffix}",
            args.model,
            args.project,
        )]
        return search_collections(query, targets, provider=provider,
                                  n_results=n_results, bm25_paths=bm25_paths,
                                  check_markers=False)

    print("scoring voyage arm (live collections) ...")
    a = run_arm(golden, arm_voyage)
    print("scoring qwen arm (shadow collections) ...")
    b = run_arm(golden, arm_qwen)
    report = ab_report(a, b)
    report["meta"] = {"golden_n": len(golden), "model_b": args.model,
                      "suffix": args.suffix, "project": args.project,
                      "note": "delta/ci = arm_b(qwen) minus arm_a(voyage)"}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-shadow")
    b.add_argument("--collections", default="livly-code,livly-docs")
    b.add_argument("--model", default="qwen3-embedding-4b@mrl1024")
    b.add_argument("--suffix", default="qwen4b")
    b.add_argument("--batch", type=int, default=32)
    r = sub.add_parser("run")
    r.add_argument("--golden", required=True)
    r.add_argument("--project", default="livly")
    r.add_argument("--model", default="qwen3-embedding-4b@mrl1024")
    r.add_argument("--suffix", default="qwen4b")
    r.add_argument("--out", default="ab-report.json")
    args = ap.parse_args()
    if args.cmd == "build-shadow":
        build_shadow(args)
    else:
        run_ab(args)


if __name__ == "__main__":
    main()
