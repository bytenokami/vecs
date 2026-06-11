"""L2 perf spike (local-embed-ab): measure Qwen3 embedding throughput on Apple
Silicon with REAL chunks from the live store, per design.md L2.1.

Phases per model: load (timed), warmup, burst per batch size, then a SUSTAINED
leg (default 20 min continuous) whose tokens/sec is the projection basis — a
minutes-long burst overstates an hours-long reindex (thermal throttling,
sustained memory pressure). Peak RSS via ru_maxrss.

Run ONE MODEL PER PROCESS for clean RSS attribution:
    uv run python scripts/spike_local_embed.py --model qwen3-embedding-0.6b --out spike-0.6b.json
    uv run python scripts/spike_local_embed.py --model qwen3-embedding-4b@mrl1024 --out spike-4b.json

Requires the vecs[local] extra (uv sync --extra local). Reads chunk text from
the local chroma store only; writes an aggregate-only JSON report (token
counts and timings, no chunk text) — safe to export off the work mac.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path


def _sample_chunks(collections: list[str], n: int) -> list[str]:
    """Evenly-spaced sample of chunk documents across the given collections."""
    from vecs.clients import get_chromadb_client

    db = get_chromadb_client()
    texts: list[str] = []
    per_col = max(1, n // len(collections))
    for name in collections:
        try:
            col = db.get_collection(name)
        except Exception:
            print(f"  collection {name} missing, skipping")
            continue
        total = col.count()
        if total == 0:
            continue
        step = max(1, total // per_col)
        got: list[str] = []
        offset = 0
        while offset < total and len(got) < per_col:
            batch = col.get(limit=1, offset=offset, include=["documents"])
            docs = batch.get("documents") or []
            got.extend(d for d in docs if d)
            offset += step
        texts.extend(got)
        print(f"  {name}: sampled {len(got)}/{total}")
    return texts


def _batches(texts: list[str], size: int):
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


def _timed_pass(provider, model: str, texts: list[str], batch_size: int) -> dict:
    t0 = time.monotonic()
    tokens = 0
    for batch in _batches(texts, batch_size):
        out = provider.embed(batch, model=model, input_type="document")
        tokens += out.total_tokens or 0
    dt = time.monotonic() - t0
    return {"seconds": round(dt, 2), "tokens": tokens,
            "tokens_per_sec": round(tokens / dt, 1) if dt > 0 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--collections", default="livly-code,livly-docs")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--batch-sizes", default="16,64")
    ap.add_argument("--sustained-minutes", type=float, default=20.0)
    ap.add_argument("--project-tokens", type=int, default=40_000_000,
                    help="full-corpus token count to project reindex wall-clock for")
    ap.add_argument("--out", default="spike-report.json")
    args = ap.parse_args()

    from vecs.embed_provider import QwenLocalProvider

    report: dict = {"model": args.model, "machine": {}, "phases": {}}
    import platform
    import subprocess
    report["machine"] = {
        "platform": platform.platform(),
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
    }

    print(f"sampling ~{args.sample} chunks from {args.collections} ...")
    texts = _sample_chunks(args.collections.split(","), args.sample)
    if not texts:
        raise SystemExit("no chunks sampled — wrong machine or empty store?")
    report["sample"] = {"n_chunks": len(texts),
                        "avg_chars": round(sum(map(len, texts)) / len(texts))}

    provider = QwenLocalProvider()
    print(f"loading {args.model} (first run downloads from HF) ...")
    t0 = time.monotonic()
    provider.embed(texts[:2], model=args.model, input_type="document")  # load+warmup
    report["phases"]["load_plus_first_batch_s"] = round(time.monotonic() - t0, 1)

    best = None
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        print(f"burst @ batch={bs} ...")
        r = _timed_pass(provider, args.model, texts, bs)
        report["phases"][f"burst_batch{bs}"] = r
        if best is None or (r["tokens_per_sec"] or 0) > (report["phases"][f"burst_batch{best}"]["tokens_per_sec"] or 0):
            best = bs
    report["best_batch_size"] = best

    print(f"sustained leg: {args.sustained_minutes} min @ batch={best} ...")
    t0 = time.monotonic()
    tokens = 0
    laps = 0
    deadline = t0 + args.sustained_minutes * 60
    while time.monotonic() < deadline:
        for batch in _batches(texts, best):
            out = provider.embed(batch, model=args.model, input_type="document")
            tokens += out.total_tokens or 0
            if time.monotonic() >= deadline:
                break
        laps += 1
    dt = time.monotonic() - t0
    tps = tokens / dt if dt > 0 else 0
    report["phases"]["sustained"] = {
        "minutes": round(dt / 60, 1), "tokens": tokens, "laps_over_sample": laps,
        "tokens_per_sec": round(tps, 1),
    }
    report["projection"] = {
        "corpus_tokens": args.project_tokens,
        "full_reindex_hours_at_sustained": round(args.project_tokens / tps / 3600, 2) if tps else None,
    }
    # macOS ru_maxrss is BYTES
    report["peak_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2
    )

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
