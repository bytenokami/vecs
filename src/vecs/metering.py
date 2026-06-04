"""Inc 1-A: LLM-call metering spike.

A thin instrument around the prose-drift extraction/judge LLM calls: every real
(cache-miss) call emits a cost record (model, input/output tokens, USD), and a
``MAX_CALLS_PER_DAY`` cap stops further calls once reached. This is a SPIKE, not
a dashboard — records append to a human-inspectable JSONL under
``~/.vecs/metering/`` and pricing is a small static prefix-matched table.

It is a prerequisite *instrument* (gives us per-call cost + a daily ceiling); it
deliberately does NOT auto-gate Inc 2/6. A cost-ceiling kill criterion, if
wanted, is a §7 program decision, not something baked in here.

No import of ``prose_drift`` (which imports this) — keep the dependency one-way.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vecs.config import VECS_DIR

# Hard daily ceiling on real (cache-miss) LLM calls. Env-overridable for ops.
MAX_CALLS_PER_DAY = int(os.environ.get("VECS_MAX_CALLS_PER_DAY", "500"))

DEFAULT_METERING_PATH = VECS_DIR / "metering" / "calls.jsonl"

# USD per 1M tokens (input, output), matched by model-family prefix (longest
# prefix wins). VERIFY against current Anthropic list pricing before trusting the
# cost report — these are list prices noted 2026-06 and are rough by design (a
# spike instrument). A stale/missing entry over- or under-states cost but NEVER
# blocks a call (unknown -> priced at 0, i.e. unpriced).
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
}
_UNKNOWN_PRICE = (0.0, 0.0)


class MeteringCapExceeded(RuntimeError):
    """Raised by :func:`metered_create` when the daily call cap is reached
    (before the API is called — no record, no spend)."""


def _price_for(model: str) -> tuple[float, float]:
    best = ""
    for prefix in PRICING_USD_PER_MTOK:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return PRICING_USD_PER_MTOK.get(best, _UNKNOWN_PRICE)


def price_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one call at list pricing (0 for an unpriced model family)."""
    pin, pout = _price_for(model)
    return (input_tokens / 1_000_000) * pin + (output_tokens / 1_000_000) * pout


def _today() -> str:
    """UTC date 'YYYY-MM-DD'. Module-level so tests can monkeypatch it."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def _store(store_path: Path | None) -> Path:
    return store_path if store_path is not None else DEFAULT_METERING_PATH


def calls_today(store_path: Path | None = None) -> int:
    """Count of recorded calls dated today (malformed lines ignored)."""
    path = _store(store_path)
    if not path.exists():
        return 0
    today = _today()
    n = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date") == today:
                n += 1
    return n


def record_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    store_path: Path | None = None,
) -> dict:
    """Append a cost record and return it."""
    path = _store(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "date": _today(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": round(price_call(model, input_tokens, output_tokens), 6),
    }
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def metered_create(client, *, store_path: Path | None = None, **create_kwargs):
    """Chokepoint around ``client.messages.create``: enforce the daily cap, then
    record the call's token/USD cost.

    Raises :class:`MeteringCapExceeded` BEFORE touching the API once the day's
    recorded calls reach :data:`MAX_CALLS_PER_DAY` (no record, no spend). On
    success, reads ``resp.usage`` and writes a cost record.
    """
    if calls_today(store_path) >= MAX_CALLS_PER_DAY:
        raise MeteringCapExceeded(
            f"daily LLM call cap reached ({MAX_CALLS_PER_DAY}); skipping "
            f"{create_kwargs.get('model', '?')} call"
        )
    resp = client.messages.create(**create_kwargs)
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    record_call(create_kwargs.get("model", "?"), in_tok, out_tok, store_path)
    return resp


def estimate_extraction_cost(
    input_tokens: int, output_tokens: int, model: str
) -> dict:
    """Cost breakdown for a hypothetical extraction volume (the est-cost report
    helper). Pure: just prices the given token totals under ``model``."""
    return {
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "usd": round(price_call(model, input_tokens, output_tokens), 4),
    }
