# prose-staleness-detector — v2 roadmap (placeholder)

Items deferred from v1. Each entry is a sketch, not a contract. The v2 design doc reopens these when v1 ships and stabilizes.

## Time-travel query (`--as-of <ISO-date>`)

Deferred from v1 per Fix 6 / MAJOR 5. The bi-temporal columns `valid_from` and `invalid_at` on `<project>-prose-facts` already support point-in-time queries; only the CLI/MCP surface is missing.

Sketch:
- `vecs prose-drift -p <project> --as-of <ISO-date>` filters facts by `valid_from <= ts AND (invalid_at IS NULL OR invalid_at > ts)` where `ts = parse_iso(<ISO-date>).timestamp_ms`.
- Returns drift as of `ts` rather than now.
- MCP variant: `mcp__vecs__prose_drift(project, as_of: str | None = None)`.

Open questions for v2:
- ISO-8601 parser: stdlib `datetime.fromisoformat` vs `dateutil.parser.isoparse`.
- Timezone handling for naive datetimes (assume UTC vs reject vs prompt operator).
- Read-path tests: at least three time-bracket cases (before chain, mid-chain, after chain).

## Symmetric doc-side persistence

If v2 introduces a `<project>-prose-facts` row per doc-extracted triple (mirroring the session path), the `source_type: "session" | "doc"` field is reintroduced. A one-shot migration backfills the field as `"session"` on every v1-era row.

## Compaction

`<project>-prose-facts` grows monotonically under v1. v2 may add `vecs prose-drift compact --older-than <days>` to prune superseded rows older than the threshold while preserving the most-recent current row per chain.

## Cost ceiling

Optional env-var `VECS_PROSE_DRIFT_MAX_CALLS_PER_DAY` to short-circuit extraction once the daily call budget is exhausted. v1 ships without a ceiling.

## Assistant-turn re-enablement

v1 hard-excludes `role != "user"` turns from extraction (hallucination risk). v2 may re-enable assistant turns gated by a pre-classifier.
