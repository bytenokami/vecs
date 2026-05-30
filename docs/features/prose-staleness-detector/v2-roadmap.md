# prose-staleness-detector — v2 roadmap

Feature-scoped deferrals for **prose-drift**. Platform-level directions (bundle assembly, code graph, hierarchical retrieval) live in `docs/vecs-roadmap.md`.

Each entry is a sketch, not a contract. v1 shipped 2026-05-30 (47/47 acceptance, live-LLM verified). The recall/cost items below are sourced from the architecture review at `docs/research/prose-drift-review-and-sota-2026-05-29.md`.

## Stage-2 recall — paraphrase / cross-predicate (highest value) — ✅ SHIPPED 2026-05-30

Design: `stage2-recall-design.md`. On a `chain_key` MISS, `find_prose_drift` embeds the doc triple, takes the single most cosine-similar `is_current` fact (`STAGE2_SIM_THRESHOLD = 0.85`), and escalates that one pair to ONE Opus contradiction-judge (`PROSE_JUDGE_MODEL`), cached in `judge_cache`. Positive verdict → a `match_type="semantic"` drift entry carrying `similarity` + `confidence`. This is Graphiti's per-edge invalidation method **without** a graph DB. The per-row Voyage embedding (previously dead weight on the collision path) is now load-bearing. `test_cross_predicate_paraphrase_drift_is_detected` was promoted from `xfail(strict)` to a passing test.

Deferred from this increment (still open below): a calibrated confidence band to replace the raw judge `confidence` float; top-k candidates per miss (shipped as top-1); a cost ceiling on judge calls.

## Valid-time axis — soft / temporal contradictions

v1 stores ONE timeline (`valid_from` = ingestion / transaction time). Add a second axis (`valid_at` / `invalid_at` = when the fact was true in the world, LLM-extracted) for the full Snodgrass bi-temporal model (matches Zep/Graphiti). Catches "we *used to have* a BE dev" temporal contradictions that a single timeline silently overwrites.

## Drift-confidence score (replace the boolean)

v1 emits boolean drift. ML-observability norm is a score with calibrated thresholds (PSI 0.1 stable / 0.25 significant) + dual-detector confirmation. Relevant once the stage-2 LLM judge exists — a confidence band suppresses judge false positives (the spike's zero-false-positive bar).

## Fact-store — SQLite migration OR drop the dead embedding write

v1 writes a Voyage embedding on every INSERT/SUPERSEDE that the pure-collision path never reads (dead weight; verified). Either (a) the stage-2 similarity fallback makes it load-bearing, or (b) migrate the fact store to SQLite (already in vecs; natively expresses the current-view + time-travel without the `is_current` companion field and `invalid_at=0` sentinel Chroma workarounds). Also adds the `extracted_at` columns the design DDL lists but the impl omits (gap `sinker-w1.1`), enabling compaction.

## Extraction model + cost metering

v1 calls Opus 4.7 on every extraction — the most expensive point on the curve; the ~$0.80/mo projection is UNMEASURED. Add call-count metering; run an extraction-accuracy spike on a labeled fixture set before claiming production scale; reconsider Sonnet as the extraction default, reserving Opus for the escalated contradiction-judge (matches the SOTA cheap-retrieval → selective-judge cost curve).

## Time-travel query (`--as-of <ISO-date>`)

The bi-temporal columns `valid_from` / `invalid_at` already support point-in-time queries; only the CLI/MCP surface is missing.

Sketch:
- `vecs prose-drift -p <project> --as-of <ISO-date>` filters facts by `valid_from <= ts AND (invalid_at == 0 OR invalid_at > ts)` where `ts = parse_iso(<ISO-date>)` in epoch ms. Returns drift as of `ts` rather than now.
- MCP variant: `mcp__vecs__prose_drift(project, as_of: str | None = None)`.

Open questions: ISO-8601 parser (stdlib `datetime.fromisoformat` vs `dateutil`); timezone handling for naive datetimes (assume UTC vs reject); read-path tests for before/mid/after-chain brackets.

## Symmetric doc-side persistence

If v2 writes a `<project>-prose-facts` row per doc-extracted triple (mirroring the session path), reintroduce the `source_type: "session" | "doc"` field (removed in v1 per Fix 4). A one-shot migration backfills `"session"` on every v1-era row.

## Compaction

`<project>-prose-facts` grows monotonically under v1 (no deletes). v2 may add `vecs prose-drift compact --older-than <days>` to prune superseded rows older than the threshold while preserving the current row per chain. Depends on the `extracted_at` column landing (see fact-store above).

## Cost ceiling

Optional env-var `VECS_PROSE_DRIFT_MAX_CALLS_PER_DAY` to short-circuit extraction once the daily budget is exhausted. v1 ships without a ceiling.

## Assistant-turn re-enablement

v1 hard-excludes `role != "user"` turns from extraction (hallucination risk). v2 may re-enable assistant turns gated by a pre-classifier.
