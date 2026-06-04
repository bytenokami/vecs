Authored by Claude

# Acceptance — Increment 1-instrumentation (A + E)

Format: checklist. Pass threshold: all-pass. Parent: `docs/features/kb-foundations/design.md`; program §6.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-foundations-instrumentation`.
Note: E depends on C (`version_id`) from kb-foundations-pipeline.

## A — metering spike

- [x] Every extraction/judge LLM call emits a cost record (model, input/output tokens, USD). — `metering.metered_create` chokepoint wraps all 3 `prose_drift` calls; `record_call` writes `{date,model,in/out tokens,usd}`. Tests: `test_metering.py` + `test_find_prose_drift_meters_doc_extraction_and_judge` (judge+doc halves).
- [x] A `MAX_CALLS_PER_DAY` cap exists and is enforced: extraction stops at the cap and logs that it stopped. — `metered_create` raises `MeteringCapExceeded` before the API call when capped; `find_prose_drift` catches it, sets `cap_hit=True`, logs to stderr. Tests: `test_metering_create_raises_and_skips_api_when_capped` + `test_find_prose_drift_real_metered_cap_stops_and_logs` (capsys) + `..._judge_cap_is_not_swallowed_by_inner_except`.
- [x] Default extraction model is Sonnet (overridable), not Opus. — `PROSE_EXTRACTION_MODEL = "claude-sonnet-4-6"` (`test_prose_extraction_model_constant_is_pinned`); judge stays Opus.
- [x] A per-project est-cost-to-populate-facts report is produced and committed under `docs/features/kb-foundations-instrumentation/`. — `est-cost-to-populate-facts.md` (vecs: measured ~$1.12 one-pass at Sonnet).
- [x] Metering is documented as a prerequisite instrument (it does not auto-gate Inc 2/6; a cost-ceiling kill criterion, if wanted, is a §7 decision). — report §"Not a gate" + `metering.py` module docstring.

## E — measurement-harness seed

- [x] `stale-retrieval-rate` is computable over a collection, defined against the per-chunk `version_id`/embed-hash anchor, with a graceful "legacy/unknown" bucket for chunks not yet re-stamped. — `eval_harness.collection_stale_rate` / `stale_stats_for_chunks` / `bucket_chunk`; unknown excluded from `rate`; current version reuses the indexer's exact anchors. Tests: `test_eval_harness.py`.
- [x] A small local eval-set scaffold (query → expected source/answer) exists with a runner stub that reports the metric. — `DEFAULT_EVAL_SET` + `run_eval` (injectable `search_fn`); reports `hit_rate` + retrieved-chunk `StaleStats`.

## Global (this sub-feature)

- [x] `uv run pytest -q` green; new/updated tests for the metering + harness helpers. — 382 passed / 2 skipped.
- [x] `src/vecs/CLAUDE.md` updated for touched modules. — `eval_harness`, `metering` rows + stale-rate & metering invariants.
- [x] Phase 4 review verdict = approve. — 4-dimension review + adversarial verify: 7 raw → 4 confirmed, all LOW (no live defect; test/doc-coverage gaps), all addressed (judge-path metering + cap tests, capsys log assert, `cap_hit` docstrings).
