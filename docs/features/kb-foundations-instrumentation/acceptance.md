# Acceptance — Increment 1-instrumentation (A + E)

Format: checklist. Pass threshold: all-pass. Parent: `docs/features/kb-foundations/design.md`; program §6.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-foundations-instrumentation`.
Note: E depends on C (`version_id`) from kb-foundations-pipeline.

## A — metering spike

- [ ] Every extraction/judge LLM call emits a cost record (model, input/output tokens, USD).
- [ ] A `MAX_CALLS_PER_DAY` cap exists and is enforced: extraction stops at the cap and logs that it stopped.
- [ ] Default extraction model is Sonnet (overridable), not Opus.
- [ ] A per-project est-cost-to-populate-facts report is produced and committed under `docs/features/kb-foundations-instrumentation/`.
- [ ] Metering is documented as a prerequisite instrument (it does not auto-gate Inc 2/6; a cost-ceiling kill criterion, if wanted, is a §7 decision).

## E — measurement-harness seed

- [ ] `stale-retrieval-rate` is computable over a collection, defined against the per-chunk `version_id`/embed-hash anchor, with a graceful "legacy/unknown" bucket for chunks not yet re-stamped.
- [ ] A small local eval-set scaffold (query → expected source/answer) exists with a runner stub that reports the metric.

## Global (this sub-feature)

- [ ] `uv run pytest -q` green; new/updated tests for the metering + harness helpers.
- [ ] `src/vecs/CLAUDE.md` updated for touched modules.
- [ ] Phase 4 review verdict = approve.
