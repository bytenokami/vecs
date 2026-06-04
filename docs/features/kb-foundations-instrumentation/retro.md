Authored by Claude

# Retrospective: kb-foundations-instrumentation

Filled per `docs/workflow-vecs-profile-v0.1.md` Phase 8. One retro per feature.

Scope: Increment 1-instrumentation (E + A), shipped together (`e6bd9a8`):
- E — stale-retrieval-rate harness + eval-set scaffold (`src/vecs/eval_harness.py`)
- A — LLM-call metering spike (`src/vecs/metering.py`) wired into `prose_drift`

## What worked
- **Grounding the metric in the indexer's exact anchors before writing E.** Reading `_git_sha`, `Manifest._file_hash` (sha256 of *bytes*, not text), and the docs `str(mtime)` stamping up front meant `_current_version_id` reproduces "current version" by the same rule the indexer stamps with — so a fresh chunk can't be misclassified as stale by a hash-flavour mismatch. The review's E dimension found nothing real.
- **A single metering chokepoint (`metered_create`).** Routing all three `prose_drift` calls through one wrapper made "every call emits a record" + the cap a two-line change per site, and kept the cap logic in one testable place.
- **Designing for offline testability.** Pure `bucket_chunk`, injectable `_current_version_id` / `search_fn`, an injectable metering `store_path` — the whole feature tests with no chroma, no git, no network, no real LLM.
- **The adversarial-verify pass kept severities honest again.** All 7 raw findings that survived to verification were correctly graded LOW (no live defect); the verifiers confirmed the production code was right and the gaps were test/doc coverage — so the fixes were tests + docstrings, not code churn.

## What didn't
- **First test pass under-asserted the metered paths.** `find_prose_drift` actually calls `extract_facts_from_doc` (doc path) and the stage-2 judge, but the initial record-emission tests only covered the session-message `extract_facts` path — so the judge half of acceptance A line 1, and the doc path's routing through `metered_create`, were exercised-but-unasserted. Caught by the review (findings 3, 4); closed with judge/doc-path + end-to-end cap tests. Lesson: assert the path the *orchestrator* uses, not just the nearest unit.
- **The cap-stop's correctness rests on an implicit invariant** (`MeteringCapExceeded` being a `RuntimeError` that stays outside the judge's narrow `except (JSONDecodeError, KeyError, ...)`), which was initially unpinned. A test now nails it down.

## Phase-by-phase notes

- Phase 1 — Acceptance: `acceptance.md` (A: 5, E: 2, Global: 3) pre-written; mapped each box to a test/artifact at close.
- Phase 2 — Context: `src/vecs/CLAUDE.md` — added `eval_harness`/`metering` module rows + the stale-rate and metering invariants (including the code-HEAD coarseness note and the one-way `metering`↛`prose_drift` dep).
- Phase 3 — Pipeline: n/a (no indexing-pipeline change; E reads collections read-only, A wraps existing LLM calls).
- Phase 4 — Review Loop: 4-dimension Workflow (E-metric / A-metering / regression-wiring-isolation / test-acceptance) → per-finding adversarial verify. 7 raw → 4 confirmed, all LOW, all addressed in-thread. Verdict = approve.
- Phase 5 — Tests: TDD throughout. New `test_eval_harness.py` (21), `test_metering.py` (10); `test_prose_drift.py` extended (metering wiring + judge/doc/e2e cap tests) with autouse metering-store isolation. 382 passed / 2 skipped.
- Phase 6 — Roster: n/a.
- Phase 7 — Dry-Run: full `uv run pytest -q` green + `ruff check src/` clean before commit; cost report measured against the real vecs doc corpus.

## Gap log
See `docs/features/kb-foundations-instrumentation/gaps.md`.

## Feedback applied

| Target phase | Artifact | Owner |
|---|---|---|
| Phase 5 — Tests | judge-path + doc-path metering-record tests; judge-cap-not-swallowed test; e2e real-`metered_create` cap-stop + stderr-log (capsys) test (Phase-4 findings 1/3/4) | self |
| Phase 2 — Context | `find_prose_drift` + `mcp_server.prose_drift` docstrings updated with the new `cap_hit` key (Phase-4 finding 2) | self |
