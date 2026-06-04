Authored by Claude

# Retrospective: kb-freshness-hotfix

Filled per `docs/workflow-vecs-profile-v0.1.md` Phase 8. One retro per feature.

Scope: Increment 1.5 — three independent freshness/correctness fixes shipped before Inc 2:
- 1.5a — prune-orphan fix (`34e7a8a`)
- 1.5b — searcher `-docs` gate one-liner (`ab9c4fe`)
- 1.5c — freshness/trust signal + query-time model-flip interlock (`53dd5fe`)

(1.5a/1.5b executed in prior sessions — notes below drawn from their commits + the landed code; 1.5c executed directly this session.)

## What worked
- **Splitting one "hotfix" into three independent sub-features** kept each change small, separately testable, and individually revertable. None depended on Inc 4a's `valid_from`/`valid_to`, so the correctness fixes shipped without waiting on the lifecycle work.
- **Reusing already-shipped machinery.** 1.5a leaned on the existing deletion/BM25-sidecar paths; 1.5c's interlock leaned on the `EmbedCache.collection_models` marker (Inc 1-B2) and the per-chunk `version_id` (Inc 1-C) — both already in place, so the surface change was thin.
- **Phase-4 adversarial review with a per-finding verify pass earned its keep on 1.5c.** The verifier *empirically disproved* the headline severity of the hot-path finding (it actually tested the real `EmbedCache` against a held write lock and measured ~0.1ms, not the claimed up-to-5s block) — preventing an over-weighted fix — while confirming two real test-coverage gaps that a single-pass review would likely have stated with false confidence either way.
- **Triaging findings in the main thread, not blindly applying them.** All three confirmed findings collapsed into one coherent change (open-once `_collection_markers`) rather than three patches.

## What didn't
- **The first 1.5c interlock cut opened a fresh `EmbedCache` per target** instead of once per search — a divergence from the codebase's established one-cache-per-operation lifecycle that the review caught. The harm was negligible in practice, but it was avoidable up front by checking how the indexer manages cache lifetime before writing the search-side version.
- **The first test pass stubbed `_collection_model` in every interlock test**, leaving the load-bearing fail-open-on-error branch with zero real-function coverage. A "test the seam, not just the consumers" instinct would have caught this without needing the review to flag it.
- **Acceptance drift from the session rip.** Two 1.5a acceptance lines (session-chunk deletion, `-sessions` collection) describe machinery that commit `5dedb17` later removed. The boxes were left as superseded history rather than rewritten — correct for an audit trail, but the acceptance doc no longer cleanly matches the shipped system for 1.5a.

## Phase-by-phase notes

- Phase 1 — Acceptance: `acceptance.md` pre-written with the 1.5c section before build; checklist mapped 1:1 to tests. Clean.
- Phase 2 — Context: `src/vecs/CLAUDE.md` updated for both 1.5c invariants (freshness surface + interlock) and again after the review-driven refactor renamed `_collection_model` → `_collection_markers`.
- Phase 3 — Pipeline: n/a for 1.5c (no indexing-pipeline change; query-side only).
- Phase 4 — Review Loop: 4-dimension Workflow (interlock-correctness / freshness-tag / regression-hotpath / test-acceptance) → per-finding adversarial verify. 4 raw → 3 confirmed (1 low hot-path cleanup, 1 medium test gap, 1 low test gap). All triaged in-thread and addressed in the same commit. No critical/high; verdict = approve.
- Phase 5 — Tests: TDD red→green throughout. `test_mcp_server.py` created (freshness surface); `test_searcher.py` extended (interlock + real `_collection_markers` fail-open). 345 passed / 2 skipped.
- Phase 6 — Roster: n/a (no new agent roster for this feature).
- Phase 7 — Dry-Run: full `uv run pytest -q` green + `ruff check` clean on touched src before commit.

## Gap log
See `docs/features/kb-freshness-hotfix/gaps.md`.

## Feedback applied

| Target phase | Artifact | Owner |
|---|---|---|
| Phase 2 — Context | `src/vecs/CLAUDE.md`: added interlock + freshness invariants; corrected `_collection_model` → `_collection_markers` after refactor | self |
| Phase 5 — Tests | Added real-function `_collection_markers` fail-open test + per-collection mixed-marker test (Phase-4 findings) | self |
| Phase 3 — Pipeline (searcher) | `searcher.py`: open one `EmbedCache` per search, not per target (Phase-4 hot-path finding) | self |
