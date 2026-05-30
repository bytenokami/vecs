# Retrospective: prose-staleness-detector v1 (V+) — Phase 7 dry-run

Filled per `docs/workflow-vecs-profile-v0.1.md` Phase 8. Scope: design pipeline (passes 1-4) + Phase 7 dry-run only. A second retro will follow Phase 4 (indexer wire-in) + Phase 5 (full test matrix) + Phase 6 (roster execution).

## What worked

- **Multi-pass review loop converged.** 4 design passes (20 → 4 → 3 → 0 BLOCKER findings) + 2 dry-run passes (7 → 0 findings). Each pass closed strictly more than it opened. No regressions across passes.
- **Spike-first beat synthesis-first.** Research synthesis tie-broke on Mem0 ("vibes"); empirical spike (Mem0 v2.0.2 ADD-only finding) invalidated the synthesis pick before any production code was written. Approach 1 spike (LLM-judge) and V+ pivot followed in hours, not weeks.
- **Bi-temporal vecs-only collapsed scope.** Zero new vendor deps (Mem0 not added). One new runtime dep: `anthropic`. State machine logic owned in-repo; Mem0's opaque update prompt traded for an explicit `extract_facts` + `add_fact_with_state_machine` pair.
- **Dry-run validated four boundaries in one subtask.** Opus 4.7 auth + no-temperature, INSERT/NOOP/SUPERSEDE end-to-end, SQLite verdict cache, Voyage `voyage-3` signature. Live integration test (`VECS_TEST_REAL_LLM=1`) passed first try on real API.
- **Workflow framework v0.1 held its shape.** Profile-driven phases (`dryrun_selection: smallest-real-subtask`, `dryrun_pass_criteria: [pipeline-pass, review-loop-satisfied]`, `abort_policy: branch-drop`) executed without amendment. Roster slots (`architect`/`builder-small`/`critical-sinker`/`reviewer`) mapped cleanly to available agents.

## What didn't

- **`m["content"]` vs `m["text"]` schema bug survived design pass 2.** Pass-1 wrote the cache helper using `m["content"]` but `preprocess_session` emits `m["text"]`. Caught at pass-3 sinker pass via cross-cite of `src/vecs/chunkers.py:46`. Lesson: structural cites must be verified against current code, not inferred from name similarity.
- **`source_type` field churned twice.** Specified in pass 1, removed in pass 1 Fix 4, lingered in two call sites until pass-4 surgical edits. Lesson: when a field is removed, grep for all call sites in the same edit cycle.
- **`INVALID_AT_NONE_SENTINEL = 0` deviation was undocumented at impl time.** Module shipped with sentinel; docs said `int | None`. Sinker caught at dry-run pass 1. Lesson: when implementation diverges from spec at code-write time (because Chroma rejects None), amend the spec in the same commit.
- **Multi-key Chroma `where` syntax not pinned until pass-4.** Initial design relied on flat-dict form; sinker raised "may need `$and` wrapping" three passes in. Lesson: vendor-specific filter syntax is a runtime constraint; pin via dry-run earlier, not by review.
- **Live API key handling was ad-hoc.** Operator pasted real key in chat (now rotated). `/tmp/.spike-anthropic-key` (mode 600) used as a one-off bridge for spike agents. Profile `pipeline_setup_runbook` has no canonical guidance on key storage. Documented as runbook addendum candidate (gaps.md #1).
- **Six worktree branches accumulated** during spike + design phases. Hook-blocked `git branch -D` requires operator hand-cleanup. Lesson: profile should specify worktree-cleanup as a Phase-8 mandatory step, not a footnote.

## Phase-by-phase notes

- **Phase 1 — Acceptance.** Started as Mem0-based; pivoted to V+ after spike. Final acceptance.md = 79 bullets including post-dry-run pin amendments (`invalid_at: int + 0 sentinel`, multi-key `$and` form).
- **Phase 2 — Context.** Reused `src/vecs/CLAUDE.md` unchanged; no context-tree edits needed for prose-drift module.
- **Phase 3 — Pipeline.** Set up `anthropic==0.103.1` pin via `uv sync`. No CI changes. Live-LLM integration test gated by `VECS_TEST_REAL_LLM=1` (default-skipped).
- **Phase 4 — Review Loop.** 4 design passes converged in one session. Two reviewer/sinker dispatch waves per pass. Caveman-compressed sinker output saved ~60% context vs vanilla.
- **Phase 5 — Tests.** Full 67-test matrix authored in design; only the Phase 7 dry-run subset (22 tests) implemented. Remaining ~45 land with Phase 4 wire-in.
- **Phase 6 — Roster.** Used `builder-small` analog (main thread executed dry-run directly, ~580 LOC under the ≤2-file scope by counting the new module + new test file). `critical-sinker` + `caveman:cavecrew-reviewer` ran per pass.
- **Phase 7 — Dry-Run.** Sequenced FIRST per profile `dryrun_sequencing`. All 4 boundaries hit. Dry-run pass 1 surfaced 7 findings, all applied; pass 2 shipped.

## Gap log

See `docs/features/prose-staleness-detector/gaps.md`. 34 findings tracked; 33 applied; 1 parked to v2; 0 wontfix.

## Feedback applied

| Target phase | Artifact | Owner |
|---|---|---|
| Phase 0 (Bootstrap) | `README.md` Install section addendum: `ANTHROPIC_API_KEY` storage guidance (Keychain helper for macOS; env-var for CI) | **applied** in `README.md:19-30` |
| Phase 2 (Context) | None this pass. `src/vecs/CLAUDE.md` will get a `Tests` bullet (gated `VECS_TEST_REAL_LLM` integration test) when Phase 5 lands. | self |
| Profile (workflow-vecs-profile-v0.1.md) | Add `worktree_cleanup: mandatory` slot to Phase 8 with explicit cleanup commands | **applied** in `docs/workflow-vecs-profile-v0.1.md:163` + new paragraph |
