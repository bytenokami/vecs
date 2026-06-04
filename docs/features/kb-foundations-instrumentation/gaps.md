Authored by Claude

# Gap log: kb-foundations-instrumentation

Phase-8 gap log per `docs/workflow-vecs-profile-v0.1.md`. Records what E + A deliberately did NOT cover and the follow-ups they surfaced — so a scope boundary isn't mistaken for coverage.

## Deliberately out of scope (seed / spike by design)
- **E: code staleness is coarse (repo-HEAD, not per-file).** A code chunk's `version_id` is the code_dir's HEAD at index time, so ANY commit marks every code chunk in that dir stale until reindex. This is a true "index lags HEAD" signal but over-counts file-level staleness vs docs' per-file mtime. A per-file blob-sha anchor would refine it; deferred (would change the indexer's stamping, out of E's seed scope).
- **E: no retrieval-quality threshold / pass-fail gate.** `run_eval` reports `hit_rate` + retrieved-chunk staleness; it does not assert a target. Wiring a green-number gate (the precondition for Inc 3/4b/6/7) is a later step once the live store is reindexed and the eval set is tuned to real projects.
- **E: the seed eval-set targets the vecs repo's own docs.** It is a scaffold, not a curated suite; the live-store quality check (`tests/test_searcher.py::test_reembed_quality_*`) stays the gated real-Voyage path. Repoint/extend per project.
- **A: pricing is a static table, not fetched.** `PRICING_USD_PER_MTOK` is list prices noted 2026-06; verify before trusting the cost report. Unknown model families price at 0 (unpriced) and never block a call — intentional fail-open for a spike.
- **A: the cost report's output-token figure (200/chunk) is assumed, not measured.** Input tokens are a real `chars//3` floor over the actual corpus; output can only be pinned by sampling real extraction responses. Flagged in the report.
- **A: the daily cap is best-effort under concurrency.** `calls_today` reads the JSONL and `metered_create` checks-then-calls without a lock, so two concurrent prose-drift runs could each pass the check near the boundary and slightly overshoot the cap. Acceptable for a single-user spike (low QPS; cap re-trips on the next call); a hard cap would need a lock or an atomic counter. Not built.

## Follow-ups surfaced (candidates for later increments)
- **A green E number is the hard precondition** for Inc 3 / 4b / 6 / 7 and the heavy half of 4a (per the program). E provides the harness; producing the number needs the live migrating reindex (owner's manual step) so the store is voyage-4 + version_id-stamped end-to-end.
- **Metering observability beyond a JSONL.** A `vecs cost` summary (today's spend, per-model rollup) would make the spike usable day-to-day; deliberately not built (spike, not dashboard).
- **Cost-ceiling kill criterion.** If wanted, a $/run or $/day ceiling that gates Inc 2/6 is a §7 program decision — the instrument exposes the numbers but does not gate.

## Note
No worktrees were created during this feature, so the mandatory Phase-8 worktree-cleanup slot is a no-op.
