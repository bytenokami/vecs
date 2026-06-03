Authored by Claude

# Acceptance — Increment 1-search (D)

Format: checklist. Pass threshold: all-pass. Parent: `docs/features/kb-foundations/design.md`; program §6.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-foundations-search`.

## D — per-collection RRF refactor + facts FTS5 sidecar

- [ ] `reciprocal_rank_fusion` is restructured to fuse PER-COLLECTION rank lists (not one concatenated list) with config-driven per-collection weights.
- [ ] Sessions are down-weighted relative to code/docs by default.
- [ ] Correctness test asserts the NEW weighted fused order on a fixture (the prior concatenation order is explicitly NOT treated as a baseline to preserve).
- [ ] Down-weighting sessions does not spuriously trip the 2×/3× refetch escalation (`searcher.py:154-216`) or wrongly interact with `deduplicate_results` — tested.
- [ ] `<p>-prose-facts` gets an FTS5 BM25 sidecar created lazily and kept in lockstep via `_sync_bm25`.
- [ ] The facts collection/sidecar remain empty and unsearched (no behavior change to default results) — facts search + blend is Increment 2.

## Global (this sub-feature)

- [ ] `uv run pytest -q` green; new/updated tests in `test_searcher.py`, `test_bm25.py`.
- [ ] `src/vecs/CLAUDE.md` updated for `searcher.py`, `bm25_index.py`.
- [ ] Phase 4 review verdict = approve.
