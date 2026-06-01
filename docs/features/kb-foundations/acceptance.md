# Acceptance — Increment 1: Foundations & No-Regret Wins

Format: checklist. Pass threshold: all-pass (per `docs/workflow-vecs-profile-v0.1.md` Phase 1).
Design: `docs/features/kb-foundations/design.md`. Parent: `docs/vecs-kb-curation-design-2026-06.md` §6.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-foundations`.

## A — Metering spike

- [ ] Every extraction/judge LLM call emits a cost record (model, input/output tokens, USD).
- [ ] A `MAX_CALLS_PER_DAY` cap exists and is enforced: extraction stops at the cap and logs that it stopped.
- [ ] Default extraction model is Sonnet (overridable by config), not Opus.
- [ ] A per-project est-cost-to-populate-facts report is produced and committed under `docs/features/kb-foundations/`.

## B — voyage-3.5 upgrade

- [ ] voyage-3.5 embedding dimension is checked against the existing voyage-3 docs/sessions collections; the result is recorded in this dir.
- [ ] If dims match: docs + sessions models switched to voyage-3.5 with no re-embed. If they differ: a gated re-embed path is documented and executed under the content-hash cache (C).
- [ ] Code collection still uses voyage-code-3 (unchanged).
- [ ] `semantic_search` smoke returns results post-switch.

## C — version_id + content-hash embedding cache

- [ ] Each stored chunk carries a `version_id` (git SHA for code, mtime/revision for docs, session/run id for sessions).
- [ ] An embedding cache keyed by chunk content-hash exists; re-indexing a content-unchanged chunk produces a cache hit and makes no Voyage call.
- [ ] A reindex of a project with no source changes performs zero new Voyage embeddings (verified via metering/log).
- [ ] A cache hit still counts toward the per-file `succeeded == expected` invariant (no file is left perpetually un-indexed by a cache hit). Test asserts this explicitly.

## D — per-collection RRF weights + facts FTS5 sidecar

- [ ] `search()` supports config-driven per-collection RRF weights; sessions weight defaults below code/docs.
- [ ] Default weights preserve current ranking for code/docs (regression test on a fixture).
- [ ] `<p>-prose-facts` has an FTS5 BM25 sidecar created lazily and kept in lockstep via `_sync_bm25`.
- [ ] The facts collection/sidecar remain empty (no behavior change to default search results) — facts blend is Increment 2.

## E — measurement-harness seed

- [ ] A `stale-retrieval-rate` metric is computable over a collection (fraction of returned chunks whose source changed since embed).
- [ ] A small local eval-set scaffold exists (query → expected source/answer) with a runner stub that reports the metric.

## F — .md reroute (multi-path docs)

- [ ] `.md` removed from every `code_dirs` extension list in `~/.vecs/config.yaml`.
- [ ] After reindex, every `-code` collection contains zero `.md`-sourced chunks.
- [ ] `ProjectConfig` has `docs_dirs: list[Path]`; an existing single `docs_dir` is coerced in with no breakage (migration test).
- [ ] In-repo `.md` under `code_dirs` is routed to the project `-docs` collection, heading-chunked.
- [ ] bloomly, eric, and livly each have a `-docs` collection containing their in-repo `.md` after reindex.
- [ ] No `.md` content lost: per project, count of `.md` files on disk == `.md` source files tracked in the docs manifest.

## Global

- [ ] `uv run pytest -q` is green.
- [ ] New or updated tests exist per touched module: `test_config.py`, `test_indexer.py`, `test_searcher.py`, `test_bm25.py`, `test_prose_drift.py`, `test_clients.py` (as touched).
- [ ] `src/vecs/CLAUDE.md` updated for touched modules (Phase 2 staleness).
- [ ] Phase 7 dry-run passed (per-collection RRF weight subtask): pipeline-pass + review-loop-satisfied.
- [ ] Phase 4 multi-agent review verdict = approve (architect → critical-sinker → reviewer).
