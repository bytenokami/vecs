Authored by Claude

# Increment 1 — Foundations & No-Regret Wins (parent)

**Parent program:** `docs/vecs-kb-curation-design-2026-06.md` (v3, §6 Increment 1).
**Workflow profile:** `docs/workflow-vecs-profile-v0.1.md`.
**Status:** spec. This increment **decomposes into three independently-gated sub-features**, each its own workflow-profile feature with its own `acceptance.md`, Phase-4 review, and sign-off (per the program's decompose rule — a regression in one must not block the others):

| Sub-feature | Dir | Deliverables |
|---|---|---|
| 1-pipeline | `docs/features/kb-foundations-pipeline/` | F (`.md`→docs reroute + `-code` sweep), B (voyage-3.5 re-embed), C (`version_id` + content-hash cache) — share one reindex |
| 1-search | `docs/features/kb-foundations-search/` | D (per-collection RRF refactor + facts FTS5 sidecar) |
| 1-instrumentation | `docs/features/kb-foundations-instrumentation/` | A (metering spike), E (measurement-harness seed) |

**Build order:** 1-pipeline → 1-search → 1-instrumentation. Within 1-pipeline: C (cache + version_id) before B (so the voyage-3.5 re-embed is cached) and before F's reroute embed.

## Goal

The cheap, safe base every later increment needs, plus the no-regret quality wins. No LLM in the agent hot path; the one LLM touch (metering, sub-feature A) only *measures* cost.

## Resolved open item

**docs sources → multi-path (contract-first).** `ProjectConfig` gains `docs_dirs: list[Path]`; the existing single `docs_dir` is coerced into the list (mirrors the `sessions_dir`→`sessions_dirs` precedent, `config.py:89-95,213-217`). In-repo `.md` under `code_dirs` is routed to the project `-docs` collection. Rejected: auto-creating a synthetic `docs_dir` (magic). This is the mechanism Increment 3 extends for per-repo `docs/` discovery.

## Shared design notes (apply across sub-features)

These are the traps the Phase-4 review surfaced; the acceptance files test for them explicitly.

1. **`.md` sweep from `-code` is not automatic (F).** Dropping `.md` from `code_dirs` extensions stops *new* indexing but does NOT delete the ~431 already-embedded livly `.md` code chunks — `index_code` only adds; `prune_out_of_scope` (`indexer.py:189-227`) keys on full-path manifest keys and is exclude-dirs-oriented, not an asserted "extension removed" path. So add an **explicit sweep** of `.md`-sourced chunks from each `-code` collection + its BM25 sidecar, and assert the sweep ran (not just an end-count).
2. **`.md`→docs is `index_docs` surgery, not config (F).** `index_docs` (`indexer.py:1071-1140`) `return 0`s without a `docs_dir` and computes `rel_path = relative_to(docs_dir)` — which **raises ValueError** for a `.md` under a code_dir — with chunk ids `docs:{rel_path}` and cleanup keyed on bare `rel_path`. Two repos' `README.md` would collide and mutually delete. Required: multi-source ingestion that accepts a **per-file base dir** and emits a **source-root-qualified `rel_path`** (e.g. prefix the root basename, like `index_code` does at `:741`, or carry `source_root` in metadata) so cleanup can't over-match across roots; and handle projects with no `docs_dir`.
3. **The new model needs a re-embed, not a model flip (B).** (Target is **voyage-4**, superseding the originally-specced voyage-3.5 — current frontier, same 1024 dim; see acceptance.md §B.) Equal dimension ≠ equal vector space: querying voyage-3-embedded docs with voyage-4 query vectors silently degrades ranking, and a non-empty smoke test cannot catch it. Re-embed docs/sessions under the C cache; validate with known query→expected-source pairs. The flip takes effect at import (search queries with voyage-4 immediately), but the store is re-embedded only by the next `run_index`, so a reindex MUST be run promptly after deploy to close the transient degraded-search window. Note `_voyage_embed` (facts) also reads `SESSIONS_MODEL` (`prose_drift.py:488-491`) — pin/stamp the facts model so a swap can't strand facts (facts are empty until Inc 2, which already prevents corruption — make it explicit).
4. **The content-hash cache must preserve the `succeeded == expected` invariant (C).** `succeeded_ids` is built only from upserted chunks (`indexer.py:455`); a file is marked indexed only if `succeeded_per_file == file_expected_count` (`:604-607`). A cache hit that skips upsert would make `succeeded < expected` → the file is never marked, reprocessed forever, cleanup never fires (the bug that sank the v1 dedup idea). So cache hits must contribute their ids to `succeeded_ids` and idempotently ensure presence. Test the **mixed changed+unchanged-chunk file** case.
5. **Per-collection RRF is a refactor (D).** `reciprocal_rank_fusion` (`searcher.py:82-114`) fuses one concatenated list (`:209`); code is appended first (`:145-146`) so current order is partly append-order — NOT a baseline to "preserve." Restructure into per-collection rank lists + weights; assert the new weighted order is correct and that down-weighting sessions doesn't spuriously trip the 2×/3× refetch (`:154-216`) or `deduplicate_results`.
6. **stale-retrieval-rate depends on `version_id` (E after C).** Define the metric against the `version_id`/embed-hash anchor with a graceful "legacy/unknown" bucket for un-restamped chunks.

## Out of scope (deferred)

Populating/searching/blending facts → Inc 2. Per-repo `docs/` dir discovery (beyond in-repo `.md`) + chunk-size/window → Inc 3. Recency, tombstones, supersession filter, near-dup dedup → Inc 4. Stage-0 bundle → Inc 5.

## Phase 7 — Dry-run (parent)

- `dryrun_selection`: smallest **additive** real subtask = the **`docs_dirs` back-compat coercion** in `config.py` (config-load migration mirroring `sessions_dir`→`sessions_dirs`). Additive, no ranking semantics.
- `dryrun_acceptance` (inline): `uv run pytest -q tests/test_config.py` green; a new test asserts an existing `docs_dir`-only config loads to an equivalent `docs_dirs=[docs_dir]` with identical downstream behavior.
- `dryrun_pass_criteria`: `[pipeline-pass, review-loop-satisfied]`. `abort_policy`: branch-drop.

## Phase 2 — Context docs

Touched modules get `src/vecs/CLAUDE.md` updates on each sub-feature's acceptance: `config.py` (docs_dirs, voyage-3.5), `indexer.py` (version_id, content-hash cache, `.md` sweep, multi-source docs), `searcher.py` (per-collection RRF), `bm25_index.py` (prose-facts sidecar), `prose_drift.py` (facts model pin), `clients.py`.
