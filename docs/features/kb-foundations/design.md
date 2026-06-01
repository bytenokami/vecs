# Increment 1 — Foundations & No-Regret Wins

**Parent program:** `docs/vecs-kb-curation-design-2026-06.md` (§6 Increment 1). This doc is the per-increment design; the program doc holds the vision, gap-map, principles, and SOTA references.
**Workflow profile:** `docs/workflow-vecs-profile-v0.1.md`.
**Status:** spec (Phase 1). Acceptance in `acceptance.md` (this dir).

## Goal

Land the cheap, safe, no-hot-path-LLM base that every later increment depends on, plus the two no-regret quality wins (voyage-3.5, the `.md`→docs reroute). Nothing here runs an LLM in the agent's hot path; the one LLM touch (metering spike) only *measures* cost so later increments can be gated.

## Resolved open item

**docs sources → multi-path (contract-first).** `ProjectConfig` gains `docs_dirs: list[Path]`; the existing single `docs_dir` is coerced into the list for back-compat. In-repo `.md` files under `code_dirs` (now that `.md` is dropped from code extensions) are routed to the project's `-docs` collection, heading-chunked. Projects with no configured docs source still get a `-docs` collection to hold their rerouted in-repo `.md`. Rejected alternative: auto-creating a synthetic `docs_dir` per project (magic, hides the contract). This same multi-path mechanism is what Increment 3 extends for per-repo `docs/` discovery, so we build it once here.

## Scope (sub-features — build/commit independently, unified acceptance)

| ID | Sub-feature | Touches |
|---|---|---|
| A | **Metering spike** — per-call cost record (model/tokens/$), `MAX_CALLS_PER_DAY` cap, default extraction model = Sonnet. Output a per-project est-cost-to-populate-facts report committed to this dir. | `prose_drift.py`, `clients.py`, new `metering` helper |
| B | **voyage-3 → voyage-3.5** for docs/sessions after a recorded dim-compat check (code stays voyage-code-3). | `config.py`, `clients.py` |
| C | **`version_id` + content-hash embedding cache** — stamp every chunk; cache embeddings by chunk content-hash so unchanged chunks aren't re-embedded. | `indexer.py`, `clients.py` |
| D | **Per-collection RRF weights + facts FTS5 sidecar** — config-driven per-collection weights (sessions down-weighted); give `-prose-facts` a BM25 sidecar synced via `_sync_bm25`. | `searcher.py`, `bm25_index.py`, `indexer.py`, `config.py` |
| E | **Measurement-harness seed** — `stale-retrieval-rate` metric + a small local eval-set scaffold + runner stub. | new `eval/` helper, `tests/` |
| F | **`.md`→docs reroute** — drop `.md` from all `code_dirs`; add `docs_dirs` multi-path; route in-repo `.md` to `-docs` (heading-chunked) for all 3 projects. | `~/.vecs/config.yaml`, `config.py`, `indexer.py` |

**Recommended build order:** F (clears the biggest noise, exercises the docs path) → D (search-layer contract) → C (cache, makes F's reindex cheap on reruns) → B (model swap) → A (metering) → E (harness). C ideally before F's first full reindex to save Voyage cost; if sequenced after, accept the one-time cost.

## Out of scope (deferred, per program doc)

- Populating/searching/blending facts → Increment 2 (needs metered extraction + the merge gate). D only builds the *plumbing* (FTS5 sidecar + weights); the facts collection stays empty until Inc 2.
- Near-dup dedup, recency, tombstones, supersession filter → Increment 4.
- Stage-0 bundle → Increment 3 (this increment delivers its prerequisites: manifest model/dim stamp groundwork + content-hash cache).

## Contract changes

- `ProjectConfig.docs_dirs: list[Path]` (new); `docs_dir` retained, coerced into `docs_dirs` on load. Config loader migrates transparently (mirrors the `sessions_dir`→`sessions_dirs` precedent in `config.py:89-95`).
- `search()` gains per-collection RRF weights (config-driven; defaults preserve current behavior except sessions down-weight).
- Chunk metadata gains `version_id`.
- A new `-prose-facts` FTS5 `.db` per project under `~/.vecs/bm25/` (created lazily; empty until Inc 2 writes facts).

## Risks

- **Dim-compat (B):** if voyage-3.5 dim ≠ voyage-3, docs/sessions need a full re-embed. Verify *before* switching; record the result. If incompatible, gate B on the C cache to bound cost.
- **Config migration (F):** `docs_dirs` coercion must not break existing `docs_dir`-only configs. Covered by a test.
- **No corpus reindex implied by dropping `.md`** (remaining code is not re-embedded, `indexer.py:799-815`); the reroute *does* embed the `.md` files as docs (cheap, voyage-3.5).
- **C invariant:** the embedding cache must not interact with the `succeeded==expected` file-marking invariant (`indexer.py:604-607`) — a cache hit still counts as a "succeeded" chunk id. Test this explicitly (this is the class of bug that sank the v1 dedup idea).

## Phase 7 — Dry-run

- `dryrun_selection`: smallest-real-subtask = **sub-feature D's per-collection RRF weight** — add a config-driven `w_sessions` (default preserves current ranking) to `search()` + a test asserting weight changes ranking on a fixture. Self-contained, exercises the search layer end-to-end.
- `dryrun_acceptance` (inline): `uv run pytest -q tests/test_searcher.py` green; the new weight test passes; `semantic_search` smoke returns results unchanged at default weights.
- `dryrun_pass_criteria`: `[pipeline-pass, review-loop-satisfied]`.
- `abort_policy`: branch-drop.

## Phase 2 — Context docs

Touched modules get `src/vecs/CLAUDE.md` updates on feature acceptance (`context_update_trigger: per-feature-acceptance`): `config.py` (docs_dirs, voyage-3.5), `searcher.py` (per-collection weights), `indexer.py` (version_id, content-hash cache), `bm25_index.py` (prose-facts sidecar).
