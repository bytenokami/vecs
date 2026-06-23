Authored by Claude

# Acceptance — Increment 2 (Memory bridge: kb-memory-bridge)

Format: checklist. Pass threshold: all-pass. Parent: `docs/vecs-kb-curation-design-2026-06.md` §6 (Increment 2, v5-fix sessions-removed reconciliation).
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-memory-bridge`.

**Scope (sessions-removed reconciliation, owner decision 2026-06-04; project memory `vecs-sessions-removed`).** vecs no longer indexes chat transcripts. Inc 2's live source is **curated `memory/*.md` promotion (the memory↔vecs inversion)** — make the agent's durable curated knowledge retrievable through semantic search. Chat-transcript triples are parked with Inc 6; write-time persistence of docs/code-prose-extracted facts is deferred (query-time `find_prose_drift` already covers docs drift and overlaps Inc 6).

**Build surface is mostly wiring** — verified by a 3-reader code map (2026-06-19). Already present and reused unchanged: the exact `chain_key` INSERT/NOOP/SUPERSEDE state machine (`add_fact_with_state_machine`, `prose_drift.py:717-790`), the `-prose-facts` collection + naming (`ProjectConfig.prose_facts_collection`, `config.py:97-98`), `FACTS_MODEL='voyage-4'` 1024-dim pinned (`config.py:25`), the Sonnet extraction kernel `extract_facts_from_doc` (`prose_drift.py:306-351`, cached + metered), the metering chokepoint `metered_create` + `MAX_CALLS_PER_DAY` (`metering.py`), and the query side `find_prose_drift` (works the moment facts exist). What is absent is the ingestion source, the write-time merge gate, provenance metadata, the facts BM25 sidecar, and the search wiring.

Three sub-features, dependency-ordered: **2a populate → 2b merge-gate → 2c search**. Each independently testable.

## 2a — Populate: memory-file ingestion + provenance

Config:

- [ ] `ProjectConfig` gains `memory_dirs: list[Path] = []` (`config.py:67-98`), round-tripped through both `VecsConfig.save()` (`config.py:140-162`) and `load_config()` (`config.py:230-235`). Points at a project's curated memory dir (e.g. `~/.claude/projects/<repo-key>/memory/`).
- [ ] **Fix pre-existing round-trip bug:** `save()` currently serializes only `code_dirs`/`docs_dirs`/`embed_provider` and silently drops `prose_drift_enabled` (`config.py:140-162`) — any `save()` (triggered by `add_document` auto-configure) strips it. `save()` now persists `prose_drift_enabled` **and** `memory_dirs`.
- [ ] `add_project()` (or a dedicated setter) accepts `memory_dirs` (`config.py:113-123`).

Ingestion:

- [ ] New `ingest_memory_facts(project)` (in `indexer.py` or `prose_drift.py`): scans `memory_dirs` from disk for `*.md`, chunks each file, calls `extract_facts_from_doc` per chunk (reuses Sonnet + `metered_create` + `extraction_cache` — no new extraction code), feeds each `Triple` to the (2b-enhanced) `add_fact_with_state_machine`. Uses the shared `get_chromadb_client()`, **not** `prose_drift._get_prose_facts_collection`'s separate `PersistentClient` (`prose_drift.py:361-367`).
- [ ] Expose point decided and wired: a `run_index` hook gated on `memory_dirs` non-empty (and/or `prose_drift_enabled`) (`run_index`, `indexer.py:1774-1844`), **or** a dedicated CLI command / MCP tool. (Design Phase picks one; do not silently route through the existing `add_document`, which is the raw `-docs` store and must stay unchanged — `mcp_server.py:158-195`.)
- [ ] Every extraction call routes through `metered_create`; `MeteringCapExceeded` is caught mid-batch → already-extracted facts persist, a `cap_hit` flag is returned, no crash (mirror `find_prose_drift`'s pattern). No new pricing table.

Provenance metadata:

- [ ] `_new_row_metadata` (`prose_drift.py:701-714`) extended with: `source_path` (repo-relative memory-file path, a distinct structured field — `source_id` today is an opaque string), `commit_sha` (git HEAD of the memory repo at ingestion, `''` if not a repo), `scope_tier` (`'curated'` for memory promotion; enum reserves `'extracted'` for the deferred docs-prose path).
- [ ] `EmbedCache.set_collection_model` marker written for the `-prose-facts` collection on first population (so the `searcher.py` model-flip interlock from Inc-1.5c protects fact vectors; today `_remodel_record` covers only code/docs).

Tests (2a):

- [ ] A project with `memory_dirs` → a temp dir of `*.md` → `ingest_memory_facts` populates `-prose-facts` with `is_current=True` rows carrying `source_path`, `commit_sha`, `scope_tier='curated'`.
- [ ] Re-ingest an unchanged file → NOOP (no new rows; extraction served from `extraction_cache`).
- [ ] Edit a fact's object in a memory file + re-ingest → SUPERSEDE (old row `is_current=False`, `invalid_at` set; new row `version+1`, `is_current=True`).
- [ ] `memory_dirs` and `prose_drift_enabled` both survive a `save()` → `load_config()` round-trip (the bug-fix regression test).
- [ ] Metering cap hit mid-batch → partial facts persisted, `cap_hit=True`, no exception escapes.

## 2b — Semantic merge gate (write-time)

- [ ] A write-time semantic gate runs **before** the exact `chain_key` resolution in the ingestion path: embed the candidate triple, fetch top-k nearest `is_current=True` facts (Chroma vector query), and for any neighbor with similarity ≥ a threshold but a **different** `chain_key`, call the Opus contradiction-judge (`PROSE_JUDGE_MODEL='claude-opus-4-8'`, `prose_drift.py:32-38`, via `metered_create`) → decision MERGE (collapse onto the matched chain — NOOP/SUPERSEDE there, do not create a near-dup row) / DISTINCT (proceed to the normal exact-`chain_key` INSERT) / SUPERSEDE. This is new code atop `add_fact_with_state_machine` (`prose_drift.py:717`), not a change to the exact state machine itself.
- [ ] Top-k and the similarity threshold are named constants; the judge call routes through `metered_create` and is cap-aware.
- [ ] Idempotent: re-running ingestion over the same source does not accumulate near-dup rows the gate should have merged.

Tests (2b):

- [ ] Two memory facts, same meaning, different `subject|predicate` wording → gate MERGES (one operative `is_current` row), not two.
- [ ] Two genuinely distinct but embedding-close facts → judge returns DISTINCT/SUPERSEDE per its decision; assert **no false merge** (both meanings retained per the decision).
- [ ] Judge cap-exceeded → fail-safe to exact `chain_key` only (store as INSERT; never silently drop a candidate fact).

## 2c — Wire facts into search

- [ ] `reciprocal_rank_fusion` refactored from a single concatenated list (`searcher.py:120-152`, fused at `:306` over append-ordered `all_results` at `:289`) into **per-collection rank lists with independent weights** (folds old Inc-1-D). Facts receive a distinct blend weight. The append-order bias (code-first out-ranks docs regardless of relevance) is removed; "current order" is **not** a baseline to preserve (it is partly an append-order artifact).
- [ ] `search()` adds the facts target (`proj.prose_facts_collection`, `FACTS_MODEL`) when the project has facts; `collection_name` accepts `'facts'` (`searcher.py:180-204`).
- [ ] `is_current=True` applied as a **pre-retrieval** Chroma `where`-filter on the facts vector query, merged with any `path_filter` via Chroma `$and` (`searcher.py` query path ~`:270-278`).
- [ ] BM25 sidecar for `-prose-facts`: `_sync_bm25(collection, project_name, 'prose-facts')` after ingestion (gated `total_stored > 0`, mirroring code/docs); targeted `_delete_ids_from_bm25` on every SUPERSEDE-retired row (so a retired fact does not linger in BM25); the searcher `bm25_paths` suffix router recognizes `'-prose-facts'` instead of falling through to the `'docs'` db (`searcher.py:193-195`).
- [ ] Refetch/dedup guard: adding the facts collection does not spuriously trip the 2×/3× refetch loop or `deduplicate_results` (`searcher.py:255-315`) when fact hits are near-duplicate to docs hits.

Tests (2c):

- [ ] Per-collection RRF: given known per-collection rank lists + weights, fused order matches the weighted expectation (not append order).
- [ ] `semantic_search` over a project with populated facts returns fact hits; superseded (`is_current=False`) facts are **never** returned.
- [ ] Facts BM25 sidecar is built; a keyword-only fact query returns via the BM25 path; `bm25_paths` resolves `-prose-facts` to the facts db, not the docs db.
- [ ] A SUPERSEDE-retired fact is removed from BM25 (not just flipped in Chroma).
- [ ] Near-dup facts do not spuriously trigger the 3× refetch.

## Phase 7 — Dry-run (inline)

- [ ] **Dry-run subtask = the 2a config round-trip** (`memory_dirs` + `prose_drift_enabled` persist through `save()` → `load_config()`) — the smallest real subtask, no LLM calls, fast. `dryrun_pass_criteria = [pipeline-pass, review-loop-satisfied]` must hold on this slice before building 2a ingestion / 2b / 2c. `iterate_before_real: true`; abort = branch-drop.

## Global (this feature)

- [ ] `uv run pytest -q` green; new/updated tests in `test_config.py` (2a config + round-trip), `test_prose_drift.py` (2a ingest, 2b gate), `test_searcher.py` (2c). `new_test_required_per_feature: true`.
- [ ] `src/vecs/CLAUDE.md` updated for every touched module (`memory_dirs` + `save()` round-trip fix, `ingest_memory_facts`, the write-time merge gate, the `-prose-facts` BM25 sidecar, the facts search wiring + `is_current` pre-filter + per-collection RRF, the provenance metadata schema, the EmbedCache facts marker).
- [ ] Phase-4 multi-agent adversarial review (`architect` + `critical-sinker` + `reviewer`, vecs profile §Phase-4) verdict = approve; every finding triaged against the code; reviewer emits a per-iteration verdict line.
- [ ] A full memory ingest of the vecs project's own memory dir stays under `MAX_CALLS_PER_DAY`; record the actual call/cost count in the retro (binds the §7 cost-ceiling open question with a real number).

## Contingent / explicitly deferred (decide in design; do not build blind)

- **Docs/code-prose write-time fact persistence** (beyond memory files) — deferred. Query-time `find_prose_drift` already detects docs drift; persisting docs-extracted facts overlaps Inc 6. Inc 2 source = curated memory only.
- **Shared-tier + redaction** → Inc 5 (the bundle is their consumer).
- **Human/critic gate on promotion** → contingent (§7).
- **Binding a `$` cost ceiling** for fact population → from the Inc-1 metering report (§7 open); Inc 2 records the number, the threshold decision is the owner's.
