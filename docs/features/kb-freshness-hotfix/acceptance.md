Authored by Claude

# Acceptance — Increment 1.5 (freshness/correctness hotfix: 1.5a + 1.5b + 1.5c)

Format: checklist. Pass threshold: all-pass. Parent: `docs/vecs-kb-curation-design-2026-06.md` §6 (Increment 1.5); direction review `docs/vecs-direction-review-2026-06.md`.
Runner: `uv run pytest -q`. Adapter: `scripts/check_acceptance.py kb-freshness-hotfix`.
Ships before Inc 2. Three independent fixes; all reuse already-shipped deletion/cache machinery; none depend on Inc 4a's `valid_from`/`valid_to`.

## 1.5a — prune-orphan fix (the only verified "agent retrieves lies" defect)

- [ ] `Manifest.prune()` returns the **list of removed keys** (was a bare count), and removes manifest entries whose source file no longer exists on disk — including `session:{path}` keys (previously skipped, so deleted sessions leaked their manifest entry forever).
- [ ] `run_index` deletes the chunks for pruned **session** files (by `session_id` = file stem) from the `-sessions` chroma collection AND its BM25 sidecar (via `_delete_ids_from_bm25` directly — the prune path does not go through `_sync_bm25`, which only runs when `total_stored > 0`).
- [ ] A **metadata-driven orphan sweep** runs every reindex over the `-code` and `-docs` collections: it resolves each chunk's source-root-qualified `file_path` back to disk and deletes chunks whose source is gone (chroma + BM25). This clears both the already-accumulated backlog (manifest already forgot them) and newly-deleted files.
- [ ] The sweep is **conservative on unknown roots**: a `file_path` whose first segment is not a current source-root name is left to the existing `_partition_docs_by_root` migration (no double-deletion, no basename-collision regression). An **empty source-root set never wipes a collection** (degenerate-prefix guard, mirroring `_partition_docs_by_root`).
- [ ] Test: index a code file + a docs file, delete each on disk, reindex → their chunks are gone from chroma AND BM25; a sibling file's chunks survive.
- [ ] Test: a backlog orphan (chunk present in chroma, **no** manifest entry, source gone) is removed by the sweep on a reindex where nothing new is embedded.
- [ ] Test: deleted session file → its `session_id` chunks are removed from chroma + BM25; a surviving session's chunks remain.
- [ ] Independence: the fix touches neither `valid_from`/`valid_to` nor the recency prior (those stay in Inc 4a).

## 1.5b — searcher `-docs` gate one-liner

- [ ] `searcher.py` always attempts the `-docs` collection with skip-on-miss (like `-code`/`-sessions`), instead of gating on `proj.docs_dir` — so `bloomly`/`eric` `-docs` (populated by F with in-repo `.md`, but lacking a `docs_dir`) are searched.
- [ ] Test: a project with a populated `-docs` collection but **no** `docs_dir`/`docs_dirs` configured returns its docs hits from search.

## 1.5c — freshness / trust signal

- [x] Search results surface a freshness/trust signal per hit (the chunk `version_id` and/or a freshness bucket) through the MCP `semantic_search` result shape. — `mcp_server._freshness_tag`; header `[v:<version_id>]` (40-hex sha → 8 chars, mtime verbatim, absent → `v:unknown`). Tests: `test_mcp_server.py`.
- [x] Query-time model-flip interlock: when a collection's recorded model marker (`EmbedCache.get_collection_model`) differs from the configured embedding model, the searcher warns and/or falls back to BM25 for that collection instead of silently scoring new-model query vectors against old-model stored vectors. — `searcher._collection_markers` (one cache open/search) + the `vector_targets` partition; mismatched collection dropped from the vector path, BM25 still runs, stderr warning. `None` marker / cache-read error is fail-open.
- [x] Test: marker ≠ configured model triggers the interlock (warn/fallback) for that collection; marker == configured model does not. — `test_searcher.py`: skips-vector / inactive-when-matches / none-fails-open / per-collection-not-all-or-nothing + real-function `_collection_markers` fail-open-on-error coverage.

## Global (this feature)

- [x] `uv run pytest -q` green; new/updated tests in `test_indexer.py` (1.5a; now split into `test_indexer_{manifest,embed,code,docs,run}.py`), `test_searcher.py` (1.5b/1.5c), `test_mcp_server.py` (1.5c surface) as touched. — 345 passed / 2 skipped.
- [x] `src/vecs/CLAUDE.md` updated for touched modules (prune contract, orphan sweep, `-docs` search, freshness signal + model-flip interlock).
- [x] Phase 4 multi-agent adversarial review verdict = approve; every finding triaged against the code. — 1.5c review (4 dimensions + adversarial verify): 3 findings, all triaged in-thread and addressed (interlock cache opened once/search; fail-open-on-error + per-collection regression tests added). No critical/high.
