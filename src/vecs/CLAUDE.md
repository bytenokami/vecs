# vecs — Module Context

Hybrid semantic + BM25 search over code, AI agent sessions (Claude Code + Codex), and docs. Exposes both a CLI (`vecs`) and an MCP server.

Targets workflow profile at `docs/workflow-vecs-profile-v0.1.md` (Phase 2 `context_tree_root`).

## Entry points

- `cli.py:main` — Click group. Subcommands: `index`, `search`, `add-document`, `status`, `prose-drift`, `codex` group.
- `mcp_server.py` — FastMCP server named `vecs`. Tools: `semantic_search`, `reindex`, `index_status`, `add_document`, `prose_drift`, `codex_orphans`, `codex_assign`, `codex_ignore`.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Per-project config loader; paths under `~/.vecs/` (chromadb, manifests, BM25 `.db`, reindex.log). |
| `clients.py` | Voyage AI embedding client + Chroma DB client. |
| `indexer.py` | Main indexing loop. Adaptive batching, manifest updates, sync between vector and BM25 stores. |
| `searcher.py` | Hybrid search: vector + BM25, fused output. |
| `bm25_index.py` | SQLite FTS5 BM25 index — one `.db` per collection. |
| `embed_cache.py` | Content-addressable embedding cache (SQLite at `~/.vecs/embed_cache.db`), keyed by `(model, content_hash)`. Lets the indexer skip Voyage calls for byte-identical chunks across runs. Also holds the `collection_models` marker table (per-collection last-embedded model) that drives the model-change re-embed trigger. |
| `chunkers.py` | Dispatch routing files to language-specific chunkers. |
| `ast_chunker.py` | Tree-sitter chunking (C#, TypeScript, plus generic). |
| `doc_chunker.py` | Markdown / PDF / plain doc chunking. |
| `codex_chunker.py` | Parses Codex CLI `rollout-*.jsonl`; filters `response_item` messages. |
| `codex_routing.py` | Discovers Codex sessions under `~/.codex/sessions/`; routes by `cwd` to a project; state in `~/.vecs/manifests/_codex_routing.json` under fcntl lock. |
| `cli.py` | Click CLI wiring. |
| `mcp_server.py` | MCP tool definitions consumed by AI agents. |
| `utils.py` | Logging, paths, small helpers. |

## Invariants

- Embedding models (`config.py`): `voyage-code-3` for code; `voyage-4` for sessions and docs (Inc 1-B, current frontier); `FACTS_MODEL = voyage-4` for prose-drift facts, pinned separately so a sessions/docs model swap cannot strand fact vectors. All three default to a **1024-dim** vector (`EMBED_DIMS`; we never send an `output_dimension` override), so a model change re-embeds in place over the same chunk ids — no collection recreate.
- Embedding-model change ⇒ re-embed, not a bare constant flip. `run_index` runs a PRE-pass (`_remodel_clear`) and POST-pass (`_remodel_record`): if the model recorded for a collection (`EmbedCache.get_collection_model`) differs from the configured `DOCS_MODEL`/`SESSIONS_MODEL` and the collection is non-empty, it clears that source's manifest entries (docs file-keys under `docs_dirs`; all `session:` keys) so the indexers re-embed every chunk under the new model; the new marker is written only after all indexers run. Centralized in `run_index` (NOT per-indexer) because `index_sessions` and `index_codex_sessions` share the `-sessions` collection — a per-indexer marker would be flipped by whichever ran first and strand the other agent's chunks. **Code has no trigger** (it stays `voyage-code-3`), so code chunks are never needlessly recomputed. The marker lives in `EmbedCache` (`collection_models` table), NOT the Manifest, because `Manifest.prune()` would delete a non-path key. **Operational note:** a model flip takes effect at import (search embeds queries with the new model immediately), but the store is re-embedded only by the next `run_index` — so a reindex must be run promptly after deploying a model change to close the transient window where new-model query vectors hit old-model stored vectors (the silent ranking degradation). The clear pass only invalidates files `index_docs`/the session indexers actually re-scan: docs are cleared by the EXACT file set `_docs_sources(project)` enumerates (all `docs_dirs` ∪ in-repo `.md` under `code_dirs`), the SAME enumerator `index_docs` iterates, so clear-scope ≡ rescan-scope by construction (F; `_clear_docs_manifest_entries` takes that file list, not a directory list). Deleted-source orphan chunks are removed every reindex by the prune-orphan sweep (`_prune_and_sweep_orphans`, Inc 1.5a — see the prune invariant below), not left for Inc 4a.
- Vector store: ChromaDB. Collections per project: `<project>-code`, `<project>-sessions`, `<project>-docs`.
- `-docs` sourcing (F): `index_docs` indexes doc files from EVERY `docs_dir` (`.md`/`.txt`/`.pdf`) PLUS in-repo `.md` under each `code_dir` (only `.md`, using the code_dir's own include/exclude scope via the shared `_scan_code_dir`). `.md` is NOT a `code_dirs` extension — it routes to `-docs`, not `-code` (and `index_code` defensively strips `.md` from its scan + warns if a stale config still lists it, so it can never fight the sweep). Every doc chunk id and `file_path` metadata is **source-root-qualified**: `docs:{root.name}/{rel}` (mirrors `index_code`'s `{code_dir.path.name}/{rel}`), so two roots' `README.md` get distinct ids + distinct cleanup filters and cannot mutually overwrite/delete. `index_single_doc` (add-document) resolves the file's owning root via `_owning_doc_root` (docs_dirs win over code_dirs, mirroring `_docs_sources`) and qualifies by it, so add + a full reindex agree on the id. Caveat: two source roots sharing a basename would collide (pre-existing `index_code` limitation; current roots are unique).
- Self-converging id-scheme migration (F, **load-bearing — do not regress**): the chunk-id scheme changed (bare `relative_to` → root-qualified) but the manifest KEY scheme did not (still bare absolute path, shared across code/docs). So deleting an old-scheme chunk is NOT enough to migrate — `needs_indexing` keys on the unchanged path-hash and would skip re-embed, losing the content. `index_docs` therefore does ONE scan (`_partition_docs_by_root`) returning `(orphan_ids, present_qualified_file_paths)`: it deletes the orphans (chunks under no current source-root prefix — legacy pre-F bare ids, removed roots; chroma + BM25), then **force-clears the manifest key of any source file whose qualified `file_path` is absent from `present`**, so `needs_indexing` returns True and the file re-embeds under the new id THIS run — independent of any embedding-model change. This converges both the `.md`→`-docs` move and the bare→qualified migration on the first post-F reindex and is a no-op in steady state. Load-bearing empty-roots guard: an empty source-root set returns `([], set())` WITHOUT scanning, so a misconfigured project never has its `-docs` wiped. `_sweep_md_code_chunks` similarly purges leftover `.md` chunks from `-code` + BM25; `index_code`'s `prune_out_of_scope` is extension-scoped so it does NOT delete in-repo `.md` keys (`index_docs` owns them), preventing a cross-indexer key thrash. Known low-risk gap (current config unaffected, roots unique): a legacy bare chunk whose first path segment equals a current root basename is misread as already-qualified and survives as a duplicate.
- Prune-orphan fix (Inc 1.5a, **the deleted-file correctness guarantee — do not regress**): `Manifest.prune()` returns the **removed keys** (not a count) and now also prunes deleted `session:{path}` keys. `run_index` calls `_prune_and_sweep_orphans(project, db)` after the index passes (wrapped per-project in try/except so one project's I/O error can't skip the rest): (1) deleted session files → chunks deleted by `session_id` (`_sweep_deleted_session_chunks`, since session chunks carry no `file_path`); (2) every reindex, `-code` and `-docs` are swept by `_sweep_deleted_source_chunks`, which resolves each chunk's root-qualified `file_path` (`{root.name}/{rel}`) back to disk and deletes chunks whose source is gone (existence cached per `file_path` — one stat per distinct file). Catches both newly-deleted files AND the backlog the *old* buggy prune (cleared the manifest, never the chunks) leaked — deleted files no longer rank against live content. **Three false-positive-deletion guards (an orphan is safer than deleting live data):** (a) the sweep root map is built by `_safe_sweep_root_map`, which includes only roots **present on disk** (a transiently-unmounted root would otherwise make every `(root/rel).exists()` False and wipe the whole collection — the index passes skip a missing root, so the sweep must too) **with a unique basename** (a collision can't disambiguate `{root.name}/{rel}`, so colliding roots are dropped); (b) unknown-root chunks are left to `_partition_docs_by_root` (legacy bare-scheme migration); (c) an `OSError` on a stat keeps the chunk. **Crash-safety ordering:** the session sweep runs and BM25 is mirrored (`_delete_ids_from_bm25` directly) **before** `manifest.save()` persists the prune — so an interruption leaves the `session:` key for next run to retry (sessions are the only non-self-healing path; code/docs self-heal via the disk scan regardless of manifest state). Independent of Inc 4a's `valid_from`/`valid_to` (4a keeps only recurring tombstone semantics + the `is_full=False` append-cleanup).
- BM25 sidecar: SQLite FTS5; one `.db` per collection. Kept in lockstep with Chroma via `_sync_bm25` in `indexer.py` (full reconcile when `total_stored > 0`); the targeted F sweeps **and the Inc 1.5a prune-orphan sweep** additionally call `_delete_ids_from_bm25` because a sweep-only run (chunks deleted, nothing new embedded) would otherwise skip `_sync_bm25` and leave stale BM25 rows.
- Sessions are agent-tagged (`metadata.agent ∈ {claude_code, codex}`). Same collection, single query covers both.
- Index storage lives under `~/.vecs/` only. Never write inside the repo.
- Codex routing state is locked via `fcntl.flock` — concurrent indexers do not corrupt it.
- Each stored chunk carries a `version_id` in metadata (set at chunk construction in `indexer.py`): git HEAD sha for code (per code_dir, falls back to file content hash for non-git trees), file mtime for docs, session id for sessions. Anchor for stale-retrieval detection.
- Embedding cache key MUST include the model. A cache hit returns a stored vector verbatim, so serving a `voyage-3` vector for a `voyage-3.5` request would silently corrupt ranking. Changing the embedding model invalidates the cache by construction (all misses → full re-embed). Cache hits are still upserted and counted in `succeeded_ids`, so the manifest's `succeeded == expected` mark-indexed invariant (`_track_embed_success`) holds — a hit that skipped upsert would leave the file reprocessed forever.

## Tests

- `tests/` holds **one or more** test files per source module. New feature touching module M MUST add or update its test file(s) (Phase 5 `new_test_required_per_feature: true`). Small modules use a single `tests/test_M.py`. A large module is split by concern to keep each file readable/cheap-to-load: **`indexer.py` → `tests/test_indexer_{manifest,embed,code,docs,sessions,run}.py`** plus shared fakes/fixtures in **`tests/indexer_helpers.py`** (a plain importable module, not `test_*`, so pytest doesn't collect it as tests; the concern files `from indexer_helpers import ...`).
- Runner: `uv run pytest -q`.
- Opt-in integration tests gated by VECS_TEST_REAL_LLM=1 — real LLM calls; default-skipped in CI. See tests/test_prose_drift.py::test_integration_real_anthropic.

## Workflow context

This repo authors the workflow framework base (`docs/workflow-framework-v0.1.md`) and applies it via the vecs profile (`docs/workflow-vecs-profile-v0.1.md`). Features land via the profile's phases.

## Roadmap

Platform direction (search → agent memory; knowledge shapes & bundles): `docs/vecs-roadmap.md`. prose-drift feature deferrals: `docs/features/prose-staleness-detector/v2-roadmap.md`.

## prose-drift v1 boundary

`vecs prose-drift` / `mcp__vecs__prose_drift` is an on-demand recrawl, not a write-time detector. It reports two kinds of contradiction between indexed docs and current chat facts: (1) **exact** — same `(subject, predicate)` chain, differing object; (2) **semantic** (stage-2) — on a `chain_key` MISS, the most cosine-similar current fact above `STAGE2_SIM_THRESHOLD` is escalated to ONE Opus contradiction-judge, catching cross-predicate/paraphrase drift (`match_type` distinguishes them; see `docs/features/prose-staleness-detector/stage2-recall-design.md`). Still out of scope (v2 — see `v2-roadmap.md`): omission (doc silent on a now-true fact) and soft/temporal "used to have" contradictions.

## Staleness baseline

This file is the Phase 2 context-tree starter for `context_tree_root: src/vecs/`. `staleness_check: [commit-sha-tag, custom:prose-vplus]` baselines from the commit that introduces this file.
