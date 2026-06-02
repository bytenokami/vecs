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
- Embedding-model change ⇒ re-embed, not a bare constant flip. `run_index` runs a PRE-pass (`_remodel_clear`) and POST-pass (`_remodel_record`): if the model recorded for a collection (`EmbedCache.get_collection_model`) differs from the configured `DOCS_MODEL`/`SESSIONS_MODEL` and the collection is non-empty, it clears that source's manifest entries (docs file-keys under `docs_dirs`; all `session:` keys) so the indexers re-embed every chunk under the new model; the new marker is written only after all indexers run. Centralized in `run_index` (NOT per-indexer) because `index_sessions` and `index_codex_sessions` share the `-sessions` collection — a per-indexer marker would be flipped by whichever ran first and strand the other agent's chunks. **Code has no trigger** (it stays `voyage-code-3`), so code chunks are never needlessly recomputed. The marker lives in `EmbedCache` (`collection_models` table), NOT the Manifest, because `Manifest.prune()` would delete a non-path key. **Operational note:** a model flip takes effect at import (search embeds queries with the new model immediately), but the store is re-embedded only by the next `run_index` — so a reindex must be run promptly after deploying a model change to close the transient window where new-model query vectors hit old-model stored vectors (the silent ranking degradation). The clear pass only invalidates files `index_docs`/the session indexers actually re-scan (today `docs_dir`, i.e. `docs_dirs[0]`); F widens both together. Deleted-source orphan chunks are not re-embedded and remain in the old vector space until the Inc 4a orphan sweep.
- Vector store: ChromaDB. Collections per project: `<project>-code`, `<project>-sessions`, `<project>-docs`.
- BM25 sidecar: SQLite FTS5; one `.db` per collection. Kept in lockstep with Chroma via `_sync_bm25` in `indexer.py`.
- Sessions are agent-tagged (`metadata.agent ∈ {claude_code, codex}`). Same collection, single query covers both.
- Index storage lives under `~/.vecs/` only. Never write inside the repo.
- Codex routing state is locked via `fcntl.flock` — concurrent indexers do not corrupt it.
- Each stored chunk carries a `version_id` in metadata (set at chunk construction in `indexer.py`): git HEAD sha for code (per code_dir, falls back to file content hash for non-git trees), file mtime for docs, session id for sessions. Anchor for stale-retrieval detection.
- Embedding cache key MUST include the model. A cache hit returns a stored vector verbatim, so serving a `voyage-3` vector for a `voyage-3.5` request would silently corrupt ranking. Changing the embedding model invalidates the cache by construction (all misses → full re-embed). Cache hits are still upserted and counted in `succeeded_ids`, so the manifest's `succeeded == expected` mark-indexed invariant (`_track_embed_success`) holds — a hit that skipped upsert would leave the file reprocessed forever.

## Tests

- `tests/` contains one module per source module (13 total). New feature touching module M MUST add or update `tests/test_M.py` (Phase 5 `new_test_required_per_feature: true`).
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
