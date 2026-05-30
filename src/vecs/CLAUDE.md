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
| `chunkers.py` | Dispatch routing files to language-specific chunkers. |
| `ast_chunker.py` | Tree-sitter chunking (C#, TypeScript, plus generic). |
| `doc_chunker.py` | Markdown / PDF / plain doc chunking. |
| `codex_chunker.py` | Parses Codex CLI `rollout-*.jsonl`; filters `response_item` messages. |
| `codex_routing.py` | Discovers Codex sessions under `~/.codex/sessions/`; routes by `cwd` to a project; state in `~/.vecs/manifests/_codex_routing.json` under fcntl lock. |
| `cli.py` | Click CLI wiring. |
| `mcp_server.py` | MCP tool definitions consumed by AI agents. |
| `utils.py` | Logging, paths, small helpers. |

## Invariants

- Embedding models: `voyage-code-3` (free) for code; `voyage-3` ($0.06/1M tokens) for sessions and docs. Wired in `config.py`.
- Vector store: ChromaDB. Collections per project: `<project>-code`, `<project>-sessions`, `<project>-docs`.
- BM25 sidecar: SQLite FTS5; one `.db` per collection. Kept in lockstep with Chroma via `_sync_bm25` in `indexer.py`.
- Sessions are agent-tagged (`metadata.agent ∈ {claude_code, codex}`). Same collection, single query covers both.
- Index storage lives under `~/.vecs/` only. Never write inside the repo.
- Codex routing state is locked via `fcntl.flock` — concurrent indexers do not corrupt it.

## Tests

- `tests/` contains one module per source module (13 total). New feature touching module M MUST add or update `tests/test_M.py` (Phase 5 `new_test_required_per_feature: true`).
- Runner: `uv run pytest -q`.
- Opt-in integration tests gated by VECS_TEST_REAL_LLM=1 — real LLM calls; default-skipped in CI. See tests/test_prose_drift.py::test_integration_real_anthropic.

## Workflow context

This repo authors the workflow framework base (`docs/workflow-framework-v0.1.md`) and applies it via the vecs profile (`docs/workflow-vecs-profile-v0.1.md`). Features land via the profile's phases.

## Roadmap

Platform direction (search → agent memory; knowledge shapes & bundles): `docs/vecs-roadmap.md`. prose-drift feature deferrals: `docs/features/prose-staleness-detector/v2-roadmap.md`.

## prose-drift v1 boundary

`vecs prose-drift` / `mcp__vecs__prose_drift` is an on-demand recrawl, not a write-time detector. v1 reports only exact `(subject, predicate)` object-collisions between indexed docs and current chat facts. Out of scope (v2 — see `docs/features/prose-staleness-detector/v2-roadmap.md`): cross-predicate/paraphrase contradictions (needs the embedding-similarity + LLM contradiction judge), omission (doc silent on a now-true fact), and soft/temporal "used to have" contradictions.

## Staleness baseline

This file is the Phase 2 context-tree starter for `context_tree_root: src/vecs/`. `staleness_check: [commit-sha-tag, custom:prose-vplus]` baselines from the commit that introduces this file.
