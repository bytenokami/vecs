# Codex CLI session-transcript support

Date: 2026-05-05
Status: Approved (user "EM mode")

## Goal

Add Codex CLI session JSONL files (`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`) as a second session source alongside Claude Code, so semantic + BM25 search returns transcripts from both agents in one query, per project.

## Non-goals

- Indexing Codex `developer` / `system` prompts, tool calls, or `reasoning` items (later, behind a flag).
- Supporting third agents in this round (Cursor, Aider). Architecture leaves room.
- Auto-discovering Codex location across machines / multiple Codex installs (single global root only).

## Decisions locked

| # | Decision |
|---|---|
| Q1 | Routing: `cwd` from `session_meta.payload` decides project. Bidirectional resolved-path containment (cwd-under-code-dir OR code-dir-under-cwd). Ambiguous ancestor cwds → orphan, never greedy. |
| Q2 | Config: top-level `codex_sessions_root` (default `~/.codex/sessions`). Top-level `codex_disabled`, `codex_ignore_cwds`. Per-project `codex_cwds` for explicit override. |
| Q3 | Storage: same `sessions` collection per project. New chunk metadata field `agent ∈ {claude_code, codex}`. |
| Q4 | Format dispatch: by source path. Files under `codex_sessions_root` → Codex parser. Files under per-project `sessions_dirs` → Claude Code parser. No content sniffing. |
| Q5 | Codex content filter: `response_item` with `payload.type=="message"` and `role ∈ {user, assistant}` only. Skip everything else. |
| Q6 | Orphans: persisted in manifests dir (under existing `fcntl.flock` locking). Surfaced via MCP banners on `index_status` / `semantic_search`. CLI: `vecs codex orphans/assign/ignore`. MCP tools: `codex_orphans`, `codex_assign`, `codex_ignore`. |
| Q7 | Implementation: separate `codex_chunker.py`. Refactor `index_sessions` (`indexer.py:848`) to call shared `_index_session_files(project, files, parser, agent_tag)` core. Both pipelines = thin wrappers. |

## Schema (verified against `~/.codex/sessions`)

Per-line shape:

```json
{ "timestamp": "...", "type": "<top>", "payload": {...} }
```

Top-level types observed in the wild: `session_meta`, `event_msg`, `response_item`, `turn_context`. Default-skip-unknown.

Indexed lines: `type=="response_item"` AND `payload.type=="message"` AND `payload.role ∈ {user, assistant}`. Content parts: `[{type, text}]` where `type ∈ {text, input_text, output_text}` (Codex uses `input_text`/`output_text`; tolerate `text` for forward-compat).

Session metadata (used for routing): line 1 has `type=="session_meta"`, `payload.cwd`, `payload.id`. Line 1 read once and cached by `(path, mtime)`.

## Architecture

```
config.yaml ─┬─► per-project sessions_dirs ──► chunkers.preprocess_session ──┐
             │                                                               │
             └─► codex_sessions_root  ───────► codex_chunker.preprocess ─────┤
                              │                                              │
                              └─► route by cwd:                              │
                                  1. project.codex_cwds (explicit)           │
                                  2. bidirectional containment vs            │
                                     code_dirs[].path (resolved)             │
                                  3. ambiguous → orphan                      │
                                  4. no match → orphan                       │
                                                                             ▼
                  ┌──────────────────────────────────────────────────────────┘
                  │
                  ▼
         _index_session_files(project, files, parser, agent_tag)
              ├─► byte-offset resume (existing)
              ├─► identity-hash full re-emit (existing)
              ├─► chunk_index_offset on append (existing)
              ├─► tag chunks: metadata.agent = agent_tag
              └─► write to {project}-sessions collection (chroma) + BM25 sidecar

         orphan list ─► ~/.vecs/manifests/_codex_routing.json
                        ├─► consumed by MCP tools
                        └─► surfaced in MCP banners
```

## Files touched / added

| File | Action | Notes |
|------|--------|-------|
| `src/vecs/config.py` | Modify | Add `codex_sessions_root`, `codex_disabled`, `codex_ignore_cwds` to `VecsConfig`. Add `codex_cwds: list[Path]` to `ProjectConfig`. Update `load_config` + `save`. |
| `src/vecs/codex_chunker.py` | New | `preprocess_codex_session(raw_jsonl) -> list[dict]` (Claude-compat shape: `{role, text, timestamp}`). Schema-tolerant: skip unknown types. Logs unknown `payload.type` once per run. |
| `src/vecs/codex_routing.py` | New | `CodexRoutingState` (loads/saves `_codex_routing.json` under flock). `extract_session_meta(path) -> dict\|None` (line 1 read, mtime cache). `route_cwd(cwd, config) -> str\|None` (longest-bidirectional, ambiguity → None). `discover_codex_sessions(config) -> dict[str, list[Path]]` (project_name → files). |
| `src/vecs/indexer.py` | Modify | Extract `_index_session_files(project, files, parser_fn, agent_tag, collection)` from `index_sessions`. `index_sessions` becomes the Claude wrapper. Add `index_codex_sessions(routed, project, ...)`. Wire into `run_index`. Tag every emitted session chunk with `metadata.agent`. `codex_assign` cleanup: drop `session:{path}` entries + sweep stale chunks by `session_id` from old project's collection. |
| `src/vecs/mcp_server.py` | Modify | New tools: `codex_orphans()`, `codex_assign(cwd, project)`, `codex_ignore(cwd)`. Banner injection in `index_status` and `semantic_search` (only when sessions involved + orphans non-empty + dedupe-by-day). |
| `src/vecs/cli.py` | Modify | New subcommand group `vecs codex` → `orphans`, `assign`, `ignore`. |
| `tests/test_codex_chunker.py` | New | Schema parsing, role filter, content-part type tolerance, schema-drift skip. |
| `tests/test_codex_routing.py` | New | Bidirectional containment, ambiguity rejection, ignore list, explicit `codex_cwds` precedence, mtime cache. |
| `tests/test_indexer.py` | Modify | Add cases for `_index_session_files` shared core + agent tagging + reassignment cleanup. |
| `README.md` | Modify | Lead paragraph now mentions Codex sessions. New "Codex sessions" recipe. New CLI/MCP tool docs. |

## `_index_session_files` signature

```python
def _index_session_files(
    project: ProjectConfig,
    files: list[Path],                    # already routed to this project
    parser_fn: Callable[[str], list[dict]], # preprocess_session OR preprocess_codex_session
    agent_tag: Literal["claude_code", "codex"],
    vo: voyageai.Client,
    db: chromadb.ClientAPI,
    suffix: str = "sessions",             # for BM25 sidecar naming
) -> int:
    """Existing index_sessions body, parameterized.

    Reuses byte-offset resume, identity-hash rewrite detection,
    chunk_index_offset on append. Stamps `metadata.agent = agent_tag`
    on every emitted chunk before embed.
    """
```

`index_sessions(project, vo, db)` becomes a 5-line wrapper that walks `project.sessions_dirs`, calls `_index_session_files(..., parser_fn=preprocess_session, agent_tag="claude_code")`.

`index_codex_sessions(project, codex_files, vo, db)` calls `_index_session_files(..., parser_fn=preprocess_codex_session, agent_tag="codex")`.

## Routing algorithm

```
function route_cwd(cwd: Path, config: VecsConfig) -> Optional[ProjectName]:
    if cwd in config.codex_ignore_cwds: return None  # silent drop
    cwd_resolved = cwd.resolve()

    # 1. Explicit override wins.
    for proj in config.projects.values():
        for explicit in proj.codex_cwds:
            if explicit.resolve() == cwd_resolved or
               cwd_resolved.is_relative_to(explicit.resolve()):
                return proj.name

    # 2. Bidirectional containment.
    candidates: list[(score, project_name)] = []
    for proj in config.projects.values():
        for cd in proj.code_dirs:
            cd_resolved = cd.path.resolve()
            if cwd_resolved.is_relative_to(cd_resolved):
                # cwd inside code_dir; score = depth match
                candidates.append((len(cd_resolved.parts), proj.name))
            elif cd_resolved.is_relative_to(cwd_resolved):
                # code_dir inside cwd (cwd is ancestor)
                # Reject ambiguous ancestor: only accept if exactly one match.
                candidates.append((-1, proj.name))  # ancestor flag

    # Ambiguous ancestor (cwd matches > 1 project as ancestor) → orphan
    ancestors = [c for c in candidates if c[0] == -1]
    if ancestors:
        unique_ancestors = {c[1] for c in ancestors}
        if len(unique_ancestors) > 1:
            return None  # ambiguous, orphan
        return unique_ancestors.pop()

    if not candidates:
        return None  # orphan

    # Longest-prefix wins among non-ancestor matches
    candidates.sort(reverse=True)
    return candidates[0][1]
```

## Orphan persistence

Path: `~/.vecs/manifests/_codex_routing.json` (in manifests dir to share `fcntl.flock` infra).

```json
{
  "version": 1,
  "updated_at": "2026-05-05T...",
  "orphans": {
    "/Users/foo/Repositories": {
      "sessions": 28,
      "first_seen": "2026-04-26T...",
      "last_seen": "2026-05-05T..."
    }
  },
  "cwd_cache": {
    "<abs path to rollout-*.jsonl>": {
      "mtime": 1730000000.123,
      "cwd": "/Users/foo/repo/vecs",
      "session_id": "019dcbab-..."
    }
  }
}
```

Cache pruning: any cache entry whose file no longer exists is dropped on load. Orphan map is rebuilt every run (current routing snapshot, not cumulative).

## MCP banner injection

Trigger conditions (any of):
- `index_status` ALWAYS includes orphan summary line if orphans non-empty.
- `semantic_search` includes one-line banner ONLY when results contain at least one `sessions` collection hit AND orphan count > 0 AND not seen today (per-day dedupe stored in `_codex_routing.json` under `last_banner_day`).

Banner format:
```
[vecs] 12 codex sessions skipped (4 unmapped cwds). Triage: codex_orphans MCP tool.
```

## codex_assign / codex_ignore semantics

`codex_assign(cwd, project)`:
1. Add `cwd` to `projects[project].codex_cwds` in `config.yaml` (atomic write + `.bak`).
2. Find session files in `_codex_routing.json` cache whose `cwd` matches.
3. Sweep old project's `sessions` collection: for each `session_id`, delete chunks via `_paginated_delete`.
4. Drop `session:{path}` manifest entries for those files (forces full re-emit).
5. Next index run picks them up under new project, re-embedding from offset 0.

`codex_ignore(cwd)`:
1. Add `cwd` to top-level `codex_ignore_cwds`.
2. Sweep ANY project's chunks whose `session_id` came from a cached file with this `cwd`. Drop manifest `session:{path}` entries.
3. Future index runs skip silently.

Cost: re-embedding sessions (`voyage-3` @ $0.06/1M tokens). Documented; user accepts when assigning.

## Schema-tolerance behavior

Codex parser:
- Unknown `obj["type"]` → skip silently (no log).
- Known `obj["type"]=="response_item"` but unknown `payload["type"]` → debug log once per run via a module-level set.
- Unknown content-part `type` → skip that part, keep other parts of the message.
- Malformed JSON line → skip (existing pattern in Claude parser).

Never raise on schema drift.

## Testing strategy

- `test_codex_chunker.py`:
  - parses fixture with all observed top types
  - filters `developer`/`system` correctly
  - accepts `input_text`/`output_text`/`text` content parts
  - skips unknown payload types without raising
  - logs unknown payload type at most once per run

- `test_codex_routing.py`:
  - bidirectional containment matches both directions
  - ambiguous ancestor (`/Users/foo` cwd, 3 projects) → orphan
  - explicit `codex_cwds` overrides bidirectional match
  - `codex_ignore_cwds` returns None silently
  - mtime cache hits skip line-1 read
  - cache invalidation on mtime change

- `test_indexer.py` additions:
  - `_index_session_files` produces identical chunks for a given input regardless of agent_tag (only metadata.agent differs)
  - reassignment via `codex_assign` drops manifest entry + sweeps old chunks
  - chunks tagged `agent=claude_code` for existing path
  - chunks tagged `agent=codex` for new path

- Run full `uv run pytest -v`. Existing 100% pass rate must hold.

## Backward compatibility

- Existing `sessions_dirs` config: unchanged behavior. Claude Code path completely untouched.
- Already-indexed chunks lacking `metadata.agent`: search still works; default to `claude_code` in display logic.
- `codex_disabled=true` (or env `VECS_CODEX_DISABLED=1`): full opt-out, no Codex indexing, no orphan tracking.
- Codex root missing: behave as if disabled (zero noise).

## Open risks / mitigations

| Risk | Mitigation |
|------|------------|
| Codex schema bumps mid-development | Schema-tolerant parser, log unknowns once, cover by fixture-based tests. |
| Cwd cache grows unbounded | Prune on load: drop entries whose path no longer exists. |
| 10K+ Codex session files: walk + line-1 read each run | mtime-keyed cache: line-1 only re-read when file changes. |
| Banner UX is silent (agent doesn't surface) | `index_status` always includes orphan line — agent will see it on routine status checks. |
| Re-embed cost on `codex_assign` | Documented up front; user opts in by assigning. |
| Two projects share an ancestor cwd | Ambiguity rejection → orphan + actionable triage flow. |

## Out of scope (follow-up)

- Indexing Codex `reasoning` items behind `--include-reasoning` flag.
- Indexing tool calls + outputs.
- Cursor / Aider / Cody adapters (the `_index_session_files(parser_fn, agent_tag)` shape is ready for them).
