# VecS (Vector Search)

Hybrid search for codebase, AI agent session transcripts (Claude Code + Codex CLI), and docs. Voyage AI **embeddings** power **semantic** **vector** recall (catch meaning, not keywords); SQLite FTS5 **BM25** sidecar pins exact symbols and rare terms; results fused via Reciprocal Rank Fusion. AST-aware chunking via tree-sitter splits code at real class/method boundaries, not random line windows. Codex sessions auto-route to projects by `cwd`, so one query searches every transcript from every agent, per project. Result: agent finds the right function by intent ("auth retry logic") AND by exact name (`refreshSessionToken`), in one query, across every repo + transcript + doc you point at it. Wired to Claude Code, Codex, Cursor (and any MCP agent) as drop-in tool.

---

## Install (MCP)

Most users run vecs as MCP server inside Claude Code. Three steps:

```bash
# 1. Clone + install CLI (need Python 3.12+ and uv: https://docs.astral.sh/uv/)
git clone <repo-url> ~/Repositories/vecs && cd ~/Repositories/vecs
uv tool install --from . vecs

# 2. Set Voyage API key — grab one at https://www.voyageai.com
echo 'export VOYAGE_API_KEY="your-key-here"' >> ~/.zshrc && source ~/.zshrc
```

3. Register MCP server with agent. JSON shape same for most clients — only config path differs:

```json
{
  "mcpServers": {
    "vecs": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/YOU/Repositories/vecs", "mcp", "run", "src/vecs/mcp_server.py"],
      "env": { "VOYAGE_API_KEY": "your-key-here" }
    }
  }
}
```

| Agent           | Config file                                                          |
|-----------------|----------------------------------------------------------------------|
| Claude Code     | `~/.claude/settings.json`                                            |
| Codex CLI       | `~/.codex/config.toml` — convert JSON to equivalent TOML block       |
| Cursor          | Settings → MCP → Add server                                          |
| Other           | See agent's MCP docs — `command` / `args` / `env` map directly       |

Restart agent. Tools `semantic_search`, `reindex`, `index_status`, `add_document` now live.

## Let agent do rest

Everything else — register projects, pick extensions, exclude noisy dirs, prune, reclaim disk, troubleshoot — lives in AI Guide below. Written so any MCP-aware agent (Claude Code, Codex, Cursor, …) reads it and drives vecs end-to-end. Try:

> *"Set up vecs for my livly project at `~/Repositories/livly`. Unity client + Go server monorepo. Exclude `Library/`, `.venv/`, `node_modules/`, generated `proto/` dirs."*

> *"My `~/.vecs/` huge. Find inflators and prune."*

> *"Re-index just server repo. Tell me how many new chunks added."*

Agent reads AI Guide, runs `vecs ...` commands + SQLite queries, reports back. No CLI memorization needed.

Prefer hands-on? AI Guide doubles as full human reference — start at "Setup".

---

# AI Guide

> Section written for any MCP-aware AI agent driving vecs on user's behalf (Claude Code, Codex, Cursor, …). Every recipe copy-pasteable. Humans benefit too.

## Setup (end-to-end, first time)

```bash
# Already done above if you followed MCP install:
#   git clone <repo> && cd vecs && uv tool install --from . vecs
#   export VOYAGE_API_KEY="..."

# Register project:
vecs project add myproject \
  --code-dir ~/Repositories/MyProject:.py,.md

# Index it (embeds + builds BM25):
vecs index -p myproject

# Search:
vecs search "your query" -p myproject
```

Minimum done. Multi-repo projects, sessions, docs, exclusions — see recipes below.

## Recipes

### Codex CLI sessions (auto-routed by `cwd`)

Codex stores sessions globally at `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`, not per-project. vecs walks that tree, reads `session_meta.payload.cwd` from each file, and routes the session to the project whose `code_dirs` match (bidirectional resolved-path containment). No per-project config needed — sessions land in the right project automatically the first time you run `vecs index`.

```yaml
# Optional top-level overrides in ~/.vecs/config.yaml
codex_sessions_root: ~/.codex/sessions    # default; rarely changed
codex_disabled: false                     # or set VECS_CODEX_DISABLED=1
codex_ignore_cwds:                        # never index sessions from these
  - /tmp/scratch
projects:
  livly:
    codex_cwds:                           # explicit routing overrides cwd matching
      - /Users/you/Repositories/livly-old-path
```

Sessions whose `cwd` matches no project (e.g. `cwd=/Users/you/Repositories`, ancestor of multiple projects → ambiguous → orphan) get tracked, not silently dropped. Triage with the agent or CLI:

```bash
vecs codex orphans                                  # list unmapped cwds + suggested projects
vecs codex assign /Users/you/Repositories/foo -p livly   # route + invalidate, re-index next run
vecs codex ignore /tmp/scratch                      # never index sessions from this cwd
```

The same operations are exposed as MCP tools (`codex_orphans`, `codex_assign`, `codex_ignore`); your agent surfaces them when `index_status` or `semantic_search` notice orphans.

Both Claude Code and Codex transcripts share one `sessions` collection per project. Each chunk carries `agent ∈ {claude_code, codex}` metadata so search results display the source. Search both with one query:

```bash
vecs search "auth retry discussion" -c sessions
```

### Add project spanning multiple repos

`--code-dir` repeats. Each takes `path:ext1,ext2,...`.

```bash
vecs project add livly \
  --code-dir ~/Repositories/livly/client:.cs,.shader,.md \
  --code-dir ~/Repositories/livly/server:.go,.proto,.yml \
  --sessions-dir ~/.claude/projects/-Users-you-Repositories-livly-client \
  --docs-dir ~/Repositories/livly/docs
```

`--sessions-dir` repeatable. `--docs-dir` single (one docs root per project).

### Index everything

```bash
vecs index                          # all projects
vecs index -p myproject             # one project
vecs index --detect-project ~/Repositories/MyProject   # auto-detect by path
```

Subsequent runs incremental — only files with changed content hash re-embed.

### Search

```bash
vecs search "auth flow"                          # all projects, all collections, top 5
vecs search "animation state machine" -n 10      # top 10
vecs search "analytics" -f Services/Analytics/   # path filter (substring match)
vecs search "that bug" -p livly -c sessions      # one project, sessions only
vecs search "fork plan" -c docs                  # docs only
```

`-c` accepts `code | sessions | docs`.

### Whitelist subdirs (include_dirs)

Restrict `code_dir` to specific subpaths. Edit `~/.vecs/config.yaml` directly:

```yaml
projects:
  myproject:
    code_dirs:
    - path: /abs/path/to/repo
      extensions: [.cs]
      include_dirs:               # only these subpaths get indexed
      - Assets/Scripts
      - Assets/Tests
      - Assets/Editor
```

After save, run `vecs index -p myproject`. Anything outside `include_dirs` treated out-of-scope — existing chunks under non-allowed paths swept from chromadb automatically.

### Blacklist subdirs (exclude_dirs)

Same file, opposite direction. Use for vendored libs, build output, virtualenvs, generated code:

```yaml
projects:
  myproject:
    code_dirs:
    - path: /abs/path/to/python-repo
      extensions: [.py, .md]
      exclude_dirs:               # any path under these dropped
      - .venv
      - node_modules
      - dist
      - generated
    - path: /abs/path/to/unity-client
      extensions: [.cs]
      exclude_dirs:
      - Library            # Unity's package cache
      - LibraryOrigin
      - Temp
      - obj
      - Logs
      - Build
      - Assets/Plugins                       # third-party SDKs
      - Assets/App/Scripts/p2-proto          # protobuf-generated
```

`exclude_dirs` wins over `include_dirs` when path matches both. Each entry is path **relative to `code_dir.path`** — not glob.

### Prune chunks for paths just excluded

```bash
vecs index -p myproject
```

Next index run sweeps existing chromadb chunks whose `file_path` under newly-excluded subdir. Log line: `pruned 4156 manifest entries now out of scope (excluded subdirs).` followed by chunk deletion. BM25 sidecar synced same pass.

### Remove project entirely

```bash
vecs project remove myproject
```

Drops chromadb collections, BM25 `.db` files, config entry. Embedding cost sunk — re-add means re-embed from scratch.

### Reclaim disk space after big pruning

ChromaDB's underlying SQLite doesn't return free pages to OS automatically. After big sweep, run VACUUM:

```bash
# Stop any process holding ~/.vecs/chromadb/chroma.sqlite3 first.
# If you use MCP server, your editor respawns it on next use.
lsof ~/.vecs/chromadb/chroma.sqlite3      # find holders
kill <pid>                                # graceful stop

# Reclaim free pages:
python -c "import sqlite3; c=sqlite3.connect('$HOME/.vecs/chromadb/chroma.sqlite3'); c.execute('PRAGMA journal_mode=DELETE'); c.execute('VACUUM'); c.execute('PRAGMA journal_mode=WAL'); c.close()"

du -sh ~/.vecs/chromadb/chroma.sqlite3    # confirm shrink
```

ChromaDB's HNSW vector dirs (`~/.vecs/chromadb/<uuid>/`) don't auto-compact either. Full reclaim needs drop + re-index of collection (Voyage API cost). Tombstoned vectors harmless — skipped at query time.

### Find and delete orphan vector dirs

Vector dir "orphan" when no live ChromaDB segment references it (collection drops, aborted runs):

```bash
python <<'PY'
import sqlite3, os
c = sqlite3.connect(os.path.expanduser('~/.vecs/chromadb/chroma.sqlite3'))
live = {r[0] for r in c.execute('SELECT id FROM segments')}
root = os.path.expanduser('~/.vecs/chromadb')
for d in os.listdir(root):
    p = f'{root}/{d}'
    if os.path.isdir(p) and d not in live:
        size_mb = sum(os.path.getsize(f'{p}/{f}') for f in os.listdir(p) if os.path.isfile(f'{p}/{f}'))/1024/1024
        print(f'  ORPHAN  {size_mb:>7.1f} MB  {d}')
PY
# Delete each one with: rm -rf ~/.vecs/chromadb/<orphan-uuid>
```

Always run with MCP server stopped (can re-create them mid-script otherwise).

### Find which paths inflate a collection

Use this output to pick what to add to `exclude_dirs`:

```bash
python <<'PY'
import sqlite3, os
from collections import Counter
db = sqlite3.connect(os.path.expanduser('~/.vecs/chromadb/chroma.sqlite3'))
db.row_factory = sqlite3.Row
COLLECTION = 'livly-code'   # change me
cid = db.execute('SELECT id FROM collections WHERE name=?', (COLLECTION,)).fetchone()['id']
seg_ids = [r['id'] for r in db.execute('SELECT id FROM segments WHERE collection=?', (cid,))]
qmarks = ','.join('?' * len(seg_ids))
paths = [r['string_value'] for r in db.execute(
    f"SELECT string_value FROM embedding_metadata "
    f"WHERE key='file_path' AND id IN (SELECT id FROM embeddings WHERE segment_id IN ({qmarks}))",
    seg_ids)]
print(f'{COLLECTION}: {len(paths)} chunks')
print('\nTop 15 directories (first 2 path segments):')
for d, n in Counter('/'.join(p.split('/')[:2]) for p in paths).most_common(15):
    print(f'  {n:>5d}  {d}')
print('\nTop 10 files by chunk count:')
for p, n in Counter(paths).most_common(10):
    print(f'  {n:>5d}  {p}')
PY
```

### Add one-off document to project's `docs` collection

```bash
echo "Notes from today's standup..." | vecs add -p myproject -t "2026-04-27 standup"
```

Doc stored under project's `docs_dir`, embedded immediately, searchable via `vecs search ... -c docs`.

### Check status

```bash
vecs status                # all projects
vecs status -p myproject   # one project
```

### Inspect storage size

```bash
du -sh ~/.vecs/                                # total
du -sh ~/.vecs/*                               # by component (chromadb, bm25, manifests)
du -sh ~/.vecs/chromadb/* | sort -hr | head    # biggest segments
```

## Configuration reference

`~/.vecs/config.yaml` is source of truth. CLI commands (`project add`, `project remove`) edit this file, but you can edit directly — next `vecs index` picks up changes.

```yaml
projects:
  myproject:
    code_dirs:                  # list, repeatable
    - path: /abs/path/to/repo   # required, absolute path
      extensions: [.cs, .md]    # required, list of file suffixes incl. dot
      include_dirs:             # optional allowlist (relative to path)
      - Assets/Scripts
      exclude_dirs:             # optional blocklist (relative to path)
      - Library
      - .venv
    sessions_dir: /Users/you/.claude/projects/-Users-you-Repositories-myproject   # optional
    docs_dir: /abs/path/to/docs                                                   # optional
```

Rules:
- `extensions` includes leading dot.
- `include_dirs` and `exclude_dirs` are paths relative to that `code_dir.path`. Not globs — directory prefixes.
- `exclude_dirs` wins over `include_dirs` when both match.
- Project must have at least one `code_dir`. `sessions_dir` and `docs_dir` optional.
- `~/.vecs/config.yaml.bak` is snapshot CLI writes before each edit.

CLI:

```bash
vecs project list
vecs project add <name> --code-dir <path:ext> [--code-dir ...] [--sessions-dir ...] [--docs-dir ...]
vecs project remove <name>
```

## Storage layout

Everything vecs writes lives under `~/.vecs/`:

| Path                                 | What it is                                                   |
|--------------------------------------|--------------------------------------------------------------|
| `~/.vecs/config.yaml`                | Project registry (source of truth)                           |
| `~/.vecs/config.yaml.bak`            | Auto-snapshot from last CLI write                            |
| `~/.vecs/chromadb/chroma.sqlite3`    | ChromaDB metadata + document store                           |
| `~/.vecs/chromadb/<uuid>/`           | HNSW vector index for one collection segment                 |
| `~/.vecs/bm25/<project>_<suffix>.db` | SQLite FTS5 BM25 sidecar (`suffix` ∈ `code/sessions/docs`)   |
| `~/.vecs/manifests/<project>.json`   | Content hashes per file for incremental indexing             |
| `~/.vecs/manifests/_codex_routing.json` | Codex orphan census + cwd-cache for fast re-walks         |
| `~/.vecs/manifest.json`              | Legacy single-file manifest (still read for migration)       |
| `~/.vecs/docs/<project>/`            | Auto-created docs target for `vecs add` if no `docs_dir`     |
| `~/.vecs/reindex.log`                | launchd / cron index logs (if you set up auto-reindex)       |

## MCP tools (already wired by install above)

- `semantic_search(query, collection?, n_results?, path_filter?, project?)` — hybrid vector + BM25 search across code / sessions / docs. Session results display `{claude_code}` or `{codex}` agent tag.
- `reindex(project?)` — incremental reindex (Claude Code + Codex sessions in one pass)
- `index_status(project?)` — per-project chunk counts; appends Codex orphan summary when present
- `add_document(content, title, project)` — save and index document into project's `docs_dir` (auto-creates one under `~/.vecs/docs/{project}/` if not configured)
- `codex_orphans()` — list Codex `cwd`s that didn't match any project, with project suggestions
- `codex_assign(cwd, project)` — route Codex sessions for a `cwd` to a project, invalidate stale chunks/manifest entries
- `codex_ignore(cwd)` — stop indexing Codex sessions captured at `cwd`, sweep already-indexed chunks

Server holds open handle on `~/.vecs/chromadb/chroma.sqlite3`. Stop it before VACUUM or moving files.

## Auto-reindex (launchd)

Run `vecs index` periodically so search results stay fresh. macOS example, every 6 hours, wrapped in `caffeinate`:

`~/Library/LaunchAgents/com.vecs.reindex.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.vecs.reindex</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/caffeinate</string>
        <string>-i</string><string>-m</string><string>-s</string>
        <string>/Users/YOU/.local/bin/vecs</string>
        <string>index</string>
    </array>
    <key>StartInterval</key><integer>21600</integer>
    <key>RunAtLoad</key><false/>
    <key>StandardOutPath</key><string>/Users/YOU/.vecs/reindex.log</string>
    <key>StandardErrorPath</key><string>/Users/YOU/.vecs/reindex.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>VOYAGE_API_KEY</key><string>your-key</string>
        <key>PATH</key><string>/Users/YOU/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.vecs.reindex.plist     # enable
launchctl start com.vecs.reindex                                  # force run now
launchctl unload ~/Library/LaunchAgents/com.vecs.reindex.plist   # disable
tail -f ~/.vecs/reindex.log                                       # watch
```

Logs carry ISO timestamps, per-project durations, chunk-id context on truncation/retry — grep by date or `attempt N/5` to forensically reconstruct run.

## How it works

1. **Discovery** — for each project's `code_dirs`, walk tree, filter by `extensions`, then by `include_dirs` / `exclude_dirs`. Hash file content, compare against per-project manifest, queue only changed/new files.
2. **Chunking**
   - Code: `.cs` / `.ts` / `.tsx` parsed with tree-sitter, split at class/method boundaries; other extensions use line-based windows (200 lines, 50 overlap).
   - Sessions: append-only JSONL chunked into 10-message windows with 2-message overlap; `byte_offset` stored so subsequent runs only read new bytes. Two parsers (Claude Code, Codex) feed one shared chunking pipeline; chunks tagged with `agent` metadata.
   - Codex routing: each `rollout-*.jsonl` is keyed by `session_meta.payload.cwd` (mtime-cached so subsequent walks skip parsing line 1). Bidirectional resolved-path containment matches `cwd` to projects via `code_dirs[].path`; ambiguous ancestor cwds go to orphan tracking, never silently misrouted.
   - Docs: Markdown and PDF split at heading boundaries; plain text falls back to size-based chunks.
3. **Embedding** — Voyage AI (`voyage-code-3` for code, `voyage-3` for sessions and docs) via adaptive batcher that calibrates char-to-token ratio from API responses. Transient errors retry with backoff; permanent errors fail fast.
4. **Storage** — vectors + metadata go into ChromaDB; parallel SQLite FTS5 BM25 sidecar (`~/.vecs/bm25/{project}_{collection}.db`) incrementally upserted as chunks change.
5. **Cleanup** — files removed from disk pruned from manifest; files now under `exclude_dirs` pruned and chromadb chunks swept.
6. **Search** — query embedded once (cached, 5-min TTL), runs against vectors and BM25, results merged via Reciprocal Rank Fusion, deduplicated, returned.

## Embedding cost

| Collection     | Model            | Cost              |
|----------------|------------------|-------------------|
| code           | `voyage-code-3`  | free              |
| sessions, docs | `voyage-3`       | $0.06 / 1M tokens |

Re-embedding project costs roughly: `total_chunks × ~500 tokens × model_rate`. For 10K-chunk Python repo on `voyage-3`, ≈ $0.30. Code free, so most users only pay for sessions and docs.

## Development

```bash
uv sync
uv run pytest -v                                       # full suite
uv run pytest tests/test_bm25.py -v                    # one file
uv run python scripts/bench_bm25.py                    # A/B benchmark vs rank_bm25
                                                       # (needs: uv pip install rank_bm25)
```

Plan and benchmark notes live under `docs/superpowers/plans/`.
