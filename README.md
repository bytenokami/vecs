# VecS (Vector Search)

Semantic search for your codebase, Claude Code session transcripts, and documentation. AST-aware chunking via tree-sitter, hybrid Voyage AI vectors + SQLite FTS5 BM25, exposed to Claude Code as an MCP server.

---

## Install (MCP)

Most users run vecs as an MCP server inside Claude Code. Three steps:

```bash
# 1. Clone and install the CLI (needs Python 3.12+ and uv: https://docs.astral.sh/uv/)
git clone <repo-url> ~/Repositories/vecs && cd ~/Repositories/vecs
uv tool install --from . vecs

# 2. Set your Voyage API key — get one at https://www.voyageai.com
echo 'export VOYAGE_API_KEY="your-key-here"' >> ~/.zshrc && source ~/.zshrc
```

3. Register the MCP server in `~/.claude/settings.json`:

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

Restart Claude Code. The `vecs` MCP tools (`semantic_search`, `reindex`, `index_status`, `add_document`) are now available.

## Let Claude do the rest

Everything else — registering projects, choosing extensions, excluding noisy directories, pruning, reclaiming disk space, troubleshooting — lives in the [**AI Guide**](#ai-guide) below. It's written so Claude Code (or any MCP-aware agent) can read it and drive vecs for you end-to-end. Try:

> *"Set up vecs for my livly project at `~/Repositories/livly`. It's a Unity client + Go server monorepo. Exclude `Library/`, `.venv/`, `node_modules/`, and any generated `proto/` dirs."*

> *"My `~/.vecs/` is huge. Find the inflators and prune them."*

> *"Re-index just my server repo and tell me how many new chunks were added."*

Claude reads the AI Guide, runs the right `vecs ...` commands and SQLite queries, and reports back. You don't need to memorize the CLI.

If you'd rather drive it yourself, the AI Guide is also a complete human reference — start at "Setup".

---

# AI Guide

> This section is written for AI agents that drive vecs on a user's behalf — Claude Code, Cursor, etc. Every recipe is copy-pasteable. Humans benefit too.

## Setup (end-to-end, first time)

```bash
# Already done above if you followed the MCP install:
#   git clone <repo> && cd vecs && uv tool install --from . vecs
#   export VOYAGE_API_KEY="..."

# Register a project:
vecs project add myproject \
  --code-dir ~/Repositories/MyProject:.py,.md

# Index it (embeds + builds BM25):
vecs index -p myproject

# Search:
vecs search "your query" -p myproject
```

That's the minimum. For multi-repo projects, sessions, docs, and exclusions, see the recipes below.

## Recipes

### Add a project that spans multiple repos

`--code-dir` repeats. Each takes `path:ext1,ext2,...`.

```bash
vecs project add livly \
  --code-dir ~/Repositories/livly/client:.cs,.shader,.md \
  --code-dir ~/Repositories/livly/server:.go,.proto,.yml \
  --sessions-dir ~/.claude/projects/-Users-you-Repositories-livly-client \
  --docs-dir ~/Repositories/livly/docs
```

`--sessions-dir` is repeatable. `--docs-dir` is single (one docs root per project).

### Index everything

```bash
vecs index                          # all projects
vecs index -p myproject             # one project
vecs index --detect-project ~/Repositories/MyProject   # auto-detect by path
```

Subsequent runs are incremental — only files whose content hash changed are re-embedded.

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

Restrict a `code_dir` to specific subpaths. Edit `~/.vecs/config.yaml` directly:

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

After saving, run `vecs index -p myproject`. Anything outside `include_dirs` is treated as out-of-scope — existing chunks under non-allowed paths are swept from chromadb automatically.

### Blacklist subdirs (exclude_dirs)

Same file, opposite direction. Use for vendored libs, build output, virtualenvs, generated code:

```yaml
projects:
  myproject:
    code_dirs:
    - path: /abs/path/to/python-repo
      extensions: [.py, .md]
      exclude_dirs:               # any path under these is dropped
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

`exclude_dirs` wins over `include_dirs` when a path matches both. Each entry is a path **relative to the `code_dir.path`** — not a glob.

### Prune chunks for paths you just excluded

```bash
vecs index -p myproject
```

The next index run sweeps any existing chromadb chunks whose `file_path` is under a newly-excluded subdir. You'll see a log line like `pruned 4156 manifest entries now out of scope (excluded subdirs).` followed by the chunk deletion. The BM25 sidecar is synced in the same pass.

### Remove a project entirely

```bash
vecs project remove myproject
```

Drops the chromadb collections, BM25 `.db` files, and config entry. Embedding cost is sunk — re-adding the project means re-embedding from scratch.

### Reclaim disk space after large pruning

ChromaDB's underlying SQLite doesn't return free pages to the OS automatically. After a big sweep, run VACUUM:

```bash
# Stop any process holding ~/.vecs/chromadb/chroma.sqlite3 first.
# If you use the MCP server, your editor will respawn it on next use.
lsof ~/.vecs/chromadb/chroma.sqlite3      # find holders
kill <pid>                                # graceful stop

# Reclaim free pages:
python -c "import sqlite3; c=sqlite3.connect('$HOME/.vecs/chromadb/chroma.sqlite3'); c.execute('PRAGMA journal_mode=DELETE'); c.execute('VACUUM'); c.execute('PRAGMA journal_mode=WAL'); c.close()"

du -sh ~/.vecs/chromadb/chroma.sqlite3    # confirm shrink
```

ChromaDB's HNSW vector dirs (`~/.vecs/chromadb/<uuid>/`) don't auto-compact either. To fully reclaim those you'd need to drop and re-index a collection (Voyage API cost). Tombstoned vectors are harmless — they're skipped at query time.

### Find and delete orphan vector dirs

A vector dir is "orphan" when no live ChromaDB segment references it (collection drops, aborted runs):

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

Always run this with the MCP server stopped (it can re-create them mid-script otherwise).

### Find which paths are inflating a collection

Use this output to decide what to add to `exclude_dirs`:

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

### Add a one-off document to a project's `docs` collection

```bash
echo "Notes from today's standup..." | vecs add -p myproject -t "2026-04-27 standup"
```

The doc gets stored under the project's `docs_dir`, embedded immediately, and is searchable via `vecs search ... -c docs`.

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

`~/.vecs/config.yaml` is the source of truth. The CLI commands (`project add`, `project remove`) edit this file, but you can also edit it directly — the next `vecs index` picks up changes.

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
- `extensions` includes the leading dot.
- `include_dirs` and `exclude_dirs` are paths relative to that `code_dir.path`. Not globs — directory prefixes.
- `exclude_dirs` wins over `include_dirs` when both match.
- A project must have at least one `code_dir`. `sessions_dir` and `docs_dir` are optional.
- `~/.vecs/config.yaml.bak` is a snapshot the CLI writes before each edit.

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
| `~/.vecs/config.yaml`                | The project registry (source of truth)                       |
| `~/.vecs/config.yaml.bak`            | Auto-snapshot from the last CLI write                        |
| `~/.vecs/chromadb/chroma.sqlite3`    | ChromaDB metadata + document store                           |
| `~/.vecs/chromadb/<uuid>/`           | HNSW vector index for one collection segment                 |
| `~/.vecs/bm25/<project>_<suffix>.db` | SQLite FTS5 BM25 sidecar (`suffix` ∈ `code/sessions/docs`)   |
| `~/.vecs/manifests/<project>.json`   | Content hashes per file for incremental indexing             |
| `~/.vecs/manifest.json`              | Legacy single-file manifest (still read for migration)       |
| `~/.vecs/docs/<project>/`            | Auto-created docs target for `vecs add` if no `docs_dir`     |
| `~/.vecs/reindex.log`                | launchd / cron index logs (if you set up auto-reindex)       |

## MCP tools (already wired by the install above)

- `semantic_search(query, collection?, n_results?, path_filter?, project?)` — hybrid vector + BM25 search across code / sessions / docs
- `reindex(project?)` — incremental reindex
- `index_status(project?)` — per-project chunk counts
- `add_document(content, title, project)` — save and index a document into the project's `docs_dir` (auto-creates one under `~/.vecs/docs/{project}/` if not configured)

The server holds an open handle on `~/.vecs/chromadb/chroma.sqlite3`. Stop it before running VACUUM or moving files.

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

Logs carry ISO timestamps, per-project durations, and chunk-id context on truncation/retry — grep by date or `attempt N/5` to forensically reconstruct a run.

## How it works

1. **Discovery** — for each project's `code_dirs`, walk the tree, filter by `extensions`, then by `include_dirs` / `exclude_dirs`. Hash file content, compare against the per-project manifest, queue only changed/new files.
2. **Chunking**
   - Code: `.cs` / `.ts` / `.tsx` parsed with tree-sitter and split at class/method boundaries; other extensions use line-based windows (200 lines, 50 overlap).
   - Sessions: append-only JSONL chunked into 10-message windows with 2-message overlap; `byte_offset` stored so subsequent runs only read new bytes.
   - Docs: Markdown and PDF split at heading boundaries; plain text falls back to size-based chunks.
3. **Embedding** — Voyage AI (`voyage-code-3` for code, `voyage-3` for sessions and docs) via an adaptive batcher that calibrates the char-to-token ratio from API responses. Transient errors retry with backoff; permanent errors fail fast.
4. **Storage** — vectors + metadata go into ChromaDB; a parallel SQLite FTS5 BM25 sidecar (`~/.vecs/bm25/{project}_{collection}.db`) is incrementally upserted as chunks change.
5. **Cleanup** — files removed from disk are pruned from the manifest; files now under `exclude_dirs` are pruned and their chromadb chunks swept.
6. **Search** — query embedded once (cached, 5-min TTL), runs against vectors and BM25, results merged via Reciprocal Rank Fusion, deduplicated, returned.

## Embedding cost

| Collection     | Model            | Cost              |
|----------------|------------------|-------------------|
| code           | `voyage-code-3`  | free              |
| sessions, docs | `voyage-3`       | $0.06 / 1M tokens |

Re-embedding a project costs roughly: `total_chunks × ~500 tokens × model_rate`. For a 10K-chunk Python repo on `voyage-3`, that's ≈ $0.30. Code is free, so most users only pay for sessions and docs.

## Development

```bash
uv sync
uv run pytest -v                                       # full suite
uv run pytest tests/test_bm25.py -v                    # one file
uv run python scripts/bench_bm25.py                    # A/B benchmark vs rank_bm25
                                                       # (needs: uv pip install rank_bm25)
```

Plan and benchmark notes live under `docs/superpowers/plans/`.
