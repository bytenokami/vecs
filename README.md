# VecS (Vector Search)

Semantic search for your codebase, Claude Code session transcripts, and documentation. AST-aware chunking via tree-sitter, hybrid vector + BM25 ranking, MCP server for Claude Code integration.

## Features

- **AST-aware code chunking** — C# / TypeScript / TSX split at class/method boundaries via tree-sitter; line-based fallback for other languages
- **Document chunking** — Markdown and PDF split at heading boundaries; plain text falls back to size-based chunks
- **Incremental session indexing** — append-only JSONL transcripts tracked by byte offset; only new bytes are re-embedded
- **Hybrid search** — Voyage AI vectors + BM25 keyword index, merged with Reciprocal Rank Fusion (`w_vector=1.0`, `w_bm25=0.6`)
- **Code-aware BM25 tokenizer** — splits `camelCase`, `PascalCase`, `ACRONYMS`, `snake_case` so identifier searches work
- **Multi-repo projects** — one project can span many `code_dirs` and `sessions_dirs`, plus a single `docs_dir`
- **`include_dirs` / `exclude_dirs`** — per-`code_dir` allowlist or blocklist of subpaths (e.g. exclude Unity's `Library/`)
- **Adaptive batching** — token-count EMA calibration to fit Voyage's 120K-token request limit without over- or under-packing
- **Per-project manifests** — content-hash tracking at `~/.vecs/manifests/{project}.json`; only changed files re-embed
- **Transient-error retry** — Voyage timeouts, rate limits, and 5xx are retried with backoff; non-transient errors fail fast
- **Orphan sweep** — chromadb chunks under newly-excluded subdirs are removed on the next index pass
- **MCP server** — exposes `semantic_search`, `reindex`, `index_status`, `add_document` to Claude Code
- **Path filtering** — scope searches to file paths matching a substring
- **Result deduplication + fetch escalation** — removes near-duplicate chunks; widens the vector fetch window when dedup eats too many

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd vecs
uv tool install --from . vecs
```

Set your Voyage API key:

```bash
export VOYAGE_API_KEY="your-key-here"
```

## Quick Start

### 1. Register a project

`--code-dir` takes the form `path:ext1,ext2,...` and is repeatable for multi-repo projects:

```bash
vecs project add myproject \
  --code-dir ~/Repositories/MyProject/client:.cs,.shader \
  --code-dir ~/Repositories/MyProject/server:.go,.proto \
  --sessions-dir ~/.claude/projects/-Users-you-Repositories-MyProject \
  --docs-dir ~/Repositories/MyProject/docs
```

`--sessions-dir` and `--docs-dir` are optional. `--sessions-dir` is repeatable.

### 2. Index

```bash
vecs index                          # all projects
vecs index -p myproject             # one project
vecs index --detect-project ~/Repositories/MyProject/client
```

### 3. Search

```bash
vecs search "animation state machine"
vecs search "auth flow" -c code -n 10
vecs search "analytics" -f Services/Analytics/
vecs search "that bug" -p myproject -c sessions
vecs search "fork plan" -c docs
```

### 4. Status

```bash
vecs status
vecs status -p myproject
```

### 5. Add a one-off document

Pipe content into the `docs` collection of a project:

```bash
echo "Notes from today's meeting…" | vecs add -p myproject -t "2026-04-27 standup"
```

## Project configuration

`~/.vecs/config.yaml` is the source of truth. Schema per project:

```yaml
projects:
  myproject:
    code_dirs:
    - path: /abs/path/to/client
      extensions: [.cs, .shader, .md]
      exclude_dirs: [Library, Temp, obj, Logs]   # optional
      include_dirs: [Assets, Packages]           # optional, allowlist
    - path: /abs/path/to/server
      extensions: [.go, .proto]
      exclude_dirs: [vendor, ap/src/infrastructure/grpc/proto]
    sessions_dir: /Users/you/.claude/projects/-Users-you-Repositories-myproject
    docs_dir: /abs/path/to/docs
```

`exclude_dirs` wins over `include_dirs` when a path matches both.

```bash
vecs project list
vecs project add <name> ...
vecs project remove <name>
```

## MCP Server (Claude Code integration)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "vecs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vecs", "mcp", "run", "src/vecs/mcp_server.py"]
    }
  }
}
```

Tools exposed:

- `semantic_search(query, collection?, n_results?, path_filter?, project?)` — hybrid vector + BM25 search across code / sessions / docs
- `reindex(project?)` — incremental reindex
- `index_status(project?)` — per-project chunk counts
- `add_document(content, title, project)` — save and index a document into the project's `docs_dir` (auto-creates one under `~/.vecs/docs/{project}/` if not configured)

## Auto-reindex (launchd / cron)

Run `vecs index` periodically so your search results stay fresh. macOS example using launchd, every 6 hours, wrapped in `caffeinate` so it survives a closed lid on AC power:

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

## How It Works

1. **Discovery** — for each project's `code_dirs`, walk the tree, filter by `extensions`, then by `include_dirs` / `exclude_dirs`. Hash file content, compare against the per-project manifest, queue only changed/new files.
2. **Chunking**
   - Code: `.cs` / `.ts` / `.tsx` parsed with tree-sitter and split at class/method boundaries; other extensions use line-based windows (200 lines, 50 overlap).
   - Sessions: append-only JSONL chunked into 10-message windows with 2-message overlap; `byte_offset` stored so subsequent runs only read new bytes.
   - Docs: Markdown and PDF split at heading boundaries; plain text falls back to size-based chunks.
3. **Embedding** — Voyage AI (`voyage-code-3` for code, `voyage-3` for sessions and docs) via an adaptive batcher that calibrates the char-to-token ratio from API responses. Transient errors (timeout, rate limit, 5xx) retry with backoff; permanent errors fail fast.
4. **Storage** — vectors + metadata go into ChromaDB; a parallel BM25 sidecar (`~/.vecs/bm25/{project}_{collection}.pkl`) is rebuilt when chunks change.
5. **Cleanup** — files removed from disk are pruned from the manifest; files now under `exclude_dirs` are pruned and their chromadb chunks swept (catches orphans from prior partially-completed runs too).
6. **Search** — query embedded once (cached, 5-min TTL), runs against vectors and BM25, results merged via Reciprocal Rank Fusion, deduplicated, returned.

## Embedding cost

| Collection | Model | Cost |
|------------|-------|------|
| code | `voyage-code-3` | free |
| sessions, docs | `voyage-3` | $0.06/1M tokens |

## Development

```bash
uv sync
uv run pytest -v
```
