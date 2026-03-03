# VecS (Vector Search)

Semantic search for your codebase and Claude Code sessions. Indexes code files using AST-aware chunking (tree-sitter) and session transcripts, then lets you search with hybrid vector + keyword matching.

## Features

- **AST-aware chunking** — splits C# and TypeScript at class/method boundaries via tree-sitter (line-based fallback for other languages)
- **Hybrid search** — vector similarity (Voyage AI) + BM25 keyword matching, merged with Reciprocal Rank Fusion
- **Multi-project** — register multiple codebases, each with its own file extensions and session directory
- **MCP server** — exposes `semantic_search`, `reindex`, and `index_status` as tools for Claude Code
- **Path filtering** — scope searches to specific directories
- **Result deduplication** — removes near-duplicate chunks from overlapping windows
- **Query caching** — 5-minute TTL cache for embedding calls

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd vecs
uv tool install --editable .
```

Set your Voyage API key:

```bash
export VOYAGE_API_KEY="your-key-here"
```

## Quick Start

### 1. Register a project

```bash
vecs project add myproject \
  --code-dir ~/Repositories/MyProject/Assets \
  --ext .cs,.ts,.json
```

Optionally include Claude Code sessions:

```bash
vecs project add myproject \
  --code-dir ~/Repositories/MyProject/Assets \
  --ext .cs,.ts \
  --sessions-dir ~/.claude/projects/-Users-you-Repositories-MyProject
```

### 2. Index

```bash
vecs index                    # index all projects
vecs index --project myproject  # index one project
```

### 3. Search

```bash
vecs search "animation state machine"
vecs search "auth flow" --collection code --limit 10
vecs search "analytics" --path-filter Services/Analytics/
vecs search "that bug" --project myproject --collection sessions
```

### 4. Check status

```bash
vecs status
```

## MCP Server (Claude Code integration)

Add to your Claude Code settings (`~/.claude/settings.json`):

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

Available tools:
- `semantic_search(query, collection?, n_results?, path_filter?, project?)` — search code and sessions
- `reindex(project?)` — trigger incremental reindex
- `index_status(project?)` — check chunk counts

## Project Management

```bash
vecs project list              # show registered projects
vecs project add <name> ...    # register a project
vecs project remove <name>     # unregister a project
```

Config is stored at `~/.vecs/config.yaml`.

## How It Works

1. **Chunking** — C# and TypeScript files are parsed with tree-sitter and split at class/method boundaries. Other file types use line-based chunking (200 lines, 50 overlap). Session transcripts are chunked in 10-message windows with 2-message overlap.
2. **Embedding** — chunks are embedded with Voyage AI (`voyage-code-3` for code, `voyage-3` for sessions) and stored in ChromaDB.
3. **BM25 index** — a keyword index is built alongside the vector index for hybrid search.
4. **Search** — queries run against both vector and BM25 indexes, results are merged with Reciprocal Rank Fusion, deduplicated, and returned.

## Development

```bash
uv sync
uv run pytest -v
```
