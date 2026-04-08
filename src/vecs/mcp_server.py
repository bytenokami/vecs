from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vecs.indexer import run_index, get_status
from vecs.searcher import search
from vecs.utils import slugify

mcp = FastMCP("vecs")


@mcp.tool()
def semantic_search(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
    path_filter: str | None = None,
    project: str | None = None,
) -> str:
    """Search code and session transcripts semantically.

    Args:
        query: Natural language search query.
        collection: Optional filter — "code", "sessions", or "docs". Searches all if omitted.
        n_results: Number of results to return (default 5).
        path_filter: Filter results to file paths containing this substring (e.g. "Services/Analytics/").
        project: Search a specific project (default: all).
    """
    results = search(
        query,
        collection_name=collection,
        n_results=n_results,
        path_filter=path_filter,
        project=project,
    )
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        proj = r.get("project", "?")
        header = f"--- Result {i} [{proj}:{r.get('collection', '?')}] {source}{dist_str} ---"
        text = r["text"]
        if len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


@mcp.tool()
def reindex(project: str | None = None) -> str:
    """Trigger incremental reindexing of code and session files.

    Args:
        project: Reindex a specific project (default: all configured projects).
    """
    try:
        run_index(project_name=project)
        status = get_status(project_name=project)
        return f"Reindex complete. {status['total_code_chunks']} code, {status['total_session_chunks']} session, {status['total_docs_chunks']} doc chunks."
    except Exception as e:
        return f"Reindex failed: {e}"


@mcp.tool()
def index_status(project: str | None = None) -> str:
    """Check the current index status — chunk counts and tracked files.

    Args:
        project: Status for a specific project (default: all).
    """
    status = get_status(project_name=project)
    lines = []
    for name, info in status.get("projects", {}).items():
        lines.append(f"[{name}] code: {info['code_chunks']}, sessions: {info['session_chunks']}, docs: {info['docs_chunks']} chunks")
    lines.append(f"Total: {status['total_code_chunks']} code + {status['total_session_chunks']} sessions + {status['total_docs_chunks']} docs")
    lines.append(f"Tracked files: {status.get('manifest_entries', 0)}")
    return "\n".join(lines)


@mcp.tool()
def add_document(
    content: str,
    title: str,
    project: str,
) -> str:
    """Save and index a document from the current conversation.

    Args:
        content: The document text to store.
        title: Document title (used as filename).
        project: Which project to store this under.
    """
    from vecs.config import load_config, VECS_DIR
    from vecs.indexer import index_single_doc

    config = load_config()
    if project not in config.projects:
        return f"Project '{project}' not found. Available: {', '.join(config.projects.keys())}"

    proj = config.projects[project]

    # Auto-configure docs_dir if not set
    if not proj.docs_dir:
        proj.docs_dir = VECS_DIR / "docs" / project
        config.save()

    proj.docs_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(title)
    file_path = proj.docs_dir / f"{slug}.md"
    file_path.write_text(content)

    try:
        stored = index_single_doc(project, file_path)
        return f"Saved '{title}' to {file_path} and indexed ({stored} chunks)."
    except Exception as e:
        return f"Saved '{title}' to {file_path} but indexing failed: {e}"
