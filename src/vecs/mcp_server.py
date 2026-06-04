from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from vecs.config import load_config
from vecs.indexer import get_status, run_index
from vecs.searcher import search
from vecs.utils import slugify

mcp = FastMCP("vecs")


def _freshness_tag(meta: dict) -> str:
    """Per-hit freshness/trust signal from chunk metadata (Inc 1.5c).

    The chunk's ``version_id`` (git HEAD sha for code, file mtime for docs;
    stamped by the indexer, C) is the freshness anchor. A 40-hex git sha is
    shortened to 8 chars for a readable header; an mtime/content-hash proxy is
    left verbatim (shortening would mangle it). A chunk with no version_id
    (legacy, pre-C) surfaces an explicit ``unknown`` bucket rather than hiding
    the absent signal.
    """
    v = meta.get("version_id")
    if not v:
        return "v:unknown"
    v = str(v)
    if len(v) == 40 and all(c in "0123456789abcdef" for c in v.lower()):
        v = v[:8]
    return f"v:{v}"


@mcp.tool()
def semantic_search(
    query: str,
    collection: str | None = None,
    n_results: int = 5,
    path_filter: str | None = None,
    project: str | None = None,
) -> str:
    """Search code and docs semantically.

    Args:
        query: Natural language search query.
        collection: Optional filter — "code" or "docs". Searches all if omitted.
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
        source = meta.get("file_path") or "?"
        collection_name = r.get("collection", "?") or ""
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        proj = r.get("project", "?")
        tag = _freshness_tag(meta)
        header = f"--- Result {i} [{proj}:{collection_name}] {source}{dist_str} [{tag}] ---"
        text = r["text"]
        if len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


@mcp.tool()
def reindex(project: str | None = None) -> str:
    """Trigger incremental reindexing of code and docs.

    Args:
        project: Reindex a specific project (default: all configured projects).
    """
    try:
        run_index(project_name=project)
        status = get_status(project_name=project)
        return (
            f"Reindex complete. {status['total_code_chunks']} code, "
            f"{status['total_docs_chunks']} doc chunks."
        )
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
        lines.append(
            f"[{name}] code: {info['code_chunks']}, docs: {info['docs_chunks']} chunks"
        )
    lines.append(
        f"Total: {status['total_code_chunks']} code + {status['total_docs_chunks']} docs"
    )
    lines.append(f"Tracked files: {status.get('manifest_entries', 0)}")
    return "\n".join(lines)


@mcp.tool()
def prose_drift(project: str | None = None) -> dict:
    """Report contradictions between indexed docs and current chat-session facts.

    On-demand recrawl. With project=None, scans every project where
    prose_drift_enabled is true and returns a dict keyed by project name
    ({} if none enabled). With a named project, returns that project's payload.

    Two detection stages: (1) exact — same (subject, predicate) chain, differing
    object; (2) semantic — on a chain_key miss, the most similar current fact above a
    threshold is escalated to one contradiction-judge, catching cross-predicate /
    paraphrase drift. Each drift entry carries match_type ("exact" | "semantic"); a
    semantic entry adds similarity + confidence and a chat block with the chat-side
    subject/predicate/object/session_id. The payload also includes facts_scanned,
    facts_scanned_docs, stage2_judge_calls, and stage2_judge_errors. Still out of
    scope: omission and soft/temporal "used to have" contradictions.

    Args:
        project: Project name, or None to scan all enabled projects.
    """
    from vecs.prose_drift import _preflight_global, _preflight_project

    config = load_config()

    g = _preflight_global(config)
    if not g.ok:
        return {"error": g.code} if g.detail is None else {"error": g.code, "detail": g.detail}

    from vecs.prose_drift import find_prose_drift

    if project is None:
        out: dict = {}
        for name, proj in config.projects.items():
            if proj.prose_drift_enabled:
                out[name] = find_prose_drift(proj)
        return out

    p = _preflight_project(config, project)
    if not p.ok:
        return {"error": p.code, "detail": p.detail}
    return find_prose_drift(config.projects[project])


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
    from vecs.config import VECS_DIR
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
