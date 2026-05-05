from __future__ import annotations

import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vecs.codex_routing import CodexRoutingState
from vecs.config import load_config
from vecs.indexer import get_status, purge_session_files_from_project, run_index
from vecs.searcher import search
from vecs.utils import slugify

mcp = FastMCP("vecs")

# Banner thresholds: dedupe by calendar day, only fire when there's something
# meaningful to say.
_BANNER_MIN_ORPHAN_SESSIONS = 1


def _orphan_banner(state: CodexRoutingState | None, *, fire_once_per_day: bool) -> str:
    """Build the orphan-summary banner for MCP tool returns.

    Returns "" when there's nothing to surface.
    When `fire_once_per_day` is True, returns "" if a banner has already
    been emitted today (per `state.last_banner_day`) and persists the new
    day on first fire of a calendar day.
    """
    if state is None or not state.orphans:
        return ""
    total = state.total_orphan_sessions()
    if total < _BANNER_MIN_ORPHAN_SESSIONS:
        return ""
    today = time.strftime("%Y-%m-%d")
    if fire_once_per_day and state.last_banner_day == today:
        return ""
    if fire_once_per_day:
        state.last_banner_day = today
        try:
            state.save()
        except Exception:
            pass
    return (
        f"\n\n[vecs] {total} codex sessions skipped ({len(state.orphans)} unmapped cwds). "
        f"Triage: codex_orphans MCP tool."
    )


def _suggest_project_for_cwd(cwd: str, project_paths: dict[str, list[Path]]) -> str | None:
    """Pick the project whose code_dir paths share the most leading segments with cwd.

    Returns None if nothing remotely matches (suggestion is a hint only — the
    user still confirms via codex_assign).
    """
    if not cwd or not project_paths:
        return None
    cwd_parts = Path(cwd).parts
    best: tuple[int, str] | None = None
    for name, paths in project_paths.items():
        for p in paths:
            p_parts = p.parts
            common = 0
            for a, b in zip(cwd_parts, p_parts):
                if a != b:
                    break
                common += 1
            if common >= 3 and (best is None or common > best[0]):
                best = (common, name)
    return best[1] if best else None


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
        body = "No results found."
    else:
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
            collection_name = r.get("collection", "?") or ""
            # Session-collection chunks always carry an agent label. Pre-existing
            # chunks (indexed before the codex change) lack `metadata.agent`; we
            # default them to claude_code per spec backward-compat clause.
            if collection_name.endswith("sessions"):
                agent = meta.get("agent") or "claude_code"
                agent_tag = f" {{{agent}}}"
            else:
                agent_tag = ""
            dist = r.get("distance")
            dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
            proj = r.get("project", "?")
            header = f"--- Result {i} [{proj}:{collection_name}{agent_tag}] {source}{dist_str} ---"
            text = r["text"]
            if len(text) > 2000:
                text = text[:2000] + "\n... [truncated]"
            parts.append(f"{header}\n{text}")
        body = "\n\n".join(parts)

    # Banner only when results touched a sessions collection AND orphans exist.
    touched_sessions = any(
        (r.get("collection") or "").endswith("sessions") for r in results
    )
    if touched_sessions:
        try:
            state = CodexRoutingState.load()
            body += _orphan_banner(state, fire_once_per_day=True)
        except Exception:
            pass
    return body


@mcp.tool()
def reindex(project: str | None = None) -> str:
    """Trigger incremental reindexing of code and session files.

    Args:
        project: Reindex a specific project (default: all configured projects).
    """
    try:
        run_index(project_name=project)
        status = get_status(project_name=project)
        return (
            f"Reindex complete. {status['total_code_chunks']} code, "
            f"{status['total_session_chunks']} session, {status['total_docs_chunks']} doc chunks."
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
            f"[{name}] code: {info['code_chunks']}, sessions: {info['session_chunks']}, "
            f"docs: {info['docs_chunks']} chunks"
        )
    lines.append(
        f"Total: {status['total_code_chunks']} code + {status['total_session_chunks']} sessions "
        f"+ {status['total_docs_chunks']} docs"
    )
    lines.append(f"Tracked files: {status.get('manifest_entries', 0)}")
    body = "\n".join(lines)

    # index_status always surfaces orphans (no per-day dedupe — this is the
    # canonical place for the agent to learn about pending triage work).
    try:
        state = CodexRoutingState.load()
        if state.orphans:
            body += (
                f"\n\nCodex orphans: {state.total_orphan_sessions()} sessions across "
                f"{len(state.orphans)} unmapped cwd(s). Call `codex_orphans` to triage."
            )
    except Exception:
        pass
    return body


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


@mcp.tool()
def codex_orphans() -> str:
    """List Codex sessions whose `cwd` matched no project, with project suggestions.

    Output is plain text the calling agent can show the user. Each orphan cwd
    is annotated with a project suggestion based on path-segment overlap.
    """
    state = CodexRoutingState.load()
    if not state.orphans:
        return "No Codex orphan cwds. Routing is clean."

    config = load_config()
    project_paths = {
        name: [cd.path for cd in proj.code_dirs]
        for name, proj in config.projects.items()
    }

    lines = [
        f"{state.total_orphan_sessions()} Codex sessions across "
        f"{len(state.orphans)} unmapped cwd(s):",
        "",
    ]
    for cwd, info in sorted(state.orphans.items()):
        suggestion = _suggest_project_for_cwd(cwd, project_paths)
        sessions = info.get("sessions", 0)
        last = info.get("last_seen", "?")
        suggest_str = f"  -> suggest project: {suggestion}" if suggestion else "  -> no obvious match"
        lines.append(f"  {cwd}  ({sessions} sessions, last_seen={last})")
        lines.append(suggest_str)
        lines.append(f"  Assign: codex_assign(cwd=\"{cwd}\", project=\"{suggestion or '<name>'}\")")
        lines.append(f"  Ignore: codex_ignore(cwd=\"{cwd}\")")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def codex_assign(cwd: str, project: str) -> str:
    """Route Codex sessions for a given `cwd` to a specific project.

    Adds `cwd` to `projects[project].codex_cwds`, sweeps any chunks that were
    indexed under another project for those sessions, and drops manifest
    entries so the next `reindex` re-emits them under the new project.

    Args:
        cwd: The Codex session working directory to route.
        project: Project name to route to.
    """
    config = load_config()
    if project not in config.projects:
        return (
            f"Project '{project}' not found. Available: "
            + (", ".join(sorted(config.projects.keys())) or "(none)")
        )

    cwd_path = Path(cwd).expanduser()

    # Mutate config.
    proj = config.projects[project]
    if cwd_path not in proj.codex_cwds:
        proj.codex_cwds.append(cwd_path)
        config.save()

    # Find all currently-cached files for this cwd; drop them from any project
    # that previously owned them (could be a different project, or this same
    # one re-keyed by a code_dir path).
    state = CodexRoutingState.load()
    affected_files: list[Path] = []
    affected_session_ids: list[str] = []
    for path_str, info in state.cwd_cache.items():
        if info.get("cwd") == cwd:
            p = Path(path_str)
            if p.exists():
                affected_files.append(p)
                sid = info.get("session_id") or p.stem
                if sid:
                    affected_session_ids.append(sid)

    # Drop from EVERY project's manifest+collection — we don't know which
    # project previously owned these. Cheap: each call is O(N session_ids)
    # against a single chroma where-filter.
    summary = []
    for name in config.projects:
        result = purge_session_files_from_project(
            project_name=name,
            file_paths=affected_files,
            session_ids=affected_session_ids,
        )
        if result["manifest_entries_dropped"] or result["chunks_deleted"]:
            summary.append(
                f"  [{name}] dropped {result['manifest_entries_dropped']} manifest entries, "
                f"swept {result['chunks_deleted']} chunks across {result['session_ids_swept']} session_ids"
            )

    # Refresh state's ignore-bookkeeping irrelevant; just persist.
    state.save()

    msg = [
        f"Routed cwd={cwd} -> project={project}.",
        f"Affected {len(affected_files)} session file(s).",
    ]
    msg.extend(summary)
    msg.append("Run `reindex` to re-emit these sessions under the new project.")
    return "\n".join(msg)


@mcp.tool()
def codex_ignore(cwd: str) -> str:
    """Skip future Codex indexing for sessions captured at `cwd`.

    Adds `cwd` to top-level `codex_ignore_cwds` and removes any previously-
    indexed chunks for those sessions from every project. Subsequent `reindex`
    runs silently skip these sessions.

    Args:
        cwd: The Codex session working directory to ignore.
    """
    config = load_config()
    cwd_path = Path(cwd).expanduser()
    if cwd_path not in config.codex_ignore_cwds:
        config.codex_ignore_cwds.append(cwd_path)
        config.save()

    state = CodexRoutingState.load()
    affected_files: list[Path] = []
    affected_session_ids: list[str] = []
    for path_str, info in state.cwd_cache.items():
        if info.get("cwd") == cwd:
            p = Path(path_str)
            affected_files.append(p)
            sid = info.get("session_id") or p.stem
            if sid:
                affected_session_ids.append(sid)

    deleted_total = 0
    manifest_dropped_total = 0
    for name in config.projects:
        result = purge_session_files_from_project(
            project_name=name,
            file_paths=affected_files,
            session_ids=affected_session_ids,
        )
        deleted_total += result["chunks_deleted"]
        manifest_dropped_total += result["manifest_entries_dropped"]

    # Remove this cwd from the orphan map immediately so banners stop firing.
    state.orphans.pop(cwd, None)
    state.save()

    return (
        f"Ignoring cwd={cwd}. "
        f"Dropped {manifest_dropped_total} manifest entries, swept {deleted_total} chunks. "
        f"Future indexing will skip these sessions silently."
    )
