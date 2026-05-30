from pathlib import Path

import click


@click.group()
def main():
    """vecs — Semantic search for your codebase."""
    pass


@main.command()
@click.option("--project", "-p", default=None, help="Index a specific project (default: all).")
@click.option("--detect-project", default=None, type=click.Path(), help="Auto-detect project from directory path.")
def index(project: str | None, detect_project: str | None):
    """Index code, sessions, and docs (incremental)."""
    from vecs.indexer import run_index

    if detect_project:
        from vecs.config import load_config
        target = Path(detect_project).resolve()
        config = load_config()
        matched = config.find_project_by_path(target)
        if not matched:
            return  # silent exit — no project matches
        run_index(project_name=matched)
    else:
        run_index(project_name=project)


@main.command()
@click.argument("query")
@click.option(
    "--collection", "-c",
    type=click.Choice(["code", "sessions", "docs"], case_sensitive=False),
    default=None,
    help="Search a specific collection (default: all).",
)
@click.option("--limit", "-n", default=5, help="Number of results.")
@click.option("--path-filter", "-f", default=None, help="Filter to paths containing this substring.")
@click.option("--project", "-p", default=None, help="Search a specific project.")
def search_cmd(query: str, collection: str | None, limit: int, path_filter: str | None, project: str | None):
    """Search code and sessions semantically."""
    from vecs.searcher import search
    results = search(query, collection_name=collection, n_results=limit, path_filter=path_filter, project=project)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        proj = r.get("project", "?")
        click.echo(f"\n--- Result {i} [{proj}:{r.get('collection', '?')}] {source}{dist_str} ---")
        text = r["text"]
        if len(text) > 1000:
            text = text[:1000] + "\n... [truncated]"
        click.echo(text)


@main.command()
@click.option("--project", "-p", default=None, help="Status for a specific project.")
def status(project: str | None):
    """Show index status."""
    from vecs.indexer import get_status
    s = get_status(project_name=project)
    for name, info in s.get("projects", {}).items():
        click.echo(f"\n  [{name}]")
        click.echo(f"    Code chunks:    {info['code_chunks']}")
        click.echo(f"    Session chunks: {info['session_chunks']}")
        click.echo(f"    Doc chunks:     {info.get('docs_chunks', 0)}")
    click.echo(f"\nTotal code chunks:    {s['total_code_chunks']}")
    click.echo(f"Total session chunks: {s['total_session_chunks']}")
    click.echo(f"Total doc chunks:     {s.get('total_docs_chunks', 0)}")
    click.echo(f"Tracked files:        {s.get('manifest_entries', 0)}")


@main.command("prose-drift")
@click.option("--project", "-p", required=True, help="Project to scan for prose drift.")
@click.option("--limit", default=50, help="Max drift lines to print (default 50).")
def prose_drift_cmd(project: str, limit: int):
    """Recrawl indexed docs and report contradictions vs current chat facts.

    On-demand recrawl (not write-time): compares (subject, predicate) facts
    extracted from indexed docs against the current state extracted from chat
    sessions. v1 detects exact (subject, predicate) object-collisions only.
    """
    from vecs.config import load_config
    from vecs.prose_drift import _preflight_global, _preflight_project

    config = load_config()

    g = _preflight_global(config)
    if not g.ok:
        if g.code == "anthropic_unavailable":
            click.echo("anthropic not installed: pip install anthropic", err=True)
        else:
            click.echo("ANTHROPIC_API_KEY not set", err=True)
        raise SystemExit(3)

    p = _preflight_project(config, project)
    if not p.ok:
        if p.code == "project_unknown":
            click.echo(f"unknown project: {project}", err=True)
        else:
            click.echo(f"prose drift not enabled for project {project}", err=True)
        raise SystemExit(2)

    from vecs.prose_drift import find_prose_drift

    report = find_prose_drift(config.projects[project])

    if report["facts_scanned"] == 0:
        click.echo(f"no chat sessions for project {project}")
        raise SystemExit(0)

    drift = report["drift"]
    if not drift:
        click.echo("no prose drift")
        raise SystemExit(0)

    total = len(drift)
    shown = drift[:limit]
    for d in shown:
        if d.get("match_type") == "semantic":
            click.echo(
                f"{d['subject']} | {d['predicate']} | "
                f"doc=\"{d['doc']['object']}\" @ {project}/{d['doc']['source']} "
                f"≠ chat[{d['chat']['subject']}|{d['chat']['predicate']}]="
                f"\"{d['chat']['object']}\" "
                f"@ session={d['chat']['session_id']} "
                f"[semantic sim={d['similarity']:.2f} conf={d['confidence']:.2f}] "
                f"(chat_history_versions={d['chat_history_versions']})"
            )
        else:
            click.echo(
                f"{d['subject']} | {d['predicate']} | "
                f"doc=\"{d['doc']['object']}\" @ {project}/{d['doc']['source']} "
                f"≠ chat=\"{d['chat']['object']}\" @ session={d['chat']['session_id']} "
                f"(chat_history_versions={d['chat_history_versions']})"
            )
    if total > limit:
        click.echo(f"drift truncated: showing {limit} of {total}", err=True)
    judge_errors = report.get("stage2_judge_errors", 0)
    if judge_errors:
        judge_calls = report.get("stage2_judge_calls", 0)
        click.echo(
            f"stage-2: {judge_calls} judge call(s), {judge_errors} errored and "
            "were skipped (possible missed contradictions)",
            err=True,
        )
    click.echo(
        "note: exact (subject,predicate) collisions + stage-2 semantic "
        "similarity-judge are covered; omission and soft/temporal contradictions "
        "remain out of scope (see v2-roadmap).",
        err=True,
    )
    raise SystemExit(1)


@main.command()
@click.option("--project", "-p", required=True, help="Project to add the doc to.")
@click.option("--title", "-t", required=True, help="Document title (used as filename).")
def add(project: str, title: str):
    """Add a document from stdin. Pipe or paste content, then Ctrl+D."""
    import sys
    from vecs.config import load_config, VECS_DIR
    from vecs.indexer import index_single_doc
    from vecs.utils import slugify

    content = sys.stdin.read()
    if not content.strip():
        click.echo("No content received.")
        return

    config = load_config()
    if project not in config.projects:
        click.echo(f"Project '{project}' not found.")
        return

    proj = config.projects[project]
    if not proj.docs_dir:
        proj.docs_dir = VECS_DIR / "docs" / project
        config.save()

    proj.docs_dir.mkdir(parents=True, exist_ok=True)

    file_path = proj.docs_dir / f"{slugify(title)}.md"
    file_path.write_text(content)

    try:
        stored = index_single_doc(project, file_path)
        click.echo(f"Saved '{title}' to {file_path} and indexed ({stored} chunks).")
    except Exception as e:
        click.echo(f"Saved '{title}' to {file_path} but indexing failed: {e}")


@main.group()
def project():
    """Manage indexed projects."""
    pass


@project.command("add")
@click.argument("name")
@click.option("--code-dir", required=True, multiple=True, help="Code directory. Format: path:ext1,ext2")
@click.option("--sessions-dir", multiple=True, type=click.Path(exists=True), help="Claude Code sessions directory (repeatable).")
@click.option("--docs-dir", default=None, type=click.Path(), help="Documentation directory.")
def project_add(name: str, code_dir: tuple[str, ...], sessions_dir: tuple[str, ...], docs_dir: str | None):
    """Register a project for indexing.

    Code dirs use format: path:ext1,ext2
    Example: vecs project add livly --code-dir ~/repos/client:.ts,.tsx --sessions-dir ~/.claude/projects/livly
    """
    from vecs.config import load_config, CodeDir

    config = load_config()
    code_dirs = []
    for entry in code_dir:
        if ":" in entry:
            path_str, ext_str = entry.rsplit(":", 1)
            extensions = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in ext_str.split(",")}
        else:
            click.echo(f"Error: code-dir '{entry}' must include extensions. Format: path:ext1,ext2")
            click.echo(f"Example: --code-dir {entry}:.py,.ts")
            raise SystemExit(1)
        resolved = Path(path_str).expanduser().resolve()
        if not resolved.exists():
            click.echo(f"Warning: path does not exist: {resolved}")
        code_dirs.append(CodeDir(
            path=resolved,
            extensions=extensions,
        ))

    sessions_dirs = [Path(s).resolve() for s in sessions_dir] if sessions_dir else []

    config.add_project(
        name=name,
        code_dirs=code_dirs,
        sessions_dirs=sessions_dirs,
        docs_dir=Path(docs_dir).resolve() if docs_dir else None,
    )
    config.save()
    click.echo(f"Added project '{name}' with {len(code_dirs)} code dir(s)")


@project.command("remove")
@click.argument("name")
def project_remove(name: str):
    """Unregister a project."""
    from vecs.config import load_config
    config = load_config()
    if name not in config.projects:
        click.echo(f"Project '{name}' not found.")
        return
    config.remove_project(name)
    config.save()
    click.echo(f"Removed project '{name}'")


@project.command("list")
def project_list():
    """List registered projects."""
    from vecs.config import load_config
    config = load_config()
    if not config.projects:
        click.echo("No projects configured. Use 'vecs project add' to register one.")
        return
    for name, p in config.projects.items():
        click.echo(f"\n  {name}:")
        for cd in p.code_dirs:
            exts = ", ".join(sorted(cd.extensions))
            click.echo(f"    code: {cd.path} [{exts}]")
        for sd in p.sessions_dirs:
            click.echo(f"    sessions: {sd}")
        if p.docs_dir:
            click.echo(f"    docs: {p.docs_dir}")


# Alias so `vecs search` works
main.add_command(search_cmd, "search")


@main.group()
def codex():
    """Manage Codex CLI session indexing (orphan triage)."""
    pass


@codex.command("orphans")
def codex_orphans_cmd():
    """List Codex sessions whose cwd matches no project."""
    from vecs.codex_routing import CodexRoutingState
    from vecs.config import load_config

    state = CodexRoutingState.load()
    if not state.orphans:
        click.echo("No Codex orphan cwds. Routing is clean.")
        return

    config = load_config()
    project_paths = {
        name: [cd.path for cd in proj.code_dirs]
        for name, proj in config.projects.items()
    }

    click.echo(
        f"{state.total_orphan_sessions()} Codex sessions across "
        f"{len(state.orphans)} unmapped cwd(s):\n"
    )
    for cwd, info in sorted(state.orphans.items()):
        suggestion = _suggest_project_for_cwd_cli(cwd, project_paths)
        sessions = info.get("sessions", 0)
        last = info.get("last_seen", "?")
        click.echo(f"  {cwd}  ({sessions} sessions, last_seen={last})")
        if suggestion:
            click.echo(f"    suggest: vecs codex assign {cwd} -p {suggestion}")
        else:
            click.echo("    suggest: no obvious project match — pick manually")
        click.echo(f"    ignore:  vecs codex ignore {cwd}")
        click.echo("")


@codex.command("assign")
@click.argument("cwd")
@click.option("--project", "-p", required=True, help="Project to route this cwd to.")
def codex_assign_cmd(cwd: str, project: str):
    """Route Codex sessions for CWD to PROJECT."""
    from vecs.config import load_config
    from vecs.codex_routing import CodexRoutingState
    from vecs.indexer import purge_session_files_from_project

    config = load_config()
    if project not in config.projects:
        click.echo(f"Project '{project}' not found.")
        raise SystemExit(1)

    cwd_path = Path(cwd).expanduser()
    proj = config.projects[project]
    if cwd_path not in proj.codex_cwds:
        proj.codex_cwds.append(cwd_path)
        config.save()

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

    for name in config.projects:
        result = purge_session_files_from_project(
            project_name=name,
            file_paths=affected_files,
            session_ids=affected_session_ids,
        )
        if result["manifest_entries_dropped"] or result["chunks_deleted"]:
            click.echo(
                f"  [{name}] dropped {result['manifest_entries_dropped']} manifest, "
                f"swept {result['chunks_deleted']} chunks"
            )

    state.save()
    click.echo(
        f"Routed cwd={cwd} -> project={project} "
        f"({len(affected_files)} session file(s) affected). "
        f"Run `vecs index` to re-emit."
    )


@codex.command("ignore")
@click.argument("cwd")
def codex_ignore_cmd(cwd: str):
    """Stop indexing Codex sessions for CWD."""
    from vecs.config import load_config
    from vecs.codex_routing import CodexRoutingState
    from vecs.indexer import purge_session_files_from_project

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

    state.orphans.pop(cwd, None)
    state.save()
    click.echo(
        f"Ignoring cwd={cwd}. "
        f"Dropped {manifest_dropped_total} manifest entries, swept {deleted_total} chunks."
    )


def _suggest_project_for_cwd_cli(cwd: str, project_paths: dict[str, list[Path]]) -> str | None:
    """CLI-side helper: pick the best project match by leading path-segment overlap."""
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
