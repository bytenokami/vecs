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
    """Index code and docs (incremental)."""
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
    type=click.Choice(["code", "docs"], case_sensitive=False),
    default=None,
    help="Search a specific collection (default: all).",
)
@click.option("--limit", "-n", default=5, help="Number of results.")
@click.option("--path-filter", "-f", default=None, help="Filter to paths containing this substring.")
@click.option("--project", "-p", default=None, help="Search a specific project.")
def search_cmd(query: str, collection: str | None, limit: int, path_filter: str | None, project: str | None):
    """Search code and docs semantically."""
    from vecs.searcher import search
    results = search(query, collection_name=collection, n_results=limit, path_filter=path_filter, project=project)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or "?"
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
        click.echo(f"    Doc chunks:     {info.get('docs_chunks', 0)}")
    click.echo(f"\nTotal code chunks:    {s['total_code_chunks']}")
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
@click.option("--docs-dir", default=None, type=click.Path(), help="Documentation directory.")
def project_add(name: str, code_dir: tuple[str, ...], docs_dir: str | None):
    """Register a project for indexing.

    Code dirs use format: path:ext1,ext2
    Example: vecs project add livly --code-dir ~/repos/client:.ts,.tsx --docs-dir ~/repos/client/docs
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

    config.add_project(
        name=name,
        code_dirs=code_dirs,
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
        if p.docs_dir:
            click.echo(f"    docs: {p.docs_dir}")


# Alias so `vecs search` works
main.add_command(search_cmd, "search")
