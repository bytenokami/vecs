import click

from vecs.indexer import run_index
from vecs.searcher import search


@click.group()
def main():
    """vecs — Semantic search for Bloomly."""
    pass


@main.command()
def index():
    """Index code and session transcripts (incremental)."""
    run_index()


@main.command()
@click.argument("query")
@click.option(
    "--collection", "-c",
    type=click.Choice(["code", "sessions"], case_sensitive=False),
    default=None,
    help="Search a specific collection (default: both).",
)
@click.option("--limit", "-n", default=5, help="Number of results.")
def search_cmd(query: str, collection: str | None, limit: int):
    """Search code and sessions semantically."""
    results = search(query, collection_name=collection, n_results=limit)
    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("file_path") or f"session:{meta.get('session_id', '?')}"
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        click.echo(f"\n--- Result {i} [{r.get('collection', '?')}] {source}{dist_str} ---")
        # Truncate long results for display
        text = r["text"]
        if len(text) > 1000:
            text = text[:1000] + "\n... [truncated]"
        click.echo(text)


# Alias so `vecs search` works
main.add_command(search_cmd, "search")
