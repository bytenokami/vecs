"""Tests for the MCP tool surface (mcp_server.py).

Inc 1.5c part (a): semantic_search surfaces a per-hit freshness/trust signal
(the chunk version_id) in its result shape, so the calling agent can judge how
fresh/trustworthy each hit is. version_id is stamped into chunk metadata by the
indexer (C): git HEAD sha for code, file mtime for docs.
"""

from __future__ import annotations

from vecs import mcp_server


def _one_result(metadata: dict) -> list[dict]:
    return [
        {
            "id": "x",
            "text": "result body",
            "metadata": metadata,
            "distance": 0.1234,
            "collection": "vecs-code",
            "project": "vecs",
        }
    ]


def test_semantic_search_surfaces_version_id_short_sha(monkeypatch):
    """A code hit's git-sha version_id is surfaced (shortened for display)."""
    sha = "0123456789abcdef0123456789abcdef01234567"  # 40-hex git HEAD sha
    monkeypatch.setattr(
        mcp_server, "search",
        lambda *a, **k: _one_result({"file_path": "src/f.py", "version_id": sha}),
    )
    out = mcp_server.semantic_search("q")
    assert "src/f.py" in out
    assert "v:" + sha[:8] in out  # sha shortened to 8 chars for the header


def test_semantic_search_version_unknown_when_absent(monkeypatch):
    """A legacy chunk with no version_id surfaces an explicit 'unknown' bucket
    rather than silently omitting the trust signal."""
    monkeypatch.setattr(
        mcp_server, "search",
        lambda *a, **k: _one_result({"file_path": "src/f.py"}),
    )
    out = mcp_server.semantic_search("q")
    assert "v:unknown" in out


def test_semantic_search_version_id_mtime_kept_verbatim(monkeypatch):
    """A docs hit's mtime version_id is NOT a 40-hex sha, so it is surfaced
    verbatim (no sha-shortening that would mangle a timestamp)."""
    mtime = "1717459200.123456"
    monkeypatch.setattr(
        mcp_server, "search",
        lambda *a, **k: _one_result({"file_path": "docs/d.md", "version_id": mtime}),
    )
    out = mcp_server.semantic_search("q")
    assert "v:" + mtime in out


def test_semantic_search_no_results_unchanged(monkeypatch):
    """Empty result set still returns the no-results sentinel."""
    monkeypatch.setattr(mcp_server, "search", lambda *a, **k: [])
    assert mcp_server.semantic_search("q") == "No results found."
