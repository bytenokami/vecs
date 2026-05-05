"""Tests for Codex cwd routing and orphan persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vecs.codex_routing import (
    CodexRoutingState,
    discover_codex_sessions,
    route_cwd,
)
from vecs.config import CodeDir, ProjectConfig, VecsConfig


def _meta_line(cwd: str, sid: str = "s1") -> str:
    return json.dumps({
        "type": "session_meta",
        "timestamp": "t",
        "payload": {"cwd": cwd, "id": sid},
    })


def _make_session(path: Path, cwd: str, sid: str = "s1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_meta_line(cwd, sid) + "\n")
    return path


@pytest.fixture
def cfg(tmp_path):
    """A two-project config: alpha at tmp/repos/alpha, beta at tmp/repos/beta."""
    cfg_path = tmp_path / "config.yaml"
    config = VecsConfig(path=cfg_path)
    config.codex_sessions_root = tmp_path / "codex_sessions"
    alpha_root = tmp_path / "repos" / "alpha"
    beta_root = tmp_path / "repos" / "beta"
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    config.projects["alpha"] = ProjectConfig(
        name="alpha",
        code_dirs=[CodeDir(path=alpha_root, extensions={".py"})],
    )
    config.projects["beta"] = ProjectConfig(
        name="beta",
        code_dirs=[CodeDir(path=beta_root, extensions={".py"})],
    )
    return config


def test_route_cwd_inside_code_dir(cfg, tmp_path):
    name = route_cwd(str(tmp_path / "repos" / "alpha" / "src" / "lib"), cfg)
    assert name == "alpha"


def test_route_cwd_equal_to_code_dir(cfg, tmp_path):
    name = route_cwd(str(tmp_path / "repos" / "alpha"), cfg)
    assert name == "alpha"


def test_route_cwd_ancestor_of_single_code_dir_accepted(cfg, tmp_path):
    """cwd is parent of exactly one project's code_dirs -> accept."""
    # Add a project whose code_dir is deeper than `tmp_path/repos/alpha/sub`.
    cfg.projects.clear()
    deep = tmp_path / "repos" / "alpha" / "sub"
    deep.mkdir(parents=True)
    cfg.projects["alpha"] = ProjectConfig(
        name="alpha",
        code_dirs=[CodeDir(path=deep, extensions={".py"})],
    )
    name = route_cwd(str(tmp_path / "repos" / "alpha"), cfg)
    assert name == "alpha"


def test_route_cwd_ancestor_of_multiple_projects_rejected(cfg, tmp_path):
    """Ambiguous: cwd is ancestor of code_dirs in 2+ projects -> orphan."""
    name = route_cwd(str(tmp_path / "repos"), cfg)
    assert name is None


def test_route_cwd_no_match_orphan(cfg, tmp_path):
    name = route_cwd(str(tmp_path / "elsewhere"), cfg)
    assert name is None


def test_explicit_codex_cwds_overrides_bidirectional(cfg, tmp_path):
    """A path that would otherwise route to alpha lands on beta when explicitly set."""
    weird_cwd = tmp_path / "scratch" / "something"
    weird_cwd.mkdir(parents=True)
    cfg.projects["beta"].codex_cwds = [weird_cwd]
    assert route_cwd(str(weird_cwd / "deeper"), cfg) == "beta"


def test_codex_ignore_cwds_returns_none(cfg, tmp_path):
    cfg.codex_ignore_cwds = [tmp_path / "repos" / "alpha"]
    assert route_cwd(str(tmp_path / "repos" / "alpha" / "src"), cfg) is None


def test_route_cwd_empty_string_orphan(cfg):
    assert route_cwd("", cfg) is None


def test_state_load_missing_file_returns_empty(tmp_path):
    state = CodexRoutingState.load(path=tmp_path / "missing.json")
    assert state.orphans == {}
    assert state.cwd_cache == {}


def test_state_save_and_reload_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    state = CodexRoutingState(path=p)
    state.orphans = {"/x": {"sessions": 3, "first_seen": "a", "last_seen": "b"}}
    state.cwd_cache = {"/file": {"mtime": 1.0, "cwd": "/x", "session_id": "s1"}}
    state.last_banner_day = "2026-05-05"
    state.save()
    reloaded = CodexRoutingState.load(path=p)
    assert reloaded.orphans == state.orphans
    assert reloaded.cwd_cache == state.cwd_cache
    assert reloaded.last_banner_day == "2026-05-05"


def test_get_or_load_meta_uses_mtime_cache(tmp_path):
    sess_file = _make_session(tmp_path / "rollout-x.jsonl", cwd="/cwd1")
    state = CodexRoutingState(path=tmp_path / "state.json")
    meta1 = state.get_or_load_meta(sess_file)
    assert meta1 == {"cwd": "/cwd1", "id": "s1"}
    # Tamper with file content but keep mtime: should still serve cached cwd.
    cached_mtime = state.cwd_cache[str(sess_file)]["mtime"]
    state.cwd_cache[str(sess_file)] = {
        "mtime": cached_mtime,
        "cwd": "/CACHED",
        "session_id": "cached_id",
    }
    meta2 = state.get_or_load_meta(sess_file)
    assert meta2["cwd"] == "/CACHED"


def test_get_or_load_meta_invalidates_on_mtime_change(tmp_path):
    sess_file = _make_session(tmp_path / "rollout-y.jsonl", cwd="/cwd1")
    state = CodexRoutingState(path=tmp_path / "state.json")
    state.get_or_load_meta(sess_file)
    # Rewrite file with different content + bump mtime far ahead.
    sess_file.write_text(_meta_line("/cwd2", "s2") + "\n")
    import os
    os.utime(sess_file, (sess_file.stat().st_atime, sess_file.stat().st_mtime + 100))
    meta = state.get_or_load_meta(sess_file)
    assert meta["cwd"] == "/cwd2"


def test_get_or_load_meta_handles_huge_first_line(tmp_path):
    """Real Codex session_meta lines can exceed 14KB; read-line, not fixed-size.

    Regression: an earlier implementation read 8KB then decoded, which
    truncated session_meta mid-JSON and parsing failed silently for every
    file -- routing returned 0 matches.
    """
    big_blob = "X" * 20000
    huge_meta = json.dumps({
        "type": "session_meta",
        "timestamp": "t",
        "payload": {"cwd": "/x", "id": "s", "filler": big_blob},
    })
    sess_file = tmp_path / "rollout-big.jsonl"
    sess_file.write_text(huge_meta + "\n")
    state = CodexRoutingState(path=tmp_path / "state.json")
    meta = state.get_or_load_meta(sess_file)
    assert meta is not None
    assert meta["cwd"] == "/x"
    assert meta["id"] == "s"


def test_prune_dead_cache_removes_missing_files(tmp_path):
    state = CodexRoutingState(path=tmp_path / "state.json")
    state.cwd_cache = {
        str(tmp_path / "exists.jsonl"): {"mtime": 1.0, "cwd": "/x", "session_id": "s"},
        "/missing/file.jsonl": {"mtime": 1.0, "cwd": "/y", "session_id": "s2"},
    }
    (tmp_path / "exists.jsonl").write_text("")
    pruned = state.prune_dead_cache()
    assert pruned == 1
    assert "/missing/file.jsonl" not in state.cwd_cache


def test_discover_codex_sessions_routes_files(cfg, tmp_path):
    root = cfg.codex_sessions_root
    _make_session(root / "2026" / "04" / "01" / "rollout-a.jsonl",
                  cwd=str(tmp_path / "repos" / "alpha"))
    _make_session(root / "2026" / "04" / "02" / "rollout-b.jsonl",
                  cwd=str(tmp_path / "repos" / "beta" / "sub"))
    # Orphan
    _make_session(root / "2026" / "04" / "03" / "rollout-c.jsonl",
                  cwd=str(tmp_path / "elsewhere"))
    routing, state = discover_codex_sessions(cfg)
    assert set(routing.keys()) == {"alpha", "beta"}
    assert len(routing["alpha"]) == 1
    assert len(routing["beta"]) == 1
    assert state.total_orphan_sessions() == 1


def test_discover_codex_sessions_disabled_returns_empty(cfg, tmp_path):
    root = cfg.codex_sessions_root
    _make_session(root / "rollout-a.jsonl", cwd=str(tmp_path / "repos" / "alpha"))
    cfg.codex_disabled = True
    routing, state = discover_codex_sessions(cfg)
    assert routing == {}


def test_discover_codex_sessions_root_missing_is_silent(cfg):
    # codex_sessions_root never created -> silent no-op.
    routing, state = discover_codex_sessions(cfg)
    assert routing == {}
    assert state.orphans == {}


def test_discover_records_orphans_with_counts(cfg, tmp_path):
    root = cfg.codex_sessions_root
    orphan_cwd = str(tmp_path / "elsewhere")
    _make_session(root / "rollout-a.jsonl", cwd=orphan_cwd)
    _make_session(root / "rollout-b.jsonl", cwd=orphan_cwd)
    routing, state = discover_codex_sessions(cfg)
    assert state.orphans[orphan_cwd]["sessions"] == 2
