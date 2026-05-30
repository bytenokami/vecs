"""Phase 5 wire-in tests: config flag, CLI, MCP, indexer facet."""
from __future__ import annotations

from pathlib import Path

import pytest

from vecs.config import ProjectConfig, load_config


def test_project_config_defaults_prose_drift_disabled():
    p = ProjectConfig(name="x")
    assert p.prose_drift_enabled is False


def test_prose_facts_collection_name():
    p = ProjectConfig(name="vecs")
    assert p.prose_facts_collection == "vecs-prose-facts"


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body)
    return cfg


def test_config_yaml_missing_field_loads_false(tmp_path):
    cfg = _write_config(tmp_path, """
projects:
  vecs:
    docs_dir: /tmp/docs
""")
    config = load_config(cfg)
    assert config.projects["vecs"].prose_drift_enabled is False


def test_config_yaml_prose_drift_enabled_true_loads(tmp_path):
    cfg = _write_config(tmp_path, """
projects:
  vecs:
    docs_dir: /tmp/docs
    prose_drift_enabled: true
""")
    config = load_config(cfg)
    assert config.projects["vecs"].prose_drift_enabled is True


# ---------------------------------------------------------------------------
# Task 8: CLI subcommand `vecs prose-drift`
# ---------------------------------------------------------------------------

from click.testing import CliRunner

from vecs.cli import main


def _mock_report(monkeypatch, drift, facts_scanned=1, facts_docs=1, project="vecs"):
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "find_prose_drift", lambda proj: {
        "drift": drift, "facts_scanned": facts_scanned,
        "facts_scanned_docs": facts_docs, "project": project,
    })


def _enable_project(monkeypatch, name="vecs"):
    from vecs.config import ProjectConfig, VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"),
                     projects={name: ProjectConfig(name=name, prose_drift_enabled=True)})
    monkeypatch.setattr("vecs.cli.load_config", lambda *a, **k: cfg, raising=False)
    import vecs.config
    monkeypatch.setattr(vecs.config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def test_cli_no_drift_exit_0(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[], facts_scanned=3)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 0
    assert "no prose drift" in res.output


def test_cli_no_chat_sessions_exit_0(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[], facts_scanned=0)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 0
    assert "no chat sessions" in res.output


def test_cli_with_drift_exit_1(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[{
        "subject": "team", "predicate": "has_role",
        "doc": {"object": "no backend developer", "source": "team.md"},
        "chat": {"object": "sasha", "session_id": "be_dev_announce"},
        "chat_history_versions": 1,
    }])
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 1
    assert 'team | has_role | doc="no backend developer" @ vecs/team.md' in res.output
    assert 'chat="sasha" @ session=be_dev_announce (chat_history_versions=1)' in res.output


def test_cli_unknown_project_exit_2(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)  # only 'vecs' exists
    res = CliRunner().invoke(main, ["prose-drift", "-p", "ghost"])
    assert res.exit_code == 2
    assert "unknown project: ghost" in res.output


def test_cli_disabled_project_exit_2(monkeypatch, fake_anthropic):
    from vecs.config import ProjectConfig, VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"),
                     projects={"vecs": ProjectConfig(name="vecs")})  # disabled
    import vecs.config
    monkeypatch.setattr(vecs.config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 2
    assert "prose drift not enabled for project vecs" in res.output


def test_cli_key_missing_exit_3(monkeypatch):
    _enable_project(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 3
    assert "ANTHROPIC_API_KEY not set" in res.output


def test_cli_limit_truncates_exit_1(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    many = [{
        "subject": f"s{i:03d}", "predicate": "p",
        "doc": {"object": "d", "source": "x.md"},
        "chat": {"object": "c", "session_id": "sess"},
        "chat_history_versions": 1,
    } for i in range(100)]
    _mock_report(monkeypatch, drift=many)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs", "--limit", "10"])
    assert res.exit_code == 1
    printed = [ln for ln in res.output.splitlines() if " | " in ln]
    assert len(printed) == 10
    assert "drift truncated: showing 10 of 100" in res.output
    # sort order preserved: first printed is s000
    assert printed[0].startswith("s000 | p |")


def test_cli_anthropic_import_missing_exit_3(monkeypatch):
    """Acceptance item 9 / design line 457: anthropic not importable -> exit 3.

    Force _preflight_global to yield ("anthropic_unavailable", ...) by patching
    _anthropic_importable, with the API key present so the failure is purely the
    missing-module path.
    """
    _enable_project(monkeypatch)  # sets ANTHROPIC_API_KEY
    import vecs.prose_drift as pd
    monkeypatch.setattr(
        pd, "_anthropic_importable",
        lambda: (False, "no module named anthropic"),
    )
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 3
    assert "anthropic not installed: pip install anthropic" in res.output


def test_cli_prose_drift_rerun_deterministic(monkeypatch, fake_anthropic):
    """Acceptance item 34 / line 62: re-running prose-drift with no new sessions
    or doc changes returns the same drift list. With find_prose_drift mocked to a
    fixed 2-entry list, two invocations produce byte-identical stdout (exit 1 both).
    """
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[
        {
            "subject": "team", "predicate": "has_role",
            "doc": {"object": "no backend developer", "source": "team.md"},
            "chat": {"object": "sasha", "session_id": "be_dev_announce"},
            "chat_history_versions": 1,
        },
        {
            "subject": "stack", "predicate": "database",
            "doc": {"object": "mysql", "source": "stack.md"},
            "chat": {"object": "postgres", "session_id": "infra_chat"},
            "chat_history_versions": 2,
        },
    ])
    first = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    second = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert first.exit_code == 1 and second.exit_code == 1
    assert first.output == second.output


def test_cli_as_of_flag_rejected(monkeypatch, fake_anthropic):
    """Acceptance item 41 / design line 141: no --as-of surface in v1.

    Click rejects the unknown option with a usage error -> exit 2. (Distinct from
    the project-preflight exit-2 path; here Click never reaches the command body.)
    """
    _enable_project(monkeypatch)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs", "--as-of", "2026-01-01"])
    assert res.exit_code == 2


# ---------------------------------------------------------------------------
# Task 9: MCP tool `prose_drift`
# ---------------------------------------------------------------------------


def _cfg_with(monkeypatch, projects):
    from vecs.config import VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"), projects=projects)
    import vecs.mcp_server
    monkeypatch.setattr(vecs.mcp_server, "load_config", lambda *a, **k: cfg)
    return cfg


def _enabled(name):
    from vecs.config import ProjectConfig
    return ProjectConfig(name=name, prose_drift_enabled=True)


def _disabled(name):
    from vecs.config import ProjectConfig
    return ProjectConfig(name=name)


def _patch_find(monkeypatch):
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "find_prose_drift", lambda proj: {
        "drift": [], "facts_scanned": 1, "facts_scanned_docs": 1, "project": proj.name,
    })


def test_mcp_named_project_returns_payload(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"vecs": _enabled("vecs")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    out = prose_drift(project="vecs")
    assert out == {"drift": [], "facts_scanned": 1, "facts_scanned_docs": 1, "project": "vecs"}


def test_mcp_none_scans_only_enabled(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"a": _enabled("a"), "b": _enabled("b"), "c": _disabled("c")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    out = prose_drift(project=None)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"]["project"] == "a"


def test_mcp_none_no_enabled_returns_empty_dict(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"c": _disabled("c")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    assert prose_drift(project=None) == {}


def test_mcp_none_global_preflight_failure_is_error_dict(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _cfg_with(monkeypatch, {"a": _enabled("a")})
    from vecs.mcp_server import prose_drift
    assert prose_drift(project=None) == {"error": "anthropic_key_missing"}


def test_mcp_named_disabled_is_error_dict(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"vecs": _disabled("vecs")})
    from vecs.mcp_server import prose_drift
    assert prose_drift(project="vecs") == {"error": "prose_drift_disabled", "detail": "vecs"}


# ---------------------------------------------------------------------------
# Task 10: Indexer wire-in (prose-drift facet block)
# ---------------------------------------------------------------------------

import vecs.indexer as indexer


class _FakeColl:
    def __init__(self):
        self.added = []
    def get_or_create_collection(self, *a, **k):
        return self
    def add(self, **k):
        self.added.append(k)
    def get(self, **k):
        return {"ids": [], "metadatas": []}


def _prep_indexer(monkeypatch, tmp_path, enabled):
    # Pin the session pipeline's external deps to no-ops.
    monkeypatch.setattr(indexer, "_get_session_new_content",
                        lambda f, m: ("raw-content", 10, True))
    monkeypatch.setattr(indexer, "chunk_session",
                        lambda msgs, sid, n, overlap: [
                            {"text": "[user]: x", "metadata": {"chunk_index": 0}}])
    monkeypatch.setattr(indexer, "_make_chunk_id", lambda a, b: f"{a}#{b}")
    monkeypatch.setattr(indexer, "_embed_and_store", lambda chunks, c, m, vo: {ch["id"] for ch in chunks})
    monkeypatch.setattr(indexer, "_track_embed_success",
                        lambda ids, c2f, fec, fc, coll: set(c2f.values()))
    monkeypatch.setattr(indexer, "_sync_bm25", lambda *a, **k: None)
    logs = []
    monkeypatch.setattr(indexer, "_log", lambda m: logs.append(m))

    class _Manifest:
        def __init__(self, *a): pass
        def mark_session_indexed(self, *a, **k): pass
        def save(self): pass
        def get_session_info(self, f): return None
    monkeypatch.setattr(indexer, "Manifest", _Manifest)

    sm_calls = []
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "extract_facts", lambda msgs, project: [pd.Triple("team", "has_role", "x")])
    monkeypatch.setattr(pd, "add_fact_with_state_machine",
                        lambda t, source_id, project: sm_calls.append((t, source_id, project)) or "INSERT")

    proj = ProjectConfig(name="vecs", prose_drift_enabled=enabled)
    return proj, sm_calls, logs


def test_state_machine_runs_when_enabled(monkeypatch, tmp_path):
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    f = tmp_path / "be_dev_announce.jsonl"
    f.write_text("{}")
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "we hired Sasha", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert len(sm_calls) == 1
    assert sm_calls[0][1] == "be_dev_announce"  # source_id == file stem
    assert sm_calls[0][2] == "vecs"


def test_state_machine_skipped_when_disabled(monkeypatch, tmp_path):
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=False)
    f = tmp_path / "s.jsonl"; f.write_text("{}")
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "x", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert sm_calls == []


def test_extract_failure_does_not_abort_indexer(monkeypatch, tmp_path):
    proj, sm_calls, logs = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "extract_facts",
                        lambda msgs, project: (_ for _ in ()).throw(RuntimeError("boom")))
    f = tmp_path / "s.jsonl"; f.write_text("{}")
    # Must NOT raise.
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "x", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert any("prose extract failed" in m for m in logs)


def test_state_machine_only_for_fully_succeeded_files(monkeypatch, tmp_path):
    """Acceptance item 21: only files in fully_succeeded get state-machine calls."""
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    fa = tmp_path / "a.jsonl"; fa.write_text("{}")
    fb = tmp_path / "b.jsonl"; fb.write_text("{}")
    # File A fully succeeds; file B is withheld from fully_succeeded (partial fail).
    monkeypatch.setattr(indexer, "_track_embed_success", lambda *a, **k: {fa})
    indexer._index_session_files(
        proj, [fa, fb], lambda content: [{"role": "user", "text": "claim", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    source_ids = [c[1] for c in sm_calls]
    assert source_ids == ["a"], "state machine must run only for fully-succeeded file A"
    assert "b" not in source_ids


def test_state_machine_skips_partial_file_runs_next_time(monkeypatch, tmp_path):
    """Acceptance item 21: a partial-fail file is faceted exactly once, on the
    later run where its chunks fully succeed — no duplicate INSERTs across runs."""
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    fa = tmp_path / "a.jsonl"; fa.write_text("{}")
    fb = tmp_path / "b.jsonl"; fb.write_text("{}")
    parser = lambda content: [{"role": "user", "text": "claim", "timestamp": "0"}]

    # Run 1: only A succeeds; B partial.
    monkeypatch.setattr(indexer, "_track_embed_success", lambda *a, **k: {fa})
    indexer._index_session_files(
        proj, [fa, fb], parser, "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    # Run 2: B now fully succeeds.
    monkeypatch.setattr(indexer, "_track_embed_success", lambda *a, **k: {fb})
    indexer._index_session_files(
        proj, [fa, fb], parser, "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )

    source_ids = [c[1] for c in sm_calls]
    assert source_ids.count("b") == 1, "B must be faceted exactly once across the two runs"


def test_mcp_anthropic_missing_returns_error_dict(monkeypatch):
    """Acceptance item 9 (MCP half): anthropic unavailable → error dict, no raise."""
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "_anthropic_importable", lambda: (False, "no module named anthropic"))
    _cfg_with(monkeypatch, {"vecs": _enabled("vecs")})
    from vecs.mcp_server import prose_drift
    out = prose_drift(project="vecs")
    assert out == {"error": "anthropic_unavailable", "detail": "no module named anthropic"}


# ---------------------------------------------------------------------------
# Code-path isolation guards (acceptance items 24, 33)
# ---------------------------------------------------------------------------

import inspect


def test_index_docs_does_not_write_facts():
    """Acceptance item 24: the doc-indexing path never invokes the state machine."""
    src = inspect.getsource(indexer.index_docs)
    for token in ("add_fact_with_state_machine", "extract_facts", "prose_drift", "prose-facts"):
        assert token not in src, f"index_docs must not reference {token!r}"


def test_code_vecs_path_untouched():
    """Acceptance item 33: the code-vecs path imports neither anthropic nor prose_drift.

    The legitimate lazy `from vecs.prose_drift import ...` lives inside
    `_index_session_files` (sessions path), NOT index_code — so this scopes to the
    index_code FUNCTION body plus the four code-path modules, never the whole
    indexer module.
    """
    import vecs.ast_chunker
    import vecs.bm25_index
    import vecs.chunkers
    import vecs.searcher

    code_fn_src = inspect.getsource(indexer.index_code)
    assert "anthropic" not in code_fn_src
    assert "prose_drift" not in code_fn_src

    for mod in (vecs.chunkers, vecs.ast_chunker, vecs.bm25_index, vecs.searcher):
        msrc = inspect.getsource(mod)
        assert "import anthropic" not in msrc, f"{mod.__name__} must not import anthropic"
        assert "prose_drift" not in msrc, f"{mod.__name__} must not reference prose_drift"
