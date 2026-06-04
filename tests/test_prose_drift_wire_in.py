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

    Scopes to the index_code FUNCTION body plus the four code-path modules, never
    the whole indexer module. (The session indexer that once carried a lazy
    `from vecs.prose_drift import ...` was removed, so no such import remains.)
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
