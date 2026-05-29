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
