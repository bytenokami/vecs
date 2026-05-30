"""CLI rendering tests. find_prose_drift is monkeypatched to a canned report so
these exercise only the click output layer (no Chroma / no LLM / no ~/.vecs)."""
from __future__ import annotations

from click.testing import CliRunner

from vecs import cli as cli_mod
from vecs import prose_drift as pd
from vecs.prose_drift import PreflightResult


class _Cfg:
    projects = {"vecs": object()}


def _patch_preflight_and_config(monkeypatch, report):
    monkeypatch.setattr("vecs.config.load_config", lambda: _Cfg())
    monkeypatch.setattr(pd, "_preflight_global", lambda config: PreflightResult(True))
    monkeypatch.setattr(pd, "_preflight_project", lambda config, p: PreflightResult(True))
    monkeypatch.setattr(pd, "find_prose_drift", lambda proj: report)


def test_cli_prose_drift_renders_semantic_and_exact_lines(monkeypatch):
    report = {
        "facts_scanned": 2,
        "facts_scanned_docs": 1,
        "stage2_judge_calls": 1,
        "stage2_judge_errors": 0,
        "project": "vecs",
        "drift": [
            {
                "subject": "team", "predicate": "has_role", "match_type": "exact",
                "doc": {"object": "no BE dev", "source": "team.md"},
                "chat": {"subject": "team", "predicate": "has_role",
                         "object": "sasha is BE", "session_id": "s1"},
                "chat_history_versions": 1,
            },
            {
                "subject": "team", "predicate": "has_role", "match_type": "semantic",
                "similarity": 0.91, "confidence": 0.88,
                "doc": {"object": "no BE dev", "source": "team.md"},
                "chat": {"subject": "team", "predicate": "employs",
                         "object": "sasha", "session_id": "s2"},
                "chat_history_versions": 1,
            },
        ],
    }
    _patch_preflight_and_config(monkeypatch, report)

    result = CliRunner().invoke(cli_mod.main, ["prose-drift", "-p", "vecs"])
    assert result.exit_code == 1
    out = result.output
    # Semantic line carries a [semantic ...] tag with similarity + confidence,
    # and surfaces the chat-side predicate (which differs from the doc-side one).
    assert "[semantic" in out
    assert "sim=" in out and "conf=" in out
    assert "employs" in out
    # Exact line is unchanged in spirit: doc + chat objects on one line.
    assert "sasha is BE" in out
    # Trailing note: tightened to a unique substring of the actual note text.
    assert "remain out of scope" in out


def test_cli_semantic_line_shows_chat_subject(monkeypatch):
    """A cross-subject semantic pairing must render the chat subject, not hide it
    behind the doc subject."""
    report = {
        "facts_scanned": 1, "facts_scanned_docs": 1,
        "stage2_judge_calls": 1, "stage2_judge_errors": 0, "project": "vecs",
        "drift": [
            {
                "subject": "team", "predicate": "has_role", "match_type": "semantic",
                "similarity": 0.9, "confidence": 0.8,
                "doc": {"object": "no BE dev", "source": "team.md"},
                "chat": {"subject": "org", "predicate": "funds",
                         "object": "two BE hires", "session_id": "s2"},
                "chat_history_versions": 1,
            },
        ],
    }
    _patch_preflight_and_config(monkeypatch, report)
    result = CliRunner().invoke(cli_mod.main, ["prose-drift", "-p", "vecs"])
    assert result.exit_code == 1
    # chat-side subject AND predicate both visible (cross-subject is not hidden).
    assert "org" in result.output and "funds" in result.output


def test_cli_surfaces_judge_errors(monkeypatch):
    report = {
        "facts_scanned": 1, "facts_scanned_docs": 1,
        "stage2_judge_calls": 3, "stage2_judge_errors": 2, "project": "vecs",
        "drift": [
            {
                "subject": "team", "predicate": "has_role", "match_type": "exact",
                "doc": {"object": "no BE dev", "source": "team.md"},
                "chat": {"subject": "team", "predicate": "has_role",
                         "object": "sasha is BE", "session_id": "s1"},
                "chat_history_versions": 1,
            },
        ],
    }
    _patch_preflight_and_config(monkeypatch, report)
    result = CliRunner().invoke(cli_mod.main, ["prose-drift", "-p", "vecs"])
    # When judge calls errored, the operator is told (else a missed contradiction is invisible).
    assert "2" in result.output and "error" in result.output.lower()
