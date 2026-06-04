"""Tests for the Inc 1-E stale-retrieval-rate harness + eval-set scaffold.

The metric is defined against the per-chunk version_id anchor (Inc 1-C): a chunk
is STALE when its recorded version_id no longer matches the version_id its source
would be stamped with NOW; a chunk with no version_id (or whose current version
can't be computed) is UNKNOWN and excluded from the rate. The harness reuses the
indexer's exact anchors (git sha / content-hash for code, mtime for docs).
"""

from __future__ import annotations

import hashlib

import pytest

from vecs.config import CodeDir, ProjectConfig, VecsConfig
from vecs import eval_harness as eh


# --- pure bucketing core -----------------------------------------------------


def test_bucket_fresh_when_versions_match():
    assert eh.bucket_chunk("abc123", "abc123") == eh.FRESH


def test_bucket_stale_when_versions_differ():
    assert eh.bucket_chunk("old", "new") == eh.STALE


def test_bucket_unknown_when_stored_missing():
    """A chunk with no recorded version_id (pre-C / never re-stamped) is UNKNOWN,
    not guessed."""
    assert eh.bucket_chunk(None, "new") == eh.UNKNOWN
    assert eh.bucket_chunk("", "new") == eh.UNKNOWN


def test_bucket_unknown_when_current_uncomputable():
    """If the source's current version can't be computed (gone/error), UNKNOWN."""
    assert eh.bucket_chunk("abc", None) == eh.UNKNOWN


def test_bucket_compares_as_strings():
    assert eh.bucket_chunk(123, "123") == eh.FRESH


# --- StaleStats --------------------------------------------------------------


def test_stale_stats_rate_excludes_unknown():
    s = eh.StaleStats(fresh=3, stale=1, unknown=10)
    assert s.classified == 4
    assert s.total == 14
    assert s.rate == pytest.approx(0.25)  # 1 / (3+1); the 10 unknown are excluded


def test_stale_stats_rate_none_when_nothing_classified():
    s = eh.StaleStats(fresh=0, stale=0, unknown=5)
    assert s.rate is None


def test_stale_stats_iadd_merges():
    a = eh.StaleStats(fresh=1, stale=2, unknown=3)
    a += eh.StaleStats(fresh=10, stale=20, unknown=30)
    assert (a.fresh, a.stale, a.unknown) == (11, 22, 33)


# --- current-version anchors (reuse the indexer's exact rule) ----------------


def _proj_with_docs(tmp_path):
    d = tmp_path / "mydocs"
    d.mkdir()
    return ProjectConfig(name="p", docs_dirs=[d]), d


def test_current_version_docs_is_mtime(tmp_path):
    proj, d = _proj_with_docs(tmp_path)
    f = d / "note.md"
    f.write_text("hello")
    expected = str(f.stat().st_mtime)
    assert eh._current_version_id("mydocs/note.md", "docs", proj) == expected


def test_current_version_code_uses_git_sha(tmp_path, monkeypatch):
    cd = tmp_path / "repo"
    cd.mkdir()
    (cd / "a.py").write_text("x = 1")
    proj = ProjectConfig(name="p", code_dirs=[CodeDir(path=cd, extensions={".py"})])
    monkeypatch.setattr(eh, "_git_sha", lambda path: "deadbeefsha")
    assert eh._current_version_id("repo/a.py", "code", proj) == "deadbeefsha"


def test_current_version_code_falls_back_to_content_hash(tmp_path, monkeypatch):
    """Non-git tree: _git_sha is None, so the anchor is sha256 of the file BYTES
    (matching Manifest._file_hash, not a text hash)."""
    cd = tmp_path / "repo"
    cd.mkdir()
    body = b"x = 1\n"
    (cd / "a.py").write_bytes(body)
    proj = ProjectConfig(name="p", code_dirs=[CodeDir(path=cd, extensions={".py"})])
    monkeypatch.setattr(eh, "_git_sha", lambda path: None)
    expected = hashlib.sha256(body).hexdigest()
    assert eh._current_version_id("repo/a.py", "code", proj) == expected


def test_current_version_none_when_source_missing(tmp_path):
    proj, d = _proj_with_docs(tmp_path)
    assert eh._current_version_id("mydocs/gone.md", "docs", proj) is None


def test_current_version_none_when_root_unresolvable(tmp_path):
    proj, d = _proj_with_docs(tmp_path)
    # root segment "other" matches no configured root
    assert eh._current_version_id("other/note.md", "docs", proj) is None


# --- stale_stats_for_chunks + collection_stale_rate --------------------------


def test_stale_stats_for_chunks_mixes_buckets(monkeypatch):
    proj = ProjectConfig(name="p", docs_dirs=[])
    metas = [
        {"version_id": "v1", "file_path": "d/fresh.md"},
        {"version_id": "v1", "file_path": "d/stale.md"},
        {"file_path": "d/legacy.md"},  # no version_id -> unknown
    ]

    def fake_current(fp, kind, p):
        return {"d/fresh.md": "v1", "d/stale.md": "v2", "d/legacy.md": "v9"}.get(fp)

    monkeypatch.setattr(eh, "_current_version_id", fake_current)
    stats = eh.stale_stats_for_chunks(metas, "docs", proj)
    assert (stats.fresh, stats.stale, stats.unknown) == (1, 1, 1)
    assert stats.rate == pytest.approx(0.5)


def test_collection_stale_rate_skips_missing_collection():
    """An absent collection (or unconfigured project) yields empty stats, not an
    error -- the harness tolerates a live store without the collection yet."""
    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["vecs"] = ProjectConfig(name="vecs")

    class _DB:
        def get_collection(self, name):
            raise Exception("no such collection")

    stats = eh.collection_stale_rate("vecs-docs", config=cfg, db=_DB())
    assert stats.total == 0
    assert stats.rate is None


def test_collection_stale_rate_unknown_collection_kind():
    cfg = VecsConfig(path="/tmp/x.yaml")
    stats = eh.collection_stale_rate("vecs-prose-facts", config=cfg, db=object())
    assert stats.total == 0


def test_collection_stale_rate_buckets_real_metadatas(monkeypatch):
    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["vecs"] = ProjectConfig(name="vecs")

    class _Col:
        def get(self, include):
            return {"metadatas": [
                {"version_id": "v1", "file_path": "vecs/a.md"},
                {"version_id": "old", "file_path": "vecs/b.md"},
            ]}

    class _DB:
        def get_collection(self, name):
            return _Col()

    monkeypatch.setattr(
        eh, "_current_version_id",
        lambda fp, kind, p: "v1" if fp == "vecs/a.md" else "new",
    )
    stats = eh.collection_stale_rate("vecs-docs", config=cfg, db=_DB())
    assert (stats.fresh, stats.stale) == (1, 1)


# --- eval-set scaffold + runner stub -----------------------------------------


def test_default_eval_set_is_a_nonempty_scaffold():
    assert len(eh.DEFAULT_EVAL_SET) >= 1
    for case in eh.DEFAULT_EVAL_SET:
        assert case.query and case.project and case.collection in ("code", "docs")
        assert case.expected_path_substring


def test_run_eval_reports_hit_and_stale_metric(monkeypatch):
    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["vecs"] = ProjectConfig(name="vecs")
    cases = [eh.EvalCase("q", "vecs", "docs", "wanted")]

    def fake_search(query, collection_name, n_results, project):
        return [
            {"metadata": {"file_path": "vecs/wanted.md", "version_id": "v1"}},
            {"metadata": {"file_path": "vecs/other.md", "version_id": "old"}},
        ]

    monkeypatch.setattr(
        eh, "_current_version_id",
        lambda fp, kind, p: "v1" if fp == "vecs/wanted.md" else "new",
    )
    report = eh.run_eval(eval_set=cases, config=cfg, search_fn=fake_search)
    assert report.hit_rate == pytest.approx(1.0)  # "wanted" found in sources
    assert (report.retrieved_stale.fresh, report.retrieved_stale.stale) == (1, 1)


def test_run_eval_miss_when_expected_absent(monkeypatch):
    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["vecs"] = ProjectConfig(name="vecs")
    cases = [eh.EvalCase("q", "vecs", "docs", "missing-thing")]

    def fake_search(query, collection_name, n_results, project):
        return [{"metadata": {"file_path": "vecs/other.md", "version_id": "v1"}}]

    monkeypatch.setattr(eh, "_current_version_id", lambda fp, kind, p: "v1")
    report = eh.run_eval(eval_set=cases, config=cfg, search_fn=fake_search)
    assert report.hit_rate == pytest.approx(0.0)


def test_run_eval_tolerates_search_failure(monkeypatch):
    cfg = VecsConfig(path="/tmp/x.yaml")
    cfg.projects["vecs"] = ProjectConfig(name="vecs")
    cases = [eh.EvalCase("q", "vecs", "docs", "x")]

    def boom(*a, **k):
        raise RuntimeError("search exploded")

    report = eh.run_eval(eval_set=cases, config=cfg, search_fn=boom)
    assert report.hit_rate == pytest.approx(0.0)
    assert report.results[0].n_results == 0
