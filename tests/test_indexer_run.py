from vecs.embed_provider import VoyageProvider
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vecs.indexer import (
    Manifest,
    _embed_and_store,
    _delete_stale_chunks_after_embed,
    _index_collection,
    _make_batches,
    _paginated_get,
    _sync_bm25,
    _sweep_excluded_chunks,
    _track_embed_success,
    index_code,
    migrate_global_manifest,
)
from vecs.config import VecsConfig, ProjectConfig, CodeDir
from indexer_helpers import (  # noqa: F401
    FakeEmbedResult, _embedded_texts, _make_index_db, _capture_files,
    _git_init_commit, _capture_chunks_via_index_collection, _StatefulDocsChroma,
    _capture_embed_ids, _seed_manifest_with_doc_code, _remodel_fixture,
    _FakeChromaCollection, _FakeDB,
)


# --- Invalid project name validation (M4) ---

def test_run_index_invalid_project_raises(tmp_path, monkeypatch):
    """Indexing with a non-existent project name raises ValueError."""
    from vecs.config import _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "real_project": {
                "code_dirs": [{"path": "/tmp/code", "extensions": [".cs"]}],
            }
        }
    }))

    _clear_config_cache()

    monkeypatch.setattr("vecs.indexer.load_config", lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file))

    from vecs.indexer import run_index

    with pytest.raises(ValueError, match="ghost_project"):
        run_index(project_name="ghost_project")

def test_run_index_threads_one_cache_to_all_indexers(tmp_path, monkeypatch):
    """run_index constructs a single EmbedCache and hands it to every indexer."""
    from vecs.config import _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {"p": {"code_dirs": [{"path": str(tmp_path / "code"), "extensions": [".cs"]}]}},
    }))
    (tmp_path / "code").mkdir()
    _clear_config_cache()

    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file),
    )
    monkeypatch.setattr("vecs.indexer.migrate_global_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer.get_provider", lambda config=None, name=None: VoyageProvider(client=MagicMock()))
    # Fresh project: collections are empty, so the B2 pre-pass finds nothing to
    # re-embed. count() must be a real int (the pre-pass compares it > 0).
    db = MagicMock()
    db.get_collection.return_value.count.return_value = 0
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)
    monkeypatch.setattr("vecs.indexer.VECS_DIR", tmp_path)
    monkeypatch.setattr("vecs.indexer.CHROMADB_DIR", tmp_path / "chromadb")
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    seen: dict = {}

    def cap(name):
        def f(*a, cache=None, **kw):
            seen[name] = cache
            return 0
        return f

    monkeypatch.setattr("vecs.indexer.index_code", cap("code"))
    monkeypatch.setattr("vecs.indexer.index_docs", cap("docs"))

    from vecs.indexer import run_index
    run_index()

    assert seen["code"] is not None
    assert seen["code"] is seen["docs"]

def test_run_index_prune_isolation_one_project_failure_does_not_skip_others(tmp_path, monkeypatch):
    """run_index wraps the per-project prune+sweep in try/except so one project's
    prune/save I/O error can't skip the sweep for the rest of the run."""
    from vecs.config import _clear_config_cache

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {
            "p1": {"code_dirs": [{"path": str(tmp_path / "c1"), "extensions": [".cs"]}]},
            "p2": {"code_dirs": [{"path": str(tmp_path / "c2"), "extensions": [".cs"]}]},
        },
    }))
    (tmp_path / "c1").mkdir()
    (tmp_path / "c2").mkdir()
    _clear_config_cache()

    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file),
    )
    monkeypatch.setattr("vecs.indexer.migrate_global_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer.get_provider", lambda config=None, name=None: VoyageProvider(client=MagicMock()))
    db = MagicMock()
    db.get_collection.return_value.count.return_value = 0
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)
    monkeypatch.setattr("vecs.indexer.VECS_DIR", tmp_path)
    monkeypatch.setattr("vecs.indexer.CHROMADB_DIR", tmp_path / "chromadb")
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    for fn in ("index_code", "index_docs"):
        monkeypatch.setattr(f"vecs.indexer.{fn}", lambda *a, **kw: 0)

    swept: list[str] = []

    def fake_prune(project, db):
        swept.append(project.name)
        if project.name == "p1":
            raise OSError("manifest write failed")
        return {"pruned_keys": 0, "code_orphans": 0, "docs_orphans": 0}

    monkeypatch.setattr("vecs.indexer._prune_and_sweep_orphans", fake_prune)

    from vecs.indexer import run_index
    run_index()  # must NOT raise despite p1's prune failure

    assert swept == ["p1", "p2"]  # p2 still swept after p1 raised

# --- B2: model-change re-embed trigger (run_index pre/post pass) -----------

def test_remodel_docs_clear_backfills_unmarked_code_without_clearing(tmp_path, monkeypatch):
    """Docs model change clears docs file-keys only. The unmarked non-empty code
    collection is BACKFILLED (marker recorded = CODE_MODEL, NO clear) — it
    predates code markers and is assumed current (L1.4 no-regret backfill;
    replaces the retired 'code has no trigger' invariant)."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, doc_f, code_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code(tmp_path, doc_f, code_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")      # stale

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5  # non-empty

    cleared = _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) not in reloaded.data  # docs cleared -> re-embed
    assert str(code_f) in reloaded.data      # code keys untouched (backfill, no clear)
    assert cleared["code"] == 0
    from vecs.config import CODE_MODEL
    assert cache.get_collection_model("p-code") == CODE_MODEL
    cache.close()


def test_remodel_clears_code_on_real_model_mismatch(tmp_path, monkeypatch):
    """A RECORDED code marker != configured CODE_MODEL + non-empty collection
    clears the code file-keys (scoped by _code_sources), leaving docs alone."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, doc_f, code_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code(tmp_path, doc_f, code_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-4")        # docs current
    cache.set_collection_model("p-code", "voyage-code-2")   # code stale

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5  # non-empty

    cleared = _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(code_f) not in reloaded.data  # code cleared -> re-embed
    assert str(doc_f) in reloaded.data        # docs untouched
    assert cleared["code"] == 1
    cache.close()


def test_remodel_record_marks_both_docs_and_code(tmp_path, monkeypatch):
    """POST-pass records the current model for BOTH collections (code markers
    are new in L1.4 — they arm the searcher interlock for code)."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_record

    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, _doc_f, _code_f = _remodel_fixture(tmp_path)
    cache = EmbedCache(tmp_path / "embed_cache.db")

    _remodel_record(project, cache)

    from vecs.config import CODE_MODEL
    assert cache.get_collection_model("p-docs") == "voyage-4"
    assert cache.get_collection_model("p-code") == CODE_MODEL
    cache.close()

def test_remodel_clear_scope_matches_index_docs_scan_all_docs_dirs(tmp_path, monkeypatch):
    """F widens index_docs to scan ALL docs_dirs, so the model-change clear must
    widen with it: a stale-model clear drops docs keys for files under EVERY
    docs_dir (clear-scope == index_docs rescan-scope, by construction)."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    docs0 = tmp_path / "docs0"
    docs0.mkdir()
    docs1 = tmp_path / "docs1"
    docs1.mkdir()
    f0 = docs0 / "a.md"
    f0.write_text("x")
    f1 = docs1 / "b.md"
    f1.write_text("y")

    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(f0, "h0")
    manifest.mark_indexed(f1, "h1")
    manifest.save()

    project = ProjectConfig(name="p", docs_dirs=[docs0, docs1])

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")  # stale

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(f0) not in reloaded.data, "docs_dirs[0] file must be cleared"
    assert str(f1) not in reloaded.data, "docs_dirs[1] file must ALSO be cleared (F widens)"
    cache.close()

def test_remodel_clear_clears_inrepo_md_but_not_code_keys(tmp_path, monkeypatch):
    """Docs clear must drop in-repo .md keys (index_docs re-embeds them under the
    new docs model) but leave .cs code keys alone (code stays voyage-code-3 and
    is never re-scanned by index_docs)."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    code = tmp_path / "client-uk"
    (code / "Assets").mkdir(parents=True)
    md = code / "README.md"
    md.write_text("# r\n\nbody\n")
    cs = code / "Assets" / "Player.cs"
    cs.write_text("public class P {}")

    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(md, "hmd")
    manifest.mark_indexed(cs, "hcs")
    manifest.save()

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".cs"})])

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")  # stale -> docs clear fires

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(md) not in reloaded.data, "in-repo .md is a docs source -> cleared"
    assert str(cs) in reloaded.data, "code .cs key must NOT be cleared"
    cache.close()

def test_remodel_noop_when_model_unchanged(tmp_path, monkeypatch):
    """Recorded == configured: nothing cleared even though collection is non-empty."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, doc_f, code_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code(tmp_path, doc_f, code_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-4")      # matches
    cache.close()
    cache = EmbedCache(tmp_path / "embed_cache.db")

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 5

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    assert str(code_f) in reloaded.data
    cache.close()

def test_remodel_noop_when_collection_empty(tmp_path, monkeypatch):
    """Recorded differs but collection is empty (count==0): no re-embed needed."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, doc_f, code_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code(tmp_path, doc_f, code_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")      # stale...

    db = MagicMock()
    db.get_collection.return_value.count.return_value = 0  # ...but empty

    _remodel_clear(project, db, cache)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    cache.close()

def test_remodel_treats_missing_collection_as_empty(tmp_path, monkeypatch):
    """A collection that does not exist yet (get_collection raises) is empty -> no clear."""
    from vecs.embed_cache import EmbedCache
    from vecs.indexer import _remodel_clear

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    project, doc_f, code_f = _remodel_fixture(tmp_path)
    _seed_manifest_with_doc_code(tmp_path, doc_f, code_f)

    cache = EmbedCache(tmp_path / "embed_cache.db")
    cache.set_collection_model("p-docs", "voyage-3")

    db = MagicMock()
    db.get_collection.side_effect = Exception("collection not found")

    _remodel_clear(project, db, cache)  # must not raise

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) in reloaded.data
    cache.close()

def test_run_index_remodel_migrates_docs_manifest_and_records_marker(tmp_path, monkeypatch):
    """End-to-end migration: an existing voyage-3 docs collection (recorded
    voyage-3, non-empty) under configured voyage-4 has its docs manifest entry
    cleared by the pre-pass, and the post-pass records voyage-4 as the new
    marker so the next run is a no-op."""
    from vecs.config import _clear_config_cache
    from vecs.embed_cache import EmbedCache

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc_f = docs_dir / "a.md"
    doc_f.write_text("doc body")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "projects": {"p": {
            "code_dirs": [{"path": str(tmp_path / "code"), "extensions": [".py"]}],
            "docs_dirs": [str(docs_dir)],
        }},
    }))
    (tmp_path / "code").mkdir()
    _clear_config_cache()

    monkeypatch.setattr(
        "vecs.indexer.load_config",
        lambda: __import__("vecs.config", fromlist=["load_config"]).load_config(config_file),
    )
    monkeypatch.setattr("vecs.indexer.migrate_global_manifest", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer.get_provider", lambda config=None, name=None: VoyageProvider(client=MagicMock()))
    monkeypatch.setattr("vecs.indexer.VECS_DIR", tmp_path)
    monkeypatch.setattr("vecs.indexer.CHROMADB_DIR", tmp_path / "chromadb")
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer.DOCS_MODEL", "voyage-4")

    # Existing voyage-3 docs collection: non-empty, recorded under the old model.
    db = MagicMock()
    db.get_collection.return_value.count.return_value = 12
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    # Pre-seed: a docs manifest entry + the old marker (simulates pre-B2 state).
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    m.mark_indexed(doc_f, "h")
    m.save()
    seed = EmbedCache(tmp_path / "embed_cache.db")
    seed.set_collection_model("p-docs", "voyage-3")
    seed.close()

    # Indexers are no-ops: we are testing the orchestration pre/post pass only.
    monkeypatch.setattr("vecs.indexer.index_code", lambda *a, **kw: 0)
    monkeypatch.setattr("vecs.indexer.index_docs", lambda *a, **kw: 0)

    from vecs.indexer import run_index
    run_index()

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(doc_f) not in reloaded.data, "pre-pass should clear the stale docs entry"

    chk = EmbedCache(tmp_path / "embed_cache.db")
    assert chk.get_collection_model("p-docs") == "voyage-4", "post-pass should record new marker"
    chk.close()

# --- Inc 1.5a: prune-orphan fix (delete chunks for deleted sources) ---------

def test_sweep_deleted_source_chunks_deletes_when_source_gone(tmp_path):
    """Chunks whose root-qualified file_path no longer resolves on disk are
    deleted; chunks of an existing file survive."""
    from vecs.indexer import _sweep_deleted_source_chunks

    root = tmp_path / "repo"
    root.mkdir()
    (root / "keep.cs").write_text("// k")  # gone.cs is never created

    col = _FakeChromaCollection([
        ("code:repo/keep.cs:0", {"file_path": "repo/keep.cs"}),
        ("code:repo/gone.cs:0", {"file_path": "repo/gone.cs"}),
        ("code:repo/gone.cs:1", {"file_path": "repo/gone.cs"}),
    ])

    deleted = _sweep_deleted_source_chunks(col, {"repo": root})

    assert set(deleted) == {"code:repo/gone.cs:0", "code:repo/gone.cs:1"}
    assert "code:repo/keep.cs:0" in col._rows
    assert "code:repo/gone.cs:0" not in col._rows

def test_sweep_deleted_source_chunks_skips_unknown_root(tmp_path):
    """A file_path whose first segment is not a current root is LEFT ALONE
    (legacy bare-scheme -docs chunks are owned by _partition_docs_by_root)."""
    from vecs.indexer import _sweep_deleted_source_chunks

    root = tmp_path / "repo"
    root.mkdir()

    col = _FakeChromaCollection([
        ("docs:HQ/old.md:0", {"file_path": "HQ/old.md"}),       # unknown root
        ("docs:repo/gone.md:0", {"file_path": "repo/gone.md"}),  # known root, gone
    ])

    deleted = _sweep_deleted_source_chunks(col, {"repo": root})

    assert deleted == ["docs:repo/gone.md:0"]
    assert "docs:HQ/old.md:0" in col._rows  # untouched

def test_sweep_deleted_source_chunks_empty_root_map_is_noop():
    """An empty root map must NEVER wipe a collection (degenerate-prefix guard)."""
    from vecs.indexer import _sweep_deleted_source_chunks

    col = _FakeChromaCollection([
        ("code:repo/a.cs:0", {"file_path": "repo/a.cs"}),
    ])

    deleted = _sweep_deleted_source_chunks(col, {})

    assert deleted == []
    assert "code:repo/a.cs:0" in col._rows

def test_prune_and_sweep_orphans_deletes_code_docs_and_bm25(tmp_path, monkeypatch):
    """End-to-end: deleting a code file and a docs file makes a reindex's
    prune+sweep delete their chunks from chroma AND BM25; siblings survive."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"
    code_root.mkdir()
    (code_root / "keep.cs").write_text("// k")  # gone.cs absent on disk

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "keep.md").write_text("# k")  # gone.md absent on disk

    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code_root, extensions={".cs"})],
        docs_dirs=[docs_root],
    )

    code_col = _FakeChromaCollection([
        ("code:repo/keep.cs:0", {"file_path": "repo/keep.cs"}),
        ("code:repo/gone.cs:0", {"file_path": "repo/gone.cs"}),
        ("code:repo/gone.cs:1", {"file_path": "repo/gone.cs"}),
    ])
    docs_col = _FakeChromaCollection([
        ("docs:docs/keep.md:0", {"file_path": "docs/keep.md"}),
        ("docs:docs/gone.md:0", {"file_path": "docs/gone.md"}),
    ])
    db = _FakeDB({
        "p-code": code_col,
        "p-docs": docs_col,
        "p-sessions": _FakeChromaCollection([]),
    })

    bm25: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda proj, suffix, ids: bm25.append((suffix, sorted(ids))),
    )

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["code_orphans"] == 2
    assert stats["docs_orphans"] == 1
    assert "code:repo/keep.cs:0" in code_col._rows
    assert "code:repo/gone.cs:0" not in code_col._rows
    assert "code:repo/gone.cs:1" not in code_col._rows
    assert "docs:docs/keep.md:0" in docs_col._rows
    assert "docs:docs/gone.md:0" not in docs_col._rows
    assert ("code", ["code:repo/gone.cs:0", "code:repo/gone.cs:1"]) in bm25
    assert ("docs", ["docs:docs/gone.md:0"]) in bm25

def test_prune_and_sweep_orphans_clears_backlog_orphan(tmp_path, monkeypatch):
    """A chunk whose source is gone is swept even with NO manifest entry --
    the already-accumulated backlog the buggy prune() leaked."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"
    code_root.mkdir()
    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})]
    )

    code_col = _FakeChromaCollection([
        ("code:repo/ghost.cs:0", {"file_path": "repo/ghost.cs"}),
    ])
    db = _FakeDB({
        "p-code": code_col,
        "p-docs": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["pruned_keys"] == 0  # manifest empty -> nothing pruned
    assert stats["code_orphans"] == 1  # ...but the orphan chunk is still swept
    assert "code:repo/ghost.cs:0" not in code_col._rows

# --- Inc 1.5a: Phase-4 review hardening (data-loss guards) -------------------

def test_safe_sweep_root_map_excludes_missing_and_colliding(tmp_path):
    """The sweep root map includes only roots present on disk with a UNIQUE
    basename -- both guards degrade to leaving an orphan (never deleting live)."""
    from vecs.indexer import _safe_sweep_root_map

    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing"  # never created
    a_shared = tmp_path / "a" / "shared"
    b_shared = tmp_path / "b" / "shared"  # collides with a_shared on basename
    a_shared.mkdir(parents=True)
    b_shared.mkdir(parents=True)

    root_map = _safe_sweep_root_map([present, missing, a_shared, b_shared])

    assert root_map == {"present": present}  # missing dropped; "shared" collides -> dropped

def test_prune_and_sweep_orphans_skips_transiently_missing_root(tmp_path, monkeypatch):
    """A configured root that is missing on disk at sweep time (e.g. unmounted)
    must NOT be read as 'all its files deleted' -- its live chunks survive."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"  # deliberately NOT created -> transiently missing
    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})]
    )
    code_col = _FakeChromaCollection([
        ("code:repo/a.cs:0", {"file_path": "repo/a.cs"}),
        ("code:repo/b.cs:0", {"file_path": "repo/b.cs"}),
    ])
    db = _FakeDB({
        "p-code": code_col,
        "p-docs": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["code_orphans"] == 0  # root missing -> presumed transient, not swept
    assert "code:repo/a.cs:0" in code_col._rows
    assert "code:repo/b.cs:0" in code_col._rows

def test_prune_and_sweep_orphans_skips_colliding_basename_roots(tmp_path, monkeypatch):
    """Two roots sharing a basename make a chunk resolve against the WRONG root;
    rather than risk deleting a live file's chunks, colliding roots are skipped."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    a_shared = tmp_path / "a" / "shared"
    b_shared = tmp_path / "b" / "shared"
    a_shared.mkdir(parents=True)
    b_shared.mkdir(parents=True)
    (a_shared / "keep.cs").write_text("// live")  # exists under /a/shared, NOT /b/shared

    project = ProjectConfig(
        name="p",
        code_dirs=[
            CodeDir(path=a_shared, extensions={".cs"}),
            CodeDir(path=b_shared, extensions={".cs"}),
        ],
    )
    # Chunk came from /a/shared/keep.cs (live), but code_map would collapse to
    # the last "shared" (=/b/shared) and falsely delete it without the guard.
    code_col = _FakeChromaCollection([
        ("code:shared/keep.cs:0", {"file_path": "shared/keep.cs"}),
    ])
    db = _FakeDB({
        "p-code": code_col,
        "p-docs": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["code_orphans"] == 0
    assert "code:shared/keep.cs:0" in code_col._rows  # live chunk preserved

def test_prune_and_sweep_orphans_no_bm25_when_clean(tmp_path, monkeypatch):
    """A reindex where every source still exists deletes nothing and never opens
    the BM25 sidecar (the prune path must not call _delete_ids_from_bm25 with [])."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"
    code_root.mkdir()
    (code_root / "a.cs").write_text("// a")

    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})]
    )
    code_col = _FakeChromaCollection([("code:repo/a.cs:0", {"file_path": "repo/a.cs"})])
    db = _FakeDB({
        "p-code": code_col,
        "p-docs": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    bm25_calls: list = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda *a: bm25_calls.append(a),
    )

    stats = _prune_and_sweep_orphans(project, db)

    assert stats == {"pruned_keys": 0, "code_orphans": 0, "docs_orphans": 0}
    assert bm25_calls == []
    assert "code:repo/a.cs:0" in code_col._rows

def test_prune_and_sweep_orphans_keeps_unknown_root_docs_end_to_end(tmp_path, monkeypatch):
    """End-to-end: a legacy bare/unknown-root -docs chunk (owned by
    _partition_docs_by_root) survives the orphan sweep; only a known-root gone
    chunk is swept."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "keep.md").write_text("# k")  # gone.md absent

    project = ProjectConfig(name="p", docs_dirs=[docs_root])
    docs_col = _FakeChromaCollection([
        ("docs:HQ/old.md:0", {"file_path": "HQ/old.md"}),        # unknown root "HQ"
        ("docs:docs/keep.md:0", {"file_path": "docs/keep.md"}),  # known, present
        ("docs:docs/gone.md:0", {"file_path": "docs/gone.md"}),  # known, gone
    ])
    db = _FakeDB({
        "p-docs": docs_col,
        "p-code": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["docs_orphans"] == 1
    assert "docs:HQ/old.md:0" in docs_col._rows       # unknown root left alone
    assert "docs:docs/keep.md:0" in docs_col._rows
    assert "docs:docs/gone.md:0" not in docs_col._rows

def test_prune_and_sweep_orphans_sweeps_inrepo_md_docs_by_code_dir_root(tmp_path, monkeypatch):
    """A deleted in-repo .md (a -docs chunk rooted at a CODE_DIR) is swept via
    the code_dir entry in the docs root map."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"
    code_root.mkdir()
    (code_root / "KEEP.md").write_text("# k")  # GONE.md absent

    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})]
    )
    docs_col = _FakeChromaCollection([
        ("docs:repo/KEEP.md:0", {"file_path": "repo/KEEP.md"}),
        ("docs:repo/GONE.md:0", {"file_path": "repo/GONE.md"}),
    ])
    db = _FakeDB({
        "p-docs": docs_col,
        "p-code": _FakeChromaCollection([]),
        "p-sessions": _FakeChromaCollection([]),
    })
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["docs_orphans"] == 1
    assert "docs:repo/KEEP.md:0" in docs_col._rows
    assert "docs:repo/GONE.md:0" not in docs_col._rows

def test_prune_and_sweep_orphans_missing_collection_is_noop(tmp_path, monkeypatch):
    """A never-created collection (get_collection raises) is skipped, not fatal."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    from vecs.indexer import _prune_and_sweep_orphans

    code_root = tmp_path / "repo"
    code_root.mkdir()
    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})]
    )
    # _FakeDB has no p-code collection -> get_collection raises -> _col returns None.
    db = _FakeDB({"p-sessions": _FakeChromaCollection([])})
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)

    stats = _prune_and_sweep_orphans(project, db)

    assert stats["code_orphans"] == 0  # no collection -> skipped, no exception

def test_sweep_deleted_source_chunks_keeps_chunk_on_oserror(tmp_path, monkeypatch):
    """A path whose .exists() raises OSError is KEPT (never deleted) -- the
    safety-critical guard against deleting a live-but-unstattable file."""
    from vecs.indexer import _sweep_deleted_source_chunks

    root = tmp_path / "repo"
    root.mkdir()
    col = _FakeChromaCollection([("code:repo/x.cs:0", {"file_path": "repo/x.cs"})])

    real_exists = Path.exists

    def boom(self):
        if self.name == "x.cs":
            raise OSError("io error")
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", boom)

    deleted = _sweep_deleted_source_chunks(col, {"repo": root})

    assert deleted == []
    assert "code:repo/x.cs:0" in col._rows
