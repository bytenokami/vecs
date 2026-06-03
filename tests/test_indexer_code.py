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
    _get_session_new_content,
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
    _capture_embed_ids, _seed_manifest_with_doc_code_session, _remodel_fixture,
    _FakeChromaCollection, _FakeDB,
)


# --- exclude_dirs filtering on index_code ---

def test_index_code_exclude_dirs_drops_files(tmp_path, monkeypatch):
    """Files under exclude_dirs are not indexed."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Scripts" / "keep.cs"
    keep.write_text("public class K {}")
    drop = code_root / "Library" / "drop.cs"
    drop.write_text("public class D {}")

    project = ProjectConfig(
        name="exclproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert keep in captured["files"]
    assert drop not in captured["files"]

def test_index_code_exclude_wins_over_include(tmp_path, monkeypatch):
    """When include_dirs and exclude_dirs overlap, exclude wins."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts" / "Editor").mkdir(parents=True)
    (code_root / "Scripts" / "Runtime").mkdir(parents=True)
    editor_file = code_root / "Scripts" / "Editor" / "e.cs"
    editor_file.write_text("// editor")
    runtime_file = code_root / "Scripts" / "Runtime" / "r.cs"
    runtime_file.write_text("// runtime")

    project = ProjectConfig(
        name="overlapproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            include_dirs=["Scripts"],
            exclude_dirs=["Scripts/Editor"],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert runtime_file in captured["files"]
    assert editor_file not in captured["files"]

def test_index_code_empty_exclude_dirs_is_noop(tmp_path, monkeypatch):
    """Empty exclude_dirs walks the full tree just like before."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "UI").mkdir(parents=True)
    a = code_root / "Scripts" / "a.cs"
    a.write_text("// a")
    b = code_root / "UI" / "b.cs"
    b.write_text("// b")

    project = ProjectConfig(
        name="noexclproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=[],
        )],
    )

    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert a in captured["files"]
    assert b in captured["files"]

def test_index_code_prunes_manifest_for_now_excluded_files(tmp_path, monkeypatch):
    """Files previously indexed but now under exclude_dirs are pruned from the manifest."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    (code_root / "Scripts").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Scripts" / "keep.cs"
    keep.write_text("// keep")
    stale = code_root / "Library" / "stale.cs"
    stale.write_text("// stale")

    # Pre-populate the manifest as if both files were previously indexed
    manifest = Manifest("pruneproj", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(keep, hashlib.sha256(keep.read_bytes()).hexdigest())
    manifest.mark_indexed(stale, hashlib.sha256(stale.read_bytes()).hexdigest())
    manifest.save()

    project = ProjectConfig(
        name="pruneproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    reloaded = Manifest("pruneproj", manifests_dir=tmp_path / "manifests")
    assert str(keep) in reloaded.data
    assert str(stale) not in reloaded.data

def test_index_code_prune_leaves_non_code_extension_keys_for_docs(tmp_path, monkeypatch):
    """A .md under a code_dir is a DOCS source after F, so index_code's prune must
    NOT remove its manifest key. .md keys share the manifest namespace with code
    (both bare abs paths); if index_code pruned them every run, index_docs would
    re-embed the .md perpetually (thrash) since its key keeps vanishing."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    cs = code_root / "A.cs"
    cs.write_text("public class A {}")
    md = code_root / "README.md"
    md.write_text("# r\n\nbody\n")

    # Manifest already tracks the .md (index_docs owns it now; key = str(md)).
    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, hmd = manifest.needs_indexing(md)
    manifest.mark_indexed(md, hmd)
    manifest.save()

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs"})])
    _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    reloaded = Manifest("p", manifests_dir=tmp_path / "manifests")
    assert str(md) in reloaded.data, "index_code must not prune the .md key (docs owns it)"

def test_index_code_never_indexes_md_even_if_still_listed(tmp_path, monkeypatch, capsys):
    """Defensive (F): .md routes to -docs. If a code_dir still lists .md in
    extensions (live config not yet updated), index_code must NOT embed .md as
    code -- otherwise it fights the .md sweep (embed-then-sweep thrash + dual
    collection). It indexes the real code extension and warns about the .md."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "repo"
    code_root.mkdir()
    cs = code_root / "A.cs"
    cs.write_text("public class A {}")
    md = code_root / "R.md"
    md.write_text("# r\n\nbody\n")

    project = ProjectConfig(
        name="p", code_dirs=[CodeDir(path=code_root, extensions={".cs", ".md"})]
    )
    captured = _capture_files(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert cs in captured["files"]
    assert md not in captured["files"], ".md must never be indexed as code"
    assert ".md" in capsys.readouterr().err  # warned about the stale extension

# --- _sweep_excluded_chunks tests (Problem 2: orphan chunks for excluded files) ---

def test_sweep_excluded_chunks_deletes_orphans_under_excluded_subdirs(tmp_path):
    """Chunks whose file_path is under an excluded subdir are deleted."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    collection = MagicMock()
    # Single page is fine for this case
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Library/PackageCache/foo.cs:0",
            "code:client-uk/Assets/Scripts/Player.cs:0",
            "code:client-uk/Library/Cached.cs:1",
        ],
        "documents": ["", "", ""],
        "metadatas": [
            {"file_path": "client-uk/Library/PackageCache/foo.cs"},
            {"file_path": "client-uk/Assets/Scripts/Player.cs"},
            {"file_path": "client-uk/Library/Cached.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 2
    collection.delete.assert_called_once()
    deleted_ids = collection.delete.call_args.kwargs.get("ids") or collection.delete.call_args.args[0]
    assert set(deleted_ids) == {
        "code:client-uk/Library/PackageCache/foo.cs:0",
        "code:client-uk/Library/Cached.cs:1",
    }

def test_sweep_excluded_chunks_skips_chunks_outside_excluded_subdirs(tmp_path):
    """Chunks whose file_path is NOT under an excluded subdir stay put."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Assets/Scripts/A.cs:0",
            "code:client-uk/Assets/Scripts/B.cs:0",
        ],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "client-uk/Assets/Scripts/A.cs"},
            {"file_path": "client-uk/Assets/Scripts/B.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 0
    collection.delete.assert_not_called()

def test_sweep_excluded_chunks_noop_with_empty_exclude_dirs(tmp_path):
    """When exclude_dirs is empty, the helper does nothing -- no get, no delete."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=[],
    )

    collection = MagicMock()

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 0
    collection.get.assert_not_called()
    collection.delete.assert_not_called()

def test_sweep_excluded_chunks_paginates_delete_for_huge_orphan_sets(tmp_path):
    """When orphan count exceeds the SQLite IN-clause cap, delete must be batched.

    Same SQL-variable bug that hit collection.get() rebirths on collection.delete()
    if we hand it a single list of 10K+ ids. Must chunk into batches <= 5000.
    """
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Library"],
    )

    n_orphans = 12000
    ids = [f"code:client-uk/Library/file{i}.cs:0" for i in range(n_orphans)]
    metadatas = [{"file_path": f"client-uk/Library/file{i}.cs"} for i in range(n_orphans)]

    collection = MagicMock()
    collection.get.side_effect = [
        {"ids": ids, "documents": [""] * n_orphans, "metadatas": metadatas},
        {"ids": [], "documents": [], "metadatas": []},
    ]

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == n_orphans
    assert collection.delete.call_count >= 2, (
        f"delete called {collection.delete.call_count} times for {n_orphans} orphans "
        "— must batch to avoid SQL variable limit"
    )
    all_deleted: list[str] = []
    for call in collection.delete.call_args_list:
        batch = call.kwargs.get("ids") or call.args[0]
        assert len(batch) <= 5000, f"batch size {len(batch)} exceeds safe SQLite limit"
        all_deleted.extend(batch)
    assert set(all_deleted) == set(ids)

def test_sweep_excluded_chunks_does_not_match_path_prefix_collisions(tmp_path):
    """Excluding 'Lib' must not match 'Library/' -- match on path component boundary."""
    from vecs.config import CodeDir

    code_root = tmp_path / "client-uk"
    code_root.mkdir()
    code_dir = CodeDir(
        path=code_root,
        extensions={".cs"},
        exclude_dirs=["Lib"],
    )

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "code:client-uk/Library/foo.cs:0",  # NOT under "Lib/"
            "code:client-uk/Lib/bar.cs:0",      # IS  under "Lib/"
        ],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "client-uk/Library/foo.cs"},
            {"file_path": "client-uk/Lib/bar.cs"},
        ],
    }

    swept = _sweep_excluded_chunks(collection, code_dir)

    assert swept == 1
    collection.delete.assert_called_once()
    deleted_ids = collection.delete.call_args.kwargs.get("ids") or collection.delete.call_args.args[0]
    assert deleted_ids == ["code:client-uk/Lib/bar.cs:0"]

def test_index_code_sweeps_orphan_chunks_for_excluded_dirs(tmp_path, monkeypatch):
    """index_code sweeps chromadb for orphan chunks under exclude_dirs.

    Regression: prune_out_of_scope only removes manifest-tracked entries. If a
    file was upserted to chromadb but the run died before manifest.save(),
    chunks remain. After exclude_dirs is configured, those chunks must be
    deleted regardless of manifest state.
    """
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "client-uk"
    (code_root / "Assets").mkdir(parents=True)
    (code_root / "Library").mkdir(parents=True)
    keep = code_root / "Assets" / "keep.cs"
    keep.write_text("public class K {}")

    project = ProjectConfig(
        name="sweepproj",
        code_dirs=[CodeDir(
            path=code_root,
            extensions={".cs"},
            exclude_dirs=["Library"],
        )],
    )

    # Simulate orphan chunks living in chromadb under Library/
    orphan_ids = [
        "code:client-uk/Library/PackageCache/orphan_a.cs:0",
        "code:client-uk/Library/orphan_b.cs:0",
    ]

    collection = MagicMock()

    def fake_get(*args, **kwargs):
        # First call: paginate ALL chunks for the sweep
        # Subsequent calls (from _delete_stale_chunks_after_embed via _track_embed_success
        # or other paths) get an empty result -- not under test here.
        if "where" in kwargs:
            return {"ids": []}
        offset = kwargs.get("offset", 0)
        if offset == 0:
            return {
                "ids": orphan_ids + ["code:client-uk/Assets/keep.cs:0"],
                "documents": ["", "", ""],
                "metadatas": [
                    {"file_path": "client-uk/Library/PackageCache/orphan_a.cs"},
                    {"file_path": "client-uk/Library/orphan_b.cs"},
                    {"file_path": "client-uk/Assets/keep.cs"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    _capture_files(monkeypatch)
    index_code(project, vo=MagicMock(), db=db)

    # Find the sweep-driven delete call (by orphan ids), tolerating other delete calls
    sweep_calls = [
        c for c in collection.delete.call_args_list
        if set(c.kwargs.get("ids") or (c.args[0] if c.args else [])) == set(orphan_ids)
    ]
    assert len(sweep_calls) == 1, (
        f"Expected exactly one sweep delete with orphan ids, "
        f"got delete calls: {collection.delete.call_args_list}"
    )

# --- F: .md sweep out of -code collections ---

def test_sweep_md_code_chunks_deletes_md_sourced_chunks(tmp_path):
    """Chunks whose file_path ends in .md are deleted; others survive."""
    from vecs.indexer import _sweep_md_code_chunks

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "code:client-uk/README.md:0",
            "code:client-uk/Assets/Player.cs:0",
            "code:client-uk/docs/GUIDE.md:1",
        ],
        "documents": ["", "", ""],
        "metadatas": [
            {"file_path": "client-uk/README.md"},
            {"file_path": "client-uk/Assets/Player.cs"},
            {"file_path": "client-uk/docs/GUIDE.md"},
        ],
    }

    swept = _sweep_md_code_chunks(collection)

    assert set(swept) == {"code:client-uk/README.md:0", "code:client-uk/docs/GUIDE.md:1"}
    collection.delete.assert_called_once()
    deleted = collection.delete.call_args.kwargs.get("ids") or collection.delete.call_args.args[0]
    assert set(deleted) == {"code:client-uk/README.md:0", "code:client-uk/docs/GUIDE.md:1"}

def test_sweep_md_code_chunks_noop_when_no_md(tmp_path):
    """No .md chunks present -> nothing swept, no delete call."""
    from vecs.indexer import _sweep_md_code_chunks

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["code:client-uk/A.cs:0", "code:client-uk/B.shader:0"],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "client-uk/A.cs"},
            {"file_path": "client-uk/B.shader"},
        ],
    }

    swept = _sweep_md_code_chunks(collection)

    assert swept == []
    collection.delete.assert_not_called()

def test_sweep_md_code_chunks_paginates_delete_for_huge_sets(tmp_path):
    """>5000 .md chunks must be deleted in batches under the SQLite var cap."""
    from vecs.indexer import _sweep_md_code_chunks

    n = 12000
    ids = [f"code:client-uk/doc{i}.md:0" for i in range(n)]
    metadatas = [{"file_path": f"client-uk/doc{i}.md"} for i in range(n)]
    collection = MagicMock()
    collection.get.side_effect = [
        {"ids": ids, "documents": [""] * n, "metadatas": metadatas},
        {"ids": [], "documents": [], "metadatas": []},
    ]

    swept = _sweep_md_code_chunks(collection)

    assert set(swept) == set(ids)
    assert collection.delete.call_count >= 2
    all_deleted: list[str] = []
    for call in collection.delete.call_args_list:
        batch = call.kwargs.get("ids") or call.args[0]
        assert len(batch) <= 5000
        all_deleted.extend(batch)
    assert set(all_deleted) == set(ids)

def test_index_code_sweeps_md_chunks_and_syncs_bm25(tmp_path, monkeypatch):
    """index_code sweeps .md chunks out of -code (chroma + BM25) every run.

    Asserts the SWEEP RAN (a delete with the .md ids fired), not merely that the
    end state has zero .md. .md is no longer in extensions, so the only .md in
    the collection is leftover residue from before F dropped the extension.
    """
    from vecs.indexer import index_code

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code_root = tmp_path / "client-uk"
    (code_root / "Assets").mkdir(parents=True)
    keep = code_root / "Assets" / "keep.cs"
    keep.write_text("public class K {}")
    # An on-disk .md that is NO LONGER in extensions (so index_code won't index it)
    (code_root / "README.md").write_text("# readme")

    project = ProjectConfig(
        name="mdproj",
        code_dirs=[CodeDir(path=code_root, extensions={".cs"})],
    )

    md_ids = ["code:client-uk/README.md:0", "code:client-uk/Docs/X.md:0"]

    collection = MagicMock()

    def fake_get(*args, **kwargs):
        if "where" in kwargs:
            return {"ids": []}
        offset = kwargs.get("offset", 0)
        if offset == 0:
            return {
                "ids": md_ids + ["code:client-uk/Assets/keep.cs:0"],
                "documents": ["", "", ""],
                "metadatas": [
                    {"file_path": "client-uk/README.md"},
                    {"file_path": "client-uk/Docs/X.md"},
                    {"file_path": "client-uk/Assets/keep.cs"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    bm25_deletes: list[list[str]] = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda proj, suffix, ids: bm25_deletes.append((proj, suffix, list(ids))),
    )
    # Patch the embed pipeline (the sweep runs before it); keep.cs would
    # otherwise hit a real Voyage embed against MagicMocks.
    _capture_files(monkeypatch)

    index_code(project, vo=MagicMock(), db=db)

    md_delete_calls = [
        c for c in collection.delete.call_args_list
        if set(c.kwargs.get("ids") or (c.args[0] if c.args else [])) == set(md_ids)
    ]
    assert len(md_delete_calls) == 1, (
        f"Expected exactly one .md sweep delete; got {collection.delete.call_args_list}"
    )
    assert bm25_deletes == [("mdproj", "code", md_ids)], (
        f"BM25 sidecar must be swept with the same .md ids; got {bm25_deletes}"
    )

# --- C: version_id stamping (git SHA for code, mtime for docs, session id) ---

def test_git_sha_returns_head_for_repo(tmp_path):
    from vecs.indexer import _git_sha
    repo = tmp_path / "r"
    sha = _git_init_commit(repo, "a.cs", "class A{}")
    assert _git_sha(repo) == sha
    sub = repo / "sub"
    sub.mkdir()
    assert _git_sha(sub) == sha  # resolves from a subdir too

def test_git_sha_none_for_non_git_dir(tmp_path):
    from vecs.indexer import _git_sha
    d = tmp_path / "nogit"
    d.mkdir()
    assert _git_sha(d) is None

def test_index_code_stamps_git_sha_version_id(tmp_path, monkeypatch):
    """Each stored code chunk carries version_id == the repo HEAD sha."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    repo = tmp_path / "repo"
    sha = _git_init_commit(repo, "Main.cs", "public class Main { void M() { int x = 1; } }")

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=repo, extensions={".cs"})])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_code(project, vo=MagicMock(), db=db)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == sha for c in captured["chunks"])

# --- C: production wiring of the EmbedCache through run_index / indexers ---

def test_index_code_uses_embed_cache_end_to_end(tmp_path, monkeypatch):
    """A cache passed to index_code is threaded down to _embed_and_store: a
    second index of unchanged files makes zero Voyage calls."""
    from vecs.embed_cache import EmbedCache
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    repo = tmp_path / "repo"
    _git_init_commit(repo, "A.cs", "public class A { void M() { int x = 1; } }")
    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=repo, extensions={".cs"})])

    cache = EmbedCache(tmp_path / "c.db")
    db, _collection = _make_index_db(tmp_path)
    vo = MagicMock()
    vo.embed.side_effect = lambda texts, model, input_type: FakeEmbedResult(len(texts))

    n1 = index_code(project, vo=vo, db=db, cache=cache)
    assert n1 >= 1
    assert vo.embed.call_count >= 1

    # Force a re-index (drop the manifest) but keep the cache warm.
    (tmp_path / "manifests" / "p.json").unlink()
    vo.embed.reset_mock()
    n2 = index_code(project, vo=vo, db=db, cache=cache)
    assert n2 >= 1                    # same chunks re-stored
    assert vo.embed.call_count == 0   # all served from cache -> wiring intact
    cache.close()
