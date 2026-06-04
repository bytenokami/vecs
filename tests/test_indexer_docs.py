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
    index_docs,
    migrate_global_manifest,
)
from vecs.config import VecsConfig, ProjectConfig, CodeDir
from indexer_helpers import (  # noqa: F401
    FakeEmbedResult, _embedded_texts, _make_index_db, _capture_files,
    _git_init_commit, _capture_chunks_via_index_collection, _StatefulDocsChroma,
    _capture_embed_ids, _seed_manifest_with_doc_code, _remodel_fixture,
    _FakeChromaCollection, _FakeDB,
)


def test_index_single_doc_stamps_mtime_version_id(tmp_path, monkeypatch):
    """index_single_doc (vecs add-document / MCP add_document) must stamp the
    same mtime version_id as index_docs on the shared -docs collection."""
    from vecs.indexer import index_single_doc
    from vecs.config import VecsConfig

    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "note.md"
    md.write_text("# Note\n\nBody text long enough to chunk into something useful.\n")
    expected = str(md.stat().st_mtime)

    cfg = VecsConfig(path=tmp_path / "config.yaml")
    cfg.projects["p"] = ProjectConfig(name="p", docs_dirs=[docs])
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    captured = _capture_chunks_via_index_collection(monkeypatch)
    index_single_doc("p", md)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == expected for c in captured["chunks"])

def test_index_docs_stamps_mtime_version_id(tmp_path, monkeypatch):
    """Each stored docs chunk carries version_id == the file mtime."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "readme.md"
    md.write_text("# Title\n\nSome documentation body text that is long enough to chunk.\n")
    expected = str(md.stat().st_mtime)

    project = ProjectConfig(name="p", docs_dirs=[docs])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    assert captured["chunks"]
    assert all(c["metadata"]["version_id"] == expected for c in captured["chunks"])

# --- F: multi-source index_docs (source-root-qualified rel_path) ---

def test_docs_sources_enumerates_docs_dirs_and_inrepo_md(tmp_path):
    """_docs_sources returns (root, file): every doc-ext file under each docs_dir
    plus in-repo .md under each code_dir (root = code_dir.path), de-duplicated."""
    from vecs.indexer import _docs_sources

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("g")
    (docs / "notes.txt").write_text("n")

    code = tmp_path / "client-uk"
    (code / "Assets").mkdir(parents=True)
    (code / "README.md").write_text("r")
    (code / "Assets" / "Player.cs").write_text("c")  # code, not a doc source

    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code, extensions={".cs"})],
        docs_dirs=[docs],
    )

    sources = _docs_sources(project)
    by_file = {f: root for root, f in sources}

    assert (docs / "guide.md") in by_file and by_file[docs / "guide.md"] == docs
    assert (docs / "notes.txt") in by_file and by_file[docs / "notes.txt"] == docs
    assert (code / "README.md") in by_file and by_file[code / "README.md"] == code
    assert (code / "Assets" / "Player.cs") not in by_file  # .cs is not a doc source

def test_index_docs_qualifies_chunk_id_with_source_root_basename(tmp_path, monkeypatch):
    """docs chunk id + file_path are source-root-qualified: docs:{root.name}/{rel}."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    md = docs / "sub" / "readme.md"
    md.write_text("# Title\n\nBody text long enough to chunk into something.\n")

    project = ProjectConfig(name="p", docs_dirs=[docs])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    assert captured["chunks"]
    for c in captured["chunks"]:
        assert c["id"].startswith("docs:docs/sub/readme.md:")
        assert c["metadata"]["file_path"] == "docs/sub/readme.md"

def test_index_docs_two_roots_same_readme_do_not_collide(tmp_path, monkeypatch):
    """Collision test (deliverable 3): two distinct source roots each holding
    README.md produce DISTINCT chunk ids + distinct cleanup file_path values, so
    neither's reindex can over-match and delete the other's chunks."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    repo_a = tmp_path / "repoA"
    repo_a.mkdir()
    (repo_a / "README.md").write_text("# A\n\nAlpha body text long enough to chunk.\n")
    repo_b = tmp_path / "repoB"
    repo_b.mkdir()
    (repo_b / "README.md").write_text("# B\n\nBeta body text long enough to chunk.\n")

    project = ProjectConfig(name="p", docs_dirs=[repo_a, repo_b])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    ids = [c["id"] for c in captured["chunks"]]
    paths = {c["metadata"]["file_path"] for c in captured["chunks"]}
    assert "repoA/README.md" in paths
    assert "repoB/README.md" in paths
    assert len(ids) == len(set(ids)), "chunk ids must be globally unique across roots"
    # The two files' chunk-id namespaces must be disjoint (no shared id => no
    # upsert overwrite, and per-file cleanup filters on the qualified path).
    a_ids = {i for i in ids if i.startswith("docs:repoA/")}
    b_ids = {i for i in ids if i.startswith("docs:repoB/")}
    assert a_ids and b_ids and not (a_ids & b_ids)

def test_index_docs_two_roots_same_readme_real_cleanup_no_mutual_delete(tmp_path, monkeypatch):
    """Stronger collision test (drives REAL _index_collection cleanup): two roots
    each with README.md both survive -- per-file _delete_stale_chunks_after_embed
    filters on the qualified file_path so neither file's cleanup matches the
    other's chunks."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    repo_a = tmp_path / "repoA"
    repo_a.mkdir()
    (repo_a / "README.md").write_text("# A\n\nAlpha body long enough to chunk.\n")
    repo_b = tmp_path / "repoB"
    repo_b.mkdir()
    (repo_b / "README.md").write_text("# B\n\nBeta body long enough to chunk.\n")

    collection = _StatefulDocsChroma()
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    # Real _index_collection (so cleanup runs); fake only the embed -> upsert.
    def fake_embed(chunks, coll, model, vo, batcher=None, cache=None):
        coll.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a, **kw: None)

    project = ProjectConfig(name="p", docs_dirs=[repo_a, repo_b])
    index_docs(project, vo=MagicMock(), db=db)

    paths = {e["metadata"]["file_path"] for e in collection.store.values()}
    assert "repoA/README.md" in paths, "repoA survived"
    assert "repoB/README.md" in paths, "repoB survived (not deleted by repoA cleanup)"

def test_docs_sources_dedups_overlapping_roots_docs_wins(tmp_path):
    """When a docs_dir is nested in a code_dir, the same .md is reachable from
    both; _docs_sources must emit it ONCE, rooted at the docs_dir (docs win)."""
    from vecs.indexer import _docs_sources

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    shared = repo / "docs" / "x.md"
    shared.write_text("shared")

    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=repo, extensions={".cs"})],  # .md scan would also see it
        docs_dirs=[repo / "docs"],
    )

    sources = _docs_sources(project)
    matches = [(root, f) for root, f in sources if f.resolve() == shared.resolve()]
    assert len(matches) == 1, f"shared .md must appear once; got {matches}"
    assert matches[0][0] == repo / "docs", "docs_dir must win as the root"

def test_index_docs_qualifies_txt_and_pdf_under_docs_dir(tmp_path, monkeypatch):
    """Deliverable 3 covers .md/.txt/.pdf: non-.md docs are also source-root
    qualified (id + file_path = docs:{root.name}/{rel})."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr(
        "vecs.indexer.extract_pdf_text", lambda p: "PDF body long enough to chunk here.\n"
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("Plain text body long enough to chunk here.\n")
    (docs / "manual.pdf").write_bytes(b"%PDF-1.4 stub")

    project = ProjectConfig(name="p", docs_dirs=[docs])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    paths = {c["metadata"]["file_path"] for c in captured["chunks"]}
    assert "docs/notes.txt" in paths
    assert "docs/manual.pdf" in paths
    for c in captured["chunks"]:
        assert c["id"].startswith("docs:docs/")

def test_docs_source_root_names_unions_docs_and_code_basenames(tmp_path):
    """The orphan-sweep keep-set must include code_dir basenames, else qualified
    in-repo-.md chunks (docs:client-uk/...) would be classed unrooted and wiped."""
    from vecs.indexer import _docs_source_root_names

    project = ProjectConfig(
        name="p",
        docs_dirs=[tmp_path / "docs"],
        code_dirs=[CodeDir(path=tmp_path / "client-uk", extensions={".cs"})],
    )
    assert _docs_source_root_names(project) == {"docs", "client-uk"}

def test_index_docs_routes_inrepo_md_when_no_docs_dir(tmp_path, monkeypatch):
    """A project with NO docs_dir still indexes in-repo .md under its code_dirs
    into the -docs collection (deliverable 4)."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "eric"
    (code / "src").mkdir(parents=True)
    (code / "README.md").write_text("# Eric\n\nProject readme body long enough.\n")
    (code / "src" / "app.ts").write_text("export const x = 1")

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".ts"})])
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    n = index_docs(project, vo=MagicMock(), db=db)

    assert n > 0
    paths = {c["metadata"]["file_path"] for c in captured["chunks"]}
    assert paths == {"eric/README.md"}  # only the .md, never the .ts

def test_index_docs_inrepo_md_respects_code_dir_exclude(tmp_path, monkeypatch):
    """In-repo .md discovery uses the code_dir's own exclude scope, so a .md
    under an excluded subdir is NOT routed to docs (no third-party junk)."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"
    (code / "Assets").mkdir(parents=True)
    (code / "Library" / "PackageCache").mkdir(parents=True)
    (code / "Assets" / "GUIDE.md").write_text("# guide\n\nkept body long enough.\n")
    (code / "Library" / "PackageCache" / "VENDOR.md").write_text("# vendor\n\ndropped.\n")

    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code, extensions={".cs"}, exclude_dirs=["Library"])],
    )
    captured = _capture_chunks_via_index_collection(monkeypatch)
    db, _collection = _make_index_db(tmp_path)
    index_docs(project, vo=MagicMock(), db=db)

    paths = {c["metadata"]["file_path"] for c in captured["chunks"]}
    assert paths == {"client-uk/Assets/GUIDE.md"}

def test_index_docs_no_md_content_lost(tmp_path, monkeypatch):
    """Deliverable 7: every in-scope .md under code_dirs (∪ docs_dirs) is tracked
    in the docs manifest after a docs index pass."""
    from vecs.indexer import index_docs, _docs_sources, Manifest
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"
    (code / "Assets").mkdir(parents=True)
    (code / "Library").mkdir(parents=True)
    (code / "README.md").write_text("# a\n\nbody long enough to chunk here.\n")
    (code / "Assets" / "B.md").write_text("# b\n\nbody long enough to chunk here.\n")
    (code / "Library" / "skip.md").write_text("# skip\n\nexcluded body.\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "C.md").write_text("# c\n\nbody long enough to chunk here.\n")

    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code, extensions={".cs"}, exclude_dirs=["Library"])],
        docs_dirs=[docs],
    )
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    # Real _index_collection so the manifest is actually written; fake the embed.
    monkeypatch.setattr(
        "vecs.indexer._embed_and_store",
        lambda chunks, *a, **kw: [c["id"] for c in chunks],
    )
    index_docs(project, vo=MagicMock(), db=db)

    expected_md = {f for _root, f in _docs_sources(project) if f.suffix == ".md"}
    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    tracked_md = {Path(k) for k in manifest.data if not k.startswith("session:") and k.endswith(".md")}

    assert expected_md == tracked_md
    assert (code / "Library" / "skip.md") not in tracked_md  # excluded, never tracked
    assert len(expected_md) == 3

def test_index_docs_returns_zero_without_any_source(tmp_path, monkeypatch):
    """No docs_dir and no .md under code_dirs -> index_docs is a no-op (0)."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "eric"
    (code / "src").mkdir(parents=True)
    (code / "src" / "app.ts").write_text("export const x = 1")

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".ts"})])
    db, _collection = _make_index_db(tmp_path)
    assert index_docs(project, vo=MagicMock(), db=db) == 0

# --- F: -docs partition by source root (orphans + present qualified paths) ---

def test_partition_docs_by_root_separates_orphans_from_present(tmp_path):
    """Chunks not under any current source-root prefix are returned as orphans;
    correctly qualified chunks are returned in the present-path set. The
    partition does NOT delete -- the caller does."""
    from vecs.indexer import _partition_docs_by_root

    collection = MagicMock()
    collection.get.return_value = {
        "ids": [
            "docs:HQ/old-bare.md:0",        # legacy bare rel (root "docs" missing) -> orphan
            "docs:docs/keep.md:0",          # qualified under "docs" -> present
            "docs:client-uk/README.md:0",   # qualified under "client-uk" -> present
            "docs:removed-repo/x.md:0",     # root no longer configured -> orphan
        ],
        "documents": ["", "", "", ""],
        "metadatas": [
            {"file_path": "HQ/old-bare.md"},
            {"file_path": "docs/keep.md"},
            {"file_path": "client-uk/README.md"},
            {"file_path": "removed-repo/x.md"},
        ],
    }

    orphan_ids, present = _partition_docs_by_root(collection, {"docs", "client-uk"})

    assert set(orphan_ids) == {"docs:HQ/old-bare.md:0", "docs:removed-repo/x.md:0"}
    assert present == {"docs/keep.md", "client-uk/README.md"}
    collection.delete.assert_not_called()  # partition never deletes

def test_partition_docs_by_root_all_qualified_no_orphans(tmp_path):
    """When every chunk is already root-qualified, there are no orphans and all
    file_paths are present."""
    from vecs.indexer import _partition_docs_by_root

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["docs:docs/a.md:0", "docs:client-uk/b.md:0"],
        "documents": ["", ""],
        "metadatas": [
            {"file_path": "docs/a.md"},
            {"file_path": "client-uk/b.md"},
        ],
    }

    orphan_ids, present = _partition_docs_by_root(collection, {"docs", "client-uk"})

    assert orphan_ids == []
    assert present == {"docs/a.md", "client-uk/b.md"}

def test_partition_docs_by_root_empty_roots_never_scans(tmp_path):
    """Safety: an empty source-root set returns ([], set()) WITHOUT scanning, so
    no chunk is ever classified as an orphan (which would wipe the collection
    via the caller). Guards the `str.startswith(())` always-False degenerate."""
    from vecs.indexer import _partition_docs_by_root

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["docs:docs/a.md:0"],
        "documents": [""],
        "metadatas": [{"file_path": "docs/a.md"}],
    }

    orphan_ids, present = _partition_docs_by_root(collection, set())

    assert orphan_ids == []
    assert present == set()
    collection.get.assert_not_called()

def test_index_docs_sweeps_unrooted_chunks_even_with_nothing_new(tmp_path, monkeypatch):
    """The orphan sweep runs on every docs pass (chroma + BM25), including runs
    where there is nothing new to index -- so legacy bare-id chunks get purged."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "a.md"
    md.write_text("# a\n\nbody long enough to chunk.\n")
    # Pre-mark it indexed so to_index is empty (nothing new this run).
    from vecs.indexer import Manifest
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(md)
    m.mark_indexed(md, h)
    m.save()

    legacy_ids = ["docs:HQ/legacy.md:0"]
    collection = MagicMock()

    def fake_get(*args, **kwargs):
        if "where" in kwargs:
            return {"ids": []}
        offset = kwargs.get("offset", 0)
        if offset == 0:
            return {
                "ids": legacy_ids + ["docs:docs/a.md:0"],
                "documents": ["", ""],
                "metadatas": [
                    {"file_path": "HQ/legacy.md"},
                    {"file_path": "docs/a.md"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    bm25_deletes: list = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda proj, suffix, ids: bm25_deletes.append((proj, suffix, list(ids))),
    )
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)

    project = ProjectConfig(name="p", docs_dirs=[docs])
    index_docs(project, vo=MagicMock(), db=db)

    sweep_calls = [
        c for c in collection.delete.call_args_list
        if set(c.kwargs.get("ids") or (c.args[0] if c.args else [])) == set(legacy_ids)
    ]
    assert len(sweep_calls) == 1, f"orphan sweep must fire; got {collection.delete.call_args_list}"
    assert bm25_deletes == [("p", "docs", legacy_ids)]

def test_index_docs_migrates_legacy_bare_id_without_model_change(tmp_path, monkeypatch):
    """Phase-4 regression: the bare-id -> qualified id migration must self-heal
    even with NO docs model change. A legacy bare-id chunk + a steady-state
    manifest key (matching hash) must be RE-EMBEDDED under the qualified id, not
    merely deleted (which would silently lose the content)."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    (docs / "HQ").mkdir(parents=True)
    md = docs / "HQ" / "guide.md"
    md.write_text("# guide\n\nbody long enough to chunk into something.\n")
    # Steady-state manifest entry (matching current hash) -> needs_indexing False
    # WITHOUT the fix. No model-change clear is simulated.
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(md)
    m.mark_indexed(md, h)
    m.save()

    collection = MagicMock()

    def fake_get(*args, **kwargs):
        if "where" in kwargs:
            return {"ids": []}
        if kwargs.get("offset", 0) == 0:
            return {
                "ids": ["docs:HQ/guide.md:0"],            # legacy BARE id
                "documents": [""],
                "metadatas": [{"file_path": "HQ/guide.md"}],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    captured = _capture_embed_ids(monkeypatch)
    project = ProjectConfig(name="p", docs_dirs=[docs])
    index_docs(project, vo=MagicMock(), db=db)

    assert any(cid.startswith("docs:docs/HQ/guide.md:") for cid in captured), (
        f"file must be re-embedded under the qualified id; embedded ids={captured}"
    )

def test_index_docs_migrates_inrepo_md_without_model_change(tmp_path, monkeypatch):
    """Phase-4 regression: an in-repo .md previously code-indexed (manifest key set,
    matching hash) must be embedded into -docs under the qualified id even with no
    model change and an empty -docs collection -- else it is lost from both."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"
    code.mkdir()
    md = code / "README.md"
    md.write_text("# readme\n\nbody long enough to chunk into something.\n")
    # Pre-mark as if index_code (pre-F) tracked it: shared manifest key = str(md).
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(md)
    m.mark_indexed(md, h)
    m.save()

    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}  # -docs empty
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    captured = _capture_embed_ids(monkeypatch)
    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".cs"})])
    index_docs(project, vo=MagicMock(), db=db)

    assert any(cid.startswith("docs:client-uk/README.md:") for cid in captured), (
        f"in-repo .md must be (re-)embedded into -docs; embedded ids={captured}"
    )

def test_md_reroute_converges_end_to_end_no_model_change(tmp_path, monkeypatch):
    """Capstone (Phase-4): the .md->docs reroute converges across BOTH collections
    in run_index order (index_code then index_docs), with NO model change and the
    .md previously code-indexed. End state: 0 .md in -code, the .md present in
    -docs under the qualified id. This is the exact loss scenario the review
    reproduced; it must now self-heal."""
    from vecs.indexer import index_code, index_docs, Manifest
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"
    code.mkdir()
    cs = code / "A.cs"
    cs.write_text("public class A {}")
    md = code / "README.md"
    md.write_text("# r\n\nReadme body long enough to chunk into something.\n")

    # Pre-state: .md was code-indexed (manifest key + a -code chunk); -docs empty.
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, hmd = m.needs_indexing(md)
    m.mark_indexed(md, hmd)
    m.save()

    code_coll = _StatefulDocsChroma()
    code_coll.upsert(
        ids=["code:client-uk/README.md:0"],
        documents=["# r"],
        metadatas=[{"file_path": "client-uk/README.md"}],
    )
    docs_coll = _StatefulDocsChroma()
    collections = {"p-code": code_coll, "p-docs": docs_coll}
    db = MagicMock()
    db.get_or_create_collection.side_effect = lambda name: collections[name]

    def fake_embed(chunks, coll, model, vo, batcher=None, cache=None):
        coll.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return [c["id"] for c in chunks]

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a, **kw: None)

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".cs"})])

    # run_index order: code first (sweeps .md out of -code), then docs.
    index_code(project, vo=MagicMock(), db=db)
    index_docs(project, vo=MagicMock(), db=db)

    code_paths = {e["metadata"]["file_path"] for e in code_coll.store.values()}
    docs_paths = {e["metadata"]["file_path"] for e in docs_coll.store.values()}
    assert not any(p.endswith(".md") for p in code_paths), f"-code still has .md: {code_paths}"
    assert "client-uk/README.md" in docs_paths, f"-docs missing the rerouted .md: {docs_paths}"

def test_index_docs_steady_state_second_run_is_noop(tmp_path, monkeypatch):
    """Idempotency: once chunks are present under qualified ids and the manifest
    matches, a subsequent docs pass re-embeds nothing and deletes nothing."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    docs.mkdir()
    md = docs / "a.md"
    md.write_text("# a\n\nbody long enough to chunk into something.\n")
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(md)
    m.mark_indexed(md, h)
    m.save()

    collection = MagicMock()

    def fake_get(*args, **kwargs):
        if "where" in kwargs:
            return {"ids": []}
        if kwargs.get("offset", 0) == 0:
            return {  # already qualified + present
                "ids": ["docs:docs/a.md:0"],
                "documents": [""],
                "metadatas": [{"file_path": "docs/a.md"}],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    captured = _capture_embed_ids(monkeypatch)
    project = ProjectConfig(name="p", docs_dirs=[docs])
    index_docs(project, vo=MagicMock(), db=db)

    assert captured == [], f"steady state must re-embed nothing; got {captured}"
    collection.delete.assert_not_called()

def test_index_single_doc_qualifies_chunk_id_with_docs_dir_basename(tmp_path, monkeypatch):
    """index_single_doc (add-document) must emit the SAME source-root-qualified
    ids index_docs does, so add + reindex don't double-store the same file."""
    from vecs.indexer import index_single_doc
    from vecs.config import VecsConfig
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    md = docs / "sub" / "note.md"
    md.write_text("# Note\n\nBody text long enough to chunk into something.\n")

    cfg = VecsConfig(path=tmp_path / "config.yaml")
    cfg.projects["p"] = ProjectConfig(name="p", docs_dirs=[docs])
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    captured = _capture_chunks_via_index_collection(monkeypatch)
    index_single_doc("p", md)

    assert captured["chunks"]
    for c in captured["chunks"]:
        assert c["id"].startswith("docs:docs/sub/note.md:")
        assert c["metadata"]["file_path"] == "docs/sub/note.md"

def test_index_single_doc_qualifies_by_owning_root_not_docs_dirs0(tmp_path, monkeypatch):
    """index_single_doc must qualify by the file's OWN source root (matching
    index_docs), not blindly by docs_dirs[0]. A file under docs_dirs[1] must get
    docs:{docs_dirs[1].name}/... -- not raise relative_to(docs_dirs[0])."""
    from vecs.indexer import index_single_doc
    from vecs.config import VecsConfig
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    d0 = tmp_path / "docs0"
    d0.mkdir()
    d1 = tmp_path / "docs1"
    d1.mkdir()
    md = d1 / "note.md"
    md.write_text("# Note\n\nBody text long enough to chunk into something.\n")

    cfg = VecsConfig(path=tmp_path / "config.yaml")
    cfg.projects["p"] = ProjectConfig(name="p", docs_dirs=[d0, d1])
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    captured = _capture_chunks_via_index_collection(monkeypatch)
    index_single_doc("p", md)

    assert captured["chunks"]
    for c in captured["chunks"]:
        assert c["id"].startswith("docs:docs1/note.md:")
        assert c["metadata"]["file_path"] == "docs1/note.md"

def test_index_single_doc_qualifies_inrepo_md_by_code_dir_root(tmp_path, monkeypatch):
    """add-document of an in-repo .md under a code_dir qualifies by the code_dir
    basename (so it agrees with a full reindex's _docs_sources qualification)."""
    from vecs.indexer import index_single_doc
    from vecs.config import VecsConfig
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    docs.mkdir()
    code = tmp_path / "client-uk"
    code.mkdir()
    md = code / "README.md"
    md.write_text("# r\n\nBody text long enough to chunk into something.\n")

    cfg = VecsConfig(path=tmp_path / "config.yaml")
    cfg.projects["p"] = ProjectConfig(
        name="p", docs_dirs=[docs], code_dirs=[CodeDir(path=code, extensions={".cs"})]
    )
    monkeypatch.setattr("vecs.indexer.load_config", lambda: cfg)
    monkeypatch.setattr("vecs.indexer.get_voyage_client", lambda: MagicMock())
    db, _collection = _make_index_db(tmp_path)
    monkeypatch.setattr("vecs.indexer.get_chromadb_client", lambda: db)

    captured = _capture_chunks_via_index_collection(monkeypatch)
    index_single_doc("p", md)

    assert captured["chunks"]
    for c in captured["chunks"]:
        assert c["id"].startswith("docs:client-uk/README.md:")
        assert c["metadata"]["file_path"] == "client-uk/README.md"


# --- Inc: hidden (dot) dirs skipped in docs scan + swept from -docs (option-2) ---

def test_docs_sources_skips_hidden_dirs(tmp_path):
    """_docs_sources drops files under hidden dirs in BOTH a docs_dir
    (.research/) and in-repo .md under a code_dir (.claude/)."""
    from vecs.indexer import _docs_sources
    docs = tmp_path / "docs"
    (docs / ".research").mkdir(parents=True)
    (docs / "real").mkdir(parents=True)
    (docs / "real" / "a.md").write_text("# a\n\nbody.\n")
    (docs / ".research" / "notes.md").write_text("# n\n\nbody.\n")
    code = tmp_path / "client-uk"
    (code / ".claude").mkdir(parents=True)
    (code / "Assets").mkdir(parents=True)
    (code / "Assets" / "B.md").write_text("# b\n\nbody.\n")
    (code / ".claude" / "skill.md").write_text("# s\n\nbody.\n")
    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code, extensions={".cs"})],
        docs_dirs=[docs],
    )
    names = {f.name for _r, f in _docs_sources(project)}
    assert names == {"a.md", "B.md"}  # .research/notes.md + .claude/skill.md dropped


def test_index_docs_sweeps_now_hidden_chunk_under_valid_root(tmp_path, monkeypatch):
    """A -docs chunk under a still-valid root but now under a hidden dir (no
    longer in _docs_sources) is swept on reindex, even with nothing new; the
    in-scope chunk survives. (Closes the gap: _partition_docs_by_root keeps it
    -- valid root -- and the 1.5a deleted-source sweep keeps it -- file present.)"""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"
    (code / ".claude" / "commands").mkdir(parents=True)
    (code / "Assets").mkdir(parents=True)
    keep = code / "Assets" / "GUIDE.md"
    keep.write_text("# guide\n\nbody long enough.\n")
    (code / ".claude" / "commands" / "create-pr.md").write_text("# pr\n\nbody.\n")

    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(keep)
    m.mark_indexed(keep, h)
    m.save()

    hidden_id = "docs:client-uk/.claude/commands/create-pr.md:0"
    keep_id = "docs:client-uk/Assets/GUIDE.md:0"
    collection = MagicMock()

    def fake_get(*a, **k):
        if "where" in k:
            return {"ids": []}
        if k.get("offset", 0) == 0:
            return {
                "ids": [hidden_id, keep_id],
                "documents": ["", ""],
                "metadatas": [
                    {"file_path": "client-uk/.claude/commands/create-pr.md"},
                    {"file_path": "client-uk/Assets/GUIDE.md"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    bm25_deletes: list = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda proj, suffix, ids: bm25_deletes.append((proj, suffix, list(ids))),
    )
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **k: None)

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".cs"})])
    index_docs(project, vo=MagicMock(), db=db)

    deleted = []
    for c in collection.delete.call_args_list:
        ids = c.kwargs.get("ids") or (c.args[0] if c.args else [])
        deleted.extend(ids)
    assert hidden_id in deleted       # now-hidden chunk swept
    assert keep_id not in deleted     # in-scope chunk survives
    assert ("p", "docs", [hidden_id]) in bm25_deletes


# --- out-of-scope -docs sweep: safety guards (review F1/F2/F4/F5) ---------

def test_out_of_scope_sweep_keeps_missing_root_and_gone_file(tmp_path, monkeypatch):
    """The out-of-scope sweep must only delete chunks whose source file EXISTS
    on disk under a present, unique-basename root (present-but-out-of-scope).
    A transiently-MISSING root's chunks (F1) and a GONE file's chunks (F2 /
    1.5a's job) are KEPT -- an orphan is safer than deleting live data. The
    hidden chunk under the present root is still swept."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    present_code = tmp_path / "client-uk"
    (present_code / "Assets").mkdir(parents=True)
    (present_code / ".claude").mkdir(parents=True)
    guide = present_code / "Assets" / "GUIDE.md"
    guide.write_text("# g\n\nbody long enough.\n")
    (present_code / ".claude" / "x.md").write_text("# x\n\nbody.\n")
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(guide)
    m.mark_indexed(guide, h)
    m.save()

    missing = tmp_path / "server-uk"  # configured but NOT created on disk

    guide_id = "docs:client-uk/Assets/GUIDE.md:0"
    hidden_id = "docs:client-uk/.claude/x.md:0"
    missing_root_id = "docs:server-uk/doc/REFERENCE.md:0"
    gone_file_id = "docs:client-uk/Assets/GONE.md:0"  # valid present root, file absent
    collection = MagicMock()

    def fake_get(*a, **k):
        if "where" in k:
            return {"ids": []}
        if k.get("offset", 0) == 0:
            return {
                "ids": [guide_id, hidden_id, missing_root_id, gone_file_id],
                "documents": ["", "", "", ""],
                "metadatas": [
                    {"file_path": "client-uk/Assets/GUIDE.md"},
                    {"file_path": "client-uk/.claude/x.md"},
                    {"file_path": "server-uk/doc/REFERENCE.md"},
                    {"file_path": "client-uk/Assets/GONE.md"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a: None)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **k: None)

    project = ProjectConfig(name="p", code_dirs=[
        CodeDir(path=present_code, extensions={".cs"}),
        CodeDir(path=missing, extensions={".go"}),
    ])
    index_docs(project, vo=MagicMock(), db=db)

    deleted = []
    for c in collection.delete.call_args_list:
        deleted.extend(c.kwargs.get("ids") or (c.args[0] if c.args else []))
    assert hidden_id in deleted              # present-root hidden file -> swept
    assert guide_id not in deleted           # in-scope -> survives
    assert missing_root_id not in deleted    # F1: missing root -> kept
    assert gone_file_id not in deleted       # F2: gone file -> kept (1.5a's job)


def test_index_docs_empty_sources_does_not_wipe_populated_collection(tmp_path, monkeypatch):
    """F4: a populated -docs collection whose scope collapses to empty
    (_docs_sources == []) must NOT be swept -- the `if not sources: return 0`
    guard precedes the out-of-scope sweep."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    code = tmp_path / "client-uk"  # exists but has NO .md -> sources empty
    code.mkdir()
    collection = MagicMock()

    def fake_get(*a, **k):
        if "where" in k:
            return {"ids": []}
        if k.get("offset", 0) == 0:
            return {
                "ids": ["docs:client-uk/a.md:0"],
                "documents": [""],
                "metadatas": [{"file_path": "client-uk/a.md"}],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection

    project = ProjectConfig(name="p", code_dirs=[CodeDir(path=code, extensions={".cs"})])
    n = index_docs(project, vo=MagicMock(), db=db)
    assert n == 0
    collection.delete.assert_not_called()


def test_out_of_scope_sweep_docs_dir_rooted_hidden_subdir(tmp_path, monkeypatch):
    """F5: the sweep reclaims a hidden-subdir chunk rooted at a DOCS_DIR (not
    just a code_dir), end-to-end through index_docs."""
    monkeypatch.setattr("vecs.indexer.MANIFESTS_DIR", tmp_path / "manifests")

    docs = tmp_path / "docs"
    (docs / ".research").mkdir(parents=True)
    (docs / "real").mkdir(parents=True)
    real = docs / "real" / "a.md"
    real.write_text("# a\n\nbody long enough.\n")
    (docs / ".research" / "notes.md").write_text("# n\n\nbody.\n")
    m = Manifest("p", manifests_dir=tmp_path / "manifests")
    _, h = m.needs_indexing(real)
    m.mark_indexed(real, h)
    m.save()

    keep_id = "docs:docs/real/a.md:0"
    hidden_id = "docs:docs/.research/notes.md:0"
    collection = MagicMock()

    def fake_get(*a, **k):
        if "where" in k:
            return {"ids": []}
        if k.get("offset", 0) == 0:
            return {
                "ids": [keep_id, hidden_id],
                "documents": ["", ""],
                "metadatas": [
                    {"file_path": "docs/real/a.md"},
                    {"file_path": "docs/.research/notes.md"},
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    collection.get.side_effect = fake_get
    db = MagicMock()
    db.get_or_create_collection.return_value = collection
    bm25_deletes = []
    monkeypatch.setattr(
        "vecs.indexer._delete_ids_from_bm25",
        lambda proj, suffix, ids: bm25_deletes.append((proj, suffix, list(ids))),
    )
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **k: None)

    project = ProjectConfig(name="p", docs_dirs=[docs])
    index_docs(project, vo=MagicMock(), db=db)

    deleted = []
    for c in collection.delete.call_args_list:
        deleted.extend(c.kwargs.get("ids") or (c.args[0] if c.args else []))
    assert hidden_id in deleted
    assert keep_id not in deleted
    assert ("p", "docs", [hidden_id]) in bm25_deletes
