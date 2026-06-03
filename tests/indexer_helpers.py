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


class FakeEmbedResult:
    """Minimal fake for voyageai embed result."""
    def __init__(self, n: int):
        self.embeddings = [[0.1] * 128 for _ in range(n)]
        self.usage = MagicMock(total_tokens=n * 50)

def _embedded_texts(call):
    """Extract the `texts` arg from a vo.embed(...) call (positional or kw)."""
    if call.args:
        return call.args[0]
    return call.kwargs.get("texts")

def _make_index_db(tmp_path):
    """Build a chromadb-like mock that records upserted ids per file_path metadata."""
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    db = MagicMock()
    db.get_or_create_collection.return_value = collection
    return db, collection

def _capture_files(monkeypatch):
    """Patch _index_collection to capture files_to_process and skip embed work."""
    captured: dict = {}

    def fake_index_collection(*, files_to_process, file_hashes, manifest, **kw):
        captured["files"] = list(files_to_process)
        captured["file_hashes"] = dict(file_hashes)
        # Mark all files as indexed so manifest persists -- mirrors successful path
        for f in files_to_process:
            manifest.mark_indexed(f, file_hashes[f])
        manifest.save()
        return len(files_to_process)

    monkeypatch.setattr("vecs.indexer._index_collection", fake_index_collection)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    return captured

def _git_init_commit(repo: Path, relfile: str, content: str) -> str:
    """Init a git repo with one committed file; return the HEAD sha."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    fp = repo / relfile
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"],
        cwd=repo, check=True,
    )
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()

def _capture_chunks_via_index_collection(monkeypatch):
    """Patch _index_collection to capture the chunks passed in (code/docs path)."""
    captured: dict = {}

    def fake_index_collection(*, chunks, manifest, files_to_process, file_hashes, **kw):
        captured["chunks"] = chunks
        for f in files_to_process:
            manifest.mark_indexed(f, file_hashes[f])
        manifest.save()
        return len(chunks)

    monkeypatch.setattr("vecs.indexer._index_collection", fake_index_collection)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    return captured

class _StatefulDocsChroma:
    """Minimal stateful chroma stand-in: stores upserts, answers get(where=
    {file_path}) and paginated get, supports delete. Lets a test drive the REAL
    _index_collection cleanup path (_delete_stale_chunks_after_embed)."""

    def __init__(self):
        self.store: dict[str, dict] = {}  # id -> {"document", "metadata"}

    def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
        for i, cid in enumerate(ids):
            self.store[cid] = {
                "document": (documents or [""] * len(ids))[i],
                "metadata": (metadatas or [{}] * len(ids))[i] or {},
            }

    def get(self, ids=None, where=None, limit=None, offset=None, include=None):
        items = list(self.store.items())
        if where:
            (k, v), = where.items()
            items = [(cid, e) for cid, e in items if (e["metadata"] or {}).get(k) == v]
        if offset is not None or limit is not None:
            off = offset or 0
            items = items[off: off + limit if limit is not None else None]
        return {
            "ids": [cid for cid, _ in items],
            "documents": [e["document"] for _, e in items],
            "metadatas": [e["metadata"] for _, e in items],
        }

    def delete(self, ids=None):
        for cid in ids or []:
            self.store.pop(cid, None)

def _capture_embed_ids(monkeypatch):
    """Fake _embed_and_store: record every chunk id it would store, return them."""
    captured: list[str] = []

    def fake_embed(chunks, collection, model, vo, batcher=None, cache=None):
        ids = [c["id"] for c in chunks]
        captured.extend(ids)
        return ids

    monkeypatch.setattr("vecs.indexer._embed_and_store", fake_embed)
    monkeypatch.setattr("vecs.indexer._sync_bm25", lambda *a, **kw: None)
    monkeypatch.setattr("vecs.indexer._delete_ids_from_bm25", lambda *a, **kw: None)
    return captured

def _seed_manifest_with_doc_code_session(tmp_path, doc_f, code_f, sess_f):
    manifest = Manifest("p", manifests_dir=tmp_path / "manifests")
    manifest.mark_indexed(doc_f, "hdoc")
    manifest.mark_indexed(code_f, "hcode")
    manifest.mark_session_indexed(sess_f, byte_offset=2, chunk_count=1)
    manifest.save()

def _remodel_fixture(tmp_path):
    """Build (project, doc_f, code_f, sess_f) sharing one manifest namespace."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    doc_f = docs_dir / "a.md"
    doc_f.write_text("doc body")
    code_f = code_dir / "m.py"
    code_f.write_text("print(1)")
    sess_f = tmp_path / "sess.jsonl"
    sess_f.write_text("{}")
    project = ProjectConfig(
        name="p",
        code_dirs=[CodeDir(path=code_dir, extensions={".py"})],
        docs_dirs=[docs_dir],
    )
    return project, doc_f, code_f, sess_f

class _FakeChromaCollection:
    """Minimal in-memory chroma collection: supports the get/delete/count the
    orphan sweeps use (paginated get, where={session_id}, id delete)."""

    def __init__(self, rows):  # rows: list[(id, metadata-dict)]
        self._rows = {cid: dict(meta) for cid, meta in rows}

    def get(self, limit=None, offset=0, where=None, include=None):
        items = list(self._rows.items())
        if where is not None:
            key, val = next(iter(where.items()))
            items = [(c, m) for c, m in items if m.get(key) == val]
        elif limit is not None:
            items = items[offset:offset + limit]
        # Honor `include` like real chromadb: only the requested keys are
        # returned (ids always present). include=None defaults to all.
        inc = include if include is not None else ["metadatas", "documents"]
        out = {"ids": [c for c, _ in items]}
        if "metadatas" in inc:
            out["metadatas"] = [m for _, m in items]
        if "documents" in inc:
            out["documents"] = ["" for _ in items]
        return out

    def delete(self, ids=None):
        for cid in ids or []:
            self._rows.pop(cid, None)

    def count(self):
        return len(self._rows)

class _FakeDB:
    def __init__(self, collections):
        self._cols = dict(collections)

    def get_collection(self, name):
        if name not in self._cols:
            raise ValueError(f"no such collection: {name}")
        return self._cols[name]

    def get_or_create_collection(self, name):
        return self._cols.setdefault(name, _FakeChromaCollection([]))
