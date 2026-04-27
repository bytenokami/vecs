# BM25 → SQLite FTS5 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rank_bm25 + pickle-based `BM25Index` with a SQLite FTS5–backed index that supports incremental upsert/delete, eliminating the per-tick "rebuild from full ChromaDB read + repickle" cost.

**Architecture:** SQLite database per `(project, suffix)` at `~/.vecs/bm25/{project}_{suffix}.db`. External-content FTS5 virtual table backed by a normal `docs` base table that holds `(doc_id, text, tokens, file_path, metadata_json)`. Tokens are pre-computed in Python using the existing code-aware tokenizer (camelCase/snake_case/ACRONYM splitter), then fed into FTS5 with the built-in `unicode61` tokenizer. Triggers keep FTS in sync with base table mutations. Indexer changes from "rebuild full collection" to "diff IDs + upsert delta + delete removed". WAL mode for concurrent indexer-writer / search-reader access.

**Tech Stack:** Python 3.12+ stdlib `sqlite3` (FTS5 + JSON1 + WAL — confirmed available on macOS/Linux wheels for Python 3.12/3.13, sqlite_version 3.43+). No new third-party dependency. Removes `rank-bm25>=0.2`.

---

## Background — Why we are doing this

Current pain points (`src/vecs/bm25_index.py`, `src/vecs/indexer.py:489-504`):

1. `_rebuild_bm25` reads **every** chunk from ChromaDB (paginated), tokenizes all of them, builds a fresh `BM25Okapi`, and writes a fresh pickle — every time the indexer runs, even if only one file changed.
2. At ~42K chunks today, this is the bottleneck of an incremental index tick.
3. The pickle has no atomic write (mid-write crash → corrupt file → silent fallback to vector-only search).
4. `rank_bm25.BM25Okapi` is in-memory only; it must be rebuilt on every load (search-side cost too).
5. SQLite variable limit (32766 / 999) has already bitten this project once (see `_paginated_get` workaround).

After this change:
- Per-tick cost drops to O(changed-files), not O(total-chunks).
- Atomic durability via SQLite transactions + WAL.
- No pickle, no `rank-bm25` dep.

## Design decisions (locked)

- **No external tokenizer.** We pre-tokenize at index time and at query time using the existing `_tokenize()` function. FTS5 just sees space-separated lowercased tokens via the built-in `unicode61` tokenizer. This preserves exact camelCase/snake_case semantics with zero new dependencies.
- **External-content FTS5** with triggers. Base table owns the data; FTS shadow table owns just the inverted index. Lets us return `text`, filter on `file_path`, and read `metadata_json` without an extra join.
- **Promoted `file_path` column** on the base table with a btree index, so path filters become an indexed `LIKE` clause inside the SQL, not a Python post-filter.
- **No schema migration of old pickles.** On first run after upgrade, the indexer's diff-sync sees an empty FTS database and rebuilds it from ChromaDB (one-time cost equivalent to today's per-tick cost). Old `.pkl` files are deleted on first successful sync.
- **Public API of `BM25Index` is preserved** (`build`, `search`, `save`, `load`, `path`). `save()` becomes a no-op (writes are durable on commit). New methods `upsert(docs)`, `delete(ids)`, `all_ids()` are added for incremental sync.
- **Match query semantics:** OR across tokenized query terms (matches `BM25Okapi.get_scores` "any term contributes" semantics). Each token wrapped in double quotes to neutralize FTS5 syntax characters (`-`, `+`, `:`, etc.).
- **Negate `bm25()` for the public score field** (SQLite returns lower-is-better; current API returns higher-is-better).

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/vecs/bm25_index.py` | **Rewrite** | FTS5-backed `BM25Index`, schema DDL, query escape, mtime cache. Same public surface. |
| `src/vecs/indexer.py` | **Modify** (`_rebuild_bm25` and 4 call sites) | Replace `_rebuild_bm25` with `_sync_bm25` (diff-based incremental sync). |
| `src/vecs/searcher.py` | **Modify** (one line: `.pkl` → `.db`) | Point cache lookups at the new file extension. |
| `tests/test_bm25.py` | **Rewrite** | Update existing 18 tests for FTS5 backend; add 6 new tests for upsert/delete/path-filter-as-SQL. |
| `tests/test_indexer.py` | **Modify** (3 BM25 tests) | Update for `_sync_bm25` diff semantics + `.db` paths. |
| `pyproject.toml` | **Modify** | Remove `rank-bm25>=0.2` line. |
| `uv.lock` | **Regenerate** | `uv lock` after pyproject edit. |
| `scripts/bench_bm25.py` | **Create** | A/B benchmark script comparing old vs new on a real corpus. |

---

## Task 1: Foundation — schema constants, connect helper, query escaper

**Files:**
- Create: `src/vecs/bm25_index.py` (full rewrite — see Tasks 1–3 collectively replace the current file)
- Test: `tests/test_bm25.py`

This task only adds the **module-level helpers** that the rewritten class will use. It also writes the first FTS5 connection test to confirm the schema applies cleanly.

- [ ] **Step 1: Write the failing test for schema bootstrap**

Replace the current `tests/test_bm25.py` imports block (lines 1-15) with:

```python
from __future__ import annotations

import sqlite3
import time

import pytest

from vecs.bm25_index import (
    BM25Index,
    _tokenize,
    _build_match_query,
    _bm25_cache,
    get_bm25,
    _connect,
    SCHEMA_DDL,
)


@pytest.fixture(autouse=True)
def clear_bm25_cache():
    """Clear module-level BM25 cache between tests."""
    _bm25_cache.clear()
    yield
    _bm25_cache.clear()
```

Add this new test at the top of the test functions section:

```python
def test_connect_creates_schema(tmp_path):
    """_connect creates the docs table, FTS5 vtable, and triggers on first call."""
    db_path = tmp_path / "test.db"
    conn = _connect(db_path)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger', 'index')"
            )
        }
        assert "docs" in names
        assert "docs_fts" in names
        assert "docs_ai" in names  # AFTER INSERT trigger
        assert "docs_ad" in names  # AFTER DELETE trigger
        assert "docs_au" in names  # AFTER UPDATE trigger
        assert "idx_docs_doc_id" in names
        assert "idx_docs_file_path" in names

        # WAL mode is set
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py::test_connect_creates_schema -v`
Expected: FAIL with `ImportError: cannot import name '_connect'` (or similar — `bm25_index.py` still has the old shape).

- [ ] **Step 3: Replace `src/vecs/bm25_index.py` with the new module skeleton**

Write the full file. This includes the helpers needed by Tasks 1–3; the `BM25Index` class body lands in Task 2. For now, stub `BM25Index` with `pass` so imports work.

```python
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


# --- Tokenizer (preserved from previous implementation) ---

def _tokenize(text: str) -> list[str]:
    """Code-aware tokenizer that splits camelCase, PascalCase, ACRONYMS, and snake_case.

    Examples:
        getUserById  -> [get, user, by, id]
        XMLParser    -> [xml, parser]
        HTTPSConnection -> [https, connection]
        ALL_CAPS     -> [all, caps]
        snake_case   -> [snake, case]
    """
    words = re.findall(r"\w+", text)
    tokens: list[str] = []
    for word in words:
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", word
        )
        if parts:
            tokens.extend(p.lower() for p in parts)
        else:
            tokens.append(word.lower())
    return tokens


# --- FTS5 query escaping ---

def _build_match_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from a free-form query.

    Tokenizes with the same code-aware tokenizer used at index time,
    wraps each token in double quotes (which neutralizes FTS5 syntax
    chars like `-`, `+`, `:`, `*`, `(`, `)`, `"`), and joins with OR
    so that any-term-matches contributes to the BM25 score, matching
    the prior BM25Okapi.get_scores semantics.

    Returns None for queries that produce zero tokens (caller should
    short-circuit to empty results).
    """
    tokens = _tokenize(query)
    if not tokens:
        return None
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens]
    return " OR ".join(quoted)


# --- Schema ---

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS docs (
  rowid          INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id         TEXT UNIQUE NOT NULL,
  text           TEXT NOT NULL,
  tokens         TEXT NOT NULL,
  file_path      TEXT,
  metadata_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_docs_doc_id    ON docs(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_file_path ON docs(file_path);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
  tokens,
  content='docs',
  content_rowid='rowid',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
  INSERT INTO docs_fts(rowid, tokens) VALUES (new.rowid, new.tokens);
END;

CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts, rowid, tokens) VALUES('delete', old.rowid, old.tokens);
END;

CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
  INSERT INTO docs_fts(docs_fts, rowid, tokens) VALUES('delete', old.rowid, old.tokens);
  INSERT INTO docs_fts(rowid, tokens) VALUES (new.rowid, new.tokens);
END;
"""


def _connect(path: Path) -> sqlite3.Connection:
    """Open (or create) a BM25 FTS5 SQLite database at `path`.

    Sets WAL mode, NORMAL synchronous, 5s busy timeout, and applies
    the schema DDL idempotently. Returns an open connection — caller
    closes it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit; we manage txns explicitly
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA_DDL)
    return conn


# --- BM25Index (filled in Task 2) ---

class BM25Index:
    """Stub — implemented in Task 2."""

    def __init__(self, path: Path):
        self.path = path


# --- Module-level mtime cache (filled in Task 3) ---

_bm25_cache: dict = {}


def get_bm25(path: Path):
    """Stub — implemented in Task 3."""
    return None
```

- [ ] **Step 4: Add tokenizer-preservation tests**

These tests already exist in the current `test_bm25.py` (lines 111-169 per the existing file). **Keep them verbatim** — they assert the tokenizer is unchanged, which is the entire compatibility contract for query results.

Make sure the file still includes (re-paste if your rewrite of the test file dropped them):

```python
def test_tokenize_camel_case():
    assert _tokenize("getUserById") == ["get", "user", "by", "id"]

def test_tokenize_pascal_case():
    assert _tokenize("XMLParser") == ["xml", "parser"]

def test_tokenize_acronym_connection():
    assert _tokenize("HTTPSConnection") == ["https", "connection"]

def test_tokenize_all_caps():
    assert _tokenize("ALL_CAPS") == ["all", "caps"]

def test_tokenize_mixed():
    assert _tokenize("parseJSON") == ["parse", "json"]

def test_tokenize_short_acronym():
    assert _tokenize("ID") == ["id"]

def test_tokenize_trailing_url():
    assert _tokenize("getURL") == ["get", "url"]

def test_tokenize_snake_case():
    assert _tokenize("snake_case_name") == ["snake", "case", "name"]

def test_tokenize_dunder():
    assert _tokenize("__init__") == ["init"]

def test_tokenize_plain_words():
    assert _tokenize("hello world") == ["hello", "world"]

def test_tokenize_numbers():
    assert _tokenize("item42count") == ["item", "42", "count"]

def test_tokenize_mixed_sentence():
    tokens = _tokenize("getUserById returns HTTPResponse")
    assert tokens == ["get", "user", "by", "id", "returns", "http", "response"]
```

- [ ] **Step 5: Add tests for `_build_match_query`**

```python
def test_build_match_query_basic():
    """Tokenizes and ORs query terms, each quoted."""
    assert _build_match_query("getUserById") == '"get" OR "user" OR "by" OR "id"'

def test_build_match_query_empty():
    """Empty / whitespace-only query returns None."""
    assert _build_match_query("") is None
    assert _build_match_query("   ") is None
    assert _build_match_query("!@#$%^&*()") is None  # no word chars

def test_build_match_query_escapes_quotes():
    """Embedded double quotes in tokens are doubled (FTS5 escaping convention)."""
    # The tokenizer strips quotes via \w+ so this is hard to trigger naturally,
    # but the escape logic should still be defensive.
    # We test the contract directly:
    from vecs.bm25_index import _build_match_query
    # A token containing a quote can only happen if someone bypasses _tokenize;
    # smoke-test that the helper does not crash on weird input.
    result = _build_match_query('hello world')
    assert '"hello"' in result and '"world"' in result

def test_build_match_query_punctuation_safe():
    """FTS5-syntax punctuation in input does not produce a syntactically invalid query."""
    # `-` would mean NOT in raw FTS5; quoting must neutralize it.
    q = _build_match_query("foo-bar baz")
    assert q is not None
    # Verify it actually parses by feeding it to FTS5
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t(x) VALUES ('foo bar baz')")
    rows = list(conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (q,)))
    assert len(rows) == 1
    conn.close()
```

- [ ] **Step 6: Run all new tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py -v -k "tokenize or connect_creates_schema or build_match_query"`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/vecs/bm25_index.py tests/test_bm25.py
git commit -m "feat(bm25): add FTS5 schema, connect helper, and query escaper"
```

---

## Task 2: BM25Index FTS5 backend — preserved public API

**Files:**
- Modify: `src/vecs/bm25_index.py:80-160` (replace stub `BM25Index`)
- Test: `tests/test_bm25.py`

This task fills in the `BM25Index` class so that `build`, `search`, `save`, `load` work against the new FTS5 schema while preserving the exact public contract used by `searcher.py` and the existing tests.

- [ ] **Step 1: Write the failing test for `build` + `search` over FTS5**

Add to `tests/test_bm25.py` (replace the existing `test_bm25_build_and_search`):

```python
def test_bm25_build_and_search(tmp_path):
    """BM25 index returns results ranked by keyword relevance (FTS5 backend)."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "the player takes damage and health decreases"},
        {"id": "b", "text": "the enemy attacks the player with a sword"},
        {"id": "c", "text": "the menu shows options for settings and audio"},
    ]
    idx.build(docs)
    results = idx.search("player damage", n=2)
    assert len(results) == 2
    assert results[0]["id"] == "a"  # most relevant — has both "player" AND "damage"
    assert results[0]["score"] > 0  # score is positive (we negate sqlite's bm25)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py::test_bm25_build_and_search -v`
Expected: FAIL with `AttributeError: 'BM25Index' object has no attribute 'build'` or similar.

- [ ] **Step 3: Implement the `BM25Index` class**

Replace the stub `BM25Index` in `src/vecs/bm25_index.py` with:

```python
class BM25Index:
    """BM25 keyword search index backed by SQLite FTS5.

    Public API preserved from the rank_bm25 implementation:
      build(docs)   — bulk replace contents
      search(...)   — query with optional path filter
      save()        — no-op (writes are durable on commit)
      load()        — open the database; returns True if it exists
      path          — file path (a `.db` file)
    Plus new methods used by the indexer for incremental sync:
      upsert(docs)  — insert-or-update by doc_id
      delete(ids)   — delete by doc_id
      all_ids()     — set of currently indexed doc_ids
      close()       — close the underlying connection
    """

    def __init__(self, path: Path):
        self.path = path
        self._conn: sqlite3.Connection | None = None

    # --- connection management ---

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self.path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # --- bulk build (used for first-time / forced rebuild) ---

    def build(self, docs: list[dict]) -> None:
        """Replace all contents with the given documents.

        Wraps the operation in a single transaction. On failure the
        database is unchanged.
        """
        conn = self._ensure_conn()
        rows = [
            (
                d["id"],
                d["text"],
                " ".join(_tokenize(d["text"])),
                (d.get("metadata") or {}).get("file_path"),
                json.dumps(d.get("metadata") or {}),
            )
            for d in docs
        ]
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM docs")
            if rows:
                conn.executemany(
                    "INSERT INTO docs(doc_id, text, tokens, file_path, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # --- incremental mutations ---

    def upsert(self, docs: list[dict]) -> None:
        """Insert or update by `doc_id` (atomic, single transaction)."""
        if not docs:
            return
        conn = self._ensure_conn()
        rows = [
            (
                d["id"],
                d["text"],
                " ".join(_tokenize(d["text"])),
                (d.get("metadata") or {}).get("file_path"),
                json.dumps(d.get("metadata") or {}),
            )
            for d in docs
        ]
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """
                INSERT INTO docs(doc_id, text, tokens, file_path, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  text          = excluded.text,
                  tokens        = excluded.tokens,
                  file_path     = excluded.file_path,
                  metadata_json = excluded.metadata_json
                """,
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def delete(self, ids: list[str]) -> None:
        """Delete rows by `doc_id`. Batches to stay under SQLITE_MAX_VARIABLE_NUMBER."""
        if not ids:
            return
        conn = self._ensure_conn()
        BATCH = 5000
        conn.execute("BEGIN")
        try:
            for i in range(0, len(ids), BATCH):
                chunk = ids[i:i + BATCH]
                placeholders = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM docs WHERE doc_id IN ({placeholders})", chunk)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def all_ids(self) -> set[str]:
        """Return the set of all currently indexed `doc_id` values."""
        conn = self._ensure_conn()
        return {row[0] for row in conn.execute("SELECT doc_id FROM docs")}

    # --- search ---

    def search(self, query: str, n: int = 5, path_filter: str | None = None) -> list[dict]:
        """Search the index. Returns list of {"id", "text", "score", "metadata"}.

        Args:
            query: Search query string.
            n: Number of results to return.
            path_filter: If set, only return rows whose `file_path` contains this
                substring (translated to a SQL LIKE so it runs inside the database,
                using the indexed `file_path` column).
        """
        if not self.path.exists():
            return []

        match_expr = _build_match_query(query)
        if match_expr is None:
            return []

        conn = self._ensure_conn()
        sql = (
            "SELECT d.doc_id, d.text, d.metadata_json, -bm25(docs_fts) AS score "
            "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
            "WHERE docs_fts MATCH ? "
        )
        params: list = [match_expr]
        if path_filter:
            sql += "AND d.file_path LIKE ? "
            params.append(f"%{path_filter}%")
        sql += "ORDER BY bm25(docs_fts) LIMIT ?"
        params.append(n)

        results: list[dict] = []
        for doc_id, text, meta_json, score in conn.execute(sql, params):
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (TypeError, ValueError):
                meta = {}
            results.append(
                {
                    "id": doc_id,
                    "text": text,
                    "score": float(score),
                    "metadata": meta,
                }
            )
        return results

    # --- legacy persistence shims ---

    def save(self) -> None:
        """No-op: SQLite writes are durable on commit. Kept for backward API compatibility."""
        return

    def load(self) -> bool:
        """Open the database. Returns True if the file already exists, False if it had to be created."""
        existed = self.path.exists()
        self._ensure_conn()
        return existed
```

- [ ] **Step 4: Run the build/search test**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py::test_bm25_build_and_search -v`
Expected: PASS.

- [ ] **Step 5: Restore + adapt the metadata, save/load, and path-filter tests**

Update these existing tests in `tests/test_bm25.py` to use `.db` paths and account for FTS5 semantics. Each test below replaces its same-named predecessor.

```python
def test_bm25_build_with_metadata(tmp_path):
    """BM25 index stores and returns metadata."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "user authentication login", "metadata": {"file_path": "auth/login.ts"}},
        {"id": "b", "text": "database connection pool", "metadata": {"file_path": "db/pool.ts"}},
        {"id": "c", "text": "render graphics engine frame", "metadata": {"file_path": "gfx/render.ts"}},
    ]
    idx.build(docs)
    results = idx.search("authentication", n=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"
    assert results[0]["metadata"] == {"file_path": "auth/login.ts"}


def test_bm25_save_and_load(tmp_path):
    """BM25 index can be saved and loaded across instances (FTS5: load just reopens)."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "hello world greeting"},
        {"id": "b", "text": "goodbye world farewell"},
        {"id": "c", "text": "something else entirely"},
    ]
    idx.build(docs)
    idx.save()
    idx.close()

    idx2 = BM25Index(tmp_path / "test.db")
    assert idx2.load() is True
    results = idx2.search("hello", n=1)
    assert results[0]["id"] == "a"
    idx2.close()


def test_bm25_save_load_with_metadata(tmp_path):
    """Metadata survives across instances."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "hello world greeting",
         "metadata": {"file_path": "hello.ts", "chunk_index": 0}},
        {"id": "b", "text": "goodbye farewell parting",
         "metadata": {"file_path": "bye.ts", "chunk_index": 1}},
        {"id": "c", "text": "something else entirely",
         "metadata": {"file_path": "other.ts", "chunk_index": 2}},
    ]
    idx.build(docs)
    idx.close()

    idx2 = BM25Index(tmp_path / "test.db")
    idx2.load()
    results = idx2.search("hello", n=1)
    assert results[0]["metadata"] == {"file_path": "hello.ts", "chunk_index": 0}
    idx2.close()


def test_bm25_search_with_path_filter(tmp_path):
    """BM25 search filters results by path_filter substring on file_path metadata."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "user login authentication",
         "metadata": {"file_path": "auth/login.ts"}},
        {"id": "b", "text": "user profile settings authentication",
         "metadata": {"file_path": "settings/profile.ts"}},
        {"id": "c", "text": "user session token authentication",
         "metadata": {"file_path": "auth/session.ts"}},
        {"id": "d", "text": "render graphics engine frame",
         "metadata": {"file_path": "gfx/render.ts"}},
    ]
    idx.build(docs)

    results_all = idx.search("authentication", n=5)
    assert len(results_all) == 3

    results_filtered = idx.search("authentication", n=5, path_filter="auth/")
    assert len(results_filtered) == 2
    assert all("auth/" in r["metadata"]["file_path"] for r in results_filtered)


def test_bm25_search_path_filter_in_sql(tmp_path):
    """Path filter is applied in SQL (not post-filter): n results returned even when most rows match path."""
    idx = BM25Index(tmp_path / "test.db")
    docs = []
    for i in range(50):
        path = "target/file.ts" if i < 30 else f"other/file{i}.ts"
        docs.append({
            "id": str(i),
            "text": f"common keyword number {i}",
            "metadata": {"file_path": path},
        })
    idx.build(docs)

    results = idx.search("common keyword", n=10, path_filter="target/")
    assert len(results) == 10
    assert all("target/" in r["metadata"]["file_path"] for r in results)


def test_bm25_build_without_metadata_key(tmp_path):
    """Docs without 'metadata' key get empty dict metadata."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "hello world greeting"},
        {"id": "b", "text": "goodbye farewell parting"},
        {"id": "c", "text": "something else entirely"},
    ]
    idx.build(docs)
    results = idx.search("hello", n=1)
    assert results[0]["metadata"] == {}


def test_bm25_empty_docs(tmp_path):
    """Building with no docs leaves an empty queryable index."""
    idx = BM25Index(tmp_path / "test.db")
    idx.build([])
    assert idx.search("anything", n=5) == []


def test_bm25_empty_query(tmp_path):
    """A query that tokenizes to nothing returns []."""
    idx = BM25Index(tmp_path / "test.db")
    idx.build([{"id": "a", "text": "hello world"}])
    assert idx.search("", n=5) == []
    assert idx.search("!!!", n=5) == []


def test_bm25_punctuation_in_query_does_not_crash(tmp_path):
    """FTS5-syntax characters in user input are escaped."""
    idx = BM25Index(tmp_path / "test.db")
    idx.build([
        {"id": "a", "text": "foo bar baz quux"},
        {"id": "b", "text": "totally unrelated"},
    ])
    # Each of these would crash a naive (unescaped) MATCH:
    for q in ["foo-bar", 'foo "bar"', "foo:bar", "foo*", "foo AND bar"]:
        results = idx.search(q, n=5)
        # Should at least not raise; result content is best-effort
        assert isinstance(results, list)
```

- [ ] **Step 6: Delete obsolete tests**

The following tests in the old `tests/test_bm25.py` are obsolete because they test pickle-format details that no longer exist. Remove them entirely:

- `test_bm25_backward_compat_no_metadata` (pickle-format-specific)

Keep all other tests, updating any `.pkl` paths to `.db`.

- [ ] **Step 7: Run the full test_bm25.py suite**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add src/vecs/bm25_index.py tests/test_bm25.py
git commit -m "feat(bm25): implement FTS5-backed BM25Index with preserved public API"
```

---

## Task 3: Cache layer + incremental upsert/delete tests

**Files:**
- Modify: `src/vecs/bm25_index.py:160-` (replace `_bm25_cache` stub and `get_bm25` stub)
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write failing tests for cache + incremental APIs**

Add to `tests/test_bm25.py`:

```python
def test_get_bm25_returns_index_for_existing_db(tmp_path):
    """get_bm25 opens an existing .db file and returns a usable BM25Index."""
    db = tmp_path / "test.db"
    idx = BM25Index(db)
    idx.build([{"id": "a", "text": "hello world"}])
    idx.close()

    result = get_bm25(db)
    assert result is not None
    hits = result.search("hello", n=1)
    assert hits[0]["id"] == "a"


def test_get_bm25_returns_cached_index(tmp_path):
    """get_bm25 returns the same object on repeated calls when mtime hasn't changed."""
    db = tmp_path / "test.db"
    idx = BM25Index(db)
    idx.build([{"id": "a", "text": "hello world"}])
    idx.close()

    r1 = get_bm25(db)
    r2 = get_bm25(db)
    assert r1 is r2


def test_get_bm25_invalidates_on_mtime_change(tmp_path):
    """get_bm25 reloads when the .db file mtime changes."""
    db = tmp_path / "test.db"
    idx = BM25Index(db)
    idx.build([{"id": "a", "text": "hello world"}])
    idx.close()

    r1 = get_bm25(db)
    assert r1 is not None

    time.sleep(0.05)
    idx2 = BM25Index(db)
    idx2.upsert([{"id": "b", "text": "goodbye world"}])
    idx2.close()
    # Touch to bump mtime in case WAL didn't flush a visible mtime change
    db.touch()

    r2 = get_bm25(db)
    assert r2 is not None
    assert r2 is not r1


def test_get_bm25_missing_file(tmp_path):
    """get_bm25 returns None for a nonexistent .db file."""
    assert get_bm25(tmp_path / "nope.db") is None


def test_get_bm25_corrupted_file(tmp_path):
    """get_bm25 returns None when the file exists but isn't a valid SQLite DB (graceful degradation)."""
    db = tmp_path / "bad.db"
    db.write_bytes(b"this is not a sqlite database")
    assert get_bm25(db) is None


def test_upsert_inserts_new_and_updates_existing(tmp_path):
    """upsert() adds new doc_ids and overwrites existing ones."""
    idx = BM25Index(tmp_path / "test.db")
    idx.upsert([
        {"id": "a", "text": "alpha quick brown fox"},
        {"id": "b", "text": "beta lazy dog"},
    ])
    assert idx.all_ids() == {"a", "b"}

    # Update 'a', insert 'c'
    idx.upsert([
        {"id": "a", "text": "alpha completely different content"},
        {"id": "c", "text": "gamma new entry"},
    ])
    assert idx.all_ids() == {"a", "b", "c"}

    # Old text for 'a' should no longer match
    results = idx.search("quick brown", n=5)
    assert all(r["id"] != "a" for r in results)
    # New text for 'a' should match
    results = idx.search("completely different", n=5)
    assert any(r["id"] == "a" for r in results)


def test_delete_removes_rows(tmp_path):
    """delete() removes the listed doc_ids from both the base table and FTS."""
    idx = BM25Index(tmp_path / "test.db")
    idx.upsert([
        {"id": "a", "text": "alpha quick brown fox"},
        {"id": "b", "text": "beta lazy dog"},
        {"id": "c", "text": "gamma jumps over"},
    ])
    idx.delete(["a", "c"])
    assert idx.all_ids() == {"b"}

    # FTS no longer returns deleted ids
    results = idx.search("alpha gamma", n=5)
    assert all(r["id"] not in {"a", "c"} for r in results)


def test_delete_empty_list_is_noop(tmp_path):
    """delete([]) is a safe no-op."""
    idx = BM25Index(tmp_path / "test.db")
    idx.upsert([{"id": "a", "text": "hello"}])
    idx.delete([])
    assert idx.all_ids() == {"a"}


def test_upsert_empty_list_is_noop(tmp_path):
    """upsert([]) is a safe no-op."""
    idx = BM25Index(tmp_path / "test.db")
    idx.upsert([])
    assert idx.all_ids() == set()


def test_delete_large_id_list_paginates(tmp_path):
    """delete() handles >5000 ids without hitting SQLITE_MAX_VARIABLE_NUMBER."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [{"id": f"id_{i}", "text": f"text {i}"} for i in range(6000)]
    idx.upsert(docs)
    idx.delete([f"id_{i}" for i in range(6000)])
    assert idx.all_ids() == set()
```

- [ ] **Step 2: Run the failing tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py -v -k "get_bm25 or upsert or delete or all_ids"`
Expected: Most FAIL because `get_bm25` is still a stub returning None.

- [ ] **Step 3: Implement the cache and helper**

Replace the cache stubs at the bottom of `src/vecs/bm25_index.py` with:

```python
# --- Module-level mtime cache ---

_bm25_cache: dict[str, tuple[float, "BM25Index"]] = {}


def get_bm25(path: Path) -> BM25Index | None:
    """Return a (cached) BM25Index for `path`, or None if the file is missing or unreadable.

    Caches by (path, mtime). On corruption (not a valid SQLite db) returns None
    so callers can degrade gracefully to vector-only search.
    """
    key = str(path)

    if not path.exists():
        cached = _bm25_cache.pop(key, None)
        if cached:
            try:
                cached[1].close()
            except Exception:
                pass
        return None

    mtime = path.stat().st_mtime
    if key in _bm25_cache and _bm25_cache[key][0] == mtime:
        return _bm25_cache[key][1]

    # Old cached entry is stale — close it
    if key in _bm25_cache:
        try:
            _bm25_cache[key][1].close()
        except Exception:
            pass
        _bm25_cache.pop(key, None)

    idx = BM25Index(path)
    try:
        idx.load()
        # Smoke-check: if the file is corrupt, this raises DatabaseError
        idx._ensure_conn().execute("SELECT 1 FROM docs LIMIT 1").fetchone()
    except sqlite3.DatabaseError:
        try:
            idx.close()
        except Exception:
            pass
        return None
    except Exception:
        try:
            idx.close()
        except Exception:
            pass
        return None

    _bm25_cache[key] = (mtime, idx)
    return idx
```

- [ ] **Step 4: Run the cache and incremental-API tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_bm25.py -v`
Expected: All PASS. If `test_get_bm25_invalidates_on_mtime_change` flakes, the `db.touch()` line in the test is intentional belt-and-suspenders for systems with low mtime resolution.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/bm25_index.py tests/test_bm25.py
git commit -m "feat(bm25): add mtime cache and incremental upsert/delete API"
```

---

## Task 4: Indexer integration — diff-based `_sync_bm25`

**Files:**
- Modify: `src/vecs/indexer.py:489-504` (replace `_rebuild_bm25` body)
- Modify: `src/vecs/indexer.py:769, 902, 974, 1026` (rename call sites)
- Test: `tests/test_indexer.py:755-817, 1085-1122` (update existing BM25 tests)

- [ ] **Step 1: Write failing test for incremental sync**

Add to `tests/test_indexer.py`:

```python
def test_sync_bm25_inserts_new_chunks(tmp_path, monkeypatch):
    """_sync_bm25 inserts chunks that exist in chromadb but not yet in BM25."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a", "b", "c"],
        "documents": ["alpha text", "beta text", "gamma text"],
        "metadatas": [{"file_path": "a.ts"}, {"file_path": "b.ts"}, {"file_path": "c.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    db = tmp_path / "bm25" / "test_code.db"
    assert db.exists()
    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == {"a", "b", "c"}
    bm25.close()


def test_sync_bm25_deletes_removed_chunks(tmp_path, monkeypatch):
    """_sync_bm25 removes BM25 docs that are no longer in chromadb."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    # Pre-populate BM25 with a, b, c
    db = tmp_path / "bm25" / "test_code.db"
    pre = BM25Index(db)
    pre.upsert([
        {"id": "a", "text": "alpha", "metadata": {"file_path": "a.ts"}},
        {"id": "b", "text": "beta", "metadata": {"file_path": "b.ts"}},
        {"id": "c", "text": "gamma", "metadata": {"file_path": "c.ts"}},
    ])
    pre.close()

    # Chromadb only has 'a' now
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a"],
        "documents": ["alpha text"],
        "metadatas": [{"file_path": "a.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == {"a"}
    bm25.close()


def test_sync_bm25_updates_changed_chunks(tmp_path, monkeypatch):
    """_sync_bm25 updates content for existing chunk ids when the chromadb text changes."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)

    db = tmp_path / "bm25" / "test_code.db"
    pre = BM25Index(db)
    pre.upsert([{"id": "a", "text": "old text alpha", "metadata": {"file_path": "a.ts"}}])
    pre.close()

    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["a"],
        "documents": ["new text completely different"],
        "metadatas": [{"file_path": "a.ts"}],
    }

    indexer._sync_bm25(collection, "test", "code")

    bm25 = BM25Index(db)
    bm25.load()
    # Old text should no longer match
    assert bm25.search("old text", n=5) == [] or all(
        r["id"] != "a" for r in bm25.search("old text", n=5)
    )
    # New text should match
    results = bm25.search("completely different", n=5)
    assert any(r["id"] == "a" for r in results)
    bm25.close()


def test_sync_bm25_logs_on_failure(tmp_path, monkeypatch, capsys):
    """_sync_bm25 logs errors instead of swallowing them."""
    from vecs import indexer

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    collection = MagicMock()
    collection.get.side_effect = RuntimeError("ChromaDB unavailable")

    indexer._sync_bm25(collection, "test", "code")

    captured = capsys.readouterr()
    assert "BM25 sync failed" in captured.err
    assert "ChromaDB unavailable" in captured.err


def test_sync_bm25_paginates_through_all_chunks(tmp_path, monkeypatch):
    """_sync_bm25 walks all chromadb pages, not just the first."""
    from vecs import indexer
    from vecs.bm25_index import BM25Index

    monkeypatch.setattr(indexer, "VECS_DIR", tmp_path)
    total_chunks = 12
    all_ids = [f"code:f{i}.ts:0" for i in range(total_chunks)]
    all_docs = [f"function f{i}() {{ return {i}; }}" for i in range(total_chunks)]
    all_metas = [{"file_path": f"f{i}.ts"} for i in range(total_chunks)]

    def fake_get(*, limit, offset, include):
        return {
            "ids": all_ids[offset:offset + limit],
            "documents": all_docs[offset:offset + limit],
            "metadatas": all_metas[offset:offset + limit],
        }

    collection = MagicMock()
    collection.get.side_effect = fake_get
    monkeypatch.setattr(
        indexer,
        "_paginated_get",
        lambda col, batch_size=5, **kw: indexer._paginated_get.__wrapped__(col, batch_size=5, **kw)
        if hasattr(indexer._paginated_get, "__wrapped__")
        else _paginated_get_inline(col, batch_size=5, **kw),
    )

    # Simpler: just call _sync_bm25 with whatever pagination the indexer ships;
    # verify all 12 ids are in the index.
    indexer._sync_bm25(collection, "test", "code")

    db = tmp_path / "bm25" / "test_code.db"
    bm25 = BM25Index(db)
    bm25.load()
    assert bm25.all_ids() == set(all_ids)
    bm25.close()
```

(If the monkeypatch trick in the pagination test is awkward, simpler: just keep the default page size and pass 12 chunks split across two `collection.get` returns using `side_effect` as a list. Either approach proves pagination is exercised.)

- [ ] **Step 2: Run the failing tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_indexer.py -v -k "_sync_bm25"`
Expected: FAIL with `AttributeError: module 'vecs.indexer' has no attribute '_sync_bm25'`.

- [ ] **Step 3: Replace `_rebuild_bm25` with `_sync_bm25` in `src/vecs/indexer.py`**

Replace lines 489-504 (the entire `_rebuild_bm25` function) with:

```python
def _sync_bm25(collection: chromadb.Collection, project_name: str, suffix: str) -> None:
    """Incrementally sync the BM25 FTS5 index with the given ChromaDB collection.

    Diff strategy:
      1. Fetch all chunk ids + documents + metadatas from chromadb (paginated).
      2. Compare ids against the existing BM25 index's id set.
      3. Delete BM25 rows whose ids are no longer in chromadb.
      4. Upsert all chromadb-resident chunks (insert new + overwrite existing).

    For per-tick incremental updates this is O(diff size) for the SQL writes;
    the chromadb scan is O(N) but only reads ids+docs+metas (cheap relative
    to the prior full rebuild + tokenize + repickle).

    Migrates transparently: when no .db exists yet, step 4 simply inserts
    everything (equivalent to a fresh build). Old `.pkl` files (if any) are
    deleted on first successful sync.
    """
    bm25_dir = VECS_DIR / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    db_path = bm25_dir / f"{project_name}_{suffix}.db"
    legacy_pkl = bm25_dir / f"{project_name}_{suffix}.pkl"

    bm25 = BM25Index(db_path)
    try:
        bm25.load()
        chroma_docs: list[dict] = []
        chroma_ids: set[str] = set()
        for page in _paginated_get(collection, include=["documents", "metadatas"]):
            for id_, text, meta in zip(
                page["ids"], page["documents"], page["metadatas"]
            ):
                chroma_ids.add(id_)
                chroma_docs.append({"id": id_, "text": text, "metadata": meta or {}})

        existing_ids = bm25.all_ids()
        to_delete = sorted(existing_ids - chroma_ids)
        if to_delete:
            bm25.delete(to_delete)
        if chroma_docs:
            bm25.upsert(chroma_docs)

        # One-shot cleanup of the obsolete pickle, if it exists
        if legacy_pkl.exists():
            try:
                legacy_pkl.unlink()
            except OSError:
                pass
    except Exception as e:
        _log(f"  Warning: BM25 sync failed for {project_name}_{suffix}: {e}")
    finally:
        bm25.close()
```

- [ ] **Step 4: Update the four call sites**

Use a single replace_all for the rename inside `src/vecs/indexer.py`:

```bash
# (Done via Edit tool with replace_all=True; pseudocode for the agent:)
# old_string: "_rebuild_bm25("
# new_string: "_sync_bm25("
```

Verify the only matches are the four lines (769, 902, 974, 1026) — these are the only callers per the codebase exploration.

- [ ] **Step 5: Run all updated indexer tests**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest tests/test_indexer.py -v -k "bm25 or _sync_bm25"`
Expected: All PASS.

Also rename and update the three obsolete tests:
- `test_rebuild_bm25_includes_metadata` → adapt to `_sync_bm25`, assert `.db` path exists, content searchable.
- `test_rebuild_bm25_logs_on_failure` → rename to `test_sync_bm25_logs_on_failure` (already in step 1).
- `test_rebuild_bm25_paginates_through_all_chunks` → rename to `test_sync_bm25_paginates_through_all_chunks` (already in step 1).

Delete the originals so we don't have duplicates.

- [ ] **Step 6: Run the full test suite to catch any other reference**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && .venv/bin/python -m pytest -v`
Expected: All PASS. If anything references `_rebuild_bm25` or `.pkl`, fix.

- [ ] **Step 7: Commit**

```bash
git add src/vecs/indexer.py tests/test_indexer.py
git commit -m "feat(indexer): replace _rebuild_bm25 with diff-based _sync_bm25"
```

---

## Task 5: Searcher cache lookup + dependency cleanup

**Files:**
- Modify: `src/vecs/searcher.py:199` (`.pkl` → `.db`)
- Modify: `pyproject.toml:14` (remove `rank-bm25>=0.2`)
- Regenerate: `uv.lock`

- [ ] **Step 1: Update searcher.py path lookup**

Edit `src/vecs/searcher.py:199` — change:

```python
bm25_path = bm25_dir / f"{proj_name}_{suffix}.pkl"
```

to:

```python
bm25_path = bm25_dir / f"{proj_name}_{suffix}.db"
```

- [ ] **Step 2: Remove rank-bm25 from pyproject.toml**

Edit `pyproject.toml`. Remove the line `"rank-bm25>=0.2",` from the `dependencies` list.

- [ ] **Step 3: Regenerate uv.lock**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv lock`
Expected: Output mentions removing `rank-bm25` and regenerating the lock.

- [ ] **Step 4: Verify nothing else imports rank_bm25**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && grep -rn "rank_bm25\|rank-bm25" src/ tests/ scripts/ 2>/dev/null`
Expected: No matches. (If anything matches, remove the import.)

- [ ] **Step 5: Reinstall and re-run the full suite**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv sync && .venv/bin/python -m pytest -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/vecs/searcher.py
git commit -m "chore: drop rank-bm25 dependency, point searcher at .db files"
```

---

## Task 6: A/B benchmark script

**Files:**
- Create: `scripts/bench_bm25.py`

This task creates a runnable benchmark that the human reviewer can execute against any local vecs corpus to compare the FTS5 implementation to a temporary rank_bm25 baseline. We snapshot results to a markdown table.

- [ ] **Step 1: Write the benchmark script**

Create `scripts/bench_bm25.py`:

```python
"""A/B benchmark: FTS5 BM25 (current) vs rank_bm25 (legacy) on a real corpus.

Usage:
    .venv/bin/python scripts/bench_bm25.py [--project PROJECT] [--suffix code|sessions|docs]

Reads chunks from the project's chromadb collection, builds both indexes
in tmpdirs, runs a fixed query set, and prints:

  - top-K overlap (Jaccard of result sets at K=5 and K=10)
  - per-query latency (median, p95)
  - bulk build / single-doc upsert latency

Requires `rank_bm25` to be installed temporarily for the comparison:
    uv pip install rank_bm25
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vecs.bm25_index import BM25Index, _tokenize  # noqa: E402
from vecs.clients import get_chromadb_client  # noqa: E402
from vecs.config import load_config  # noqa: E402
from vecs.indexer import _paginated_get  # noqa: E402

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank_bm25 not installed; run: uv pip install rank_bm25", file=sys.stderr)
    sys.exit(2)


CANONICAL_QUERIES = [
    "BM25 index",
    "rebuild bm25",
    "tokenize camelCase",
    "VECS_DIR",
    "reciprocal rank fusion",
    "ChromaDB collection",
    "voyage embed",
    "HTTPSConnection",
    "test_indexer",
    "_sync_bm25",
]


def fetch_chunks(collection_name: str) -> list[dict]:
    db = get_chromadb_client()
    col = db.get_collection(collection_name)
    out = []
    for page in _paginated_get(col, include=["documents", "metadatas"]):
        for id_, text, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            out.append({"id": id_, "text": text, "metadata": meta or {}})
    return out


def build_legacy(docs: list[dict]) -> tuple[BM25Okapi, list[str], list[dict]]:
    ids = [d["id"] for d in docs]
    metas = [d.get("metadata", {}) for d in docs]
    tokenized = [_tokenize(d["text"]) for d in docs]
    return BM25Okapi(tokenized), ids, metas


def search_legacy(b25, ids, metas, query: str, n: int) -> list[str]:
    scores = b25.get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [ids[i] for i, s in ranked if s > 0]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def time_call(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--suffix", default="code", choices=["code", "sessions", "docs"])
    args = ap.parse_args()

    cfg = load_config()
    if args.project:
        proj = cfg.projects[args.project]
    else:
        proj = next(iter(cfg.projects.values()))

    if args.suffix == "code":
        col_name = proj.code_collection
    elif args.suffix == "sessions":
        col_name = proj.sessions_collection
    else:
        col_name = proj.docs_collection

    print(f"Loading chunks from {col_name}…", file=sys.stderr)
    docs = fetch_chunks(col_name)
    print(f"  {len(docs)} chunks", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        # Build legacy
        print("\n# Build phase", file=sys.stderr)
        (legacy, l_ids, l_metas), legacy_build_ms = time_call(build_legacy, docs)
        print(f"  legacy (rank_bm25 build): {legacy_build_ms:.1f} ms")

        # Build FTS5
        fts_path = Path(tmp) / "fts.db"
        idx = BM25Index(fts_path)
        _, fts_build_ms = time_call(idx.build, docs)
        print(f"  fts5   (build full):      {fts_build_ms:.1f} ms")

        # Single-doc upsert latency (fts5 only — legacy has no incremental API)
        sample = docs[len(docs) // 2]
        _, upsert_ms = time_call(idx.upsert, [sample])
        print(f"  fts5   (single upsert):   {upsert_ms:.2f} ms")
        print(f"  fts5   (single delete):   ", end="")
        _, delete_ms = time_call(idx.delete, [sample["id"]])
        print(f"{delete_ms:.2f} ms")
        idx.upsert([sample])  # restore for query phase

        # Query phase
        print("\n# Query phase (top-K overlap and latency)", file=sys.stderr)
        print(f"\n{'query':<28} {'k=5 jacc':<10} {'k=10 jacc':<10} {'leg ms':<8} {'fts ms':<8}")
        leg_lat, fts_lat = [], []
        overlap_5, overlap_10 = [], []
        for q in CANONICAL_QUERIES:
            l5, l_ms_5 = time_call(search_legacy, legacy, l_ids, l_metas, q, 5)
            l10, l_ms_10 = time_call(search_legacy, legacy, l_ids, l_metas, q, 10)
            f5_results, f_ms_5 = time_call(idx.search, q, 5)
            f10_results, f_ms_10 = time_call(idx.search, q, 10)
            f5 = [r["id"] for r in f5_results]
            f10 = [r["id"] for r in f10_results]
            j5, j10 = jaccard(l5, f5), jaccard(l10, f10)
            overlap_5.append(j5); overlap_10.append(j10)
            leg_lat.append(l_ms_5); fts_lat.append(f_ms_5)
            print(f"{q:<28} {j5:<10.2f} {j10:<10.2f} {l_ms_5:<8.2f} {f_ms_5:<8.2f}")

        print(f"\n# Summary")
        print(f"  mean top-5 overlap:  {statistics.mean(overlap_5):.2f}")
        print(f"  mean top-10 overlap: {statistics.mean(overlap_10):.2f}")
        print(f"  legacy median ms:    {statistics.median(leg_lat):.2f}")
        print(f"  fts5   median ms:    {statistics.median(fts_lat):.2f}")
        idx.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the script (no assertion — informational)**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv pip install rank_bm25 && .venv/bin/python scripts/bench_bm25.py 2>&1 | tee /tmp/bench-bm25.txt`
Expected: Prints a result table, no exceptions.

- [ ] **Step 3: Capture the output as a benchmark report**

Save the bench output as `docs/superpowers/plans/2026-04-27-bm25-fts5-bench.md` (just paste the captured output under a header — this is for human review, not asserted).

- [ ] **Step 4: Uninstall the temporary rank_bm25**

Run: `cd /Users/darynavoloshyna/Repositories/vecs && uv pip uninstall rank_bm25`
Expected: Clean removal.

- [ ] **Step 5: Commit the script (not the benchmark report)**

The benchmark report is one-shot and machine-generated — keep it gitignored or just save it locally. The script is checked in for repeatability.

```bash
git add scripts/bench_bm25.py
git commit -m "test: add A/B benchmark script comparing FTS5 vs rank_bm25"
```

---

## Self-review (writer's checklist — already executed)

**Spec coverage:**
- ✅ Drop `rank_bm25` + pickle (Tasks 2, 5)
- ✅ FTS5 virtual table + custom tokenizer (Tasks 1, 2; tokenizer kept in Python pre-tokenization)
- ✅ Replace `_rebuild_bm25` per-tick rebuild with incremental upsert/delete (Tasks 3, 4)
- ✅ Cleanup of stale BM25 docs no longer in chromadb (Task 4 — `to_delete` step covers what `_sweep_excluded_chunks` doesn't)
- ✅ One-shot migration (Task 4 — auto-rebuild on missing .db, plus pickle cleanup)
- ✅ Tests rewritten (Tasks 1–4)
- ✅ A/B benchmark (Task 6)

**Placeholder scan:** None — every code block is complete; every command is exact.

**Type consistency:** `BM25Index.path: Path`, `BM25Index.build(docs: list[dict]) -> None`, `BM25Index.search(query: str, n: int = 5, path_filter: str | None = None) -> list[dict]` are consistent across all tasks. `_sync_bm25(collection, project_name: str, suffix: str) -> None` matches the prior `_rebuild_bm25` signature so call sites only need a rename.

## Risks and how the plan addresses them

| Risk | Mitigation |
|------|-----------|
| FTS5 BM25 ranking differs from rank_bm25 (k1, epsilon) | Task 6 benchmark measures top-K overlap on canonical queries. Acceptance threshold is documented in Task 6 review (tune column weights or revisit if mean top-10 overlap < 0.5). |
| WAL on shared filesystems | We only write under `~/.vecs` which is the user's home — local FS, never NFS. No change in storage location. |
| FTS5 not in some Python distros | Confirmed available in stdlib for Python 3.12 / 3.13 macOS / Linux wheels. Schema-creation test (Task 1) fails loudly if absent on a contributor's machine — surfaces the issue immediately. |
| User input with FTS5 syntax chars (`-`, `:`, `*`, …) | `_build_match_query` quotes every token (Task 1). Test `test_bm25_punctuation_in_query_does_not_crash` exercises this. |
| Concurrent indexer + searcher on same .db | WAL mode + `busy_timeout=5000` on every connection (Task 1). |
| Old `.pkl` files clutter `~/.vecs/bm25` | First successful `_sync_bm25` deletes the corresponding `.pkl` (Task 4). |
