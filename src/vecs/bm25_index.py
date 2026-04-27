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


# --- BM25Index ---

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
                substring (translated to a SQL LIKE so it runs inside the database
                rather than as a Python post-filter on a small candidate set).
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


# --- Module-level mtime cache (filled in Task 3) ---

_bm25_cache: dict = {}


def get_bm25(path: Path):
    """Stub — implemented in Task 3."""
    return None
