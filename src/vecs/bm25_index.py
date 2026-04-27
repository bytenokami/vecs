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
