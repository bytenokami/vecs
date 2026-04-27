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
        assert "idx_docs_file_path" in names

        # WAL mode is set
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


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
    result = _build_match_query('hello world')
    assert '"hello"' in result and '"world"' in result


def test_build_match_query_punctuation_safe():
    """FTS5-syntax punctuation in input does not produce a syntactically invalid query."""
    q = _build_match_query("foo-bar baz")
    assert q is not None
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t(x) VALUES ('foo bar baz')")
    rows = list(conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (q,)))
    assert len(rows) == 1
    conn.close()


# ---------------------------------------------------------------------------
# Tokenizer tests (preserved from original implementation)
# ---------------------------------------------------------------------------

def test_tokenize_camel_case():
    """camelCase identifiers are split into subwords."""
    assert _tokenize("getUserById") == ["get", "user", "by", "id"]


def test_tokenize_pascal_case():
    """PascalCase identifiers are split into subwords."""
    assert _tokenize("XMLParser") == ["xml", "parser"]


def test_tokenize_acronym_connection():
    """ACRONYM followed by PascalCase word splits correctly."""
    assert _tokenize("HTTPSConnection") == ["https", "connection"]


def test_tokenize_all_caps():
    """ALL_CAPS identifiers split on underscore, each word separate."""
    assert _tokenize("ALL_CAPS") == ["all", "caps"]


def test_tokenize_mixed():
    """Mixed camelCase identifier splits correctly."""
    assert _tokenize("parseJSON") == ["parse", "json"]


def test_tokenize_short_acronym():
    """Short all-caps like ID are preserved as a single token."""
    assert _tokenize("ID") == ["id"]


def test_tokenize_trailing_url():
    """Trailing acronym in camelCase splits correctly."""
    assert _tokenize("getURL") == ["get", "url"]


def test_tokenize_snake_case():
    """snake_case names split into subwords."""
    assert _tokenize("snake_case_name") == ["snake", "case", "name"]


def test_tokenize_dunder():
    """Dunder methods split correctly (underscores removed by \\w+)."""
    assert _tokenize("__init__") == ["init"]


def test_tokenize_plain_words():
    """Plain lowercase words pass through unchanged."""
    assert _tokenize("hello world") == ["hello", "world"]


def test_tokenize_numbers():
    """Numbers are preserved as tokens."""
    assert _tokenize("item42count") == ["item", "42", "count"]


def test_tokenize_mixed_sentence():
    """Realistic code line with mixed identifiers."""
    tokens = _tokenize("getUserById returns HTTPResponse")
    assert tokens == ["get", "user", "by", "id", "returns", "http", "response"]


# ---------------------------------------------------------------------------
# Legacy BM25Index tests (will be rewritten in Task 2 — kept for reference)
# ---------------------------------------------------------------------------

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


def test_bm25_load_returns_false_for_new_file(tmp_path):
    """load() returns False when the .db file did not exist before opening."""
    db_path = tmp_path / "fresh.db"
    assert not db_path.exists()
    idx = BM25Index(db_path)
    result = idx.load()
    try:
        assert result is False
        # And it's now usable
        idx.upsert([{"id": "a", "text": "hello"}])
        assert idx.all_ids() == {"a"}
    finally:
        idx.close()


def test_bm25_empty_index(tmp_path):
    """Empty index returns no results."""
    idx = BM25Index(tmp_path / "test.db")
    idx.build([])
    results = idx.search("anything", n=5)
    assert results == []


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
    db.touch()  # belt-and-suspenders mtime bump for low-resolution filesystems

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


def test_get_bm25_cache_cleared_between_tests():
    """Verify _bm25_cache is accessible for test cleanup."""
    _bm25_cache.clear()
    assert len(_bm25_cache) == 0


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
    for q in ["foo-bar", 'foo "bar"', "foo:bar", "foo*", "foo AND bar"]:
        results = idx.search(q, n=5)
        assert isinstance(results, list)


def test_bm25_search_path_filter_no_matches(tmp_path):
    """BM25 path_filter that matches nothing returns empty."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "hello world greeting", "metadata": {"file_path": "src/hello.ts"}},
        {"id": "b", "text": "goodbye farewell parting", "metadata": {"file_path": "src/bye.ts"}},
        {"id": "c", "text": "something else entirely", "metadata": {"file_path": "src/other.ts"}},
    ]
    idx.build(docs)
    results = idx.search("hello", n=5, path_filter="nonexistent/")
    assert results == []


def test_bm25_search_path_filter_no_metadata(tmp_path):
    """Path filter skips docs with no file_path in metadata."""
    idx = BM25Index(tmp_path / "test.db")
    docs = [
        {"id": "a", "text": "hello world greeting", "metadata": {"session_id": "abc"}},
        {"id": "b", "text": "hello earth planet", "metadata": {"file_path": "src/hello.ts"}},
        {"id": "c", "text": "something else entirely", "metadata": {"file_path": "lib/other.ts"}},
    ]
    idx.build(docs)
    results = idx.search("hello", n=5, path_filter="src/")
    assert len(results) == 1
    assert results[0]["id"] == "b"
