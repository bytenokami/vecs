import pytest

from vecs.embed_cache import EmbedCache


def test_content_hash_deterministic_and_distinct():
    h1 = EmbedCache.content_hash("hello world")
    h2 = EmbedCache.content_hash("hello world")
    h3 = EmbedCache.content_hash("hello mars")
    assert h1 == h2
    assert h1 != h3
    assert isinstance(h1, str) and len(h1) == 64  # sha256 hex


def test_put_then_get_roundtrip(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    h = EmbedCache.content_hash("abc")
    cache.put("voyage-3", [(h, [0.1, 0.2, 0.3])])
    got = cache.get("voyage-3", [h])
    assert h in got
    assert got[h] == pytest.approx([0.1, 0.2, 0.3])
    cache.close()


def test_get_miss_returns_empty(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    got = cache.get("voyage-3", [EmbedCache.content_hash("nope")])
    assert got == {}
    cache.close()


def test_model_scoping_same_hash_different_model_misses(tmp_path):
    """A content hash stored under one model must NOT be served for another.

    Vectors are model-specific: serving a voyage-3 vector for a voyage-3.5
    request silently corrupts ranking. This scoping is what lets B re-embed
    under a new model without a cross-model false hit.
    """
    cache = EmbedCache(tmp_path / "c.db")
    h = EmbedCache.content_hash("abc")
    cache.put("voyage-3", [(h, [1.0, 2.0])])
    assert cache.get("voyage-3.5", [h]) == {}
    assert h in cache.get("voyage-3", [h])
    cache.close()


def test_persists_across_instances(tmp_path):
    db = tmp_path / "c.db"
    h = EmbedCache.content_hash("abc")
    c1 = EmbedCache(db)
    c1.put("voyage-3", [(h, [0.5, 0.6])])
    c1.close()
    c2 = EmbedCache(db)
    got = c2.get("voyage-3", [h])
    assert got[h] == pytest.approx([0.5, 0.6])
    c2.close()


def test_get_returns_only_hits(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    h1 = EmbedCache.content_hash("a")
    h2 = EmbedCache.content_hash("b")
    cache.put("voyage-3", [(h1, [1.0])])
    got = cache.get("voyage-3", [h1, h2])
    assert set(got) == {h1}
    cache.close()


def test_put_overwrites_existing(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    h = EmbedCache.content_hash("a")
    cache.put("voyage-3", [(h, [1.0])])
    cache.put("voyage-3", [(h, [2.0])])
    assert cache.get("voyage-3", [h])[h] == pytest.approx([2.0])
    cache.close()


def test_get_empty_hash_list_returns_empty(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    assert cache.get("voyage-3", []) == {}
    cache.close()


def test_put_empty_items_noop(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    cache.put("voyage-3", [])  # must not raise
    cache.close()


def test_collection_model_absent_returns_none(tmp_path):
    """No marker recorded yet -> None. Existing (pre-B2) collections read None,
    which the run_index pre-pass treats as 'differs from configured model'."""
    cache = EmbedCache(tmp_path / "c.db")
    assert cache.get_collection_model("proj-docs") is None
    cache.close()


def test_set_then_get_collection_model_roundtrip(tmp_path):
    cache = EmbedCache(tmp_path / "c.db")
    cache.set_collection_model("proj-docs", "voyage-4")
    assert cache.get_collection_model("proj-docs") == "voyage-4"
    cache.close()


def test_set_collection_model_overwrites(tmp_path):
    """The marker is updated in place when a collection is re-embedded."""
    cache = EmbedCache(tmp_path / "c.db")
    cache.set_collection_model("proj-docs", "voyage-3")
    cache.set_collection_model("proj-docs", "voyage-4")
    assert cache.get_collection_model("proj-docs") == "voyage-4"
    cache.close()


def test_collection_model_persists_across_instances(tmp_path):
    """The post-pass marker must survive process exit (cron run -> next run)."""
    db = tmp_path / "c.db"
    c1 = EmbedCache(db)
    c1.set_collection_model("proj-sessions", "voyage-4")
    c1.close()
    c2 = EmbedCache(db)
    assert c2.get_collection_model("proj-sessions") == "voyage-4"
    c2.close()


def test_collection_model_isolated_per_collection(tmp_path):
    """Per-collection key: a docs re-embed must not flip the sessions marker."""
    cache = EmbedCache(tmp_path / "c.db")
    cache.set_collection_model("a-docs", "voyage-4")
    cache.set_collection_model("b-sessions", "voyage-3")
    assert cache.get_collection_model("a-docs") == "voyage-4"
    assert cache.get_collection_model("b-sessions") == "voyage-3"
    cache.close()


def test_uses_wal_and_busy_timeout(tmp_path):
    """Concurrency: overlapping reindex runs write the same db. WAL lets a
    reader proceed during a write; busy_timeout makes a competing writer wait
    instead of failing instantly with 'database is locked' (mirrors bm25_index).
    """
    cache = EmbedCache(tmp_path / "c.db")
    journal_mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy_timeout = cache._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert journal_mode.lower() == "wal"
    assert busy_timeout == 5000
    cache.close()
