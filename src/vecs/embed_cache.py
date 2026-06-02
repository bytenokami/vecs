"""Content-addressable embedding cache.

Embeddings are expensive (Voyage API calls cost money and latency). When a file
changes, the indexer re-chunks the WHOLE file, but typically only one chunk's
text actually changed — the rest are byte-identical. This cache lets the
indexer skip the Voyage call for unchanged chunks while still upserting them
(so the manifest's ``succeeded == expected`` invariant holds; see
``indexer._track_embed_success``).

Key = (model, content_hash). The model is part of the key because vectors are
model-specific: a voyage-3 vector served for a voyage-3.5 request would
silently degrade ranking. Scoping by model is exactly what lets a re-embed
under a new model (Increment 1-B) treat every chunk as a miss and recompute,
rather than reuse a stale-space vector.

Vectors are stored as packed float32 blobs (4 bytes/float) — compact and
lossless for retrieval (ChromaDB stores float32 too).
"""

from __future__ import annotations

import array
import hashlib
import sqlite3
from pathlib import Path

from vecs.config import VECS_DIR

DEFAULT_CACHE_PATH = VECS_DIR / "embed_cache.db"


class EmbedCache:
    """SQLite-backed (model, content_hash) -> embedding store."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_CACHE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # WAL + busy_timeout (mirrors bm25_index._connect): the reindex cron can
        # overlap an MCP `reindex`, and there is no run-level lock, so two
        # processes may write this db concurrently. WAL lets a reader proceed
        # during a write; busy_timeout makes a competing writer wait rather than
        # fail instantly with "database is locked".
        self._conn = sqlite3.connect(
            str(self.db_path), isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model        TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                PRIMARY KEY (model, content_hash)
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def content_hash(text: str) -> str:
        """SHA-256 hex of the chunk text (the content-addressable key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, model: str, content_hashes: list[str]) -> dict[str, list[float]]:
        """Return {content_hash: embedding} for the cached subset of the inputs.

        Missing hashes are simply absent from the result.
        """
        if not content_hashes:
            return {}
        out: dict[str, list[float]] = {}
        # Chunk the IN-clause to stay well under SQLITE_MAX_VARIABLE_NUMBER.
        for i in range(0, len(content_hashes), 900):
            window = content_hashes[i : i + 900]
            placeholders = ",".join("?" * len(window))
            rows = self._conn.execute(
                f"SELECT content_hash, embedding FROM embeddings "
                f"WHERE model = ? AND content_hash IN ({placeholders})",
                (model, *window),
            ).fetchall()
            for chash, blob in rows:
                vec = array.array("f")
                vec.frombytes(blob)
                out[chash] = vec.tolist()
        return out

    def put(self, model: str, items: list[tuple[str, list[float]]]) -> None:
        """Insert/overwrite (model, content_hash) -> embedding pairs."""
        if not items:
            return
        rows = [
            (model, chash, array.array("f", embedding).tobytes())
            for chash, embedding in items
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (model, content_hash, embedding) "
            "VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
