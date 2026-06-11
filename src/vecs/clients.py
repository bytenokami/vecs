from __future__ import annotations

import chromadb
import voyageai
from chromadb.config import Settings

from vecs.config import CHROMADB_DIR

_vo_client: voyageai.Client | None = None
_db_client: chromadb.ClientAPI | None = None


def get_voyage_client() -> voyageai.Client:
    """Return a singleton Voyage client."""
    global _vo_client
    if _vo_client is None:
        _vo_client = voyageai.Client()
    return _vo_client


def get_chromadb_client() -> chromadb.ClientAPI:
    """Return a singleton ChromaDB persistent client."""
    global _db_client
    if _db_client is None:
        CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
        _db_client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _db_client
