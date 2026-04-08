from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


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
    tokens = []
    for word in words:
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", word
        )
        if parts:
            tokens.extend(p.lower() for p in parts)
        else:
            tokens.append(word.lower())
    return tokens


class BM25Index:
    """BM25 keyword search index with persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_metadatas: list[dict] = []

    def build(self, docs: list[dict]) -> None:
        """Build index from list of {"id", "text", optional "metadata"} dicts."""
        self.doc_ids = [d["id"] for d in docs]
        self.doc_texts = [d["text"] for d in docs]
        self.doc_metadatas = [d.get("metadata", {}) for d in docs]
        tokenized = [_tokenize(d["text"]) for d in docs]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def search(self, query: str, n: int = 5, path_filter: str | None = None) -> list[dict]:
        """Search the index. Returns list of {"id", "text", "score", "metadata"}.

        Args:
            query: Search query string.
            n: Number of results to return.
            path_filter: If set, only return results whose file_path metadata
                         contains this substring. Fetches n*5 internally to
                         compensate for post-filter loss.
        """
        if self.bm25 is None or not self.doc_ids:
            return []

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Fetch more candidates when filtering to compensate for post-filter loss
        fetch_n = n * 5 if path_filter else n

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:fetch_n]

        results = [
            {
                "id": self.doc_ids[i],
                "text": self.doc_texts[i],
                "score": float(score),
                "metadata": self.doc_metadatas[i] if i < len(self.doc_metadatas) else {},
            }
            for i, score in ranked
            if score > 0
        ]

        if path_filter:
            results = [
                r for r in results
                if path_filter in r["metadata"].get("file_path", "")
            ]

        return results[:n]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "doc_texts": self.doc_texts,
                    "doc_metadatas": self.doc_metadatas,
                },
                f,
            )

    def load(self) -> bool:
        """Load from disk. Returns True if loaded, False if file missing."""
        if not self.path.exists():
            return False
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        # Backward compat: old pickles may not have doc_metadatas
        self.doc_metadatas = data.get("doc_metadatas", [{} for _ in self.doc_ids])
        # Rebuild BM25 from tokenized texts (BM25Okapi doesn't pickle well)
        tokenized = [_tokenize(text) for text in self.doc_texts]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
        return True


# --- Module-level BM25 cache (H5 + H8) ---

_bm25_cache: dict[str, tuple[float, BM25Index]] = {}


def get_bm25(path: Path) -> BM25Index | None:
    """Load a BM25 index with mtime-based caching.

    Returns cached index if file hasn't changed. On load failure
    (corrupted pickle, missing keys, etc.), returns None for graceful
    degradation to vector-only search.
    """
    key = str(path)

    if not path.exists():
        _bm25_cache.pop(key, None)
        return None

    mtime = path.stat().st_mtime

    if key in _bm25_cache and _bm25_cache[key][0] == mtime:
        return _bm25_cache[key][1]

    bm25 = BM25Index(path)
    try:
        if bm25.load():
            _bm25_cache[key] = (mtime, bm25)
            return bm25
    except Exception:
        return None

    return None
