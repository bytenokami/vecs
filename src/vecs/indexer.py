from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import chromadb
import voyageai

from vecs.chunkers import chunk_code_file, preprocess_session, chunk_session
from vecs.config import (
    BLOOMLY_CODE_DIR,
    BLOOMLY_SESSIONS_DIR,
    CHROMADB_DIR,
    CODE_CHUNK_LINES,
    CODE_CHUNK_OVERLAP,
    CODE_COLLECTION,
    CODE_EXTENSIONS,
    CODE_MODEL,
    MANIFEST_PATH,
    SESSION_CHUNK_MESSAGES,
    SESSIONS_COLLECTION,
    SESSIONS_MODEL,
    VECS_DIR,
    VOYAGE_BATCH_SIZE,
)


class Manifest:
    """Tracks which files have been indexed and their content hashes."""

    def __init__(self, path: Path = MANIFEST_PATH):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def _file_hash(self, file_path: Path) -> str:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def needs_indexing(self, file_path: Path) -> bool:
        key = str(file_path)
        if key not in self.data:
            return True
        return self.data[key] != self._file_hash(file_path)

    def mark_indexed(self, file_path: Path) -> None:
        self.data[str(file_path)] = self._file_hash(file_path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: str,
    vo: voyageai.Client,
    id_prefix: str,
) -> int:
    """Embed chunks in batches and store in ChromaDB. Returns count stored."""
    if not chunks:
        return 0

    batch_size = min(VOYAGE_BATCH_SIZE, 10)
    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        for attempt in range(5):
            try:
                result = vo.embed(texts, model=model, input_type="document")
                break
            except Exception as e:
                if "RateLimitError" in type(e).__name__ or "rate" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    _log(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            _log(f"  Failed after 5 retries, skipping batch at {i}")
            continue

        ids = [f"{id_prefix}-{i + j}" for j in range(len(batch))]
        metadatas = [c["metadata"] for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=result.embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        stored += len(batch)
        _log(f"  Indexed {stored}/{len(chunks)} chunks")

        if i + batch_size < len(chunks):
            time.sleep(21)

    return stored


def index_code(vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Bloomly .cs files from Assets/. Returns count of new chunks."""
    manifest = Manifest()
    collection = db.get_or_create_collection(CODE_COLLECTION)

    files = [
        f
        for f in BLOOMLY_CODE_DIR.rglob("*")
        if f.suffix in CODE_EXTENSIONS and f.is_file()
    ]

    to_index = [f for f in files if manifest.needs_indexing(f)]
    if not to_index:
        _log("Code: nothing new to index.")
        return 0

    _log(f"Code: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        content = f.read_text(errors="replace")
        rel_path = str(f.relative_to(BLOOMLY_CODE_DIR))
        chunks = chunk_code_file(
            content, rel_path, CODE_CHUNK_LINES, CODE_CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)

    stored = _embed_and_store(all_chunks, collection, CODE_MODEL, vo, "code")

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def index_sessions(vo: voyageai.Client, db: chromadb.ClientAPI) -> int:
    """Index Claude Code session transcripts. Returns count of new chunks."""
    manifest = Manifest()
    collection = db.get_or_create_collection(SESSIONS_COLLECTION)

    files = sorted(BLOOMLY_SESSIONS_DIR.glob("*.jsonl"))
    to_index = [f for f in files if manifest.needs_indexing(f)]

    if not to_index:
        _log("Sessions: nothing new to index.")
        return 0

    _log(f"Sessions: {len(to_index)} files to index ({len(files)} total)")

    all_chunks = []
    for f in to_index:
        raw = f.read_text(errors="replace")
        session_id = f.stem
        messages = preprocess_session(raw)
        chunks = chunk_session(messages, session_id, SESSION_CHUNK_MESSAGES)
        all_chunks.extend(chunks)

    stored = _embed_and_store(
        all_chunks, collection, SESSIONS_MODEL, vo, "session"
    )

    for f in to_index:
        manifest.mark_indexed(f)
    manifest.save()

    return stored


def run_index() -> None:
    """Run full incremental index."""
    VECS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

    vo = voyageai.Client()
    db = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    _log("Starting index...")
    code_count = index_code(vo, db)
    session_count = index_sessions(vo, db)
    _log(f"Done. Indexed {code_count} code chunks, {session_count} session chunks.")
