"""prose_drift: bi-temporal INSERT/NOOP/SUPERSEDE state machine + extraction cache.

Design: docs/features/prose-staleness-detector-design-v1.md (V+).
Phase 7 dry-run subtask. v1 covers the state machine + extract_facts +
SQLite verdict cache. CLI / MCP / indexer wire-in land later.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from vecs.clients import get_voyage_client
from vecs.config import CHROMADB_DIR, SESSIONS_MODEL

PROSE_EXTRACTION_MODEL = "claude-opus-4-7"
EXTRACTION_PROMPT_VERSION = "v2"

EVENT_INSERT = "INSERT"
EVENT_NOOP = "NOOP"
EVENT_SUPERSEDE = "SUPERSEDE"

# Chroma rejects None in metadata values; use 0 sentinel for "still current".
# is_current bool is the operative filter; invalid_at is informational.
INVALID_AT_NONE_SENTINEL = 0

_PROSE_DRIFT_CACHE_DIR = Path.home() / ".vecs" / "prose_drift_cache"

EXTRACTION_PROMPT = """Extract structured factual claims from the user-authored messages below as a JSON array of objects.

Each object: {{"subject": <str>, "predicate": <str>, "object": <str>}}.

Canonicalization rules (critical — drift detection joins on exact (subject, predicate)):
- subject and predicate are SHORT, lowercase, snake_case canonical strings.
- Collapse synonyms to ONE canonical predicate. Prefer these when they fit:
  has_role, headcount, status, owns, uses, decision, deadline, location, name.
- Map paraphrases to the same (subject, predicate). e.g. "we have no backend dev",
  "the team lacks a server engineer", and "no one owns the backend" all become
  subject="team", predicate="has_role" (object describes the role state).
- object may be a natural-language phrase (verbatim is fine).
- Only extract assertions the user makes about the project, team, or work state.
  Skip questions, hypotheticals, and meta-talk. If no factual claims, return [].

Example input:  [user]: We still have no backend developer on the team.
Example output: [{{"subject":"team","predicate":"has_role","object":"no backend developer"}}]

Example input:  [user]: Sasha just joined as our backend engineer.
Example output: [{{"subject":"team","predicate":"has_role","object":"sasha is backend engineer"}}]

Messages:
{messages}

Return ONLY the JSON array, no prose."""

DOC_EXTRACTION_PROMPT = """Extract structured factual claims from the documentation text below as a JSON array of objects.

Each object: {{"subject": <str>, "predicate": <str>, "object": <str>}}.

Canonicalization rules (critical — drift detection joins on exact (subject, predicate)):
- subject and predicate are SHORT, lowercase, snake_case canonical strings.
- Collapse synonyms to ONE canonical predicate. Prefer these when they fit:
  has_role, headcount, status, owns, uses, decision, deadline, location, name.
- Map paraphrases to the same (subject, predicate) as the equivalent chat claim would.
- object may be a natural-language phrase. Only extract asserted facts. If none, return [].

Text:
{text}

Return ONLY the JSON array, no prose."""


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


def _cache_dir() -> Path:
    return _PROSE_DRIFT_CACHE_DIR


def _chroma_path() -> Path:
    return CHROMADB_DIR


def _extract_cache_key_text(messages: list[dict]) -> str:
    minimal = [{"role": m["role"], "text": m["text"]} for m in messages]
    return json.dumps(
        minimal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_path(project: str) -> Path:
    return _cache_dir() / f"{project}.db"


def _init_cache(project: str) -> sqlite3.Connection:
    cdir = _cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_cache_path(project)))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_facts (
            source_relpath TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            triples_json TEXT NOT NULL,
            PRIMARY KEY (source_relpath, sha256)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extraction_cache (
            text_sha TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            triples_json TEXT NOT NULL,
            PRIMARY KEY (text_sha, model, prompt_version)
        )
        """
    )
    conn.commit()
    return conn


def _strip_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.startswith("json"):
                body = body[4:]
            return body.strip()
    return raw


def extract_facts(messages: list[dict], project: str) -> list[Triple]:
    """Extract triples from user-authored messages; cached per (text_sha, model, prompt_version)."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return []
    key_text = _extract_cache_key_text(user_msgs)
    text_sha = _sha256(key_text)

    conn = _init_cache(project)
    try:
        row = conn.execute(
            "SELECT triples_json FROM extraction_cache "
            "WHERE text_sha=? AND model=? AND prompt_version=?",
            (text_sha, PROSE_EXTRACTION_MODEL, EXTRACTION_PROMPT_VERSION),
        ).fetchone()
        if row is not None:
            return [Triple(**t) for t in json.loads(row[0])]

        import anthropic

        client = anthropic.Anthropic()
        prompt = EXTRACTION_PROMPT.format(
            messages="\n".join(
                f"[{m['role']}]: {m['text']}" for m in user_msgs
            )
        )
        # NOTE: no `temperature` kwarg. claude-opus-4-7 rejects it (400).
        resp = client.messages.create(
            model=PROSE_EXTRACTION_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        parsed: list[dict[str, Any]] = json.loads(_strip_fence(raw))
        conn.execute(
            "INSERT INTO extraction_cache VALUES (?, ?, ?, ?)",
            (
                text_sha,
                PROSE_EXTRACTION_MODEL,
                EXTRACTION_PROMPT_VERSION,
                json.dumps(parsed),
            ),
        )
        conn.commit()
        return [Triple(**t) for t in parsed]
    finally:
        conn.close()


def extract_facts_from_doc(
    text: str, source_relpath: str, project: str = "default"
) -> list[Triple]:
    """Query-time doc-side extraction. Caches in extraction_cache + doc_facts.

    Doc text is hashed raw (no canonicalization — no role/timestamp metadata).
    doc_facts.sha256 == extraction_cache.text_sha by construction.
    Never writes Chroma; v1 doc triples are compared in-memory only.
    """
    text_sha = _sha256(text)
    conn = _init_cache(project)
    try:
        row = conn.execute(
            "SELECT triples_json FROM extraction_cache "
            "WHERE text_sha=? AND model=? AND prompt_version=?",
            (text_sha, PROSE_EXTRACTION_MODEL, EXTRACTION_PROMPT_VERSION),
        ).fetchone()
        if row is not None:
            triples = [Triple(**t) for t in json.loads(row[0])]
            _ensure_doc_facts_row(conn, source_relpath, text_sha, row[0])
            conn.commit()
            return triples

        import anthropic

        client = anthropic.Anthropic()
        prompt = DOC_EXTRACTION_PROMPT.format(text=text)
        # NOTE: no `temperature` kwarg. claude-opus-4-7 rejects it (400).
        resp = client.messages.create(
            model=PROSE_EXTRACTION_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        parsed: list[dict[str, Any]] = json.loads(_strip_fence(raw))
        triples_json = json.dumps(parsed)
        conn.execute(
            "INSERT OR REPLACE INTO extraction_cache VALUES (?, ?, ?, ?)",
            (text_sha, PROSE_EXTRACTION_MODEL, EXTRACTION_PROMPT_VERSION, triples_json),
        )
        _ensure_doc_facts_row(conn, source_relpath, text_sha, triples_json)
        conn.commit()
        return [Triple(**t) for t in parsed]
    finally:
        conn.close()


def _ensure_doc_facts_row(conn, source_relpath: str, sha256: str, triples_json: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO doc_facts VALUES (?, ?, ?)",
        (source_relpath, sha256, triples_json),
    )


def _get_prose_facts_collection(project: str):
    path = _chroma_path()
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=f"{project}-prose-facts")


def _voyage_embed(text: str) -> list[float]:
    vo = get_voyage_client()
    result = vo.embed([text], model=SESSIONS_MODEL, input_type="document")
    return result.embeddings[0]


def _new_row_metadata(
    triple: Triple, source_id: str, chain_key: str, now_ms: int, version: int
) -> dict[str, Any]:
    return {
        "subject": triple.subject,
        "predicate": triple.predicate,
        "object": triple.object,
        "chain_key": chain_key,
        "valid_from": now_ms,
        "invalid_at": INVALID_AT_NONE_SENTINEL,
        "is_current": True,
        "source_id": source_id,
        "version": version,
    }


def add_fact_with_state_machine(
    triple: Triple, source_id: str, project: str
) -> str:
    """Apply INSERT / NOOP / SUPERSEDE semantics for one triple. Returns event name."""
    collection = _get_prose_facts_collection(project)
    chain_key = f"{triple.subject}|{triple.predicate}"
    now_ms = int(time.time() * 1000)

    result = collection.get(
        where={
            "$and": [
                {"chain_key": chain_key},
                {"is_current": True},
            ]
        }
    )
    ids = result.get("ids") or []
    metas = result.get("metadatas") or []

    # Fix 2 repair branch: multiple is_current=True for same chain_key
    if len(ids) > 1:
        rows = sorted(
            zip(ids, metas),
            key=lambda x: x[1].get("version", 0),
            reverse=True,
        )
        operative_id, operative_meta = rows[0]
        for lower_id, lower_meta in rows[1:]:
            repaired = dict(lower_meta)
            repaired["is_current"] = False
            repaired["invalid_at"] = now_ms
            collection.update(ids=[lower_id], metadatas=[repaired])
        ids = [operative_id]
        metas = [operative_meta]

    if not ids:
        new_id = str(uuid.uuid4())
        doc_text = f"{triple.subject} {triple.predicate} {triple.object}"
        embedding = _voyage_embed(doc_text)
        collection.add(
            ids=[new_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[
                _new_row_metadata(triple, source_id, chain_key, now_ms, version=1)
            ],
        )
        return EVENT_INSERT

    operative_meta = metas[0]
    if operative_meta["object"] == triple.object:
        return EVENT_NOOP

    # SUPERSEDE (Fix 2 reorder: add new FIRST, flip old SECOND)
    old_id = ids[0]
    old_version = int(operative_meta.get("version", 1))
    new_id = str(uuid.uuid4())
    doc_text = f"{triple.subject} {triple.predicate} {triple.object}"
    embedding = _voyage_embed(doc_text)
    collection.add(
        ids=[new_id],
        embeddings=[embedding],
        documents=[doc_text],
        metadatas=[
            _new_row_metadata(
                triple, source_id, chain_key, now_ms, version=old_version + 1
            )
        ],
    )
    flipped = dict(operative_meta)
    flipped["invalid_at"] = now_ms
    flipped["is_current"] = False
    collection.update(ids=[old_id], metadatas=[flipped])
    return EVENT_SUPERSEDE
