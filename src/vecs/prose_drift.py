"""prose_drift: bi-temporal INSERT/NOOP/SUPERSEDE state machine + extraction cache.

Design: docs/features/prose-staleness-detector-design-v1.md (V+).
Phase 7 dry-run subtask. v1 covers the state machine + extract_facts +
SQLite verdict cache. CLI / MCP / indexer wire-in land later.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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

# stage-2 recall: embedding-similarity fallback + selective contradiction-judge.
PROSE_JUDGE_MODEL = "claude-opus-4-7"
JUDGE_PROMPT_VERSION = "v1"
STAGE2_SIM_THRESHOLD = 0.85

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

JUDGE_PROMPT = """You are judging whether two factual claims about the same project or team CONTRADICT each other.

Claim A (from a document): {doc}
Claim B (from chat history): {chat}

Do these two claims contradict — i.e., they cannot both be true of the same real-world subject at the same time? Account for paraphrase, synonyms, and different predicates that describe the same underlying fact (e.g. "team employs Sasha as backend engineer" contradicts "team has no backend developer").

Return ONLY a JSON object, no other prose:
{{"contradicts": <true|false>, "confidence": <float 0.0-1.0>, "reason": <one short sentence justifying the verdict>}}"""


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class Verdict:
    contradicts: bool
    confidence: float
    reason: str


class ProseDriftError(Exception):
    """Base for prose-drift errors."""


class AnthropicUnavailable(ProseDriftError):
    pass


class AnthropicKeyMissing(ProseDriftError):
    pass


class ProseDriftDisabled(ProseDriftError):
    pass


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    code: str | None = None
    detail: str | None = None


def _anthropic_importable() -> tuple[bool, str]:
    try:
        import anthropic  # noqa: F401
        return (True, "")
    except Exception as e:  # ImportError or transitive failure
        return (False, str(e))


def _preflight_global(config) -> PreflightResult:
    """Invariants that apply to ANY prose-drift invocation."""
    importable, err = _anthropic_importable()
    if not importable:
        return PreflightResult(False, "anthropic_unavailable", err)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return PreflightResult(False, "anthropic_key_missing")
    return PreflightResult(True)


def _preflight_project(config, project_name: str) -> PreflightResult:
    """Invariants specific to one project."""
    if project_name not in config.projects:
        return PreflightResult(False, "project_unknown", project_name)
    if not config.projects[project_name].prose_drift_enabled:
        return PreflightResult(False, "prose_drift_disabled", project_name)
    return PreflightResult(True)


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS judge_cache (
            doc_triple_json TEXT NOT NULL,
            chat_triple_json TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            verdict_json TEXT NOT NULL,
            PRIMARY KEY (doc_triple_json, chat_triple_json, model, prompt_version)
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


def _get_docs_collection(project: str):
    path = _chroma_path()
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=f"{project}-docs")


def iterate_indexed_docs(project: str):
    """Yield (chunk_text, source_relpath) for every indexed doc-chunk.

    source_relpath is read from metadata key `file_path` (cite doc_chunker.py:103),
    relative to project.docs_dir. No fallback key: chunks lacking `file_path`
    are skipped.
    """
    coll = _get_docs_collection(project)
    res = coll.get(include=["documents", "metadatas"])
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for text, meta in zip(docs, metas):
        relpath = (meta or {}).get("file_path")
        if relpath is None:
            continue
        yield (text, relpath)


def _exact_drift_entry(dt: Triple, cur: dict, source_relpath: str) -> dict:
    return {
        "subject": dt.subject,
        "predicate": dt.predicate,
        "match_type": "exact",
        "doc": {"object": dt.object, "source": source_relpath},
        "chat": {
            "subject": cur.get("subject", dt.subject),
            "predicate": cur.get("predicate", dt.predicate),
            "object": cur["object"],
            "session_id": cur.get("source_id", "?"),
        },
        "chat_history_versions": int(cur.get("version", 1)),
    }


def _semantic_drift_entry(
    dt: Triple, cand_meta: dict, source_relpath: str, sim: float, confidence: float
) -> dict:
    return {
        "subject": dt.subject,
        "predicate": dt.predicate,
        "match_type": "semantic",
        "similarity": round(float(sim), 4),
        "confidence": float(confidence),
        "doc": {"object": dt.object, "source": source_relpath},
        "chat": {
            "subject": cand_meta.get("subject"),
            "predicate": cand_meta.get("predicate"),
            "object": cand_meta.get("object"),
            "session_id": cand_meta.get("source_id", "?"),
        },
        "chat_history_versions": int(cand_meta.get("version", 1)),
    }


def find_prose_drift(project) -> dict:
    """Compare doc-side triples (query-time) against current session-side facts.

    Scheduled/on-demand recrawl over indexed doc-chunks. Two detection stages:
      1. exact — a doc (subject, predicate) whose object differs from the current
         chat fact on the same chain_key.
      2. semantic (stage-2) — on a chain_key MISS, the single most cosine-similar
         current fact (>= STAGE2_SIM_THRESHOLD) is escalated to ONE contradiction
         judge; a positive verdict surfaces a cross-predicate / paraphrase drift.

    Returns the MCP payload dict:
      {"drift": [...], "facts_scanned": int, "facts_scanned_docs": int,
       "stage2_judge_calls": int, "stage2_judge_errors": int, "project": str}
    """
    name = project.name
    facts = _get_prose_facts_collection(name)
    current_rows = _load_current_rows(facts)
    by_chain = {meta["chain_key"]: meta for meta, _emb in current_rows}
    drift: list[dict] = []
    docs_scanned = 0
    judge_calls = 0
    judge_errors = 0
    for chunk_text, source_relpath in iterate_indexed_docs(name):
        docs_scanned += 1
        for dt in extract_facts_from_doc(chunk_text, source_relpath, project=name):
            chain_key = f"{dt.subject}|{dt.predicate}"
            cur = by_chain.get(chain_key)
            if cur is not None:
                if cur["object"] != dt.object:
                    drift.append(_exact_drift_entry(dt, cur, source_relpath))
                continue
            # chain_key MISS -> stage-2 semantic fallback.
            if not current_rows:
                continue
            doc_emb = _voyage_embed(f"{dt.subject} {dt.predicate} {dt.object}")
            cand = _best_semantic_candidate(doc_emb, current_rows)
            if cand is None or cand[1] < STAGE2_SIM_THRESHOLD:
                continue
            cand_meta, sim = cand
            judge_calls += 1
            try:
                verdict = _judge_contradiction(dt, cand_meta, name)
            except Exception:
                judge_errors += 1
                continue
            if verdict.contradicts:
                drift.append(
                    _semantic_drift_entry(
                        dt, cand_meta, source_relpath, sim, verdict.confidence
                    )
                )
    drift.sort(key=lambda d: (d["subject"], d["predicate"], d.get("match_type", "")))
    facts_scanned = len(facts.get(where={"is_current": True}).get("ids") or [])
    return {
        "drift": drift,
        "facts_scanned": facts_scanned,
        "facts_scanned_docs": docs_scanned,
        "stage2_judge_calls": judge_calls,
        "stage2_judge_errors": judge_errors,
        "project": name,
    }


def _voyage_embed(text: str) -> list[float]:
    vo = get_voyage_client()
    result = vo.embed([text], model=SESSIONS_MODEL, input_type="document")
    return result.embeddings[0]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Zero-norm vectors return 0.0 (no ZeroDivisionError)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_current_rows(collection) -> list[tuple[dict, list[float]]]:
    """All is_current facts as (meta, embedding), loaded once per scan.

    Deduped to the max-version row per chain_key (read-only; mirrors
    _current_row_for_chain's crash tolerance without repair writes).
    """
    res = collection.get(
        where={"is_current": True}, include=["metadatas", "embeddings"]
    )
    metas = res.get("metadatas") or []
    embs = res.get("embeddings")
    if embs is None:
        embs = []
    best: dict[str, tuple[dict, list[float]]] = {}
    for meta, emb in zip(metas, embs):
        ck = meta.get("chain_key")
        if ck is None:
            continue
        prev = best.get(ck)
        if prev is None or meta.get("version", 0) > prev[0].get("version", 0):
            best[ck] = (meta, list(emb))
    return list(best.values())


def _best_semantic_candidate(
    doc_emb: list[float], rows: list[tuple[dict, list[float]]]
) -> tuple[dict, float] | None:
    """Highest-cosine (meta, similarity) among rows; None if rows is empty.

    Threshold gating is the caller's job — this only ranks.
    """
    best: tuple[dict, float] | None = None
    for meta, emb in rows:
        sim = _cosine(doc_emb, emb)
        if best is None or sim > best[1]:
            best = (meta, sim)
    return best


def _triple_key(subject: str, predicate: str, object: str) -> str:
    return json.dumps(
        {"subject": subject, "predicate": predicate, "object": object},
        sort_keys=True,
        separators=(",", ":"),
    )


def _judge_contradiction(doc_triple: Triple, chat_meta: dict, project: str) -> Verdict:
    """Ask ONE Opus judge whether a doc triple contradicts a chat fact. Cached.

    chat_meta is a `<project>-prose-facts` row metadata dict; only its
    (subject, predicate, object) participate in the verdict + cache key.
    """
    doc_key = _triple_key(doc_triple.subject, doc_triple.predicate, doc_triple.object)
    chat_key = _triple_key(
        chat_meta.get("subject", ""),
        chat_meta.get("predicate", ""),
        chat_meta.get("object", ""),
    )
    conn = _init_cache(project)
    try:
        row = conn.execute(
            "SELECT verdict_json FROM judge_cache "
            "WHERE doc_triple_json=? AND chat_triple_json=? AND model=? AND prompt_version=?",
            (doc_key, chat_key, PROSE_JUDGE_MODEL, JUDGE_PROMPT_VERSION),
        ).fetchone()
        if row is not None:
            return Verdict(**json.loads(row[0]))

        import anthropic

        client = anthropic.Anthropic()
        prompt = JUDGE_PROMPT.format(
            doc=f'subject="{doc_triple.subject}" predicate="{doc_triple.predicate}" object="{doc_triple.object}"',
            chat=f'subject="{chat_meta.get("subject", "")}" predicate="{chat_meta.get("predicate", "")}" object="{chat_meta.get("object", "")}"',
        )
        # NOTE: no `temperature` kwarg. claude-opus-4-7 rejects it (400).
        resp = client.messages.create(
            model=PROSE_JUDGE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(_strip_fence(resp.content[0].text))
        verdict = Verdict(
            contradicts=bool(parsed["contradicts"]),
            confidence=float(parsed.get("confidence", 0.0)),
            reason=str(parsed.get("reason", "")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO judge_cache VALUES (?, ?, ?, ?, ?)",
            (
                doc_key,
                chat_key,
                PROSE_JUDGE_MODEL,
                JUDGE_PROMPT_VERSION,
                json.dumps(
                    {
                        "contradicts": verdict.contradicts,
                        "confidence": verdict.confidence,
                        "reason": verdict.reason,
                    }
                ),
            ),
        )
        conn.commit()
        return verdict
    finally:
        conn.close()


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
