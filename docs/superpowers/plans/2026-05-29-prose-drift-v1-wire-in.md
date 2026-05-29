# Prose-Drift v1 Wire-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the shipped prose-drift state-machine core (`src/vecs/prose_drift.py`, dry-run subtask only) into a working `vecs prose-drift` CLI subcommand + `mcp__vecs__prose_drift` MCP tool that detects contradictions between indexed docs and indexed chat sessions, wired into the indexer.

**Architecture:** Keep the converged bi-temporal hand-build (verdict from `docs/research/prose-drift-review-and-sota-2026-05-29.md`: PROCEED-WITH-MODS). Session-side triples are written to the existing `<project>-prose-facts` Chroma collection at index time via the existing state machine. Doc-side triples are extracted on-demand at query time (cached in SQLite), then compared against the current-state session rows. v1 detects exact `(subject, predicate)` object-collisions only; the recall upgrade (embedding-similarity + LLM contradiction judge) is parked for v2. v1 adds cheap predicate canonicalization to the extraction prompt to raise exact-collision rate, an xfail test encoding the known paraphrase boundary, and frames the feature as scheduled/on-demand recrawl.

**Tech Stack:** Python 3, Click (CLI), FastMCP (MCP), ChromaDB (vector store + fact store), Voyage AI (`voyage-3` embeddings), Anthropic (`claude-opus-4-7` extraction), SQLite (verdict cache), pytest + monkeypatch (tests, no new dev deps).

**Source of truth for behavior:** `docs/features/prose-staleness-detector-design-v1.md` (Phases 1–8) and `docs/features/prose-staleness-detector/acceptance.md`. This plan implements the parts the dry-run did NOT: `extract_facts_from_doc`, `iterate_indexed_docs`, `find_prose_drift`, preflight, CLI, MCP, indexer wire-in, config flag — plus the 5 review fold-ins.

**Pre-flight (run once before Task 1):**
```bash
cd /Users/darynavoloshyna/repo/vecs
git checkout -b feat/prose-drift-v1-wire-in
uv run pytest -q tests/test_prose_drift.py
```
Expected: existing 22 dry-run tests PASS (green baseline). `anthropic==0.103.1` is already pinned in `pyproject.toml`, so `test_pyproject_pins_anthropic_exact_version` passes.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/vecs/config.py` | Modify | Add `prose_drift_enabled: bool = False` to `ProjectConfig`; parse YAML key in `load_config`; add `prose_facts_collection` property. |
| `src/vecs/prose_drift.py` | Modify | Add `DOC_EXTRACTION_PROMPT`, canonicalized `EXTRACTION_PROMPT` (bump version), `extract_facts_from_doc`, `_get_docs_collection`, `iterate_indexed_docs`, `_current_row_for_chain`, `find_prose_drift`, `PreflightResult` + `_anthropic_importable` + `_preflight_global` + `_preflight_project`, exception hierarchy. |
| `src/vecs/cli.py` | Modify | New `@main.command()` `prose-drift` (lazy import; scheduled-recrawl framing; exit codes 0/1/2/3; `--limit`; v1-boundary footer). |
| `src/vecs/mcp_server.py` | Modify | New `@mcp.tool()` `prose_drift(project=None)` (lazy import; dict return; `project=None` fan-out). |
| `src/vecs/indexer.py` | Modify | Stash `file_messages` in the `:904` loop; insert prose-drift facet block after success-path `manifest.save()` at `:964`. |
| `src/vecs/CLAUDE.md` | Modify | `staleness_check` scalar→list; new Tests bullet; v1-boundary note. |
| `tests/test_prose_drift.py` | Modify | Add Phase-5 unit tests for all new functions + the paraphrase-miss xfail. |
| `tests/test_prose_drift_wire_in.py` | Create | Indexer/CLI/MCP integration tests (kept separate so the core test file stays focused). |
| `tests/fixtures/prose_drift/docs/team.md` | Create | BE-dev doc fixture ("no backend developer"). |
| `tests/fixtures/prose_drift/sessions/be_dev_announce.jsonl` | Create | BE-dev session fixture ("hired Sasha as BE dev"). |

**Existing test helpers to reuse** (already in `tests/test_prose_drift.py`): `_isolate_chroma_and_cache` (autouse; pins `_chroma_path`/`_cache_dir` to `tmp_path`), `fake_voyage` (recording embed fake), `fake_anthropic` (recording `anthropic.Anthropic` fake with mutable `state["response_text"]`).

---

## Task 1: Config flag `prose_drift_enabled`

**Files:**
- Modify: `src/vecs/config.py` (ProjectConfig dataclass ~`:44-65`; `load_config` project loop ~`:217-223`)
- Test: `tests/test_prose_drift_wire_in.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_prose_drift_wire_in.py`:
```python
"""Phase 5 wire-in tests: config flag, CLI, MCP, indexer facet."""
from __future__ import annotations

from pathlib import Path

import pytest

from vecs.config import ProjectConfig, load_config


def test_project_config_defaults_prose_drift_disabled():
    p = ProjectConfig(name="x")
    assert p.prose_drift_enabled is False


def test_prose_facts_collection_name():
    p = ProjectConfig(name="vecs")
    assert p.prose_facts_collection == "vecs-prose-facts"


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body)
    return cfg


def test_config_yaml_missing_field_loads_false(tmp_path):
    cfg = _write_config(tmp_path, """
projects:
  vecs:
    docs_dir: /tmp/docs
""")
    config = load_config(cfg)
    assert config.projects["vecs"].prose_drift_enabled is False


def test_config_yaml_prose_drift_enabled_true_loads(tmp_path):
    cfg = _write_config(tmp_path, """
projects:
  vecs:
    docs_dir: /tmp/docs
    prose_drift_enabled: true
""")
    config = load_config(cfg)
    assert config.projects["vecs"].prose_drift_enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q`
Expected: FAIL — `AttributeError: 'ProjectConfig' object has no attribute 'prose_drift_enabled'`.

- [ ] **Step 3: Add the field + property to `ProjectConfig`**

In `src/vecs/config.py`, add the field after `codex_cwds` (`:53`):
```python
    codex_cwds: list[Path] = field(default_factory=list)
    prose_drift_enabled: bool = False
```
And add the property after `docs_collection` (`:65`):
```python
    @property
    def prose_facts_collection(self) -> str:
        return f"{self.name}-prose-facts"
```

- [ ] **Step 4: Parse the YAML key in `load_config`**

In `src/vecs/config.py`, in the `config.projects[name] = ProjectConfig(...)` constructor inside `load_config` (`:217-223`), add the field:
```python
        config.projects[name] = ProjectConfig(
            name=name,
            code_dirs=code_dirs,
            sessions_dirs=sessions_dirs,
            docs_dir=Path(proj["docs_dir"]) if proj.get("docs_dir") else None,
            codex_cwds=codex_cwds,
            prose_drift_enabled=bool(proj.get("prose_drift_enabled", False)),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/vecs/config.py tests/test_prose_drift_wire_in.py
git commit -m "feat(prose-drift): add prose_drift_enabled config flag + prose_facts_collection"
```

---

## Task 2: Predicate canonicalization in the extraction prompt (fold-in B)

**Files:**
- Modify: `src/vecs/prose_drift.py` (`EXTRACTION_PROMPT` `:36-46`, `EXTRACTION_PROMPT_VERSION` `:24`)
- Test: `tests/test_prose_drift.py`

**Why:** Review B1 — the exact-string `chain_key` misses paraphrase. Canonicalization guidance in the prompt raises the exact-collision rate cheaply (does NOT fix cross-predicate misses — that is the v2 judge). Bumping `EXTRACTION_PROMPT_VERSION` invalidates the cache so old extractions re-run under the new guidance.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift.py`:
```python
def test_extraction_prompt_version_bumped_to_v2():
    assert prose_drift.EXTRACTION_PROMPT_VERSION == "v2"


def test_extraction_prompt_has_canonicalization_guidance():
    p = prose_drift.EXTRACTION_PROMPT
    assert "canonical" in p.lower()
    # A controlled-vocabulary hint and at least one worked example must be present.
    assert "has_role" in p
    assert "Example" in p
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift.py -q -k "prompt_version_bumped or canonicalization_guidance"`
Expected: FAIL — version is still `"v1"`; guidance substrings absent.

- [ ] **Step 3: Bump version + rewrite the prompt**

In `src/vecs/prose_drift.py`, change `:24`:
```python
EXTRACTION_PROMPT_VERSION = "v2"
```
Replace `EXTRACTION_PROMPT` (`:36-46`) with:
```python
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
```

- [ ] **Step 4: Run to verify pass (and no regressions)**

Run: `uv run pytest tests/test_prose_drift.py -q`
Expected: PASS. Note: `test_cache_invalidates_on_prompt_version_bump` still passes (it monkeypatches the version independently). Cache-hit tests still pass (they use a fixed version within the test).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "feat(prose-drift): canonicalization guidance in extraction prompt, bump to v2"
```

---

## Task 3: `extract_facts_from_doc` + doc-side cache

**Files:**
- Modify: `src/vecs/prose_drift.py` (add `DOC_EXTRACTION_PROMPT`, `extract_facts_from_doc`)
- Test: `tests/test_prose_drift.py`

**Spec:** design `:177`, acceptance `:48-49`, `:431`. Reads `extraction_cache` keyed on `(sha256(raw_text), MODEL, PROMPT_VERSION)`; on miss calls Anthropic; writes BOTH `extraction_cache` AND `doc_facts` with `doc_facts.sha256 == extraction_cache.text_sha`. Doc text is NOT canonicalized (no role/timestamp to strip). Never writes Chroma.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift.py`:
```python
def test_extract_facts_from_doc_returns_triples(fake_anthropic):
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    out = prose_drift.extract_facts_from_doc("Our team has no backend developer.", "team.md")
    assert len(out) == 1
    assert out[0] == prose_drift.Triple("team", "has_role", "no backend developer")


def test_doc_extract_writes_both_tables_with_matching_sha(fake_anthropic):
    import sqlite3
    prose_drift.extract_facts_from_doc("Our team has no backend developer.", "team.md")
    db = prose_drift._cache_path("default")
    conn = sqlite3.connect(str(db))
    doc_sha = conn.execute("SELECT sha256 FROM doc_facts").fetchone()[0]
    ext_sha = conn.execute("SELECT text_sha FROM extraction_cache").fetchone()[0]
    conn.close()
    assert doc_sha == ext_sha


def test_doc_extract_cache_hit_skips_anthropic(fake_anthropic):
    prose_drift.extract_facts_from_doc("same text", "team.md")
    prose_drift.extract_facts_from_doc("same text", "team.md")
    assert len(fake_anthropic["calls"]) == 1


def test_doc_extract_cache_miss_on_text_change(fake_anthropic):
    prose_drift.extract_facts_from_doc("text A", "team.md")
    prose_drift.extract_facts_from_doc("text B", "team.md")
    assert len(fake_anthropic["calls"]) == 2
```
Note: `extract_facts_from_doc` uses a fixed cache project name `"default"` (doc cache is per running query; the per-project DB is selected by `find_prose_drift`, which passes the project name — see Task 5). For these unit tests the helper must accept a `project` argument. Adjust signature accordingly (see Step 3).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift.py -q -k extract_facts_from_doc or doc_extract`
Expected: FAIL — `AttributeError: ... has no attribute 'extract_facts_from_doc'`.

- [ ] **Step 3: Implement `extract_facts_from_doc`**

In `src/vecs/prose_drift.py`, add after the `EXTRACTION_PROMPT` block:
```python
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
```
Add the function after `extract_facts` (`:167`):
```python
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
```
Note: existing `_init_cache` DDL for `doc_facts` is `(source_relpath, sha256, triples_json, PRIMARY KEY(source_relpath, sha256))` — 3 columns. The `INSERT OR REPLACE` above matches that 3-column shape. Existing `extraction_cache` INSERT in `extract_facts` uses plain `INSERT`; the doc path uses `INSERT OR REPLACE` to tolerate a pre-existing session-side row with the same `text_sha` (rare; harmless overwrite with identical content).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift.py -q -k "extract_facts_from_doc or doc_extract"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "feat(prose-drift): query-time doc-side extraction with dual-table cache"
```

---

## Task 4: `iterate_indexed_docs`

**Files:**
- Modify: `src/vecs/prose_drift.py` (add `_get_docs_collection`, `iterate_indexed_docs`)
- Test: `tests/test_prose_drift.py`

**Spec:** design `:180`, acceptance `:60`, `:439`. Iterates `<project>-docs` Chroma collection, yields `(chunk_text, source_relpath)` reading `metadatas[i]["file_path"]` with NO fallback key.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift.py`:
```python
def test_iterate_indexed_docs_yields_text_and_file_path(fake_voyage):
    project = "p_docs"
    coll = prose_drift._get_docs_collection(project)
    coll.add(
        ids=["d1", "d2"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
        documents=["team has no backend dev", "we use postgres"],
        metadatas=[{"file_path": "team.md"}, {"file_path": "stack.md"}],
    )
    out = sorted(prose_drift.iterate_indexed_docs(project))
    assert out == [("team has no backend dev", "team.md"), ("we use postgres", "stack.md")]


def test_iterate_indexed_docs_no_fallback_key(fake_voyage):
    project = "p_docs_nofallback"
    coll = prose_drift._get_docs_collection(project)
    coll.add(
        ids=["d1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["orphan chunk"],
        metadatas=[{"path": "team.md"}],  # wrong key — must NOT be read as file_path
    )
    out = list(prose_drift.iterate_indexed_docs(project))
    assert out == [], "chunk without file_path metadata must be skipped (no fallback)"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift.py -q -k iterate_indexed_docs`
Expected: FAIL — no attribute `_get_docs_collection`.

- [ ] **Step 3: Implement**

In `src/vecs/prose_drift.py`, add after `_get_prose_facts_collection` (`:174`):
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift.py -q -k iterate_indexed_docs`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "feat(prose-drift): iterate_indexed_docs over <project>-docs collection"
```

---

## Task 5: `find_prose_drift` — the drift comparison (core deliverable)

**Files:**
- Modify: `src/vecs/prose_drift.py` (add `_current_row_for_chain`, `find_prose_drift`)
- Test: `tests/test_prose_drift.py`

**Spec:** design `:181`, acceptance `:22` (return shape), `:140` (`facts_scanned`/`facts_scanned_docs`), `:440`, `:448-450`, `:472-474`. Returns the exact MCP payload dict. `doc.source` stores the BARE relpath (CLI prepends `<project>/` at format time per acceptance `:21-22`).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift.py`:
```python
def _seed_doc(project, text, relpath):
    coll = prose_drift._get_docs_collection(project)
    coll.add(
        ids=[f"d-{relpath}"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=[text],
        metadatas=[{"file_path": relpath}],
    )


class _Proj:
    def __init__(self, name):
        self.name = name


def test_find_prose_drift_no_drift_when_no_facts(fake_anthropic):
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc("p_nd", "team has no backend dev", "team.md")
    report = prose_drift.find_prose_drift(_Proj("p_nd"))
    assert report["drift"] == []
    assert report["facts_scanned"] == 0
    assert report["facts_scanned_docs"] == 1
    assert report["project"] == "p_nd"


def test_find_prose_drift_surfaces_collision(fake_anthropic, fake_voyage):
    project = "p_drift"
    # Session-side current row: team has_role "Sasha is backend engineer"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "has_role", "sasha is backend engineer"),
        "be_dev_announce", project,
    )
    # Doc-side claims the opposite.
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert len(report["drift"]) == 1
    d = report["drift"][0]
    assert d["subject"] == "team" and d["predicate"] == "has_role"
    assert d["doc"] == {"object": "no backend developer", "source": "team.md"}
    assert d["chat"]["object"] == "sasha is backend engineer"
    assert d["chat"]["session_id"] == "be_dev_announce"
    assert d["chat_history_versions"] == 1
    assert report["facts_scanned"] == 1
    assert report["facts_scanned_docs"] == 1


def test_find_prose_drift_sorted_by_subject_predicate(fake_anthropic, fake_voyage):
    project = "p_sort"
    for subj in ("zeta", "alpha"):
        prose_drift.add_fact_with_state_machine(
            prose_drift.Triple(subj, "p", "chat_val"), "s", project,
        )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"zeta","predicate":"p","object":"doc_val"},'
        '{"subject":"alpha","predicate":"p","object":"doc_val"}]'
    )
    _seed_doc(project, "irrelevant", "x.md")
    report = prose_drift.find_prose_drift(_Proj(project))
    subjects = [d["subject"] for d in report["drift"]]
    assert subjects == ["alpha", "zeta"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift.py -q -k find_prose_drift`
Expected: FAIL — no attribute `find_prose_drift`.

- [ ] **Step 3: Implement**

In `src/vecs/prose_drift.py`, add after `iterate_indexed_docs`:
```python
def _current_row_for_chain(collection, chain_key: str) -> dict | None:
    """Read-only lookup of the operative current row for a chain. No repair writes."""
    res = collection.get(
        where={"$and": [{"chain_key": chain_key}, {"is_current": True}]}
    )
    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    if not ids:
        return None
    if len(ids) > 1:
        # Transient post-crash state: highest-version row is operative (read path
        # does not mutate; the next write-path call repairs via add_fact_with_state_machine).
        return max(metas, key=lambda m: m.get("version", 0))
    return metas[0]


def find_prose_drift(project) -> dict:
    """Compare doc-side triples (query-time) against current session-side facts.

    Scheduled/on-demand recrawl: iterates indexed doc-chunks, extracts triples,
    and reports each (subject, predicate) whose doc-object collides with a
    DIFFERENT current chat-object. v1 detects exact (subject, predicate)
    object-collisions only (see v1-boundary note in docs).

    Returns the MCP payload dict:
      {"drift": [...], "facts_scanned": int, "facts_scanned_docs": int, "project": str}
    """
    name = project.name
    facts = _get_prose_facts_collection(name)
    drift: list[dict] = []
    docs_scanned = 0
    for chunk_text, source_relpath in iterate_indexed_docs(name):
        docs_scanned += 1
        for dt in extract_facts_from_doc(chunk_text, source_relpath, project=name):
            chain_key = f"{dt.subject}|{dt.predicate}"
            cur = _current_row_for_chain(facts, chain_key)
            if cur is None or cur["object"] == dt.object:
                continue
            drift.append(
                {
                    "subject": dt.subject,
                    "predicate": dt.predicate,
                    "doc": {"object": dt.object, "source": source_relpath},
                    "chat": {
                        "object": cur["object"],
                        "session_id": cur.get("source_id", "?"),
                    },
                    "chat_history_versions": int(cur.get("version", 1)),
                }
            )
    drift.sort(key=lambda d: (d["subject"], d["predicate"]))
    facts_scanned = len(facts.get(where={"is_current": True}).get("ids") or [])
    return {
        "drift": drift,
        "facts_scanned": facts_scanned,
        "facts_scanned_docs": docs_scanned,
        "project": name,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift.py -q -k find_prose_drift`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "feat(prose-drift): find_prose_drift drift comparison + facts_scanned counts"
```

---

## Task 6: Paraphrase-miss xfail test (fold-in C)

**Files:**
- Test: `tests/test_prose_drift.py`

**Why:** Review B1 / `docs/spikes/approach1-spike.md:85-89` — the exact-string `chain_key` join MISSES cross-predicate/paraphrase contradictions. Encode the known recall hole as an `xfail` so it is regression-visible and flips to a real pass when the v2 LLM-judge stage lands. No production change in this task — this test documents the v1 boundary.

- [ ] **Step 1: Write the xfail test**

Add to `tests/test_prose_drift.py`:
```python
@pytest.mark.xfail(
    reason="v1 boundary: exact (subject,predicate) chain_key cannot match cross-predicate "
    "paraphrase. 'team|has_role:none' vs 'team|employs:sasha' do not collide. "
    "Recovered by the v2 embedding-similarity + LLM contradiction-judge stage "
    "(docs/features/prose-staleness-detector/v2-roadmap.md).",
    strict=True,
)
def test_cross_predicate_paraphrase_drift_is_detected(fake_anthropic, fake_voyage):
    project = "p_paraphrase"
    # Chat says the team EMPLOYS Sasha (predicate 'employs').
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "be_dev_announce", project,
    )
    # Doc says the team HAS_ROLE none (predicate 'has_role') — a real contradiction
    # a human would spot, but a different (subject,predicate) chain.
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    # v1 will report ZERO drift here (the hole). xfail(strict) asserts this fails today
    # and turns RED the day v2 closes the gap — forcing this test to be promoted.
    assert len(report["drift"]) == 1
```

- [ ] **Step 2: Run to verify it xfails (not errors)**

Run: `uv run pytest tests/test_prose_drift.py -q -k cross_predicate_paraphrase`
Expected: `xfailed` (1 xfailed, 0 failed). If it `XPASS`es, the join behavior changed — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_prose_drift.py
git commit -m "test(prose-drift): xfail encoding v1 cross-predicate paraphrase recall hole"
```

---

## Task 7: Preflight (`_preflight_global`, `_preflight_project`) + exception hierarchy

**Files:**
- Modify: `src/vecs/prose_drift.py`
- Test: `tests/test_prose_drift.py`

**Spec:** design `:338-377`, acceptance `:53-56`, `:429`, `:454-455`, `:463-464`. Two-stage split so MCP `project=None` is well-defined. Returns a small `PreflightResult`; CLI/MCP map codes. Codes: `anthropic_unavailable`, `anthropic_key_missing` (global); `project_unknown`, `prose_drift_disabled` (project).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift.py`:
```python
class _Cfg:
    def __init__(self, projects):
        self.projects = projects


def test_preflight_global_ok_when_key_set_and_anthropic_importable(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = prose_drift._preflight_global(_Cfg({}))
    assert r.ok is True


def test_preflight_global_err_key_missing(monkeypatch, fake_anthropic):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = prose_drift._preflight_global(_Cfg({}))
    assert r.ok is False and r.code == "anthropic_key_missing"


def test_preflight_global_err_anthropic_unavailable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(prose_drift, "_anthropic_importable", lambda: (False, "no module named anthropic"))
    r = prose_drift._preflight_global(_Cfg({}))
    assert r.ok is False and r.code == "anthropic_unavailable"
    assert r.detail


def test_preflight_project_err_unknown():
    r = prose_drift._preflight_project(_Cfg({}), "ghost")
    assert r.ok is False and r.code == "project_unknown" and r.detail == "ghost"


def test_preflight_project_err_disabled():
    p = ProjectConfig(name="vecs")  # prose_drift_enabled defaults False
    r = prose_drift._preflight_project(_Cfg({"vecs": p}), "vecs")
    assert r.ok is False and r.code == "prose_drift_disabled"


def test_preflight_project_ok_when_enabled():
    p = ProjectConfig(name="vecs", prose_drift_enabled=True)
    r = prose_drift._preflight_project(_Cfg({"vecs": p}), "vecs")
    assert r.ok is True
```
Add the import at the top of the test file if missing: `from vecs.config import ProjectConfig`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift.py -q -k preflight`
Expected: FAIL — no `_preflight_global` / `PreflightResult`.

- [ ] **Step 3: Implement**

In `src/vecs/prose_drift.py`, add near the top (after imports, before `PROSE_EXTRACTION_MODEL`):
```python
import os
```
And add (after the `Triple` dataclass `:53`):
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift.py -q -k preflight`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/prose_drift.py tests/test_prose_drift.py
git commit -m "feat(prose-drift): two-stage preflight + exception hierarchy"
```

---

## Task 8: CLI subcommand `vecs prose-drift` (fold-in D + E)

**Files:**
- Modify: `src/vecs/cli.py`
- Test: `tests/test_prose_drift_wire_in.py`

**Spec:** design `:362-365`, acceptance `:20-21`, `:37-41`, `:63`, `:452-458`, `:466-467`. Exit codes: 0 (no drift / no chat sessions), 1 (drift, possibly truncated), 2 (unknown / disabled project), 3 (anthropic missing / key missing). `--limit` default 50. Scheduled-recrawl framing in help + a v1-boundary footer to stderr when drift prints (fold-in D).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift_wire_in.py`:
```python
from click.testing import CliRunner

from vecs.cli import main


def _mock_report(monkeypatch, drift, facts_scanned=1, facts_docs=1, project="vecs"):
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "find_prose_drift", lambda proj: {
        "drift": drift, "facts_scanned": facts_scanned,
        "facts_scanned_docs": facts_docs, "project": project,
    })


def _enable_project(monkeypatch, name="vecs"):
    from vecs.config import ProjectConfig, VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"),
                     projects={name: ProjectConfig(name=name, prose_drift_enabled=True)})
    monkeypatch.setattr("vecs.cli.load_config", lambda *a, **k: cfg, raising=False)
    import vecs.config
    monkeypatch.setattr(vecs.config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def test_cli_no_drift_exit_0(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[], facts_scanned=3)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 0
    assert "no prose drift" in res.output


def test_cli_no_chat_sessions_exit_0(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[], facts_scanned=0)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 0
    assert "no chat sessions" in res.output


def test_cli_with_drift_exit_1(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    _mock_report(monkeypatch, drift=[{
        "subject": "team", "predicate": "has_role",
        "doc": {"object": "no backend developer", "source": "team.md"},
        "chat": {"object": "sasha", "session_id": "be_dev_announce"},
        "chat_history_versions": 1,
    }])
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 1
    assert 'team | has_role | doc="no backend developer" @ vecs/team.md' in res.output
    assert 'chat="sasha" @ session=be_dev_announce (chat_history_versions=1)' in res.output


def test_cli_unknown_project_exit_2(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)  # only 'vecs' exists
    res = CliRunner().invoke(main, ["prose-drift", "-p", "ghost"])
    assert res.exit_code == 2
    assert "unknown project: ghost" in res.output


def test_cli_disabled_project_exit_2(monkeypatch, fake_anthropic):
    from vecs.config import ProjectConfig, VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"),
                     projects={"vecs": ProjectConfig(name="vecs")})  # disabled
    import vecs.config
    monkeypatch.setattr(vecs.config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 2
    assert "prose drift not enabled for project vecs" in res.output


def test_cli_key_missing_exit_3(monkeypatch):
    _enable_project(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs"])
    assert res.exit_code == 3
    assert "ANTHROPIC_API_KEY not set" in res.output


def test_cli_limit_truncates_exit_1(monkeypatch, fake_anthropic):
    _enable_project(monkeypatch)
    many = [{
        "subject": f"s{i:03d}", "predicate": "p",
        "doc": {"object": "d", "source": "x.md"},
        "chat": {"object": "c", "session_id": "sess"},
        "chat_history_versions": 1,
    } for i in range(100)]
    _mock_report(monkeypatch, drift=many)
    res = CliRunner().invoke(main, ["prose-drift", "-p", "vecs", "--limit", "10"])
    assert res.exit_code == 1
    printed = [ln for ln in res.output.splitlines() if " | " in ln]
    assert len(printed) == 10
    assert "drift truncated: showing 10 of 100" in res.output
    # sort order preserved: first printed is s000
    assert printed[0].startswith("s000 | p |")
```
Note: `CliRunner` merges stderr into `res.output` by default, so stderr assertions work against `res.output`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q -k cli`
Expected: FAIL — `No such command 'prose-drift'`.

- [ ] **Step 3: Implement the CLI command**

In `src/vecs/cli.py`, add after the `status` command (before the `add` command, ~`:79`):
```python
@main.command("prose-drift")
@click.option("--project", "-p", required=True, help="Project to scan for prose drift.")
@click.option("--limit", default=50, help="Max drift lines to print (default 50).")
def prose_drift_cmd(project: str, limit: int):
    """Recrawl indexed docs and report contradictions vs current chat facts.

    On-demand recrawl (not write-time): compares (subject, predicate) facts
    extracted from indexed docs against the current state extracted from chat
    sessions. v1 detects exact (subject, predicate) object-collisions only.
    """
    import sys

    from vecs.config import load_config
    from vecs.prose_drift import _preflight_global, _preflight_project

    config = load_config()

    g = _preflight_global(config)
    if not g.ok:
        if g.code == "anthropic_unavailable":
            click.echo(f"anthropic not installed: pip install anthropic", err=True)
        else:
            click.echo("ANTHROPIC_API_KEY not set", err=True)
        raise SystemExit(3)

    p = _preflight_project(config, project)
    if not p.ok:
        if p.code == "project_unknown":
            click.echo(f"unknown project: {project}", err=True)
        else:
            click.echo(f"prose drift not enabled for project {project}", err=True)
        raise SystemExit(2)

    from vecs.prose_drift import find_prose_drift

    report = find_prose_drift(config.projects[project])

    if report["facts_scanned"] == 0:
        click.echo(f"no chat sessions for project {project}")
        raise SystemExit(0)

    drift = report["drift"]
    if not drift:
        click.echo("no prose drift")
        raise SystemExit(0)

    total = len(drift)
    shown = drift[:limit]
    for d in shown:
        click.echo(
            f"{d['subject']} | {d['predicate']} | "
            f"doc=\"{d['doc']['object']}\" @ {project}/{d['doc']['source']} "
            f"≠ chat=\"{d['chat']['object']}\" @ session={d['chat']['session_id']} "
            f"(chat_history_versions={d['chat_history_versions']})"
        )
    if total > limit:
        click.echo(f"drift truncated: showing {limit} of {total}", err=True)
    click.echo(
        "note: v1 detects exact (subject,predicate) object-collisions only; "
        "omission/temporal/cross-predicate drift is out of scope (see v2-roadmap).",
        err=True,
    )
    raise SystemExit(1)
```
Note: the `≠` is the `≠` character required by the acceptance line format (`:21`). Use the literal `≠` if the editor supports it; the escape is shown for safety.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q -k cli`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/cli.py tests/test_prose_drift_wire_in.py
git commit -m "feat(prose-drift): vecs prose-drift CLI subcommand (recrawl framing + exit codes)"
```

---

## Task 9: MCP tool `prose_drift`

**Files:**
- Modify: `src/vecs/mcp_server.py`
- Test: `tests/test_prose_drift_wire_in.py`

**Spec:** design `:367-377`, acceptance `:22-34`, `:40-41`, `:459-465`. `project=None` → fan out over `prose_drift_enabled` projects (skip disabled silently, NOT via `_preflight_project`); `{}` if none enabled. Global preflight failure → single-error dict. Named project → `_preflight_project` maps to single-error dict.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift_wire_in.py`:
```python
def _cfg_with(monkeypatch, projects):
    from vecs.config import VecsConfig
    cfg = VecsConfig(path=Path("/tmp/none.yaml"), projects=projects)
    import vecs.mcp_server
    monkeypatch.setattr(vecs.mcp_server, "load_config", lambda *a, **k: cfg)
    return cfg


def _enabled(name):
    from vecs.config import ProjectConfig
    return ProjectConfig(name=name, prose_drift_enabled=True)


def _disabled(name):
    from vecs.config import ProjectConfig
    return ProjectConfig(name=name)


def _patch_find(monkeypatch):
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "find_prose_drift", lambda proj: {
        "drift": [], "facts_scanned": 1, "facts_scanned_docs": 1, "project": proj.name,
    })


def test_mcp_named_project_returns_payload(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"vecs": _enabled("vecs")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    out = prose_drift(project="vecs")
    assert out == {"drift": [], "facts_scanned": 1, "facts_scanned_docs": 1, "project": "vecs"}


def test_mcp_none_scans_only_enabled(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"a": _enabled("a"), "b": _enabled("b"), "c": _disabled("c")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    out = prose_drift(project=None)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"]["project"] == "a"


def test_mcp_none_no_enabled_returns_empty_dict(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"c": _disabled("c")})
    _patch_find(monkeypatch)
    from vecs.mcp_server import prose_drift
    assert prose_drift(project=None) == {}


def test_mcp_none_global_preflight_failure_is_error_dict(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _cfg_with(monkeypatch, {"a": _enabled("a")})
    from vecs.mcp_server import prose_drift
    assert prose_drift(project=None) == {"error": "anthropic_key_missing"}


def test_mcp_named_disabled_is_error_dict(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    _cfg_with(monkeypatch, {"vecs": _disabled("vecs")})
    from vecs.mcp_server import prose_drift
    assert prose_drift(project="vecs") == {"error": "prose_drift_disabled", "detail": "vecs"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q -k mcp`
Expected: FAIL — cannot import `prose_drift` from `vecs.mcp_server`.

- [ ] **Step 3: Implement the MCP tool**

In `src/vecs/mcp_server.py`, add after the `index_status` tool (~`:185`):
```python
@mcp.tool()
def prose_drift(project: str | None = None) -> dict:
    """Report contradictions between indexed docs and current chat-session facts.

    On-demand recrawl. With project=None, scans every project where
    prose_drift_enabled is true and returns a dict keyed by project name
    ({} if none enabled). With a named project, returns that project's payload.
    Detects exact (subject, predicate) object-collisions only (v1).

    Args:
        project: Project name, or None to scan all enabled projects.
    """
    from vecs.prose_drift import _preflight_global, _preflight_project

    config = load_config()

    g = _preflight_global(config)
    if not g.ok:
        return {"error": g.code} if g.detail is None else {"error": g.code, "detail": g.detail}

    from vecs.prose_drift import find_prose_drift

    if project is None:
        out: dict = {}
        for name, proj in config.projects.items():
            if proj.prose_drift_enabled:
                out[name] = find_prose_drift(proj)
        return out

    p = _preflight_project(config, project)
    if not p.ok:
        return {"error": p.code, "detail": p.detail}
    return find_prose_drift(config.projects[project])
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q -k mcp`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vecs/mcp_server.py tests/test_prose_drift_wire_in.py
git commit -m "feat(prose-drift): mcp__vecs__prose_drift tool with project=None fan-out"
```

---

## Task 10: Indexer wire-in (fold-in E)

**Files:**
- Modify: `src/vecs/indexer.py` (`_index_session_files`: stash at `:904` loop; facet block after `manifest.save()` at `:964`)
- Test: `tests/test_prose_drift_wire_in.py`

**Spec:** design `:70-73`, `:195`, `:209-254`, acceptance `:43-46`, `:442-447`. Block runs ONLY when `project.prose_drift_enabled`; per fully-succeeded file; AFTER success-path `manifest.save()`; failures logged via `_log`, never abort the indexer.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prose_drift_wire_in.py`. These call `_index_session_files` directly with a fake parser/voyage/chroma; the key assertions are (a) state machine runs only when enabled, (b) only for fully-succeeded files, (c) extraction failure does not abort.
```python
import vecs.indexer as indexer
from vecs.config import ProjectConfig


class _FakeColl:
    def __init__(self):
        self.added = []
    def get_or_create_collection(self, *a, **k):
        return self
    def add(self, **k):
        self.added.append(k)
    def get(self, **k):
        return {"ids": [], "metadatas": []}


def _prep_indexer(monkeypatch, tmp_path, enabled):
    # Pin the session pipeline's external deps to no-ops.
    monkeypatch.setattr(indexer, "_get_session_new_content",
                        lambda f, m: ("raw-content", 10, True))
    monkeypatch.setattr(indexer, "chunk_session",
                        lambda msgs, sid, n, overlap: [
                            {"text": "[user]: x", "metadata": {"chunk_index": 0}}])
    monkeypatch.setattr(indexer, "_make_chunk_id", lambda a, b: f"{a}#{b}")
    monkeypatch.setattr(indexer, "_embed_and_store", lambda chunks, c, m, vo: {ch["id"] for ch in chunks})
    monkeypatch.setattr(indexer, "_track_embed_success",
                        lambda ids, c2f, fec, fc, coll: set(c2f.values()))
    monkeypatch.setattr(indexer, "_sync_bm25", lambda *a, **k: None)
    logs = []
    monkeypatch.setattr(indexer, "_log", lambda m: logs.append(m))

    class _Manifest:
        def __init__(self, *a): pass
        def mark_session_indexed(self, *a, **k): pass
        def save(self): pass
        def get_session_info(self, f): return None
    monkeypatch.setattr(indexer, "Manifest", _Manifest)

    sm_calls = []
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "extract_facts", lambda msgs, project: [pd.Triple("team", "has_role", "x")])
    monkeypatch.setattr(pd, "add_fact_with_state_machine",
                        lambda t, source_id, project: sm_calls.append((t, source_id, project)) or "INSERT")

    proj = ProjectConfig(name="vecs", prose_drift_enabled=enabled)
    return proj, sm_calls, logs


def test_state_machine_runs_when_enabled(monkeypatch, tmp_path):
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    f = tmp_path / "be_dev_announce.jsonl"
    f.write_text("{}")
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "we hired Sasha", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert len(sm_calls) == 1
    assert sm_calls[0][1] == "be_dev_announce"  # source_id == file stem
    assert sm_calls[0][2] == "vecs"


def test_state_machine_skipped_when_disabled(monkeypatch, tmp_path):
    proj, sm_calls, _ = _prep_indexer(monkeypatch, tmp_path, enabled=False)
    f = tmp_path / "s.jsonl"; f.write_text("{}")
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "x", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert sm_calls == []


def test_extract_failure_does_not_abort_indexer(monkeypatch, tmp_path):
    proj, sm_calls, logs = _prep_indexer(monkeypatch, tmp_path, enabled=True)
    import vecs.prose_drift as pd
    monkeypatch.setattr(pd, "extract_facts",
                        lambda msgs, project: (_ for _ in ()).throw(RuntimeError("boom")))
    f = tmp_path / "s.jsonl"; f.write_text("{}")
    # Must NOT raise.
    indexer._index_session_files(
        proj, [f], lambda content: [{"role": "user", "text": "x", "timestamp": "0"}],
        "claude_code", vo=object(), db=_FakeColl(), log_label="s",
    )
    assert any("prose extract failed" in m for m in logs)
```
Note: if `_index_session_files` references helpers not patched above (e.g. constants), patch them too; run the test and follow the `AttributeError`/`NameError` to add the missing monkeypatch. Keep patches minimal — the goal is to isolate the new facet block, not re-test the embedding pipeline.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q -k state_machine or extract_failure`
Expected: FAIL — `sm_calls` empty (no facet block exists yet).

- [ ] **Step 3: Add the `file_messages` stash**

In `src/vecs/indexer.py`, inside `_index_session_files`, just before the `for f in files:` loop (`:904`), add:
```python
    file_messages: dict[Path, list[dict]] = {}
```
And inside that loop, immediately after `messages = parser_fn(content)` (`:912`), add:
```python
        file_messages[f] = messages
```

- [ ] **Step 4: Add the facet block after the success-path `manifest.save()`**

In `src/vecs/indexer.py`, immediately after `manifest.save()` at `:964` (and before `if len(succeeded_ids) > 0:` at `:966`), insert:
```python
    if project.prose_drift_enabled:
        try:
            from vecs.prose_drift import add_fact_with_state_machine, extract_facts
            for f in fully_succeeded:
                session_id = f.stem
                messages = file_messages.get(f)
                if not messages:
                    continue
                user_messages = [m for m in messages if m.get("role") == "user"]
                if not user_messages:
                    continue
                try:
                    triples = extract_facts(user_messages, project.name)
                except Exception as e:
                    _log(f"[{project.name}] prose extract failed for {f.name}: {e}")
                    continue
                if not triples:
                    _log(f"[{project.name}] prose-drift {f.name}: triples=0")
                    continue
                counts = {"INSERT": 0, "NOOP": 0, "SUPERSEDE": 0}
                for t in triples:
                    try:
                        event = add_fact_with_state_machine(
                            t, source_id=session_id, project=project.name,
                        )
                        counts[event] = counts.get(event, 0) + 1
                    except Exception as e:
                        _log(f"[{project.name}] prose state-machine failed for {f.name} triple {t}: {e}")
                        continue
                _log(
                    f"[{project.name}] prose-drift {f.name}: "
                    f"INSERT={counts['INSERT']} NOOP={counts['NOOP']} SUPERSEDE={counts['SUPERSEDE']}"
                )
        except ImportError as e:
            _log(f"[{project.name}] anthropic not installed; skipping prose-drift facet: {e}")
```

- [ ] **Step 5: Run to verify pass + full file green**

Run: `uv run pytest tests/test_prose_drift_wire_in.py -q`
Expected: PASS. Then `uv run pytest tests/test_indexer.py -q` (or the indexer test module name) to confirm no regression in the session pipeline.

- [ ] **Step 6: Commit**

```bash
git add src/vecs/indexer.py tests/test_prose_drift_wire_in.py
git commit -m "feat(prose-drift): wire state-machine facet into session indexer (enabled-gated, per fully-succeeded file)"
```

---

## Task 11: BE-dev fixtures + end-to-end test (mocked Anthropic)

**Files:**
- Create: `tests/fixtures/prose_drift/docs/team.md`
- Create: `tests/fixtures/prose_drift/sessions/be_dev_announce.jsonl`
- Test: `tests/test_prose_drift.py`

**Spec:** design `:187`, acceptance `:23`, `:448`. The canonical contradiction scenario, end-to-end through `find_prose_drift` with mocked Anthropic returning a doc-side triple that collides with a seeded session-side row.

- [ ] **Step 1: Create the fixtures**

Create `tests/fixtures/prose_drift/docs/team.md`:
```markdown
# Team

The team has no backend developer. All server work is currently unowned.
```

Create `tests/fixtures/prose_drift/sessions/be_dev_announce.jsonl`:
```json
{"role": "user", "text": "Good news — we hired Sasha as our BE dev. She owns the backend now.", "timestamp": "2026-05-20T10:00:00Z"}
```

- [ ] **Step 2: Write the end-to-end test**

Add to `tests/test_prose_drift.py`:
```python
def test_be_dev_contradiction_surfaces(fake_anthropic, fake_voyage):
    """Canonical scenario: doc says 'no backend developer', chat says 'Sasha is BE dev'."""
    project = "vecs-bedev"
    # Session-side current fact (as the indexer would have written it).
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "has_role", "sasha is backend engineer"),
        "be_dev_announce", project,
    )
    # Doc-side extraction returns the contradicting fact.
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, "The team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert len(report["drift"]) == 1
    d = report["drift"][0]
    assert d["subject"] == "team" and d["predicate"] == "has_role"
    assert d["doc"]["object"] == "no backend developer"
    assert d["chat"]["object"] == "sasha is backend engineer"
    assert d["chat"]["session_id"] == "be_dev_announce"
    assert d["chat_history_versions"] >= 1
```

- [ ] **Step 3: Run to verify pass**

Run: `uv run pytest tests/test_prose_drift.py -q -k be_dev_contradiction`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/prose_drift tests/test_prose_drift.py
git commit -m "test(prose-drift): BE-dev contradiction fixtures + end-to-end drift test"
```

---

## Task 12: Docs + CLAUDE.md baseline (fold-in D) + full acceptance

**Files:**
- Modify: `src/vecs/CLAUDE.md`
- Test: `uv run pytest -q` (full suite) + `scripts/check_acceptance.py`

- [ ] **Step 1: Update `src/vecs/CLAUDE.md`**

Under `## Entry points`, append to the CLI bullet: add `prose-drift` to the subcommand list. Under the MCP bullet, add `prose_drift` to the tools list.

Under `## Tests`, add the bullet (verbatim per acceptance `:139`):
```markdown
- Opt-in integration tests gated by VECS_TEST_REAL_LLM=1 — real LLM calls; default-skipped in CI. See tests/test_prose_drift.py::test_integration_real_anthropic.
```

Replace the `## Staleness baseline` section's `staleness_check: commit-sha-tag` line with:
```markdown
This file is the Phase 2 context-tree starter for `context_tree_root: src/vecs/`. `staleness_check: [commit-sha-tag, custom:prose-vplus]` baselines from the commit that introduces this file.
```

Add a new short subsection documenting the v1 boundary (fold-in D):
```markdown
## prose-drift v1 boundary

`vecs prose-drift` / `mcp__vecs__prose_drift` is an on-demand recrawl, not a write-time detector. v1 reports only exact `(subject, predicate)` object-collisions between indexed docs and current chat facts. Out of scope (v2 — see `docs/features/prose-staleness-detector/v2-roadmap.md`): cross-predicate/paraphrase contradictions (needs the embedding-similarity + LLM contradiction judge), omission (doc silent on a now-true fact), and soft/temporal "used to have" contradictions.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: ALL pass, with exactly one `xfailed` (the Task 6 paraphrase test). Zero `failed`, zero `xpassed`. If `test_code_vecs_path_untouched` (acceptance `:451`) exists/fails, confirm the indexer facet imports `prose_drift` only inside `_index_session_files`, never in `index_code`/`chunkers`/`ast_chunker`/`bm25_index`/`searcher`.

- [ ] **Step 3: Run the acceptance checker (non-interactive)**

Run: `uv run python scripts/check_acceptance.py prose-staleness-detector --non-interactive`
Expected: the executable checklist section reports pass for the implemented items. Operator-sign-off items (live-LLM) remain `[ ]` and are skipped. Tick implemented `[ ]` boxes in `docs/features/prose-staleness-detector/acceptance.md` to `[x]` as they are verified by the passing tests, then re-run.

- [ ] **Step 4: Optional live integration check (operator)**

Run (only with a real key): `VECS_TEST_REAL_LLM=1 uv run pytest -q -k test_integration_real_anthropic`
Expected: PASS (real Opus 4.7 extraction + INSERT/NOOP/SUPERSEDE). Tick the operator-sign-off BE-dev live box manually.

- [ ] **Step 5: Commit**

```bash
git add src/vecs/CLAUDE.md docs/features/prose-staleness-detector/acceptance.md
git commit -m "docs(prose-drift): CLAUDE.md baseline list + v1 boundary; tick verified acceptance"
```

---

## Self-review (completed by plan author)

**1. Spec coverage** — every acceptance pin maps to a task:
- CLI exits 0/1/2/3, `--limit`, drift-line format, no-chat-sessions → Task 8. MCP dict / `project=None` / `{}` / error dicts → Task 9. Config flag + YAML parse → Task 1. Row schema / state machine / no-deletes / Chroma where → already shipped (dry-run; covered by existing tests). `extract_facts_from_doc` + dual-table cache + join invariant → Task 3. `iterate_indexed_docs` `file_path` no-fallback → Task 4. `find_prose_drift` + `facts_scanned`/`facts_scanned_docs` → Task 5. Preflight two-stage → Task 7. Indexer wire-in (`file_messages` stash, post-`manifest.save()` block, enabled-gate, per-fully-succeeded, failure isolation, role filter) → Task 10. BE-dev fixture + e2e → Task 11. CLAUDE.md baseline list + Tests bullet + boundary → Task 12. Anthropic exact pin → already in `pyproject.toml` (existing test passes). Fold-ins B (canonicalization) → Task 2, C (paraphrase xfail) → Task 6, D (recrawl framing + boundary doc) → Tasks 8/12.

**2. Placeholder scan** — no `TBD`/`handle edge cases`/`similar to`; every code step shows complete code; every run step shows the command + expected result.

**3. Type/name consistency** — `find_prose_drift(project)` takes a ProjectConfig-like object with `.name` (tests use `_Proj`/real `ProjectConfig`); returns the dict `{"drift","facts_scanned","facts_scanned_docs","project"}` consumed identically by CLI (Task 8) and MCP (Task 9). `extract_facts_from_doc(text, source_relpath, project="default")` signature is consistent between Task 3 (definition) and Task 5 (call passes `project=name`). `PreflightResult(ok, code, detail)` fields consistent across Tasks 7/8/9. `_get_docs_collection`/`iterate_indexed_docs` consistent Tasks 4/5. Event constants + `INVALID_AT_NONE_SENTINEL` reused from the shipped module unchanged.

**Known intentional gap (parked v2, not a plan miss):** cross-predicate/paraphrase recall (Task 6 xfail), the embedding-similarity + LLM contradiction judge, valid-time axis, SQLite fact-store migration, cost metering — all per `docs/research/prose-drift-review-and-sota-2026-05-29.md` park-v2 list and `docs/features/prose-staleness-detector/v2-roadmap.md`.
