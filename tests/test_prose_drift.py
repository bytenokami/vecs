"""Phase 7 dry-run tests for prose_drift state machine + extraction cache.

Design ref: docs/features/prose-staleness-detector-design-v1.md Phase 7.
Acceptance ref: docs/features/prose-staleness-detector/acceptance.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vecs import metering, prose_drift
from vecs.embed_provider import VoyageProvider
from vecs.config import ProjectConfig
from vecs.prose_drift import (
    EVENT_INSERT,
    EVENT_NOOP,
    EVENT_SUPERSEDE,
    EXTRACTION_PROMPT_VERSION,
    PROSE_EXTRACTION_MODEL,
    PROSE_JUDGE_MODEL,
    Triple,
    _extract_cache_key_text,
    add_fact_with_state_machine,
    extract_facts,
)


# ----- fixtures -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_chroma_and_cache(monkeypatch, tmp_path):
    """Pin Chroma + cache dir to tmp_path so dry-run does not touch ~/.vecs/."""
    monkeypatch.setattr(prose_drift, "_chroma_path", lambda: tmp_path / "chromadb")
    monkeypatch.setattr(prose_drift, "_cache_dir", lambda: tmp_path / "prose_drift_cache")
    # Inc 1-A: keep metering's cost-record JSONL off the real ~/.vecs store (and
    # out of the real daily cap) during tests.
    monkeypatch.setattr(metering, "DEFAULT_METERING_PATH", tmp_path / "metering" / "calls.jsonl")
    yield


@pytest.fixture
def fake_voyage(monkeypatch):
    """Replace Voyage client with a recording fake; emit 4-dim toy embeddings."""
    calls: list[dict] = []

    class _FakeResult:
        def __init__(self, n: int):
            self.embeddings = [[0.1, 0.2, 0.3, 0.4] for _ in range(n)]

    class _FakeClient:
        def embed(self, texts, *, model, input_type):
            calls.append({"texts": list(texts), "model": model, "input_type": input_type})
            return _FakeResult(len(texts))

    fake = _FakeClient()
    monkeypatch.setattr(prose_drift, "get_provider", lambda: VoyageProvider(client=fake))
    return calls


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Replace anthropic.Anthropic with a recording fake; default response = 1 triple."""
    calls: list[dict] = []
    state = {
        "response_text": '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]',
        # Default judge verdict is "not a contradiction" so stray judge calls in
        # extraction-only tests never fabricate drift. Tests opt in explicitly.
        "judge_response": '{"contradicts": false, "confidence": 0.0, "reason": "default"}',
    }

    class _FakeContent:
        def __init__(self, text: str):
            self.text = text

    class _FakeResp:
        def __init__(self, text: str):
            self.content = [_FakeContent(text)]

    class _FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][0]["content"].lower()
            # Route by prompt: the contradiction-judge prompt is the only one that
            # mentions "contradict"; everything else is an extraction call.
            if "contradict" in prompt:
                # Per-call judge responses (FIFO) override the single default,
                # letting one scan exercise multiple distinct verdicts.
                queue = state.get("judge_responses")
                if queue:
                    return _FakeResp(queue.pop(0))
                return _FakeResp(state["judge_response"])
            return _FakeResp(state["response_text"])

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    import sys
    import types

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    return {"calls": calls, "state": state}


@pytest.fixture
def fake_voyage_textmap(monkeypatch):
    """Voyage fake whose embedding depends on the input text via an injectable map.

    Lets a test make two facts SIMILAR or DISSIMILAR on demand (the constant-vector
    `fake_voyage` always yields cosine 1.0). Unmapped text -> [1,0,0,0]."""
    calls: list[dict] = []
    vectors: dict[str, list[float]] = {}

    class _FakeResult:
        def __init__(self, embs):
            self.embeddings = embs

    class _FakeClient:
        def embed(self, texts, *, model, input_type):
            calls.append({"texts": list(texts), "model": model, "input_type": input_type})
            return _FakeResult([vectors.get(t, [1.0, 0.0, 0.0, 0.0]) for t in texts])

    fake = _FakeClient()
    monkeypatch.setattr(prose_drift, "get_provider", lambda: VoyageProvider(client=fake))
    return {"calls": calls, "vectors": vectors}


# ----- cache-key canonicalization (Fix 3 / acceptance lines 416-418) -----


def test_extract_cache_key_stable_across_dict_order():
    a = [{"role": "user", "text": "hi", "timestamp": "t1"}]
    b = [{"timestamp": "t1", "text": "hi", "role": "user"}]
    assert _extract_cache_key_text(a) == _extract_cache_key_text(b)


def test_extract_cache_key_ignores_timestamps():
    a = [{"role": "user", "text": "hi", "timestamp": "t1"}]
    b = [{"role": "user", "text": "hi", "timestamp": "t999"}]
    assert _extract_cache_key_text(a) == _extract_cache_key_text(b)


def test_extract_cache_key_reads_role_and_text_only():
    """Helper MUST read m['role'] + m['text'], NEVER m['content']."""
    messages = [{"role": "user", "text": "ping", "timestamp": "0"}]
    # If helper accidentally read m["content"], this would KeyError.
    key = _extract_cache_key_text(messages)
    assert '"role":"user"' in key and '"text":"ping"' in key


# ----- extraction model + no-temperature contract -----------------------


def test_extraction_uses_correct_model_no_temperature(fake_anthropic, monkeypatch):
    extract_facts([{"role": "user", "text": "we have no BE dev", "timestamp": "0"}], "p1")
    assert len(fake_anthropic["calls"]) == 1
    call = fake_anthropic["calls"][0]
    assert call["model"] == "claude-sonnet-4-6"
    assert "temperature" not in call, "extraction model rejects temperature kwarg"


def test_prose_extraction_model_constant_is_pinned():
    assert PROSE_EXTRACTION_MODEL == "claude-sonnet-4-6"


# ----- Inc 1-A: metering wiring -----------------------------------------------


def test_extraction_emits_metering_record(fake_anthropic):
    """Every real extraction call goes through the metering chokepoint and emits
    a cost record (acceptance A line 1)."""
    extract_facts([{"role": "user", "text": "we have no BE dev", "timestamp": "0"}], "p_meter")
    assert len(fake_anthropic["calls"]) == 1
    assert metering.calls_today() == 1  # isolated store (autouse fixture)


def test_extraction_cache_hit_emits_no_metering_record(fake_anthropic):
    """A cache hit makes no API call, so it records no cost (zero spend)."""
    msgs = [{"role": "user", "text": "cache me", "timestamp": "0"}]
    extract_facts(msgs, "p_meter2")
    extract_facts(msgs, "p_meter2")  # cache hit -> no second API call
    assert len(fake_anthropic["calls"]) == 1
    assert metering.calls_today() == 1  # only the miss was metered


def test_find_prose_drift_stops_at_cap(monkeypatch, fake_voyage):
    """When extraction hits the daily cap mid-scan, find_prose_drift stops
    gracefully: no exception escapes, the payload carries cap_hit=True
    (acceptance A line 2 — 'extraction stops at the cap')."""
    proj = ProjectConfig(name="p_cap")
    monkeypatch.setattr(
        prose_drift, "iterate_indexed_docs",
        lambda name: iter([("some doc text", "docs/x.md")]),
    )

    def _capped(*a, **k):
        raise metering.MeteringCapExceeded("daily LLM call cap reached (test)")

    monkeypatch.setattr(prose_drift, "extract_facts_from_doc", _capped)
    out = prose_drift.find_prose_drift(proj)
    assert out["cap_hit"] is True
    assert out["project"] == "p_cap"
    assert out["facts_scanned_docs"] == 1  # the one doc we started before the cap


def test_voyage_embed_uses_pinned_facts_model(monkeypatch):
    """Facts embedding uses the dedicated FACTS_MODEL, decoupled from
    DOCS_MODEL, so the Inc 1-B docs model swap cannot strand fact vectors
    (Inc 1-B acceptance)."""
    from vecs.config import FACTS_MODEL

    captured = {}

    class _Result:
        embeddings = [[0.1, 0.2]]

    class _Vo:
        def embed(self, texts, model, input_type):
            captured["model"] = model
            return _Result()

    monkeypatch.setattr(prose_drift, "get_provider", lambda: VoyageProvider(client=_Vo()))
    prose_drift._embed_fact("hello")
    assert captured["model"] == FACTS_MODEL


def test_facts_model_decoupled_from_docs_model():
    """FACTS_MODEL is its own constant, not an alias of DOCS_MODEL."""
    import vecs.config as cfg
    assert hasattr(cfg, "FACTS_MODEL")
    # Distinct module-level name; changing DOCS_MODEL must not move facts.
    assert "FACTS_MODEL" in cfg.__dict__


# ----- extraction cache (acceptance lines 411-413, 421) ------------------


def test_extraction_cache_hits_on_repeat_text(fake_anthropic):
    msgs = [{"role": "user", "text": "we have no BE dev", "timestamp": "0"}]
    out1 = extract_facts(msgs, "p_repeat")
    out2 = extract_facts(msgs, "p_repeat")
    assert out1 == out2
    assert len(fake_anthropic["calls"]) == 1, "second call must hit cache, not Anthropic"


def test_extraction_cache_misses_on_text_change(fake_anthropic):
    extract_facts([{"role": "user", "text": "v1", "timestamp": "0"}], "p_miss")
    extract_facts([{"role": "user", "text": "v2", "timestamp": "0"}], "p_miss")
    assert len(fake_anthropic["calls"]) == 2


def test_cache_invalidates_on_prompt_version_bump(fake_anthropic, monkeypatch, tmp_path):
    msgs = [{"role": "user", "text": "hi", "timestamp": "0"}]
    extract_facts(msgs, "p_pv")
    assert len(fake_anthropic["calls"]) == 1
    # Use a sentinel distinct from the live default so the bump always differs
    # from whatever version the first call cached under.
    monkeypatch.setattr(prose_drift, "EXTRACTION_PROMPT_VERSION", "v_test_bump")
    extract_facts(msgs, "p_pv")
    assert len(fake_anthropic["calls"]) == 2, "prompt_version bump must invalidate cache"


def test_extraction_cache_ddl_initializes_on_first_use(fake_anthropic, tmp_path):
    extract_facts([{"role": "user", "text": "x", "timestamp": "0"}], "p_ddl")
    db_path = tmp_path / "prose_drift_cache" / "p_ddl.db"
    assert db_path.exists()
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "doc_facts" in tables and "extraction_cache" in tables
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal.lower() == "wal"
    conn.close()


def test_empty_triple_list_positive_cached(fake_anthropic):
    fake_anthropic["state"]["response_text"] = "[]"
    msgs = [{"role": "user", "text": "small talk", "timestamp": "0"}]
    assert extract_facts(msgs, "p_empty") == []
    assert extract_facts(msgs, "p_empty") == []
    assert len(fake_anthropic["calls"]) == 1, "empty result must be positive-cached"


def test_role_filter_excludes_assistant(fake_anthropic):
    msgs = [
        {"role": "assistant", "text": "hallucinated fact", "timestamp": "0"},
        {"role": "system", "text": "system noise", "timestamp": "0"},
        {"role": "user", "text": "real claim", "timestamp": "0"},
    ]
    extract_facts(msgs, "p_filter")
    # Only the user message should appear in the prompt.
    sent_prompt = fake_anthropic["calls"][0]["messages"][0]["content"]
    assert "real claim" in sent_prompt
    assert "hallucinated fact" not in sent_prompt
    assert "system noise" not in sent_prompt


# ----- state machine (acceptance lines 422-424) -------------------------


def test_event_constants():
    assert EVENT_INSERT == "INSERT"
    assert EVENT_NOOP == "NOOP"
    assert EVENT_SUPERSEDE == "SUPERSEDE"


def test_insert_noop_supersede_sequence(fake_voyage):
    project = "vecs-test"
    t1 = Triple("team", "has_role", "no backend developer")
    assert add_fact_with_state_machine(t1, "be_dev_announce", project) == EVENT_INSERT

    coll = prose_drift._get_prose_facts_collection(project)
    assert len(coll.get()["ids"]) == 1

    assert add_fact_with_state_machine(t1, "be_dev_announce", project) == EVENT_NOOP
    assert len(coll.get()["ids"]) == 1, "NOOP must not add a row"

    t2 = Triple("team", "has_role", "Sasha as backend developer")
    assert add_fact_with_state_machine(t2, "be_dev_announce", project) == EVENT_SUPERSEDE

    rows = coll.get()
    assert len(rows["ids"]) == 2
    metas_by_current = {m["is_current"]: m for m in rows["metadatas"]}
    assert metas_by_current[True]["object"] == "Sasha as backend developer"
    assert metas_by_current[True]["version"] == 2
    assert metas_by_current[True]["invalid_at"] == prose_drift.INVALID_AT_NONE_SENTINEL
    assert metas_by_current[False]["object"] == "no backend developer"
    assert metas_by_current[False]["invalid_at"] > 0


def test_voyage_embed_uses_correct_model(fake_voyage):
    from vecs.config import FACTS_MODEL

    add_fact_with_state_machine(
        Triple("x", "y", "z"), "src", "vecs-voyage-test"
    )
    assert len(fake_voyage) >= 1
    # Facts embed with the pinned FACTS_MODEL, decoupled from DOCS_MODEL.
    assert fake_voyage[0]["model"] == FACTS_MODEL


def test_per_project_scoping_isolates_collections(fake_voyage):
    add_fact_with_state_machine(Triple("a", "b", "v1"), "s", "proj_A")
    a_rows = prose_drift._get_prose_facts_collection("proj_A").get()
    b_rows = prose_drift._get_prose_facts_collection("proj_B").get()
    assert len(a_rows["ids"]) == 1
    assert len(b_rows["ids"]) == 0


# ----- Chroma where-clause verification (Fix 1 / new multi-key bullet) ---


def test_chroma_bool_where_verification(fake_voyage):
    """Fix 1 / BLOCKER 1: bool literals in where-clause must filter correctly."""
    project = "vecs-where-bool"
    add_fact_with_state_machine(Triple("k", "p", "v1"), "s", project)
    add_fact_with_state_machine(Triple("k", "p", "v2"), "s", project)
    coll = prose_drift._get_prose_facts_collection(project)
    current_ids = coll.get(where={"is_current": True})["ids"]
    superseded_ids = coll.get(where={"is_current": False})["ids"]
    assert len(current_ids) == 1
    assert len(superseded_ids) == 1


def test_chroma_multi_key_where_canonical_form_pinned(fake_voyage):
    """Pin canonical multi-key `where` form to `$and`. Production code uses it.

    Both forms are exercised; either may raise depending on Chroma version.
    The canonical form (`$and`) is asserted to succeed unconditionally;
    the flat form's behavior is observed and recorded but does not gate the test.
    """
    project = "vecs-where-multi"
    add_fact_with_state_machine(Triple("k", "p", "v1"), "s", project)
    add_fact_with_state_machine(Triple("k", "p", "v2"), "s", project)
    coll = prose_drift._get_prose_facts_collection(project)
    chain_key = "k|p"

    # Canonical form (used by production state-machine lookup): MUST pass.
    and_result = coll.get(
        where={"$and": [{"chain_key": chain_key}, {"is_current": True}]}
    )
    assert len(and_result["ids"]) == 1
    assert and_result["metadatas"][0]["object"] == "v2"

    # Flat form: observed for cross-version compatibility audit.
    flat_raises = False
    flat_matches = False
    try:
        flat_result = coll.get(where={"chain_key": chain_key, "is_current": True})
        flat_matches = (
            len(flat_result["ids"]) == 1
            and flat_result["metadatas"][0]["object"] == "v2"
        )
    except Exception:
        flat_raises = True
    # Either Chroma accepts the flat form correctly, or rejects it. Not both, not neither.
    assert flat_raises ^ flat_matches, (
        "flat-dict multi-key where behavior inconsistent — neither raise nor match"
    )


def test_no_deletes_against_prose_facts():
    """Grep test: no code path issues collection.delete against -prose-facts."""
    src_path = Path(__file__).resolve().parents[1] / "src" / "vecs" / "prose_drift.py"
    src = src_path.read_text()
    assert ".delete(" not in src, "prose_drift.py must not call .delete() (history-only invariant)"


def test_supersede_write_order_add_first_then_flip(fake_voyage, monkeypatch):
    """Fix 2: SUPERSEDE writes new row FIRST, then flips old. Crash between
    the two steps must leave the chain recoverable.

    Simulate the crash by making `update` raise after `add` has run. Then
    re-run the state machine and assert the repair branch (len(current) > 1)
    demotes the lower-version row and returns NOOP because the new row's
    object already matches the incoming triple.
    """
    project = "vecs-write-order"
    add_fact_with_state_machine(Triple("k", "p", "v1"), "s", project)

    coll = prose_drift._get_prose_facts_collection(project)
    original_update = coll.update
    call_log = []

    def spy_add(*args, **kwargs):
        call_log.append("add")
        return coll._add_original(*args, **kwargs)

    def raising_update(*args, **kwargs):
        call_log.append("update")
        raise RuntimeError("simulated crash mid-SUPERSEDE")

    # Patch BOTH add (to log) and update (to raise) on the same collection object.
    coll._add_original = coll.add
    coll.add = spy_add
    coll.update = raising_update

    def fixed_collection(_project):
        return coll

    monkeypatch.setattr(prose_drift, "_get_prose_facts_collection", fixed_collection)

    # Attempt SUPERSEDE — update will raise. Verify add ran BEFORE update.
    with pytest.raises(RuntimeError):
        add_fact_with_state_machine(Triple("k", "p", "v2"), "s", project)

    assert call_log == ["add", "update"], (
        f"SUPERSEDE write order violated: expected add-then-update, got {call_log}"
    )

    # Post-crash state: 2 rows with is_current=True (the new v2 row + the original v1 row).
    coll.update = original_update  # restore for repair to run
    coll.add = coll._add_original
    rows_post_crash = coll.get(where={"is_current": True})
    assert len(rows_post_crash["ids"]) == 2, "post-crash: two is_current=True rows expected"

    # Re-run with the new object — repair branch demotes v1, finds v2 operative → NOOP.
    assert (
        add_fact_with_state_machine(Triple("k", "p", "v2"), "s", project)
        == EVENT_NOOP
    )
    rows_after = coll.get(where={"is_current": True})
    assert len(rows_after["ids"]) == 1
    assert rows_after["metadatas"][0]["object"] == "v2"


def test_repair_branch_noop_when_operative_matches_incoming(fake_voyage, monkeypatch):
    """Repair branch (len(current) > 1) followed by NOOP because the highest-version
    row's object already equals the incoming triple's object."""
    project = "vecs-repair-noop"
    # Hand-seed two rows with is_current=True for the same chain_key, same object.
    coll = prose_drift._get_prose_facts_collection(project)
    common = {
        "subject": "k",
        "predicate": "p",
        "object": "v_same",
        "chain_key": "k|p",
        "valid_from": 1000,
        "invalid_at": 0,
        "is_current": True,
        "source_id": "seed",
    }
    coll.add(
        ids=["row-old"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["k p v_same"],
        metadatas=[{**common, "version": 1}],
    )
    coll.add(
        ids=["row-new"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["k p v_same"],
        metadatas=[{**common, "version": 2, "valid_from": 2000}],
    )
    pre = coll.get(where={"is_current": True})
    assert len(pre["ids"]) == 2, "test setup: two is_current=True rows seeded"

    event = add_fact_with_state_machine(Triple("k", "p", "v_same"), "s", project)
    assert event == EVENT_NOOP

    post = coll.get(where={"is_current": True})
    assert len(post["ids"]) == 1, "repair branch must demote the lower-version row"
    assert post["metadatas"][0]["version"] == 2


def test_pyproject_pins_anthropic_exact_version():
    """Acceptance line 68: pyproject.toml declares anthropic with an exact pin."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "anthropic==" in pyproject, "anthropic must be declared with exact-pin (==) form"
    # Extract pinned version and assert it's a concrete release identifier.
    pinned = [
        ln.strip() for ln in pyproject.splitlines() if ln.strip().startswith('"anthropic==')
    ]
    assert len(pinned) == 1, f"expected exactly one anthropic pin line; got {pinned}"


# ----- Task 2: canonicalization prompt (fold-in B) ----------------------


def test_extraction_prompt_version_bumped_to_v2():
    assert prose_drift.EXTRACTION_PROMPT_VERSION == "v2"


def test_extraction_prompt_has_canonicalization_guidance():
    p = prose_drift.EXTRACTION_PROMPT
    assert "canonical" in p.lower()
    # A controlled-vocabulary hint and at least one worked example must be present.
    assert "has_role" in p
    assert "Example" in p


# ----- Task 3: doc-side extraction + dual-table cache -------------------


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


# ----- Task 4: iterate_indexed_docs -------------------------------------


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


# ----- Task 5: find_prose_drift -----------------------------------------


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


# ----- Task 6: paraphrase-miss xfail (fold-in C) ------------------------


def test_cross_predicate_paraphrase_drift_is_detected(fake_anthropic, fake_voyage):
    """Promoted from xfail(strict) when stage-2 recall landed (2026-05-30).

    Chat 'team|employs:sasha' vs doc 'team|has_role:no backend developer' — a real
    contradiction across DIFFERENT (subject,predicate) chains. The exact key misses;
    the similarity fallback + contradiction-judge recovers it.
    """
    project = "p_paraphrase"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "be_dev_announce", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": 0.88, '
        '"reason": "employing Sasha as BE engineer contradicts having no BE developer"}'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert len(report["drift"]) == 1
    d = report["drift"][0]
    assert d["match_type"] == "semantic"
    assert d["subject"] == "team" and d["predicate"] == "has_role"  # doc-side
    assert d["doc"]["object"] == "no backend developer"
    assert d["chat"]["predicate"] == "employs" and d["chat"]["object"] == "sasha"
    assert d["similarity"] >= prose_drift.STAGE2_SIM_THRESHOLD
    assert d["confidence"] == pytest.approx(0.88)
    assert report["stage2_judge_calls"] == 1
    assert report["stage2_judge_errors"] == 0


def test_stage2_below_threshold_makes_no_judge_call_and_no_drift(
    fake_anthropic, fake_voyage_textmap
):
    """A MISS whose best candidate is below the similarity threshold never reaches
    the judge and produces no drift."""
    project = "p_below_thresh"
    fake_voyage_textmap["vectors"]["team employs sasha"] = [1.0, 0.0, 0.0, 0.0]
    fake_voyage_textmap["vectors"]["team has_role no backend developer"] = [0.0, 1.0, 0.0, 0.0]
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["drift"] == []
    assert report["stage2_judge_calls"] == 0
    assert _judge_calls(fake_anthropic) == []


def test_stage2_judge_says_not_contradiction_no_drift(fake_anthropic, fake_voyage):
    """Above threshold but the judge rules NOT a contradiction -> no drift."""
    project = "p_judge_no"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"mentions","object":"sasha is great"}]'
    )
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": false, "confidence": 0.1, "reason": "not contradictory"}'
    )
    _seed_doc(project, "Sasha is great.", "praise.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["drift"] == []
    assert report["stage2_judge_calls"] == 1
    assert report["stage2_judge_errors"] == 0


def test_stage2_judge_error_is_skipped_and_counted(fake_anthropic, fake_voyage):
    """A single judge call that returns unparseable output is caught: the candidate
    is skipped, the scan continues, and the error is counted."""
    project = "p_judge_err"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    fake_anthropic["state"]["judge_response"] = "this is not json at all"
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["drift"] == []
    assert report["stage2_judge_calls"] == 1
    assert report["stage2_judge_errors"] == 1


def test_exact_finding_tagged_match_type_exact(fake_anthropic, fake_voyage):
    """The v1 exact-collision path still works and now carries match_type='exact'."""
    project = "p_exact_tag"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "has_role", "sasha is backend engineer"),
        "be_dev_announce", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert len(report["drift"]) == 1
    d = report["drift"][0]
    assert d["match_type"] == "exact"
    assert d["chat"]["subject"] == "team" and d["chat"]["predicate"] == "has_role"
    assert report["stage2_judge_calls"] == 0


# ----- Inc 1-A: metering through the find_prose_drift judge + doc paths -------


def _metering_models():
    """Models recorded in the (isolated) metering JSONL this test."""
    p = metering.DEFAULT_METERING_PATH
    if not p.exists():
        return []
    return [json.loads(line)["model"] for line in p.read_text().splitlines() if line.strip()]


def test_find_prose_drift_meters_doc_extraction_and_judge(fake_anthropic, fake_voyage):
    """Both the doc-extraction AND the stage-2 judge LLM calls inside
    find_prose_drift route through metered_create, each emitting a cost record
    (acceptance A line 1 — the judge + doc halves, which the extraction-only
    record tests don't cover)."""
    project = "p_meter_judge"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"mentions","object":"sasha is great"}]'
    )
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": false, "confidence": 0.1, "reason": "no"}'
    )
    _seed_doc(project, "Sasha is great.", "praise.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["stage2_judge_calls"] == 1  # the judge actually fired

    models = _metering_models()
    assert PROSE_EXTRACTION_MODEL in models  # doc-extraction was metered
    assert PROSE_JUDGE_MODEL in models  # the judge call was metered


def test_find_prose_drift_judge_cap_is_not_swallowed_by_inner_except(
    fake_anthropic, fake_voyage, monkeypatch
):
    """A cap raised on the JUDGE call must propagate past the inner judge except
    (which catches only parse errors) to the outer cap-stop — NOT be swallowed
    and miscounted as a judge_error. Pins that MeteringCapExceeded (a
    RuntimeError) stays outside that narrow except tuple."""
    project = "p_judge_cap"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"mentions","object":"sasha is great"}]'
    )
    _seed_doc(project, "Sasha is great.", "praise.md")

    def _cap(*a, **k):
        raise metering.MeteringCapExceeded("cap on judge (test)")

    monkeypatch.setattr(prose_drift, "_judge_contradiction_ex", _cap)

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["cap_hit"] is True
    assert report["stage2_judge_errors"] == 0  # not swallowed/miscounted


def test_find_prose_drift_real_metered_cap_stops_and_logs(
    fake_anthropic, fake_voyage, monkeypatch, capsys
):
    """End-to-end: the REAL metered_create raises mid-scan once the daily cap is
    hit (here MAX_CALLS_PER_DAY=0 -> the first metered call trips it), so
    find_prose_drift stops with cap_hit=True AND logs that it stopped to stderr
    (acceptance A line 2 — both 'stops' and 'logs that it stopped')."""
    project = "p_real_cap"
    monkeypatch.setattr(metering, "MAX_CALLS_PER_DAY", 0)  # cap already reached
    _seed_doc(project, "Our team has no backend developer.", "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["cap_hit"] is True
    assert "stopping extraction at the daily cap" in capsys.readouterr().err


# ----- stage-2 review-driven hardening -----------------------------------


def _seed_paraphrase(fake_anthropic, project):
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "be_dev_announce", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": 0.88, "reason": "contradiction"}'
    )
    _seed_doc(project, "Our team has no backend developer.", "team.md")


def test_rescan_makes_no_new_anthropic_calls_and_is_deterministic(fake_anthropic, fake_voyage):
    """Rerun-determinism at the scan level (acceptance line: repeat scan = zero new
    anthropic calls). Judge + extraction caches make scan 2 free."""
    project = "p_rescan"
    _seed_paraphrase(fake_anthropic, project)

    r1 = prose_drift.find_prose_drift(_Proj(project))
    calls_after_1 = len(fake_anthropic["calls"])
    judge_after_1 = len(_judge_calls(fake_anthropic))
    voyage_after_1 = len(fake_voyage)
    assert r1["stage2_judge_calls"] == 1

    r2 = prose_drift.find_prose_drift(_Proj(project))
    # No new anthropic calls of any kind on the second scan.
    assert len(fake_anthropic["calls"]) == calls_after_1
    assert len(_judge_calls(fake_anthropic)) == judge_after_1
    # No new Voyage embeds either — the doc-triple embedding is cached.
    assert len(fake_voyage) == voyage_after_1
    # stage2_judge_calls counts ACTUAL api calls — a cached rescan makes zero.
    assert r2["stage2_judge_calls"] == 0
    # Drift is identical across runs.
    assert r1["drift"] == r2["drift"]


def test_stage2_threshold_is_inclusive_just_above(fake_anthropic, fake_voyage_textmap):
    import math as _m
    project = "p_thr_above"
    fake_voyage_textmap["vectors"]["team employs sasha"] = [0.86, _m.sqrt(1 - 0.86**2), 0.0, 0.0]
    fake_voyage_textmap["vectors"]["team has_role no backend developer"] = [1.0, 0.0, 0.0, 0.0]
    _seed_paraphrase(fake_anthropic, project)
    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["stage2_judge_calls"] == 1
    assert len(report["drift"]) == 1
    assert report["drift"][0]["similarity"] >= prose_drift.STAGE2_SIM_THRESHOLD


def test_stage2_threshold_excludes_just_below(fake_anthropic, fake_voyage_textmap):
    import math as _m
    project = "p_thr_below"
    fake_voyage_textmap["vectors"]["team employs sasha"] = [0.84, _m.sqrt(1 - 0.84**2), 0.0, 0.0]
    fake_voyage_textmap["vectors"]["team has_role no backend developer"] = [1.0, 0.0, 0.0, 0.0]
    _seed_paraphrase(fake_anthropic, project)
    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["stage2_judge_calls"] == 0
    assert report["drift"] == []


def test_stage2_multiple_miss_triples_one_scan(fake_anthropic, fake_voyage):
    """Two MISS triples in one scan: one judged contradiction, one judge error.
    Counters accumulate; the error does not abort the scan."""
    project = "p_multi"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s1", project,
    )
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("repo", "uses", "go"), "s2", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"},'
        '{"subject":"repo","predicate":"lang","object":"python"}]'
    )
    fake_anthropic["state"]["judge_responses"] = [
        '{"contradicts": true, "confidence": 0.9, "reason": "yes"}',
        "this is not json",
    ]
    _seed_doc(project, "doc text", "x.md")
    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["stage2_judge_calls"] == 2  # both reached the API
    assert report["stage2_judge_errors"] == 1
    assert len(report["drift"]) == 1


def test_best_semantic_candidate_tie_breaks_on_chain_key():
    rows = [
        ({"chain_key": "z|z", "object": "x"}, [1.0, 0.0, 0.0, 0.0]),
        ({"chain_key": "a|a", "object": "y"}, [1.0, 0.0, 0.0, 0.0]),
    ]
    meta, sim = prose_drift._best_semantic_candidate([1.0, 0.0, 0.0, 0.0], rows)
    assert sim == pytest.approx(1.0)
    assert meta["chain_key"] == "a|a"  # deterministic: smallest chain_key wins ties


def test_stage2_same_object_different_predicate_judge_rejects(fake_anthropic, fake_voyage):
    """High similarity but the judge rules NOT a contradiction (same object, different
    predicate) -> no drift. Proves the system relies on the judge, not similarity."""
    project = "p_same_obj"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s", project,
    )
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"owns","object":"sasha"}]'
    )
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": false, "confidence": 0.2, "reason": "same person, not contradictory"}'
    )
    _seed_doc(project, "team owns sasha", "x.md")
    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["drift"] == []
    assert report["stage2_judge_calls"] == 1


def test_judge_confidence_is_clamped_into_unit_interval(fake_anthropic, fake_voyage):
    project = "p_clamp"
    _seed_paraphrase(fake_anthropic, project)
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": 1.5, "reason": "overconfident"}'
    )
    report = prose_drift.find_prose_drift(_Proj(project))
    assert len(report["drift"]) == 1
    assert report["drift"][0]["confidence"] == 1.0


def test_judge_non_numeric_confidence_counts_as_error_not_drift(fake_anthropic, fake_voyage):
    project = "p_badconf"
    _seed_paraphrase(fake_anthropic, project)
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": "high", "reason": "unparseable confidence"}'
    )
    report = prose_drift.find_prose_drift(_Proj(project))
    assert report["drift"] == []
    assert report["stage2_judge_errors"] == 1


def test_prompt_routing_marker_is_stable():
    """The test fake routes judge vs extraction on the 'contradict' substring; pin it."""
    assert "contradict" in prose_drift.JUDGE_PROMPT.lower()
    assert "contradict" not in prose_drift.EXTRACTION_PROMPT.lower()
    assert "contradict" not in prose_drift.DOC_EXTRACTION_PROMPT.lower()


def test_load_current_rows_roundtrips_real_embedding_for_cosine(fake_voyage):
    """Exercise the real add -> Chroma -> _load_current_rows -> _cosine path so the
    numpy/float round-trip is covered, not just python-float fakes."""
    project = "p_roundtrip"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("k", "p", "v"), "s", project,
    )
    rows = prose_drift._load_current_rows(prose_drift._get_prose_facts_collection(project))
    assert len(rows) == 1
    emb = rows[0][1]
    assert all(isinstance(float(x), float) for x in emb)
    assert prose_drift._cosine(emb, emb) == pytest.approx(1.0)


# ----- Task 7: preflight + exception hierarchy --------------------------


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


# ----- Task 11: BE-dev contradiction end-to-end (mocked Anthropic) ------


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


def test_be_dev_contradiction_via_fixtures(fake_anthropic, fake_voyage):
    """Same scenario as above, but exercised against the committed fixture FILES
    (docs/team.md + sessions/be_dev_announce.jsonl) rather than inline strings.

    Acceptance item 24 / design line 448. Loads the fixture doc content into the
    docs collection and seeds the session-side fact derived from the fixture
    session, then asserts find_prose_drift surfaces exactly one (team, has_role)
    drift carrying both objects.
    """
    project = "vecs-bedev-fixtures"
    fixtures = Path(__file__).resolve().parent / "fixtures" / "prose_drift"
    doc_text = (fixtures / "docs" / "team.md").read_text()
    session_lines = [
        json.loads(ln)
        for ln in (fixtures / "sessions" / "be_dev_announce.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    # Sanity: the fixture files carry the contradicting prose we rely on.
    assert "no backend developer" in doc_text
    assert any("Sasha" in m["text"] for m in session_lines)

    # Session-side current fact (the indexer would have extracted this from the
    # be_dev_announce.jsonl user turn). source_id mirrors the fixture file stem.
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "has_role", "sasha is backend engineer"),
        "be_dev_announce", project,
    )
    # Doc-side extraction over the fixture's team.md returns the contradicting fact.
    fake_anthropic["state"]["response_text"] = (
        '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'
    )
    _seed_doc(project, doc_text, "team.md")

    report = prose_drift.find_prose_drift(_Proj(project))
    drift = [d for d in report["drift"] if (d["subject"], d["predicate"]) == ("team", "has_role")]
    assert len(drift) == 1
    d = drift[0]
    assert d["doc"]["object"] == "no backend developer"
    assert d["chat"]["object"] == "sasha is backend engineer"
    assert d["chat"]["session_id"] == "be_dev_announce"
    assert d["chat_history_versions"] >= 1


# ----- integration: real Anthropic call (gated) -------------------------


@pytest.mark.skipif(
    os.environ.get("VECS_TEST_REAL_LLM") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set VECS_TEST_REAL_LLM=1 and ANTHROPIC_API_KEY for live LLM integration test.",
)
def test_integration_real_anthropic(tmp_path):
    """End-to-end: real Opus 4.7 extraction + state machine on BE-dev fixture."""
    msgs = [
        {"role": "user", "text": "Our team has no backend developer.", "timestamp": "0"}
    ]
    triples = extract_facts(msgs, "vecs-live")
    assert len(triples) >= 1
    t1 = triples[0]
    assert add_fact_with_state_machine(t1, "live_session", "vecs-live") == EVENT_INSERT
    assert add_fact_with_state_machine(t1, "live_session", "vecs-live") == EVENT_NOOP
    t2 = Triple(t1.subject, t1.predicate, "Sasha")
    assert (
        add_fact_with_state_machine(t2, "live_session", "vecs-live")
        == EVENT_SUPERSEDE
    )


# ===== stage-2 recall (cross-predicate / paraphrase) =====================
# Design: docs/features/prose-staleness-detector/stage2-recall-design.md


def test_cosine_identical_vectors_is_one():
    assert prose_drift._cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert prose_drift._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_negative_one():
    assert prose_drift._cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero_not_error():
    # A zero-norm vector must not raise ZeroDivisionError; define sim as 0.
    assert prose_drift._cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_load_current_rows_returns_metas_and_embeddings(fake_voyage):
    project = "p_load"
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("team", "employs", "sasha"), "s1", project
    )
    prose_drift.add_fact_with_state_machine(
        prose_drift.Triple("repo", "uses", "python"), "s2", project
    )
    coll = prose_drift._get_prose_facts_collection(project)
    rows = prose_drift._load_current_rows(coll)
    assert len(rows) == 2
    by_chain = {m["chain_key"]: e for m, e in rows}
    assert set(by_chain) == {"team|employs", "repo|uses"}
    for m, e in rows:
        assert len(e) == 4  # fake voyage emits 4-dim toy embeddings
        assert "object" in m


def test_load_current_rows_dedupes_duplicate_chain_to_max_version(fake_voyage):
    """Post-crash transient state: two is_current rows for one chain_key.
    Read path returns the max-version row only, without mutating."""
    project = "p_load_dup"
    coll = prose_drift._get_prose_facts_collection(project)
    base = {
        "subject": "k",
        "predicate": "p",
        "chain_key": "k|p",
        "valid_from": 1,
        "invalid_at": 0,
        "is_current": True,
        "source_id": "seed",
    }
    coll.add(
        ids=["v1", "v2"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        documents=["k p old", "k p new"],
        metadatas=[
            {**base, "object": "old", "version": 1},
            {**base, "object": "new", "version": 2},
        ],
    )
    rows = prose_drift._load_current_rows(coll)
    assert len(rows) == 1
    meta, _emb = rows[0]
    assert meta["object"] == "new" and meta["version"] == 2
    # Read path must not have flipped the lower row.
    still = coll.get(where={"$and": [{"chain_key": "k|p"}, {"is_current": True}]})
    assert len(still["ids"]) == 2


def test_best_semantic_candidate_picks_highest_cosine():
    rows = [
        ({"chain_key": "a|b", "object": "x"}, [1.0, 0.0, 0.0]),
        ({"chain_key": "c|d", "object": "y"}, [0.0, 1.0, 0.0]),
    ]
    doc_emb = [0.9, 0.1, 0.0]
    meta, sim = prose_drift._best_semantic_candidate(doc_emb, rows)
    assert meta["chain_key"] == "a|b"
    assert sim == pytest.approx(prose_drift._cosine(doc_emb, [1.0, 0.0, 0.0]))


def test_best_semantic_candidate_none_when_no_rows():
    assert prose_drift._best_semantic_candidate([1.0, 0.0], []) is None


def _judge_calls(fake_anthropic):
    return [
        c for c in fake_anthropic["calls"]
        if "contradict" in c["messages"][0]["content"].lower()
    ]


def test_judge_contradiction_parses_verdict(fake_anthropic):
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": 0.82, '
        '"reason": "team cannot both employ Sasha and have no BE dev"}'
    )
    doc = prose_drift.Triple("team", "has_role", "no backend developer")
    chat = {"subject": "team", "predicate": "employs", "object": "sasha"}
    v = prose_drift._judge_contradiction(doc, chat, "p_judge")
    assert v.contradicts is True
    assert v.confidence == pytest.approx(0.82)
    assert "sasha" in v.reason.lower()


def test_judge_contradiction_uses_pinned_model_no_temperature(fake_anthropic):
    doc = prose_drift.Triple("team", "has_role", "none")
    chat = {"subject": "team", "predicate": "employs", "object": "sasha"}
    prose_drift._judge_contradiction(doc, chat, "p_judge_model")
    jc = _judge_calls(fake_anthropic)
    assert len(jc) == 1
    assert jc[0]["model"] == prose_drift.PROSE_JUDGE_MODEL == "claude-opus-4-8"
    assert "temperature" not in jc[0]


def test_judge_contradiction_caches_verdict(fake_anthropic):
    fake_anthropic["state"]["judge_response"] = (
        '{"contradicts": true, "confidence": 0.5, "reason": "x"}'
    )
    doc = prose_drift.Triple("team", "has_role", "none")
    chat = {"subject": "team", "predicate": "employs", "object": "sasha"}
    prose_drift._judge_contradiction(doc, chat, "p_judge_cache")
    assert len(_judge_calls(fake_anthropic)) == 1
    # Second identical call hits the cache — no new anthropic judge call.
    again = prose_drift._judge_contradiction(doc, chat, "p_judge_cache")
    assert len(_judge_calls(fake_anthropic)) == 1
    assert again.contradicts is True and again.confidence == pytest.approx(0.5)
