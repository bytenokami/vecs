"""Phase 7 dry-run tests for prose_drift state machine + extraction cache.

Design ref: docs/features/prose-staleness-detector-design-v1.md Phase 7.
Acceptance ref: docs/features/prose-staleness-detector/acceptance.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vecs import prose_drift
from vecs.prose_drift import (
    EVENT_INSERT,
    EVENT_NOOP,
    EVENT_SUPERSEDE,
    EXTRACTION_PROMPT_VERSION,
    PROSE_EXTRACTION_MODEL,
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
    monkeypatch.setattr(prose_drift, "get_voyage_client", lambda: fake)
    return calls


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Replace anthropic.Anthropic with a recording fake; default response = 1 triple."""
    calls: list[dict] = []
    state = {"response_text": '[{"subject":"team","predicate":"has_role","object":"no backend developer"}]'}

    class _FakeContent:
        def __init__(self, text: str):
            self.text = text

    class _FakeResp:
        def __init__(self, text: str):
            self.content = [_FakeContent(text)]

    class _FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
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
    assert call["model"] == "claude-opus-4-7"
    assert "temperature" not in call, "claude-opus-4-7 rejects temperature kwarg"


def test_prose_extraction_model_constant_is_pinned():
    assert PROSE_EXTRACTION_MODEL == "claude-opus-4-7"


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
    monkeypatch.setattr(prose_drift, "EXTRACTION_PROMPT_VERSION", "v2")
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
    add_fact_with_state_machine(
        Triple("x", "y", "z"), "src", "vecs-voyage-test"
    )
    assert len(fake_voyage) >= 1
    assert fake_voyage[0]["model"] == "voyage-3"


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
