"""Tests for the Inc 1-A metering spike.

Every real (cache-miss) extraction/judge LLM call emits a cost record
(model, input/output tokens, USD); a MAX_CALLS_PER_DAY cap stops further calls.
Spike-grade: JSONL store, static prefix-matched pricing table.
"""

from __future__ import annotations

import json

import pytest

from vecs import metering as m


# --- pricing -----------------------------------------------------------------


def test_price_call_sonnet():
    # 1M in @ $3, 1M out @ $15 -> $18 for the pair
    assert m.price_call("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_price_call_opus_costs_more_than_sonnet():
    opus = m.price_call("claude-opus-4-8", 1000, 1000)
    sonnet = m.price_call("claude-sonnet-4-6", 1000, 1000)
    assert opus > sonnet > 0


def test_price_longest_prefix_wins_and_unknown_is_zero():
    # an unknown model family prices at 0 (never blocks; just unpriced)
    assert m.price_call("gpt-9", 1000, 1000) == 0.0


# --- record + count ----------------------------------------------------------


def test_record_call_writes_record_with_cost(tmp_path):
    store = tmp_path / "calls.jsonl"
    rec = m.record_call("claude-sonnet-4-6", 1000, 500, store_path=store)
    assert rec["model"] == "claude-sonnet-4-6"
    assert rec["input_tokens"] == 1000 and rec["output_tokens"] == 500
    assert rec["usd"] > 0
    lines = store.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["usd"] == rec["usd"]


def test_calls_today_counts_only_today(tmp_path, monkeypatch):
    store = tmp_path / "calls.jsonl"
    monkeypatch.setattr(m, "_today", lambda: "2026-06-04")
    m.record_call("claude-sonnet-4-6", 10, 10, store_path=store)
    m.record_call("claude-sonnet-4-6", 10, 10, store_path=store)
    assert m.calls_today(store_path=store) == 2
    # a record from another day is not counted
    monkeypatch.setattr(m, "_today", lambda: "2026-06-05")
    assert m.calls_today(store_path=store) == 0


def test_calls_today_zero_when_no_store(tmp_path):
    assert m.calls_today(store_path=tmp_path / "nope.jsonl") == 0


def test_calls_today_ignores_malformed_lines(tmp_path, monkeypatch):
    store = tmp_path / "calls.jsonl"
    store.write_text('not json\n{"date": "2026-06-04"}\n\n')
    monkeypatch.setattr(m, "_today", lambda: "2026-06-04")
    assert m.calls_today(store_path=store) == 1


# --- metered_create chokepoint ----------------------------------------------


class _Usage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _Resp:
    def __init__(self, i, o):
        self.usage = _Usage(i, o)


class _Client:
    """Minimal anthropic-client double: client.messages.create(**kw)."""

    def __init__(self, resp):
        self._resp = resp
        self.created: list[dict] = []

    @property
    def messages(self):
        return self

    def create(self, **kw):
        self.created.append(kw)
        return self._resp


def test_metered_create_records_on_success(tmp_path):
    store = tmp_path / "calls.jsonl"
    client = _Client(_Resp(200, 80))
    resp = m.metered_create(
        client,
        store_path=store,
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is client._resp
    assert len(client.created) == 1  # the underlying API was called once
    assert m.calls_today(store_path=store) == 1
    rec = json.loads(store.read_text().strip())
    assert rec["input_tokens"] == 200 and rec["output_tokens"] == 80


def test_metered_create_raises_and_skips_api_when_capped(tmp_path, monkeypatch):
    store = tmp_path / "calls.jsonl"
    monkeypatch.setattr(m, "MAX_CALLS_PER_DAY", 2)
    monkeypatch.setattr(m, "_today", lambda: "2026-06-04")
    m.record_call("claude-sonnet-4-6", 10, 10, store_path=store)
    m.record_call("claude-sonnet-4-6", 10, 10, store_path=store)  # at cap

    client = _Client(_Resp(1, 1))
    with pytest.raises(m.MeteringCapExceeded):
        m.metered_create(
            client,
            store_path=store,
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "x"}],
        )
    assert client.created == []  # API never called past the cap
    assert m.calls_today(store_path=store) == 2  # no new record


# --- cost estimate -----------------------------------------------------------


def test_estimate_extraction_cost_breakdown():
    est = m.estimate_extraction_cost(1_000_000, 100_000, model="claude-sonnet-4-6")
    # 1M in @ $3 + 0.1M out @ $15 = 3 + 1.5 = 4.5
    assert est["usd"] == pytest.approx(4.5)
    assert est["input_tokens"] == 1_000_000
    assert est["model"] == "claude-sonnet-4-6"
