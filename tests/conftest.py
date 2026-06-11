"""Shared pytest fixtures for the vecs test suite.

The prose-drift wire-in tests (``tests/test_prose_drift_wire_in.py``) reuse the
``fake_voyage`` and ``fake_anthropic`` recording fakes originally defined in
``tests/test_prose_drift.py``. pytest only shares fixtures across modules when
they live in a ``conftest.py``, so the shared ones are re-exported here.

These are non-autouse, so they only activate when a test requests them by name.
The autouse ``_isolate_chroma_and_cache`` fixture is intentionally NOT lifted
here: the wire-in tests mock the prose-drift boundary entirely and never touch
real Chroma/cache, so global isolation is unnecessary.
"""
from __future__ import annotations

import sys
import types

import pytest

from vecs import prose_drift
from vecs.embed_provider import VoyageProvider


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

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    return {"calls": calls, "state": state}
