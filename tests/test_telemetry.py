"""Chroma telemetry must be off at every PersistentClient construction site."""
from unittest.mock import MagicMock

import vecs.clients as clients
import vecs.prose_drift as prose_drift


def _capture_pc(captured):
    def fake_pc(path, settings=None):
        captured["settings"] = settings
        return MagicMock()
    return fake_pc


def test_get_chromadb_client_disables_telemetry(monkeypatch):
    captured = {}
    monkeypatch.setattr(clients.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(clients, "_db_client", None)
    clients.get_chromadb_client()
    assert captured["settings"] is not None
    assert captured["settings"].anonymized_telemetry is False
    monkeypatch.setattr(clients, "_db_client", None)


def test_prose_facts_collection_disables_telemetry(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(prose_drift.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(prose_drift, "_chroma_path", lambda: tmp_path)
    prose_drift._get_prose_facts_collection("p")
    assert captured["settings"].anonymized_telemetry is False


def test_docs_collection_disables_telemetry(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(prose_drift.chromadb, "PersistentClient", _capture_pc(captured))
    monkeypatch.setattr(prose_drift, "_chroma_path", lambda: tmp_path)
    prose_drift._get_docs_collection("p")
    assert captured["settings"].anonymized_telemetry is False
