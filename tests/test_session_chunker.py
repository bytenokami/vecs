import json

from vecs.chunkers import preprocess_session, chunk_session


def _make_message(role: str, content: str, msg_type: str = "user") -> str:
    """Create a minimal JSONL line matching Claude Code format."""
    obj = {
        "type": msg_type,
        "message": {"role": role, "content": content},
        "sessionId": "test-session",
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": "fake-uuid",
    }
    return json.dumps(obj)


def test_preprocess_strips_system_prompts():
    """System reminder content is stripped."""
    lines = [
        _make_message("user", "<system-reminder>blah</system-reminder>"),
        _make_message("user", "real question"),
        _make_message("assistant", "real answer", msg_type="assistant"),
    ]
    messages = preprocess_session("\n".join(lines))
    texts = [m["text"] for m in messages]
    assert not any("<system-reminder>" in t for t in texts)
    assert any("real question" in t for t in texts)


def test_preprocess_strips_base64():
    """Base64 image data is replaced with placeholder."""
    lines = [
        _make_message("user", "here is image data:iVBORw0KGgoAAAAN" + "A" * 200),
    ]
    messages = preprocess_session("\n".join(lines))
    assert not any("iVBORw0KGgo" in m["text"] for m in messages)


def test_preprocess_skips_metadata_lines():
    """Lines with type file-history-snapshot or progress are skipped."""
    lines = [
        json.dumps({"type": "file-history-snapshot", "snapshot": {}}),
        _make_message("user", "real message"),
    ]
    messages = preprocess_session("\n".join(lines))
    assert len(messages) == 1


def test_chunk_session_groups_messages():
    """Messages are grouped into chunks of N."""
    lines = [_make_message("user", f"msg {i}") for i in range(25)]
    messages = preprocess_session("\n".join(lines))
    chunks = chunk_session(messages, session_id="test", chunk_size=10)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["session_id"] == "test"
