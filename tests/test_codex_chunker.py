"""Tests for Codex JSONL session parser."""
from __future__ import annotations

import json

from vecs.codex_chunker import (
    INDEXED_ROLES,
    extract_session_meta,
    preprocess_codex_session,
)


def _line(obj: dict) -> str:
    return json.dumps(obj)


def test_extracts_user_assistant_messages():
    raw = "\n".join([
        _line({"timestamp": "t0", "type": "session_meta", "payload": {"cwd": "/tmp", "id": "s1"}}),
        _line({"timestamp": "t1", "type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }}),
        _line({"timestamp": "t2", "type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "world"}],
        }}),
    ])
    msgs = preprocess_codex_session(raw)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["text"] for m in msgs] == ["hello", "world"]
    assert msgs[0]["timestamp"] == "t1"


def test_skips_developer_and_system_roles():
    raw = "\n".join([
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "<huge prompt>"}],
        }}),
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "system",
            "content": [{"type": "input_text", "text": "system msg"}],
        }}),
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "kept"}],
        }}),
    ])
    msgs = preprocess_codex_session(raw)
    assert [m["role"] for m in msgs] == ["user"]


def test_skips_non_message_payload_types():
    raw = "\n".join([
        _line({"type": "response_item", "payload": {"type": "function_call", "name": "bash"}}),
        _line({"type": "response_item", "payload": {"type": "reasoning", "text": "..."}}),
        _line({"type": "event_msg", "payload": {"type": "task_started"}}),
        _line({"type": "turn_context", "payload": {}}),
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "kept"}],
        }}),
    ])
    msgs = preprocess_codex_session(raw)
    assert len(msgs) == 1


def test_accepts_text_input_text_and_output_text_parts():
    raw = "\n".join([
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "output_text", "text": "c"},
                {"type": "image", "url": "ignored"},
            ],
        }}),
    ])
    msgs = preprocess_codex_session(raw)
    assert msgs[0]["text"] == "a\nb\nc"


def test_strips_system_reminders_and_base64():
    long_b64 = "A" * 200
    raw = _line({"type": "response_item", "payload": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": f"hello <system-reminder>secret</system-reminder> {long_b64}"}],
    }})
    msgs = preprocess_codex_session(raw)
    assert "system-reminder" not in msgs[0]["text"]
    assert "[binary data removed]" in msgs[0]["text"]


def test_unknown_payload_type_logged_via_set():
    seen: set[str] = set()
    raw = _line({"type": "response_item", "payload": {"type": "brand_new_thing", "foo": 1}})
    preprocess_codex_session(raw, unknown_payload_seen=seen)
    assert "brand_new_thing" in seen


def test_known_payload_type_not_added_to_unknown_seen():
    seen: set[str] = set()
    raw = _line({"type": "response_item", "payload": {"type": "function_call", "name": "x"}})
    preprocess_codex_session(raw, unknown_payload_seen=seen)
    assert seen == set()


def test_malformed_json_lines_are_skipped():
    raw = "\n".join([
        "{not valid json",
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "kept"}],
        }}),
        "",
        "garbage",
    ])
    msgs = preprocess_codex_session(raw)
    assert len(msgs) == 1


def test_extract_session_meta_returns_payload():
    raw = "\n".join([
        _line({"type": "session_meta", "payload": {"cwd": "/repo/foo", "id": "abc"}}),
        _line({"type": "event_msg", "payload": {}}),
    ])
    meta = extract_session_meta(raw)
    assert meta == {"cwd": "/repo/foo", "id": "abc"}


def test_extract_session_meta_returns_none_when_first_line_is_not_meta():
    raw = _line({"type": "event_msg", "payload": {}})
    assert extract_session_meta(raw) is None


def test_indexed_roles_constant_is_user_and_assistant_only():
    """Lock the role filter so a careless edit can't silently widen it."""
    assert INDEXED_ROLES == {"user", "assistant"}


def test_string_content_accepted_as_forward_compat_hedge():
    raw = _line({"type": "response_item", "payload": {
        "type": "message", "role": "assistant", "content": "plain string",
    }})
    msgs = preprocess_codex_session(raw)
    assert msgs[0]["text"] == "plain string"


def test_empty_content_skipped():
    raw = "\n".join([
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [],
        }}),
        _line({"type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "  "}],
        }}),
    ])
    msgs = preprocess_codex_session(raw)
    assert msgs == []
