"""Codex CLI session JSONL parser.

Codex stores transcripts at ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl with
shape: {"timestamp", "type", "payload"}. We index only `response_item` lines
whose `payload.type == "message"` and `payload.role in {"user", "assistant"}`.

Output shape mirrors `chunkers.preprocess_session` so `chunkers.chunk_session`
can consume both Claude Code and Codex outputs with no branching.
"""
from __future__ import annotations

import json
import re

# Codex top-level types observed in the wild. Anything not in this set is
# silently ignored. Adding a known type prevents the once-per-run debug log.
KNOWN_TOP_TYPES = {"session_meta", "event_msg", "response_item", "turn_context"}

# response_item payload types. Only "message" is indexed; others are tolerated.
KNOWN_PAYLOAD_TYPES = {
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "tool_call",
    "tool_call_output",
}

# Content-part types accepted as text. Codex emits input_text / output_text;
# we tolerate plain "text" for forward-compat with format changes.
TEXT_PART_TYPES = {"text", "input_text", "output_text"}

# Roles indexed. developer / system / tool are skipped (giant prompts, noise).
INDEXED_ROLES = {"user", "assistant"}

SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


def extract_session_meta(raw_jsonl: str) -> dict | None:
    """Parse only the first valid JSON line and return session_meta payload.

    Returns the payload dict (with `cwd`, `id`, etc.) or None if the file is
    empty / malformed / does not start with a session_meta record.
    """
    for line in raw_jsonl.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if obj.get("type") != "session_meta":
            return None
        payload = obj.get("payload")
        if isinstance(payload, dict):
            return payload
        return None
    return None


def preprocess_codex_session(
    raw_jsonl: str,
    unknown_payload_seen: set[str] | None = None,
) -> list[dict]:
    """Parse a Codex JSONL session and return user/assistant messages.

    The output shape matches `chunkers.preprocess_session` so the same
    `chunk_session` function can group either source into chunks.

    Args:
        raw_jsonl: Full file contents (or new-bytes tail for incremental runs).
        unknown_payload_seen: If provided, payload-type strings we did not
            recognize are added to this set. Callers should log new entries
            once per index run for telemetry.

    Tolerant of schema drift: unknown top types and unknown payload types
    are skipped. Unknown content-part types are skipped. Never raises.
    """
    messages: list[dict] = []

    for line in raw_jsonl.split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        top_type = obj.get("type", "")
        if top_type != "response_item":
            # Silent skip. Unknown top-level types stay quiet (most volume is
            # event_msg / turn_context which we already know about).
            continue

        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type", "")
        if payload_type != "message":
            if (
                unknown_payload_seen is not None
                and payload_type
                and payload_type not in KNOWN_PAYLOAD_TYPES
            ):
                unknown_payload_seen.add(payload_type)
            continue

        role = payload.get("role", "")
        if role not in INDEXED_ROLES:
            continue

        content_parts = payload.get("content")
        text = _extract_text(content_parts)
        if not text:
            continue

        # Strip embedded system-reminders + base64 (same hygiene as Claude path).
        text = SYSTEM_REMINDER_RE.sub("", text)
        text = BASE64_RE.sub("[binary data removed]", text)
        text = text.strip()
        if not text:
            continue

        messages.append(
            {
                "role": role,
                "text": text,
                "timestamp": obj.get("timestamp", ""),
            }
        )

    return messages


def _extract_text(content: object) -> str:
    """Pull text out of Codex content list. Returns empty string on any fail.

    Codex content is `[{type: input_text|output_text|text, text: "..."}]`.
    Plain string content is also accepted (forward-compat hedge).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in TEXT_PART_TYPES:
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)
