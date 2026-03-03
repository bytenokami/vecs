from __future__ import annotations

import json
import re

SKIP_TYPES = {"file-history-snapshot", "progress"}
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


def chunk_code_file(
    content: str,
    file_path: str,
    chunk_lines: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Split a code file into overlapping line-based chunks."""
    if not content.strip():
        return []

    lines = content.split("\n")
    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    "file_path": file_path,
                    "chunk_index": len(chunks),
                    "start_line": start + 1,
                    "end_line": end,
                },
            }
        )
        if end >= len(lines):
            break
        start += chunk_lines - overlap

    return chunks


def preprocess_session(raw_jsonl: str) -> list[dict]:
    """Parse a session JSONL file and extract clean messages."""
    messages = []
    for line in raw_jsonl.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")
        if msg_type in SKIP_TYPES:
            continue

        message = obj.get("message")
        if not message:
            continue

        role = message.get("role", "")
        content = message.get("content", "")

        # Handle content that's a list (multimodal messages)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part["text"])
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            continue

        # Strip system reminders
        content = SYSTEM_REMINDER_RE.sub("", content)
        # Strip base64
        content = BASE64_RE.sub("[binary data removed]", content)
        content = content.strip()

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "text": content,
                "timestamp": obj.get("timestamp", ""),
            }
        )

    return messages


def chunk_session(
    messages: list[dict],
    session_id: str,
    chunk_size: int = 10,
) -> list[dict]:
    """Group preprocessed messages into chunks."""
    chunks = []
    for i in range(0, len(messages), chunk_size):
        group = messages[i : i + chunk_size]
        combined = "\n\n".join(
            f"[{m['role']}]: {m['text']}" for m in group
        )
        first_ts = group[0].get("timestamp", "")
        last_ts = group[-1].get("timestamp", "")
        chunks.append(
            {
                "text": combined,
                "metadata": {
                    "session_id": session_id,
                    "chunk_index": len(chunks),
                    "start_timestamp": first_ts,
                    "end_timestamp": last_ts,
                    "message_count": len(group),
                },
            }
        )
    return chunks
