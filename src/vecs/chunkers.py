from __future__ import annotations


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
