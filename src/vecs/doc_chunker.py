from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
MIN_CHUNK_CHARS = 30
MAX_SECTION_LINES = 200


def chunk_doc(content: str, file_path: str) -> list[dict]:
    """Chunk a markdown/text document by headings, with paragraph fallback."""
    if not content.strip():
        return []

    heading_sections = _split_by_headings(content)
    if heading_sections:
        sections = heading_sections
        apply_min_chars = True
    else:
        sections = _split_by_paragraphs(content)
        apply_min_chars = False

    chunks = []
    for title, text in sections:
        lines = text.split("\n")
        if len(lines) > MAX_SECTION_LINES:
            sub_parts = _split_by_paragraphs(text)
            for _, sub_text in sub_parts:
                if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(_make_chunk(sub_text.strip(), file_path, title, len(chunks)))
        else:
            # For heading sections: skip only if the body (after heading line) is empty
            body = _body_text(text) if apply_min_chars else text.strip()
            if not body:
                continue
            chunks.append(_make_chunk(text.strip(), file_path, title, len(chunks)))

    return chunks


def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file."""
    import pymupdf
    doc = pymupdf.open(file_path)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    """Split content at H1/H2 heading boundaries.

    H1 and H2 always start new chunks. H3 under an H2 stays in the H2 chunk.
    """
    # Only treat H1 and H2 as chunk boundaries; H3 is absorbed as body text.
    split_re = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(split_re.finditer(content))
    if not matches:
        # No H1/H2 found — check if there are any headings at all (H3+)
        if not HEADING_RE.search(content):
            return []
        # Only H3 headings present: treat the whole content as one section
        first = HEADING_RE.search(content)
        title = first.group(2).strip() if first else "untitled"
        return [(title, content)]

    sections = []

    # Capture preamble text before the first heading
    if matches[0].start() > 0:
        preamble = content[:matches[0].start()]
        if preamble.strip():
            sections.append(("untitled", preamble))

    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        sections.append((title, content[start:end]))

    return sections


def _split_by_paragraphs(content: str) -> list[tuple[str, str]]:
    """Split content by double newlines. Returns [("untitled", text), ...]."""
    parts = re.split(r"\n{2,}", content.strip())
    return [("untitled", p.strip()) for p in parts if p.strip()]


def _body_text(text: str) -> str:
    """Return text with the leading heading line removed, for size checks."""
    lines = text.strip().splitlines()
    if lines and HEADING_RE.match(lines[0]):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _make_chunk(text: str, file_path: str, title: str, index: int) -> dict:
    return {
        "text": text,
        "metadata": {
            "file_path": file_path,
            "title": title,
            "chunk_index": index,
        },
    }
