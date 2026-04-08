from __future__ import annotations

from pathlib import PurePath

import tree_sitter_c_sharp as ts_cs
import tree_sitter_typescript as ts_ts
from tree_sitter import Language, Parser

from vecs.chunkers import chunk_code_file

CS_LANGUAGE = Language(ts_cs.language())
TS_LANGUAGE = Language(ts_ts.language_typescript())
TSX_LANGUAGE = Language(ts_ts.language_tsx())

# Map file extensions to tree-sitter languages
LANGUAGE_MAP: dict[str, Language] = {
    ".cs": CS_LANGUAGE,
    ".ts": TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}

# Node types that represent top-level declarations worth chunking.
# For C# we do NOT include namespace_declaration — we recurse into it
# to find class/struct/etc. declarations inside.
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    ".cs": {
        "class_declaration",
        "struct_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    },
    ".ts": {
        "class_declaration",
        "function_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "export_statement",
    },
    ".tsx": {
        "class_declaration",
        "function_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
        "export_statement",
    },
}

# Minimum lines for a chunk to stand alone (otherwise merge with adjacent)
MIN_CHUNK_LINES = 5


def _extract_declarations(root, node_types: set[str]) -> list[tuple[int, int]]:
    """Walk the AST and return (start_line, end_line) for top-level declarations.

    Recurses up to depth 3 to handle structures like
    ``namespace > declaration_list > class_declaration`` in C#.
    When a matching node is found we record its span and do NOT recurse
    further into it (nested classes stay inside their parent chunk).
    """
    declarations: list[tuple[int, int]] = []

    def walk(node, depth=0):
        if node.type in node_types and depth >= 1:
            declarations.append((node.start_point[0], node.end_point[0]))
            return  # Don't recurse into matched nodes
        if depth <= 4:
            for child in node.children:
                walk(child, depth + 1)

    walk(root)
    return declarations


def chunk_code_file_ast(
    content: str,
    file_path: str,
    max_chunk_lines: int = 500,
    chunk_lines: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Chunk a code file using AST boundaries when possible.

    Falls back to line-based chunking for unsupported languages or
    when AST parsing yields no declarations.

    Args:
        content: File content as string.
        file_path: Relative file path (used to detect language and for metadata).
        max_chunk_lines: Maximum lines per chunk; declarations exceeding this
            are sub-split using line-based chunking.
        chunk_lines: Line-based chunk size for fallback.
        overlap: Line overlap for fallback chunking.

    Returns:
        A list of chunk dicts, each with ``text`` and ``metadata`` keys.
    """
    if not content.strip():
        return []

    ext = PurePath(file_path).suffix
    language = LANGUAGE_MAP.get(ext)

    if language is None:
        return chunk_code_file(content, file_path, chunk_lines, overlap)

    parser = Parser(language)
    tree = parser.parse(content.encode())

    node_types = CHUNK_NODE_TYPES.get(ext, set())
    declarations = _extract_declarations(tree.root_node, node_types)

    if not declarations:
        return chunk_code_file(content, file_path, chunk_lines, overlap)

    lines = content.split("\n")
    total_lines = len(lines)

    # Sort declarations by start line
    declarations.sort()

    # Build regions from declarations, merging adjacent small ones
    raw_regions: list[tuple[int, int]] = []
    for start, end in declarations:
        if (
            raw_regions
            and (start - raw_regions[-1][1] <= 1)
            and (end - raw_regions[-1][0] + 1 < MIN_CHUNK_LINES * 3)
        ):
            # Merge with previous if adjacent and combined is still small
            raw_regions[-1] = (raw_regions[-1][0], end)
        else:
            raw_regions.append((start, end))

    # Assign preamble (imports, usings) before first declaration to that chunk
    if raw_regions and raw_regions[0][0] > 0:
        raw_regions[0] = (0, raw_regions[0][1])

    # Assign any trailing code after last declaration to that chunk
    if raw_regions and raw_regions[-1][1] < total_lines - 1:
        raw_regions[-1] = (raw_regions[-1][0], total_lines - 1)

    # Fill gaps between declarations — attach gap lines to the preceding chunk
    filled: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(raw_regions):
        if filled:
            prev_end = filled[-1][1]
            if start > prev_end + 1:
                # Gap between previous chunk end and this chunk start;
                # extend previous chunk to cover the gap
                filled[-1] = (filled[-1][0], start - 1)
        filled.append((start, end))

    # Build final chunks, splitting any that exceed max_chunk_lines
    chunks: list[dict] = []
    for start, end in filled:
        chunk_text = "\n".join(lines[start : end + 1])
        line_count = end - start + 1

        if line_count > max_chunk_lines:
            # Sub-split large declarations using line-based chunking
            sub_chunks = chunk_code_file(chunk_text, file_path, chunk_lines, overlap)
            for sc in sub_chunks:
                sc["metadata"]["start_line"] = start + sc["metadata"]["start_line"]
                sc["metadata"]["end_line"] = start + sc["metadata"]["end_line"]
                sc["metadata"]["chunk_index"] = len(chunks)
                chunks.append(sc)
        else:
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_index": len(chunks),
                        "start_line": start + 1,
                        "end_line": end + 1,
                    },
                }
            )

    return chunks
