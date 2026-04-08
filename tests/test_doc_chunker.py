from vecs.doc_chunker import chunk_doc


def test_chunk_by_headings():
    content = """# Introduction
Some intro text here.

## Section A
Content of section A.
More content.

## Section B
Content of section B.
"""
    chunks = chunk_doc(content, "readme.md")
    assert len(chunks) == 3
    assert chunks[0]["metadata"]["title"] == "Introduction"
    assert "Some intro text" in chunks[0]["text"]
    assert chunks[1]["metadata"]["title"] == "Section A"
    assert chunks[2]["metadata"]["title"] == "Section B"
    assert all(c["metadata"]["file_path"] == "readme.md" for c in chunks)


def test_chunk_no_headings_falls_back_to_paragraphs():
    content = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    chunks = chunk_doc(content, "notes.md")
    assert len(chunks) == 3
    assert "First paragraph" in chunks[0]["text"]
    assert "Third paragraph" in chunks[2]["text"]


def test_chunk_long_section_subsplit():
    long_section = "# Big Section\n\n" + "\n\n".join(
        f"Paragraph {i}.\n" + "Extra line.\n" * 50 for i in range(5)
    )
    chunks = chunk_doc(long_section, "big.md")
    assert len(chunks) > 1
    assert all(c["metadata"]["title"] == "Big Section" for c in chunks)


def test_chunk_skips_empty_sections():
    content = "# Empty\n\n## Real\nSome actual content that is long enough."
    chunks = chunk_doc(content, "test.md")
    titles = [c["metadata"]["title"] for c in chunks]
    assert "Real" in titles


def test_chunk_empty_content():
    assert chunk_doc("", "empty.md") == []
    assert chunk_doc("   ", "empty.md") == []


def test_chunk_index_increments():
    content = "# A\nContent A.\n\n# B\nContent B.\n\n# C\nContent C."
    chunks = chunk_doc(content, "test.md")
    indexes = [c["metadata"]["chunk_index"] for c in chunks]
    assert indexes == [0, 1, 2]


def test_subsections_stay_in_parent_chunk():
    content = """## Section A
Intro.

### Subsection A1
Detail.

### Subsection A2
More detail.

## Section B
Other content.
"""
    chunks = chunk_doc(content, "test.md")
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["title"] == "Section A"
    assert "Subsection A1" in chunks[0]["text"]
    assert "Subsection A2" in chunks[0]["text"]
    assert chunks[1]["metadata"]["title"] == "Section B"
