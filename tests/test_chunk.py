from itertools import pairwise
from pathlib import Path

import pytest

from mini_rag.chunk import (
    Block,
    build_section,
    chunk_blocks,
    encoding,
    merge_small_sections,
    split_section,
    split_sections,
)

VOCABULARY = {
    "a": ["cat", "dog", "sun", "sky", "red", "big", "map", "run", "sea", "car"],
    "b": ["one", "two", "six", "ten", "new", "old", "hot", "low", "top", "end"],
}


def words(count: int, vocabulary: str = "a") -> str:
    """Text of exactly count tokens: every word (with its leading space) is one token."""
    vocab = VOCABULARY[vocabulary]
    return " ".join(vocab[i % len(vocab)] for i in range(count))


def tokens(text: str) -> int:
    return len(encoding.encode(text))


# --- sections -----------------------------------------------------------------


def test_new_section_starts_at_heading() -> None:
    blocks = [
        Block("1 Intro", 1, heading=True),
        Block("intro text", 1),
        Block("2 Methods", 2, heading=True),
        Block("methods text", 2),
    ]

    assert [[b.text for b in s] for s in split_sections(blocks)] == [
        ["1 Intro", "intro text"],
        ["2 Methods", "methods text"],
    ]


def test_consecutive_headings_stay_in_one_section() -> None:
    blocks = [
        Block("text before", 1),
        Block("3 Results", 1, heading=True),
        Block("3.1 First result", 1, heading=True),
        Block("result text", 1),
    ]

    assert [[b.text for b in s] for s in split_sections(blocks)] == [
        ["text before"],
        ["3 Results", "3.1 First result", "result text"],
    ]


def test_section_continues_across_pages() -> None:
    blocks = [Block("1 Intro", 1, heading=True), Block("page one", 1), Block("page two", 2)]

    (section,) = split_sections(blocks)
    built = build_section(section)

    assert (built.page_start, built.page_end) == (1, 2)


def test_blank_blocks_are_ignored() -> None:
    assert split_sections([Block("  ", 1), Block("text", 1)]) == [[Block("text", 1)]]


def test_joined_block_uses_line_break_others_paragraph_break() -> None:
    section = build_section(
        [
            Block("Heading", 1, heading=True),
            Block("first half", 1),
            Block("second half", 2, joined=True),
        ]
    )

    assert section.text == "Heading\n\nfirst half\nsecond half"


# --- merging small sections -----------------------------------------------------


def test_small_section_merges_forward_into_next() -> None:
    sections = [
        [Block("1.3 Parent", 1, heading=True), Block("short intro", 1)],
        [Block("1.3.1 Child", 1, heading=True), Block(words(300), 2)],
    ]

    (merged,) = merge_small_sections(sections, min_tokens=50)

    assert merged.text.startswith("1.3 Parent\n\nshort intro\n\n1.3.1 Child")
    assert (merged.page_start, merged.page_end) == (1, 2)


def test_small_last_section_merges_backward() -> None:
    sections = [
        [Block("1 Big", 1, heading=True), Block(words(300), 1)],
        [Block("2 Tiny", 3, heading=True), Block("tail", 3)],
    ]

    (merged,) = merge_small_sections(sections, min_tokens=50)

    assert merged.text.endswith("2 Tiny\n\ntail")
    assert merged.page_end == 3


def test_large_sections_are_not_merged() -> None:
    sections = [
        [Block("1 A", 1, heading=True), Block(words(100), 1)],
        [Block("2 B", 2, heading=True), Block(words(100), 2)],
    ]

    assert len(merge_small_sections(sections, min_tokens=50)) == 2


def test_merged_section_starts_after_paragraph_break_even_if_joined() -> None:
    sections = [[Block("tiny", 1)], [Block(words(100), 2, joined=True)]]

    (merged,) = merge_small_sections(sections, min_tokens=50)

    assert merged.text.startswith("tiny\n\n")


# --- splitting large sections ---------------------------------------------------


def test_words_helper_is_one_token_per_word() -> None:
    assert tokens(words(50)) == 50
    assert tokens(words(50, "b")) == 50


def test_section_within_chunk_size_is_one_chunk() -> None:
    section = build_section([Block(words(50), 4)])

    (chunk,) = split_section(section, chunk_size=100, chunk_overlap=20)

    assert chunk.text == section.text
    assert (chunk.page_start, chunk.page_end) == (4, 4)
    assert chunk.token_count == tokens(chunk.text)


@pytest.mark.parametrize("length", [101, 150, 181, 250, 999])
def test_windows_respect_size_and_share_exact_overlap(length: int) -> None:
    section = build_section([Block(words(length), 1)])
    assert len(section.tokens) == length

    chunks = split_section(section, chunk_size=100, chunk_overlap=20)

    assert all(c.token_count <= 100 for c in chunks)
    for previous, current in pairwise(chunks):
        assert encoding.encode(previous.text)[-20:] == encoding.encode(current.text)[:20]
    assert encoding.encode(chunks[-1].text)[-5:] == list(section.tokens[-5:])


def test_windows_are_balanced_no_near_duplicate_tail() -> None:
    # Greedy windows of 100 with step 80 would leave a last window of 21 tokens
    # (1 new token + 20 overlap). Balanced windows split the section evenly.
    section = build_section([Block(words(181), 1)])

    sizes = [c.token_count for c in split_section(section, chunk_size=100, chunk_overlap=20)]

    assert len(sizes) == 3
    assert max(sizes) - min(sizes) <= 1


def test_window_pages_follow_the_text() -> None:
    section = build_section([Block(words(150, "a"), 1), Block(words(150, "b"), 2)])

    chunks = split_section(section, chunk_size=100, chunk_overlap=10)
    pages = [(c.page_start, c.page_end) for c in chunks]

    assert pages[0] == (1, 1)
    assert (1, 2) in pages
    assert pages[-1] == (2, 2)
    for chunk, (first, last) in zip(chunks, pages, strict=True):
        found = set(chunk.text.split())
        assert bool(found & set(VOCABULARY["a"])) == (first == 1)
        assert bool(found & set(VOCABULARY["b"])) == (last == 2)


# --- whole document -----------------------------------------------------------


def test_chunk_blocks_end_to_end() -> None:
    blocks = [
        Block("Title", 1, heading=True),
        Block("Authors", 1),
        Block("1 Introduction", 2, heading=True),
        Block(words(250), 2),
        Block(words(250), 3, joined=True),
        Block("2 Conclusion", 4, heading=True),
        Block(words(120), 4),
    ]

    chunks = chunk_blocks(blocks, chunk_size=300, chunk_overlap=30, min_tokens=50)

    assert chunks[0].text.startswith("Title\n\nAuthors\n\n1 Introduction")
    assert chunks[-1].text.startswith("2 Conclusion")
    assert all(c.token_count <= 300 for c in chunks)
    assert [(c.page_start, c.page_end) for c in chunks][-1] == (4, 4)
    assert any(c.page_start != c.page_end for c in chunks)


def test_chunk_blocks_empty_document() -> None:
    assert chunk_blocks([], chunk_size=300, chunk_overlap=30, min_tokens=50) == []


# --- ingest wiring --------------------------------------------------------------


def test_process_document_uses_real_page_numbers_and_page_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mini_rag import ingest
    from mini_rag.config import IngestConfig
    from mini_rag.models import Manifest
    from mini_rag.pdf_parsers import PdfBlock, PdfDocument, PdfPage

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"mock PDF")
    # Page 2 (contents) was skipped by the parser: pages are 1 and 3.
    document = PdfDocument(
        id="doc",
        source_path=str(pdf_path),
        title="Doc",
        collection="test",
        pages=[
            PdfPage(
                number=1,
                text="",
                blocks=[
                    PdfBlock(text="1 Intro", kind="section-header"),
                    PdfBlock(text=words(80), kind="text"),
                ],
            ),
            PdfPage(
                number=3,
                text="",
                blocks=[
                    PdfBlock(text=words(80), kind="text", joined=True),
                    PdfBlock(text="2 Next", kind="section-header"),
                    PdfBlock(text=words(80), kind="text"),
                ],
            ),
        ],
    )
    monkeypatch.setattr(ingest, "parse_pdf_text", lambda *_: document)
    manifest = Manifest(
        id="doc", title="Doc", url="https://example.test", year="2026", collection="test"
    )
    config = IngestConfig.model_construct(
        pdf_dir=tmp_path, chunk_size=800, chunk_overlap=120, min_chunk_tokens=10
    )

    chunks, stats = ingest.process_document(manifest, config)

    assert [(c.chunk_id, c.page, c.page_end) for c in chunks] == [
        ("doc_p001_c001", 1, 3),
        ("doc_p003_c001", 3, 3),
    ]
    assert stats["chunks"] == 2
