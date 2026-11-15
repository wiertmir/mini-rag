import re
from itertools import pairwise
from pathlib import Path

import pytest

from mini_rag.chunk import (
    Block,
    build_section,
    chunk_blocks,
    continues,
    drop_skipped,
    encoding,
    merge_small_sections,
    move_boxes,
    split_section,
    split_sections,
)
from mini_rag.config import DEFAULT_BOX_PATTERN, DEFAULT_SKIP_PATTERNS

SKIP = [re.compile(p) for p in DEFAULT_SKIP_PATTERNS]
BOX = re.compile(DEFAULT_BOX_PATTERN)

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


def test_joined_block_is_glued_others_get_paragraph_break() -> None:
    section = build_section(
        [
            Block("Heading", 1, heading=True),
            Block("first half", 1),
            Block(" second half", 2, joined=True),
        ]
    )

    assert section.text == "Heading\n\nfirst half second half"


def test_joined_block_starting_with_punctuation_is_glued_without_space() -> None:
    section = build_section(
        [Block("observed changes", 1), Block(", including more", 2, joined=True)]
    )

    assert section.text == "observed changes, including more"


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
    # A paragraph continuing on the next page (joined): windows may span the page break.
    section = build_section([Block(words(150, "a"), 1), Block(words(150, "b"), 2, joined=True)])

    chunks = split_section(section, chunk_size=100, chunk_overlap=10)
    pages = [(c.page_start, c.page_end) for c in chunks]

    assert pages[0] == (1, 1)
    assert (1, 2) in pages
    assert pages[-1] == (2, 2)
    for chunk, (first, last) in zip(chunks, pages, strict=True):
        found = set(chunk.text.split())
        assert bool(found & set(VOCABULARY["a"])) == (first == 1)
        assert bool(found & set(VOCABULARY["b"])) == (last == 2)


# --- paragraph-aware windows ---------------------------------------------------


def paragraphs_section(*sizes: int) -> str:
    """Paragraphs of the given token sizes, separated by blank lines; vocabularies alternate."""
    return "\n\n".join(words(size, "ab"[i % 2]) for i, size in enumerate(sizes))


def test_next_window_starts_at_paragraph_inside_previous_window() -> None:
    # The previous window keeps its token cut; the next one starts at a paragraph start.
    text = paragraphs_section(50, 40, 50)
    paragraphs = text.split("\n\n")
    section = build_section([Block(text, 1)])

    first, second = split_section(section, chunk_size=100, chunk_overlap=30)

    assert second.text == "\n\n".join(paragraphs[1:])
    assert first.text.startswith(paragraphs[0])
    assert not first.text.endswith(paragraphs[1])  # token cut, not shortened to a blank line
    assert max(first.token_count, second.token_count) <= 100


@pytest.mark.parametrize("sizes", [(60, 90, 60), (30, 150, 30), (70, 70, 70), (95, 95, 95)])
def test_neighbouring_windows_always_share_text(sizes: tuple[int, ...]) -> None:
    # Regression (WGI_CH01 p. 7): snapping to paragraphs must never remove the overlap.
    text = paragraphs_section(*sizes)
    section = build_section([Block(text, 1)])

    chunks = split_section(section, chunk_size=100, chunk_overlap=30)

    assert all(c.token_count <= 100 for c in chunks)
    for previous, current in pairwise(chunks):
        assert current.text[:20] in previous.text
    assert text.startswith(chunks[0].text)
    assert text.endswith(chunks[-1].text)


def test_window_start_falls_back_to_token_cut_when_paragraph_does_not_fit() -> None:
    # One 300-token paragraph: no blank line to snap to, so plain token windows remain.
    section = build_section([Block(words(300), 1)])

    chunks = split_section(section, chunk_size=100, chunk_overlap=20)

    for previous, current in pairwise(chunks):
        assert encoding.encode(previous.text)[-20:] == encoding.encode(current.text)[:20]


def test_no_gap_between_windows() -> None:
    text = paragraphs_section(40, 90, 15, 120, 60, 33)
    section = build_section([Block(text, 1)])

    chunks = split_section(section, chunk_size=100, chunk_overlap=20)

    covered = "".join(chunks[0].text)
    for chunk in chunks[1:]:
        assert chunk.token_count <= 100
        # the next chunk starts inside text already covered (overlap) or right after a break
        head = chunk.text[:30]
        assert head in covered or head in text
        covered += chunk.text
    for paragraph in text.split("\n\n"):
        assert any(paragraph[:20] in c.text for c in chunks)


# --- skipped blocks ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("from the near", "to the long term", True),
        ("observed changes", ", including", True),
        ("ends here.", "lower start", False),
        ("no period", "Upper start", False),
    ],
)
def test_continues(previous: str, current: str, expected: bool) -> None:
    assert continues(previous, current) is expected


@pytest.mark.parametrize(
    "caption",
    [
        "Figure 1.1 | The structure of the report.",
        "Table 2.3 | Summary.",
        "Box 1.1, Figure 1 | The approach.",
        "Cross-Chapter Box 1.1, Table 1 | Findings.",
        "Box 1.1 (continued)",
        "Cross-Chapter Box 1.2 (continued)",
    ],
)
def test_caption_is_skipped(caption: str) -> None:
    assert drop_skipped([Block(caption, 1)], SKIP) == []


@pytest.mark.parametrize(
    "sentence",
    ["Figure 1.2 shows the distribution.", "The Figure 1.1 shows it.", "Table 1.1 summarises."],
)
def test_reference_to_figure_is_kept(sentence: str) -> None:
    assert drop_skipped([Block(sentence, 1)], SKIP) == [Block(sentence, 1)]


def test_text_around_caption_is_glued_when_it_continues() -> None:
    blocks = [
        Block("Chapter 4 assesses projections, from the near", 1),
        Block("Figure 1.1 | The structure of the AR6 WGI Report.", 1),
        Block("to the long term, including 2100.", 2),
    ]

    section = build_section(drop_skipped(blocks, SKIP))

    assert (
        section.text
        == "Chapter 4 assesses projections, from the near to the long term, including 2100."
    )


def test_text_around_caption_stays_separate_paragraphs_otherwise() -> None:
    blocks = [Block("First paragraph.", 1), Block("Figure 1.1 | Caption.", 1), Block("Second.", 1)]

    assert build_section(drop_skipped(blocks, SKIP)).text == "First paragraph.\n\nSecond."


# --- boxes ------------------------------------------------------------------------


def test_box_interrupting_text_moves_before_next_heading() -> None:
    blocks = [
        Block("1.1 Intro", 1, heading=True),
        Block("The main text is cut by the", 1),
        Block("Box 1.1 | A box", 1),
        Block("Box paragraph is split across the", 1),
        Block("box rest, which the reader skips.", 2),
        Block("main text again after the box.", 2),
        Block("Another main paragraph.", 2),
        Block("1.2 Next", 3, heading=True),
    ]

    texts = [b.text for b in move_boxes(blocks, BOX)]

    assert texts == [
        "1.1 Intro",
        "The main text is cut by the",
        "main text again after the box.",
        "Another main paragraph.",
        "Box 1.1 | A box",
        "Box paragraph is split across the",
        "box rest, which the reader skips.",
        "1.2 Next",
    ]


def test_box_title_is_a_heading_and_starts_a_section() -> None:
    blocks = [Block("Main.", 1), Block("Box 1.1 | A box", 1), Block("Box text.", 1)]

    sections = split_sections(move_boxes(blocks, BOX))

    assert [[b.text for b in s] for s in sections] == [["Main."], ["Box 1.1 | A box", "Box text."]]


def test_box_ending_at_numbered_heading_stays_in_place() -> None:
    blocks = [
        Block("Main paragraph.", 1),
        Block("Box 1.1 | A box", 1),
        Block("Box text.", 1),
        Block("1.2 Next", 2, heading=True),
    ]

    assert [b.text for b in move_boxes(blocks, BOX)] == [b.text for b in blocks]


def test_box_at_document_end_is_kept() -> None:
    blocks = [Block("Main text without end", 1), Block("Box 1.1 | A box", 1), Block("Box text.", 1)]

    assert [b.text for b in move_boxes(blocks, BOX)] == [b.text for b in blocks]


def test_continued_main_text_is_glued_after_box_is_moved() -> None:
    blocks = [
        Block("1 Intro", 1, heading=True),
        Block("The main text is cut by the", 1),
        Block("Box 1.1 | A box", 1),
        Block("Box text.", 1),
        Block("box that follows.", 2),
    ]

    chunks = chunk_blocks(
        blocks,
        chunk_size=300,
        chunk_overlap=30,
        min_tokens=0,
        skip_patterns=(),
        box_pattern=DEFAULT_BOX_PATTERN,
    )

    assert chunks[0].text == "1 Intro\n\nThe main text is cut by the box that follows."
    assert chunks[1].text.startswith("Box 1.1 | A box")


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
