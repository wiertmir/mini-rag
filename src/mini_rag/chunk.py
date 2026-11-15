"""Section-aware chunking.

1. Blocks matching a skip pattern (figure/table captions, "Box 1.1 (continued)") are dropped;
   text interrupted by them is glued back together.
2. Boxes interrupting the main text are moved to the end of their section.
3. Sections start at headings (consecutive headings stay together) and may span pages;
   sections below min_tokens merge forward.
4. Sections above chunk_size are split into balanced windows whose boundaries snap to
   paragraph breaks (blank lines) where possible.
"""

import re
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import rich_struct.log
import tiktoken

log = rich_struct.log.get_logger()

CHUNK_MODEL = "text-embedding-3-small"

encoding = tiktoken.encoding_for_model(CHUNK_MODEL)

PARAGRAPH_BREAK = "\n\n"
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s")
"""A numbered section heading such as "1.2.3 Title"; it always ends a box."""

GLUE_WITHOUT_SPACE = ",.;:)]"
CONTINUATION_PUNCTUATION = ",.;:)]-–—"


class Block(NamedTuple):
    """One text region of the document, in reading order.

    joined: the block continues the previous one (column, page or caption interruption),
    so the two are glued into one paragraph instead of separated by a blank line.
    """

    text: str
    page: int
    heading: bool = False
    joined: bool = False


class TextChunk(NamedTuple):
    text: str
    token_count: int
    page_start: int
    page_end: int


class _Span(NamedTuple):
    start: int
    end: int
    page: int


class Section(NamedTuple):
    text: str
    spans: tuple[_Span, ...]
    tokens: tuple[int, ...]

    @property
    def page_start(self) -> int:
        return self.spans[0].page

    @property
    def page_end(self) -> int:
        return max(span.page for span in self.spans)


# --- cleaning: skipped blocks and boxes ------------------------------------------


def continues(previous: str, current: str) -> bool:
    """True when current reads as the continuation of previous (an interrupted sentence).

    previous must not end with a period; current must start with a lowercase letter
    or a punctuation mark.
    """
    previous, current = previous.rstrip(), current.lstrip()
    if not previous or not current or previous.endswith("."):
        return False
    return current[0].islower() or current[0] in CONTINUATION_PUNCTUATION


def drop_skipped(blocks: Iterable[Block], skip_patterns: Sequence[re.Pattern[str]]) -> list[Block]:
    """Drop blocks matching a skip pattern; glue the text around them when it continues."""
    kept: list[Block] = []
    skipped = False
    for block in blocks:
        if not block.text.strip():
            continue
        if any(pattern.search(block.text.strip()) for pattern in skip_patterns):
            skipped = True
            continue
        if skipped and kept and not kept[-1].heading and continues(kept[-1].text, block.text):
            block = block._replace(joined=True)
        kept.append(block)
        skipped = False
    return kept


def _ends_box(block: Block, box_pattern: re.Pattern[str]) -> bool:
    text = block.text.strip()
    return bool(box_pattern.search(text)) or (block.heading and bool(NUMBERED_HEADING.match(text)))


def move_boxes(blocks: Sequence[Block], box_pattern: re.Pattern[str]) -> list[Block]:
    """Move boxes that interrupt the main text to the end of the section (before the next heading).

    A box starts at a block matching box_pattern and ends before the first block that
    - is a numbered section heading or another box title: the box already ends its section
      and stays where it is, or
    - continues the main text interrupted by the box (see continues()): the box is cut out,
      the main text is glued together and the box is re-inserted before the next heading.
    A box title is treated as a heading, so a box always starts its own section.
    """
    out: list[Block] = []
    pending: list[Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if not box_pattern.search(block.text.strip()):
            if block.heading and pending:
                out.extend(pending)
                pending = []
            out.append(block)
            i += 1
            continue

        main = out[-1] if out and not out[-1].heading else None
        box = [block._replace(heading=True, joined=False)]
        i += 1
        resumed = False
        while i < len(blocks):
            candidate = blocks[i]
            if _ends_box(candidate, box_pattern):
                break
            box_paragraph_closed = box[-1].heading or box[-1].text.rstrip().endswith(".")
            if main is not None and box_paragraph_closed and continues(main.text, candidate.text):
                resumed = True
                break
            box.append(candidate if len(box) > 1 else candidate._replace(joined=False))
            i += 1

        if resumed:
            pending.extend(box)
            out.append(blocks[i]._replace(joined=True))
            i += 1
        else:
            if pending:
                out.extend(pending)
                pending = []
            out.extend(box)
    out.extend(pending)
    return out


# --- sections ------------------------------------------------------------------


def split_sections(blocks: Iterable[Block]) -> list[list[Block]]:
    """Start a new section at every heading, unless the previous block is a heading too.

    Consecutive headings ("3", "3.1 Introduction") stay together with the text below them.
    """
    sections: list[list[Block]] = []
    current: list[Block] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if block.heading and current and not current[-1].heading:
            sections.append(current)
            current = []
        current.append(block)
    if current:
        sections.append(current)
    return sections


def _separator(block: Block) -> str:
    if not block.joined:
        return PARAGRAPH_BREAK
    return "" if block.text.lstrip()[:1] in tuple(GLUE_WITHOUT_SPACE) else " "


def build_section(blocks: Sequence[Block]) -> Section:
    parts: list[str] = []
    spans: list[_Span] = []
    offset = 0
    for i, block in enumerate(blocks):
        text = block.text.strip()
        if i:
            separator = _separator(block)
            parts.append(separator)
            offset += len(separator)
        parts.append(text)
        spans.append(_Span(offset, offset + len(text), block.page))
        offset += len(text)
    text = "".join(parts)
    return Section(text, tuple(spans), tuple(encoding.encode(text)))


def merge_small_sections(sections: Sequence[Sequence[Block]], min_tokens: int) -> list[Section]:
    """Merge every section below min_tokens forward into the section that follows it.

    A short parent section ("1.3 ..." directly followed by "1.3.1 ...") thus stays with the text
    below it. A small last section merges backward instead. A merged section may exceed
    chunk_size; it is then split into windows like any other large section.
    """
    merged: list[tuple[list[Block], Section]] = []
    pending: list[Block] = []
    for blocks in sections:
        combined = [*pending, *_as_new_paragraph(blocks)] if pending else list(blocks)
        section = build_section(combined)
        if len(section.tokens) < min_tokens:
            pending = combined
            continue
        merged.append((combined, section))
        pending = []
    if pending:
        if merged:
            previous_blocks, _ = merged.pop()
            pending = [*previous_blocks, *_as_new_paragraph(pending)]
        merged.append((pending, build_section(pending)))
    return [section for _, section in merged]


def _as_new_paragraph(blocks: Sequence[Block]) -> list[Block]:
    """A merged section always starts after a paragraph break."""
    return [blocks[0]._replace(joined=False), *blocks[1:]]


# --- windows -------------------------------------------------------------------


def _pages_between(section: Section, start: int, end: int) -> tuple[int, int]:
    pages = [span.page for span in section.spans if span.start < end and span.end > start]
    if not pages:  # a window of separators only; cannot happen with real text
        pages = [section.page_start]
    return min(pages), max(pages)


def _balanced_windows(length: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Token windows of at most chunk_size, evenly sized, sharing chunk_overlap tokens."""
    max_step = chunk_size - chunk_overlap
    count = -(-(length - chunk_size) // max_step) + 1
    step = -(-(length - chunk_overlap) // count)
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + step + chunk_overlap, length)
        windows.append((start, end))
        if end == length:
            return windows
        start += step


def split_section(section: Section, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Split a section into windows of at most chunk_size tokens.

    Windows start from balanced token windows sharing chunk_overlap tokens, then snap to
    paragraph breaks:
    - a window ends at the last blank line inside its overlap with the next window;
    - the next window starts at the nearest paragraph start at or before its nominal start,
      provided the window still fits in chunk_size; otherwise right after the first blank line
      inside the overlap; otherwise it keeps the token cut.
    """
    tokens = section.tokens
    text = section.text
    if len(tokens) <= chunk_size:
        return [TextChunk(text, len(tokens), section.page_start, section.page_end)]

    _, offsets = encoding.decode_with_offsets(list(tokens))

    def char_at(token: int) -> int:
        return offsets[token] if token < len(tokens) else len(text)

    def fits(start: int, end: int) -> bool:
        return len(encoding.encode(text[start:end])) <= chunk_size

    windows = _balanced_windows(len(tokens), chunk_size, chunk_overlap)
    breaks = [m.start() for m in re.finditer(PARAGRAPH_BREAK, text)]
    last = len(windows) - 1

    chunks: list[TextChunk] = []
    start = 0
    for i, (_, nominal_end) in enumerate(windows):
        end = char_at(nominal_end)
        if i < last:
            overlap_start = char_at(windows[i + 1][0])
            inside = [b for b in breaks if overlap_start <= b < end and b > start]
            if inside:
                end = inside[-1]

        chunk_text = text[start:end]
        page_start, page_end = _pages_between(section, start, end)
        chunks.append(TextChunk(chunk_text, len(encoding.encode(chunk_text)), page_start, page_end))

        if i < last:
            next_start = char_at(windows[i + 1][0])
            next_end = char_at(windows[i + 1][1])
            paragraph_starts = [
                b + len(PARAGRAPH_BREAK)
                for b in breaks
                if start < b + len(PARAGRAPH_BREAK) <= next_start
            ]
            in_overlap = [b + len(PARAGRAPH_BREAK) for b in breaks if next_start <= b <= end]
            if paragraph_starts and fits(paragraph_starts[-1], next_end):
                next_start = paragraph_starts[-1]
            elif in_overlap:
                next_start = in_overlap[0]
            start = next_start
    return chunks


def chunk_blocks(
    blocks: Iterable[Block],
    chunk_size: int,
    chunk_overlap: int,
    min_tokens: int,
    skip_patterns: Sequence[str] = (),
    box_pattern: str | None = None,
) -> list[TextChunk]:
    """Split a document into chunks of at most chunk_size tokens (see module docstring)."""
    cleaned = drop_skipped(blocks, [re.compile(p) for p in skip_patterns])
    if box_pattern:
        cleaned = move_boxes(cleaned, re.compile(box_pattern))
    sections = merge_small_sections(split_sections(cleaned), min_tokens)
    log.debug("Chunking {} sections, size {}, overlap {}", len(sections), chunk_size, chunk_overlap)
    return [
        chunk for section in sections for chunk in split_section(section, chunk_size, chunk_overlap)
    ]
