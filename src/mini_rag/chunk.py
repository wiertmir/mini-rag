"""Section-aware chunking: sections start at headings, may span pages, small ones merge."""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import rich_struct.log
import tiktoken

log = rich_struct.log.get_logger()

CHUNK_MODEL = "text-embedding-3-small"

encoding = tiktoken.encoding_for_model(CHUNK_MODEL)

PARAGRAPH_BREAK = "\n\n"
LINE_BREAK = "\n"


class Block(NamedTuple):
    """One text region of the document, in reading order.

    joined: the block continues the previous one (column or page continuation),
    so it is separated by a line break instead of a paragraph break.
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


def build_section(blocks: Sequence[Block]) -> Section:
    parts: list[str] = []
    spans: list[_Span] = []
    offset = 0
    for i, block in enumerate(blocks):
        if i:
            separator = LINE_BREAK if block.joined else PARAGRAPH_BREAK
            parts.append(separator)
            offset += len(separator)
        parts.append(block.text)
        spans.append(_Span(offset, offset + len(block.text), block.page))
        offset += len(block.text)
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


def _pages_between(section: Section, start: int, end: int) -> tuple[int, int]:
    pages = [span.page for span in section.spans if span.start < end and span.end > start]
    if not pages:  # a window of separators only; cannot happen with real text
        pages = [section.page_start]
    return min(pages), max(pages)


def split_section(section: Section, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Cut a section into evenly sized token windows sharing exactly chunk_overlap tokens.

    The window count is the minimum needed for windows of at most chunk_size; the windows are
    then balanced, so the last one is never a near-duplicate tail of the previous one.
    """
    tokens = section.tokens
    if len(tokens) <= chunk_size:
        return [TextChunk(section.text, len(tokens), section.page_start, section.page_end)]

    _, offsets = encoding.decode_with_offsets(list(tokens))
    max_step = chunk_size - chunk_overlap
    windows = -(-(len(tokens) - chunk_size) // max_step) + 1
    step = -(-(len(tokens) - chunk_overlap) // windows)
    chunks: list[TextChunk] = []
    start = 0
    while True:
        end = min(start + step + chunk_overlap, len(tokens))
        text = encoding.decode(list(tokens[start:end]))
        char_start = offsets[start]
        char_end = offsets[end] if end < len(tokens) else len(section.text)
        page_start, page_end = _pages_between(section, char_start, char_end)
        chunks.append(TextChunk(text, len(encoding.encode(text)), page_start, page_end))
        if end == len(tokens):
            return chunks
        start += step


def chunk_blocks(
    blocks: Iterable[Block], chunk_size: int, chunk_overlap: int, min_tokens: int
) -> list[TextChunk]:
    """Split a document into chunks of at most chunk_size tokens (see module docstring)."""
    sections = merge_small_sections(split_sections(blocks), min_tokens)
    log.debug("Chunking {} sections, size {}, overlap {}", len(sections), chunk_size, chunk_overlap)
    return [
        chunk for section in sections for chunk in split_section(section, chunk_size, chunk_overlap)
    ]
