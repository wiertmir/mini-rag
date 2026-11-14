"""Show all chunks of a document, highlighting text shared by neighbouring chunks."""

import argparse
import asyncio
from collections.abc import Sequence
from itertools import pairwise
from typing import NamedTuple

import rich_struct.log
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mini_rag.config import DisplayConfig, HelpFormatter, load_config
from mini_rag.errors import DocumentNotFoundError
from mini_rag.helpers import ErrorHandler
from mini_rag.models import Chunk
from mini_rag.pages import PageRange
from mini_rag.storage import load_embedding_manifest_async, read_chunks_async

logger = rich_struct.log.get_logger()

OVERLAP_STYLE = "yellow"
MIN_OVERLAP_CHARS = 20
"""Shorter suffix/prefix matches (a shared "." or a space) are coincidences, not overlap."""


class ChunkRow(NamedTuple):
    label: str
    text: str
    is_overlap: bool


def overlap_length(first: str, second: str, min_length: int = MIN_OVERLAP_CHARS) -> int:
    """Length of the longest end of first that is also the start of second.

    Matches shorter than min_length count as no overlap.
    """
    for length in range(min(len(first), len(second)), min_length - 1, -1):
        if length and first.endswith(second[:length]):
            return length
    return 0


def _short_id(chunk: Chunk) -> str:
    return chunk.chunk_id.rsplit("_", 1)[-1]


def _label(chunk: Chunk) -> str:
    return f"{chunk.chunk_id}\np. {chunk.pages}"


def split_chunks(chunks: Sequence[Chunk]) -> list[list[ChunkRow]]:
    """Split consecutive chunks into unique and shared (overlap) rows.

    Chunks that share no text start a new group: with section chunking a group is one
    section split into overlapping windows.
    """
    overlaps = [overlap_length(a.text, b.text) for a, b in pairwise(chunks)]
    groups: list[list[ChunkRow]] = []
    rows: list[ChunkRow] = []
    for i, chunk in enumerate(chunks):
        text = chunk.text
        start = overlaps[i - 1] if i > 0 else 0
        end = len(text) - overlaps[i] if i < len(overlaps) else len(text)
        rows.append(ChunkRow(_label(chunk), text[start : max(start, end)], is_overlap=False))
        if i < len(overlaps) and overlaps[i]:
            label = f"{_short_id(chunk)} ↔ {_short_id(chunks[i + 1])}"
            rows.append(ChunkRow(label, text[len(text) - overlaps[i] :], is_overlap=True))
        elif rows:
            groups.append(rows)
            rows = []
    return groups


def build_table(chunks: Sequence[Chunk], title: str, pages: str | None = None) -> Table:
    selection = f" on pages {pages}" if pages else ""
    table = Table(
        title=title,
        caption=f"{len(chunks)} chunks{selection}; overlapping text in {OVERLAP_STYLE}",
        header_style="bold cyan",
        expand=not chunks,
    )
    table.add_column("Chunk ID", no_wrap=True)
    table.add_column("Text", overflow="fold")

    for group_no, rows in enumerate(split_chunks(chunks)):
        if group_no:
            table.add_section()
        for row in rows:
            if row.is_overlap:
                table.add_row(row.label, Text(row.text), style=OVERLAP_STYLE)
            elif row.text:
                table.add_row(row.label, Text(row.text))
            else:
                table.add_row(row.label, Text("(entirely overlapped)", style="dim italic"))
    return table


def select_pages(chunks: Sequence[Chunk], page_ranges: Sequence[PageRange]) -> list[Chunk]:
    """Keep chunks that share at least one page with the selection."""
    return [c for c in chunks if any(r.overlaps(c.page, c.page_end) for r in page_ranges)]


async def load_document_chunks_async(config: DisplayConfig) -> list[Chunk]:
    chunks_path = config.chunks
    if chunks_path is None:
        chunks_path = (await load_embedding_manifest_async(config.index_dir)).chunks_path

    chunks = [
        chunk
        async for chunk in read_chunks_async(chunks_path)
        if chunk.document_id == config.document_id
    ]
    if not chunks:
        raise DocumentNotFoundError(
            f"No chunks for document {config.document_id!r} in {chunks_path}"
        )
    return chunks


def main(argv: Sequence[str] | None = None) -> None:
    with ErrorHandler(OSError, ValueError):
        parser = argparse.ArgumentParser(
            prog="python -m mini_rag.display",
            description=__doc__,
            formatter_class=HelpFormatter,
            allow_abbrev=False,
        )
        config = load_config(DisplayConfig, argv, parser)
        rich_struct.log.configure(console=False, file=config.log_folder)

        chunks = asyncio.run(load_document_chunks_async(config))
        title = f"{chunks[0].document_id}: {chunks[0].title}"
        if config.page_ranges is not None:
            chunks = select_pages(chunks, config.page_ranges)
        Console().print(build_table(chunks, title, config.pages))


if __name__ == "__main__":
    main()
