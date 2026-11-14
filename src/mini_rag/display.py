"""Show all chunks of a document, highlighting text shared by neighbouring chunks."""

import argparse
import asyncio
from collections.abc import Sequence
from itertools import groupby, pairwise
from typing import NamedTuple

import rich_struct.log
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mini_rag.config import DisplayConfig, HelpFormatter, load_config
from mini_rag.errors import DocumentNotFoundError
from mini_rag.helpers import ErrorHandler
from mini_rag.models import Chunk, EmbeddingManifest
from mini_rag.storage import read_chunks_async

logger = rich_struct.log.get_logger()

OVERLAP_STYLE = "yellow"


class ChunkRow(NamedTuple):
    label: str
    text: str
    is_overlap: bool


def overlap_length(first: str, second: str) -> int:
    """Length of the longest end of ``first`` that is also the start of ``second``."""
    for length in range(min(len(first), len(second)), 0, -1):
        if first.endswith(second[:length]):
            return length
    return 0


def _short_id(chunk: Chunk) -> str:
    return chunk.chunk_id.rsplit("_", 1)[-1]


def split_page_chunks(chunks: Sequence[Chunk]) -> list[ChunkRow]:
    """Split consecutive chunks of one page into unique and shared (overlap) rows."""
    overlaps = [overlap_length(a.text, b.text) for a, b in pairwise(chunks)]
    rows: list[ChunkRow] = []
    for i, chunk in enumerate(chunks):
        text = chunk.text
        start = overlaps[i - 1] if i > 0 else 0
        end = len(text) - overlaps[i] if i < len(overlaps) else len(text)
        rows.append(ChunkRow(chunk.chunk_id, text[start : max(start, end)], is_overlap=False))
        if i < len(overlaps) and overlaps[i]:
            label = f"{_short_id(chunk)} ↔ {_short_id(chunks[i + 1])}"
            rows.append(ChunkRow(label, text[len(text) - overlaps[i] :], is_overlap=True))
    return rows


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

    for page_no, (_, page_chunks) in enumerate(groupby(chunks, key=lambda c: c.page)):
        if page_no:
            table.add_section()
        for row in split_page_chunks(list(page_chunks)):
            if row.is_overlap:
                table.add_row(row.label, Text(row.text), style=OVERLAP_STYLE)
            elif row.text:
                table.add_row(row.label, Text(row.text))
            else:
                table.add_row(row.label, Text("(entirely overlapped)", style="dim italic"))
    return table


async def load_document_chunks_async(config: DisplayConfig) -> list[Chunk]:
    chunks_path = config.chunks
    if chunks_path is None:
        manifest_path = config.index_dir / "embedding_manifest.json"
        logger.info("Loading embeddings manifest from {}", manifest_path)
        manifest = EmbeddingManifest.model_validate_json(manifest_path.read_text("utf-8"))
        chunks_path = manifest.chunks_path

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
        if (page_ranges := config.page_ranges) is not None:
            chunks = [c for c in chunks if any(r.includes(c.page) for r in page_ranges)]
        Console().print(build_table(chunks, title, config.pages))


if __name__ == "__main__":
    main()
