"""Show the top-k chunks from the embedding index most similar to a question."""

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence

import numpy as np
import rich_struct.log
from numpy.typing import NDArray
from openai import AsyncOpenAI, OpenAIError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mini_rag.config import HelpFormatter, RetrieveConfig, load_config
from mini_rag.embed import embed_texts_async, normalize
from mini_rag.errors import (
    DimensionMismatchError,
    EmptyIndex,
    EmptyInputData,
    IndexAndChunksNotSyncError,
)
from mini_rag.models import Chunk, SearchResult
from mini_rag.storage import load_index_async

logger = rich_struct.log.get_logger()


def search(
    query_vector: Sequence[float] | NDArray[np.float32],
    matrix: NDArray[np.float32],
    chunks: Sequence[Chunk],
    top_k: int,
) -> list[SearchResult]:
    """Rank chunks by cosine similarity, breaking score ties by chunk ID."""
    query = np.asarray(query_vector, dtype=np.float32)
    n = matrix.shape[0]

    if n == 0:
        raise EmptyIndex("index has 0 rows")

    if query.ndim != 1:
        raise ValueError(f"Expected a 1-D query vector, got shape {query.shape}")

    d = query.shape[0]
    if d != matrix.shape[1]:
        raise DimensionMismatchError(
            f"Question vector's dimension is {d}, where the index dimension is {matrix.shape[1]}"
        )

    if top_k < 1:
        raise ValueError("`top_k` must be at least 1")

    if matrix.shape[0] != len(chunks):
        raise IndexAndChunksNotSyncError(
            f"Index has {matrix.shape[0]} rows, where chunks have {len(chunks)} rows"
        )

    norm_query_vector = normalize(query.reshape(1, -1))[0]
    scores = matrix @ norm_query_vector
    # lexsort uses the last key as primary: descending score, then ascending ID.
    # full sort as data is not that big and `argpartition` have some unnecessary challanges
    sort_indices = np.lexsort(([chunk.chunk_id for chunk in chunks], -scores))

    return [
        SearchResult(rank=rank, score=float(scores[i]), chunk=chunks[i])
        for (rank, i) in enumerate(sort_indices[:top_k], start=1)
    ]


async def _yield_one(text: str) -> AsyncIterator[str]:
    yield text


async def _read_one_async(iter: AsyncIterator[list[float]]) -> list[float]:
    first_result: list[float] | None = None
    async for lst in iter:
        first_result = lst
        break

    if first_result is not None:
        return first_result
    raise EmptyInputData("`iter` has no elements")


async def retrieve_async(
    config: RetrieveConfig, client: AsyncOpenAI
) -> list[SearchResult]:
    index, chunks, manifest = await load_index_async(config.index_dir, config.chunks)
    logger.debug("Searching for text {}", config.question)
    questions = _yield_one(config.question)
    model = manifest.model
    question_vector = await _read_one_async(
        embed_texts_async(questions, client, model, 1)
    )
    logger.debug("Semantic searching")
    return search(question_vector, index, chunks, config.top_k)


def preview(text: str, limit: int = 200) -> str:
    return text[:limit]


async def _run_async(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.retrieve",
        description=__doc__,
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )

    config = load_config(RetrieveConfig, argv, parser)
    rich_struct.log.configure(console=False, file=config.log_folder)

    async with AsyncOpenAI() as client:
        result = await retrieve_async(config, client)

    table = Table(header_style="bold cyan")
    for column in ("Rank", "Score", "ChunkId", "DocumentId", "Page", "Text"):
        table.add_column(column)
    for hit in result:
        table.add_row(
            str(hit.rank),
            f"{hit.score:.4f}",
            Text(hit.chunk.chunk_id),
            Text(hit.chunk.document_id),
            str(hit.chunk.page),
            Text(preview(hit.chunk.text)),
        )
    Console().print(table)


def main(argv: Sequence[str] | None = None) -> None:
    try:
        asyncio.run(_run_async(argv))
    except (OpenAIError, OSError, ValueError, TypeError) as exc:
        Console(stderr=True).print(
            f"Error: {exc}", style="red", markup=False, highlight=False
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
