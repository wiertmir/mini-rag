"""
Embed chunks from a JSONL file and save a normalized vector index.

Writes embeddings.npy (float32, one L2-normalized row per chunk, in chunk-file
order) and embedding_manifest.json (model, dimension, record count, and the
SHA-256 of the chunk file, used to detect a stale index).
"""

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import rich_struct.log
from asyncstdlib import batched
from numpy.typing import NDArray
from openai import AsyncOpenAI

from mini_rag.config import EmbedConfig, HelpFormatter, load_config
from mini_rag.errors import InconsistentManifestError
from mini_rag.models import Chunk, EmbeddingManifest
from mini_rag.storage import calculate_file_sha, read_chunks_async

rich_struct.log.configure()

logger = rich_struct.log.get_logger()


async def normalize_async(
    embeddings_iter: AsyncIterator[list[float]],
) -> NDArray[np.float32]:
    raw_embeddings = await _collect_async(embeddings_iter)

    return normalize(raw_embeddings)


def normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize rows, reporting invalid norms with one-based row numbers."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {matrix.shape}")

    if matrix.shape[0] == 0:
        raise ValueError("Expecting matrix with rows")

    if matrix.dtype != np.dtype(np.float32):
        raise TypeError(f"Expected float32 matrix, got {matrix.dtype}")

    logger.info("Normalizing {} rows", matrix.shape[0])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # NaN/inf rows are as bad as zero rows: they also yield NaN scores.
    # I decided to ignore cases close to zero (around epsilon)
    invalid_rows = np.flatnonzero(~np.isfinite(norms) | (norms == 0.0))
    if invalid_rows.size:
        raise ValueError(
            "Data can't be normalized since it contains NaN, infinity or Zero "
            f"norms at rows (1-based): {(invalid_rows + 1).tolist()}"
        )

    return matrix / norms


def save_index(
    output_dir: Path, matrix: NDArray[np.float32], manifest: EmbeddingManifest
) -> None:
    """Save a float32 index; reject other dtypes before writing any files."""
    if matrix.dtype != np.dtype(np.float32):
        raise TypeError(f"Expected float32 matrix, got {matrix.dtype}")

    if manifest.records != matrix.shape[0]:
        raise InconsistentManifestError(
            f"Inconsistent number of rows in manifest ({manifest.records}) and matrix ({matrix.shape[0]})"
        )
    if manifest.dimension != matrix.shape[1]:
        raise InconsistentManifestError(
            f"Inconsistent number of columns (dimension) in manifest ({manifest.dimension}) and matrix ({matrix.shape[1]})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    logger.info(
        "Saving embeddings matrix to {}. Size={}", embeddings_path, matrix.shape
    )
    np.save(embeddings_path, matrix)
    manifest_path = output_dir / "embedding_manifest.json"
    logger.info("Saving embeddings manifest to {}", manifest_path)
    manifest_path.write_text(manifest.model_dump_json(indent=2))


async def _collect_async(
    embeddings_iter: AsyncIterator[list[float]],
) -> NDArray[np.float32]:
    embeddings = [np.asarray(row, dtype=np.float32) async for row in embeddings_iter]

    if not len(embeddings):
        raise ValueError("Expecting at leat one embedding")
    return np.stack(embeddings)


def embed_chunks(
    chunks: AsyncIterator[Chunk],
    client: AsyncOpenAI,
    model: str,
    batch_size: int,
) -> AsyncIterator[list[float]]:
    return embed_texts_async(
        (chunk.text async for chunk in chunks), client, model, batch_size
    )


async def embed_texts_async(
    texts: AsyncIterator[str], client: AsyncOpenAI, model: str, batch_size: int
) -> AsyncIterator[list[float]]:
    batch_no = 0
    total_texts = 0
    total_tokens = 0
    start_time = perf_counter()
    async for batch in batched(texts, batch_size):
        batch_no += 1
        batch_seq = batch_no
        total_texts += len(batch)
        logger.info(
            "Calling OpenAI embeddings (model={}) with {} strings [Batch: {}]",
            model,
            len(batch),
            batch_seq,
        )
        response = await client.embeddings.create(model=model, input=batch)

        total_tokens += response.usage.total_tokens
        logger.debug(
            "OpenAI returned {} embeddings (total_tokens: {}) [Batch: {}]",
            len(response.data),
            response.usage.total_tokens,
            batch_seq,
        )
        for item in sorted(response.data, key=lambda x: x.index):
            yield item.embedding

    logger.info(
        "Embedded {} texts in {:.2f} seconds (total_tokens: {})",
        total_texts,
        perf_counter() - start_time,
        total_tokens,
    )


async def run_embed(config: EmbedConfig, client: AsyncOpenAI) -> EmbeddingManifest:
    logger.info("Reading chunks from file {}", config.chunks)

    chunks_sha = calculate_file_sha(config.chunks)
    chunks = read_chunks_async(config.chunks)
    chunk_count = 0

    async def count_chunks() -> AsyncIterator[Chunk]:
        nonlocal chunk_count
        async for chunk in chunks:
            chunk_count += 1
            yield chunk

    raw_embeddings = embed_chunks(
        count_chunks(), client, config.embedding_model, config.batch_size
    )
    embedding_count = 0

    async def validate_embedding_count() -> AsyncIterator[list[float]]:
        nonlocal embedding_count
        async for embedding in raw_embeddings:
            embedding_count += 1
            yield embedding
        if embedding_count != chunk_count:
            raise ValueError(
                f"Expected {chunk_count} embeddings for chunks, got {embedding_count}"
            )

    embeddings = await normalize_async(validate_embedding_count())

    manifest = EmbeddingManifest(
        model=config.embedding_model,
        dimension=embeddings.shape[1],
        records=embeddings.shape[0],
        chunks_path=config.chunks,
        chunks_sha256=chunks_sha,
    )

    save_index(config.output_dir, embeddings, manifest)

    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.embed",
        description=__doc__,
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )
    config = load_config(EmbedConfig, argv, parser)

    logger.info("Starting the embed app")

    client = AsyncOpenAI()
    asyncio.run(run_embed(config, client))


if __name__ == "__main__":
    main()
