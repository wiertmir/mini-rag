import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import openai
import rich_struct.log
from asyncstdlib import batched

from mini_rag.errors import InconsistentManifestError
from mini_rag.models import Chunk, EmbeddingManifest

rich_struct.log.configure()

logger = rich_struct.log.get_logger()


def normalize(embeddings_iter: AsyncIterator[list[float]]) -> NDArray[np.float32]:
    raw_embeddings = asyncio.run(collect(embeddings_iter))

    if raw_embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {raw_embeddings.shape}")

    logger.info("Normalizing {} rows", raw_embeddings.shape)
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

    # NaN/inf rows are as bad as zero rows: they also yield NaN scores.
    # I decided to ignore cases close to zero (around epsilon)
    if np.any(~np.isfinite(norms) | (norms == 0.0)):
        raise ValueError("Data can't be normalized since it contains NaN or Zero")

    return raw_embeddings / norms

def save_index(output_dir: Path, matrix: NDArray[np.float32], manifest: EmbeddingManifest) -> None:
    if manifest.records != matrix.shape[0]:
        raise InconsistentManifestError(f"Inconsistent number of rows in manifest ({manifest.records}) and matrix ({matrix.shape[0]})")
    if manifest.dimension != matrix.shape[1]:
        raise InconsistentManifestError(f"Inconsistent number of columns (dimension) in manifest ({manifest.dimension}) and matrix ({matrix.shape[1]})")

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    logger.info("Saving embeddings matrix to {}. Size={}", embeddings_path, matrix.shape)
    np.save(embeddings_path, matrix)
    manifest_path = output_dir / "embedding_manifest.json"
    logger.info("Saving embeddings manifest to {}", manifest_path)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    

async def collect(embeddings_iter: AsyncIterator[list[float]]) -> NDArray[np.float32]:
    embeddings = [np.asarray(row, dtype=np.float32) async for row in embeddings_iter]
    return np.stack(embeddings)


async def embed_chunks(
    chunks: AsyncIterator[Chunk],
    client: openai.AsyncOpenAI,
    model: str,
    batch_size: int,
) -> AsyncIterator[list[float]]:
    return embed_texts(
        (chunk.text async for chunk in chunks), client, model, batch_size
    )


async def embed_texts(
    texts: AsyncIterator[str], client: openai.AsyncOpenAI, model: str, batch_size: int
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
