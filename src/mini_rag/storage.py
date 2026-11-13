import hashlib
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import NamedTuple

import aiofiles
import numpy as np
import rich_struct.log
from numpy.typing import NDArray
from pydantic import ValidationError

from mini_rag.errors import (
    ChunkParseError,
    ChunkValidationError,
    DimensionMismatchError,
    InconsistentManifestError,
    IndexAndChunksNotSyncError,
    StaleIndexError,
)
from mini_rag.models import Chunk, EmbeddingManifest

logger = rich_struct.log.get_logger()


class LoadedIndex(NamedTuple):
    embeddings_matrix: NDArray[np.float32]
    chunks: Sequence[Chunk]
    manifest: EmbeddingManifest


async def load_index_async(index_dir: Path, chunks_path: Path | None) -> LoadedIndex:

    manifest_path = index_dir / "embedding_manifest.json"
    logger.info("Loading embeddings manifest from {}", manifest_path)
    async with aiofiles.open(manifest_path, mode="r", encoding="utf-8") as file:
        json_string = await file.read()
    manifest = EmbeddingManifest.model_validate_json(json_string)

    chunks_path = chunks_path or manifest.chunks_path
    chunks = await read_chunk_list_async(chunks_path)
    chunks_sha = calculate_file_sha(chunks_path)

    if chunks_sha != manifest.chunks_sha256:
        raise StaleIndexError("Stale index, re-run embed")

    embeddings_path = index_dir / "embeddings.npy"
    logger.info("Loading normalized embeddings from {}", embeddings_path)
    embeddings: NDArray[np.float32] = np.load(embeddings_path)

    if embeddings.shape[0] != len(chunks):
        raise IndexAndChunksNotSyncError(
            f"Index has {embeddings.shape[0]} rows, but chunks have {len(chunks)} records"
        )

    if embeddings.shape != (manifest.records, manifest.dimension):
        raise DimensionMismatchError(
            f"Embeddings shape: {embeddings.shape}, where manifest specifies {(manifest.records, manifest.dimension)}"
        )

    return LoadedIndex(embeddings, chunks, manifest)


def save_index(
    output_dir: Path, matrix: NDArray[np.float32], manifest: EmbeddingManifest
) -> None:
    """Remove the old manifest before rewriting the index; write the new one last."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected a NumPy array, got {type(matrix).__name__}")

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
    manifest_path = output_dir / "embedding_manifest.json"
    manifest_path.unlink(missing_ok=True)
    logger.info(
        "Saving embeddings matrix to {}. Size={}", embeddings_path, matrix.shape
    )
    np.save(embeddings_path, matrix)
    logger.info("Saving embeddings manifest to {}", manifest_path)
    manifest_path.write_text(manifest.model_dump_json(indent=2))


async def read_chunks_async(path: Path) -> AsyncIterator[Chunk]:
    """Read chunks, raising a ChunkReadError with the failing line number."""
    logger.info("Reading chunks from {}", path)
    async with aiofiles.open(path, encoding="utf-8") as file:
        line_no = 0
        async for line in file:
            line_no += 1
            if line:
                try:
                    chunk = Chunk.model_validate_json(line)
                except ValidationError as exc:
                    if any(error["type"] == "json_invalid" for error in exc.errors()):
                        raise ChunkParseError(
                            path, line_no, f"Invalid JSON: {exc}"
                        ) from exc
                    raise ChunkValidationError(
                        path, line_no, f"Invalid chunk: {exc}"
                    ) from exc
                yield chunk


async def read_chunk_list_async(path: Path) -> list[Chunk]:
    return await _collect_chunks_async(path)


def calculate_file_sha(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


async def _collect_chunks_async(path: Path) -> list[Chunk]:
    return [chunk async for chunk in read_chunks_async(path)]
