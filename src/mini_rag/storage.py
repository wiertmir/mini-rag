import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
import rich_struct.log
from pydantic import ValidationError

from mini_rag.errors import ChunkParseError, ChunkValidationError
from mini_rag.models import Chunk

rich_struct.log.configure(file=True)
logger = rich_struct.log.get_logger()


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


def read_chunks(path: Path) -> list[Chunk]:
    """Read all chunks, propagating line-numbered parsing and validation errors."""
    return asyncio.run(_collect_chunks(path))


def calculate_file_sha(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


async def _collect_chunks(path: Path) -> list[Chunk]:
    return [chunk async for chunk in read_chunks_async(path)]
