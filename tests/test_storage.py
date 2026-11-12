from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_rag.errors import ChunkParseError, ChunkReadError, ChunkValidationError
from mini_rag.models import Chunk
from mini_rag.storage import read_chunks_async


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        chunk_id="sample_p001_c001",
        document_id="sample",
        title="Sample",
        collection="test",
        source_path=Path("sample.pdf"),
        source_url="https://example.test/sample.pdf",
        page=1,
        text="Sample text",
        token_count=2,
    )


async def _collect(path: Path) -> list[Chunk]:
    return [chunk async for chunk in read_chunks_async(path)]


async def test_read_valid_chunks(tmp_path: Path, chunk: Chunk) -> None:
    second_chunk = Chunk(
        chunk_id="sample_p001_c002",
        document_id="sample",
        title="Sample",
        collection="test",
        source_path=Path("sample.pdf"),
        source_url="https://example.test/sample.pdf",
        page=1,
        text="Second chunk",
        token_count=2,
    )
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        chunk.model_dump_json() + "\n" + second_chunk.model_dump_json(),
        encoding="utf-8",
    )
    result = await _collect(path)
    assert [record.chunk_id for record in result] == [
        chunk.chunk_id,
        second_chunk.chunk_id,
    ]
    assert result == [chunk, second_chunk]


async def test_second_line_missing_text_raises_validation_error(
    tmp_path: Path, chunk: Chunk
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        chunk.model_dump_json() + "\n" + chunk.model_dump_json(exclude={"text"}),
        encoding="utf-8",
    )
    with pytest.raises(ChunkValidationError, match="line 2"):
        await _collect(path)


@pytest.mark.parametrize(
    "invalid_line,error_type",
    [
        ("{", ChunkParseError),
        ("\n", ChunkParseError),
        ("   \n", ChunkParseError),
        ("{}", ChunkValidationError),
        ("[]", ChunkValidationError),
        ('{"page": "not an integer"}', ChunkValidationError),
    ],
)
async def test_read_errors_include_path_line_number_and_cause(
    tmp_path: Path,
    chunk: Chunk,
    invalid_line: str,
    error_type: type[ChunkReadError],
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(chunk.model_dump_json() + "\n" + invalid_line, encoding="utf-8")
    with pytest.raises(error_type) as caught:
        await _collect(path)

    assert caught.value.path == path
    assert caught.value.line_no == 2
    assert f"{path}: line 2:" in str(caught.value)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert str(caught.value.__cause__) in str(caught.value)


async def test_empty_file_returns_no_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("", encoding="utf-8")
    result = await _collect(path)
    assert result == []


async def test_missing_file_preserves_filesystem_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        await _collect(path)
