from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from openai import AsyncOpenAI

from mini_rag.config import EmbedConfig
from mini_rag.embed import _collect_async as _collect
from mini_rag.embed import embed_texts_async as embed_texts
from mini_rag.embed import normalize, normalize_async, run_embed, save_index
from mini_rag.models import Chunk, EmbeddingManifest
from mini_rag.storage import calculate_file_sha


@pytest.fixture
def embedding_manifest() -> EmbeddingManifest:
    return EmbeddingManifest(
        model="test-model",
        dimension=2,
        records=2,
        chunks_path=Path("chunks.jsonl"),
        chunks_sha256="a" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def iter_embeddings(rows: list[list[float]]) -> AsyncIterator[list[float]]:
    for row in rows:
        yield row


async def test_collect_preserves_rows_as_float32_matrix() -> None:
    rows = [[0.1, 0.2, 0.3], [4.0, 5.0, 6.0]]

    result = await _collect(iter_embeddings(rows))

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array(rows, dtype=np.float32))


async def test_collect_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="one embedding"):
        await _collect(iter_embeddings([]))


async def test_normalize_async_collects_and_normalizes_rows() -> None:
    result = await normalize_async(iter_embeddings([[3.0, 4.0], [0.0, 5.0]]))

    assert result.dtype == np.float32
    assert np.allclose(result, [[0.6, 0.8], [0.0, 1.0]])


def test_normalize_rejects_non_float32_matrix() -> None:
    with pytest.raises(TypeError, match="Expected float32 matrix, got float64"):
        normalize(np.array([[3.0, 4.0]], dtype=np.float64))


def test_normalize_rejects_plain_lists() -> None:
    with pytest.raises(TypeError, match="Expected a NumPy array, got list"):
        normalize([[3.0, 4.0]])  # type: ignore[arg-type]


def test_normalize_produces_unit_row_norms() -> None:
    result = normalize(np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32))

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalize_preserves_direction() -> None:
    result = normalize(np.array([[3.0, 4.0], [-4.0, 3.0]], dtype=np.float32))

    assert np.allclose(result, [[0.6, 0.8], [-0.8, 0.6]])


def test_normalize_rejects_zero_row() -> None:
    with pytest.raises(ValueError, match=r"Zero norms at rows \(1-based\): \[2\]"):
        normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0])
def test_normalize_reports_all_invalid_row_numbers(value: float) -> None:
    matrix = np.array(
        [[value, 0.0], [3.0, 4.0], [value, 0.0], [5.0, 12.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match=r"rows \(1-based\): \[1, 3\]"):
        normalize(matrix)


def test_normalize_reports_mixed_invalid_row_numbers() -> None:
    matrix = np.array(
        [[3.0, 4.0], [np.nan, 1.0], [0.0, 0.0], [1.0, np.inf]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match=r"rows \(1-based\): \[2, 3, 4\]"):
        normalize(matrix)


def test_save_index_matrix_round_trip(
    tmp_path: Path, embedding_manifest: EmbeddingManifest
) -> None:
    matrix = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=np.float32)

    save_index(tmp_path, matrix, embedding_manifest)
    loaded = np.load(tmp_path / "embeddings.npy", allow_pickle=False)

    assert loaded.shape == matrix.shape
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, matrix)


def test_save_index_manifest_round_trip(
    tmp_path: Path, embedding_manifest: EmbeddingManifest
) -> None:
    matrix = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=np.float32)

    save_index(tmp_path, matrix, embedding_manifest)
    file_text = (tmp_path / "embedding_manifest.json").read_text(encoding="utf-8")

    assert EmbeddingManifest.model_validate_json(file_text) == embedding_manifest


def test_save_index_removes_old_manifest_before_writing_embeddings(
    tmp_path: Path,
    embedding_manifest: EmbeddingManifest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=np.float32)
    manifest_path = tmp_path / "embedding_manifest.json"
    manifest_path.write_text("old manifest", encoding="utf-8")

    def fail_save(*args: object, **kwargs: object) -> None:
        assert not manifest_path.exists()
        raise OSError("Embedding write failed")

    monkeypatch.setattr(np, "save", fail_save)

    with pytest.raises(OSError, match="Embedding write failed"):
        save_index(tmp_path, matrix, embedding_manifest)

    assert not manifest_path.exists()


@pytest.mark.parametrize("dtype", ["float64", "float16", "int32", "complex64"])
def test_save_index_rejects_non_float32_before_creating_output(
    tmp_path: Path, embedding_manifest: EmbeddingManifest, dtype: str
) -> None:
    matrix = np.array([[1, 0], [0, 1]], dtype=dtype)
    output_dir = tmp_path / "index"

    with pytest.raises(TypeError, match=f"Expected float32 matrix, got {dtype}"):
        save_index(output_dir, matrix, embedding_manifest)

    assert not output_dir.exists()


def test_save_index_rejects_float64_without_overwriting_files(
    tmp_path: Path, embedding_manifest: EmbeddingManifest
) -> None:
    matrix = np.array([[0.6, 0.8], [-0.8, 0.6]], dtype=np.float32)
    save_index(tmp_path, matrix, embedding_manifest)
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "embedding_manifest.json"
    original_embeddings = embeddings_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(TypeError, match="Expected float32 matrix, got float64"):
        # Deliberately violate the dtype contract to test runtime validation.
        save_index(tmp_path, matrix.astype(np.float64), embedding_manifest)  # pyright: ignore[reportArgumentType]

    assert embeddings_path.read_bytes() == original_embeddings
    assert manifest_path.read_bytes() == original_manifest


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = self
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def create(self, *, model: str, input: tuple[str, ...]) -> SimpleNamespace:
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(text)), 0.0])
                for index, text in reversed(list(enumerate(input)))
            ],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


async def iter_texts(texts: list[str]) -> AsyncIterator[str]:
    for text in texts:
        yield text


async def test_embed_texts_batches_with_partial_final_batch() -> None:
    client = FakeClient()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts(texts), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert [len(batch) for _, batch in client.calls] == [2, 2, 1]
    assert client.calls == [
        ("test-model", ("a", "bb")),
        ("test-model", ("ccc", "dddd")),
        ("test-model", ("eeeee",)),
    ]
    assert len(vectors) == 5


async def test_embed_texts_preserves_text_order() -> None:
    client = FakeClient()
    texts = ["four", "a", "three", "bb", "sixsix"]

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts(texts), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert vectors == [[float(len(text)), 0.0] for text in texts]


async def test_embed_texts_empty_iterator() -> None:
    client = FakeClient()

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts([]), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert vectors == []
    assert client.calls == []


@pytest.fixture
def run_embed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbedConfig:
    (tmp_path / "logs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_RAG_CONFIG", raising=False)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        Chunk(
            chunk_id=f"sample_p001_c{index:03d}",
            document_id="sample",
            title="Sample",
            collection="test",
            source_path=tmp_path / "sample.pdf",
            source_url="https://example.test/sample.pdf",
            page=1,
            text=text,
            token_count=1,
        )
        for index, text in enumerate(["a", "bb", "ccc"], start=1)
    ]
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    return EmbedConfig(
        chunks=chunks_path,
        output_dir=tmp_path / "out",
        batch_size=2,
        embedding_model="test-model",
    )


async def test_run_embed_writes_index_and_manifest(
    run_embed_config: EmbedConfig,
) -> None:
    config = run_embed_config
    client = FakeClient()

    returned_manifest = await run_embed(config, cast(AsyncOpenAI, client))

    matrix = np.load(config.output_dir / "embeddings.npy", allow_pickle=False)
    manifest = EmbeddingManifest.model_validate_json(
        (config.output_dir / "embedding_manifest.json").read_text(encoding="utf-8")
    )

    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert manifest.records == 3
    assert manifest.dimension == 2
    assert manifest.model == config.embedding_model
    assert manifest.chunks_path == config.chunks
    assert manifest.chunks_sha256 == calculate_file_sha(config.chunks)
    assert returned_manifest == manifest
    assert client.calls == [
        ("test-model", ("a", "bb")),
        ("test-model", ("ccc",)),
    ]


async def test_run_embed_rejects_mismatched_embedding_count_before_saving(
    run_embed_config: EmbedConfig,
) -> None:
    class FewerVectorsClient(FakeClient):
        async def create(
            self, *, model: str, input: tuple[str, ...]
        ) -> SimpleNamespace:
            response = await super().create(model=model, input=input)
            response.data = response.data[:-1]
            return response

    config = run_embed_config
    client = FewerVectorsClient()

    with pytest.raises(ValueError, match="Expected 3 embeddings for chunks, got 1"):
        await run_embed(config, cast(AsyncOpenAI, client))

    assert not (config.output_dir / "embeddings.npy").exists()
    assert not (config.output_dir / "embedding_manifest.json").exists()


async def test_run_embed_propagates_second_batch_failure_without_output(
    run_embed_config: EmbedConfig,
) -> None:
    failure = RuntimeError("Second batch failed")

    class FailingClient(FakeClient):
        async def create(
            self, *, model: str, input: tuple[str, ...]
        ) -> SimpleNamespace:
            if len(self.calls) == 1:
                self.calls.append((model, input))
                raise failure
            return await super().create(model=model, input=input)

    config = run_embed_config
    client = FailingClient()

    with pytest.raises(RuntimeError, match="Second batch failed") as exc:
        await run_embed(config, cast(AsyncOpenAI, client))

    assert exc.value is failure
    assert client.calls == [
        ("test-model", ("a", "bb")),
        ("test-model", ("ccc",)),
    ]
    assert not (config.output_dir / "embeddings.npy").exists()
    assert not (config.output_dir / "embedding_manifest.json").exists()
