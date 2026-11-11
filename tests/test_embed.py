from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from openai import AsyncOpenAI

from mini_rag.embed import embed_texts, normalize, save_index
from mini_rag.models import EmbeddingManifest


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
