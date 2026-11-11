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


async def iter_embeddings(rows: list[list[float]]) -> AsyncIterator[list[float]]:
    for row in rows:
        yield row


def test_normalize_produces_unit_row_norms() -> None:
    result = normalize(iter_embeddings([[3.0, 4.0], [5.0, 12.0]]))

    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalize_preserves_direction() -> None:
    result = normalize(iter_embeddings([[3.0, 4.0], [-4.0, 3.0]]))

    assert np.allclose(result, [[0.6, 0.8], [-0.8, 0.6]])


def test_normalize_rejects_zero_row() -> None:
    with pytest.raises(ValueError, match="Zero"):
        normalize(iter_embeddings([[3.0, 4.0], [0.0, 0.0]]))


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
