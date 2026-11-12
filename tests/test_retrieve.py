from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from mini_rag.errors import DimensionMismatchError
from mini_rag.models import Chunk
from mini_rag.retrieve import search


def make_chunk(document_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}_p001_c001",
        document_id=document_id,
        title="Dummy title",
        collection="test",
        source_path=Path("dummy.pdf"),
        source_url="https://example.test/dummy.pdf",
        page=1,
        text="Dummy text",
        token_count=2,
    )


@pytest.fixture
def index() -> tuple[NDArray[np.float32], list[Chunk]]:
    matrix = np.array([[1, 0], [0, 1], [0.7071, 0.7071]], dtype=np.float32)
    chunks = [make_chunk(name) for name in ("a", "b", "c")]
    return matrix, chunks


def test_search_orders_results_by_score(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    results = search([1, 0], matrix, chunks, top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        chunks[0].chunk_id,
        chunks[2].chunk_id,
        chunks[1].chunk_id,
    ]
    assert [result.score for result in results] == pytest.approx(
        [1.0, 0.7071, 0.0], abs=1e-4
    )


def test_search_breaks_score_ties_by_chunk_id() -> None:
    matrix = np.array([[1, 0], [1, 0]], dtype=np.float32)
    chunks = [make_chunk("b"), make_chunk("a")]

    results = search([1, 0], matrix, chunks, top_k=2)

    assert [result.chunk.chunk_id for result in results] == [
        chunks[1].chunk_id,
        chunks[0].chunk_id,
    ]


def test_search_top_k_overflow_returns_all_chunks_with_sequential_ranks(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    results = search([1, 0], matrix, chunks, top_k=10)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


def test_search_rejects_empty_index() -> None:
    matrix = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        search([1, 0], matrix, [], top_k=3)


def test_search_rejects_dimension_mismatch(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    with pytest.raises(
        DimensionMismatchError,
        match="Question vector's dimension is 3, where the index dimension is 2",
    ):
        search([1, 0, 0], matrix, chunks, top_k=3)


@pytest.mark.parametrize(
    ("query", "shape"),
    [
        (np.array([[1, 0]], dtype=np.float32), r"\(1, 2\)"),
        (np.array(1, dtype=np.float32), r"\(\)"),
    ],
)
def test_search_rejects_non_vector_query(
    index: tuple[NDArray[np.float32], list[Chunk]],
    query: NDArray[np.float32],
    shape: str,
) -> None:
    matrix, chunks = index

    with pytest.raises(
        ValueError, match=f"Expected a 1-D query vector, got shape {shape}"
    ):
        search(query, matrix, chunks, top_k=3)


def test_search_accepts_float32_array_query(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    results = search(np.array([1, 0], dtype=np.float32), matrix, chunks, top_k=1)

    assert results[0].chunk == chunks[0]
    assert results[0].score == pytest.approx(1.0)


def test_search_is_deterministic(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    first = search([1, 0], matrix, chunks, top_k=3)
    second = search([1, 0], matrix, chunks, top_k=3)

    assert first == second


def test_search_rejects_zero_top_k(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    with pytest.raises(ValueError, match="`top_k` must be at least 1"):
        search([1, 0], matrix, chunks, top_k=0)


def test_search_normalizes_query(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    unit_results = search([1, 0], matrix, chunks, top_k=3)
    scaled_results = search([2, 0], matrix, chunks, top_k=3)

    assert [result.chunk for result in scaled_results] == [
        result.chunk for result in unit_results
    ]
    assert [result.score for result in scaled_results] == pytest.approx(
        [result.score for result in unit_results], abs=1e-4
    )


def test_search_rejects_zero_query(
    index: tuple[NDArray[np.float32], list[Chunk]],
) -> None:
    matrix, chunks = index

    with pytest.raises(ValueError):
        search([0, 0], matrix, chunks, top_k=3)
