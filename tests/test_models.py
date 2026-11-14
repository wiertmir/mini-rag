import pytest
from pydantic import ValidationError

from mini_rag.models import Chunk, SearchResult


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(
        chunk_id="sample_p001_c001",
        document_id="sample",
        title="Sample",
        collection="test",
        source_path="sample.pdf",
        source_url="https://example.test/sample.pdf",
        page=1,
        text="Sample text",
        token_count=2,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-1.0001, -1.0),
        (-1.000000001, -1.0),
        (-1.0, -1.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (1.000000001, 1.0),
        (1.0001, 1.0),
    ],
)
def test_search_result_clamps_rounding_overshoots(
    chunk: Chunk, score: float, expected: float
) -> None:
    result = SearchResult(rank=1, score=score, chunk=chunk)

    assert result.score == expected


@pytest.mark.parametrize("score", [-1.00011, 1.00011])
def test_search_result_rejects_scores_beyond_rounding_tolerance(chunk: Chunk, score: float) -> None:
    with pytest.raises(ValidationError):
        SearchResult(rank=1, score=score, chunk=chunk)
