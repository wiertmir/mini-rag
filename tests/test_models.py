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
        page_end=1,
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


def test_chunk_page_end_defaults_to_page_for_old_files(chunk: Chunk) -> None:
    data = chunk.model_dump(exclude={"page_end"})

    assert Chunk.model_validate(data).page_end == chunk.page


def test_chunk_rejects_page_end_before_page(chunk: Chunk) -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({**chunk.model_dump(), "page": 3, "page_end": 2})


def test_chunk_pages_label(chunk: Chunk) -> None:
    assert chunk.pages == "1"
    assert chunk.model_copy(update={"page": 2, "page_end": 4}).pages == "2-4"
