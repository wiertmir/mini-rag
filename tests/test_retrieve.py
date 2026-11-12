from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mini_rag.config import RetrieveConfig, load_config
from mini_rag.errors import DimensionMismatchError, StaleIndexError
from mini_rag.models import Chunk, EmbeddingManifest, SearchResult
from mini_rag.retrieve import main, preview, retrieve_async, search
from mini_rag.storage import calculate_file_sha, load_index_async, save_index


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


# --- #10: load the index and the retrieve command -------------------------


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )


@pytest.fixture
def mini_index(tmp_path: Path) -> tuple[Path, Path, list[Chunk]]:
    """Two chunks, a normalized 2 x 2 matrix and a manifest with a valid hash."""
    chunks = [make_chunk("a"), make_chunk("b")]
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, chunks)
    matrix = np.eye(2, dtype=np.float32)
    manifest = EmbeddingManifest(
        model="index-model",
        dimension=2,
        records=2,
        chunks_path=chunks_path,
        chunks_sha256=calculate_file_sha(chunks_path),
    )
    index_dir = tmp_path / "index"
    save_index(index_dir, matrix, manifest)
    return index_dir, chunks_path, chunks


async def test_load_index_happy_path(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, chunks = mini_index

    matrix, loaded_chunks, manifest = await load_index_async(index_dir, chunks_path)

    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float32
    assert loaded_chunks == chunks
    assert manifest.model == "index-model"
    assert (manifest.records, manifest.dimension) == (2, 2)
    assert manifest.chunks_path == chunks_path


async def test_load_index_defaults_to_chunks_path_from_manifest(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, _, chunks = mini_index

    _, loaded_chunks, _ = await load_index_async(index_dir, None)

    assert loaded_chunks == chunks


async def test_load_index_rejects_stale_chunk_file(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, chunks = mini_index
    changed = chunks[0].model_copy(update={"text": "Changed after embed"})
    write_chunks(chunks_path, [changed, chunks[1]])

    with pytest.raises(StaleIndexError, match="(?i)stale"):
        await load_index_async(index_dir, chunks_path)


async def test_load_index_hashes_the_overridden_chunk_file(
    mini_index: tuple[Path, Path, list[Chunk]], tmp_path: Path
) -> None:
    index_dir, _, chunks = mini_index
    other = tmp_path / "other_chunks.jsonl"
    write_chunks(other, [chunks[1], chunks[0]])  # same records, different order

    with pytest.raises(StaleIndexError):
        await load_index_async(index_dir, other)


async def test_load_index_rejects_row_count_mismatch(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, _ = mini_index
    np.save(index_dir / "embeddings.npy", np.eye(3, 2, dtype=np.float32))

    with pytest.raises(ValueError, match=r"3.*2"):
        await load_index_async(index_dir, chunks_path)


async def test_load_index_rejects_matrix_not_matching_manifest_dimension(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, _ = mini_index
    np.save(index_dir / "embeddings.npy", np.ones((2, 3), dtype=np.float32))

    with pytest.raises(DimensionMismatchError):
        await load_index_async(index_dir, chunks_path)


def test_preview_is_at_most_200_characters() -> None:
    assert len(preview("x" * 500)) <= 200


def test_preview_keeps_short_text() -> None:
    assert preview("short text") == "short text"


def test_preview_honours_custom_limit() -> None:
    assert len(preview("x" * 500, limit=50)) <= 50


class FakeEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, input: Sequence[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": list(input)})
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=self.vector)],
            usage=SimpleNamespace(total_tokens=1),
        )


async def test_retrieve_embeds_question_with_manifest_model(
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, chunks = mini_index
    embeddings = FakeEmbeddings([0.0, 1.0])
    client: Any = SimpleNamespace(embeddings=embeddings)
    config = RetrieveConfig(
        question="Which chunk?", top_k=2, index_dir=index_dir, chunks=chunks_path
    )

    results = await retrieve_async(config, client)

    assert embeddings.calls == [{"model": "index-model", "input": ["Which chunk?"]}]
    assert [result.chunk for result in results] == [chunks[1], chunks[0]]


def test_cli_accepts_top_k_of_one() -> None:
    config = load_config(RetrieveConfig, ["Question?", "--top-k", "1"])

    assert config.top_k == 1


def test_cli_rejects_top_k_of_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        load_config(RetrieveConfig, ["Question?", "--top-k", "0"])

    assert caught.value.code != 0
    assert "top_k" in capsys.readouterr().err


def test_main_prints_score_with_four_decimals(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hit = SearchResult(rank=1, score=0.051234, chunk=make_chunk("a"))

    async def fake_retrieve(*_: object) -> list[SearchResult]:
        return [hit]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("mini_rag.retrieve.retrieve_async", fake_retrieve)

    main(["Question?"])

    out = capsys.readouterr().out
    assert "0.0512" in out
    assert "0.05123" not in out
    assert "a_p001_c001" in out


def test_main_reports_missing_index_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(SystemExit) as caught:
        main(["Question?", "--index-dir", str(tmp_path / "missing")])

    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "Error" in err
    assert "Traceback" not in err


def test_main_reports_missing_api_key_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mini_index: tuple[Path, Path, list[Chunk]],
) -> None:
    index_dir, chunks_path, _ = mini_index
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as caught:
        main(["Question?", "--index-dir", str(index_dir), "--chunks", str(chunks_path)])

    assert caught.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
