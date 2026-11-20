import re
from pathlib import Path

import pytest

from mini_rag.models import Chunk, SearchResult
from mini_rag.prompts import build_context


@pytest.fixture
def results() -> list[SearchResult]:
    return [
        SearchResult(
            rank=rank,
            score=0.9 - index * 0.1,
            chunk=Chunk(
                chunk_id=f"doc{index}_p{page:03d}_c001",
                document_id=f"doc{index}",
                title=f"Distinct title {index}",
                collection="test",
                source_path=Path(f"doc{index}.pdf"),
                source_url=f"https://example.test/doc{index}.pdf",
                page=page,
                page_end=end,
                text=f"Distinct passage {index} with evidence.",
                token_count=8,
            ),
        )
        for index, (rank, page, end) in enumerate([(2, 6, 7), (8, 12, 12), (15, 20, 22)])
    ]


def test_context_contains_ordered_numbered_blocks(results: list[SearchResult]) -> None:
    context, _ = build_context(results)
    positions = [context.index(f"[{ref}]") for ref in (1, 2, 3)]
    assert positions == sorted(positions)
    for index, result in enumerate(results):
        end = positions[index + 1] if index + 1 < len(positions) else len(context)
        block = context[positions[index] : end]
        assert result.chunk.title in block
        assert result.chunk.document_id in block
        assert result.chunk.text in block


def test_ref_map_uses_positions_not_sparse_ranks(results: list[SearchResult]) -> None:
    context, refs = build_context(results)
    assert list(refs) == [1, 2, 3]
    assert [refs[ref].chunk_id for ref in refs] == [item.chunk.chunk_id for item in results]
    assert "[8]" not in context
    assert "[15]" not in context


def test_context_empty_input() -> None:
    assert build_context([]) == ("", {})


def test_context_renders_inclusive_page_ranges(results: list[SearchResult]) -> None:
    context, _ = build_context(results)
    assert re.search(r"pages\s+6\s*[-–]\s*7", context)
    assert re.search(r"pages\s+20\s*[-–]\s*22", context)
    assert "page 12" in context
    assert not re.search(r"pages?\s+12\s*[-–]\s*\d+", context)


def test_context_keeps_original_objects_and_input_unchanged(results: list[SearchResult]) -> None:
    before = [item.model_dump() for item in results]
    original_objects = [item.chunk for item in results]
    first = build_context(results)
    second = build_context(results)
    assert first == second
    assert [item.model_dump() for item in results] == before
    assert all(first[1][i] is chunk for i, chunk in enumerate(original_objects, start=1))


def test_context_preserves_input_order_without_resorting(results: list[SearchResult]) -> None:
    reordered = list(reversed(results))
    context, refs = build_context(reordered)
    assert refs[1] is reordered[0].chunk
    assert context.index(reordered[0].chunk.text) < context.index(reordered[1].chunk.text)


def test_blocks_have_clear_boundaries(results: list[SearchResult]) -> None:
    context, _ = build_context(results)
    for previous, ref in zip(results, (2, 3), strict=False):
        gap = context[
            context.index(previous.chunk.text) + len(previous.chunk.text) : context.index(
                f"[{ref}]"
            )
        ]
        assert "\n\n" in gap or "---" in gap


@pytest.mark.parametrize(
    "text",
    [
        "Source mentions [12] and [1].",
        "Unicode: 温暖化 — CO₂\nSecond line.",
        "Long passage. " * 1000,
    ],
)
def test_context_does_not_truncate_or_rewrite_source(
    results: list[SearchResult], text: str
) -> None:
    results[0].chunk.text = text
    context, refs = build_context(results[:1])
    assert text in context
    assert refs[1].text == text


def test_repeated_chunk_still_has_a_ref_per_context_position(results: list[SearchResult]) -> None:
    context, refs = build_context([results[0], results[0]])
    assert list(refs) == [1, 2]
    assert refs[1] is refs[2] is results[0].chunk
    assert context.count(results[0].chunk.text) == 2
