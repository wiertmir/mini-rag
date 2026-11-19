"""Offline contract tests for issue #17; the richer result keeps diagnostics visible."""

from pathlib import Path

import pytest

from mini_rag.ask import InvalidCitationError, resolve_citations
from mini_rag.models import Chunk, LlmAnswer


@pytest.fixture
def ref_map() -> dict[int, Chunk]:
    return {
        ref: Chunk(
            chunk_id=f"doc{ref}_p{page:03d}_c001",
            document_id=f"doc{ref}",
            title=f"Document {ref}",
            collection="test",
            source_path=Path(f"doc{ref}.pdf"),
            source_url=f"https://example.test/doc{ref}.pdf",
            page=page,
            page_end=end,
            text=f"Evidence {ref}",
            token_count=3,
        )
        for ref, page, end in [(1, 4, 7), (2, 12, 12), (3, 20, 22)]
    }


def answer(text: str, refs: list[int], *, refusal: bool = False) -> LlmAnswer:
    return LlmAnswer(answer=text, cited_refs=refs, insufficient_evidence=refusal)


def test_maps_metadata_and_both_physical_page_endpoints(ref_map: dict[int, Chunk]) -> None:
    result = resolve_citations(answer("First [1], third [3].", [1, 3]), ref_map)
    assert [c.model_dump() for c in result.resolved_citations] == [
        {
            "ref": ref,
            "chunk_id": ref_map[ref].chunk_id,
            "document_id": ref_map[ref].document_id,
            "page": ref_map[ref].page,
            "page_end": ref_map[ref].page_end,
        }
        for ref in [1, 3]
    ]
    assert not result.non_empty_citations_error
    assert result.mismatched_citations == result.extra_references == []


def test_single_page_stays_single_page(ref_map: dict[int, Chunk]) -> None:
    citation = resolve_citations(answer("Evidence [2]", [2]), ref_map).resolved_citations[0]
    assert citation.page == citation.page_end == 12


@pytest.mark.parametrize("refs", [[], [2], [7, 0, -1]])
def test_refusal_has_no_resolved_citations_and_flags_refs(
    ref_map: dict[int, Chunk], refs: list[int]
) -> None:
    result = resolve_citations(answer("Cannot answer [7].", refs, refusal=True), ref_map)
    assert result.resolved_citations == []
    assert result.non_empty_citations_error is bool(refs)
    assert result.mismatched_citations == result.extra_references == []


@pytest.mark.parametrize("ref", [7, 0, -1])
def test_unknown_structured_ref_raises_even_without_text_marker(
    ref_map: dict[int, Chunk], ref: int
) -> None:
    with pytest.raises(InvalidCitationError, match=str(ref)) as caught:
        resolve_citations(answer("No inline marker.", [ref]), ref_map)
    assert caught.value.invalid_refs == [ref]


def test_collects_all_invalid_structured_refs(ref_map: dict[int, Chunk]) -> None:
    with pytest.raises(InvalidCitationError) as caught:
        resolve_citations(answer("Evidence [1].", [1, 7, 9, 0, -2]), ref_map)
    assert set(caught.value.invalid_refs) == {7, 9, 0, -2}
    assert all(str(ref) in str(caught.value) for ref in [7, 9, 0, -2])


def test_unknown_inline_marker_is_visible(ref_map: dict[int, Chunk]) -> None:
    with pytest.raises(InvalidCitationError, match="7") as caught:
        resolve_citations(answer("Evidence [7]", [7]), ref_map)
    assert caught.value.invalid_refs == [7]


@pytest.mark.parametrize("refs", [[1], [1, 1]])
def test_repeated_text_marker_resolves_to_one_citation(
    ref_map: dict[int, Chunk], refs: list[int]
) -> None:
    result = resolve_citations(answer("Claim [1]. Another claim [1].", refs), ref_map)
    assert [c.ref for c in result.resolved_citations] == [1]
    assert result.mismatched_citations == result.extra_references == []


def test_duplicate_structured_refs_do_not_create_false_extra_refs(
    ref_map: dict[int, Chunk],
) -> None:
    result = resolve_citations(answer("Claim [1].", [1, 1]), ref_map)
    assert [c.ref for c in result.resolved_citations] == [1]
    assert result.extra_references == []


def test_warning_only_marker_check_keeps_structured_refs_authoritative(
    ref_map: dict[int, Chunk],
) -> None:
    result = resolve_citations(answer("Claim [1].", [2]), ref_map)
    assert [c.ref for c in result.resolved_citations] == [2]
    assert result.mismatched_citations == [1]
    assert result.extra_references == [2]


def test_valid_structured_ref_without_marker_is_not_dropped(ref_map: dict[int, Chunk]) -> None:
    result = resolve_citations(answer("Claim without marker.", [3]), ref_map)
    assert [c.ref for c in result.resolved_citations] == [3]
    assert result.extra_references == [3]


def test_inputs_unchanged_and_results_deterministic(ref_map: dict[int, Chunk]) -> None:
    llm_answer = answer("Third [3], first [1].", [3, 1])
    before_answer = llm_answer.model_dump()
    before_map = {ref: chunk.model_dump() for ref, chunk in ref_map.items()}
    first = resolve_citations(llm_answer, ref_map)
    assert first == resolve_citations(llm_answer, ref_map)
    assert sorted(c.ref for c in first.resolved_citations) == [1, 3]
    assert llm_answer.model_dump() == before_answer
    assert {ref: chunk.model_dump() for ref, chunk in ref_map.items()} == before_map
