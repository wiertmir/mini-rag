import pytest
from pydantic import BaseModel, ValidationError

from mini_rag.models import Answer, Chunk, Citation, LlmAnswer, SearchResult


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


@pytest.fixture
def citation() -> Citation:
    return Citation(ref=1, chunk_id="sample_p006_c001", document_id="sample", page=6, page_end=9)


@pytest.fixture
def grounded_answer(citation: Citation) -> Answer:
    return Answer(
        answer="A grounded answer [1].",
        citations=[citation],
        insufficient_evidence=False,
        retrieved_chunk_ids=[citation.chunk_id, "uncited_p010_c001"],
    )


def test_answer_preserves_uncited_retrievals_in_ranking_order(grounded_answer: Answer) -> None:
    assert grounded_answer.retrieved_chunk_ids == ["sample_p006_c001", "uncited_p010_c001"]
    assert [item.ref for item in grounded_answer.citations] == [1]


@pytest.mark.parametrize("retrieved", [[], ["sample_p006_c001"]])
def test_refusal_can_preserve_retrievals(retrieved: list[str]) -> None:
    result = Answer(
        answer="Insufficient information.",
        citations=[],
        insufficient_evidence=True,
        retrieved_chunk_ids=retrieved,
    )
    assert result.retrieved_chunk_ids == retrieved
    assert Answer.model_validate_json(result.model_dump_json()) == result


def test_refusal_rejects_citations(citation: Citation) -> None:
    with pytest.raises(ValidationError, match="insufficient evidence"):
        Answer(
            answer="Insufficient information.",
            citations=[citation],
            insufficient_evidence=True,
            retrieved_chunk_ids=[citation.chunk_id],
        )


def test_non_refusal_requires_citation() -> None:
    with pytest.raises(ValidationError, match="at least one citation"):
        Answer(
            answer="An answer.", citations=[], insufficient_evidence=False, retrieved_chunk_ids=[]
        )


def test_answer_rejects_duplicate_refs(grounded_answer: Answer) -> None:
    data = grounded_answer.model_dump()
    data["citations"].append(data["citations"][0].copy())
    with pytest.raises(ValidationError, match="unique"):
        Answer.model_validate(data)


def test_answer_rejects_citation_outside_retrieved_set(grounded_answer: Answer) -> None:
    data = grounded_answer.model_dump()
    data["retrieved_chunk_ids"] = ["unrelated_p001_c001"]
    with pytest.raises(ValidationError, match="Citation must come from"):
        Answer.model_validate(data)


def test_answer_round_trip_preserves_multi_page_citation(grounded_answer: Answer) -> None:
    restored = Answer.model_validate_json(grounded_answer.model_dump_json())
    assert restored == grounded_answer
    assert restored.citations[0].page == 6
    assert restored.citations[0].page_end == 9


@pytest.mark.parametrize(("page", "page_end"), [(1, 1), (6, 6), (6, 9)])
def test_citation_accepts_inclusive_physical_page_range(page: int, page_end: int) -> None:
    citation = Citation(
        ref=1, chunk_id="sample_p006_c001", document_id="sample", page=page, page_end=page_end
    )
    assert Citation.model_validate_json(citation.model_dump_json()) == citation


@pytest.mark.parametrize(("page", "page_end"), [(6, 5), (9, 3)])
def test_citation_rejects_reversed_range(page: int, page_end: int) -> None:
    with pytest.raises(ValidationError):
        Citation(
            ref=1, chunk_id="sample_p006_c001", document_id="sample", page=page, page_end=page_end
        )


@pytest.mark.parametrize("field", ["ref", "page", "page_end"])
@pytest.mark.parametrize("value", [0, -1])
def test_citation_rejects_nonpositive_numbers(citation: Citation, field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate({**citation.model_dump(), field: value})


@pytest.mark.parametrize("field", ["ref", "chunk_id", "document_id", "page", "page_end"])
def test_citation_requires_metadata(citation: Citation, field: str) -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate(citation.model_dump(exclude={field}))


@pytest.mark.parametrize(
    "field", ["answer", "citations", "insufficient_evidence", "retrieved_chunk_ids"]
)
def test_answer_requires_all_fields(grounded_answer: Answer, field: str) -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate(grounded_answer.model_dump(exclude={field}))


@pytest.mark.parametrize("refusal", [False, True])
def test_llm_answer_preserves_raw_refs_even_for_inconsistent_refusal(refusal: bool) -> None:
    result = LlmAnswer(
        answer="A response [1], [3].", cited_refs=[1, 3, 1], insufficient_evidence=refusal
    )
    assert result.cited_refs == [1, 3, 1]
    assert LlmAnswer.model_validate_json(result.model_dump_json()) == result


def test_llm_answer_schema_has_required_plain_fields() -> None:
    schema = LlmAnswer.model_json_schema()
    assert list(schema["properties"]) == ["answer", "cited_refs", "insufficient_evidence"]
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["cited_refs"]["items"]["type"] == "integer"
    assert all("default" not in field for field in schema["properties"].values())


@pytest.mark.parametrize("field", ["answer", "cited_refs", "insufficient_evidence"])
def test_llm_answer_requires_all_fields(field: str) -> None:
    data = {"answer": "A response [1].", "cited_refs": [1], "insufficient_evidence": False}
    data.pop(field)
    with pytest.raises(ValidationError):
        LlmAnswer.model_validate(data)


def test_answer_models_forbid_extra_fields(citation: Citation, grounded_answer: Answer) -> None:
    instances: list[BaseModel] = [
        citation,
        grounded_answer,
        LlmAnswer(answer="Text", cited_refs=[], insufficient_evidence=True),
    ]
    for instance in instances:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            type(instance).model_validate({**instance.model_dump(), "unexpected": "value"})
