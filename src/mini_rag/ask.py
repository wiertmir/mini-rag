from dataclasses import dataclass
import re

from mini_rag.models import Chunk, Citation, LlmAnswer


class InvalidCitationError(ValueError):
    """Error indicating that AI is hallucinating and picking up non-existent citations"""

    invalid_refs: list[int]

    def __init__(self, invalid_refs: list[int]):
        self.invalid_refs = invalid_refs
        super().__init__(
            f"AI is hallucinating and quoting non-existing citations: {', '.join(str(r) for r in self.invalid_refs)}"
        )


@dataclass(frozen=True)
class ResolvedCitations:
    resolved_citations: list[Citation]
    non_empty_citations_error: bool
    mismatched_citations: list[int]
    extra_references: list[int]


def resolve_citations(llm_answer: LlmAnswer, ref_map: dict[int, Chunk]) -> ResolvedCitations:
    if llm_answer.insufficient_evidence:
        return ResolvedCitations([], len(llm_answer.cited_refs) > 0, [], [])

    invalid_citations: list[int] = []
    mismatched_citations: list[int] = []
    citations: list[Citation] = []
    ref_num_list = list(llm_answer.cited_refs)
    for llm_txt_ref in re.findall(r"\[(\d+)\]", llm_answer.answer):
        ref_num = int(llm_txt_ref)

        if ref_num in ref_map:
            chunk = ref_map[ref_num]
            citation = Citation(
                ref=ref_num,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page=chunk.page,
                page_end=chunk.page_end,
            )
            citations.append(citation)

            if ref_num in ref_num_list:
                ref_num_list.remove(ref_num)
            else:
                mismatched_citations.append(ref_num)
        else:
            invalid_citations.append(ref_num)

    if invalid_citations:
        raise InvalidCitationError(invalid_citations)

    return ResolvedCitations(citations, False, mismatched_citations, ref_num_list)
