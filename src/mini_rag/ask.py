"""Using LLM answers the user question using only provided knowledge/context"""

import argparse
import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass

import rich_struct.log
from openai import AsyncOpenAI, OpenAIError
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mini_rag.config import AskConfig, HelpFormatter, load_config
from mini_rag.helpers import ErrorHandler
from mini_rag.llm import AnswerGenerator, OpenAIAnswerGenerator, OpenAIRetriever, Retriever
from mini_rag.models import Answer, Chunk, Citation, LlmAnswer
from mini_rag.prompts import build_context

logger = rich_struct.log.get_logger()


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
    processed_ref_citations: set[int] = set()
    processed_txt_citations: set[int] = set()
    for llm_txt_ref in re.findall(r"\[(\d+)\]", llm_answer.answer):
        ref_num = int(llm_txt_ref)

        if ref_num in processed_txt_citations:
            continue

        processed_txt_citations.add(ref_num)

        if ref_num in ref_map:
            if ref_num not in processed_ref_citations:
                chunk = ref_map[ref_num]
                citation = Citation(
                    ref=ref_num,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page=chunk.page,
                    page_end=chunk.page_end,
                )
                citations.append(citation)
                processed_ref_citations.add(ref_num)

            if ref_num in ref_num_list:
                while ref_num in ref_num_list:
                    ref_num_list.remove(ref_num)
            else:
                mismatched_citations.append(ref_num)
        else:
            invalid_citations.append(ref_num)

    if ref_num_list:
        for ref_num in ref_num_list:
            if ref_num not in invalid_citations and ref_num not in ref_map:
                invalid_citations.append(ref_num)

    if invalid_citations:
        raise InvalidCitationError(invalid_citations)

    return ResolvedCitations(citations, False, mismatched_citations, ref_num_list)


async def ask_async(
    question: str, retriever: Retriever, generator: AnswerGenerator, top_k: int
) -> Answer:
    logger.info("Retrieving closest semanticly chunks for user question: {}", question)
    search_results = await retriever.retrieve_async(question, top_k)
    if not search_results:
        logger.info("No chunks retrieved; refusing without calling the answer model")
        return Answer(
            answer="I don't have enough evidence in the retrieved documents to answer this question.",
            citations=[],
            insufficient_evidence=True,
            retrieved_chunk_ids=[],
        )

    logger.info("Building question context")
    question_context = build_context(search_results)
    logger.info("Getting answer for user question from LLM")
    llm_answer = await generator.generate_async(question, question_context)

    resolved_citations = resolve_citations(llm_answer, question_context[1])
    if resolved_citations.non_empty_citations_error:
        logger.error("LLM failed to answer the question, but still returned non-empty cited refs")
    if resolved_citations.mismatched_citations:
        logger.warning(
            "LLM text answer cites aren't returned in cited refs too: {}",
            resolved_citations.mismatched_citations,
        )
    if resolved_citations.extra_references:
        logger.warning("Some of the LLM returbed cited refs aren't part of text answer")

    return Answer(
        answer=llm_answer.answer,
        citations=resolved_citations.resolved_citations,
        insufficient_evidence=llm_answer.insufficient_evidence,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in question_context[1].values()],
    )


def print_answer(answer: Answer, console: Console | None = None) -> None:
    """Render an answer and, when supported, its source metadata."""
    console = console or Console()
    border_style = "dim #92745b" if answer.insufficient_evidence else "dim #64748b"
    if answer.insufficient_evidence:
        console.print(Panel(Markdown(answer.answer), border_style=border_style))
        return

    citations = Table(box=None, header_style="dim #87cefa", row_styles=["dim #87cefa"])
    for column in ("ref", "document_id", "page", "page_end"):
        citations.add_column(column)
    for citation in answer.citations:
        citations.add_row(
            str(citation.ref),
            Text(citation.document_id),
            str(citation.page),
            str(citation.page_end),
        )
    console.print(
        Panel(Group(Markdown(answer.answer), Text(), citations), border_style=border_style)
    )


async def _run_async(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mini_rag.ask",
        description=__doc__,
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )

    config = load_config(AskConfig, argv, parser)
    rich_struct.log.configure(console=False, file=config.log_folder)

    async with AsyncOpenAI() as client:
        retriever = OpenAIRetriever(config.index_dir, client, config.min_score)
        await retriever.load_index()

        generator = OpenAIAnswerGenerator(client, config.answer_model)

        answer = await ask_async(config.question, retriever, generator, config.top_k)

    if config.json_output:
        print(answer.model_dump_json(indent=2))
    else:
        print_answer(answer)


def main(argv: Sequence[str] | None = None) -> None:
    with ErrorHandler(OpenAIError, OSError, ValueError):
        asyncio.run(_run_async(argv))


if __name__ == "__main__":
    main()
