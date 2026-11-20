import time
from collections.abc import AsyncIterator, Iterable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from statistics import mean

import aiofiles
import rich_struct.log

from mini_rag.ask import InvalidCitationError, ask_async
from mini_rag.errors import OutputAlreadyExists
from mini_rag.hashing import calculate_file_sha, calculate_text_sha
from mini_rag.llm import AnswerGenerator, Retriever
from mini_rag.models import (
    AnswerEvaluationConfiguration,
    AnswerEvaluationCounts,
    AnswerEvaluationIngestParameters,
    AnswerEvaluationLatency,
    AnswerEvaluationRecord,
    AnswerEvaluationSummary,
    Question,
    RetrievalEvaluationRecord,
)
from mini_rag.prompts import SYSTEM_PROMPT

logger = rich_struct.log.get_logger()


async def load_questions_async(questions_path: Path) -> AsyncIterator[Question]:
    async with aiofiles.open(questions_path, mode="r", encoding="utf-8") as file:
        async for line in file:
            question = Question.model_validate_json(line)
            yield question


def is_hit(
    expected_documents: AbstractSet[str], retrieved_document_ids: Iterable[str], answerable: bool
) -> bool | None:
    if not answerable:
        return None

    for doc in retrieved_document_ids:
        if doc in expected_documents:
            return True

    return False


def summarize_evaluation(
    records: Sequence[RetrievalEvaluationRecord],
) -> tuple[int, int, list[str]]:
    answerable_count = 0
    hit_count = 0
    missed_ids: list[str] = []

    for record in records:
        if record.hit is None:
            continue

        answerable_count += 1
        if record.hit:
            hit_count += 1
        else:
            missed_ids.append(record.id)

    return answerable_count, hit_count, missed_ids


async def baseline_evaluation_loop(
    questions_path: Path,
    output_dir: Path,
    retriever: Retriever,
    answer_generator: AnswerGenerator,
    top_k: int,
    force_overwrite: bool,
):
    logger.info("Loading questions from {}", questions_path)
    questions = load_questions_async(questions_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    record_file_path = output_dir / "baseline.jsonl"
    summary_file_path = output_dir / "baseline_summary.json"

    if (record_file_path.exists() or summary_file_path.exists()) and not force_overwrite:
        raise OutputAlreadyExists(
            "Baseline evaluation can't be ovewritten without extra `--force` parameter"
        )

    summary_file_path.unlink(missing_ok=True)

    logger.debug("Saving evaluation records to {}", record_file_path)

    async with aiofiles.open(record_file_path, mode="w", encoding="utf-8") as record_file:
        record_count = 0
        error_count = 0
        refusal_count = 0
        invalid_citation_count = 0
        answerable_count = 0
        answerable_hit_count = 0
        latencies: list[float] = []
        async for question in questions:
            record_count += 1
            start_time = time.perf_counter()
            record: AnswerEvaluationRecord | None = None
            try:
                if question.answerable:
                    answerable_count += 1
                answer = await ask_async(question.question, retriever, answer_generator, top_k)

                # No errors
                record = AnswerEvaluationRecord(
                    id=question.id,
                    type=question.type,
                    question=question.question,
                    reference_answer=question.reference_answer,
                    expected_documents=question.expected_documents,
                    answer=answer.answer,
                    citations=answer.citations,
                    insufficient_evidence=answer.insufficient_evidence,
                    retrieved_chunk_ids=answer.retrieved_chunk_ids,
                    cites_expected_document=None
                    if not question.answerable
                    else any(
                        citation.document_id in question.expected_documents
                        for citation in answer.citations
                    ),
                    latency_s=time.perf_counter() - start_time,
                )
                if record.insufficient_evidence == True:
                    refusal_count += 1
                if question.answerable and record.cites_expected_document == True:
                    answerable_hit_count += 1

            except InvalidCitationError:
                error_count += 1
                invalid_citation_count += 1
                record = AnswerEvaluationRecord(
                    id=question.id,
                    type=question.type,
                    question=question.question,
                    reference_answer=question.reference_answer,
                    expected_documents=question.expected_documents,
                    latency_s=time.perf_counter() - start_time,
                    error="invalid_citation",
                )
            except Exception as ex:  # noqa: BLE001
                error_count += 1
                record = AnswerEvaluationRecord(
                    id=question.id,
                    type=question.type,
                    question=question.question,
                    reference_answer=question.reference_answer,
                    expected_documents=question.expected_documents,
                    latency_s=time.perf_counter() - start_time,
                    error=type(ex).__name__,
                )

            await record_file.write(record.model_dump_json() + "\n")
            latencies.append(record.latency_s)

    summary = AnswerEvaluationSummary(
        configuration=AnswerEvaluationConfiguration(
            answer_model=answer_generator.model_name,
            embedding_model=retriever.embedding_manifest.model,
            top_k=top_k,
            chunks_sha256=retriever.embedding_manifest.chunks_sha256,
            dataset_sha256=calculate_file_sha(questions_path),
            prompt_sha256=calculate_text_sha(SYSTEM_PROMPT),
            ingest_parameters=AnswerEvaluationIngestParameters(
                tokenizer=retriever.ingest_manifest.parameters.tokenizer,
                strategy=retriever.ingest_manifest.parameters.strategy,
                chunk_size=retriever.ingest_manifest.parameters.chunk_size,
                chunk_overlap=retriever.ingest_manifest.parameters.chunk_overlap,
                min_chunk_tokens=retriever.ingest_manifest.parameters.min_chunk_tokens,
                skip_patterns=retriever.ingest_manifest.parameters.skip_patterns,
                box_pattern=retriever.ingest_manifest.parameters.box_pattern,
            ),
        ),
        counts=AnswerEvaluationCounts(
            records=record_count,
            errors=error_count,
            refusals=refusal_count,
            invalid_citation=invalid_citation_count,
            answerable=answerable_count,
            answerable_citation_hits=answerable_hit_count,
        ),
        latency_s=AnswerEvaluationLatency(mean=mean(latencies), max=max(latencies)),
    )

    logger.debug("Saving evaluation summary to {}", summary_file_path)
    async with aiofiles.open(summary_file_path, mode="w", encoding="utf-8") as summary_file:
        await summary_file.write(summary.model_dump_json(indent=2))
