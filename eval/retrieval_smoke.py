import argparse
import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import aiofiles
import rich_struct.log
from asyncstdlib import tee
from asyncstdlib import zip as zip_async
from openai import AsyncOpenAI, OpenAIError

from mini_rag.config import DEFAULT_EMBEDDING_MODEL
from mini_rag.embed import embed_texts_async
from mini_rag.evaluation import is_hit, load_questions_async, summarize_evaluation
from mini_rag.helpers import ErrorHandler
from mini_rag.models import RetrievalSmokeRecord, RetrievedChunkRecord
from mini_rag.paths import portable_path
from mini_rag.retrieve import search
from mini_rag.storage import load_index_async

logger = rich_struct.log.get_logger()


async def run_async(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluates retrieved documents against provided set of questions and expected answers"
    )

    parser.add_argument(
        "--questions", type=str, required=True, help="Path to the file with questions and answers"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many search results use to in question context",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=f"index/{DEFAULT_EMBEDDING_MODEL}",
        help="Location of embeddings index and manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"eval/results/{DEFAULT_EMBEDDING_MODEL}",
        help="Results directory",
    )

    args = parser.parse_args(argv)

    top_k = int(args.top_k)

    async with AsyncOpenAI() as client:
        await eval_async(
            client, top_k, Path(args.questions), Path(args.index_dir), Path(args.output_dir)
        )


async def eval_async(
    client: AsyncOpenAI, top_k: int, questions_path: Path, index_dir: Path, output_dir: Path
) -> dict[str, Any]:
    index = await load_index_async(index_dir)

    output_dir = portable_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_jsonl_path = output_dir / "retrieval_smoke.jsonl"
    logger.info("Saving retrieval smoke evaluation to {}", smoke_jsonl_path)

    smoke_summary_json_path = output_dir / "retrieval_smoke_summary.json"
    smoke_summary_json_path.unlink(missing_ok=True)

    evaluation_records: list[RetrievalSmokeRecord] = []

    async with (
        tee(load_questions_async(questions_path)) as (questions, embedding_questions),
        aiofiles.open(smoke_jsonl_path, mode="w", encoding="utf-8") as smoke_file,
    ):
        q_vectors = embed_texts_async(
            (q.question async for q in embedding_questions), client, index.embedding_manifest.model
        )
        async for question, vector in zip_async(questions, q_vectors, strict=True):
            results = search(vector, index.embeddings_matrix, index.chunks, top_k)

            hit = is_hit(
                frozenset(question.expected_documents),
                (r.chunk.document_id for r in results),
                question.answerable,
            )
            record = RetrievalSmokeRecord(
                id=question.id,
                type=question.type,
                expected_documents=question.expected_documents,
                top_k=[
                    RetrievedChunkRecord(
                        rank=r.rank,
                        score=r.score,
                        chunk_id=r.chunk.chunk_id,
                        document_id=r.chunk.document_id,
                        page=r.chunk.page,
                        page_end=r.chunk.page_end,
                    )
                    for r in results
                ],
                hit=hit,
            )
            evaluation_records.append(record)
            await smoke_file.write(f"{record.model_dump_json()}\n")

    answerable_count, hit_count, missed_questions = summarize_evaluation(evaluation_records)

    logger.info("Saving retrieval smoke evaluation summary to {}", smoke_summary_json_path)
    async with aiofiles.open(smoke_summary_json_path, mode="w", encoding="utf-8") as summary_file:
        summary = {
            "model": index.embedding_manifest.model,
            "top_k": top_k,
            "chunks_sha256": index.embedding_manifest.chunks_sha256,
            "answerable": answerable_count,
            "hits": hit_count,
            "missed_ids": missed_questions,
        }
        summary_json = json.dumps(summary, indent=2)
        await summary_file.write(summary_json)

    return summary


def main(argv: Iterable[str] | None = None):
    with ErrorHandler(OpenAIError, OSError, ValueError):
        rich_struct.log.configure()
        asyncio.run(run_async(argv))


if __name__ == "__main__":
    main()
