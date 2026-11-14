import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import aiofiles
import rich_struct.log
from asyncstdlib import tee
from asyncstdlib import zip as zip_async
from openai import AsyncOpenAI, OpenAIError

from mini_rag.embed import embed_texts_async
from mini_rag.evaluation import is_hit
from mini_rag.helpers import ErrorHandler
from mini_rag.models import Question
from mini_rag.retrieve import search
from mini_rag.storage import load_index_async

logger = rich_struct.log.get_logger()


async def load_questions_async(questions_path: Path) -> AsyncIterator[Question]:
    async with aiofiles.open(questions_path, mode="r", encoding="utf-8") as file:
        async for line in file:
            question = Question.model_validate_json(line)
            yield question


async def run_async(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluates retrieved documents aginst provied set of questions and expected answers"
    )

    parser.add_argument(
        "--questions", type=str, required=True, help="Path to the file with questions and answers"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many search results use to compare with exected answers",
    )
    parser.add_argument(
        "--index-dir", type=str, default="index", help="Location of embeddings index and manifest"
    )
    parser.add_argument("--output-dir", type=str, default="eval/results", help="Results directory")

    args = parser.parse_args(argv)

    top_k = int(args.top_k)

    await eval_async(top_k, Path(args.questions), Path(args.index_dir), Path(args.output_dir))


async def eval_async(top_k: int, questions_path: Path, index_dir: Path, output_dir: Path):
    index = await load_index_async(index_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_jsonl_path = output_dir / "retrieval_smoke.jsonl"
    logger.info("Saving retrieval smoke evaluation to {}", smoke_jsonl_path)

    smoke_summary_json_path = output_dir / "retrieval_smoke_summary.json"
    smoke_summary_json_path.unlink(missing_ok=True)

    answerable_count = 0
    hit_count = 0
    missed_questions: list[str] = []

    async with (
        tee(load_questions_async(questions_path)) as (questions, embedding_questions),
        AsyncOpenAI() as client,
        aiofiles.open(smoke_jsonl_path, mode="w", encoding="utf-8") as smoke_file,
    ):
        q_vectors = embed_texts_async(
            (q.question async for q in embedding_questions), client, index.manifest.model
        )
        async for question, vector in zip_async(questions, q_vectors, strict=True):
            results = search(vector, index.embeddings_matrix, index.chunks, top_k)

            if question.answerable:
                answerable_count += 1
            hit = is_hit(
                frozenset(question.expected_documents),
                (r.chunk.document_id for r in results),
                question.answerable,
            )
            if hit == True:
                hit_count += 1
            elif hit == False:
                missed_questions.append(question.id)

            record = {
                "id": question.id,
                "type": question.type.value,
                "expected_documents": question.expected_documents,
                "top_k": [
                    {
                        "rank": r.rank,
                        "score": r.score,
                        "chunk_id": r.chunk.chunk_id,
                        "document_id": r.chunk.document_id,
                        "page": r.chunk.page,
                    }
                    for r in results
                ],
                "hit": hit,
            }
            json_line = f"{json.dumps(record)}\n"
            await smoke_file.write(json_line)

    logger.info("Saving retrieval smoke evaluation summary to {}", smoke_summary_json_path)
    async with aiofiles.open(smoke_summary_json_path, mode="w", encoding="utf-8") as summary_file:
        summary = {
            "model": index.manifest.model,
            "top_k": top_k,
            "chunks_sha256": index.manifest.chunks_sha256,
            "answerable": answerable_count,
            "hits": hit_count,
            "missed_ids": missed_questions,
        }
        summary_json = json.dumps(summary, indent=2)
        await summary_file.write(summary_json)


def main(argv: Iterable[str] | None = None):
    with ErrorHandler(OpenAIError, OSError, ValueError):
        rich_struct.log.configure()
        asyncio.run(run_async(argv))


if __name__ == "__main__":
    main()
