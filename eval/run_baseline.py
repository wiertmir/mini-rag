import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import rich_struct.log
from openai import AsyncOpenAI, OpenAIError

from mini_rag.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL
from mini_rag.evaluation import baseline_evaluation_loop
from mini_rag.helpers import ErrorHandler
from mini_rag.llm import OpenAIAnswerGenerator, OpenAIRetriever

logger = rich_struct.log.get_logger()


async def _run_async(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Evaluates LLM answers against provided set of questions and expected answers"
    )
    parser.add_argument(
        "--questions", type=str, required=True, help="Path to the file with questions and answers"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=f"index/{DEFAULT_EMBEDDING_MODEL}",
        help=f"Location of embeddings index and manifest. Default: 'index/{DEFAULT_EMBEDDING_MODEL}'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Results directory. Default='eval/results/<embedding_model>/<llm_model>'",
    )
    parser.add_argument(
        "--local-ollama",
        action="store_true",
        help="If true everything runs on local ollama service",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_LLM_MODEL, help="Model to use to answer questions"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many search results use to in question context",
    )
    parser.add_argument(
        "--force", action="store_true", help="If set, existing output will be overwritten"
    )
    args = parser.parse_args(argv)

    questions_path = Path(args.questions)
    index_dir = Path(args.index_dir)
    logger.info(
        "Starting evaluation script with questions in '{}' and index in '{}'. LLM: {}",
        questions_path,
        index_dir,
        args.model,
    )
    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        logger.info("...and output_dir='{}'", output_dir)

    def get_openai_client(run_locally: bool) -> AsyncOpenAI:
        if run_locally:
            logger.info("...and we are running on local ollama")
            return AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            return AsyncOpenAI()

    async with get_openai_client(args.local_ollama) as client:
        retriever = OpenAIRetriever(index_dir, client)
        embedding_manifest = await retriever.load_index()

        llm_model: str = args.model
        generator = OpenAIAnswerGenerator(client, llm_model)

        if output_dir is None:
            output_dir = Path("eval/results") / embedding_manifest.model / llm_model
            logger.info(
                "Based on selected index directory and model, the output directory is {}",
                output_dir,
            )

        await baseline_evaluation_loop(
            questions_path, output_dir, retriever, generator, args.top_k, args.force
        )


def main(argv: Sequence[str] | None = None):
    with ErrorHandler(OpenAIError, OSError, ValueError):
        rich_struct.log.configure()
        asyncio.run(_run_async(argv))


if __name__ == "__main__":
    main()
