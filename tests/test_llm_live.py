"""Opt-in generator contract checks. Run from the repo with its existing index.

The evidence chunk is pinned by ID and text hash, not retrieved afresh per run.
No document embeddings are regenerated. An exported OPENAI_API_KEY is required.
"""

import hashlib
from pathlib import Path

import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from mini_rag.config import AskConfig
from mini_rag.llm import OpenAIAnswerGenerator
from mini_rag.models import Chunk, SearchResult
from mini_rag.prompts import build_context
from mini_rag.storage import load_index_async

pytestmark = pytest.mark.live


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def fixed_context() -> tuple[str, dict[int, Chunk]]:
    loaded = await load_index_async(Path("index"))
    chunk = next(c for c in loaded.chunks if c.chunk_id == "WGIII_CH07_p008_c001")
    assert (
        hashlib.sha256(chunk.text.encode()).hexdigest()
        == "7bc4b2edc8a8af25e41fe038b1ffd5ed285bffe7575e05c3712a5a320005d1a6"
    )
    return build_context([SearchResult(rank=1, score=1.0, chunk=chunk)])


async def test_supported_context_produces_cited_answer(
    fixed_context: tuple[str, dict[int, Chunk]],
) -> None:
    async with AsyncOpenAI(max_retries=0, timeout=45) as client:
        result = await OpenAIAnswerGenerator(client, AskConfig().answer_model).generate_async(
            "What four broad categories of mitigation measures does this chapter describe?",
            fixed_context,
        )
    assert result.insufficient_evidence is False
    assert result.cited_refs
    assert set(result.cited_refs) <= set(fixed_context[1])


async def test_q29_refuses_without_evidence(fixed_context: tuple[str, dict[int, Chunk]]) -> None:
    async with AsyncOpenAI(max_retries=0, timeout=45) as client:
        result = await OpenAIAnswerGenerator(client, AskConfig().answer_model).generate_async(
            "How much did Apple earn in fiscal year 2025?", fixed_context
        )
    assert result.insufficient_evidence is True
    assert result.cited_refs == []
