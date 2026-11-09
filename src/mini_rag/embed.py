from collections.abc import AsyncIterator
from time import perf_counter

import openai
import rich_struct.log
from asyncstdlib import batched

from mini_rag.models import Chunk

rich_struct.log.configure()

logger = rich_struct.log.get_logger()


async def embed_chunks(
    chunks: AsyncIterator[Chunk],
    client: openai.AsyncOpenAI,
    model: str,
    batch_size: int,
) -> AsyncIterator[list[float]]:
    return embed_texts(
        (chunk.text async for chunk in chunks), client, model, batch_size
    )


async def embed_texts(
    texts: AsyncIterator[str], client: openai.AsyncOpenAI, model: str, batch_size: int
) -> AsyncIterator[list[float]]:
    batch_no = 0
    total_texts = 0
    total_tokens = 0
    start_time = perf_counter()
    async for batch in batched(texts, batch_size):
        batch_no += 1
        batch_seq = batch_no
        total_texts += len(batch)
        logger.info(
            "Calling OpenAI embeddings (model={}) with {} strings [Batch: {}]",
            model,
            len(batch),
            batch_seq,
        )
        response = await client.embeddings.create(model=model, input=batch)

        total_tokens += response.usage.total_tokens
        logger.debug(
            "OpenAI returned {} embeddings (total_tokens: {}) [Batch: {}]",
            len(response.data),
            response.usage.total_tokens,
            batch_seq,
        )
        for item in sorted(response.data, key=lambda x: x.index):
            yield item.embedding

    logger.info(
        "Embedded {} texts in {:.2f} seconds (total_tokens: {})",
        total_texts,
        perf_counter() - start_time,
        total_tokens,
    )
