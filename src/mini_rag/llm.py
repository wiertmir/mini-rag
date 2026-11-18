from time import perf_counter
from typing import Protocol, runtime_checkable

import rich_struct.log
from openai import AsyncOpenAI

from mini_rag.errors import MissingParsedOutputError
from mini_rag.models import Chunk, LlmAnswer
from mini_rag.prompts import SYSTEM_PROMPT

logger = rich_struct.log.get_logger()


@runtime_checkable
class AnswerGenerator(Protocol):
    """Structural interface for anything that can generate an answer."""

    async def generate_async(
        self, question: str, context: tuple[str, dict[int, Chunk]]
    ) -> LlmAnswer: ...


class OpenAIAnswerGenerator:
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self._client = client
        self._model_name = model_name

    async def generate_async(
        self, question: str, context: tuple[str, dict[int, Chunk]]
    ) -> LlmAnswer:
        logger.info("Quering OpenAI's model '{}'", self._model_name)

        user_content = f"""
User question is below (wrapped by tripple ```)
```
{question}
```

Data fragments, your all knowledge base is below (wrapped by tripple ```):
```
{context[0]}
```
"""

        start_time = perf_counter()
        response = await self._client.responses.parse(
            model=self._model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            text_format=LlmAnswer,
        )
        latency = perf_counter() - start_time

        usage = response.usage
        logger.info(
            "OpenAI model '{}' responded in {:.2f} seconds "
            "(input_tokens: {}, output_tokens: {}, total_tokens: {}) [Response: {}]",
            self._model_name,
            latency,
            usage.input_tokens if usage else None,
            usage.output_tokens if usage else None,
            usage.total_tokens if usage else None,
            response.id,
        )

        if response.output_parsed:
            return response.output_parsed

        refusal = next(
            (
                content.refusal
                for item in response.output
                if item.type == "message"
                for content in item.content
                if content.type == "refusal"
            ),
            None,
        )
        incomplete_reason = (
            response.incomplete_details.reason if response.incomplete_details else None
        )
        raise MissingParsedOutputError(
            model=self._model_name,
            response_id=response.id,
            status=response.status,
            refusal=refusal,
            incomplete_reason=incomplete_reason,
        )
