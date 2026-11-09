from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import cast

from openai import AsyncOpenAI

from mini_rag.embed import embed_texts


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = self
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def create(self, *, model: str, input: tuple[str, ...]) -> SimpleNamespace:
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(text)), 0.0])
                for index, text in reversed(list(enumerate(input)))
            ],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


async def iter_texts(texts: list[str]) -> AsyncIterator[str]:
    for text in texts:
        yield text


async def test_embed_texts_batches_with_partial_final_batch() -> None:
    client = FakeClient()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts(texts), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert [len(batch) for _, batch in client.calls] == [2, 2, 1]
    assert client.calls == [
        ("test-model", ("a", "bb")),
        ("test-model", ("ccc", "dddd")),
        ("test-model", ("eeeee",)),
    ]
    assert len(vectors) == 5


async def test_embed_texts_preserves_text_order() -> None:
    client = FakeClient()
    texts = ["four", "a", "three", "bb", "sixsix"]

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts(texts), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert vectors == [[float(len(text)), 0.0] for text in texts]


async def test_embed_texts_empty_iterator() -> None:
    client = FakeClient()

    vectors = [
        vector
        async for vector in embed_texts(
            iter_texts([]), cast(AsyncOpenAI, client), "test-model", batch_size=2
        )
    ]

    assert vectors == []
    assert client.calls == []
