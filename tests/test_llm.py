from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from mini_rag.config import AskConfig
from mini_rag.errors import MissingParsedOutputError
from mini_rag.llm import AnswerGenerator, OpenAIAnswerGenerator
from mini_rag.models import LlmAnswer
from mini_rag.prompts import SYSTEM_PROMPT


class FakeResponses:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def make_generator(
    parsed: LlmAnswer | None = None, **overrides: Any
) -> tuple[OpenAIAnswerGenerator, FakeResponses]:
    response = SimpleNamespace(
        output_parsed=parsed,
        usage=None,
        id="resp_test",
        status="completed",
        output=[],
        incomplete_details=None,
    )
    for key, value in overrides.items():
        setattr(response, key, value)
    responses = FakeResponses(response)
    client = cast(AsyncOpenAI, SimpleNamespace(responses=responses))
    return OpenAIAnswerGenerator(client, "configured-model"), responses


@pytest.mark.parametrize("refusal", [False, True])
async def test_generator_passes_schema_messages_and_returns_parsed_object(refusal: bool) -> None:
    parsed = LlmAnswer(
        answer="Answer", cited_refs=[] if refusal else [1], insufficient_evidence=refusal
    )
    generator, responses = make_generator(parsed)
    assert isinstance(generator, AnswerGenerator)
    result = await generator.generate_async("Test question?", ("[1] Evidence", {}))
    assert result is parsed
    request = responses.calls[0]
    assert request["model"] == "configured-model"
    assert request["text_format"] is LlmAnswer
    assert request["input"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert request["input"][1]["role"] == "user"
    assert "Test question?" in request["input"][1]["content"]
    assert "[1] Evidence" in request["input"][1]["content"]
    assert "temperature" not in request


async def test_generator_reuses_injected_client() -> None:
    parsed = LlmAnswer(answer="Text", cited_refs=[1], insufficient_evidence=False)
    generator, responses = make_generator(parsed)
    for question in ("one", "two"):
        assert await generator.generate_async(question, ("context", {})) is parsed
    assert len(responses.calls) == 2


@pytest.mark.parametrize("mode", ["safety", "incomplete", "missing"])
async def test_no_parsed_output_raises_diagnostic_error(mode: str) -> None:
    details: dict[str, Any] = {}
    if mode == "safety":
        details["output"] = [
            SimpleNamespace(
                type="message", content=[SimpleNamespace(type="refusal", refusal="Safety refusal")]
            )
        ]
    elif mode == "incomplete":
        details.update(
            status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")
        )
    generator, _ = make_generator(**details)
    with pytest.raises(MissingParsedOutputError) as caught:
        await generator.generate_async("question", ("context", {}))
    assert caught.value.model == "configured-model"
    assert caught.value.response_id == "resp_test"
    if mode == "safety":
        assert caught.value.refusal == "Safety refusal"
        assert "Safety refusal" in str(caught.value)
    if mode == "incomplete":
        assert caught.value.incomplete_reason == "max_output_tokens"
        assert caught.value.status == "incomplete"
        assert "Generation incomplete" in str(caught.value)
        assert "refused" not in str(caught.value).lower()
        assert "max_output_tokens" in str(caught.value)
    if mode == "missing":
        assert "No parsed output returned" in str(caught.value)


async def test_usage_logging_excludes_context(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: list[str] = []

    def info(message: str, *args: Any) -> None:
        entries.append(message.format(*args))

    monkeypatch.setattr("mini_rag.llm.logger", SimpleNamespace(info=info))
    parsed = LlmAnswer(answer="Text", cited_refs=[1], insufficient_evidence=False)
    generator, _ = make_generator(
        parsed, usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120)
    )
    await generator.generate_async("PRIVATE_QUESTION", ("PRIVATE_CONTEXT", {}))
    logs = "\n".join(entries)
    assert "configured-model" in logs
    assert (
        "input_tokens: 100" in logs and "output_tokens: 20" in logs and "total_tokens: 120" in logs
    )
    assert "seconds" in logs
    assert "PRIVATE_CONTEXT" not in logs and "PRIVATE_QUESTION" not in logs


@pytest.mark.parametrize("top_k", [0, -1])
def test_ask_config_rejects_invalid_top_k(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, top_k: int
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        AskConfig(top_k=top_k)


def test_ask_config_precedence(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_RAG_CONFIG", raising=False)
    monkeypatch.delenv("ANSWER_MODEL", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    (tmp_path / "config.toml").write_text('[ask]\nanswer_model="toml-model"\ntop_k=3\n')
    assert AskConfig().answer_model == "toml-model"
    (tmp_path / ".env").write_text("ANSWER_MODEL=dotenv-model\n")
    assert AskConfig().answer_model == "dotenv-model"
    monkeypatch.setenv("ANSWER_MODEL", "env-model")
    assert AskConfig().answer_model == "env-model"
    assert AskConfig(answer_model="explicit-model").answer_model == "explicit-model"
    assert AskConfig().top_k == 3


def test_ask_index_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINI_RAG_CONFIG", raising=False)
    monkeypatch.delenv("INDEX_DIR", raising=False)
    assert AskConfig().index_dir == Path("index")
    (tmp_path / "config.toml").write_text('[ask]\nindex_dir="toml-index"\n')
    assert AskConfig().index_dir == Path("toml-index")
    (tmp_path / ".env").write_text("INDEX_DIR=dotenv-index\n")
    assert AskConfig().index_dir == Path("dotenv-index")
    monkeypatch.setenv("INDEX_DIR", "env-index")
    assert AskConfig().index_dir == Path("env-index")
    assert AskConfig(index_dir=Path("explicit-index")).index_dir == Path("explicit-index")
