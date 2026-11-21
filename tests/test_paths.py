import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mini_rag.config import EmbedConfig
from mini_rag.paths import portable_path, safe_filename


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("qwen3-embedding:4b", "qwen3-embedding_4b"),
        ("org/model:tag", "org_model_tag"),
        ("CON.txt", "_CON.txt"),
        ("lpt1", "_lpt1"),
        ('bad<>:"/\\|?*\x01. ', "bad__________"),
        ("...", "_"),
        ("valid-name.npy", "valid-name.npy"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected
    assert safe_filename(expected) == expected


def test_paths_preserve_root_and_parent_but_not_model_colons(tmp_path: Path) -> None:
    path = tmp_path / ".." / "qwen:4b" / "model:12b"
    assert portable_path(path) == tmp_path / ".." / "qwen_4b" / "model_12b"
    config = EmbedConfig(
        chunks=Path("index/chunks.jsonl"),
        output_dir=Path("index/qwen:4b"),
        embedding_model="qwen:4b",
    )
    assert config.output_dir == Path("index/qwen_4b")
    assert config.embedding_model == "qwen:4b"


async def test_exact_legacy_cli_keeps_api_names_and_uses_portable_output(monkeypatch):
    from mini_rag.llm import OpenAIAnswerGenerator

    spec = importlib.util.spec_from_file_location("portable_baseline", "eval/run_baseline.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = AsyncMock()
    client.__aenter__.return_value = client
    manifest = SimpleNamespace(model="qwen3-embedding:4b")
    retriever = SimpleNamespace(load_index=AsyncMock(return_value=manifest))
    index_paths = []
    options = []

    def make_client(**kwargs):
        options.append(kwargs)
        return client

    def make_retriever(path, sdk):
        assert sdk is client
        index_paths.append(path)
        return retriever

    evaluator = AsyncMock()
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    monkeypatch.setattr(module, "OpenAIRetriever", make_retriever)
    monkeypatch.setattr(module, "baseline_evaluation_loop", evaluator)
    await module._run_async(
        [
            "--questions",
            "data/questions.jsonl",
            "--index-dir=index/qwen3-embedding:4b",
            "--local-ollama",
            "--model=gemma4:12b-q8-8k",
        ]
    )
    assert options == [{"base_url": "http://localhost:11434/v1", "api_key": "ollama"}]
    assert index_paths == [Path("index/qwen3-embedding_4b")]
    args = evaluator.await_args.args
    assert args[1] == Path("eval/results/qwen3-embedding_4b/gemma4_12b-q8-8k")
    assert isinstance(args[3], OpenAIAnswerGenerator)
    assert args[3].model_name == "gemma4:12b-q8-8k"
    assert manifest.model == "qwen3-embedding:4b"
    assert args[-1] is False


def test_repository_names_are_portable():
    import subprocess

    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    for raw in names.split(b"\0"):
        if not raw:
            continue
        path = Path(raw.decode())
        if path.exists():  # renamed paths still in the old index until staged
            assert all(part == safe_filename(part) for part in path.parts), path
