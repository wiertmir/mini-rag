import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from pydantic import ValidationError

from mini_rag.evaluation import is_hit, summarize_evaluation
from mini_rag.models import (
    Chunk,
    EmbeddingManifest,
    IngestManifest,
    IngestManifestDocument,
    IngestManifestParameters,
    IngestManifestSummary,
    Question,
    RetrievalEvaluationRecord,
)
from mini_rag.storage import calculate_file_sha, save_index

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- is_hit -----------------------------------------------------------------


def test_is_hit_when_expected_document_is_retrieved() -> None:
    assert is_hit(frozenset({"A"}), iter(["B", "A"]), answerable=True) is True


def test_is_hit_multi_document_question_needs_only_one_document() -> None:
    assert is_hit(frozenset({"A", "B"}), iter(["B"]), answerable=True) is True


def test_is_hit_false_when_expected_document_is_missing() -> None:
    assert is_hit(frozenset({"A"}), iter(["B", "C", "B"]), answerable=True) is False


def test_is_hit_is_none_for_unanswerable_question() -> None:
    assert is_hit(frozenset({"NONE"}), iter(["NONE", "A"]), answerable=False) is None


def test_summarize_evaluation_counts_answerable_hits_and_misses() -> None:
    records = [
        RetrievalEvaluationRecord(id="Q1", hit=True),
        RetrievalEvaluationRecord(id="Q2", hit=False),
        RetrievalEvaluationRecord(id="Q3", hit=None),
    ]

    assert summarize_evaluation(records) == (2, 1, ["Q2"])


# --- Question ---------------------------------------------------------------


def question_dict(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": "Q01",
        "question": "What is A?",
        "reference_answer": "A is A.",
        "expected_documents": ["A"],
        "type": "simple",
        "answerable": True,
        "review_status": "draft_not_page_verified",
    }
    record.update(overrides)
    return record


def test_question_loads_every_line_of_the_dataset() -> None:
    lines = (REPO_ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [Question.model_validate_json(line) for line in lines]

    assert len(questions) == 30
    assert [q.id for q in questions if not q.answerable] == ["Q29", "Q30"]


def test_question_rejects_unanswerable_with_real_documents() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(answerable=False, expected_documents=["A"]))


def test_question_rejects_unanswerable_without_sentinel() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(answerable=False, expected_documents=[]))


def test_question_rejects_answerable_with_none_sentinel() -> None:
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(expected_documents=["NONE"]))


def test_question_rejects_answerable_without_expected_documents() -> None:
    # The validator's own message promises "non empty expected documents";
    # an answerable question with [] can never be a hit.
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(expected_documents=[]))


# --- eval/retrieval_smoke.py end to end (fake embeddings, tiny index) --------


def load_smoke_script() -> ModuleType:
    path = REPO_ROOT / "eval" / "retrieval_smoke.py"
    spec = importlib.util.spec_from_file_location("retrieval_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_chunk(document_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}_p001_c001",
        document_id=document_id,
        title="Dummy title",
        collection="test",
        source_path=Path("dummy.pdf"),
        source_url="https://example.test/dummy.pdf",
        page=1,
        page_end=1,
        text="Dummy text",
        token_count=2,
    )


class FakeClient:
    """Question text -> fixed vector; document A is [1, 0], document B is [0, 1]."""

    VECTORS: ClassVar[dict[str, list[float]]] = {"about A": [1.0, 0.0], "about B": [0.0, 1.0]}

    def __init__(self) -> None:
        self.embeddings = self
        self.calls: list[tuple[str, list[str]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def create(self, *, model: str, input: tuple[str, ...]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=self.VECTORS[text])
                for i, text in enumerate(input)
            ],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


async def run_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, FakeClient]:
    """Hand-built run: Q1 hit, Q2 miss, Q3 unanswerable; top_k = 1."""
    chunks_path = tmp_path / "chunks_600_90.jsonl"
    chunks_path.write_text(
        "".join(make_chunk(doc).model_dump_json() + "\n" for doc in ("A", "B")),
        encoding="utf-8",
    )
    ingest_manifest = IngestManifest(
        parameters=IngestManifestParameters(
            tokenizer="test",
            chunk_size=600,
            chunk_overlap=90,
            strategy="sections",
            min_chunk_tokens=0,
            skip_patterns=[],
            box_pattern=None,
        ),
        summary=IngestManifestSummary(documents=2, pages=0, chunks=2, errors=0),
        documents=[
            IngestManifestDocument(
                document_id=doc,
                source_path=f"{doc}.pdf",
                sha_256=None,
                pages=0,
                chunks=1,
                status="ok",
                error=None,
            )
            for doc in ("A", "B")
        ],
        errors=[],
    )
    (tmp_path / "ingest_manifest_600_90.json").write_text(
        ingest_manifest.model_dump_json(), encoding="utf-8"
    )
    index_dir = tmp_path / "index"
    save_index(
        index_dir,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        EmbeddingManifest(
            model="fake-model",
            dimension=2,
            records=2,
            chunks_path=chunks_path,
            chunks_sha256=calculate_file_sha(chunks_path),
        ),
    )

    questions = [
        question_dict(id="Q1", question="about A", expected_documents=["A"]),
        question_dict(id="Q2", question="about A", expected_documents=["B"]),
        question_dict(
            id="Q3",
            question="about B",
            expected_documents=["NONE"],
            type="unanswerable",
            answerable=False,
        ),
    ]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")

    client = FakeClient()
    module = load_smoke_script()
    monkeypatch.setattr(module, "AsyncOpenAI", lambda: client)

    output_dir = tmp_path / "results"
    returned = await module.eval_async(client, 1, questions_path, index_dir, output_dir)
    assert returned == json.loads((output_dir / "retrieval_smoke_summary.json").read_text())
    return output_dir, client


async def test_smoke_summary_counts_only_answerable_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = await run_smoke(tmp_path, monkeypatch)
    summary = json.loads((output_dir / "retrieval_smoke_summary.json").read_text())

    assert summary["model"] == "fake-model"
    assert summary["top_k"] == 1
    assert summary["answerable"] == 2
    assert summary["hits"] == 1
    # hit=None is neither a hit nor a miss: Q3 must not be listed as missed.
    assert summary["missed_ids"] == ["Q2"]


async def test_smoke_writes_one_record_per_question_in_file_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = await run_smoke(tmp_path, monkeypatch)
    lines = (output_dir / "retrieval_smoke.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert [r["id"] for r in records] == ["Q1", "Q2", "Q3"]
    assert [r["hit"] for r in records] == [True, False, None]
    assert all(len(r["top_k"]) == 1 for r in records)
    assert records[0]["top_k"][0]["document_id"] == "A"


async def test_smoke_embeds_all_questions_in_one_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, client = await run_smoke(tmp_path, monkeypatch)

    assert client.calls == [("fake-model", ["about A", "about A", "about B"])]


def test_question_rejects_unanswerable_with_sentinel_and_real_document() -> None:
    # Unanswerable means exactly ["NONE"]; ["NONE", "A"] is a dataset mistake.
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(answerable=False, expected_documents=["NONE", "A"]))


@pytest.mark.parametrize("documents", [["NONE", "A"], ["A", "NONE"]])
def test_question_rejects_answerable_containing_none_sentinel(documents: list[str]) -> None:
    # "NONE" is not a document id; an answerable question must not mention it at all.
    with pytest.raises(ValidationError):
        Question.model_validate(question_dict(expected_documents=documents))


async def test_smoke_cli_owns_client_and_passes_custom_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    module = load_smoke_script()
    client = FakeClient()
    constructor_calls = []

    def make_client() -> FakeClient:
        constructor_calls.append(True)
        return client

    evaluator = AsyncMock(return_value={})
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    monkeypatch.setattr(module, "eval_async", evaluator)
    questions = tmp_path / "questions.jsonl"
    index = tmp_path / "local-index"
    output = tmp_path / "local-results"
    await module.run_async(
        [
            "--questions",
            str(questions),
            "--index-dir",
            str(index),
            "--output-dir",
            str(output),
            "--top-k",
            "3",
        ]
    )
    assert constructor_calls == [True]
    evaluator.assert_awaited_once_with(client, 3, questions, index, output)


@pytest.mark.parametrize(
    "model_dir",
    [
        "qwen3-embedding_4b",
        "qwen3-embedding_8b",
        "text-embedding-3-small",
    ],
)
def test_committed_embedding_results_match_summary(model_dir: str) -> None:
    from mini_rag.models import RetrievalSmokeRecord

    directory = REPO_ROOT / "eval" / "results" / model_dir
    records = [
        RetrievalSmokeRecord.model_validate_json(line)
        for line in (directory / "retrieval_smoke.jsonl").read_text().splitlines()
    ]
    summary = json.loads((directory / "retrieval_smoke_summary.json").read_text())
    answerable, hits, missed = summarize_evaluation(records)
    assert len(records) == 30
    assert len({record.id for record in records}) == 30
    assert (answerable, hits, missed) == (
        summary["answerable"],
        summary["hits"],
        summary["missed_ids"],
    )
    assert all(len(record.top_k) == summary["top_k"] for record in records)


# --- Answer baseline (#19): offline contract/regression checks ---------------


def baseline_dependencies() -> tuple[Any, Any]:
    parameters = IngestManifestParameters(
        tokenizer="test",
        strategy="sections",
        chunk_size=600,
        chunk_overlap=90,
        min_chunk_tokens=150,
        skip_patterns=[r"^Figure"],
        box_pattern=None,
    )
    retriever = SimpleNamespace(
        embedding_manifest=EmbeddingManifest(
            model="test-embedding",
            dimension=2,
            records=2,
            chunks_path=Path("unused"),
            chunks_sha256="a" * 64,
        ),
        ingest_manifest=SimpleNamespace(parameters=parameters),
    )
    return retriever, SimpleNamespace(model_name="test-answer")


def baseline_answer(document: str = "A", *, refused: bool = False) -> Any:
    from mini_rag.models import Answer, Citation

    return Answer(
        answer="No evidence" if refused else "Answer [1]",
        citations=[]
        if refused
        else [
            Citation(
                ref=1,
                chunk_id="chunk-A",
                document_id=document,
                page=6,
                page_end=7,
            )
        ],
        insufficient_evidence=refused,
        retrieved_chunk_ids=["chunk-A", "uncited-B"],
    )


async def run_fake_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    questions: list[dict[str, Any]],
    outcomes: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    from unittest.mock import AsyncMock

    from mini_rag import evaluation

    source = tmp_path / "questions.jsonl"
    source.write_text("".join(json.dumps(q) + "\n" for q in questions))
    ask = AsyncMock(side_effect=outcomes)
    monkeypatch.setattr(evaluation, "ask_async", ask)
    retriever, generator = baseline_dependencies()
    await evaluation.baseline_evaluation_loop(
        source, tmp_path / "out", retriever, generator, 2, False
    )
    records = [
        json.loads(line) for line in (tmp_path / "out/baseline.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "out/baseline_summary.json").read_text())
    return records, summary, ask


@pytest.mark.parametrize("error", [RuntimeError("offline"), ValueError("offline")])
async def test_baseline_continues_after_error_and_stores_exception_class_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    records, summary, ask = await run_fake_baseline(
        tmp_path,
        monkeypatch,
        [question_dict(id=f"Q{i}") for i in range(3)],
        [baseline_answer(), error, baseline_answer()],
    )
    assert ask.await_count == 3
    assert [r["id"] for r in records] == ["Q0", "Q1", "Q2"]
    assert records[0]["answer"] and records[2]["answer"]
    assert summary["counts"]["errors"] == 1
    assert all(
        records[1][key] is None
        for key in (
            "answer",
            "citations",
            "insufficient_evidence",
            "retrieved_chunk_ids",
            "cites_expected_document",
        )
    )
    assert records[1]["error"] == type(error).__name__


async def test_baseline_summary_counts_hits_refusals_errors_and_all_latencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_rag.ask import InvalidCitationError
    from mini_rag.hashing import calculate_text_sha
    from mini_rag.models import AnswerEvaluationRecord, AnswerEvaluationSummary
    from mini_rag.prompts import SYSTEM_PROMPT

    questions = [question_dict(id=f"Q{i}") for i in range(6)]
    questions[1]["expected_documents"] = ["A", "B"]
    questions[5].update(answerable=False, type="unanswerable", expected_documents=["NONE"])
    records, summary, _ = await run_fake_baseline(
        tmp_path,
        monkeypatch,
        questions,
        [
            baseline_answer(),
            baseline_answer("B"),
            baseline_answer("C"),
            baseline_answer(refused=True),
            InvalidCitationError([99]),
            baseline_answer(refused=True),
        ],
    )
    assert [r["cites_expected_document"] for r in records] == [True, True, False, False, None, None]
    assert records[4]["error"] == "invalid_citation"
    assert summary["counts"] == {
        "records": 6,
        "errors": 1,
        "refusals": 2,
        "invalid_citation": 1,
        "answerable": 5,
        "answerable_citation_hits": 2,
    }
    assert records[0]["citations"][0]["page_end"] == 7
    assert records[0]["retrieved_chunk_ids"] == ["chunk-A", "uncited-B"]
    config = summary["configuration"]
    assert (config["answer_model"], config["embedding_model"], config["top_k"]) == (
        "test-answer",
        "test-embedding",
        2,
    )
    assert config["chunks_sha256"] == "a" * 64
    assert config["dataset_sha256"] == calculate_file_sha(tmp_path / "questions.jsonl")
    assert config["prompt_sha256"] == calculate_text_sha(SYSTEM_PROMPT)
    assert (
        config["ingest_parameters"]
        == baseline_dependencies()[0].ingest_manifest.parameters.model_dump()
    )
    assert config["concurrency"] == 1
    assert summary["latency_s"]["mean"] == pytest.approx(sum(r["latency_s"] for r in records) / 6)
    assert summary["latency_s"]["max"] == max(r["latency_s"] for r in records)
    AnswerEvaluationSummary.model_validate(summary)
    for record in records:
        AnswerEvaluationRecord.model_validate(record)


@pytest.mark.parametrize("existing_name", ["baseline.jsonl", "baseline_summary.json"])
async def test_baseline_preserves_existing_reference_without_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    from unittest.mock import AsyncMock

    from mini_rag import evaluation

    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(question_dict()) + "\n")
    output = tmp_path / "out"
    output.mkdir()
    reference = output / existing_name
    reference.write_text("existing reference\n")
    ask = AsyncMock(return_value=baseline_answer())
    monkeypatch.setattr(evaluation, "ask_async", ask)
    try:
        await evaluation.baseline_evaluation_loop(
            questions, output, *baseline_dependencies(), 2, False
        )
    except (FileExistsError, ValueError):
        pass
    assert reference.read_text() == "existing reference\n"
    ask.assert_not_awaited()


def test_committed_answer_baseline_matches_dataset_summary_and_prompt() -> None:
    from statistics import mean

    from mini_rag.hashing import calculate_text_sha
    from mini_rag.models import AnswerEvaluationRecord, AnswerEvaluationSummary
    from mini_rag.prompts import SYSTEM_PROMPT

    directory = REPO_ROOT / "eval/results/text-embedding-3-small/gpt-5.4-mini"
    records = [
        AnswerEvaluationRecord.model_validate_json(line)
        for line in (directory / "baseline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = AnswerEvaluationSummary.model_validate_json(
        (directory / "baseline_summary.json").read_text(encoding="utf-8")
    )
    questions = [
        Question.model_validate_json(line)
        for line in (REPO_ROOT / "data/questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 30
    assert [r.id for r in records] == [q.id for q in questions]
    for record, question in zip(records, questions, strict=True):
        assert (
            record.question,
            record.reference_answer,
            record.expected_documents,
            record.type,
        ) == (
            question.question,
            question.reference_answer,
            question.expected_documents,
            question.type,
        )
        assert record.cites_expected_document == (
            None
            if not question.answerable or record.error
            else any(c.document_id in question.expected_documents for c in record.citations or [])
        )
        if question.id in {"Q29", "Q30"}:
            assert record.insufficient_evidence is True
    assert summary.counts.model_dump() == {
        "records": len(records),
        "errors": sum(r.error is not None for r in records),
        "refusals": sum(r.error is None and r.insufficient_evidence is True for r in records),
        "invalid_citation": sum(r.error == "invalid_citation" for r in records),
        "answerable": sum(q.answerable for q in questions),
        "answerable_citation_hits": sum(
            r.error is None and r.cites_expected_document is True for r in records
        ),
    }
    assert summary.counts.invalid_citation == 0
    # The committed run used CRLF; Git stores LF. Accept only a line-ending
    # change to this exact dataset, never changed questions or labels.
    import hashlib

    dataset_bytes = (REPO_ROOT / "data/questions.jsonl").read_bytes()
    assert summary.configuration.dataset_sha256 in {
        hashlib.sha256(dataset_bytes).hexdigest(),
        hashlib.sha256(dataset_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest(),
    }
    assert summary.configuration.prompt_sha256 == calculate_text_sha(SYSTEM_PROMPT)
    assert summary.latency_s.mean == pytest.approx(mean(r.latency_s for r in records))
    assert summary.latency_s.max == max(r.latency_s for r in records)


async def test_baseline_cli_loads_index_once_and_reuses_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from mini_rag import evaluation
    from mini_rag.llm import OpenAIRetriever

    # Reuse the real tiny-index fixture; its smoke call is offline too.
    _, client = await run_smoke(tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "baseline_script", REPO_ROOT / "eval/run_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructors = []
    retrievers: list[Any] = []

    def make_client() -> FakeClient:
        constructors.append(True)
        return client

    def make_retriever(index_dir: Path, sdk: Any) -> Any:
        retriever = OpenAIRetriever(index_dir, sdk)
        retriever.load_index = AsyncMock(wraps=retriever.load_index)  # type: ignore[method-assign]
        retrievers.append(retriever)
        return retriever

    ask = AsyncMock(return_value=baseline_answer())
    monkeypatch.setattr(evaluation, "ask_async", ask)
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    monkeypatch.setattr(module, "OpenAIRetriever", make_retriever)
    output = tmp_path / "baseline-out"
    await module._run_async(
        [
            "--questions",
            str(tmp_path / "questions.jsonl"),
            "--index-dir",
            str(tmp_path / "index"),
            "--output-dir",
            str(output),
            "--model",
            "offline-answer",
            "--top-k",
            "2",
        ]
    )
    assert constructors == [True]
    assert len(retrievers) == 1
    retrievers[0].load_index.assert_awaited_once()
    assert ask.await_count == 3
    assert all(call.args[1] is retrievers[0] and call.args[3] == 2 for call in ask.await_args_list)
    assert all(call.args[2] is ask.await_args_list[0].args[2] for call in ask.await_args_list)
    summary = json.loads((output / "baseline_summary.json").read_text())
    assert summary["configuration"]["answer_model"] == "offline-answer"
    assert summary["configuration"]["embedding_model"] == "fake-model"


async def test_baseline_cli_missing_index_aborts_before_question_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    spec = importlib.util.spec_from_file_location(
        "baseline_missing_index", REPO_ROOT / "eval/run_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = AsyncMock()
    monkeypatch.setattr(module, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(module, "baseline_evaluation_loop", evaluator)
    with pytest.raises(FileNotFoundError):
        await module._run_async(
            [
                "--questions",
                str(tmp_path / "questions.jsonl"),
                "--index-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    evaluator.assert_not_awaited()
    assert not (tmp_path / "out").exists()


async def test_baseline_cli_false_does_not_select_local_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    spec = importlib.util.spec_from_file_location(
        "baseline_local_flag", REPO_ROOT / "eval/run_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client_options: list[dict[str, Any]] = []

    def make_client(**kwargs: Any) -> FakeClient:
        client_options.append(kwargs)
        return FakeClient()

    retriever, _ = baseline_dependencies()
    retriever.load_index = AsyncMock(return_value=retriever.embedding_manifest)
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    monkeypatch.setattr(module, "OpenAIRetriever", lambda *_: retriever)
    monkeypatch.setattr(module, "baseline_evaluation_loop", AsyncMock())
    # Rejecting a value is also fine if the CLI changes to a bare boolean flag.
    try:
        await module._run_async(
            [
                "--questions",
                str(tmp_path / "questions.jsonl"),
                "--local-ollama",
                "False",
            ]
        )
    except SystemExit as exc:
        assert exc.code != 0
        assert not client_options
        return
    assert client_options == [{}]


async def test_baseline_force_replaces_both_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from mini_rag import evaluation

    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(question_dict()) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    for name in ("baseline.jsonl", "baseline_summary.json"):
        (output / name).write_text("old reference", encoding="utf-8")
    ask = AsyncMock(return_value=baseline_answer())
    monkeypatch.setattr(evaluation, "ask_async", ask)
    await evaluation.baseline_evaluation_loop(
        questions,
        output,
        *baseline_dependencies(),
        2,
        True,
    )
    ask.assert_awaited_once()
    records = (output / "baseline.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["id"] == "Q01"
    assert (
        json.loads((output / "baseline_summary.json").read_text(encoding="utf-8"))["counts"][
            "records"
        ]
        == 1
    )


async def test_baseline_force_removes_old_summary_before_interrupted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from mini_rag import evaluation

    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(question_dict()) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    summary = output / "baseline_summary.json"
    summary.write_text("old completed run", encoding="utf-8")
    monkeypatch.setattr(evaluation, "ask_async", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await evaluation.baseline_evaluation_loop(
            questions,
            output,
            *baseline_dependencies(),
            2,
            True,
        )
    assert not summary.exists()


async def test_baseline_cli_passes_force_and_selects_local_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    spec = importlib.util.spec_from_file_location(
        "baseline_force_cli", REPO_ROOT / "eval/run_baseline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options: list[dict[str, Any]] = []

    def make_client(**kwargs: Any) -> FakeClient:
        options.append(kwargs)
        return FakeClient()

    retriever, _ = baseline_dependencies()
    retriever.load_index = AsyncMock(return_value=retriever.embedding_manifest)
    evaluator = AsyncMock()
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    monkeypatch.setattr(module, "OpenAIRetriever", lambda *_: retriever)
    monkeypatch.setattr(module, "baseline_evaluation_loop", evaluator)
    await module._run_async(
        [
            "--questions",
            str(tmp_path / "questions.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--local-ollama",
            "--force",
        ]
    )
    assert options == [{"base_url": "http://localhost:11434/v1", "api_key": "ollama"}]
    assert evaluator.await_args is not None
    assert evaluator.await_args.args[-1] is True
