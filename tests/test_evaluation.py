import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from pydantic import ValidationError

from mini_rag.evaluation import is_hit, summarize_evaluation
from mini_rag.models import Chunk, EmbeddingManifest, EvaluationRecord, Question
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
        EvaluationRecord(id="Q1", hit=True),
        EvaluationRecord(id="Q2", hit=False),
        EvaluationRecord(id="Q3", hit=None),
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

    async def __aexit__(self, *exc: object) -> None:
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
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(make_chunk(doc).model_dump_json() + "\n" for doc in ("A", "B")),
        encoding="utf-8",
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
    await module.eval_async(1, questions_path, index_dir, output_dir)
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
