from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_SCORE_ROUNDING_TOLERANCE = 1e-4
_NO_DOCUMENT_SENTINEL = "NONE"


class QuestionType(Enum):
    DEFINITION = "definition"
    MULTI_DOCUMENT = "multi_document"
    SIMPLE = "simple"
    UNANSWERABLE = "unanswerable"


class ReviewStatusType(Enum):
    DRAFT_NOT_PAGE_VERIFIED = "draft_not_page_verified"


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Question id")
    question: str = Field(description="Question text")
    reference_answer: str = Field(description="Answer used as reference")
    expected_documents: list[str] = Field(description="Which document contains answer")
    type: QuestionType = Field(description="Question type (simple, definition, multidocument)")
    answerable: bool = Field(description="Based on data we gather is it possible to answer it?")
    review_status: ReviewStatusType = Field(description="Review status")

    @model_validator(mode="after")
    def enforce_answerable_docs_constraint(self) -> "Question":
        if not self.answerable and self.expected_documents != [_NO_DOCUMENT_SENTINEL]:
            raise ValueError("Non answerable question can't have any expected documents")

        if self.answerable and (
            _NO_DOCUMENT_SENTINEL in self.expected_documents or not self.expected_documents
        ):
            raise ValueError("Answerable question must have non empty expected documents")

        return self


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Question id")
    hit: bool | None = Field(description="Whether the question was answered by retrieval")


class RetrievedChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    chunk_id: str
    document_id: str
    page: int = Field(ge=1)
    page_end: int | None = Field(
        default=None, ge=1, description="Last page of a multi-page chunk; absent in v1 results"
    )


class RetrievalSmokeRecord(EvaluationRecord):
    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    expected_documents: list[str]
    top_k: list[RetrievedChunkRecord]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        pattern=r"^.+_p\d{3,}_c\d{3,}$",
        description=(
            "Stable ID in the form {document_id}_p{page:03d}_c{chunk:03d}; "
            "re-ingesting the same input produces the same ID"
        ),
    )
    document_id: str = Field(
        min_length=1,
        description="Source document ID from the manifest, shared by its chunks",
    )
    title: str = Field(min_length=1, description="Manifest document title, copied into every chunk")
    collection: str = Field(
        min_length=1,
        description="Collection the source document belongs to, used for filtering",
    )
    source_path: Path = Field(description="Local PDF path the chunk was cut from")
    source_url: str = Field(min_length=1, description="Original online document URL for citations")
    page: int = Field(ge=1, description="1-based page the chunk starts on")
    page_end: int = Field(ge=1, description="1-based page the chunk ends on (>= page)")
    text: str = Field(min_length=1, description="Chunk text that is embedded and returned to users")
    token_count: int = Field(
        ge=0,
        description="Token count using the chunking tokenizer for size and overlap",
    )

    @model_validator(mode="before")
    @classmethod
    def default_page_end(cls, data: Any) -> Any:
        """Chunk files from before section chunking are single-page: page_end = page."""
        if isinstance(data, dict) and "page_end" not in data and "page" in data:
            return {**data, "page_end": data["page"]}
        return data

    @model_validator(mode="after")
    def check_page_range(self) -> Self:
        if self.page_end < self.page:
            raise ValueError(f"page_end {self.page_end} is before page {self.page}")
        return self

    @property
    def pages(self) -> str:
        """Human-readable page or page range, e.g. "7" or "7-9"."""
        return str(self.page) if self.page == self.page_end else f"{self.page}-{self.page_end}"


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, description="1-based position in the result list (rank 1 = best)")
    score: float = Field(
        ge=-1 - _SCORE_ROUNDING_TOLERANCE,
        le=1 + _SCORE_ROUNDING_TOLERANCE,
        description=("Cosine similarity in [-1, 1]; overshoots up to 0.0001 are clamped"),
    )
    chunk: Chunk = Field(description="The full chunk (carries chunk_id, document_id, page, text)")

    @field_validator("score")
    @classmethod
    def clamp_score_rounding_error(cls, score: float) -> float:
        return min(1.0, max(-1.0, score))


class EmbeddingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Embeddings model")
    dimension: int = Field(gt=0, description="Length of embeddings vector")
    records: int = Field(ge=0, description="Number of embedding vectors in index file")
    chunks_path: Path = Field(description="Which file the index was build from")
    chunks_sha256: str = Field(
        pattern=r"^[0-9a-fA-F]{64}$",
        description="Detects a stale index after re-ingest",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="Creation time with zone offset",
    )


class Manifest(BaseModel):
    id: str = Field(description="Id of the whole source document")
    title: str = Field(description="Document title")
    url: str = Field(description="Url of the source")
    year: str = Field(description="Publish year")
    collection: str = Field(description="The bigger set document belongs to. Usefull for filtering")
