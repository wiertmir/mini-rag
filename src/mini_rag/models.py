from datetime import datetime
from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

class EmbeddingManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = Field(description="Embeddings model")
    dimension: int = Field(gt=0, description="Length of embeddings vector")
    records: int = Field(ge=0, description="Number of embedding vectors in index file")
    chunks_path: Path = Field(description="Which file the index was build from")
    chunks_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$", description="Detects a stale index after re-ingest")
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="Creation time with zone offset")


class Manifest(BaseModel):
    id: str = Field(description="Id of the whole source document")
    title: str = Field(description="Document title")
    url: str = Field(description="Url of the source")
    year: str = Field(description="Publish year")
    collection: str = Field(description="The bigger set document belongs to. Usefull for filtering")


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    collection: str
    source_path: Path
    source_url: str
    page: int
    text: str
    token_count: int
