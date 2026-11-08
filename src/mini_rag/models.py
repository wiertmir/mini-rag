from pydantic import BaseModel, Field


class Manifest(BaseModel):
    id: str = Field(..., description="Id of the whole source document")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Url of the source")
    year: str = Field(..., description="Publish year")
    collection: str = Field(
        ...,
        description="The bigger set document belongs to. Usefull for filtering",
    )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    collection: str
    source_path: str
    source_url: str
    page: int
    text: str
    token_count: int
