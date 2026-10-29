from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    collection: str
    source_url: str
    page: int
    text: str
    token_count: int
