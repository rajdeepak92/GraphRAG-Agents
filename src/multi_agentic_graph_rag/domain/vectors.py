"""Vector indexing domain contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VectorRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    ordinal: int = Field(ge=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    embedding_fingerprint: str = Field(min_length=1)


class VectorSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_version_id: str
    text: str
    distance: float
    metadata: dict[str, str | int | float | bool | None]
