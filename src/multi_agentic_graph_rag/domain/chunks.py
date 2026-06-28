"""Chunk and chunk manifest contracts."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_version_id: UUID
    ordinal: int = Field(ge=1)
    content_hash: str = Field(min_length=64, max_length=64)
    normalized_text: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Chunk:
        if self.character_end <= self.character_start:
            msg = "character_end must be greater than character_start."
            raise ValueError(msg)

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            msg = "page_end cannot be less than page_start."
            raise ValueError(msg)

        return self


class ChunkManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    ordinal: int = Field(ge=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    content_hash: str
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)


class ChunkManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_schema_version: str = "1.0"
    manifest_id: UUID
    document_version_id: UUID
    source_checksum: str
    parser_fingerprint: str
    chunker_fingerprint: str
    chunks: tuple[ChunkManifestEntry, ...]
