"""Chunk and chunk manifest contracts."""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Chunk(BaseModel):
    """A deterministic, source-traceable chunk produced from parsed document blocks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_version_id: UUID
    ordinal: int = Field(ge=1)

    source_checksum: str = Field(min_length=64, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)

    normalized_text: str = Field(min_length=1)
    raw_text: str = Field(
        min_length=1,
        validation_alias=AliasChoices("raw_text", "original_text"),
    )

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()

    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)

    @property
    def original_text(self) -> str:
        """Backward-compatible alias for older code that used original_text."""
        return self.raw_text

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
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
    """Lightweight manifest projection for location-only chunk references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()

    content_hash: str = Field(min_length=64, max_length=64)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
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


class ChunkManifest(BaseModel):
    """Validated Phase 8 manifest containing deterministic chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_schema_version: str = "1.0"
    manifest_id: UUID | None = None

    document_version_id: UUID
    source_checksum: str = Field(min_length=64, max_length=64)

    parser_fingerprint: str = Field(min_length=1)
    chunker_fingerprint: str = Field(min_length=1)

    chunks: tuple[Chunk, ...] = Field(min_length=1)

    @property
    def entries(self) -> tuple[ChunkManifestEntry, ...]:
        """Return a lightweight location-only projection of manifest chunks."""
        return tuple(
            ChunkManifestEntry(
                chunk_id=chunk.chunk_id,
                ordinal=chunk.ordinal,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                content_hash=chunk.content_hash,
                character_start=chunk.character_start,
                character_end=chunk.character_end,
            )
            for chunk in self.chunks
        )

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> Self:
        seen_ordinals: set[int] = set()
        seen_chunk_ids: set[str] = set()

        for chunk in self.chunks:
            if chunk.document_version_id != self.document_version_id:
                msg = "All chunks must use the same document_version_id as the manifest."
                raise ValueError(msg)

            if chunk.source_checksum != self.source_checksum:
                msg = "All chunks must use the same source_checksum as the manifest."
                raise ValueError(msg)

            if chunk.ordinal in seen_ordinals:
                msg = f"Duplicate chunk ordinal detected: {chunk.ordinal}."
                raise ValueError(msg)

            if chunk.chunk_id in seen_chunk_ids:
                msg = f"Duplicate chunk_id detected: {chunk.chunk_id}."
                raise ValueError(msg)

            seen_ordinals.add(chunk.ordinal)
            seen_chunk_ids.add(chunk.chunk_id)

        expected_ordinals = set(range(1, len(self.chunks) + 1))
        if seen_ordinals != expected_ordinals:
            msg = "Chunk ordinals must be contiguous and start at 1."
            raise ValueError(msg)

        return self
