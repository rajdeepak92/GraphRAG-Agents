"""Document domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from multi_agentic_graph_rag.domain.identifiers import normalize_version


class Project(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: UUID
    project_key: str = Field(min_length=1)
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: UUID
    project_id: UUID
    logical_document_name: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DocumentVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_version_id: UUID
    document_id: UUID
    supplied_version: str = Field(min_length=1)
    normalized_version: str
    source_checksum: str = Field(min_length=64, max_length=64)
    supersedes_document_version_id: UUID | None = None
    parser_fingerprint: str | None = None
    chunker_fingerprint: str | None = None
    embedding_fingerprint: str | None = None
    prompt_fingerprint: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("normalized_version")
    @classmethod
    def validate_normalized_version(cls, value: str) -> str:
        return normalize_version(value)


class ParsedBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source_path: Path
    source_checksum: str
    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    paragraph_number: int | None = Field(default=None, ge=1)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)
    raw_text: str
    normalized_text: str
    parser_name: str
    parser_version: str


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_version_id: UUID
    source_checksum: str
    parser_fingerprint: str
    blocks: tuple[ParsedBlock, ...]
