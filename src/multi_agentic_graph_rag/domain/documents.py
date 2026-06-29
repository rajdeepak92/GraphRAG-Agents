"""Document domain contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    """A source-traceable parsed text block produced by a document parser."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(min_length=1)
    source_checksum: str = Field(min_length=64, max_length=64)

    page_number: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = ()
    paragraph_number: int | None = Field(default=None, ge=1)

    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)

    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)

    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.character_end <= self.character_start:
            msg = "character_end must be greater than character_start."
            raise ValueError(msg)

        return self


class ParsedDocument(BaseModel):
    """Normalized parser output for any supported source document type.

    This model intentionally does not require document_version_id.
    Parsers should parse files. The ingestion workflow or manifest builder
    binds parser output to a canonical document version later.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(min_length=1)
    source_checksum: str = Field(min_length=64, max_length=64)

    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_fingerprint: str | None = None

    blocks: tuple[ParsedBlock, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def populate_parser_fingerprint(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get("parser_fingerprint"):
            return data

        parser_name = data.get("parser_name")
        parser_version = data.get("parser_version")

        if parser_name and parser_version:
            payload = {
                "parser_name": parser_name,
                "parser_version": parser_version,
            }
            serialized = json.dumps(payload, sort_keys=True)
            data["parser_fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return data

    @model_validator(mode="after")
    def validate_block_consistency(self) -> Self:
        for block in self.blocks:
            if block.source_checksum != self.source_checksum:
                msg = "All parsed blocks must use the same source_checksum as the parsed document."
                raise ValueError(msg)

            if block.source_path != self.source_path:
                msg = "All parsed blocks must use the same source_path as the parsed document."
                raise ValueError(msg)

            if block.parser_name != self.parser_name:
                msg = "All parsed blocks must use the same parser_name as the parsed document."
                raise ValueError(msg)

            if block.parser_version != self.parser_version:
                msg = "All parsed blocks must use the same parser_version as the parsed document."
                raise ValueError(msg)

        return self
