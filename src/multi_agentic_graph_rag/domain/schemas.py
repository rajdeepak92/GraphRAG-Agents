"""Strict Pydantic contracts used by ingestion and generated artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestionRequest(StrictModel):
    project: str
    document: Path
    version: str
    logical_name: str | None = None
    replace_version: bool = False
    reasoning_provider: str | None = None
    embedding_provider: str | None = None

    @field_validator("project", "version")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class SourceTrace(StrictModel):
    chunk_id: str
    quote: str
    start_char: int
    end_char: int
    page: int | None = None
    section: str | None = None


class ParsedBlock(StrictModel):
    block_id: str
    original_text: str
    normalized_text: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(StrictModel):
    chunk_id: str
    ordinal: int
    text: str
    normalized_text: str
    page: int | None = None
    section: str | None = None
    start_char: int
    end_char: int
    source_block_ids: list[str]


class DocumentManifest(StrictModel):
    project: str
    document_id: str
    document_version_id: str
    logical_name: str
    version: str
    source_path: str
    source_checksum: str
    parser_fingerprint: str
    chunker_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunks: list[DocumentChunk]


class LLMFactCandidate(StrictModel):
    temp_id: str
    text: str
    source_trace: SourceTrace


class LLMRequirementCandidate(StrictModel):
    temp_id: str
    statement: str
    requirement_type: str = "functional"
    priority: str = "medium"
    fact_temp_ids: list[str]
    source_trace: SourceTrace


class RequirementDiscoveryOutput(StrictModel):
    facts: list[LLMFactCandidate]
    requirements: list[LLMRequirementCandidate]


class VerifiedFact(StrictModel):
    fact_id: str
    text: str
    source_trace: SourceTrace


class VerifiedRequirement(StrictModel):
    requirement_id: str
    statement: str
    requirement_type: str
    priority: str
    fact_ids: list[str]
    source_trace: SourceTrace


class RequirementArtifact(StrictModel):
    artifact_schema_version: Literal["1.0"] = "1.0"
    project: str
    document_id: str
    document_version_id: str
    version: str
    source_checksum: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    facts: list[VerifiedFact]
    requirements: list[VerifiedRequirement]


class IngestionResult(StrictModel):
    run_id: str
    status: str
    project: str
    version: str
    document_id: str
    document_version_id: str
    checksum: str
    manifest_path: Path
    artifact_path: Path
    chunk_ids: list[str]
    fact_ids: list[str]
    requirement_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
