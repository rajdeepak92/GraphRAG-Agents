"""Strict Pydantic contracts used by ingestion and generated artifacts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_SOURCE_REQUIREMENT_ID_RE = re.compile(
    r"\b(?:BR|AC|FR|NFR)\s*-\s*[A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)*\b",
    re.I,
)
_PLACEHOLDER_REQUIREMENT_TEXT_RE = re.compile(
    r"^(?:requirement|business requirement|acceptance criteria|functional "
    r"requirement|non-functional requirement|placeholder|tbd|n/?a|none)$",
    re.I,
)


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

    @field_validator("quote")
    @classmethod
    def quote_is_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("quote must not be empty")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> SourceTrace:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


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


class LLMRequirementCandidate(StrictModel):
    temp_id: str
    statement: str
    requirement_type: str = "functional"
    priority: str = "medium"
    requirement_key: str | None = None
    source_trace: SourceTrace


class LLMFactCandidate(StrictModel):
    temp_id: str
    text: str
    source_trace: SourceTrace
    requirements: list[LLMRequirementCandidate] = Field(default_factory=list)


class LLMDiscoveredRequirement(StrictModel):
    req_id: str
    req_text: str
    requirement_type: str = "Functional Requirement"
    priority: str = "Medium"
    requirement_key: str | None = None

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> str:
        if value is None:
            return "Medium"

        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return "Medium"

        normalized = text.lower()
        if normalized in {"high", "critical", "mandatory", "must", "required"}:
            return "High"
        if normalized in {"medium", "med", "normal", "default"}:
            return "Medium"
        if normalized in {"low", "optional", "future", "nice-to-have", "nice to have"}:
            return "Low"

        return "Medium"

    @field_validator("requirement_type", mode="before")
    @classmethod
    def normalize_requirement_type(cls, value: object) -> str:
        if value is None:
            return "Functional Requirement"

        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return "Functional Requirement"

        normalized = text.lower()
        if normalized in {"functional", "fr", "functional requirement"}:
            return "Functional Requirement"
        if normalized in {"business", "br", "business requirement"}:
            return "Business Requirement"
        if normalized in {"acceptance", "ac", "acceptance criteria", "acceptance criterion"}:
            return "Acceptance Criteria"
        if normalized in {"non-functional", "non functional", "nfr", "non-functional requirement"}:
            return "Non-Functional Requirement"
        if normalized in {"security", "security requirement"}:
            return "Security Requirement"
        if normalized in {"configuration", "configuration requirement"}:
            return "Configuration Requirement"
        if normalized in {"validation", "validation requirement"}:
            return "Validation Requirement"
        if normalized in {"alerting", "alerting requirement"}:
            return "Alerting Requirement"
        if normalized in {"health", "health requirement"}:
            return "Health Requirement"
        if normalized in {"data quality", "data quality requirement"}:
            return "Data Quality Requirement"
        if normalized in {"application", "application requirement"}:
            return "Application Requirement"
        if normalized in {"offline", "offline requirement"}:
            return "Offline Requirement"

        return text


class LLMDiscoveredFact(StrictModel):
    fact_id: str
    fact_text: str
    quote: str
    requirements: list[LLMDiscoveredRequirement] = Field(default_factory=list)

    @field_validator("fact_id", "fact_text", "quote")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class RequirementDiscoveryChunkOutput(StrictModel):
    facts: list[LLMDiscoveredFact] = Field(default_factory=list)


class LLMChunkExtraction(StrictModel):
    chunk_id: str
    facts: list[LLMFactCandidate] = Field(default_factory=list)


class RequirementDiscoveryOutput(StrictModel):
    chunks: list[LLMChunkExtraction] = Field(default_factory=list)


class CanonicalFact(StrictModel):
    canonical_fact_id: str
    normalized_text: str
    representative_text: str


class VerifiedFact(StrictModel):
    fact_id: str
    canonical_fact_id: str = ""
    text: str
    source_trace: SourceTrace


class RequirementEvidence(StrictModel):
    evidence_id: str
    fact_ids: list[str]
    source_trace: SourceTrace


class VerifiedRequirement(StrictModel):
    requirement_id: str
    revision_id: str = ""
    requirement_key: str = ""
    statement: str
    normalized_statement: str = ""
    requirement_type: str
    priority: str
    status: Literal["active", "superseded"] = "active"
    fact_ids: list[str]
    source_trace: SourceTrace
    evidence: list[RequirementEvidence] = Field(default_factory=list)


class RequirementDeltaEvent(StrictModel):
    event_id: str
    event_type: Literal["new", "duplicate", "changed", "superseded"]
    requirement_id: str
    revision_id: str | None = None
    previous_revision_id: str | None = None
    superseded_by_revision_id: str | None = None
    document_version_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    impacted_artifact_types: list[str] = Field(default_factory=list)


class RequirementRevisionSnapshot(StrictModel):
    requirement_id: str
    revision_id: str
    statement: str
    normalized_statement: str


class RequirementArtifact(StrictModel):
    artifact_schema_version: Literal["1.0", "2.0"] = "2.0"
    project: str
    document_id: str
    document_version_id: str
    version: str
    source_path: str = ""
    source_checksum: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    canonical_facts: list[CanonicalFact] = Field(default_factory=list)
    facts: list[VerifiedFact]
    requirements: list[VerifiedRequirement]
    delta_events: list[RequirementDeltaEvent] = Field(default_factory=list)


class CompactRequirementOccurrence(StrictModel):
    chunk_id: str
    fact_id: str
    requirement_text: str
    requirement_type: str
    priority: Literal["High", "Medium", "Low"]
    status: Literal["Active", "Superseded"]
    doc_version: str


class CompactRequirementArtifact(StrictModel):
    artifact_schema_version: Literal["3.0-compact"] = "3.0-compact"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requirements: dict[str, list[CompactRequirementOccurrence]]


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
    full_artifact_path: Path | None = None
    chunk_ids: list[str]
    fact_ids: list[str]
    requirement_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
