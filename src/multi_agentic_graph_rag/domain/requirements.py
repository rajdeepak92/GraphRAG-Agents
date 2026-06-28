"""Requirement discovery and artifact contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from multi_agentic_graph_rag.domain.enums import (
    DerivationType,
    RequirementType,
    ValidationStatus,
)
from multi_agentic_graph_rag.domain.evidence import EvidenceReference
from multi_agentic_graph_rag.domain.facts import FactCandidate
from multi_agentic_graph_rag.domain.identifiers import validate_public_id, validate_temporary_key


class RequirementCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temporary_requirement_key: str
    statement: str = Field(min_length=1)
    requirement_type: RequirementType
    derivation_type: DerivationType
    validation_status: ValidationStatus = ValidationStatus.CANDIDATE
    supporting_fact_keys: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...] = ()

    @field_validator("temporary_requirement_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_temporary_key(value, prefix="R")

    @field_validator("supporting_fact_keys")
    @classmethod
    def validate_supporting_keys(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            msg = "RequirementCandidate must reference at least one fact key."
            raise ValueError(msg)

        return tuple(validate_temporary_key(value, prefix="F") for value in values)


class Requirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement_pk: UUID
    requirement_id: str
    statement: str = Field(min_length=1)
    requirement_type: RequirementType
    derivation_type: DerivationType
    validation_status: ValidationStatus
    supporting_fact_ids: tuple[str, ...]
    evidence: tuple[EvidenceReference, ...]

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        return validate_public_id(value, prefix="REQ")

    @model_validator(mode="after")
    def validate_traceability(self) -> Requirement:
        if not self.supporting_fact_ids:
            msg = "Requirement must have at least one supporting fact."
            raise ValueError(msg)

        if not self.evidence:
            msg = "Requirement must have at least one evidence reference."
            raise ValueError(msg)

        return self


class SourceTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_id: str
    fact_type: str
    chunk_id: str
    document_id: str
    exact_quote: str


class RequirementArtifactRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requirement_id: str
    requirement_content: str
    requirement_type: RequirementType
    derivation_type: DerivationType
    validation_status: ValidationStatus
    source_trace: tuple[SourceTrace, ...]

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        return validate_public_id(value, prefix="REQ")


class RequirementArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    project_id: str
    document_id: str
    document_version: str
    source_checksum: str
    pipeline_fingerprint: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_requirements: int = Field(ge=0)
    artifact_checksum: str | None = None
    requirements: tuple[RequirementArtifactRecord, ...]

    @model_validator(mode="after")
    def validate_total(self) -> RequirementArtifact:
        if self.total_requirements != len(self.requirements):
            msg = "total_requirements must match requirements length."
            raise ValueError(msg)

        return self


class DiscoveryBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: UUID
    document_version_id: UUID
    chunk_ids: tuple[str, ...]
    token_budget: int = Field(gt=0)

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            msg = "DiscoveryBatch must contain at least one chunk id."
            raise ValueError(msg)
        return values


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: UUID
    facts: tuple[FactCandidate, ...]
    requirements: tuple[RequirementCandidate, ...]
    provider_name: str
    provider_model: str | None = None
    prompt_fingerprint: str
    latency_ms: int | None = Field(default=None, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
