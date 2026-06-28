"""Fact discovery contracts."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from multi_agentic_graph_rag.domain.enums import FactType, ValidationStatus
from multi_agentic_graph_rag.domain.evidence import EvidenceReference
from multi_agentic_graph_rag.domain.identifiers import validate_public_id, validate_temporary_key


class FactCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temporary_fact_key: str
    statement: str = Field(min_length=1)
    fact_type: FactType = FactType.UNKNOWN
    evidence: tuple[EvidenceReference, ...]
    validation_status: ValidationStatus = ValidationStatus.CANDIDATE

    @field_validator("temporary_fact_key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_temporary_key(value, prefix="F")


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_pk: UUID
    fact_id: str
    statement: str = Field(min_length=1)
    fact_type: FactType
    evidence: tuple[EvidenceReference, ...]
    validation_status: ValidationStatus

    @field_validator("fact_id")
    @classmethod
    def validate_fact_id(cls, value: str) -> str:
        return validate_public_id(value, prefix="FACT")
