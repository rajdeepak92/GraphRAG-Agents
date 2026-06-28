"""Application command contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from multi_agentic_graph_rag.domain.enums import ReplacePolicy
from multi_agentic_graph_rag.domain.identifiers import normalize_version


class ProviderOverrides(BaseModel):
    """Optional provider overrides supplied by CLI or API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    vector_store_provider: str | None = None
    graph_store_provider: str | None = None


class IngestDocumentCommand(BaseModel):
    """Command to ingest one logical document version."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    project_key: str = Field(min_length=1)
    document_path: Path
    document_version: str = Field(min_length=1)
    logical_document_name: str | None = None
    provider_overrides: ProviderOverrides = Field(default_factory=ProviderOverrides)
    replace_policy: ReplacePolicy = ReplacePolicy.REJECT
    dry_run: bool = False

    @field_validator("project_key")
    @classmethod
    def normalize_project_key(cls, value: str) -> str:
        normalized = value.strip().upper().replace(" ", "_")
        if not normalized:
            msg = "project_key cannot be empty."
            raise ValueError(msg)
        return normalized

    @field_validator("document_version")
    @classmethod
    def validate_document_version(cls, value: str) -> str:
        normalize_version(value)
        return value.strip()

    @field_validator("logical_document_name")
    @classmethod
    def normalize_logical_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.strip().split())
        return normalized or None


class ResumeRunCommand(BaseModel):
    """Command to resume an existing ingestion run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)


class RebuildProjectionCommand(BaseModel):
    """Command to rebuild one projection from canonical records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_key: str = Field(min_length=1)
    document_version: str | None = None
    projection: str = Field(pattern=r"^(neo4j|chroma)$")


class VerifyArtifactCommand(BaseModel):
    """Command to verify an emitted requirement artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    artifact_path: Path
