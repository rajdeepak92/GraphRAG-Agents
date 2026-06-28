"""SQLAlchemy ORM models for the PostgreSQL control plane.

These classes are infrastructure models. They represent PostgreSQL rows and
must not be imported by the domain package.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from multi_agentic_graph_rag.domain.enums import (
    DerivationType,
    FactType,
    RequirementType,
    RunStatus,
    RunStepName,
    ValidationStatus,
)

RUN_STATUS_VALUES = tuple(status.value for status in RunStatus)
RUN_STEP_NAME_VALUES = tuple(step_name.value for step_name in RunStepName)
FACT_TYPE_VALUES = tuple(fact_type.value for fact_type in FactType)
REQUIREMENT_TYPE_VALUES = tuple(requirement_type.value for requirement_type in RequirementType)
DERIVATION_TYPE_VALUES = tuple(derivation_type.value for derivation_type in DerivationType)
VALIDATION_STATUS_VALUES = tuple(status.value for status in ValidationStatus)
DISCOVERY_RUN_STATUS_VALUES = (
    "pending",
    "running",
    "completed",
    "failed",
)

ARTIFACT_STATUS_VALUES = (
    "pending",
    "ready",
    "failed",
)

PROJECTION_JOB_STATUS_VALUES = (
    "pending",
    "running",
    "ready",
    "failed",
)

OUTBOX_EVENT_STATUS_VALUES = (
    "pending",
    "processing",
    "processed",
    "failed",
)


def _uuid_pk() -> UUID:
    """Generate UUID primary keys in Python."""
    return uuid4()


def _sql_in_values(values: tuple[str, ...]) -> str:
    """Render a safe SQL IN value tuple for fixed enum values."""
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


class Base(DeclarativeBase):
    """Base class for all PostgreSQL ORM rows."""


class ProjectRow(Base):
    """Canonical project registry."""

    __tablename__ = "projects"

    project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    project_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )
    name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DocumentRow(Base):
    """Logical document inside a project."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "logical_document_name",
            name="uq_documents_project_logical_name",
        ),
    )

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.project_id", ondelete="RESTRICT"),
        nullable=False,
    )
    logical_document_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DocumentVersionRow(Base):
    """Immutable version of a logical document."""

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "supplied_version",
            name="uq_document_versions_document_supplied_version",
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="RESTRICT"),
        nullable=False,
    )
    supplied_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    normalized_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    source_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    supersedes_document_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="SET NULL"),
        nullable=True,
    )
    parser_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    chunker_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    embedding_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    prompt_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SourceFileRow(Base):
    """Registered physical source file for a document version."""

    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "source_checksum",
            name="uq_source_files_document_version_checksum",
        ),
    )

    source_file_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    source_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    source_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    byte_size: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    media_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class IngestionRunRow(Base):
    """Canonical ingestion run state."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_sql_in_values(RUN_STATUS_VALUES)}",
            name="ck_ingestion_runs_status",
        ),
    )

    run_pk: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )
    project_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("projects.project_id", ondelete="RESTRICT"),
        nullable=True,
    )
    document_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="RESTRICT"),
        nullable=True,
    )
    document_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RunStatus.REQUESTED.value,
    )
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class RunStepRow(Base):
    """Per-step execution audit for an ingestion run."""

    __tablename__ = "run_steps"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "step_name",
            "attempt",
            name="uq_run_steps_run_step_attempt",
        ),
        CheckConstraint(
            f"step_name IN {_sql_in_values(RUN_STEP_NAME_VALUES)}",
            name="ck_run_steps_step_name",
        ),
        CheckConstraint(
            f"status IN {_sql_in_values(RUN_STATUS_VALUES)}",
            name="ck_run_steps_status",
        ),
    )

    run_step_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    step_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class ChunkManifestRow(Base):
    """Chunk manifest metadata for a document version."""

    __tablename__ = "chunk_manifests"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "parser_fingerprint",
            "chunker_fingerprint",
            name="uq_chunk_manifests_document_parser_chunker",
        ),
    )

    manifest_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    manifest_schema_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    source_checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    parser_fingerprint: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    chunker_fingerprint: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    manifest_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChunkRow(Base):
    """Canonical chunk registry."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "chunk_ordinal",
            name="uq_chunks_document_version_ordinal",
        ),
        UniqueConstraint(
            "document_version_id",
            "chunk_content_hash",
            "chunk_ordinal",
            name="uq_chunks_document_version_hash_ordinal",
        ),
    )

    chunk_pk: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    manifest_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chunk_manifests.manifest_id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_ordinal: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunk_content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    normalized_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    original_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    page_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    page_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    section_path: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    character_start: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    character_end: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DiscoveryRunRow(Base):
    """Requirement-discovery run metadata."""

    __tablename__ = "discovery_runs"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_sql_in_values(DISCOVERY_RUN_STATUS_VALUES)}",
            name="ck_discovery_runs_status",
        ),
    )

    discovery_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    provider_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    provider_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    prompt_fingerprint: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class DiscoveryBatchRow(Base):
    """Chunk batch submitted to the requirement-discovery provider."""

    __tablename__ = "discovery_batches"

    batch_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    discovery_run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("discovery_runs.discovery_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    chunk_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )
    token_budget: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    provider_metadata: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class FactRow(Base):
    """Validated canonical fact."""

    __tablename__ = "facts"
    __table_args__ = (
        CheckConstraint(
            f"fact_type IN {_sql_in_values(FACT_TYPE_VALUES)}",
            name="ck_facts_fact_type",
        ),
        CheckConstraint(
            f"validation_status IN {_sql_in_values(VALIDATION_STATUS_VALUES)}",
            name="ck_facts_validation_status",
        ),
    )

    fact_pk: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    fact_sequence: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
        unique=True,
    )
    fact_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    statement: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    normalized_statement_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    fact_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    validation_status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class FactEvidenceRow(Base):
    """Evidence linking facts to chunks."""

    __tablename__ = "fact_evidence"
    __table_args__ = (
        UniqueConstraint(
            "fact_id",
            "chunk_id",
            "exact_quote",
            name="uq_fact_evidence_fact_chunk_quote",
        ),
    )

    fact_evidence_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    fact_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("facts.fact_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("chunks.chunk_id", ondelete="RESTRICT"),
        nullable=False,
    )
    exact_quote: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    quote_start: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    quote_end: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RequirementRow(Base):
    """Validated canonical requirement."""

    __tablename__ = "requirements"
    __table_args__ = (
        CheckConstraint(
            f"requirement_type IN {_sql_in_values(REQUIREMENT_TYPE_VALUES)}",
            name="ck_requirements_requirement_type",
        ),
        CheckConstraint(
            f"derivation_type IN {_sql_in_values(DERIVATION_TYPE_VALUES)}",
            name="ck_requirements_derivation_type",
        ),
        CheckConstraint(
            f"validation_status IN {_sql_in_values(VALIDATION_STATUS_VALUES)}",
            name="ck_requirements_validation_status",
        ),
    )

    requirement_pk: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    requirement_sequence: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
        unique=True,
    )
    requirement_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    statement: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    normalized_statement_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    requirement_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    derivation_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    validation_status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RequirementFactLinkRow(Base):
    """Many-to-many support link from requirement to fact."""

    __tablename__ = "requirement_fact_links"
    __table_args__ = (
        UniqueConstraint(
            "requirement_id",
            "fact_id",
            name="uq_requirement_fact_links_requirement_fact",
        ),
    )

    requirement_fact_link_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    requirement_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("requirements.requirement_id", ondelete="CASCADE"),
        nullable=False,
    )
    fact_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("facts.fact_id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RequirementRelationRow(Base):
    """Requirement-to-requirement semantic relationship."""

    __tablename__ = "requirement_relations"
    __table_args__ = (
        UniqueConstraint(
            "source_requirement_id",
            "target_requirement_id",
            "relation_type",
            name="uq_requirement_relations_source_target_type",
        ),
    )

    requirement_relation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    source_requirement_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("requirements.requirement_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_requirement_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("requirements.requirement_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ArtifactRow(Base):
    """Generated artifact registry."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "artifact_path",
            name="uq_artifacts_run_path",
        ),
        CheckConstraint(
            f"status IN {_sql_in_values(ARTIFACT_STATUS_VALUES)}",
            name="ck_artifacts_status",
        ),
    )

    artifact_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="RESTRICT"),
        nullable=False,
    )
    document_version_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("document_versions.document_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    artifact_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    artifact_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    artifact_checksum: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    ready_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ProjectionJobRow(Base):
    """Projection status for derived stores such as Neo4j and Chroma."""

    __tablename__ = "projection_jobs"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "projection_type",
            "target_id",
            name="uq_projection_jobs_run_type_target",
        ),
        CheckConstraint(
            f"status IN {_sql_in_values(PROJECTION_JOB_STATUS_VALUES)}",
            name="ck_projection_jobs_status",
        ),
    )

    projection_job_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    projection_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    target_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    last_error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class OutboxEventRow(Base):
    """Transactional outbox event for resumable projections."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        CheckConstraint(
            f"status IN {_sql_in_values(OUTBOX_EVENT_STATUS_VALUES)}",
            name="ck_outbox_events_status",
        ),
    )

    outbox_event_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid_pk,
    )
    run_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    aggregate_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    aggregate_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
