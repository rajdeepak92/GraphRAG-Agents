"""phase 4 postgres control plane baseline.

Revision ID: 95d3d37e4060
Revises:
Create Date: 2026-06-28 23:54:26.417029
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "95d3d37e4060"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_key", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("project_id", name="pk_projects"),
        sa.UniqueConstraint("project_key", name="uq_projects_project_key"),
    )

    op.create_table(
        "documents",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("logical_document_name", sa.String(length=512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.project_id"],
            name="fk_documents_project_id_projects",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("document_id", name="pk_documents"),
        sa.UniqueConstraint(
            "project_id",
            "logical_document_name",
            name="uq_documents_project_logical_name",
        ),
    )

    op.create_table(
        "document_versions",
        sa.Column(
            "document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("supplied_version", sa.String(length=128), nullable=False),
        sa.Column("normalized_version", sa.String(length=128), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "supersedes_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("parser_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("chunker_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("embedding_fingerprint", sa.String(length=128), nullable=True),
        sa.Column("prompt_fingerprint", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name="fk_document_versions_document_id_documents",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_doc_versions_supersedes_version",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint(
            "document_version_id",
            name="pk_document_versions",
        ),
        sa.UniqueConstraint(
            "document_id",
            "supplied_version",
            name="uq_document_versions_document_supplied_version",
        ),
    )

    op.create_table(
        "source_files",
        sa.Column("source_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("media_type", sa.String(length=128), nullable=True),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_source_files_document_version_id_document_versions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("source_file_id", name="pk_source_files"),
        sa.UniqueConstraint(
            "document_version_id",
            "source_checksum",
            name="uq_source_files_document_version_checksum",
        ),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("run_pk", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ("
            "'requested', "
            "'validated', "
            "'source_registered', "
            "'parsed', "
            "'chunked', "
            "'chunks_persisted', "
            "'graph_projected', "
            "'vectors_indexed', "
            "'discovery_running', "
            "'requirements_validated', "
            "'requirements_persisted', "
            "'artifact_written', "
            "'completed', "
            "'failed', "
            "'partially_completed', "
            "'cancelled'"
            ")",
            name="ck_ingestion_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.project_id"],
            name="fk_ingestion_runs_project_id_projects",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name="fk_ingestion_runs_document_id_documents",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_ingestion_runs_document_version_id_document_versions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("run_pk", name="pk_ingestion_runs"),
        sa.UniqueConstraint("run_id", name="uq_ingestion_runs_run_id"),
    )

    op.create_table(
        "run_steps",
        sa.Column("run_step_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("step_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ("
            "'requested', "
            "'validated', "
            "'source_registered', "
            "'parsed', "
            "'chunked', "
            "'chunks_persisted', "
            "'graph_projected', "
            "'vectors_indexed', "
            "'discovery_running', "
            "'requirements_validated', "
            "'requirements_persisted', "
            "'artifact_written', "
            "'completed', "
            "'failed', "
            "'partially_completed', "
            "'cancelled'"
            ")",
            name="ck_run_steps_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_run_steps_run_id_ingestion_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_step_id", name="pk_run_steps"),
        sa.UniqueConstraint(
            "run_id",
            "step_name",
            "attempt",
            name="uq_run_steps_run_step_attempt",
        ),
    )
    op.create_table(
        "chunk_manifests",
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("manifest_schema_version", sa.String(length=32), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column("parser_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("chunker_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("manifest_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_chunk_manifests_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("manifest_id", name="pk_chunk_manifests"),
        sa.UniqueConstraint(
            "document_version_id",
            "parser_fingerprint",
            "chunker_fingerprint",
            name="uq_chunk_manifests_doc_parser_chunker",
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("chunk_pk", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("manifest_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=False),
        sa.Column("chunk_content_hash", sa.String(length=64), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_path", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("character_start", sa.Integer(), nullable=False),
        sa.Column("character_end", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_chunks_document_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["manifest_id"],
            ["chunk_manifests.manifest_id"],
            name="fk_chunks_manifest",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("chunk_pk", name="pk_chunks"),
        sa.UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
        sa.UniqueConstraint(
            "document_version_id",
            "chunk_ordinal",
            name="uq_chunks_document_version_ordinal",
        ),
        sa.UniqueConstraint(
            "document_version_id",
            "chunk_content_hash",
            "chunk_ordinal",
            name="uq_chunks_document_version_hash_ordinal",
        ),
    )

    op.create_table(
        "discovery_runs",
        sa.Column("discovery_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider_name", sa.String(length=128), nullable=False),
        sa.Column("provider_model", sa.String(length=255), nullable=True),
        sa.Column("prompt_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_discovery_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_discovery_runs_run",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_discovery_runs_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("discovery_run_id", name="pk_discovery_runs"),
    )

    op.create_table(
        "discovery_batches",
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("discovery_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("token_budget", sa.Integer(), nullable=False),
        sa.Column("provider_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["discovery_run_id"],
            ["discovery_runs.discovery_run_id"],
            name="fk_discovery_batches_discovery_run",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_discovery_batches_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("batch_id", name="pk_discovery_batches"),
    )

    op.create_table(
        "facts",
        sa.Column("fact_pk", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_sequence", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("normalized_statement_hash", sa.String(length=64), nullable=True),
        sa.Column("fact_type", sa.String(length=64), nullable=False),
        sa.Column("validation_status", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_facts_document_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_facts_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("fact_pk", name="pk_facts"),
        sa.UniqueConstraint("fact_sequence", name="uq_facts_fact_sequence"),
        sa.UniqueConstraint("fact_id", name="uq_facts_fact_id"),
    )

    op.create_table(
        "requirements",
        sa.Column("requirement_pk", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "requirement_sequence",
            sa.BigInteger(),
            sa.Identity(always=False),
            nullable=False,
        ),
        sa.Column("requirement_id", sa.String(length=64), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("normalized_statement_hash", sa.String(length=64), nullable=True),
        sa.Column("requirement_type", sa.String(length=64), nullable=False),
        sa.Column("derivation_type", sa.String(length=64), nullable=False),
        sa.Column("validation_status", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_requirements_document_version",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_requirements_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("requirement_pk", name="pk_requirements"),
        sa.UniqueConstraint(
            "requirement_sequence",
            name="uq_requirements_requirement_sequence",
        ),
        sa.UniqueConstraint("requirement_id", name="uq_requirements_requirement_id"),
    )

    op.create_table(
        "fact_evidence",
        sa.Column("fact_evidence_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("exact_quote", sa.Text(), nullable=False),
        sa.Column("quote_start", sa.Integer(), nullable=True),
        sa.Column("quote_end", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["fact_id"],
            ["facts.fact_id"],
            name="fk_fact_evidence_fact",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.chunk_id"],
            name="fk_fact_evidence_chunk",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("fact_evidence_id", name="pk_fact_evidence"),
        sa.UniqueConstraint(
            "fact_id",
            "chunk_id",
            "exact_quote",
            name="uq_fact_evidence_fact_chunk_quote",
        ),
    )

    op.create_table(
        "requirement_fact_links",
        sa.Column(
            "requirement_fact_link_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("requirement_id", sa.String(length=64), nullable=False),
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["requirement_id"],
            ["requirements.requirement_id"],
            name="fk_req_fact_links_requirement",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["fact_id"],
            ["facts.fact_id"],
            name="fk_req_fact_links_fact",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "requirement_fact_link_id",
            name="pk_requirement_fact_links",
        ),
        sa.UniqueConstraint(
            "requirement_id",
            "fact_id",
            name="uq_requirement_fact_links_requirement_fact",
        ),
    )

    op.create_table(
        "requirement_relations",
        sa.Column("requirement_relation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_requirement_id", sa.String(length=64), nullable=False),
        sa.Column("target_requirement_id", sa.String(length=64), nullable=False),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["source_requirement_id"],
            ["requirements.requirement_id"],
            name="fk_req_relations_source_requirement",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_requirement_id"],
            ["requirements.requirement_id"],
            name="fk_req_relations_target_requirement",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "requirement_relation_id",
            name="pk_requirement_relations",
        ),
        sa.UniqueConstraint(
            "source_requirement_id",
            "target_requirement_id",
            "relation_type",
            name="uq_requirement_relations_source_target_type",
        ),
    )

    op.create_table(
        "artifacts",
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("artifact_checksum", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name="ck_artifacts_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_artifacts_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.document_version_id"],
            name="fk_artifacts_document_version",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("artifact_id", name="pk_artifacts"),
        sa.UniqueConstraint("run_id", "artifact_path", name="uq_artifacts_run_path"),
    )

    op.create_table(
        "projection_jobs",
        sa.Column("projection_job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("projection_type", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("last_error_code", sa.String(length=128), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'ready', 'failed')",
            name="ck_projection_jobs_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_projection_jobs_run",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("projection_job_id", name="pk_projection_jobs"),
        sa.UniqueConstraint(
            "run_id",
            "projection_type",
            "target_id",
            name="uq_projection_jobs_run_type_target",
        ),
    )

    op.create_table(
        "outbox_events",
        sa.Column("outbox_event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("aggregate_type", sa.String(length=128), nullable=False),
        sa.Column("aggregate_id", sa.String(length=128), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'processed', 'failed')",
            name="ck_outbox_events_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.run_id"],
            name="fk_outbox_events_run",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("outbox_event_id", name="pk_outbox_events"),
    )


def downgrade() -> None:
    op.drop_table("outbox_events")
    op.drop_table("projection_jobs")
    op.drop_table("artifacts")
    op.drop_table("requirement_relations")
    op.drop_table("requirement_fact_links")
    op.drop_table("fact_evidence")
    op.drop_table("requirements")
    op.drop_table("facts")
    op.drop_table("discovery_batches")
    op.drop_table("discovery_runs")
    op.drop_table("chunks")
    op.drop_table("chunk_manifests")
    op.drop_table("run_steps")
    op.drop_table("ingestion_runs")
    op.drop_table("source_files")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("projects")
