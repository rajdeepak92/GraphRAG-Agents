"""Document-version safety use case for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from multi_agentic_graph_rag.domain.documents import DocumentVersion
from multi_agentic_graph_rag.domain.enums import RunStatus, RunStepName
from multi_agentic_graph_rag.domain.errors import VersionConflictError
from multi_agentic_graph_rag.domain.identifiers import normalize_version
from multi_agentic_graph_rag.domain.runs import IngestionRun, RunStep
from multi_agentic_graph_rag.infrastructure.postgres.unit_of_work import (
    PostgresUnitOfWork,
)


@dataclass(frozen=True)
class DocumentVersionSafetyRequest:
    project_key: str
    logical_document_name: str
    supplied_version: str
    source_checksum: str
    replace_version: bool = False
    project_name: str | None = None
    parser_fingerprint: str | None = None
    chunker_fingerprint: str | None = None
    embedding_fingerprint: str | None = None
    prompt_fingerprint: str | None = None


@dataclass(frozen=True)
class DocumentVersionSafetyResult:
    project_id: str
    document_id: str
    document_version_id: str
    run_id: str
    action: str


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid4().hex[:8].upper()
    return f"RUN-{timestamp}-{suffix}"


async def register_document_version_safely(
    *,
    uow: PostgresUnitOfWork,
    request: DocumentVersionSafetyRequest,
) -> DocumentVersionSafetyResult:
    project = await uow.projects.get_by_key(request.project_key)

    if project is None:
        project = await uow.projects.create(
            project_key=request.project_key,
            name=request.project_name,
        )

    document = await uow.documents.get_by_project_and_name(
        project_id=project.project_id,
        logical_document_name=request.logical_document_name,
    )

    if document is None:
        document = await uow.documents.create(
            project_id=project.project_id,
            logical_document_name=request.logical_document_name,
        )

    locked_document = await uow.documents.lock_by_id(document.document_id)

    existing_version = await uow.document_versions.get_by_document_and_version(
        document_id=locked_document.document_id,
        supplied_version=request.supplied_version,
    )

    if existing_version is not None:
        if existing_version.source_checksum == request.source_checksum:
            existing_open_run = await uow.ingestion_runs.get_latest_open_by_document_version(
                existing_version.document_version_id,
            )

            if existing_open_run is not None:
                await uow.commit()

                return DocumentVersionSafetyResult(
                    project_id=str(project.project_id),
                    document_id=str(locked_document.document_id),
                    document_version_id=str(existing_version.document_version_id),
                    run_id=existing_open_run.run_id,
                    action="resumed_existing_open_run",
                )

            run = await _create_register_run(
                uow=uow,
                project_id=project.project_id,
                document_id=locked_document.document_id,
                document_version_id=existing_version.document_version_id,
            )

            await uow.commit()

            return DocumentVersionSafetyResult(
                project_id=str(project.project_id),
                document_id=str(locked_document.document_id),
                document_version_id=str(existing_version.document_version_id),
                run_id=run.run_id,
                action="reused_existing_document_version",
            )

        if not request.replace_version:
            raise VersionConflictError(
                "Document version conflict: same document and supplied version "
                "already exist with a different source checksum."
            )

        raise VersionConflictError(
            "Explicit replacement was requested, but the current Phase 4 schema "
            "uses UNIQUE(document_id, supplied_version). True replacement lineage "
            "requires a schema change before it can be implemented safely."
        )

    document_version = DocumentVersion(
        document_version_id=uuid4(),
        document_id=locked_document.document_id,
        supplied_version=request.supplied_version,
        normalized_version=normalize_version(request.supplied_version),
        source_checksum=request.source_checksum,
        supersedes_document_version_id=None,
        parser_fingerprint=request.parser_fingerprint,
        chunker_fingerprint=request.chunker_fingerprint,
        embedding_fingerprint=request.embedding_fingerprint,
        prompt_fingerprint=request.prompt_fingerprint,
        created_at=datetime.now(UTC),
    )

    created_version = await uow.document_versions.create(document_version)

    run = await _create_register_run(
        uow=uow,
        project_id=project.project_id,
        document_id=locked_document.document_id,
        document_version_id=created_version.document_version_id,
    )

    await uow.commit()

    return DocumentVersionSafetyResult(
        project_id=str(project.project_id),
        document_id=str(locked_document.document_id),
        document_version_id=str(created_version.document_version_id),
        run_id=run.run_id,
        action="created_new_document_version",
    )


async def _create_register_run(
    *,
    uow: PostgresUnitOfWork,
    project_id: UUID,
    document_id: UUID,
    document_version_id: UUID,
) -> IngestionRun:
    now = datetime.now(UTC)

    run = IngestionRun(
        run_pk=uuid4(),
        run_id=_new_run_id(),
        project_id=project_id,
        document_id=document_id,
        document_version_id=document_version_id,
        status=RunStatus.REQUESTED,
        requested_at=now,
        completed_at=None,
    )

    created_run = await uow.ingestion_runs.create(run)

    step = RunStep(
        run_step_id=uuid4(),
        run_id=created_run.run_id,
        step_name=RunStepName.REGISTER_RUN,
        status=RunStatus.COMPLETED,
        attempt=1,
        started_at=now,
        completed_at=datetime.now(UTC),
        error_code=None,
        error_message=None,
    )

    await uow.run_steps.create(step)

    return created_run
