"""SQLAlchemy repository implementations for PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from multi_agentic_graph_rag.domain.documents import Document, DocumentVersion, Project
from multi_agentic_graph_rag.domain.enums import RunStatus, RunStepName
from multi_agentic_graph_rag.domain.runs import (
    IngestionRun,
    RunStep,
    validate_run_transition,
)
from multi_agentic_graph_rag.infrastructure.postgres.models import (
    DocumentRow,
    DocumentVersionRow,
    IngestionRunRow,
    ProjectRow,
    RunStepRow,
)

TERMINAL_RUN_STATUS_VALUES = {
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.PARTIALLY_COMPLETED.value,
}


def _project_from_row(row: ProjectRow) -> Project:
    return Project(
        project_id=row.project_id,
        project_key=row.project_key,
        name=row.name,
        created_at=row.created_at,
    )


def _document_from_row(row: DocumentRow) -> Document:
    return Document(
        document_id=row.document_id,
        project_id=row.project_id,
        logical_document_name=row.logical_document_name,
        created_at=row.created_at,
    )


def _document_version_from_row(row: DocumentVersionRow) -> DocumentVersion:
    return DocumentVersion(
        document_version_id=row.document_version_id,
        document_id=row.document_id,
        supplied_version=row.supplied_version,
        normalized_version=row.normalized_version,
        source_checksum=row.source_checksum,
        supersedes_document_version_id=row.supersedes_document_version_id,
        parser_fingerprint=row.parser_fingerprint,
        chunker_fingerprint=row.chunker_fingerprint,
        embedding_fingerprint=row.embedding_fingerprint,
        prompt_fingerprint=row.prompt_fingerprint,
        created_at=row.created_at,
    )


def _ingestion_run_from_row(row: IngestionRunRow) -> IngestionRun:
    return IngestionRun(
        run_pk=row.run_pk,
        run_id=row.run_id,
        project_id=row.project_id,
        document_id=row.document_id,
        document_version_id=row.document_version_id,
        status=RunStatus(row.status),
        requested_at=row.requested_at,
        completed_at=row.completed_at,
    )


def _run_step_from_row(row: RunStepRow) -> RunStep:
    return RunStep(
        run_step_id=row.run_step_id,
        run_id=row.run_id,
        step_name=RunStepName(row.step_name),
        status=RunStatus(row.status),
        attempt=row.attempt,
        started_at=row.started_at,
        completed_at=row.completed_at,
        error_code=row.error_code,
        error_message=row.error_message,
    )


class SqlAlchemyProjectRepository:
    """PostgreSQL implementation of project repository operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_key(self, project_key: str) -> Project | None:
        stmt = select(ProjectRow).where(ProjectRow.project_key == project_key)
        row = await self._session.scalar(stmt)

        if row is None:
            return None

        return _project_from_row(row)

    async def create(self, *, project_key: str, name: str | None = None) -> Project:
        row = ProjectRow(
            project_key=project_key,
            name=name,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)

        return _project_from_row(row)


class SqlAlchemyDocumentRepository:
    """PostgreSQL implementation of document repository operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_project_and_name(
        self,
        *,
        project_id: UUID,
        logical_document_name: str,
    ) -> Document | None:
        stmt = select(DocumentRow).where(
            DocumentRow.project_id == project_id,
            DocumentRow.logical_document_name == logical_document_name,
        )
        row = await self._session.scalar(stmt)

        if row is None:
            return None

        return _document_from_row(row)

    async def create(
        self,
        *,
        project_id: UUID,
        logical_document_name: str,
    ) -> Document:
        row = DocumentRow(
            project_id=project_id,
            logical_document_name=logical_document_name,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)

        return _document_from_row(row)

    async def lock_by_id(self, document_id: UUID) -> Document:
        stmt = select(DocumentRow).where(DocumentRow.document_id == document_id).with_for_update()
        row = await self._session.scalar(stmt)

        if row is None:
            raise ValueError(f"Document not found: {document_id}")

        return _document_from_row(row)


class SqlAlchemyDocumentVersionRepository:
    """PostgreSQL implementation of document-version repository operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_document_and_version(
        self,
        *,
        document_id: UUID,
        supplied_version: str,
    ) -> DocumentVersion | None:
        stmt = select(DocumentVersionRow).where(
            DocumentVersionRow.document_id == document_id,
            DocumentVersionRow.supplied_version == supplied_version,
        )
        row = await self._session.scalar(stmt)

        if row is None:
            return None

        return _document_version_from_row(row)

    async def create(self, version: DocumentVersion) -> DocumentVersion:
        row = DocumentVersionRow(
            document_version_id=version.document_version_id,
            document_id=version.document_id,
            supplied_version=version.supplied_version,
            normalized_version=version.normalized_version,
            source_checksum=version.source_checksum,
            supersedes_document_version_id=version.supersedes_document_version_id,
            parser_fingerprint=version.parser_fingerprint,
            chunker_fingerprint=version.chunker_fingerprint,
            embedding_fingerprint=version.embedding_fingerprint,
            prompt_fingerprint=version.prompt_fingerprint,
            created_at=version.created_at,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)

        return _document_version_from_row(row)


class SqlAlchemyIngestionRunRepository:
    """PostgreSQL implementation of ingestion-run repository operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, run: IngestionRun) -> IngestionRun:
        row = IngestionRunRow(
            run_pk=run.run_pk,
            run_id=run.run_id,
            project_id=run.project_id,
            document_id=run.document_id,
            document_version_id=run.document_version_id,
            status=run.status.value,
            requested_at=run.requested_at,
            completed_at=run.completed_at,
            error_code=None,
            error_message=None,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)

        return _ingestion_run_from_row(row)

    async def get_by_run_id(self, run_id: str) -> IngestionRun | None:
        stmt = select(IngestionRunRow).where(IngestionRunRow.run_id == run_id)
        row = await self._session.scalar(stmt)

        if row is None:
            return None

        return _ingestion_run_from_row(row)

    async def get_latest_open_by_document_version(
        self,
        document_version_id: UUID,
    ) -> IngestionRun | None:
        stmt = (
            select(IngestionRunRow)
            .where(
                IngestionRunRow.document_version_id == document_version_id,
                ~IngestionRunRow.status.in_(TERMINAL_RUN_STATUS_VALUES),
            )
            .order_by(desc(IngestionRunRow.requested_at))
            .limit(1)
        )
        row = await self._session.scalar(stmt)

        if row is None:
            return None

        return _ingestion_run_from_row(row)

    async def transition(
        self,
        *,
        run_id: str,
        target_status: RunStatus,
    ) -> IngestionRun:
        stmt = select(IngestionRunRow).where(IngestionRunRow.run_id == run_id)
        row = await self._session.scalar(stmt)

        if row is None:
            raise ValueError(f"Ingestion run not found: {run_id}")

        current_status = RunStatus(row.status)
        validate_run_transition(current_status, target_status)

        row.status = target_status.value

        if target_status.value in TERMINAL_RUN_STATUS_VALUES:
            row.completed_at = datetime.now(UTC)

        await self._session.flush()
        await self._session.refresh(row)

        return _ingestion_run_from_row(row)


class SqlAlchemyRunStepRepository:
    """PostgreSQL implementation of run-step repository operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, step: RunStep) -> RunStep:
        row = RunStepRow(
            run_step_id=step.run_step_id,
            run_id=step.run_id,
            step_name=step.step_name.value,
            status=step.status.value,
            attempt=step.attempt,
            started_at=step.started_at,
            completed_at=step.completed_at,
            error_code=step.error_code,
            error_message=step.error_message,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)

        return _run_step_from_row(row)
