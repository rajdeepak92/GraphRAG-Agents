from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.domain.documents import DocumentVersion
from multi_agentic_graph_rag.domain.enums import RunStatus, RunStepName
from multi_agentic_graph_rag.domain.identifiers import normalize_version
from multi_agentic_graph_rag.domain.runs import IngestionRun, RunStep
from multi_agentic_graph_rag.infrastructure.postgres.session import (
    create_postgres_engine,
    create_session_factory,
)
from multi_agentic_graph_rag.infrastructure.postgres.unit_of_work import (
    PostgresUnitOfWork,
)

PROJECT_KEY = "PROJECT_1"
PROJECT_NAME = "Phase 4 Smoke Project"
LOGICAL_DOCUMENT_NAME = "requirements.pdf"
SUPPLIED_VERSION = "1.0"
SOURCE_CHECKSUM = "f" * 64


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid4().hex[:8].upper()
    return f"RUN-SMOKE-{timestamp}-{suffix}"


def _status(name: str) -> RunStatus:
    try:
        return RunStatus[name]
    except KeyError as exc:
        available = ", ".join(status.name for status in RunStatus)
        raise RuntimeError(
            f"RunStatus.{name} does not exist. Available statuses: {available}"
        ) from exc


async def main() -> None:
    settings = load_settings()
    engine = create_postgres_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with PostgresUnitOfWork(session_factory) as uow:
            project = await uow.projects.get_by_key(PROJECT_KEY)

            if project is None:
                project = await uow.projects.create(
                    project_key=PROJECT_KEY,
                    name=PROJECT_NAME,
                )

            document = await uow.documents.get_by_project_and_name(
                project_id=project.project_id,
                logical_document_name=LOGICAL_DOCUMENT_NAME,
            )

            if document is None:
                document = await uow.documents.create(
                    project_id=project.project_id,
                    logical_document_name=LOGICAL_DOCUMENT_NAME,
                )

            locked_document = await uow.documents.lock_by_id(document.document_id)

            document_version = await uow.document_versions.get_by_document_and_version(
                document_id=locked_document.document_id,
                supplied_version=SUPPLIED_VERSION,
            )

            if document_version is None:
                new_document_version = DocumentVersion(
                    document_version_id=uuid4(),
                    document_id=locked_document.document_id,
                    supplied_version=SUPPLIED_VERSION,
                    normalized_version=normalize_version(SUPPLIED_VERSION),
                    source_checksum=SOURCE_CHECKSUM,
                    supersedes_document_version_id=None,
                    parser_fingerprint=None,
                    chunker_fingerprint=None,
                    embedding_fingerprint=None,
                    prompt_fingerprint=None,
                    created_at=datetime.now(UTC),
                )

                document_version = await uow.document_versions.create(new_document_version)

            elif document_version.source_checksum != SOURCE_CHECKSUM:
                raise RuntimeError(
                    "PROJECT_1 requirements.pdf version 1.0 already exists with "
                    "a different checksum. Use a clean DB or change SOURCE_CHECKSUM."
                )

            now = datetime.now(UTC)

            run = IngestionRun(
                run_pk=uuid4(),
                run_id=_new_run_id(),
                project_id=project.project_id,
                document_id=locked_document.document_id,
                document_version_id=document_version.document_version_id,
                status=RunStatus.REQUESTED,
                requested_at=now,
                completed_at=None,
            )

            created_run = await uow.ingestion_runs.create(run)

            register_step = RunStep(
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

            await uow.run_steps.create(register_step)

            validated = _status("VALIDATED")
            source_registered = _status("SOURCE_REGISTERED")

            await uow.ingestion_runs.transition(
                run_id=created_run.run_id,
                target_status=validated,
            )

            await uow.ingestion_runs.transition(
                run_id=created_run.run_id,
                target_status=source_registered,
            )

            queried_run = await uow.ingestion_runs.get_by_run_id(created_run.run_id)

            if queried_run is None:
                raise RuntimeError("Created run could not be queried back.")

            if queried_run.status != source_registered:
                raise RuntimeError(f"Unexpected final status: {queried_run.status.value}")

            await uow.commit()

            print(f"PASS Phase 4 PostgreSQL smoke test: {queried_run.run_id}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
