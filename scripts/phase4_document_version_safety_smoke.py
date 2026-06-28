from __future__ import annotations

import asyncio
from uuid import uuid4

from multi_agentic_graph_rag.application.services.document_version_safety import (
    DocumentVersionSafetyRequest,
    register_document_version_safely,
)
from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.domain.errors import VersionConflictError
from multi_agentic_graph_rag.infrastructure.postgres.session import (
    create_postgres_engine,
    create_session_factory,
)
from multi_agentic_graph_rag.infrastructure.postgres.unit_of_work import (
    PostgresUnitOfWork,
)

PROJECT_KEY = f"PROJECT_SMOKE_SAFETY_{uuid4().hex[:8].upper()}"
PROJECT_NAME = "Smoke Test Project"
LOGICAL_DOCUMENT_NAME = "requirements.pdf"
SUPPLIED_VERSION = "1.0"
ORIGINAL_CHECKSUM = "a" * 64
CONFLICTING_CHECKSUM = "b" * 64


def _build_request(*, source_checksum: str) -> DocumentVersionSafetyRequest:
    return DocumentVersionSafetyRequest(
        project_key=PROJECT_KEY,
        project_name=PROJECT_NAME,
        logical_document_name=LOGICAL_DOCUMENT_NAME,
        supplied_version=SUPPLIED_VERSION,
        source_checksum=source_checksum,
        replace_version=False,
    )


async def main() -> None:
    settings = load_settings()
    engine = create_postgres_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with PostgresUnitOfWork(session_factory) as uow:
            first = await register_document_version_safely(
                uow=uow,
                request=_build_request(source_checksum=ORIGINAL_CHECKSUM),
            )

        print(f"FIRST: {first.action} {first.run_id}")

        if first.action != "created_new_document_version":
            raise AssertionError(
                f"Expected first action to create a new document version, got: {first.action}"
            )

        async with PostgresUnitOfWork(session_factory) as uow:
            second = await register_document_version_safely(
                uow=uow,
                request=_build_request(source_checksum=ORIGINAL_CHECKSUM),
            )

        print(f"SECOND: {second.action} {second.run_id}")

        if second.action not in {
            "resumed_existing_open_run",
            "reused_existing_document_version",
        }:
            raise AssertionError(
                f"Expected second action to reuse/resume existing state, got: {second.action}"
            )

        if second.run_id != first.run_id:
            raise AssertionError(
                f"Expected second run_id to match first run_id. "
                f"first={first.run_id}, second={second.run_id}"
            )

        try:
            async with PostgresUnitOfWork(session_factory) as uow:
                await register_document_version_safely(
                    uow=uow,
                    request=_build_request(source_checksum=CONFLICTING_CHECKSUM),
                )
        except VersionConflictError:
            print("CONFLICT: rejected different checksum as expected")
        else:
            raise AssertionError("Expected VersionConflictError was not raised")

    finally:
        await engine.dispose()

    print("PASS Phase 4 document-version safety smoke test")


if __name__ == "__main__":
    asyncio.run(main())
