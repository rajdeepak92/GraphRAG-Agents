from __future__ import annotations

import asyncio

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


async def main() -> None:
    settings = load_settings()
    engine = create_postgres_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        async with PostgresUnitOfWork(session_factory) as uow:
            first = await register_document_version_safely(
                uow=uow,
                request=DocumentVersionSafetyRequest(
                    project_key="PROJECT_SMOKE",
                    project_name="Smoke Test Project",
                    logical_document_name="requirements.pdf",
                    supplied_version="1.0",
                    source_checksum="a" * 64,
                    replace_version=False,
                ),
            )

        print(f"FIRST: {first.action} {first.run_id}")

        async with PostgresUnitOfWork(session_factory) as uow:
            second = await register_document_version_safely(
                uow=uow,
                request=DocumentVersionSafetyRequest(
                    project_key="PROJECT_SMOKE",
                    project_name="Smoke Test Project",
                    logical_document_name="requirements.pdf",
                    supplied_version="1.0",
                    source_checksum="a" * 64,
                    replace_version=False,
                ),
            )

        print(f"SECOND: {second.action} {second.run_id}")

        try:
            async with PostgresUnitOfWork(session_factory) as uow:
                await register_document_version_safely(
                    uow=uow,
                    request=DocumentVersionSafetyRequest(
                        project_key="PROJECT_SMOKE",
                        project_name="Smoke Test Project",
                        logical_document_name="requirements.pdf",
                        supplied_version="1.0",
                        source_checksum="b" * 64,
                        replace_version=False,
                    ),
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
