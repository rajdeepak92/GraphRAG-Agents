"""PostgreSQL unit-of-work implementation."""

from __future__ import annotations

from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from multi_agentic_graph_rag.infrastructure.postgres.repositories import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
    SqlAlchemyIngestionRunRepository,
    SqlAlchemyProjectRepository,
    SqlAlchemyRunStepRepository,
)


class PostgresUnitOfWork:
    """Transaction boundary for PostgreSQL repository operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self._committed = False

        self.projects: SqlAlchemyProjectRepository
        self.documents: SqlAlchemyDocumentRepository
        self.document_versions: SqlAlchemyDocumentVersionRepository
        self.ingestion_runs: SqlAlchemyIngestionRunRepository
        self.run_steps: SqlAlchemyRunStepRepository

    async def __aenter__(self) -> PostgresUnitOfWork:
        self.session = self._session_factory()

        self.projects = SqlAlchemyProjectRepository(self.session)
        self.documents = SqlAlchemyDocumentRepository(self.session)
        self.document_versions = SqlAlchemyDocumentVersionRepository(self.session)
        self.ingestion_runs = SqlAlchemyIngestionRunRepository(self.session)
        self.run_steps = SqlAlchemyRunStepRepository(self.session)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return

        try:
            if exc_type is not None or not self._committed:
                await self.session.rollback()
        finally:
            await self.session.close()

    async def rollback(self) -> None:
        if self.session is None:
            return

        await self.session.rollback()
        self._committed = False

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("Cannot commit before entering unit of work.")

        await self.session.commit()
        self._committed = True
