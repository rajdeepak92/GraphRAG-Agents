"""Async PostgreSQL engine/session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from multi_agentic_graph_rag.config.settings import Settings


def create_postgres_engine(settings: Settings) -> AsyncEngine:
    if settings.postgres.dsn is None:
        raise RuntimeError("POSTGRES_DSN is required for PostgreSQL access.")

    return create_async_engine(
        settings.postgres.dsn.get_secret_value(),
        pool_size=settings.postgres.pool_size,
        max_overflow=settings.postgres.max_overflow,
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
