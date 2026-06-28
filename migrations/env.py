from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.infrastructure.postgres.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = load_settings()

if settings.postgres.dsn is None:
    raise RuntimeError("POSTGRES_DSN is required to run Alembic migrations.")

config.set_main_option(
    "sqlalchemy.url",
    settings.postgres.dsn.get_secret_value(),
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without creating an engine."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section)

    if configuration is None:
        raise RuntimeError("Alembic configuration section is missing.")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
