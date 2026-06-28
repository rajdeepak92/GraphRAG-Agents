"""Execution helpers for the ingestion graph."""

from __future__ import annotations

from typing import Literal, cast
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.domain.commands import IngestDocumentCommand

from .graph import build_ingestion_graph
from .state import IngestionState

CheckpointMode = Literal["memory", "postgres"]


def _to_psycopg_dsn(dsn: str) -> str:
    """Convert SQLAlchemy asyncpg DSN to psycopg-compatible DSN."""

    return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)


async def run_ingestion_skeleton(
    *,
    command: IngestDocumentCommand,
    resume_run: str | None = None,
    checkpoint_mode: CheckpointMode = "memory",
    setup_checkpointer: bool = False,
) -> IngestionState:
    """Run the Phase 7 placeholder workflow from START to END."""

    run_id = resume_run or str(uuid4())

    initial_state = IngestionState(
        command=command.model_dump(mode="json"),
        run_id=run_id,
        current_step="requested",
        warnings=[],
        errors=[],
    )

    config = {
        "configurable": {
            "thread_id": run_id,
            "checkpoint_ns": "ingestion",
        }
    }

    if checkpoint_mode == "memory":
        graph = build_ingestion_graph(checkpointer=InMemorySaver())
        result = await graph.ainvoke(initial_state, config=config)
        return cast(IngestionState, result)

    settings = load_settings()

    if settings.postgres.dsn is None:
        msg = "POSTGRES_DSN is required for postgres checkpoint mode."
        raise RuntimeError(msg)

    postgres_dsn = _to_psycopg_dsn(settings.postgres.dsn.get_secret_value())

    async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as checkpointer:
        if setup_checkpointer:
            await checkpointer.setup()

        graph = build_ingestion_graph(checkpointer=checkpointer)
        result = await graph.ainvoke(initial_state, config=config)
        return cast(IngestionState, result)
