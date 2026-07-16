"""Durable LangGraph PostgreSQL checkpoint helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.settings import AppSettings


@contextmanager
def workflow_checkpointer(settings: AppSettings) -> Iterator[Any]:
    """Yield PostgreSQL persistence in production and memory only for local tests."""
    if settings.postgres.mode == "local_json":
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(settings.postgres.dsn) as saver:
        saver.setup()
        yield saver


__all__ = ["workflow_checkpointer"]
