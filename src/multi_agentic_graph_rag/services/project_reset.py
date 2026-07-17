"""Explicit project maintenance reset across all stores and generated artifacts."""

from __future__ import annotations

import shutil
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import normalize_project
from multi_agentic_graph_rag.observability.logging import get_logger

_LOG = get_logger(__name__)


def reset_project(project: str, settings: AppSettings) -> dict[str, Any]:
    """Wipe one project only while holding its non-blocking maintenance lease."""
    summary: dict[str, Any] = {"project": project}
    postgres = PostgresStore(settings)
    postgres.ensure_schema()
    with postgres.project_maintenance_lease(project):
        nodes_deleted = Neo4jStore(settings).delete_project(project)
        summary["neo4j_nodes_deleted"] = nodes_deleted
        _LOG.info("reset.neo4j project=%s nodes_deleted=%s", project, nodes_deleted)

        collection_deleted = ChromaStore(settings).delete_project(project)
        summary["chroma_collection_deleted"] = collection_deleted
        _LOG.info("reset.chroma project=%s collection_deleted=%s", project, collection_deleted)

        rows_deleted = postgres.delete_project(project)
        summary["postgres_rows_deleted"] = rows_deleted
        _LOG.info("reset.postgres project=%s rows_deleted=%s", project, rows_deleted)

        generated = settings.paths.generated_dir / normalize_project(project)
        shutil.rmtree(generated, ignore_errors=True)
        summary["generated_dir_removed"] = str(generated)
        _LOG.info("reset.generated project=%s path=%s", project, generated)

    return summary


__all__ = ["reset_project"]
