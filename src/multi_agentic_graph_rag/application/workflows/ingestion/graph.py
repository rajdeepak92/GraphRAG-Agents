"""LangGraph ingestion workflow skeleton."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agentic_graph_rag.application.workflows.ingestion.nodes import placeholders

from .state import IngestionState

INGESTION_NODE_SEQUENCE: tuple[str, ...] = (
    "validate_command",
    "bootstrap_runtime",
    "check_dependencies",
    "register_run",
    "resolve_version_lineage",
    "parse_document",
    "chunk_document",
    "persist_manifest",
    "persist_source_records",
    "project_chunk_graph",
    "index_chunk_vectors",
    "create_discovery_run",
    "discover_requirements",
    "reconcile_requirements",
    "persist_requirements",
    "project_requirement_graph",
    "write_artifact",
    "verify_outputs",
    "complete_run",
)


def build_ingestion_graph(*, checkpointer: Any | None = None) -> Any:
    """Build and compile the Phase 7 ingestion graph."""

    builder = StateGraph(IngestionState)

    for node_name in INGESTION_NODE_SEQUENCE:
        builder.add_node(node_name, getattr(placeholders, node_name))

    builder.add_edge(START, INGESTION_NODE_SEQUENCE[0])

    for current_node, next_node in pairwise(INGESTION_NODE_SEQUENCE):
        builder.add_edge(current_node, next_node)

    builder.add_edge(INGESTION_NODE_SEQUENCE[-1], END)

    return builder.compile(checkpointer=checkpointer)
