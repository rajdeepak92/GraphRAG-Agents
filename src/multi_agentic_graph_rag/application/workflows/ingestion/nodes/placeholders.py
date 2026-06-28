"""Phase 7 placeholder ingestion nodes.

These nodes prove orchestration shape only. Real persistence, parsing, chunking,
projection, discovery, reconciliation, and artifact behavior comes in later phases.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import structlog

from multi_agentic_graph_rag.domain.commands import IngestDocumentCommand

from ..state import IngestionState

logger = structlog.get_logger(__name__)

NodeHandler = Callable[[IngestionState], Awaitable[IngestionState]]


def _base_update(
    state: IngestionState,
    *,
    step_name: str,
    **updates: Any,
) -> IngestionState:
    run_id = updates.get("run_id") or state.get("run_id")

    logger.info(
        "ingestion_node_completed",
        run_id=run_id,
        step_name=step_name,
    )

    next_state: IngestionState = {
        "current_step": step_name,
        "warnings": list(state.get("warnings", [])),
        "errors": list(state.get("errors", [])),
    }

    if "run_id" in updates:
        next_state["run_id"] = str(updates["run_id"])

    if "project_id" in updates:
        next_state["project_id"] = str(updates["project_id"])

    if "document_id" in updates:
        next_state["document_id"] = str(updates["document_id"])

    if "document_version_id" in updates:
        next_state["document_version_id"] = str(updates["document_version_id"])

    if "source_checksum" in updates:
        next_state["source_checksum"] = str(updates["source_checksum"])

    if "manifest_id" in updates:
        next_state["manifest_id"] = str(updates["manifest_id"])

    if "chunk_ids" in updates:
        next_state["chunk_ids"] = [str(value) for value in updates["chunk_ids"]]

    if "discovery_run_id" in updates:
        next_state["discovery_run_id"] = str(updates["discovery_run_id"])

    if "batch_ids" in updates:
        next_state["batch_ids"] = [str(value) for value in updates["batch_ids"]]

    if "artifact_id" in updates:
        next_state["artifact_id"] = str(updates["artifact_id"])

    return next_state


async def validate_command(state: IngestionState) -> IngestionState:
    IngestDocumentCommand.model_validate(state["command"])
    return _base_update(state, step_name="validate_command")


async def bootstrap_runtime(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="bootstrap_runtime")


async def check_dependencies(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="check_dependencies")


async def register_run(state: IngestionState) -> IngestionState:
    run_id = state.get("run_id") or str(uuid4())
    return _base_update(state, step_name="register_run", run_id=run_id)


async def resolve_version_lineage(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="resolve_version_lineage")


async def parse_document(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="parse_document")


async def chunk_document(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="chunk_document", chunk_ids=[])


async def persist_manifest(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="persist_manifest")


async def persist_source_records(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="persist_source_records")


async def project_chunk_graph(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="project_chunk_graph")


async def index_chunk_vectors(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="index_chunk_vectors")


async def create_discovery_run(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="create_discovery_run", batch_ids=[])


async def discover_requirements(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="discover_requirements")


async def reconcile_requirements(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="reconcile_requirements")


async def persist_requirements(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="persist_requirements")


async def project_requirement_graph(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="project_requirement_graph")


async def write_artifact(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="write_artifact")


async def verify_outputs(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="verify_outputs")


async def complete_run(state: IngestionState) -> IngestionState:
    return _base_update(state, step_name="complete_run")
