"""Serializable LangGraph state for the ingestion workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class IngestionState(TypedDict, total=False):
    """Small, checkpoint-safe ingestion state.

    Keep only identifiers, command payloads, status fields, warnings, and errors.
    Do not store DB sessions, clients, models, document bytes, or vectors here.
    """

    command: dict[str, Any]

    run_id: str
    project_id: str
    document_id: str
    document_version_id: str
    source_checksum: str
    manifest_id: str

    chunk_ids: list[str]
    discovery_run_id: str
    batch_ids: list[str]
    artifact_id: str

    current_step: str
    warnings: list[str]
    errors: list[str]
