"""Ingestion run and run-step contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from multi_agentic_graph_rag.domain.enums import RunStatus, RunStepName
from multi_agentic_graph_rag.domain.errors import InvalidRunTransitionError

_ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.REQUESTED: frozenset({RunStatus.VALIDATED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.VALIDATED: frozenset(
        {RunStatus.SOURCE_REGISTERED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.SOURCE_REGISTERED: frozenset(
        {RunStatus.PARSED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.PARSED: frozenset({RunStatus.CHUNKED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.CHUNKED: frozenset(
        {RunStatus.CHUNKS_PERSISTED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.CHUNKS_PERSISTED: frozenset(
        {
            RunStatus.GRAPH_PROJECTED,
            RunStatus.VECTORS_INDEXED,
            RunStatus.PARTIALLY_COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.GRAPH_PROJECTED: frozenset(
        {
            RunStatus.VECTORS_INDEXED,
            RunStatus.DISCOVERY_RUNNING,
            RunStatus.PARTIALLY_COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.VECTORS_INDEXED: frozenset(
        {
            RunStatus.DISCOVERY_RUNNING,
            RunStatus.PARTIALLY_COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.DISCOVERY_RUNNING: frozenset(
        {
            RunStatus.REQUIREMENTS_VALIDATED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.REQUIREMENTS_VALIDATED: frozenset(
        {
            RunStatus.REQUIREMENTS_PERSISTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.REQUIREMENTS_PERSISTED: frozenset(
        {
            RunStatus.ARTIFACT_WRITTEN,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.ARTIFACT_WRITTEN: frozenset({RunStatus.COMPLETED, RunStatus.FAILED}),
    RunStatus.PARTIALLY_COMPLETED: frozenset(
        {
            RunStatus.GRAPH_PROJECTED,
            RunStatus.VECTORS_INDEXED,
            RunStatus.DISCOVERY_RUNNING,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset({RunStatus.PARTIALLY_COMPLETED}),
    RunStatus.CANCELLED: frozenset(),
}


def validate_run_transition(current: RunStatus, target: RunStatus) -> None:
    """Reject invalid run-state transitions."""

    if target not in _ALLOWED_TRANSITIONS[current]:
        msg = f"Invalid run transition: {current.value} -> {target.value}"
        raise InvalidRunTransitionError(msg)


class IngestionRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_pk: UUID
    run_id: str = Field(min_length=1)
    project_id: UUID | None = None
    document_id: UUID | None = None
    document_version_id: UUID | None = None
    status: RunStatus = RunStatus.REQUESTED
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> IngestionRun:
        if self.status == RunStatus.COMPLETED and self.completed_at is None:
            msg = "completed_at is required when run status is completed."
            raise ValueError(msg)

        return self


class RunStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_step_id: UUID
    run_id: str
    step_name: RunStepName
    status: RunStatus
    attempt: int = Field(ge=1)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
