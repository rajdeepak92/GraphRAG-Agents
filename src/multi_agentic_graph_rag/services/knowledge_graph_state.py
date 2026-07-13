"""Shared knowledge-graph readiness state machine.

Both the inline ingestion build (`workflows/ingestion_graph.py`) and the
standalone `build-knowledge-graph` stage (`workflows/knowledge_graph.py`) drive
the KG build through :func:`run_guarded_knowledge_graph_build` so the two paths
can never drift. It owns the full transition sequence:

``building``/``rebuilding`` -> build (bounded transient retry) -> project ->
move the active-knowledge pointer -> ``ready``.

``ready`` is written *last*, only after the active pointer has moved, so a crash
anywhere in the sequence leaves a non-ready state and the graph-primary gate
blocks fail-closed. Deterministic failures (validation / schema / configuration)
are never retried; only transient infrastructure failures are.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError, ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    KnowledgeGraphArtifact,
    KnowledgeGraphStateRecord,
)
from multi_agentic_graph_rag.services.knowledge_graph_builder import (
    build_and_project_knowledge_graph,
)

# Exception classes that indicate a transient infrastructure failure worth
# retrying. Names (not imports) keep this decoupled from optional driver deps.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)
_TRANSIENT_NAMES = frozenset(
    {
        "ServiceUnavailable",
        "SessionExpired",
        "TransientError",
        "WriteServiceUnavailable",
        "ConnectionResetError",
    }
)


@dataclass(frozen=True)
class GuardedKnowledgeGraphBuild:
    """Result of a successful guarded build: the ready state and the built artifact."""

    state: KnowledgeGraphStateRecord
    artifact: KnowledgeGraphArtifact


class _KnowledgeStateStore(Protocol):
    def get_knowledge_graph_state(
        self, document_version_id: str
    ) -> KnowledgeGraphStateRecord | None: ...

    def upsert_knowledge_graph_state(self, state: KnowledgeGraphStateRecord) -> None: ...


def knowledge_graph_rebuild_command(project: str, document_version_id: str) -> str:
    """The exact recovery command an operator runs to rebuild a version's KG."""
    project_arg = project or "<PROJECT>"
    return (
        "marag build-knowledge-graph "
        f"--project {project_arg} --document-version-id {document_version_id}"
    )


def _is_transient(exc: BaseException) -> bool:
    # Deterministic failures must never be retried: retrying only wastes time and
    # re-persists the same failure.
    if isinstance(exc, (ConfigurationError, ModelOutputError, ValueError)):
        return False
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES


def _extractor_fingerprint(reasoning_model: Any) -> str:
    for attr in ("fingerprint", "model_id", "model_name"):
        value = getattr(reasoning_model, attr, None)
        if value:
            return str(value)
    return type(reasoning_model).__name__


def run_guarded_knowledge_graph_build(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    chunks: list[DocumentChunk],
    reasoning_model: Any,
    neo4j: Any,
    postgres: _KnowledgeStateStore,
    settings: AppSettings,
    run_id: str,
    logger: Any | None = None,
) -> GuardedKnowledgeGraphBuild:
    """Run the full guarded KG build and persist the readiness state.

    Returns the ``ready`` state record and the built artifact on success. On
    failure the ``failed`` state is persisted (requirements are never touched) and
    the original exception is re-raised so the caller decides between degrade and
    hard-fail.
    """
    existing = postgres.get_knowledge_graph_state(document_version_id)
    attempt = (existing.attempt if existing is not None else 0) + 1
    in_progress: str = "rebuilding" if existing is not None else "building"
    started_at = datetime.now(UTC)

    postgres.upsert_knowledge_graph_state(
        KnowledgeGraphStateRecord(
            document_version_id=document_version_id,
            project=project,
            document_id=document_id,
            doc_version=doc_version,
            status=in_progress,  # type: ignore[arg-type]
            run_id=run_id,
            attempt=attempt,
            chunk_count=len(chunks),
            extractor_fingerprint=_extractor_fingerprint(reasoning_model),
            started_at=started_at,
        )
    )

    try:
        artifact = _build_with_retry(
            settings=settings,
            logger=logger,
            document_version_id=document_version_id,
            build_call=lambda: build_and_project_knowledge_graph(
                project=project,
                document_id=document_id,
                document_version_id=document_version_id,
                doc_version=doc_version,
                chunks=chunks,
                reasoning_model=reasoning_model,
                neo4j=neo4j,
                logger=logger,
            ),
        )
        # Move the active-knowledge pointer only after a complete projection, then
        # mark ready last so any earlier crash keeps the version blocked.
        neo4j.set_active_knowledge_version(
            document_id=document_id,
            document_version_id=document_version_id,
        )
        ready = _ready_state(
            project=project,
            document_id=document_id,
            document_version_id=document_version_id,
            doc_version=doc_version,
            run_id=run_id,
            attempt=attempt,
            chunk_count=len(chunks),
            artifact=artifact,
            reasoning_model=reasoning_model,
            started_at=started_at,
        )
        postgres.upsert_knowledge_graph_state(ready)
        return GuardedKnowledgeGraphBuild(state=ready, artifact=artifact)
    except Exception as exc:
        failed = KnowledgeGraphStateRecord(
            document_version_id=document_version_id,
            project=project,
            document_id=document_id,
            doc_version=doc_version,
            status="failed",
            run_id=run_id,
            attempt=attempt,
            failure_reason=str(exc),
            chunk_count=len(chunks),
            extractor_fingerprint=_extractor_fingerprint(reasoning_model),
            started_at=started_at,
        )
        postgres.upsert_knowledge_graph_state(failed)
        raise


def _ready_state(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    run_id: str,
    attempt: int,
    chunk_count: int,
    artifact: KnowledgeGraphArtifact,
    reasoning_model: Any,
    started_at: datetime,
) -> KnowledgeGraphStateRecord:
    return KnowledgeGraphStateRecord(
        document_version_id=document_version_id,
        project=project,
        document_id=document_id,
        doc_version=doc_version,
        status="ready",
        run_id=run_id,
        attempt=attempt,
        chunk_count=chunk_count,
        assertion_count=len(artifact.assertions),
        evidence_count=len(artifact.evidence),
        extractor_fingerprint=_extractor_fingerprint(reasoning_model),
        graph_schema_version=artifact.artifact_schema_version,
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )


def _build_with_retry(
    *,
    settings: AppSettings,
    logger: Any | None,
    document_version_id: str,
    build_call: Any,
) -> KnowledgeGraphArtifact:
    attempts = max(1, settings.knowledge_graph.build_max_attempts)
    last_exc: BaseException | None = None
    for index in range(1, attempts + 1):
        try:
            return build_call()  # type: ignore[no-any-return]
        except Exception as exc:
            last_exc = exc
            if index < attempts and _is_transient(exc):
                if logger is not None:
                    logger.warning(
                        "Transient knowledge-graph build failure; retrying",
                        step="build_knowledge_graph",
                        document_version_id=document_version_id,
                        attempt=index,
                        max_attempts=attempts,
                        error=str(exc),
                        status="retrying",
                    )
                continue
            raise
    assert last_exc is not None  # pragma: no cover - loop always raises or returns
    raise last_exc
