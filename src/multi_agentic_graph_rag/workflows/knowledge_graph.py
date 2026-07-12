"""LangGraph orchestration for the source-knowledge graph build stage.

Reads the already-projected chunks of one document version from Neo4j, extracts
entities and assertions per chunk with exact-quote grounding, resolves entities
against the project's existing canonical entities, canonicalizes assertions,
and projects the result back into Neo4j. The audit copy of the build is written
to ``generated/<PROJECT>/kg/<RUN_ID>/knowledge_graph.json`` and the run is
recorded in PostgreSQL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
from multi_agentic_graph_rag.common_defs import ModeName, ProviderName, RuntimeCommand
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.schemas import (
    KnowledgeGraphArtifact,
    KnowledgeGraphRequest,
    KnowledgeGraphResult,
)
from multi_agentic_graph_rag.llm_models.factory import create_reasoning_model
from multi_agentic_graph_rag.observability.session import (
    RunSession,
    command_run_id,
    command_session,
)
from multi_agentic_graph_rag.services.assertion_canonicalization import canonicalize_assertions
from multi_agentic_graph_rag.services.entity_resolution import resolve_entities
from multi_agentic_graph_rag.services.text_unit_segmentation import (
    attach_evidence_text_units,
    segment_version_chunks,
)


class KnowledgeGraphState(TypedDict, total=False):
    request: dict[str, Any]
    run_id: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: str
    chunk_count: int
    entity_count: int
    assertion_count: int
    evidence_count: int
    warnings: list[str]
    errors: list[str]


def build_knowledge_graph_workflow(session: RunSession | None = None) -> Any:
    graph = StateGraph(KnowledgeGraphState)
    graph.add_node("validate_request", lambda state: _validate_request(state, session=session))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, session=session))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()


def _validate_request(
    state: KnowledgeGraphState,
    *,
    session: RunSession | None = None,
) -> KnowledgeGraphState:
    request = KnowledgeGraphRequest.model_validate(state["request"])
    logger = session.logger if session is not None else None
    rid = state.get("run_id") or command_run_id(RuntimeCommand.BUILD_KNOWLEDGE_GRAPH.value)
    if logger is not None:
        logger.debug(
            "Validated knowledge-graph request",
            step="validate_request",
            project=request.project,
            document_version_id=request.document_version_id,
            run_id=rid,
        )
    return {"request": request.model_dump(mode="json"), "run_id": rid, "warnings": [], "errors": []}


def _run_pipeline(
    state: KnowledgeGraphState,
    *,
    session: RunSession | None = None,
) -> KnowledgeGraphState:
    request = KnowledgeGraphRequest.model_validate(state["request"])
    settings = load_config()
    if session is not None:
        session.set_log_level(settings.log_level)
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    logger = session.logger if session is not None else None
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    run_dir = session.run_dir if session is not None else None

    def pipeline() -> KnowledgeGraphState:
        _validate_required_knowledge_stack(settings)
        if logger is not None:
            logger.info(
                "Beginning knowledge-graph build pipeline",
                step="run_pipeline",
                run_id=state["run_id"],
                document_version_id=request.document_version_id,
                status="started",
            )
        _check_store(logger, "check_postgres", postgres.check)
        _check_store(logger, "check_neo4j", neo4j.check)
        postgres.ensure_schema()

        reasoning_model = create_reasoning_model(settings, logger=logger, run_dir=run_dir)
        _warmup_reasoning_model(reasoning_model)
        neo4j.ensure_knowledge_schema()

        metadata = neo4j.fetch_version_metadata(request.document_version_id)
        if metadata is None:
            raise ConfigurationError(
                "document version "
                f"{request.document_version_id} is not projected in Neo4j; run ingest first"
            )
        if metadata["project"] and metadata["project"] != request.project:
            raise ConfigurationError(
                f"--project {request.project} does not own document version "
                f"{request.document_version_id} (owner: {metadata['project']})"
            )
        chunks = neo4j.fetch_version_chunks(request.document_version_id)
        if not chunks:
            raise ConfigurationError(
                f"document version {request.document_version_id} has no chunks in Neo4j"
            )
        if logger is not None:
            logger.info(
                "Loaded {chunk_count} chunks for knowledge extraction",
                step="load_chunks",
                chunk_count=len(chunks),
                document_version_id=request.document_version_id,
                project=request.project,
            )

        text_units = segment_version_chunks(
            document_version_id=request.document_version_id,
            chunks=chunks,
        )
        if logger is not None:
            logger.info(
                "Segmented {chunk_count} chunks into {text_unit_count} text units",
                step="segment_text_units",
                chunk_count=len(chunks),
                text_unit_count=len(text_units),
                document_version_id=request.document_version_id,
            )
        neo4j.project_text_units(request.document_version_id, text_units)

        agent = KnowledgeExtractionAgent(reasoning_model, logger=logger)
        extraction = agent.run(
            project=request.project,
            version=metadata["version"],
            chunks=chunks,
        )
        existing_entities = neo4j.fetch_project_entities(request.project)
        resolution = resolve_entities(
            project=request.project,
            extraction=extraction,
            existing_entities=existing_entities,
            chunk_text_by_id={chunk.chunk_id: chunk.text for chunk in chunks},
        )
        assertions, evidence = canonicalize_assertions(
            project=request.project,
            document_id=metadata["document_id"],
            document_version_id=request.document_version_id,
            extraction=extraction,
            resolution=resolution,
        )
        evidence = attach_evidence_text_units(
            evidence,
            chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
            text_units=text_units,
        )
        artifact = KnowledgeGraphArtifact(
            project=request.project,
            document_id=metadata["document_id"],
            document_version_id=request.document_version_id,
            doc_version=metadata["version"],
            text_units=text_units,
            entities=resolution.entities,
            mentions=resolution.mentions,
            assertions=assertions,
            evidence=evidence,
        )
        if logger is not None:
            logger.info(
                "Projecting source-knowledge graph into Neo4j",
                step="project_knowledge_graph",
                document_version_id=request.document_version_id,
                entity_count=len(artifact.entities),
                assertion_count=len(artifact.assertions),
                evidence_count=len(artifact.evidence),
                store_responsibility="source_knowledge_graph",
            )
        neo4j.project_knowledge_graph(artifact)

        out_dir = _output_dir(settings, request.project, state["run_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / "knowledge_graph.json"
        artifact_path.write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if session is not None:
            session.artifact_payload = artifact.model_dump(mode="json")
        if logger is not None:
            logger.info(
                "Knowledge-graph artifact written to {path}",
                step="write_knowledge_graph_artifact",
                path=str(artifact_path),
                status="completed",
            )

        result_payload: KnowledgeGraphState = {
            "run_id": state["run_id"],
            "project": request.project,
            "document_id": metadata["document_id"],
            "document_version_id": request.document_version_id,
            "doc_version": metadata["version"],
            "artifact_path": str(artifact_path),
            "chunk_count": len(chunks),
            "entity_count": len(artifact.entities),
            "assertion_count": len(artifact.assertions),
            "evidence_count": len(artifact.evidence),
            "warnings": [],
            "errors": [],
        }
        if session is not None:
            session.metadata.update(result_payload)
        postgres.record_run(state["run_id"], "completed", dict(result_payload))
        if logger is not None:
            logger.info(
                "Knowledge-graph build pipeline completed",
                step="run_pipeline",
                run_id=state["run_id"],
                entity_count=len(artifact.entities),
                assertion_count=len(artifact.assertions),
                status="completed",
            )
        return result_payload

    try:
        return pipeline()
    except Exception as exc:
        if logger is not None:
            logger.exception(
                "Knowledge-graph build pipeline failed",
                step="run_pipeline",
                exc=exc,
                status="failed",
            )
        if session is not None:
            session.write_failure_envelope(error=exc)
        _record_failed_run_safely(
            postgres=postgres,
            run_id=state["run_id"],
            payload={"run_id": state["run_id"], "error": str(exc)},
            logger=logger,
        )
        raise


def run_knowledge_graph_build(
    request: KnowledgeGraphRequest,
    session: RunSession | None = None,
) -> KnowledgeGraphResult:
    if session is None:
        with command_session(
            project=request.project,
            version="generated",
            command=RuntimeCommand.BUILD_KNOWLEDGE_GRAPH.value,
            run_id=command_run_id(RuntimeCommand.BUILD_KNOWLEDGE_GRAPH.value),
        ) as managed_session:
            return run_knowledge_graph_build(request, session=managed_session)
    session.request_payload = request.model_dump(mode="json")
    graph = build_knowledge_graph_workflow(session)
    final_state = graph.invoke(
        {"request": request.model_dump(mode="json"), "run_id": session.run_id}
    )
    return KnowledgeGraphResult(
        run_id=final_state["run_id"],
        status="completed",
        project=final_state["project"],
        document_id=final_state["document_id"],
        document_version_id=final_state["document_version_id"],
        doc_version=final_state["doc_version"],
        artifact_path=Path(final_state["artifact_path"]),
        chunk_count=final_state["chunk_count"],
        entity_count=final_state["entity_count"],
        assertion_count=final_state["assertion_count"],
        evidence_count=final_state["evidence_count"],
        warnings=final_state.get("warnings", []),
        errors=final_state.get("errors", []),
    )


def _output_dir(settings: AppSettings, project: str, run_identifier: str) -> Path:
    return settings.paths.generated_requirements_dir / project / "kg" / run_identifier


def _validate_required_knowledge_stack(settings: AppSettings) -> None:
    if settings.reasoning_model.provider in {ProviderName.LOCAL_HEURISTIC.value}:
        raise ConfigurationError(
            f"REASONING_MODEL_PROVIDER={settings.reasoning_model.provider} "
            "is not valid for knowledge-graph builds"
        )
    if settings.postgres.mode != ModeName.POSTGRES.value:
        raise ConfigurationError("POSTGRES_MODE=postgres is required for knowledge-graph builds")
    if settings.neo4j.mode != ModeName.NEO4J.value:
        raise ConfigurationError("NEO4J_MODE=neo4j is required for knowledge-graph builds")


def _check_store(logger: Any | None, step: str, check: Any) -> None:
    detail = check()
    if logger is not None:
        logger.info(detail, step=step, status="PASS", detail=detail)


def _warmup_reasoning_model(reasoning_model: Any) -> None:
    warmup = getattr(reasoning_model, "warmup", None)
    if callable(warmup):
        warmup()


def _record_failed_run_safely(
    *,
    postgres: PostgresStore,
    run_id: str,
    payload: dict[str, Any],
    logger: Any,
) -> None:
    try:
        postgres.record_run(run_id, "failed", payload)
    except Exception as record_error:
        if logger is not None:
            logger.warning(
                "Could not record failed knowledge-graph run in PostgreSQL; "
                "preserving original error",
                step="record_failed_run",
                run_id=run_id,
                error_type=record_error.__class__.__name__,
                error=str(record_error),
                status="warning",
            )
