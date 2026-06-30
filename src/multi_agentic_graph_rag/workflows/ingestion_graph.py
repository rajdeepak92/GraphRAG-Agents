"""LangGraph orchestration for ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.requirement_discovery_agent import RequirementDiscoveryAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.identifiers import run_id
from multi_agentic_graph_rag.domain.schemas import IngestionRequest, IngestionResult
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.observability.session import RunSession, command_session
from multi_agentic_graph_rag.services.artifacts import write_requirement_artifact
from multi_agentic_graph_rag.services.chunking import chunk_blocks
from multi_agentic_graph_rag.services.manifest import build_manifest, write_manifest
from multi_agentic_graph_rag.services.parsing import checksum_bytes, parse_document
from multi_agentic_graph_rag.services.requirement_builder import build_requirement_artifact


class IngestionState(TypedDict, total=False):
    request: dict[str, Any]
    run_id: str
    checksum: str
    document_id: str
    document_version_id: str
    manifest_path: str
    artifact_path: str
    chunk_ids: list[str]
    fact_ids: list[str]
    requirement_ids: list[str]
    warnings: list[str]
    errors: list[str]


def build_ingestion_graph(session: RunSession | None = None) -> Any:
    graph = StateGraph(IngestionState)
    graph.add_node("validate_request", lambda state: _validate_request(state, session=session))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, session=session))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()


def _validate_request(
    state: IngestionState,
    *,
    session: RunSession | None = None,
) -> IngestionState:
    request = IngestionRequest.model_validate(state["request"])
    logger = session.logger if session is not None else None
    if logger is not None:
        logger.debug(
            "Validated ingestion request for {project}:{version}",
            step="validate_request",
            project=request.project,
            version=request.version,
            document=str(request.document),
        )
    if not request.document.exists():
        raise FileNotFoundError(request.document)
    rid = state.get("run_id") or run_id(request.project, request.document, request.version)
    if logger is not None:
        logger.debug(
            "Resolved run id {run_id}",
            step="validate_request",
            run_id=rid,
            document=str(request.document),
        )
    return {"request": request.model_dump(mode="json"), "run_id": rid, "warnings": [], "errors": []}


def _run_pipeline(
    state: IngestionState,
    *,
    session: RunSession | None = None,
) -> IngestionState:
    request = IngestionRequest.model_validate(state["request"])
    settings = load_config()
    if session is not None:
        session.set_log_level(settings.log_level)
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider:
        settings.embedding_model.provider = request.embedding_provider
    logger = session.logger if session is not None else None
    project = request.project
    version = request.version
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = ChromaStore(settings)
    run_dir = (
        session.run_dir
        if session is not None
        else _ingest_run_dir(settings, project, state["run_id"])
    )

    def pipeline() -> IngestionState:
        _validate_required_ingest_stack(settings)
        if logger is not None:
            logger.info(
                "Beginning ingestion pipeline for {project}:{version}",
                step="run_pipeline",
                project=project,
                version=version,
                run_id=state["run_id"],
                status="started",
            )
            logger.debug(
                "Runtime paths resolved",
                step="run_pipeline",
                project=project,
                version=version,
                runtime_root=str(settings.paths.project_root),
                staging_dir=str(settings.paths.runtime_staging_dir),
                output_dir=str(run_dir),
            )
        if logger is not None:
            logger.debug(
                "Checking postgres store",
                step="check_postgres",
                project=project,
                version=version,
            )
        postgres_status = postgres.check()
        if logger is not None:
            logger.info(
                postgres_status,
                step="check_postgres",
                project=project,
                version=version,
                status="PASS",
                detail=postgres_status,
            )
            logger.debug(
                "Checking neo4j store",
                step="check_neo4j",
                project=project,
                version=version,
            )
        neo4j_status = neo4j.check()
        if logger is not None:
            logger.info(
                neo4j_status,
                step="check_neo4j",
                project=project,
                version=version,
                status="PASS",
                detail=neo4j_status,
            )
            logger.debug(
                "Checking chroma store",
                step="check_chroma",
                project=project,
                version=version,
            )
        chroma_status = chroma.check()
        if logger is not None:
            logger.info(
                chroma_status,
                step="check_chroma",
                project=project,
                version=version,
                status="PASS",
                detail=chroma_status,
            )
            logger.debug(
                "Ensuring postgres schema",
                step="ensure_postgres_schema",
                project=project,
                version=version,
            )
        postgres.ensure_schema()
        reasoning_model = create_reasoning_model(
            settings,
            logger=logger,
            run_dir=run_dir,
        )
        embedding_model = create_embedding_model(settings)
        reranker_model = create_reranker_model(settings)
        _warmup_reasoning_model(reasoning_model)
        if logger is not None:
            logger.info(
                "Model readiness checks completed",
                step="check_models",
                reasoning_provider=reasoning_model.provider_name,
                embedding_provider=embedding_model.provider_name,
                reranker_provider=reranker_model.provider_name,
                huggingface_reasoning_model=settings.huggingface.reasoning_model,
                huggingface_embedding_model=settings.huggingface.embedding_model,
                huggingface_reranker_model=settings.huggingface.reranker_model,
                status="PASS",
            )

        source_bytes = request.document.read_bytes()
        checksum = checksum_bytes(source_bytes)
        if logger is not None:
            logger.debug(
                "Read source document {path} ({byte_count} bytes)",
                step="read_document",
                path=str(request.document),
                byte_count=len(source_bytes),
            )
        blocks, parser_fingerprint = parse_document(request.document, logger=logger)
        logical_name = request.logical_name or request.document.stem
        preliminary = build_manifest(
            project=project,
            logical_name=logical_name,
            version=version,
            source_path=request.document,
            source_checksum=checksum,
            parser_fingerprint=parser_fingerprint,
            chunker_fingerprint="pending",
            chunks=[],
            logger=logger,
        )
        chunks, chunker_fingerprint = chunk_blocks(
            document_version_id=preliminary.document_version_id,
            blocks=blocks,
            settings=settings.chunking,
            logger=logger,
        )
        manifest = build_manifest(
            project=project,
            logical_name=logical_name,
            version=version,
            source_path=request.document,
            source_checksum=checksum,
            parser_fingerprint=parser_fingerprint,
            chunker_fingerprint=chunker_fingerprint,
            chunks=chunks,
            logger=logger,
        )
        if logger is not None:
            logger.debug(
                "Resolving version policy for {document_version_id}",
                step="resolve_version",
                document_version_id=manifest.document_version_id,
                replace_version=request.replace_version,
            )
        postgres.assert_version_allowed(manifest, request.replace_version)
        manifest_path = write_manifest(
            manifest,
            run_dir,
            logger=logger,
        )
        if logger is not None:
            logger.info(
                "Projecting document/chunk graph to Neo4j; "
                "generated requirements stay out of Neo4j",
                step="project_chunks_neo4j",
                document_version_id=manifest.document_version_id,
                chunk_count=len(manifest.chunks),
                store_responsibility="document_chunk_graph_only",
            )
        neo4j.project_manifest(manifest)
        if logger is not None:
            logger.info(
                "Indexing {chunk_count} chunk embeddings into Chroma; "
                "requirement data stays out of Chroma",
                step="index_chunks_chroma",
                chunk_count=len(manifest.chunks),
                collection=settings.chroma.collection_name,
                store_responsibility="chunk_embeddings_only",
            )
        chroma.index_chunks(manifest, embedding_model)
        discovery_agent = RequirementDiscoveryAgent(reasoning_model, logger=logger)
        if logger is not None:
            logger.info(
                "Discovering requirements from {document_version_id}",
                step="discover_requirements",
                document_version_id=manifest.document_version_id,
            )
        discovery = discovery_agent.run(manifest)
        if logger is not None:
            logger.debug(
                "Loaded requirement ledger snapshot for {document_id}",
                step="load_requirement_ledger_snapshot",
                document_id=manifest.document_id,
            )
        prior_revisions = postgres.load_requirement_revision_snapshot(
            project=project,
            document_id=manifest.document_id,
        )
        artifact = build_requirement_artifact(
            project=project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=version,
            source_path=manifest.source_path,
            source_checksum=checksum,
            discovery=discovery,
            prior_revisions=prior_revisions,
            logger=logger,
        )
        if session is not None:
            session.artifact_payload = artifact.model_dump(mode="json")
            artifact_path = session.write_artifact(session.artifact_payload)
        else:
            artifact_path = write_requirement_artifact(
                artifact,
                run_dir,
                logger=logger,
            )
        if logger is not None:
            logger.info(
                "Requirement artifact written to {path}",
                step="write_requirement_artifact",
                path=str(artifact_path),
                document_version_id=artifact.document_version_id,
                status="completed",
            )
        postgres.persist_manifest(manifest)
        if logger is not None:
            logger.info(
                "Persisting generated requirements.json payload and "
                "requirement ledger to PostgreSQL",
                step="persist_requirements_postgres",
                document_version_id=artifact.document_version_id,
                artifact_path=str(artifact_path),
                fact_count=len(artifact.facts),
                requirement_count=len(artifact.requirements),
                store_responsibility="requirement_artifact_and_ledger_only",
            )
        postgres.persist_artifact(artifact, str(artifact_path), state["run_id"])
        result_payload: IngestionState = {
            "run_id": state["run_id"],
            "checksum": checksum,
            "document_id": manifest.document_id,
            "document_version_id": manifest.document_version_id,
            "manifest_path": str(manifest_path),
            "artifact_path": str(artifact_path),
            "chunk_ids": [chunk.chunk_id for chunk in manifest.chunks],
            "fact_ids": [fact.fact_id for fact in artifact.facts],
            "requirement_ids": [req.requirement_id for req in artifact.requirements],
            "warnings": [],
            "errors": [],
        }
        if session is not None:
            session.metadata.update(result_payload)
        postgres.record_run(state["run_id"], "completed", dict(result_payload))
        if logger is not None:
            logger.info(
                "Ingestion pipeline completed for {document_version_id}",
                step="run_pipeline",
                document_version_id=manifest.document_version_id,
                run_id=state["run_id"],
                chunk_count=len(manifest.chunks),
                fact_count=len(artifact.facts),
                requirement_count=len(artifact.requirements),
                status="completed",
            )
        return result_payload

    try:
        return pipeline()
    except Exception as exc:
        if logger is not None:
            logger.exception(
                "Ingestion pipeline failed",
                step="run_pipeline",
                exc=exc,
                status="failed",
            )
        if session is not None:
            session.write_failure_envelope(error=exc)
        error_payload = {
            "run_id": state["run_id"],
            "document_version_id": state.get("document_version_id", ""),
            "error": str(exc),
        }
        _record_failed_run_safely(
            postgres=postgres,
            run_id=state["run_id"],
            payload=error_payload,
            logger=logger,
        )
        raise


def run_ingestion(
    request: IngestionRequest,
    session: RunSession | None = None,
) -> IngestionResult:
    if session is None:
        generated_run_id = run_id(request.project, request.document, request.version)
        with command_session(
            project=request.project,
            version=request.version,
            command="ingest",
            run_id=generated_run_id,
        ) as managed_session:
            return run_ingestion(request, session=managed_session)
    session.request_payload = request.model_dump(mode="json")
    graph = build_ingestion_graph(session)
    final_state = graph.invoke(
        {"request": request.model_dump(mode="json"), "run_id": session.run_id}
    )
    return IngestionResult(
        run_id=final_state["run_id"],
        status="completed",
        project=request.project,
        version=request.version,
        document_id=final_state["document_id"],
        document_version_id=final_state["document_version_id"],
        checksum=final_state["checksum"],
        manifest_path=Path(final_state["manifest_path"]),
        artifact_path=Path(final_state["artifact_path"]),
        chunk_ids=final_state["chunk_ids"],
        fact_ids=final_state["fact_ids"],
        requirement_ids=final_state["requirement_ids"],
        warnings=final_state.get("warnings", []),
        errors=final_state.get("errors", []),
    )


def _ingest_run_dir(settings: AppSettings, project: str, run_identifier: str) -> Path:
    return settings.paths.generated_requirements_dir / project / "req" / run_identifier


def _validate_required_ingest_stack(settings: AppSettings) -> None:
    invalid_reasoning = {"local_heuristic"}
    invalid_embedding = {"local_hash"}
    invalid_reranker = {"none"}
    if settings.reasoning_model.provider in invalid_reasoning:
        raise ConfigurationError(
            f"REASONING_MODEL_PROVIDER={settings.reasoning_model.provider} is not valid for ingest"
        )
    if settings.embedding_model.provider in invalid_embedding:
        raise ConfigurationError(
            f"EMBEDDING_MODEL_PROVIDER={settings.embedding_model.provider} is not valid for ingest"
        )
    if settings.reranker_model.provider in invalid_reranker:
        raise ConfigurationError(
            f"RERANKER_MODEL_PROVIDER={settings.reranker_model.provider} is not valid for ingest"
        )
    if settings.postgres.mode != "postgres":
        raise ConfigurationError("POSTGRES_MODE=postgres is required for ingest")
    if settings.neo4j.mode != "neo4j":
        raise ConfigurationError("NEO4J_MODE=neo4j is required for ingest")


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
                "Could not record failed ingest state in PostgreSQL; preserving original error",
                step="record_failed_run",
                run_id=run_id,
                error_type=record_error.__class__.__name__,
                error=str(record_error),
                status="warning",
            )
