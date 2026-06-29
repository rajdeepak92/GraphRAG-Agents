"""LangGraph orchestration for ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.requirement_discovery_agent import RequirementDiscoveryAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import run_id
from multi_agentic_graph_rag.domain.schemas import IngestionRequest, IngestionResult
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
)
from multi_agentic_graph_rag.observability.logging import RunLogger
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


def build_ingestion_graph() -> Any:
    graph = StateGraph(IngestionState)
    graph.add_node("validate_request", _validate_request)
    graph.add_node("run_pipeline", _run_pipeline)
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()


def _validate_request(state: IngestionState) -> IngestionState:
    request = IngestionRequest.model_validate(state["request"])
    if not request.document.exists():
        raise FileNotFoundError(request.document)
    rid = state.get("run_id") or run_id(request.project, request.document, request.version)
    return {"request": request.model_dump(mode="json"), "run_id": rid, "warnings": [], "errors": []}


def _run_pipeline(state: IngestionState) -> IngestionState:
    request = IngestionRequest.model_validate(state["request"])
    settings = load_config()
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider:
        settings.embedding_model.provider = request.embedding_provider
    logger = RunLogger(settings.paths.runtime_logs_dir, state["run_id"])
    project = request.project
    version = request.version
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = ChromaStore(settings)

    def pipeline() -> IngestionState:
        logger.node(project=project, version=version, step="bootstrap_runtime", fn=lambda: None)
        logger.node(project=project, version=version, step="check_postgres", fn=postgres.check)
        logger.node(project=project, version=version, step="check_neo4j", fn=neo4j.check)
        logger.node(project=project, version=version, step="check_chroma", fn=chroma.check)
        logger.node(
            project=project,
            version=version,
            step="ensure_postgres_schema",
            fn=postgres.ensure_schema,
        )

        source_bytes = request.document.read_bytes()
        checksum = checksum_bytes(source_bytes)
        blocks, parser_fingerprint = logger.node(
            project=project,
            version=version,
            step="parse_document",
            fn=lambda: parse_document(request.document),
        )
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
        )
        chunks, chunker_fingerprint = logger.node(
            project=project,
            version=version,
            step="chunk_document",
            fn=lambda: chunk_blocks(
                document_version_id=preliminary.document_version_id,
                blocks=blocks,
                settings=settings.chunking,
            ),
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
        )
        logger.node(
            project=project,
            version=version,
            step="resolve_version",
            fn=lambda: postgres.assert_version_allowed(manifest, request.replace_version),
        )
        manifest_path = logger.node(
            project=project,
            version=version,
            step="write_manifest",
            fn=lambda: write_manifest(
                manifest, settings.paths.runtime_staging_dir, state["run_id"]
            ),
        )
        logger.node(
            project=project,
            version=version,
            step="project_chunks_neo4j",
            fn=lambda: neo4j.project_manifest(manifest),
        )
        embedding_model = create_embedding_model(settings)
        logger.node(
            project=project,
            version=version,
            step="index_chunks_chroma",
            fn=lambda: chroma.index_chunks(manifest, embedding_model),
        )
        discovery_agent = RequirementDiscoveryAgent(create_reasoning_model(settings))
        discovery = logger.node(
            project=project,
            version=version,
            step="discover_requirements",
            fn=lambda: discovery_agent.run(manifest),
        )
        artifact = logger.node(
            project=project,
            version=version,
            step="build_requirement_artifact",
            fn=lambda: build_requirement_artifact(
                project=project,
                document_id=manifest.document_id,
                document_version_id=manifest.document_version_id,
                version=version,
                source_checksum=checksum,
                discovery=discovery,
            ),
        )
        logger.node(
            project=project,
            version=version,
            step="persist_manifest_postgres",
            fn=lambda: postgres.persist_manifest(manifest),
        )
        artifact_path = logger.node(
            project=project,
            version=version,
            step="write_artifact",
            fn=lambda: write_requirement_artifact(
                artifact, settings.paths.generated_requirements_dir
            ),
        )
        logger.node(
            project=project,
            version=version,
            step="persist_artifact_postgres",
            fn=lambda: postgres.persist_artifact(artifact, str(artifact_path), state["run_id"]),
        )
        logger.node(
            project=project,
            version=version,
            step="project_artifact_neo4j",
            fn=lambda: neo4j.project_artifact(artifact),
        )
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
        logger.node(
            project=project,
            version=version,
            step="record_run_postgres",
            fn=lambda: postgres.record_run(state["run_id"], "completed", dict(result_payload)),
        )
        return result_payload

    try:
        return pipeline()
    except Exception as exc:
        error_payload = {
            "run_id": state["run_id"],
            "document_version_id": state.get("document_version_id", ""),
            "error": str(exc),
        }
        postgres.record_run(state["run_id"], "failed", error_payload)
        raise


def run_ingestion(request: IngestionRequest) -> IngestionResult:
    graph = build_ingestion_graph()
    final_state = graph.invoke({"request": request.model_dump(mode="json")})
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
