"""Stage 1.2 LangGraph: combined discovery, semantic projection, and canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    RequirementDiscoveryAgent,
)
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import (
    make_checkpoint_thread_id,
    normalize_project,
)
from multi_agentic_graph_rag.domain.schemas import (
    ArtifactResult,
    CanonicalEntity,
    CanonicalRelationship,
    ChunkManifest,
    KnowledgeGraphReadiness,
    ManifestChunk,
    RequirementChunkResult,
    RequirementDiscoveryChunkResponse,
    RequirementEntityRelationshipMap,
    RequirementsArtifact,
    StageRequest,
)
from multi_agentic_graph_rag.llm_models.factory import create_reasoning_model
from multi_agentic_graph_rag.observability.logging import RunLogger, get_logger
from multi_agentic_graph_rag.services.checkpointing import workflow_checkpointer
from multi_agentic_graph_rag.services.knowledge_graph_builder import KnowledgeGraphBuilder
from multi_agentic_graph_rag.services.manifest import atomic_write_model, load_model
from multi_agentic_graph_rag.services.requirement_builder import build_requirements_artifact
from multi_agentic_graph_rag.services.retry import retry_transient_once

_LOG = get_logger(__name__)


class DiscoveryState(TypedDict, total=False):
    """Stage 1.2 parent state."""

    request: dict[str, Any]
    artifact_dir: str
    manifest: dict[str, Any]
    artifact_path: str
    requirement_ids: list[str]


class DiscoveryPipelineState(TypedDict, total=False):
    """Checkpointed per-manifest-chunk state."""

    project: str
    run_id: str
    manifest: dict[str, Any]
    current_index: int
    current_response: dict[str, Any]
    validated_response: dict[str, Any]
    current_projection: dict[str, Any]
    current_failure: dict[str, Any]
    chunk_results: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


@dataclass
class _Runtime:
    settings: AppSettings
    postgres: PostgresStore
    neo4j: Neo4jStore
    agent: RequirementDiscoveryAgent
    checkpointer: Any


def build_requirement_discovery_graph(runtime: _Runtime) -> Any:
    """Build the four-node RequirementDiscoveryRun graph."""
    graph = StateGraph(DiscoveryState)
    graph.add_node(
        "validate_session_input_stack",
        lambda state: _validate_session_input_stack(state, runtime),
    )
    graph.add_node("load_chunk_manifest", _load_chunk_manifest)
    graph.add_node(
        "run_requirement_pipeline",
        lambda state: _run_requirement_pipeline(state, runtime),
    )
    graph.add_node(
        "validate_session_exit_gate",
        lambda state: _validate_exit_gate(state, runtime),
    )
    graph.set_entry_point("validate_session_input_stack")
    graph.add_edge("validate_session_input_stack", "load_chunk_manifest")
    graph.add_edge("load_chunk_manifest", "run_requirement_pipeline")
    graph.add_edge("run_requirement_pipeline", "validate_session_exit_gate")
    graph.add_edge("validate_session_exit_gate", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _validate_session_input_stack(
    state: DiscoveryState,
    runtime: _Runtime,
) -> DiscoveryState:
    request = StageRequest.model_validate(state["request"])
    artifact_dir = (
        runtime.settings.paths.generated_dir
        / normalize_project(request.project_name)
        / request.run_id
        / "requirements"
    )
    manifest_path = artifact_dir / "chunk_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    runtime.postgres.check()
    runtime.postgres.ensure_schema()
    runtime.neo4j.check()
    runtime.neo4j.ensure_schema()
    if (
        runtime.settings.reasoning_model.provider == "huggingface"
        and runtime.settings.runtime.concurrency != 1
    ):
        raise ValueError("CPU Hugging Face Stage 1.2 requires WORKFLOW_CONCURRENCY=1")
    runtime.postgres.set_readiness(
        KnowledgeGraphReadiness(
            project=request.project_name,
            status="building",
            build_run_id=request.run_id,
            failure_reason=None,
        )
    )
    return {"request": request.model_dump(mode="json"), "artifact_dir": str(artifact_dir)}


def _load_chunk_manifest(state: DiscoveryState) -> DiscoveryState:
    request = StageRequest.model_validate(state["request"])
    path = Path(state["artifact_dir"]) / "chunk_manifest.json"
    manifest = load_model(path, ChunkManifest)
    if manifest.project != request.project_name or manifest.run_id != request.run_id:
        raise ValueError("chunk manifest project/run mismatch")
    return {"manifest": manifest.model_dump(mode="json")}


def _run_requirement_pipeline(
    state: DiscoveryState,
    runtime: _Runtime,
) -> DiscoveryState:
    request = StageRequest.model_validate(state["request"])
    subgraph = _build_discovery_subgraph(runtime)
    child_thread = make_checkpoint_thread_id(
        request.project_name, request.run_id, "stage-1.2-chunks"
    )
    with runtime.postgres.discovery_lease(request.project_name):
        final = subgraph.invoke(
            {
                "project": request.project_name,
                "run_id": request.run_id,
                "manifest": state["manifest"],
                "current_index": 0,
                "chunk_results": [],
                "entities": [],
                "relationships": [],
            },
            config={"configurable": {"thread_id": child_thread}},
        )
        requirement_map = RequirementEntityRelationshipMap(
            project=request.project_name,
            run_id=request.run_id,
            chunk_results=[
                RequirementChunkResult.model_validate(item) for item in final["chunk_results"]
            ],
        )
        artifact = build_requirements_artifact(
            project=request.project_name,
            run_id=request.run_id,
            requirement_map=requirement_map,
            entities=[CanonicalEntity.model_validate(item) for item in final["entities"]],
            relationships=[
                CanonicalRelationship.model_validate(item) for item in final["relationships"]
            ],
            existing=runtime.postgres.load_project_requirements(request.project_name),
        )
        runtime.postgres.persist_requirements(artifact)
        path = atomic_write_model(
            artifact,
            Path(state["artifact_dir"]) / "requirements.json",
        )
    return {
        "artifact_path": str(path),
        "requirement_ids": [item.requirement_id for item in artifact.requirements],
    }


def _validate_exit_gate(
    state: DiscoveryState,
    runtime: _Runtime,
) -> DiscoveryState:
    request = StageRequest.model_validate(state["request"])
    artifact = load_model(Path(state["artifact_path"]), RequirementsArtifact)
    persisted = runtime.postgres.load_requirements(request.project_name, request.run_id)
    if persisted is None or persisted.checksum != artifact.checksum:
        raise ValueError("PostgreSQL and local requirements checksums differ")
    if set(state["requirement_ids"]) != {
        requirement.requirement_id for requirement in artifact.requirements
    }:
        raise ValueError("requirement IDs differ from completed state")
    runtime.postgres.set_readiness(
        KnowledgeGraphReadiness(
            project=request.project_name,
            status="ready",
            build_run_id=request.run_id,
            failure_reason=None,
        )
    )
    return {}


def _build_discovery_subgraph(runtime: _Runtime) -> Any:
    graph = StateGraph(DiscoveryPipelineState)
    graph.add_node(
        "call_combined_reasoning_model",
        lambda state: _call_combined_model(state, runtime),
    )
    graph.add_node("checkpoint_validated_response", _checkpoint_validated_response)
    graph.add_node(
        "project_chunk_semantics",
        lambda state: _project_chunk_semantics(state, runtime),
    )
    graph.add_node("record_chunk_result", _record_chunk_result)
    graph.add_node("raise_terminal_failure", _raise_terminal_failure)
    graph.set_entry_point("call_combined_reasoning_model")
    graph.add_edge("call_combined_reasoning_model", "checkpoint_validated_response")
    graph.add_edge("checkpoint_validated_response", "project_chunk_semantics")
    graph.add_edge("project_chunk_semantics", "record_chunk_result")
    graph.add_conditional_edges(
        "record_chunk_result",
        _route_discovery,
        {
            "next": "call_combined_reasoning_model",
            "done": END,
            "failed": "raise_terminal_failure",
        },
    )
    graph.add_edge("raise_terminal_failure", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _manifest_chunk(state: DiscoveryPipelineState) -> ManifestChunk:
    manifest = ChunkManifest.model_validate(state["manifest"])
    return manifest.chunks[state["current_index"]]


def _chunk_failure(chunk: ManifestChunk, exc: Exception) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "sequence_index": chunk.sequence_index,
        "error": f"{exc.__class__.__name__}: {exc}",
    }


def _call_combined_model(
    state: DiscoveryPipelineState,
    runtime: _Runtime,
) -> DiscoveryPipelineState:
    if (
        state.get("current_response")
        or state.get("validated_response")
        or state.get("current_failure")
    ):
        return {}
    chunk = _manifest_chunk(state)
    try:
        response = retry_transient_once(lambda: runtime.agent.discover(chunk))
    except Exception as exc:  # transient-exhausted or terminal invalid response
        _LOG.warning("discovery.chunk.failed stage=model chunk_id=%s error=%s", chunk.chunk_id, exc)
        return {"current_failure": _chunk_failure(chunk, exc)}
    return {"current_response": response.model_dump(mode="json")}


def _checkpoint_validated_response(
    state: DiscoveryPipelineState,
) -> DiscoveryPipelineState:
    """Checkpoint the fully schema- and semantically-validated response before Neo4j."""
    if state.get("current_failure") or state.get("validated_response"):
        return {}
    return {"validated_response": dict(state["current_response"]), "current_response": {}}


def _project_chunk_semantics(
    state: DiscoveryPipelineState,
    runtime: _Runtime,
) -> DiscoveryPipelineState:
    if state.get("current_failure") or state.get("current_projection"):
        return {}
    chunk = _manifest_chunk(state)
    response = RequirementDiscoveryChunkResponse.model_validate(state["validated_response"])
    try:
        projection = KnowledgeGraphBuilder(runtime.neo4j).project(
            project=state["project"],
            chunk=chunk,
            response=response,
        )
    except Exception as exc:  # terminal Neo4j projection/read-back failure
        _LOG.warning("discovery.chunk.failed stage=neo4j chunk_id=%s error=%s", chunk.chunk_id, exc)
        return {"validated_response": {}, "current_failure": _chunk_failure(chunk, exc)}
    return {
        "current_projection": {
            "result": projection.result.model_dump(mode="json"),
            "entities": [entity.model_dump(mode="json") for entity in projection.entities],
            "relationships": [
                relationship.model_dump(mode="json") for relationship in projection.relationships
            ],
        }
    }


def _record_chunk_result(state: DiscoveryPipelineState) -> DiscoveryPipelineState:
    failure = state.get("current_failure")
    if failure:
        result = RequirementChunkResult(
            chunk_id=failure["chunk_id"],
            sequence_index=failure["sequence_index"],
            status="failed",
            requirements=[],
            error=failure["error"],
        )
        _LOG.info("discovery.chunk.recorded chunk_id=%s status=failed", failure["chunk_id"])
        return {
            "current_index": state["current_index"] + 1,
            "current_response": {},
            "validated_response": {},
            "current_projection": {},
            "current_failure": {},
            "chunk_results": [*state["chunk_results"], result.model_dump(mode="json")],
        }
    projection = state["current_projection"]
    _LOG.info(
        "discovery.chunk.recorded chunk_id=%s status=%s requirements=%d",
        projection["result"]["chunk_id"],
        projection["result"]["status"],
        len(projection["result"]["requirements"]),
    )
    return {
        "current_index": state["current_index"] + 1,
        "current_response": {},
        "validated_response": {},
        "current_projection": {},
        "current_failure": {},
        "chunk_results": [
            *state["chunk_results"],
            dict(projection["result"]),
        ],
        "entities": [
            *state["entities"],
            *(dict(entity) for entity in projection["entities"]),
        ],
        "relationships": [
            *state["relationships"],
            *(dict(relationship) for relationship in projection["relationships"]),
        ],
    }


def _route_discovery(state: DiscoveryPipelineState) -> Literal["next", "done", "failed"]:
    if state["chunk_results"][-1]["status"] == "failed":
        return "failed"
    manifest = ChunkManifest.model_validate(state["manifest"])
    return "done" if state["current_index"] >= len(manifest.chunks) else "next"


def _raise_terminal_failure(state: DiscoveryPipelineState) -> DiscoveryPipelineState:
    failure = state["chunk_results"][-1]
    raise RuntimeError(f"Stage 1.2 terminal failure for {failure['chunk_id']}: {failure['error']}")


def run_requirement_discovery(
    request: StageRequest,
    *,
    settings: AppSettings | None = None,
) -> ArtifactResult:
    """Execute Stage 1.2 and mark readiness failed on any required validation error."""
    resolved = settings or load_config()
    if request.reasoning_provider is not None:
        resolved.reasoning_model.provider = request.reasoning_provider
    postgres = PostgresStore(resolved)
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-1.2",
        status="started",
        payload={},
    )
    run_dir = (
        resolved.paths.generated_dir
        / normalize_project(request.project_name)
        / request.run_id
        / "requirements"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("stage-1.2.start project=%s run_id=%s", request.project_name, request.run_id)
    with workflow_checkpointer(resolved) as checkpointer:
        runtime = _Runtime(
            settings=resolved,
            postgres=postgres,
            neo4j=Neo4jStore(resolved),
            agent=RequirementDiscoveryAgent(
                create_reasoning_model(
                    resolved,
                    logger=RunLogger(_LOG),
                    run_dir=run_dir,
                    stage12=True,
                )
            ),
            checkpointer=checkpointer,
        )
        graph = build_requirement_discovery_graph(runtime)
        try:
            final = graph.invoke(
                {"request": request.model_dump(mode="json")},
                config={
                    "configurable": {
                        "thread_id": make_checkpoint_thread_id(
                            request.project_name, request.run_id, "stage-1.2"
                        )
                    }
                },
            )
        except Exception as exc:
            postgres.set_readiness(
                KnowledgeGraphReadiness(
                    project=request.project_name,
                    status="failed",
                    build_run_id=request.run_id,
                    failure_reason=exc.__class__.__name__,
                )
            )
            postgres.record_run(
                run_id=request.run_id,
                project=request.project_name,
                stage="stage-1.2",
                status="failed",
                payload={"error_type": exc.__class__.__name__},
            )
            raise
    result = ArtifactResult(
        project=request.project_name,
        run_id=request.run_id,
        artifact_path=Path(final["artifact_path"]),
        item_ids=final["requirement_ids"],
    )
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-1.2",
        status="completed",
        payload=result.model_dump(mode="json"),
    )
    return result


__all__ = ["build_requirement_discovery_graph", "run_requirement_discovery"]
