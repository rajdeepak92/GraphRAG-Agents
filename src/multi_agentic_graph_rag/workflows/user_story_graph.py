"""Stage 2 LangGraph user-story generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGenerationAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import (
    make_checkpoint_thread_id,
    normalize_project,
)
from multi_agentic_graph_rag.domain.schemas import (
    ArtifactResult,
    CanonicalRequirement,
    ChunkManifest,
    LLMUserStoryCandidate,
    ProgressItem,
    ProgressReport,
    RequirementsArtifact,
    StageRequest,
    StoryContext,
    UserStoriesArtifact,
    UserStoryGenerationResponse,
)
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.services.checkpointing import workflow_checkpointer
from multi_agentic_graph_rag.services.manifest import atomic_write_model, load_model
from multi_agentic_graph_rag.services.retrieval import RetrievalService
from multi_agentic_graph_rag.services.user_story_builder import (
    build_user_stories_artifact,
)


class UserStoryState(TypedDict, total=False):
    """Stage 2 parent state."""

    request: dict[str, Any]
    artifact_dir: str
    requirements: dict[str, Any]
    manifest: dict[str, Any]
    artifact_path: str
    story_ids: list[str]


class StoryPipelineState(TypedDict, total=False):
    """Checkpointed per-requirement state."""

    project: str
    run_id: str
    requirements: dict[str, Any]
    manifest: dict[str, Any]
    current_index: int
    current_context: dict[str, Any]
    current_response: dict[str, Any]
    contexts: list[dict[str, Any]]
    candidates: list[dict[str, Any]]


@dataclass
class _Runtime:
    settings: AppSettings
    postgres: PostgresStore
    retrieval: RetrievalService
    agent: UserStoryGenerationAgent
    checkpointer: Any


def build_user_story_graph(runtime: _Runtime) -> Any:
    """Build UserStoryGeneration with the required five parent nodes."""
    graph = StateGraph(UserStoryState)
    graph.add_node("validate_user_request", _validate_user_request)
    graph.add_node(
        "validate_environment_settings",
        lambda state: _validate_environment(state, runtime),
    )
    graph.add_node("load_requirements", lambda state: _load_requirements(state, runtime))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, runtime))
    graph.add_node(
        "validate_session_exit_gate",
        lambda state: _validate_exit_gate(state, runtime),
    )
    graph.set_entry_point("validate_user_request")
    graph.add_edge("validate_user_request", "validate_environment_settings")
    graph.add_edge("validate_environment_settings", "load_requirements")
    graph.add_edge("load_requirements", "run_pipeline")
    graph.add_edge("run_pipeline", "validate_session_exit_gate")
    graph.add_edge("validate_session_exit_gate", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _validate_user_request(state: UserStoryState) -> UserStoryState:
    request = StageRequest.model_validate(state["request"])
    return {"request": request.model_dump(mode="json")}


def _validate_environment(
    state: UserStoryState,
    runtime: _Runtime,
) -> UserStoryState:
    request = StageRequest.model_validate(state["request"])
    readiness = runtime.postgres.get_readiness(request.project_name)
    if readiness is None or readiness.status != "ready" or readiness.build_run_id != request.run_id:
        raise ValueError("knowledge graph is not ready for the selected project/run")
    runtime.postgres.check()
    runtime.retrieval.neo4j.check()
    runtime.retrieval.chroma.check(request.project_name)
    artifact_dir = (
        runtime.settings.paths.generated_dir
        / normalize_project(request.project_name)
        / request.run_id
    )
    return {"artifact_dir": str(artifact_dir)}


def _load_requirements(
    state: UserStoryState,
    runtime: _Runtime,
) -> UserStoryState:
    request = StageRequest.model_validate(state["request"])
    requirements_path = Path(state["artifact_dir"]) / "requirements" / "requirements.json"
    try:
        requirements = load_model(requirements_path, RequirementsArtifact)
    except (FileNotFoundError, ValueError) as exc:
        stored_requirements = runtime.postgres.load_requirements(
            request.project_name, request.run_id
        )
        if stored_requirements is None:
            raise ValueError("requirements are unavailable locally and in PostgreSQL") from exc
        requirements = stored_requirements
        atomic_write_model(requirements, requirements_path)
    manifest = load_model(
        Path(state["artifact_dir"]) / "requirements" / "chunk_manifest.json",
        ChunkManifest,
    )
    if (
        requirements.project != request.project_name
        or requirements.run_id != request.run_id
        or manifest.project != request.project_name
        or manifest.run_id != request.run_id
    ):
        raise ValueError("Stage 2 input project/run scope mismatch")
    allowed = {chunk.chunk_id for chunk in manifest.chunks}
    for requirement in requirements.requirements:
        evidence_ids = {evidence.chunk_id for evidence in requirement.evidence}
        if not evidence_ids <= allowed:
            raise ValueError("requirement evidence is outside the current manifest")
    return {
        "requirements": requirements.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
    }


def _run_pipeline(
    state: UserStoryState,
    runtime: _Runtime,
) -> UserStoryState:
    request = StageRequest.model_validate(state["request"])
    requirements = RequirementsArtifact.model_validate(state["requirements"]).requirements
    if requirements:
        subgraph = _build_story_subgraph(runtime)
        final = subgraph.invoke(
            {
                "project": request.project_name,
                "run_id": request.run_id,
                "requirements": state["requirements"],
                "manifest": state["manifest"],
                "current_index": 0,
                "contexts": [],
                "candidates": [],
            },
            config={
                "configurable": {
                    "thread_id": make_checkpoint_thread_id(
                        request.project_name, request.run_id, "stage-2-requirements"
                    )
                }
            },
        )
    else:
        final = {"contexts": [], "candidates": []}
    candidates = [
        (
            str(item["requirement_id"]),
            LLMUserStoryCandidate.model_validate(item["candidate"]),
        )
        for item in final["candidates"]
    ]
    artifact = build_user_stories_artifact(
        project=request.project_name,
        run_id=request.run_id,
        candidates=candidates,
        existing=runtime.postgres.load_project_user_stories(request.project_name),
    )
    runtime.postgres.persist_user_stories(artifact)
    output_dir = Path(state["artifact_dir"]) / "user-stories"
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "story_context.json"
    context_path.write_text(
        json.dumps({"contexts": final["contexts"]}, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = {requirement.requirement_id: 0 for requirement in requirements}
    for requirement_id, _candidate in candidates:
        counts[requirement_id] = counts.get(requirement_id, 0) + 1
    progress = ProgressReport(
        stage="user_story",
        project=request.project_name,
        run_id=request.run_id,
        items=[
            ProgressItem(
                anchor_id=requirement_id,
                status="generated" if count else "no_story",
                candidate_count=count,
            )
            for requirement_id, count in counts.items()
        ],
    )
    atomic_write_model(progress, output_dir / "progress_story.json")
    artifact_path = atomic_write_model(artifact, output_dir / "user-stories.json")
    return {
        "artifact_path": str(artifact_path),
        "story_ids": [story.story_id for story in artifact.stories],
    }


def _validate_exit_gate(
    state: UserStoryState,
    runtime: _Runtime,
) -> UserStoryState:
    request = StageRequest.model_validate(state["request"])
    artifact = load_model(Path(state["artifact_path"]), UserStoriesArtifact)
    persisted = runtime.postgres.load_user_stories(request.project_name, request.run_id)
    if persisted is None or persisted.checksum != artifact.checksum:
        raise ValueError("PostgreSQL and local user-story checksums differ")
    requirement_ids = {
        item.requirement_id
        for item in RequirementsArtifact.model_validate(state["requirements"]).requirements
    }
    if any(not set(story.requirement_ids) <= requirement_ids for story in artifact.stories):
        raise ValueError("user-story requirement traceability is invalid")
    return {}


def _build_story_subgraph(runtime: _Runtime) -> Any:
    graph = StateGraph(StoryPipelineState)
    graph.add_node(
        "retrieve_requirement_context",
        lambda state: _retrieve_requirement_context(state, runtime),
    )
    graph.add_node(
        "generate_user_stories",
        lambda state: _generate_user_stories(state, runtime),
    )
    graph.add_node("record_requirement_result", _record_requirement_result)
    graph.set_entry_point("retrieve_requirement_context")
    graph.add_edge("retrieve_requirement_context", "generate_user_stories")
    graph.add_edge("generate_user_stories", "record_requirement_result")
    graph.add_conditional_edges(
        "record_requirement_result",
        _route_story,
        {"next": "retrieve_requirement_context", "done": END},
    )
    return graph.compile(checkpointer=runtime.checkpointer)


def _current_requirement(state: StoryPipelineState) -> CanonicalRequirement:
    artifact = RequirementsArtifact.model_validate(state["requirements"])
    return artifact.requirements[state["current_index"]]


def _retrieve_requirement_context(
    state: StoryPipelineState,
    runtime: _Runtime,
) -> StoryPipelineState:
    if state.get("current_context"):
        return {}
    requirement = _current_requirement(state)
    context = runtime.retrieval.story_context(
        project=state["project"],
        manifest=ChunkManifest.model_validate(state["manifest"]),
        requirement=requirement,
    )
    runtime.postgres.save_generation_context(
        project=state["project"],
        run_id=state["run_id"],
        stage="user_story",
        anchor_id=requirement.requirement_id,
        payload=context.model_dump(mode="json"),
    )
    return {"current_context": context.model_dump(mode="json")}


def _generate_user_stories(
    state: StoryPipelineState,
    runtime: _Runtime,
) -> StoryPipelineState:
    if state.get("current_response"):
        return {}
    requirement = _current_requirement(state)
    response = runtime.agent.generate(
        requirement,
        StoryContext.model_validate(state["current_context"]),
    )
    return {"current_response": response.model_dump(mode="json")}


def _record_requirement_result(state: StoryPipelineState) -> StoryPipelineState:
    requirement = _current_requirement(state)
    response = UserStoryGenerationResponse.model_validate(state["current_response"])
    candidates = [
        *state["candidates"],
        *(
            {
                "requirement_id": requirement.requirement_id,
                "candidate": candidate.model_dump(mode="json"),
            }
            for candidate in response.user_stories
        ),
    ]
    return {
        "current_index": state["current_index"] + 1,
        "current_context": {},
        "current_response": {},
        "contexts": [*state["contexts"], state["current_context"]],
        "candidates": candidates,
    }


def _route_story(state: StoryPipelineState) -> Literal["next", "done"]:
    requirements = RequirementsArtifact.model_validate(state["requirements"]).requirements
    return "done" if state["current_index"] >= len(requirements) else "next"


def run_user_story_generation(
    request: StageRequest,
    *,
    settings: AppSettings | None = None,
) -> ArtifactResult:
    """Execute Stage 2 with durable per-requirement checkpoints."""
    resolved = settings or load_config()
    if request.reasoning_provider is not None:
        resolved.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider is not None:
        resolved.embedding_model.provider = request.embedding_provider
    postgres = PostgresStore(resolved)
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-2",
        status="started",
        payload={},
    )
    with workflow_checkpointer(resolved) as checkpointer:
        neo4j = Neo4jStore(resolved)
        runtime = _Runtime(
            settings=resolved,
            postgres=postgres,
            retrieval=RetrievalService(
                neo4j=neo4j,
                chroma=ChromaStore(resolved),
                embedding=create_embedding_model(resolved),
                reranker=create_reranker_model(resolved),
                settings=resolved.retrieval,
            ),
            agent=UserStoryGenerationAgent(create_reasoning_model(resolved)),
            checkpointer=checkpointer,
        )
        graph = build_user_story_graph(runtime)
        try:
            final = graph.invoke(
                {"request": request.model_dump(mode="json")},
                config={
                    "configurable": {
                        "thread_id": make_checkpoint_thread_id(
                            request.project_name, request.run_id, "stage-2"
                        )
                    }
                },
            )
        except Exception as exc:
            postgres.record_run(
                run_id=request.run_id,
                project=request.project_name,
                stage="stage-2",
                status="failed",
                payload={"error_type": exc.__class__.__name__},
            )
            raise
    result = ArtifactResult(
        project=request.project_name,
        run_id=request.run_id,
        artifact_path=Path(final["artifact_path"]),
        item_ids=final["story_ids"],
    )
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-2",
        status="completed",
        payload=result.model_dump(mode="json"),
    )
    return result


__all__ = ["build_user_story_graph", "run_user_story_generation"]
