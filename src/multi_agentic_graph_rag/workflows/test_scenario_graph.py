"""Stage 3 LangGraph test-scenario generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.test_scenario_agent import (
    TestScenarioGenerationAgent,
)
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
    CanonicalUserStory,
    ChunkManifest,
    LLMTestScenarioCandidate,
    RequirementsArtifact,
    ScenarioContext,
    StageRequest,
    TestScenarioGenerationResponse,
    TestScenariosArtifact,
    UserStoriesArtifact,
)
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.services.checkpointing import workflow_checkpointer
from multi_agentic_graph_rag.services.manifest import atomic_write_model, load_model
from multi_agentic_graph_rag.services.retrieval import RetrievalService
from multi_agentic_graph_rag.services.test_scenario_builder import (
    build_test_scenarios_artifact,
)


class ScenarioGenerationState(TypedDict, total=False):
    """Stage 3 parent state."""

    request: dict[str, Any]
    artifact_dir: str
    requirements: dict[str, Any]
    stories: dict[str, Any]
    manifest: dict[str, Any]
    artifact_path: str
    scenario_ids: list[str]


class ScenarioPipelineState(TypedDict, total=False):
    """Checkpointed per-story state."""

    project: str
    run_id: str
    requirements: dict[str, Any]
    stories: dict[str, Any]
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
    agent: TestScenarioGenerationAgent
    checkpointer: Any


def build_test_scenario_graph(runtime: _Runtime) -> Any:
    """Build TestScenarioGeneration with the required five parent nodes."""
    graph = StateGraph(ScenarioGenerationState)
    graph.add_node("validate_user_request", _validate_user_request)
    graph.add_node(
        "validate_environment_settings",
        lambda state: _validate_environment(state, runtime),
    )
    graph.add_node("load_stories", lambda state: _load_stories(state, runtime))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, runtime))
    graph.add_node(
        "validate_session_exit_gate",
        lambda state: _validate_exit_gate(state, runtime),
    )
    graph.set_entry_point("validate_user_request")
    graph.add_edge("validate_user_request", "validate_environment_settings")
    graph.add_edge("validate_environment_settings", "load_stories")
    graph.add_edge("load_stories", "run_pipeline")
    graph.add_edge("run_pipeline", "validate_session_exit_gate")
    graph.add_edge("validate_session_exit_gate", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _validate_user_request(state: ScenarioGenerationState) -> ScenarioGenerationState:
    request = StageRequest.model_validate(state["request"])
    return {"request": request.model_dump(mode="json")}


def _validate_environment(
    state: ScenarioGenerationState,
    runtime: _Runtime,
) -> ScenarioGenerationState:
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


def _load_stories(
    state: ScenarioGenerationState,
    runtime: _Runtime,
) -> ScenarioGenerationState:
    request = StageRequest.model_validate(state["request"])
    root = Path(state["artifact_dir"])
    story_path = root / "user-stories" / "user-stories.json"
    requirement_path = root / "requirements" / "requirements.json"
    try:
        stories = load_model(story_path, UserStoriesArtifact)
    except (FileNotFoundError, ValueError) as exc:
        stored_stories = runtime.postgres.load_user_stories(request.project_name, request.run_id)
        if stored_stories is None:
            raise ValueError("user stories are unavailable locally and in PostgreSQL") from exc
        stories = stored_stories
        atomic_write_model(stories, story_path)
    try:
        requirements = load_model(requirement_path, RequirementsArtifact)
    except (FileNotFoundError, ValueError) as exc:
        stored_requirements = runtime.postgres.load_requirements(
            request.project_name, request.run_id
        )
        if stored_requirements is None:
            raise ValueError("requirements are unavailable locally and in PostgreSQL") from exc
        requirements = stored_requirements
        atomic_write_model(requirements, requirement_path)
    manifest = load_model(root / "requirements" / "chunk_manifest.json", ChunkManifest)
    if any(
        artifact.project != request.project_name or artifact.run_id != request.run_id
        for artifact in (stories, requirements, manifest)
    ):
        raise ValueError("Stage 3 input project/run scope mismatch")
    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in requirements.requirements
    }
    allowed = {chunk.chunk_id for chunk in manifest.chunks}
    for story in stories.stories:
        linked = [requirements_by_id.get(value) for value in story.requirement_ids]
        if any(item is None for item in linked):
            raise ValueError("story has an unresolved requirement mapping")
        if any(
            item.source_req_id != story.source_req_id
            or item.source_req_id_type != story.source_req_id_type
            for item in linked
            if item is not None
        ):
            raise ValueError("story provenance differs from a linked requirement")
        if not set(story.traceability.evidence_chunk_ids) <= allowed:
            raise ValueError("story evidence is outside the current manifest")
    return {
        "stories": stories.model_dump(mode="json"),
        "requirements": requirements.model_dump(mode="json"),
        "manifest": manifest.model_dump(mode="json"),
    }


def _run_pipeline(
    state: ScenarioGenerationState,
    runtime: _Runtime,
) -> ScenarioGenerationState:
    request = StageRequest.model_validate(state["request"])
    stories = UserStoriesArtifact.model_validate(state["stories"]).stories
    if stories:
        subgraph = _build_scenario_subgraph(runtime)
        final = subgraph.invoke(
            {
                "project": request.project_name,
                "run_id": request.run_id,
                "requirements": state["requirements"],
                "stories": state["stories"],
                "manifest": state["manifest"],
                "current_index": 0,
                "contexts": [],
                "candidates": [],
            },
            config={
                "configurable": {
                    "thread_id": make_checkpoint_thread_id(
                        request.project_name, request.run_id, "stage-3-stories"
                    )
                }
            },
        )
    else:
        final = {"contexts": [], "candidates": []}
    candidates = [
        (
            str(item["story_id"]),
            [str(value) for value in item["requirement_ids"]],
            LLMTestScenarioCandidate.model_validate(item["candidate"]),
        )
        for item in final["candidates"]
    ]
    artifact = build_test_scenarios_artifact(
        project=request.project_name,
        run_id=request.run_id,
        candidates=candidates,
        existing=runtime.postgres.load_project_test_scenarios(request.project_name),
    )
    runtime.postgres.persist_test_scenarios(artifact)
    output_dir = Path(state["artifact_dir"]) / "test-scenario"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenario_context.json").write_text(
        json.dumps({"contexts": final["contexts"]}, indent=2),
        encoding="utf-8",
    )
    artifact_path = atomic_write_model(artifact, output_dir / "test-scenarios.json")
    return {
        "artifact_path": str(artifact_path),
        "scenario_ids": [scenario.scenario_id for scenario in artifact.scenarios],
    }


def _validate_exit_gate(
    state: ScenarioGenerationState,
    runtime: _Runtime,
) -> ScenarioGenerationState:
    request = StageRequest.model_validate(state["request"])
    artifact = load_model(Path(state["artifact_path"]), TestScenariosArtifact)
    persisted = runtime.postgres.load_test_scenarios(request.project_name, request.run_id)
    if persisted is None or persisted.checksum != artifact.checksum:
        raise ValueError("PostgreSQL and local test-scenario checksums differ")
    stories = {
        story.story_id: story
        for story in UserStoriesArtifact.model_validate(state["stories"]).stories
    }
    for scenario in artifact.scenarios:
        for story_id in scenario.story_ids:
            story = stories.get(story_id)
            if story is None:
                raise ValueError("scenario has an unresolved story mapping")
            valid_criteria = {criterion.criterion_id for criterion in story.acceptance_criteria}
            if not set(scenario.covered_acceptance_criterion_ids) <= valid_criteria:
                raise ValueError("scenario covers criteria outside a mapped story")
    return {}


def _build_scenario_subgraph(runtime: _Runtime) -> Any:
    graph = StateGraph(ScenarioPipelineState)
    graph.add_node(
        "retrieve_story_context",
        lambda state: _retrieve_story_context(state, runtime),
    )
    graph.add_node(
        "generate_test_scenarios",
        lambda state: _generate_test_scenarios(state, runtime),
    )
    graph.add_node("record_story_result", _record_story_result)
    graph.set_entry_point("retrieve_story_context")
    graph.add_edge("retrieve_story_context", "generate_test_scenarios")
    graph.add_edge("generate_test_scenarios", "record_story_result")
    graph.add_conditional_edges(
        "record_story_result",
        _route_scenario,
        {"next": "retrieve_story_context", "done": END},
    )
    return graph.compile(checkpointer=runtime.checkpointer)


def _current_story(state: ScenarioPipelineState) -> CanonicalUserStory:
    return UserStoriesArtifact.model_validate(state["stories"]).stories[state["current_index"]]


def _retrieve_story_context(
    state: ScenarioPipelineState,
    runtime: _Runtime,
) -> ScenarioPipelineState:
    if state.get("current_context"):
        return {}
    story = _current_story(state)
    all_requirements = RequirementsArtifact.model_validate(state["requirements"]).requirements
    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in all_requirements
    }
    linked: list[CanonicalRequirement] = [
        requirements_by_id[requirement_id] for requirement_id in story.requirement_ids
    ]
    context = runtime.retrieval.scenario_context(
        project=state["project"],
        manifest=ChunkManifest.model_validate(state["manifest"]),
        story=story,
        requirements=linked,
    )
    runtime.postgres.save_generation_context(
        project=state["project"],
        run_id=state["run_id"],
        stage="test_scenario",
        anchor_id=story.story_id,
        payload=context.model_dump(mode="json"),
    )
    return {"current_context": context.model_dump(mode="json")}


def _generate_test_scenarios(
    state: ScenarioPipelineState,
    runtime: _Runtime,
) -> ScenarioPipelineState:
    if state.get("current_response"):
        return {}
    story = _current_story(state)
    response = runtime.agent.generate(
        story,
        ScenarioContext.model_validate(state["current_context"]),
    )
    return {"current_response": response.model_dump(mode="json")}


def _record_story_result(state: ScenarioPipelineState) -> ScenarioPipelineState:
    story = _current_story(state)
    response = TestScenarioGenerationResponse.model_validate(state["current_response"])
    return {
        "current_index": state["current_index"] + 1,
        "current_context": {},
        "current_response": {},
        "contexts": [*state["contexts"], state["current_context"]],
        "candidates": [
            *state["candidates"],
            *(
                {
                    "story_id": story.story_id,
                    "requirement_ids": story.requirement_ids,
                    "candidate": candidate.model_dump(mode="json"),
                }
                for candidate in response.test_scenarios
            ),
        ],
    }


def _route_scenario(state: ScenarioPipelineState) -> Literal["next", "done"]:
    stories = UserStoriesArtifact.model_validate(state["stories"]).stories
    return "done" if state["current_index"] >= len(stories) else "next"


def run_test_scenario_generation(
    request: StageRequest,
    *,
    settings: AppSettings | None = None,
) -> ArtifactResult:
    """Execute Stage 3 with durable per-story checkpoints."""
    resolved = settings or load_config()
    if request.reasoning_provider is not None:
        resolved.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider is not None:
        resolved.embedding_model.provider = request.embedding_provider
    postgres = PostgresStore(resolved)
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-3",
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
            agent=TestScenarioGenerationAgent(create_reasoning_model(resolved)),
            checkpointer=checkpointer,
        )
        graph = build_test_scenario_graph(runtime)
        try:
            final = graph.invoke(
                {"request": request.model_dump(mode="json")},
                config={
                    "configurable": {
                        "thread_id": make_checkpoint_thread_id(
                            request.project_name, request.run_id, "stage-3"
                        )
                    }
                },
            )
        except Exception as exc:
            postgres.record_run(
                run_id=request.run_id,
                project=request.project_name,
                stage="stage-3",
                status="failed",
                payload={"error_type": exc.__class__.__name__},
            )
            raise
    result = ArtifactResult(
        project=request.project_name,
        run_id=request.run_id,
        artifact_path=Path(final["artifact_path"]),
        item_ids=final["scenario_ids"],
    )
    postgres.record_run(
        run_id=request.run_id,
        project=request.project_name,
        stage="stage-3",
        status="completed",
        payload=result.model_dump(mode="json"),
    )
    return result


__all__ = ["build_test_scenario_graph", "run_test_scenario_generation"]
