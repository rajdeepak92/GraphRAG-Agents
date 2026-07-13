"""LangGraph orchestration for standalone test-scenario generation (stage 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import ValidationError

from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGenerationAgent
from multi_agentic_graph_rag.common_defs import ModeName, ProviderName, RuntimeCommand
from multi_agentic_graph_rag.common_prompt_defs import PromptTestScenarioGeneration
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import (
    CheckpointError,
    ConfigurationError,
    StoreUnavailableError,
)
from multi_agentic_graph_rag.domain.schemas import (
    GenerationTrace,
    RequirementInput,
    TestScenarioArtifact,
    TestScenarioBuildResult,
    TestScenarioModel,
    TestScenarioRecord,
    TestScenarioRequest,
    TestScenarioResult,
    UserStoryArtifact,
    UserStoryRecord,
)
from multi_agentic_graph_rag.io.feedback_io import CLIFeedbackIO, FeedbackIO
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.observability.session import (
    RunSession,
    command_run_id,
    command_session,
)
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.generation_checkpoint import (
    STAGE_TEST_SCENARIO,
    ContextMapEntry,
    append_generation_error,
    context_map_filename,
    hydrate_context_map_entry,
    load_context_map_checkpoint,
    load_generation_progress,
    make_context_map_entry,
    record_completion,
    record_failure,
    trace_from_entry,
    validate_context_map,
    write_context_map_checkpoint,
    write_generation_progress,
)
from multi_agentic_graph_rag.services.knowledge_retrieval import (
    GraphPrimaryDecision,
    build_knowledge_retrieval_config,
    log_graph_primary_decision,
    require_knowledge_graph_when_primary,
)
from multi_agentic_graph_rag.services.requirement_source import (
    RequirementSource,
    load_requirement_source_from_canonical_payload,
    load_requirement_source_local,
)
from multi_agentic_graph_rag.services.retrieval import RetrievalService
from multi_agentic_graph_rag.services.scenario_dedup import DedupConfig, DedupEngine
from multi_agentic_graph_rag.services.semantic_matcher import SemanticMatcher
from multi_agentic_graph_rag.services.test_scenario_builder import (
    build_test_scenario_artifact,
    project_test_scenario_artifact,
)
from multi_agentic_graph_rag.workflows.hfil_node import (
    HFILRuntime,
    hfil_review_node,
    hfil_scenarios_from_state,
    initialize_hfil_state,
    route_hfil,
)


class TestScenarioState(TypedDict, total=False):
    request: dict[str, Any]
    run_id: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: str
    story_count: int
    scenario_ids: list[str]
    coverage: dict[str, list[str]]
    requirement_coverage: dict[str, list[str]]
    warnings: list[str]
    errors: list[str]
    output_dir: str
    artifact_payload: dict[str, Any]
    scenario_record_payloads: dict[str, Any]
    evidence_by_requirement: dict[str, list[str]]
    hfil_enabled: bool
    hfil_done: bool
    hfil_phase: str
    hfil_pending_prompt: str
    hfil_scenarios: list[dict[str, Any]]
    hfil_user_stories: list[dict[str, Any]]
    hfil_last_duplicate_groups: list[dict[str, Any]]
    hfil_messages: list[str]


@dataclass(frozen=True)
class _StorySource:
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    stories: list[UserStoryRecord]


def build_test_scenario_graph(
    session: RunSession | None = None,
    *,
    settings: AppSettings | None = None,
    hfil_enabled: bool | None = None,
    hfil_runtime: HFILRuntime | None = None,
    checkpointer: Any | None = None,
) -> Any:
    settings = settings or load_config()
    enabled = settings.enable_hfil if hfil_enabled is None else hfil_enabled
    graph = StateGraph(TestScenarioState)
    graph.add_node("validate_request", lambda state: _validate_request(state, session=session))
    graph.add_node("generate", lambda state: _generate(state, session=session, settings=settings))
    graph.add_node("validate_and_assign_ids", _validate_and_assign_ids)
    if enabled:
        if hfil_runtime is None:
            raise ConfigurationError("HFIL runtime is required when HFIL is enabled")
        graph.add_node(
            "hfil_review",
            lambda state: hfil_review_node(dict(state), runtime=hfil_runtime),
        )
    graph.add_node("finalize", _finalize)
    graph.add_node("persist", lambda state: _persist(state, session=session, settings=settings))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "generate")
    graph.add_edge("generate", "validate_and_assign_ids")
    # --- HFIL WIRING (optional).
    # Comment out this block OR set settings.enable_hfil=False to disable.
    if enabled:
        graph.add_edge("validate_and_assign_ids", "hfil_review")
        graph.add_conditional_edges(
            "hfil_review",
            route_hfil,
            {
                "loop": "hfil_review",
                "done": "finalize",
            },
        )
    else:
        graph.add_edge("validate_and_assign_ids", "finalize")
    # --- END HFIL WIRING ---
    graph.add_edge("finalize", "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer)


def _validate_request(
    state: TestScenarioState,
    *,
    session: RunSession | None = None,
) -> TestScenarioState:
    request = TestScenarioRequest.model_validate(state["request"])
    logger = session.logger if session is not None else None
    if request.user_stories_path is None and request.document_version_id is None:
        raise ConfigurationError(
            "provide --user-stories <path> or --document-version-id <id> to load user stories"
        )
    if request.user_stories_path is not None and not request.user_stories_path.exists():
        raise FileNotFoundError(request.user_stories_path)
    if request.requirements_path is not None and not request.requirements_path.exists():
        raise FileNotFoundError(request.requirements_path)
    rid = state.get("run_id") or command_run_id(RuntimeCommand.GENERATE_TEST_SCENARIOS.value)
    if logger is not None:
        logger.debug(
            "Validated test-scenario request",
            step="validate_request",
            user_stories_path=(
                str(request.user_stories_path) if request.user_stories_path else None
            ),
            requirements_path=str(request.requirements_path) if request.requirements_path else None,
            document_version_id=request.document_version_id,
            run_id=rid,
        )
    return {"request": request.model_dump(mode="json"), "run_id": rid, "warnings": [], "errors": []}


def _generate(
    state: TestScenarioState,
    *,
    session: RunSession | None,
    settings: AppSettings,
) -> TestScenarioState:
    request = TestScenarioRequest.model_validate(state["request"])
    if session is not None:
        session.set_log_level(settings.log_level)
    _apply_overrides(settings, request)
    logger = session.logger if session is not None else None
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = ChromaStore(settings)
    run_dir = session.run_dir if session is not None else None

    _validate_required_test_scenario_stack(settings)
    if logger is not None:
        logger.info(
            "Beginning test-scenario generation pipeline",
            step="generate",
            run_id=state["run_id"],
            hfil_enabled=settings.enable_hfil,
            status="started",
        )
    _check_store(logger, "check_postgres", postgres.check)
    _check_store(logger, "check_neo4j", neo4j.check)
    _check_store(logger, "check_chroma", chroma.check)
    postgres.ensure_schema()

    reasoning_model = create_reasoning_model(settings, logger=logger, run_dir=run_dir)
    embedding_model = create_embedding_model(settings)
    reranker_model = create_reranker_model(settings)
    _warmup_reasoning_model(reasoning_model)
    _apply_test_scenario_system_message(reasoning_model)
    neo4j.ensure_search_index()
    if logger is not None:
        logger.info(
            "Model readiness checks completed",
            step="check_models",
            reasoning_provider=reasoning_model.provider_name,
            embedding_provider=embedding_model.provider_name,
            reranker_provider=reranker_model.provider_name,
            status="PASS",
        )

    source = _load_story_source(request, postgres, logger)
    warnings: list[str] = []
    requirement_map = _load_requirement_map(request, source, postgres, logger, warnings)
    if logger is not None:
        logger.info(
            "Loaded {story_count} user stories for test-scenario generation",
            step="load_user_stories",
            story_count=len(source.stories),
            requirement_count=len(requirement_map),
            document_version_id=source.document_version_id,
            project=source.project,
        )
    require_knowledge_graph_when_primary(
        settings=settings,
        neo4j=neo4j,
        document_version_id=source.document_version_id,
        primary=settings.knowledge_graph.graph_primary_scenario,
        stage="test-scenario",
    )

    retrieval = RetrievalService(
        chroma=chroma,
        neo4j=neo4j,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        settings=settings.test_scenario,
        logger=logger,
        knowledge=build_knowledge_retrieval_config(
            settings,
            stage=STAGE_TEST_SCENARIO,
            project=source.project,
            primary=settings.knowledge_graph.graph_primary_scenario,
            recorder=postgres.record_generation_context,
        ),
    )
    agent = TestScenarioGenerationAgent(reasoning_model, logger=logger)
    out_dir = _output_dir(request, settings, source, state["run_id"])
    evidence_by_requirement: dict[str, list[str]] = {
        story.requirement_id: (
            list(requirement_map[story.requirement_id].evidence_chunk_ids)
            if story.requirement_id in requirement_map
            else []
        )
        for story in source.stories
    }

    entries = _build_or_load_context_map(
        source=source,
        requirement_map=requirement_map,
        retrieval=retrieval,
        out_dir=out_dir,
        run_id=state["run_id"],
        logger=logger,
    )
    generated = _generate_from_context_map(
        source=source,
        requirement_map=requirement_map,
        entries=entries,
        agent=agent,
        neo4j=neo4j,
        out_dir=out_dir,
        run_id=state["run_id"],
        logger=logger,
    )

    build_result = build_test_scenario_artifact(
        project=source.project,
        document_id=source.document_id,
        document_version_id=source.document_version_id,
        doc_version=source.doc_version,
        generated=generated,
        traces=_traces_by_story(entries),
    )
    build_result = _preserve_existing_scenario_ids(
        build_result,
        postgres.load_test_scenarios_for_generation(project=source.project),
    )
    next_state: dict[str, Any] = {
        **state,
        "project": source.project,
        "document_id": source.document_id,
        "document_version_id": source.document_version_id,
        "doc_version": source.doc_version,
        "output_dir": str(out_dir),
        "artifact_path": str(out_dir / "test_scenarios.json"),
        "artifact_payload": build_result.artifact.model_dump(mode="json"),
        "scenario_record_payloads": {
            scenario_id: record.model_dump(mode="json")
            for scenario_id, record in build_result.records.items()
        },
        "story_count": len(source.stories),
        "scenario_ids": list(build_result.records),
        "coverage": build_result.coverage,
        "requirement_coverage": build_result.requirement_coverage,
        "evidence_by_requirement": evidence_by_requirement,
        "warnings": warnings,
        "errors": [],
    }
    initialized = initialize_hfil_state(
        state=next_state,
        scenarios=list(build_result.records.values()),
        user_stories=source.stories,
        enabled=settings.enable_hfil,
    )
    return cast(TestScenarioState, initialized)


def _validate_and_assign_ids(state: TestScenarioState) -> TestScenarioState:
    TestScenarioArtifact.model_validate(state["artifact_payload"])
    _scenario_records_from_state(state)
    return state


def _finalize(state: TestScenarioState) -> TestScenarioState:
    artifact = TestScenarioArtifact.model_validate(state["artifact_payload"])
    if state.get("hfil_enabled"):
        scenarios = hfil_scenarios_from_state(dict(state))
        records = {scenario.scenario_id: scenario for scenario in scenarios}
        artifact = project_test_scenario_artifact(
            project=artifact.project,
            document_id=artifact.document_id,
            document_version_id=artifact.document_version_id,
            doc_version=artifact.doc_version,
            records=records,
        )
        artifact = artifact.model_copy(update={"updated_at": datetime.now(UTC)})
    else:
        records = _scenario_records_from_state(state)
    payload = artifact.model_dump(mode="json")
    return {
        **state,
        "artifact_payload": payload,
        "scenario_record_payloads": {
            scenario_id: record.model_dump(mode="json") for scenario_id, record in records.items()
        },
        "scenario_ids": list(records),
        "coverage": _coverage_by_story(list(records.values())),
        "requirement_coverage": _coverage_by_requirement(list(records.values())),
    }


def _persist(
    state: TestScenarioState,
    *,
    session: RunSession | None,
    settings: AppSettings,
) -> TestScenarioState:
    request = TestScenarioRequest.model_validate(state["request"])
    _apply_overrides(settings, request)
    logger = session.logger if session is not None else None
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    artifact = TestScenarioArtifact.model_validate(state["artifact_payload"])
    records = _scenario_records_from_state(state)
    build_result = TestScenarioBuildResult(
        artifact=artifact,
        records=records,
        coverage=_coverage_by_story(list(records.values())),
        requirement_coverage=_coverage_by_requirement(list(records.values())),
    )
    artifact_path = Path(state["artifact_path"])
    ArtifactMirror(postgres).persist_committed_artifact(
        artifact=build_result,
        artifact_path=artifact_path,
        run_id=state["run_id"],
    )
    if settings.hfil_emit_md:
        _write_test_scenario_markdown(build_result, artifact_path.with_suffix(".md"))
    if session is not None:
        session.artifact_payload = build_result.artifact.model_dump(mode="json")
    if logger is not None:
        logger.info(
            "Test-scenario artifact persisted to PostgreSQL and mirrored to {path}",
            step="persist",
            path=str(artifact_path),
            scenario_count=len(build_result.records),
            story_count=len(build_result.coverage),
            requirement_count=len(build_result.requirement_coverage),
            status="completed",
        )
        logger.info(
            "Projecting test-scenario coverage into Neo4j",
            step="project_test_scenario_coverage",
            document_version_id=build_result.artifact.document_version_id,
            scenario_count=len(build_result.records),
            store_responsibility="test_scenario_traceability_nodes",
        )
    neo4j.project_test_scenario_coverage(
        build_result,
        dict(state.get("evidence_by_requirement", {})),
    )
    result_payload: TestScenarioState = {
        "run_id": state["run_id"],
        "project": build_result.artifact.project,
        "document_id": build_result.artifact.document_id,
        "document_version_id": build_result.artifact.document_version_id,
        "doc_version": build_result.artifact.doc_version,
        "artifact_path": str(artifact_path),
        "story_count": state["story_count"],
        "scenario_ids": list(build_result.records),
        "coverage": build_result.coverage,
        "requirement_coverage": build_result.requirement_coverage,
        "warnings": state.get("warnings", []),
        "errors": [],
    }
    if session is not None:
        session.metadata.update(result_payload)
    postgres.record_run(state["run_id"], "completed", dict(result_payload))
    if logger is not None:
        logger.info(
            "Test-scenario generation pipeline completed",
            step="run_pipeline",
            run_id=state["run_id"],
            scenario_count=len(build_result.records),
            story_count=state["story_count"],
            status="completed",
        )
    return result_payload


def _run_pipeline(
    state: TestScenarioState,
    *,
    session: RunSession | None = None,
) -> TestScenarioState:
    request = TestScenarioRequest.model_validate(state["request"])
    settings = load_config()
    if session is not None:
        session.set_log_level(settings.log_level)
    _apply_overrides(settings, request)
    logger = session.logger if session is not None else None
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = ChromaStore(settings)
    run_dir = session.run_dir if session is not None else None

    def pipeline() -> TestScenarioState:
        _validate_required_test_scenario_stack(settings)
        if logger is not None:
            logger.info(
                "Beginning test-scenario generation pipeline",
                step="run_pipeline",
                run_id=state["run_id"],
                status="started",
            )
        _check_store(logger, "check_postgres", postgres.check)
        _check_store(logger, "check_neo4j", neo4j.check)
        _check_store(logger, "check_chroma", chroma.check)
        postgres.ensure_schema()

        reasoning_model = create_reasoning_model(settings, logger=logger, run_dir=run_dir)
        embedding_model = create_embedding_model(settings)
        reranker_model = create_reranker_model(settings)
        _warmup_reasoning_model(reasoning_model)
        _apply_test_scenario_system_message(reasoning_model)
        neo4j.ensure_search_index()
        if logger is not None:
            logger.info(
                "Model readiness checks completed",
                step="check_models",
                reasoning_provider=reasoning_model.provider_name,
                embedding_provider=embedding_model.provider_name,
                reranker_provider=reranker_model.provider_name,
                status="PASS",
            )

        source = _load_story_source(request, postgres, logger)
        warnings: list[str] = []
        requirement_map = _load_requirement_map(request, source, postgres, logger, warnings)
        if logger is not None:
            logger.info(
                "Loaded {story_count} user stories for test-scenario generation",
                step="load_user_stories",
                story_count=len(source.stories),
                requirement_count=len(requirement_map),
                document_version_id=source.document_version_id,
                project=source.project,
            )
        require_knowledge_graph_when_primary(
            settings=settings,
            neo4j=neo4j,
            document_version_id=source.document_version_id,
            primary=settings.knowledge_graph.graph_primary_scenario,
            stage="test-scenario",
        )

        retrieval = RetrievalService(
            chroma=chroma,
            neo4j=neo4j,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            settings=settings.test_scenario,
            logger=logger,
            knowledge=build_knowledge_retrieval_config(
                settings,
                stage=STAGE_TEST_SCENARIO,
                project=source.project,
                primary=settings.knowledge_graph.graph_primary_scenario,
                recorder=postgres.record_generation_context,
            ),
        )
        agent = TestScenarioGenerationAgent(reasoning_model, logger=logger)
        out_dir = _output_dir(request, settings, source, state["run_id"])
        evidence_by_requirement: dict[str, list[str]] = {
            story.requirement_id: (
                list(requirement_map[story.requirement_id].evidence_chunk_ids)
                if story.requirement_id in requirement_map
                else []
            )
            for story in source.stories
        }

        # Loop 1: retrieve context identifiers into a checkpointed context map.
        entries = _build_or_load_context_map(
            source=source,
            requirement_map=requirement_map,
            retrieval=retrieval,
            out_dir=out_dir,
            run_id=state["run_id"],
            logger=logger,
        )
        # Loop 2: hydrate full chunk text, feed the LLM, checkpoint per item.
        generated = _generate_from_context_map(
            source=source,
            requirement_map=requirement_map,
            entries=entries,
            agent=agent,
            neo4j=neo4j,
            out_dir=out_dir,
            run_id=state["run_id"],
            logger=logger,
        )

        build_result = build_test_scenario_artifact(
            project=source.project,
            document_id=source.document_id,
            document_version_id=source.document_version_id,
            doc_version=source.doc_version,
            generated=generated,
            traces=_traces_by_story(entries),
        )
        build_result = _preserve_existing_scenario_ids(
            build_result,
            postgres.load_test_scenarios_for_generation(project=source.project),
        )
        artifact_path = ArtifactMirror(postgres).persist_committed_artifact(
            artifact=build_result,
            artifact_path=out_dir / "test_scenarios.json",
            run_id=state["run_id"],
        )
        if session is not None:
            session.artifact_payload = build_result.artifact.model_dump(mode="json")
        if logger is not None:
            logger.info(
                "Test-scenario artifact written to {path}",
                step="write_test_scenario_artifact",
                path=str(artifact_path),
                scenario_count=len(build_result.records),
                story_count=len(build_result.coverage),
                requirement_count=len(build_result.requirement_coverage),
                status="completed",
            )
        if logger is not None:
            logger.info(
                "Projecting test-scenario coverage into Neo4j",
                step="project_test_scenario_coverage",
                document_version_id=source.document_version_id,
                scenario_count=len(build_result.records),
                store_responsibility="test_scenario_traceability_nodes",
            )
        neo4j.project_test_scenario_coverage(build_result, evidence_by_requirement)

        result_payload: TestScenarioState = {
            "run_id": state["run_id"],
            "project": source.project,
            "document_id": source.document_id,
            "document_version_id": source.document_version_id,
            "doc_version": source.doc_version,
            "artifact_path": str(artifact_path),
            "story_count": len(source.stories),
            "scenario_ids": list(build_result.records),
            "coverage": build_result.coverage,
            "requirement_coverage": build_result.requirement_coverage,
            "warnings": warnings,
            "errors": [],
        }
        if session is not None:
            session.metadata.update(result_payload)
        postgres.record_run(state["run_id"], "completed", dict(result_payload))
        if logger is not None:
            logger.info(
                "Test-scenario generation pipeline completed",
                step="run_pipeline",
                run_id=state["run_id"],
                scenario_count=len(build_result.records),
                story_count=len(source.stories),
                status="completed",
            )
        return result_payload

    try:
        return pipeline()
    except Exception as exc:
        if logger is not None:
            logger.exception(
                "Test-scenario generation pipeline failed",
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


def _decide_scenario_semantic(
    *,
    retrieval: RetrievalService,
    story: UserStoryRecord,
    query_text: str,
    evidence_chunk_ids: list[str],
    project: str,
    document_version_id: str,
    logger: RunLogger | None,
) -> GraphPrimaryDecision:
    """Graph-primary scenario decision (loop 1) with a structured audit log.

    Uses the story query (title + persona + acceptance criteria + linked
    requirement) and the linked requirement's evidence chunks as anchors, with
    the scenario predicate profile. Gate misses log ``graph_fallback`` and degrade
    to the legacy chunk path.
    """
    decision = retrieval.decide_primary_context(
        requirement_text=query_text,
        document_version_id=document_version_id,
        evidence_chunk_ids=evidence_chunk_ids,
        anchor_id=story.story_id,
    )
    log_graph_primary_decision(
        logger,
        stage=STAGE_TEST_SCENARIO,
        project=project,
        document_version_id=document_version_id,
        anchor_id=story.story_id,
        decision=decision,
    )
    return decision


def _traces_by_story(entries: list[ContextMapEntry]) -> dict[str, GenerationTrace]:
    """Per-story generation grounding trace, keyed by ``story_id`` for the builder."""
    return {
        entry.story_id: trace_from_entry(entry) for entry in entries if entry.story_id is not None
    }


def _build_or_load_context_map(
    *,
    source: _StorySource,
    requirement_map: dict[str, RequirementInput],
    retrieval: RetrievalService,
    out_dir: Path,
    run_id: str,
    logger: RunLogger | None,
) -> list[ContextMapEntry]:
    """Loop 1: build the context map, or reuse a valid checkpoint to skip retrieval."""
    existing = load_context_map_checkpoint(
        out_dir,
        stage=STAGE_TEST_SCENARIO,
        project=source.project,
        document_version_id=source.document_version_id,
        run_id=run_id,
        logger=logger,
    )
    if existing is not None:
        if logger is not None:
            logger.info(
                "Loaded context map from checkpoint; skipping retrieval loop",
                step="test_scenarios.context_map",
                entry_count=len(existing),
                source="checkpoint",
                status="resumed",
            )
        return existing

    entries: list[ContextMapEntry] = []
    evidence_by_story: dict[str, list[str]] = {}
    for story in source.stories:
        linked = requirement_map.get(story.requirement_id)
        linked_text = linked.requirement_text if linked else None
        evidence = list(linked.evidence_chunk_ids) if linked else []
        evidence_by_story[story.story_id] = evidence
        query_text = _story_query_text(story, linked_text)
        decision = _decide_scenario_semantic(
            retrieval=retrieval,
            story=story,
            query_text=query_text,
            evidence_chunk_ids=evidence,
            project=source.project,
            document_version_id=source.document_version_id,
            logger=logger,
        )
        provenance = retrieval.retrieve_context_map(
            requirement_text=query_text,
            document_version_id=source.document_version_id,
            evidence_chunk_ids=evidence,
            anchor_id=story.story_id,
        )
        entries.append(
            make_context_map_entry(
                requirement_id=story.requirement_id,
                document_version_id=source.document_version_id,
                provenance=provenance,
                story_id=story.story_id,
                semantic=decision.context if decision.selected else None,
                generation_context_run_id=decision.context_run_id,
            )
        )
    entries = validate_context_map(
        entries,
        expected_document_version_id=source.document_version_id,
        require_story_id=True,
        evidence_by_key=evidence_by_story,
        logger=logger,
    )
    write_context_map_checkpoint(
        out_dir,
        run_id=run_id,
        stage=STAGE_TEST_SCENARIO,
        project=source.project,
        document_version_id=source.document_version_id,
        entries=entries,
    )
    if logger is not None:
        logger.info(
            "Context map retrieval loop completed and checkpointed",
            step="test_scenarios.context_map",
            entry_count=len(entries),
            path=str(out_dir / context_map_filename(STAGE_TEST_SCENARIO)),
            status="completed",
        )
    return entries


def _generate_from_context_map(
    *,
    source: _StorySource,
    requirement_map: dict[str, RequirementInput],
    entries: list[ContextMapEntry],
    agent: TestScenarioGenerationAgent,
    neo4j: Neo4jStore,
    out_dir: Path,
    run_id: str,
    logger: RunLogger | None,
) -> list[tuple[UserStoryRecord, TestScenarioModel]]:
    """Loop 2: hydrate chunk text, generate per story, checkpoint each item."""
    stories_by_id = {story.story_id: story for story in source.stories}
    progress = load_generation_progress(
        out_dir,
        stage=STAGE_TEST_SCENARIO,
        project=source.project,
        document_version_id=source.document_version_id,
        run_id=run_id,
        logger=logger,
    )
    generated: list[tuple[UserStoryRecord, TestScenarioModel]] = []
    for index, entry in enumerate(entries, start=1):
        story_id = entry.story_id
        story = stories_by_id.get(story_id) if story_id else None
        if story is None:
            raise CheckpointError(f"context map references unknown story_id {entry.story_id}")
        completed = progress.completed.get(story.story_id)
        if completed is not None:
            for scenario in _scenarios_from_payload(completed.payload, story.story_id):
                generated.append((story, scenario))
            if logger is not None:
                logger.info(
                    "Skipping already-completed story from progress checkpoint",
                    step="generate_test_scenarios.story",
                    story_index=index,
                    story_id=story.story_id,
                    status="skipped",
                )
            continue

        linked = requirement_map.get(story.requirement_id)
        linked_text = linked.requirement_text if linked else None
        context = hydrate_context_map_entry(entry, neo4j=neo4j, logger=logger)
        try:
            output = agent.generate(
                story,
                context,
                story_index=index,
                requirement_text=linked_text,
            )
        except Exception as exc:
            record_failure(
                progress,
                input_id=story.story_id,
                error=str(exc),
                requirement_id=story.requirement_id,
            )
            write_generation_progress(out_dir, progress)
            append_generation_error(
                out_dir,
                STAGE_TEST_SCENARIO,
                {
                    "stage": STAGE_TEST_SCENARIO,
                    "input_id": story.story_id,
                    "requirement_id": story.requirement_id,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        scenarios = list(output.test_scenarios)
        for scenario in scenarios:
            generated.append((story, scenario))
        record_completion(
            progress,
            input_id=story.story_id,
            requirement_id=story.requirement_id,
            payload={
                "test_scenarios": [scenario.model_dump(mode="json") for scenario in scenarios]
            },
        )
        write_generation_progress(out_dir, progress)
    return generated


def _scenarios_from_payload(payload: dict[str, Any], story_id: str) -> list[TestScenarioModel]:
    raw = payload.get("test_scenarios")
    if not isinstance(raw, list):
        raise CheckpointError(f"progress payload for {story_id} is missing test_scenarios")
    try:
        return [TestScenarioModel.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise CheckpointError(f"progress payload for {story_id} is invalid ({exc})") from exc


def run_test_scenario_generation(
    request: TestScenarioRequest,
    session: RunSession | None = None,
) -> TestScenarioResult:
    if session is None:
        project, version = resolve_test_scenario_identity(request)
        with command_session(
            project=project,
            version=version,
            command=RuntimeCommand.GENERATE_TEST_SCENARIOS.value,
            run_id=command_run_id(RuntimeCommand.GENERATE_TEST_SCENARIOS.value),
        ) as managed_session:
            return run_test_scenario_generation(request, session=managed_session)
    session.request_payload = request.model_dump(mode="json")
    settings = load_config()
    session.set_log_level(settings.log_level)
    _apply_overrides(settings, request)
    initial_state = {"request": request.model_dump(mode="json"), "run_id": session.run_id}
    if settings.enable_hfil:
        final_state = _invoke_hfil_graph(
            request=request,
            session=session,
            settings=settings,
            initial_state=initial_state,
            feedback_io=CLIFeedbackIO(),
        )
    else:
        graph = build_test_scenario_graph(
            session,
            settings=settings,
            hfil_enabled=False,
        )
        final_state = graph.invoke(initial_state)
    return _result_from_state(final_state)


def _result_from_state(final_state: dict[str, Any]) -> TestScenarioResult:
    return TestScenarioResult(
        run_id=final_state["run_id"],
        status="completed",
        project=final_state["project"],
        document_id=final_state["document_id"],
        document_version_id=final_state["document_version_id"],
        doc_version=final_state["doc_version"],
        artifact_path=Path(final_state["artifact_path"]),
        story_count=final_state["story_count"],
        scenario_ids=final_state["scenario_ids"],
        coverage=final_state.get("coverage", {}),
        requirement_coverage=final_state.get("requirement_coverage", {}),
        warnings=final_state.get("warnings", []),
        errors=final_state.get("errors", []),
    )


def _invoke_hfil_graph(
    *,
    request: TestScenarioRequest,
    session: RunSession,
    settings: AppSettings,
    initial_state: dict[str, Any],
    feedback_io: FeedbackIO,
) -> dict[str, Any]:
    runtime = _build_hfil_runtime(settings, session)
    thread_id = request.thread_id or f"{session.run_id}:hfil"
    config = {"configurable": {"thread_id": thread_id}}
    if (
        settings.hfil_checkpointer == "postgres"
        and settings.postgres.mode == ModeName.POSTGRES.value
    ):
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(_checkpoint_dsn(settings)) as checkpointer:
            checkpointer.setup()
            graph = build_test_scenario_graph(
                session,
                settings=settings,
                hfil_enabled=True,
                hfil_runtime=runtime,
                checkpointer=checkpointer,
            )
            return _drive_hfil_interrupts(graph, initial_state, config, feedback_io)

    memory_checkpointer = InMemorySaver()
    graph = build_test_scenario_graph(
        session,
        settings=settings,
        hfil_enabled=True,
        hfil_runtime=runtime,
        checkpointer=memory_checkpointer,
    )
    return _drive_hfil_interrupts(graph, initial_state, config, feedback_io)


def _build_hfil_runtime(settings: AppSettings, session: RunSession) -> HFILRuntime:
    reasoning_model = create_reasoning_model(
        settings,
        logger=session.logger,
        run_dir=session.run_dir,
    )
    embedding_model = create_embedding_model(settings)
    reranker_model = create_reranker_model(settings)
    _warmup_reasoning_model(reasoning_model)
    _apply_test_scenario_system_message(reasoning_model)
    matcher = SemanticMatcher(
        embedding_model,
        cos_floor=settings.hfil_cos_floor,
        cos_ceil=settings.hfil_cos_ceil,
    )
    dedup = DedupEngine(
        embedding_model,
        reranker_model,
        reasoning_model,
        DedupConfig(recall_cosine=settings.dedup_recall_cosine),
    )
    return HFILRuntime(
        settings=settings,
        matcher=matcher,
        dedup=dedup,
        reasoner=reasoning_model,
    )


def _drive_hfil_interrupts(
    graph: Any,
    initial_state: dict[str, Any],
    config: dict[str, Any],
    feedback_io: FeedbackIO,
) -> dict[str, Any]:
    state = graph.invoke(initial_state, config=config)
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        _render_hfil_payload(payload, feedback_io)
        user_line = feedback_io.prompt("> ")
        state = graph.invoke(Command(resume=user_line), config=config)
    return cast(dict[str, Any], state)


def _render_hfil_payload(payload: object, feedback_io: FeedbackIO) -> None:
    if not isinstance(payload, dict):
        feedback_io.show(str(payload))
        return
    for message in payload.get("messages", []):
        feedback_io.show(str(message))
    scenarios = payload.get("scenarios", [])
    if isinstance(scenarios, list):
        feedback_io.show_scenarios([item for item in scenarios if isinstance(item, dict)])
    duplicate_groups = payload.get("duplicate_groups", [])
    if duplicate_groups:
        feedback_io.show(json.dumps(duplicate_groups, indent=2))
    feedback_io.show(str(payload.get("prompt", "")))


def _checkpoint_dsn(settings: AppSettings) -> str:
    dsn = settings.postgres.dsn
    scheme, separator, remainder = dsn.partition("://")
    if separator and "+" in scheme:
        return f"{scheme.split('+', 1)[0]}://{remainder}"
    return dsn


def resolve_test_scenario_identity(request: TestScenarioRequest) -> tuple[str, str]:
    """Resolve (project, version) for the command session before the graph runs."""
    project = request.project
    version = "generated"
    if request.user_stories_path is not None and request.user_stories_path.exists():
        data = json.loads(request.user_stories_path.read_text(encoding="utf-8"))
        if not project and data.get("project"):
            project = str(data["project"])
        version = str(data.get("doc_version") or data.get("version") or version)
    if not project:
        raise ConfigurationError(
            "--project is required when --user-stories is absent or has no project field"
        )
    return project, version


def _apply_overrides(settings: AppSettings, request: TestScenarioRequest) -> None:
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider:
        settings.embedding_model.provider = request.embedding_provider
    if request.reranker_provider:
        settings.reranker_model.provider = request.reranker_provider
    if request.top_k is not None and request.top_k > 0:
        settings.test_scenario.top_k = request.top_k
    if settings.test_scenario.max_new_tokens:
        settings.huggingface.max_new_tokens = settings.test_scenario.max_new_tokens
    if request.hfil_enabled is not None:
        settings.enable_hfil = request.hfil_enabled
    if request.emit_md:
        settings.hfil_emit_md = True


def _load_story_source(
    request: TestScenarioRequest,
    postgres: PostgresStore,
    logger: Any | None,
) -> _StorySource:
    if request.user_stories_path is not None:
        if logger is not None:
            logger.info(
                "Loading user stories from local artifact {path}",
                step="load_user_stories",
                path=str(request.user_stories_path),
                source="local_json",
            )
        local_payload = json.loads(request.user_stories_path.read_text(encoding="utf-8"))
        project = request.project or str(local_payload.get("project", ""))
        document_version_id = str(local_payload.get("document_version_id", ""))
        payload = (
            ArtifactMirror(postgres)
            .read_preferring_local(
                artifact_path=request.user_stories_path,
                project=project,
                run_id=None,
                document_version_id=document_version_id,
            )
            .payload
        )
        records = postgres.load_user_stories_for_generation(
            project=project,
            document_version_id=document_version_id,
        )
        return _load_story_source_from_payload(payload, records)

    if not request.document_version_id:
        raise ConfigurationError("provide --user-stories or --document-version-id")
    postgres_payload = postgres.load_user_story_artifact_payload(
        document_version_id=request.document_version_id
    )
    if postgres_payload is None:
        raise ConfigurationError(
            "no user-story artifact found in postgres for "
            f"document_version_id={request.document_version_id}"
        )
    if logger is not None:
        logger.info(
            "Loading user stories from postgres fallback",
            step="load_user_stories",
            document_version_id=request.document_version_id,
            source="postgres",
        )
    records = postgres.load_user_stories_for_generation(
        project=str(postgres_payload.get("project", "")),
        document_version_id=str(postgres_payload.get("document_version_id", "")),
    )
    return _load_story_source_from_payload(postgres_payload, records)


def _load_story_source_from_payload(
    payload: dict[str, Any],
    records: list[UserStoryRecord],
) -> _StorySource:
    artifact = UserStoryArtifact.model_validate(payload)
    if not records:
        raise ConfigurationError(
            "user-story artifact projection is valid, but internal user-story rows are "
            "unavailable; run reconcile against PostgreSQL or regenerate user stories"
        )
    return _StorySource(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.doc_version,
        stories=records,
    )


def _load_requirement_map(
    request: TestScenarioRequest,
    source: _StorySource,
    postgres: PostgresStore,
    logger: Any | None,
    warnings: list[str],
) -> dict[str, RequirementInput]:
    def warn(message: str, **context: object) -> None:
        warnings.append(message)
        if logger is not None:
            logger.warning(
                message,
                step="load_requirements",
                status="warning",
                **context,
            )

    if request.requirements_path is not None:
        requirement_source = load_requirement_source_local(request.requirements_path)
        _raise_on_document_version_mismatch(requirement_source, source)
        return _requirements_by_id(requirement_source)

    if request.user_stories_path is not None:
        compact_path = request.user_stories_path.parent / "requirements.json"
        if compact_path.exists():
            loaded_compact = _try_load_requirement_source_local(
                compact_path,
                source,
                warn,
            )
            if loaded_compact is not None:
                return _requirements_by_id(loaded_compact)

    payload = postgres.load_requirement_artifact_payload(
        document_version_id=source.document_version_id
    )
    if payload is not None:
        try:
            requirement_source = load_requirement_source_from_canonical_payload(payload)
            _raise_on_document_version_mismatch(requirement_source, source)
            if logger is not None:
                logger.info(
                    "Loading requirement evidence from postgres fallback",
                    step="load_requirements",
                    document_version_id=source.document_version_id,
                    source="postgres",
                )
            return _requirements_by_id(requirement_source)
        except (ValidationError, KeyError, TypeError, ValueError, ConfigurationError) as exc:
            warn(
                "postgres requirement artifact is unusable; proceeding without it",
                error=str(exc),
                source="postgres",
            )

    warn("proceeding without requirement evidence; retrieval uses dense+sparse only")
    return {}


def _try_load_requirement_source_local(
    path: Path,
    source: _StorySource,
    warn: Any,
) -> RequirementSource | None:
    try:
        requirement_source = load_requirement_source_local(path)
        _raise_on_document_version_mismatch(requirement_source, source)
    except (ValidationError, json.JSONDecodeError, OSError, ConfigurationError) as exc:
        warn(
            "sibling requirements.json is unusable; trying next requirement source",
            path=str(path),
            error=str(exc),
            source="local_json",
        )
        return None
    return requirement_source


def _raise_on_document_version_mismatch(
    requirement_source: RequirementSource,
    source: _StorySource,
) -> None:
    if requirement_source.document_version_id != source.document_version_id:
        raise ConfigurationError(
            "requirement artifact document_version_id "
            f"{requirement_source.document_version_id} does not match user-story "
            f"document_version_id {source.document_version_id}"
        )


def _requirements_by_id(source: RequirementSource) -> dict[str, RequirementInput]:
    return {requirement.requirement_id: requirement for requirement in source.requirements}


def _story_query_text(story: UserStoryRecord, requirement_text: str | None) -> str:
    parts = [story.title, story.user_story.i_want, story.user_story.so_that]
    parts.extend(story.acceptance_criteria)
    if requirement_text:
        parts.append(requirement_text)
    return "\n".join(part.strip() for part in parts if part.strip())


def _coverage_by_story(scenarios: list[TestScenarioRecord]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for scenario in scenarios:
        coverage.setdefault(scenario.story_id, []).append(scenario.scenario_id)
    return coverage


def _scenario_records_from_state(state: TestScenarioState) -> dict[str, TestScenarioRecord]:
    payloads = state.get("scenario_record_payloads", {})
    if not isinstance(payloads, dict):
        raise CheckpointError("scenario_record_payloads must be a JSON object")
    return {
        str(scenario_id): TestScenarioRecord.model_validate(payload)
        for scenario_id, payload in payloads.items()
        if isinstance(payload, dict)
    }


def _preserve_existing_scenario_ids(
    artifact: TestScenarioBuildResult,
    existing_scenarios: list[TestScenarioRecord],
) -> TestScenarioBuildResult:
    existing_by_story: dict[str, list[TestScenarioRecord]] = {}
    for scenario in existing_scenarios:
        existing_by_story.setdefault(scenario.story_id, []).append(scenario)
    for scenarios in existing_by_story.values():
        scenarios.sort(key=lambda item: item.scenario_id)

    rewritten: dict[str, TestScenarioRecord] = {}
    ordinal_by_story: dict[str, int] = {}
    for scenario in artifact.records.values():
        ordinal = ordinal_by_story.get(scenario.story_id, 0)
        ordinal_by_story[scenario.story_id] = ordinal + 1
        prior = existing_by_story.get(scenario.story_id, [])
        stable_scenario_id = (
            prior[ordinal].scenario_id if ordinal < len(prior) else scenario.scenario_id
        )
        record = scenario.model_copy(update={"scenario_id": stable_scenario_id})
        rewritten[stable_scenario_id] = record
    scenarios = list(rewritten.values())
    artifact.records = rewritten
    artifact.coverage = _coverage_by_story(scenarios)
    artifact.requirement_coverage = _coverage_by_requirement(scenarios)
    artifact.artifact = project_test_scenario_artifact(
        project=artifact.artifact.project,
        document_id=artifact.artifact.document_id,
        document_version_id=artifact.artifact.document_version_id,
        doc_version=artifact.artifact.doc_version,
        records=rewritten,
    )
    return artifact


def _coverage_by_requirement(scenarios: list[TestScenarioRecord]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for scenario in scenarios:
        coverage.setdefault(scenario.requirement_id, []).append(scenario.scenario_id)
    return coverage


def _write_test_scenario_markdown(artifact: TestScenarioBuildResult, path: Path) -> None:
    lines = [
        f"# Test Scenarios - {artifact.artifact.project}",
        "",
        f"- document_version_id: `{artifact.artifact.document_version_id}`",
        f"- scenario_count: {len(artifact.records)}",
        "",
    ]
    for scenario in artifact.records.values():
        lines.extend(
            [
                f"## {scenario.scenario_id}",
                "",
                f"- story_id: `{scenario.story_id}`",
                f"- requirement_id: `{scenario.requirement_id}`",
                f"- type: {scenario.scenario_type}",
                f"- priority: {scenario.priority}",
                "",
                scenario.description,
                "",
                f"Expected result: {scenario.expected_result}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _output_dir(
    request: TestScenarioRequest,
    settings: AppSettings,
    source: _StorySource,
    run_identifier: str,
) -> Path:
    if request.user_stories_path is not None:
        return request.user_stories_path.parent
    return (
        settings.paths.generated_requirements_dir
        / source.project
        / "test_scenarios"
        / run_identifier
    )


def _validate_required_test_scenario_stack(settings: AppSettings) -> None:
    if settings.reasoning_model.provider in {ProviderName.LOCAL_HEURISTIC.value}:
        raise ConfigurationError(
            f"REASONING_MODEL_PROVIDER={settings.reasoning_model.provider} "
            "is not valid for test-scenario generation"
        )
    if settings.embedding_model.provider in {ProviderName.LOCAL_HASH.value}:
        raise ConfigurationError(
            f"EMBEDDING_MODEL_PROVIDER={settings.embedding_model.provider} "
            "is not valid for test-scenario generation"
        )
    if settings.reranker_model.provider in {ProviderName.NONE.value}:
        raise ConfigurationError(
            f"RERANKER_MODEL_PROVIDER={settings.reranker_model.provider} "
            "is not valid for test-scenario generation"
        )
    if settings.postgres.mode != ModeName.POSTGRES.value:
        raise ConfigurationError("POSTGRES_MODE=postgres is required for test-scenario generation")
    if settings.neo4j.mode != ModeName.NEO4J.value:
        raise ConfigurationError("NEO4J_MODE=neo4j is required for test-scenario generation")


def _apply_test_scenario_system_message(reasoning_model: Any) -> None:
    setter = getattr(reasoning_model, "set_system_message", None)
    if callable(setter):
        setter(PromptTestScenarioGeneration.SYS_PROMPT_TEST_SCENARIO_GENERATION.value)


def _check_store(logger: Any | None, step: str, check: Any) -> None:
    try:
        detail = check()
    except Exception as exc:
        raise StoreUnavailableError(f"{step} failed: {exc}") from exc
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
                "Could not record failed test-scenario run in PostgreSQL; "
                "preserving original error",
                step="record_failed_run",
                run_id=run_id,
                error_type=record_error.__class__.__name__,
                error=str(record_error),
                status="warning",
            )
