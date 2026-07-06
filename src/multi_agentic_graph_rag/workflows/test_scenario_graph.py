"""LangGraph orchestration for standalone test-scenario generation (stage 4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from common_defs import ModeName, ProviderName, RuntimeCommand
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGenerationAgent
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
    RequirementInput,
    TestScenarioModel,
    TestScenarioRequest,
    TestScenarioResult,
    UserStoryArtifact,
    UserStoryRecord,
)
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
from multi_agentic_graph_rag.services.artifacts import write_test_scenario_artifact
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
    validate_context_map,
    write_context_map_checkpoint,
    write_generation_progress,
)
from multi_agentic_graph_rag.services.requirement_source import (
    RequirementSource,
    load_requirement_source_from_full_payload,
    load_requirement_source_local,
)
from multi_agentic_graph_rag.services.retrieval import RetrievalService
from multi_agentic_graph_rag.services.test_scenario_builder import build_test_scenario_artifact


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


@dataclass(frozen=True)
class _StorySource:
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    stories: list[UserStoryRecord]


def build_test_scenario_graph(session: RunSession | None = None) -> Any:
    graph = StateGraph(TestScenarioState)
    graph.add_node("validate_request", lambda state: _validate_request(state, session=session))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, session=session))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()


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

        retrieval = RetrievalService(
            chroma=chroma,
            neo4j=neo4j,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            settings=settings.test_scenario,
            logger=logger,
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

        artifact = build_test_scenario_artifact(
            project=source.project,
            document_id=source.document_id,
            document_version_id=source.document_version_id,
            doc_version=source.doc_version,
            generated=generated,
        )
        artifact_path = write_test_scenario_artifact(artifact, out_dir, logger=logger)
        if session is not None:
            session.artifact_payload = artifact.model_dump(mode="json")
        if logger is not None:
            logger.info(
                "Test-scenario artifact written to {path}",
                step="write_test_scenario_artifact",
                path=str(artifact_path),
                scenario_count=len(artifact.scenarios),
                story_count=len(artifact.coverage),
                requirement_count=len(artifact.requirement_coverage),
                status="completed",
            )
        postgres.persist_test_scenario_artifact(artifact, str(artifact_path), state["run_id"])
        if logger is not None:
            logger.info(
                "Projecting test-scenario coverage into Neo4j",
                step="project_test_scenario_coverage",
                document_version_id=source.document_version_id,
                scenario_count=len(artifact.scenarios),
                store_responsibility="test_scenario_traceability_nodes",
            )
        neo4j.project_test_scenario_coverage(artifact, evidence_by_requirement)

        result_payload: TestScenarioState = {
            "run_id": state["run_id"],
            "project": source.project,
            "document_id": source.document_id,
            "document_version_id": source.document_version_id,
            "doc_version": source.doc_version,
            "artifact_path": str(artifact_path),
            "story_count": len(source.stories),
            "scenario_ids": list(artifact.scenarios),
            "coverage": artifact.coverage,
            "requirement_coverage": artifact.requirement_coverage,
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
                scenario_count=len(artifact.scenarios),
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
        provenance = retrieval.retrieve_context_map(
            requirement_text=_story_query_text(story, linked_text),
            document_version_id=source.document_version_id,
            evidence_chunk_ids=evidence,
        )
        entries.append(
            make_context_map_entry(
                requirement_id=story.requirement_id,
                document_version_id=source.document_version_id,
                provenance=provenance,
                story_id=story.story_id,
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
    graph = build_test_scenario_graph(session)
    final_state = graph.invoke(
        {"request": request.model_dump(mode="json"), "run_id": session.run_id}
    )
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
        payload = json.loads(request.user_stories_path.read_text(encoding="utf-8"))
        return _load_story_source_from_payload(payload)

    if not request.document_version_id:
        raise ConfigurationError("provide --user-stories or --document-version-id")
    payload = postgres.load_user_story_artifact_payload(
        document_version_id=request.document_version_id
    )
    if payload is None:
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
    return _load_story_source_from_payload(payload)


def _load_story_source_from_payload(payload: dict[str, Any]) -> _StorySource:
    artifact = UserStoryArtifact.model_validate(payload)
    return _StorySource(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.doc_version,
        stories=list(artifact.stories.values()),
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

        full_path = request.user_stories_path.parent / "requirements_full.json"
        if full_path.exists():
            loaded_full = _try_load_requirement_source_full(
                full_path,
                source,
                warn,
            )
            if loaded_full is not None:
                return _requirements_by_id(loaded_full)

    payload = postgres.load_requirement_artifact_payload(
        document_version_id=source.document_version_id
    )
    if payload is not None:
        try:
            requirement_source = load_requirement_source_from_full_payload(payload)
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


def _try_load_requirement_source_full(
    path: Path,
    source: _StorySource,
    warn: Any,
) -> RequirementSource | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requirement_source = load_requirement_source_from_full_payload(payload)
        _raise_on_document_version_mismatch(requirement_source, source)
    except (ValidationError, json.JSONDecodeError, OSError, ConfigurationError) as exc:
        warn(
            "sibling requirements_full.json is unusable; trying next requirement source",
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
    for criterion in story.acceptance_criteria:
        parts.append(
            f"{criterion.title}. Given {criterion.given}, when {criterion.when}, "
            f"then {criterion.then}."
        )
    if requirement_text:
        parts.append(requirement_text)
    return "\n".join(part.strip() for part in parts if part.strip())


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
