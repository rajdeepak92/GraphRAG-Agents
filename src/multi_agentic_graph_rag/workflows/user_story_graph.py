"""LangGraph orchestration for standalone user-story generation (stage 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from common_defs import ModeName, ProviderName, RuntimeCommand
from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGenerationAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import CheckpointError, ConfigurationError
from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    UserStoryBuildResult,
    UserStoryModel,
    UserStoryRecord,
    UserStoryRequest,
    UserStoryResult,
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
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.generation_checkpoint import (
    STAGE_USER_STORY,
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
from multi_agentic_graph_rag.services.user_story_builder import (
    build_user_story_artifact,
    project_user_story_artifact,
)


class UserStoryState(TypedDict, total=False):
    request: dict[str, Any]
    run_id: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: str
    requirement_count: int
    story_ids: list[str]
    coverage: dict[str, list[str]]
    warnings: list[str]
    errors: list[str]


def build_user_story_graph(session: RunSession | None = None) -> Any:
    graph = StateGraph(UserStoryState)
    graph.add_node("validate_request", lambda state: _validate_request(state, session=session))
    graph.add_node("run_pipeline", lambda state: _run_pipeline(state, session=session))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()


def _validate_request(
    state: UserStoryState,
    *,
    session: RunSession | None = None,
) -> UserStoryState:
    request = UserStoryRequest.model_validate(state["request"])
    logger = session.logger if session is not None else None
    if request.requirements_path is None and request.document_version_id is None:
        raise ConfigurationError(
            "provide --requirements <path> or --document-version-id <id> to load requirements"
        )
    if request.requirements_path is not None and not request.requirements_path.exists():
        raise FileNotFoundError(request.requirements_path)
    rid = state.get("run_id") or command_run_id(RuntimeCommand.GENERATE_USER_STORIES.value)
    if logger is not None:
        logger.debug(
            "Validated user-story request",
            step="validate_request",
            requirements_path=str(request.requirements_path) if request.requirements_path else None,
            document_version_id=request.document_version_id,
            run_id=rid,
        )
    return {"request": request.model_dump(mode="json"), "run_id": rid, "warnings": [], "errors": []}


def _run_pipeline(
    state: UserStoryState,
    *,
    session: RunSession | None = None,
) -> UserStoryState:
    request = UserStoryRequest.model_validate(state["request"])
    settings = load_config()
    if session is not None:
        session.set_log_level(settings.log_level)
    _apply_overrides(settings, request)
    logger = session.logger if session is not None else None
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = ChromaStore(settings)
    run_dir = session.run_dir if session is not None else None

    def pipeline() -> UserStoryState:
        _validate_required_user_story_stack(settings)
        if logger is not None:
            logger.info(
                "Beginning user-story generation pipeline",
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

        source = _load_requirement_source(request, postgres, logger)
        if logger is not None:
            logger.info(
                "Loaded {requirement_count} requirements for user-story generation",
                step="load_requirements",
                requirement_count=len(source.requirements),
                document_version_id=source.document_version_id,
                project=source.project,
            )

        retrieval = RetrievalService(
            chroma=chroma,
            neo4j=neo4j,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            settings=settings.user_story,
            logger=logger,
        )
        agent = UserStoryGenerationAgent(reasoning_model, logger=logger)
        out_dir = _output_dir(request, settings, source, state["run_id"])
        evidence_by_requirement: dict[str, list[str]] = {
            requirement.requirement_id: list(requirement.evidence_chunk_ids)
            for requirement in source.requirements
        }

        # Loop 1: retrieve context identifiers into a checkpointed context map.
        entries = _build_or_load_context_map(
            source=source,
            retrieval=retrieval,
            out_dir=out_dir,
            run_id=state["run_id"],
            evidence_by_requirement=evidence_by_requirement,
            logger=logger,
        )
        # Loop 2: hydrate full chunk text, feed the LLM, checkpoint per item.
        generated = _generate_from_context_map(
            source=source,
            entries=entries,
            agent=agent,
            neo4j=neo4j,
            out_dir=out_dir,
            run_id=state["run_id"],
            logger=logger,
        )

        build_result = build_user_story_artifact(
            project=source.project,
            document_id=source.document_id,
            document_version_id=source.document_version_id,
            doc_version=source.doc_version,
            generated=generated,
        )
        build_result = _preserve_existing_story_ids(
            build_result,
            postgres.load_user_stories_for_generation(project=source.project),
        )
        artifact_path = ArtifactMirror(postgres).persist_committed_artifact(
            artifact=build_result,
            artifact_path=out_dir / "user_stories.json",
            run_id=state["run_id"],
        )
        artifact = build_result.artifact
        if session is not None:
            session.artifact_payload = artifact.model_dump(mode="json")
        if logger is not None:
            logger.info(
                "User-story artifact written to {path}",
                step="write_user_story_artifact",
                path=str(artifact_path),
                story_count=len(build_result.records),
                requirement_count=len(build_result.coverage),
                status="completed",
            )
        if logger is not None:
            logger.info(
                "Projecting user-story coverage into Neo4j and marking requirements covered",
                step="project_user_story_coverage",
                document_version_id=source.document_version_id,
                story_count=len(build_result.records),
                store_responsibility="user_story_traceability_nodes",
            )
        neo4j.project_user_story_coverage(build_result, evidence_by_requirement)

        result_payload: UserStoryState = {
            "run_id": state["run_id"],
            "project": source.project,
            "document_id": source.document_id,
            "document_version_id": source.document_version_id,
            "doc_version": source.doc_version,
            "artifact_path": str(artifact_path),
            "requirement_count": len(source.requirements),
            "story_ids": list(build_result.records),
            "coverage": build_result.coverage,
            "warnings": [],
            "errors": [],
        }
        if session is not None:
            session.metadata.update(result_payload)
        postgres.record_run(state["run_id"], "completed", dict(result_payload))
        if logger is not None:
            logger.info(
                "User-story generation pipeline completed",
                step="run_pipeline",
                run_id=state["run_id"],
                story_count=len(build_result.records),
                requirement_count=len(source.requirements),
                status="completed",
            )
        return result_payload

    try:
        return pipeline()
    except Exception as exc:
        if logger is not None:
            logger.exception(
                "User-story generation pipeline failed",
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
    source: RequirementSource,
    retrieval: RetrievalService,
    out_dir: Path,
    run_id: str,
    evidence_by_requirement: dict[str, list[str]],
    logger: RunLogger | None,
) -> list[ContextMapEntry]:
    """Loop 1: build the context map, or reuse a valid checkpoint to skip retrieval."""
    existing = load_context_map_checkpoint(
        out_dir,
        stage=STAGE_USER_STORY,
        project=source.project,
        document_version_id=source.document_version_id,
        run_id=run_id,
        logger=logger,
    )
    if existing is not None:
        if logger is not None:
            logger.info(
                "Loaded context map from checkpoint; skipping retrieval loop",
                step="user_stories.context_map",
                entry_count=len(existing),
                source="checkpoint",
                status="resumed",
            )
        return existing

    entries = [
        make_context_map_entry(
            requirement_id=requirement.requirement_id,
            document_version_id=source.document_version_id,
            provenance=retrieval.retrieve_context_map(
                requirement_text=requirement.requirement_text,
                document_version_id=source.document_version_id,
                evidence_chunk_ids=requirement.evidence_chunk_ids,
            ),
        )
        for requirement in source.requirements
    ]
    entries = validate_context_map(
        entries,
        expected_document_version_id=source.document_version_id,
        require_story_id=False,
        evidence_by_key=evidence_by_requirement,
        logger=logger,
    )
    write_context_map_checkpoint(
        out_dir,
        run_id=run_id,
        stage=STAGE_USER_STORY,
        project=source.project,
        document_version_id=source.document_version_id,
        entries=entries,
    )
    if logger is not None:
        logger.info(
            "Context map retrieval loop completed and checkpointed",
            step="user_stories.context_map",
            entry_count=len(entries),
            path=str(out_dir / context_map_filename(STAGE_USER_STORY)),
            status="completed",
        )
    return entries


def _generate_from_context_map(
    *,
    source: RequirementSource,
    entries: list[ContextMapEntry],
    agent: UserStoryGenerationAgent,
    neo4j: Neo4jStore,
    out_dir: Path,
    run_id: str,
    logger: RunLogger | None,
) -> list[tuple[RequirementInput, UserStoryModel]]:
    """Loop 2: hydrate chunk text, generate per requirement, checkpoint each item."""
    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in source.requirements
    }
    progress = load_generation_progress(
        out_dir,
        stage=STAGE_USER_STORY,
        project=source.project,
        document_version_id=source.document_version_id,
        run_id=run_id,
        logger=logger,
    )
    generated: list[tuple[RequirementInput, UserStoryModel]] = []
    for index, entry in enumerate(entries, start=1):
        requirement = requirements_by_id.get(entry.requirement_id)
        if requirement is None:
            raise CheckpointError(
                f"context map references unknown requirement_id {entry.requirement_id}"
            )
        completed = progress.completed.get(entry.requirement_id)
        if completed is not None:
            for story in _stories_from_payload(completed.payload, entry.requirement_id):
                generated.append((requirement, story))
            if logger is not None:
                logger.info(
                    "Skipping already-completed requirement from progress checkpoint",
                    step="generate_user_stories.requirement",
                    requirement_index=index,
                    requirement_id=entry.requirement_id,
                    status="skipped",
                )
            continue

        context = hydrate_context_map_entry(entry, neo4j=neo4j, logger=logger)
        try:
            output = agent.generate(requirement, context, requirement_index=index)
        except Exception as exc:
            record_failure(progress, input_id=entry.requirement_id, error=str(exc))
            write_generation_progress(out_dir, progress)
            append_generation_error(
                out_dir,
                STAGE_USER_STORY,
                {
                    "stage": STAGE_USER_STORY,
                    "input_id": entry.requirement_id,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            raise
        stories = list(output.user_stories)
        for story in stories:
            generated.append((requirement, story))
        record_completion(
            progress,
            input_id=entry.requirement_id,
            payload={"user_stories": [story.model_dump(mode="json") for story in stories]},
        )
        write_generation_progress(out_dir, progress)
    return generated


def _stories_from_payload(payload: dict[str, Any], requirement_id: str) -> list[UserStoryModel]:
    raw = payload.get("user_stories")
    if not isinstance(raw, list):
        raise CheckpointError(f"progress payload for {requirement_id} is missing user_stories")
    try:
        return [UserStoryModel.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise CheckpointError(f"progress payload for {requirement_id} is invalid ({exc})") from exc


def _preserve_existing_story_ids(
    artifact: UserStoryBuildResult,
    existing_stories: list[UserStoryRecord],
) -> UserStoryBuildResult:
    existing_by_requirement: dict[str, list[UserStoryRecord]] = {}
    for story in existing_stories:
        existing_by_requirement.setdefault(story.requirement_id, []).append(story)
    for stories in existing_by_requirement.values():
        stories.sort(key=lambda item: item.story_id)

    rewritten: dict[str, UserStoryRecord] = {}
    coverage: dict[str, list[str]] = {}
    ordinal_by_requirement: dict[str, int] = {}
    for story in artifact.records.values():
        ordinal = ordinal_by_requirement.get(story.requirement_id, 0)
        ordinal_by_requirement[story.requirement_id] = ordinal + 1
        prior = existing_by_requirement.get(story.requirement_id, [])
        stable_story_id = prior[ordinal].story_id if ordinal < len(prior) else story.story_id
        record = story.model_copy(update={"story_id": stable_story_id})
        rewritten[stable_story_id] = record
        coverage.setdefault(record.requirement_id, []).append(stable_story_id)
    artifact.records = rewritten
    artifact.coverage = coverage
    artifact.artifact = project_user_story_artifact(
        project=artifact.artifact.project,
        document_id=artifact.artifact.document_id,
        document_version_id=artifact.artifact.document_version_id,
        doc_version=artifact.artifact.doc_version,
        records=rewritten,
        requirement_display_ids={},
        story_display_ids={},
    )
    return artifact


def run_user_story_generation(
    request: UserStoryRequest,
    session: RunSession | None = None,
) -> UserStoryResult:
    if session is None:
        project, version = resolve_user_story_identity(request)
        with command_session(
            project=project,
            version=version,
            command=RuntimeCommand.GENERATE_USER_STORIES.value,
            run_id=command_run_id(RuntimeCommand.GENERATE_USER_STORIES.value),
        ) as managed_session:
            return run_user_story_generation(request, session=managed_session)
    session.request_payload = request.model_dump(mode="json")
    graph = build_user_story_graph(session)
    final_state = graph.invoke(
        {"request": request.model_dump(mode="json"), "run_id": session.run_id}
    )
    return UserStoryResult(
        run_id=final_state["run_id"],
        status="completed",
        project=final_state["project"],
        document_id=final_state["document_id"],
        document_version_id=final_state["document_version_id"],
        doc_version=final_state["doc_version"],
        artifact_path=Path(final_state["artifact_path"]),
        requirement_count=final_state["requirement_count"],
        story_ids=final_state["story_ids"],
        coverage=final_state.get("coverage", {}),
        warnings=final_state.get("warnings", []),
        errors=final_state.get("errors", []),
    )


def resolve_user_story_identity(request: UserStoryRequest) -> tuple[str, str]:
    """Resolve (project, version) for the command session before the graph runs."""
    project = request.project
    version = "generated"
    if request.requirements_path is not None and request.requirements_path.exists():
        data = json.loads(request.requirements_path.read_text(encoding="utf-8"))
        if not project and data.get("project"):
            project = str(data["project"])
        version = str(data.get("doc_version") or data.get("version") or version)
    if not project:
        raise ConfigurationError(
            "--project is required when --requirements is absent or has no project field"
        )
    return project, version


def _apply_overrides(settings: AppSettings, request: UserStoryRequest) -> None:
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider:
        settings.embedding_model.provider = request.embedding_provider
    if request.reranker_provider:
        settings.reranker_model.provider = request.reranker_provider
    if request.top_k is not None and request.top_k > 0:
        settings.user_story.top_k = request.top_k
    if settings.user_story.max_new_tokens:
        settings.huggingface.max_new_tokens = settings.user_story.max_new_tokens


def _load_requirement_source(
    request: UserStoryRequest,
    postgres: PostgresStore,
    logger: Any | None,
) -> RequirementSource:
    if request.requirements_path is not None:
        if logger is not None:
            logger.info(
                "Loading requirements from local artifact {path}",
                step="load_requirements",
                path=str(request.requirements_path),
                source="local_json",
            )
        local_payload = json.loads(request.requirements_path.read_text(encoding="utf-8"))
        project = request.project or str(local_payload.get("project", ""))
        document_version_id = str(local_payload.get("document_version_id", ""))
        ArtifactMirror(postgres).read_preferring_local(
            artifact_path=request.requirements_path,
            project=project,
            run_id=None,
            document_version_id=document_version_id,
        )
        return load_requirement_source_local(request.requirements_path)

    if not request.document_version_id:
        raise ConfigurationError("provide --requirements or --document-version-id")
    payload = postgres.load_requirement_artifact_payload(
        document_version_id=request.document_version_id
    )
    if payload is None:
        raise ConfigurationError(
            "no requirement artifact found in postgres for "
            f"document_version_id={request.document_version_id}"
        )
    if logger is not None:
        logger.info(
            "Loading requirements from postgres fallback",
            step="load_requirements",
            document_version_id=request.document_version_id,
            source="postgres",
        )
    return load_requirement_source_from_full_payload(payload)


def _output_dir(
    request: UserStoryRequest,
    settings: AppSettings,
    source: RequirementSource,
    run_identifier: str,
) -> Path:
    if request.requirements_path is not None:
        return request.requirements_path.parent
    return (
        settings.paths.generated_requirements_dir / source.project / "user_stories" / run_identifier
    )


def _validate_required_user_story_stack(settings: AppSettings) -> None:
    if settings.reasoning_model.provider in {ProviderName.LOCAL_HEURISTIC.value}:
        raise ConfigurationError(
            f"REASONING_MODEL_PROVIDER={settings.reasoning_model.provider} "
            "is not valid for user-story generation"
        )
    if settings.embedding_model.provider in {ProviderName.LOCAL_HASH.value}:
        raise ConfigurationError(
            f"EMBEDDING_MODEL_PROVIDER={settings.embedding_model.provider} "
            "is not valid for user-story generation"
        )
    if settings.reranker_model.provider in {ProviderName.NONE.value}:
        raise ConfigurationError(
            f"RERANKER_MODEL_PROVIDER={settings.reranker_model.provider} "
            "is not valid for user-story generation"
        )
    if settings.postgres.mode != ModeName.POSTGRES.value:
        raise ConfigurationError("POSTGRES_MODE=postgres is required for user-story generation")
    if settings.neo4j.mode != ModeName.NEO4J.value:
        raise ConfigurationError("NEO4J_MODE=neo4j is required for user-story generation")


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
                "Could not record failed user-story run in PostgreSQL; preserving original error",
                step="record_failed_run",
                run_id=run_id,
                error_type=record_error.__class__.__name__,
                error=str(record_error),
                status="warning",
            )
