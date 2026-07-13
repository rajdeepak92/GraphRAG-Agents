"""LangGraph orchestration for ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.requirement_discovery_agent import RequirementDiscoveryAgent
from multi_agentic_graph_rag.common_defs import ModeName, ProviderName, RuntimeCommand
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.identifiers import requirement_delta_event_id, run_id
from multi_agentic_graph_rag.domain.schemas import (
    IngestionRequest,
    IngestionResult,
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementRevisionSnapshot,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary
from multi_agentic_graph_rag.observability.session import RunSession, command_session
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.artifacts import (
    write_requirement_identity_resolution_artifact,
)
from multi_agentic_graph_rag.services.chunking import chunk_blocks
from multi_agentic_graph_rag.services.coverage_ledger import CoverageLedger
from multi_agentic_graph_rag.services.knowledge_graph_state import (
    knowledge_graph_rebuild_command,
    run_guarded_knowledge_graph_build,
)
from multi_agentic_graph_rag.services.manifest import build_manifest, write_manifest
from multi_agentic_graph_rag.services.parsing import checksum_bytes, parse_document
from multi_agentic_graph_rag.services.requirement_builder import (
    build_canonical_requirements_artifact,
    build_requirement_artifact,
)
from multi_agentic_graph_rag.services.requirement_memory import (
    MemoryEntry,
    ModelEntailmentJudge,
    RequirementMemory,
)


class IngestionState(TypedDict, total=False):
    """Describe the ingestion state state exchanged between typed workflow nodes."""

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
    ingestion_status: str
    kg_status: str | None
    kg_failure_reason: str | None
    downstream_blocked: bool
    kg_rebuild_command: str | None
    warnings: list[str]
    errors: list[str]


def build_ingestion_graph(session: RunSession | None = None) -> Any:
    """Build ingestion graph.

    Args:
        session (RunSession | None): Optional command session that owns run artifacts and
                                     diagnostics.

    Returns:
        Any: The typed result produced by the operation.
    """
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
    """Validate request against the enforced runtime contract.

    Args:
        state (IngestionState): State required by the operation's typed contract.
        session (RunSession | None): Optional command session that owns run artifacts and
                                     diagnostics.

    Returns:
        IngestionState: The typed result produced by the operation.

    Raises:
        FileNotFoundError: If validated inputs or required dependencies cannot satisfy the
        contract.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
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
    """Run pipeline.

    Args:
        state (IngestionState): State required by the operation's typed contract.
        session (RunSession | None): Optional command session that owns run artifacts and
                                     diagnostics.

    Returns:
        IngestionState: The typed result produced by the operation.

    Side Effects:
        May write transactional or derivative state through the configured store.
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
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
        """Run the workflow pipeline while preserving its persistence boundaries.

        Returns:
            IngestionState: The typed result produced by the operation.

        Side Effects:
            May write transactional or derivative state through the configured store.
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
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
        artifact, artifact_path = _resolve_and_persist_requirements(
            settings=settings,
            manifest=manifest,
            project=project,
            version=version,
            checksum=checksum,
            run_id=state["run_id"],
            run_dir=run_dir,
            replace_version=request.replace_version,
            postgres=postgres,
            reasoning_model=reasoning_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            logger=logger,
        )
        if session is not None:
            session.artifact_payload = build_canonical_requirements_artifact(artifact).model_dump(
                mode="json"
            )
        if logger is not None:
            logger.info(
                "Canonical requirement artifact persisted/written to {path}",
                step="write_requirement_artifact",
                path=str(artifact_path),
                document_version_id=artifact.document_version_id,
                status="completed",
            )
        # Knowledge-graph build runs AFTER requirements are persisted and is
        # best-effort: a failure never discards the requirement artifacts. If it
        # fails, graph-primary generation stays unavailable for this version until
        # it is rebuilt with `build-knowledge-graph`.
        kg_outcome = _build_knowledge_graph_best_effort(
            settings=settings,
            manifest=manifest,
            reasoning_model=reasoning_model,
            neo4j=neo4j,
            postgres=postgres,
            run_id=state["run_id"],
            logger=logger,
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
            "ingestion_status": kg_outcome.get("ingestion_status", "completed"),
            "kg_status": kg_outcome.get("kg_status"),
            "kg_failure_reason": kg_outcome.get("kg_failure_reason"),
            "downstream_blocked": bool(kg_outcome.get("downstream_blocked", False)),
            "kg_rebuild_command": kg_outcome.get("kg_rebuild_command"),
            "warnings": kg_outcome.get("warnings", []),
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
        if logger is None:
            return pipeline()
        with logger.span(
            step="run_pipeline",
            operation="ingestion.pipeline",
            document_version_id=state.get("document_version_id", ""),
        ):
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
            "error": sanitized_exception_summary(exc),
        }
        _record_failed_run_safely(
            postgres=postgres,
            run_id=state["run_id"],
            payload=error_payload,
            logger=logger,
        )
        raise


def _resolve_and_persist_requirements(
    *,
    settings: AppSettings,
    manifest: Any,
    project: str,
    version: str,
    checksum: str,
    run_id: str,
    run_dir: Path,
    replace_version: bool,
    postgres: PostgresStore,
    reasoning_model: Any,
    embedding_model: Any,
    reranker_model: Any,
    logger: Any | None,
) -> tuple[RequirementArtifact, Path]:
    """Resolve and persist requirements deterministically within the active scope.

    Args:
        settings (AppSettings): Validated settings that control this operation.
        manifest (Any): Manifest required by the operation's typed contract.
        project (str): Project scope that isolates persistence and retrieval.
        version (str): Document version label within the project scope.
        checksum (str): Checksum required by the operation's typed contract.
        run_id (str): Canonical run id used as a safe operational anchor.
        run_dir (Path): Filesystem location authorized for this operation.
        replace_version (bool): Replace version required by the operation's typed contract.
        postgres (PostgresStore): Postgres required by the operation's typed contract.
        reasoning_model (Any): Provider-neutral model adapter used by the operation.
        embedding_model (Any): Provider-neutral model adapter used by the operation.
        reranker_model (Any): Provider-neutral model adapter used by the operation.
        logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        tuple[RequirementArtifact, Path]: The typed result produced by the operation.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    with postgres.document_identity_lock(project, manifest.document_id):
        prior_revisions = postgres.load_requirement_revision_snapshot(
            project=project,
            document_id=manifest.document_id,
        )
        requirement_memory = RequirementMemory(
            settings=settings.requirement_identity,
            embedder=embedding_model,
            reranker=reranker_model,
            judge=ModelEntailmentJudge(reasoning_model),
        )
        requirement_memory.seed(
            MemoryEntry(
                requirement_id=snapshot.requirement_id,
                revision_id=snapshot.revision_id,
                statement=snapshot.statement,
                normalized_statement=snapshot.normalized_statement,
                requirement_type=snapshot.requirement_type,
                signature=snapshot.semantic_signature,
            )
            for snapshot in prior_revisions.values()
        )
        coverage_ledger = None
        if settings.discovery.ledger_enabled:
            coverage_ledger = CoverageLedger(
                max_entries=settings.discovery.ledger_max_entries,
                injection_top_k=settings.discovery.ledger_top_k,
                embedder=embedding_model,
            )
            primed = coverage_ledger.prime_prior_revisions(prior_revisions.values())
            if logger is not None and primed:
                logger.info(
                    "Seeded requirement memory with {primed} prior-version requirements",
                    step="seed_prior_requirements",
                    primed=primed,
                    document_id=manifest.document_id,
                )
        discovery_agent = RequirementDiscoveryAgent(
            reasoning_model,
            logger=logger,
            coverage_ledger=coverage_ledger,
        )
        if logger is not None:
            logger.info(
                "Discovering requirements from {document_version_id}",
                step="discover_requirements",
                document_version_id=manifest.document_version_id,
            )
        discovery = discovery_agent.run(manifest)
        artifact = build_requirement_artifact(
            project=project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=version,
            source_path=manifest.source_path,
            source_checksum=checksum,
            discovery=discovery,
            prior_revisions=prior_revisions,
            requirement_memory=requirement_memory,
            logger=logger,
        )
        artifact = _apply_strictly_outdated_cascade(
            artifact=artifact,
            prior_revisions=prior_revisions,
            postgres=postgres,
            replace_version=replace_version,
        )
        postgres.persist_manifest(manifest)
        artifact_target = run_dir / "requirements.json"
        if logger is not None:
            logger.info(
                "Persisting canonical requirements and requirement ledger to PostgreSQL",
                step="persist_requirements_postgres",
                document_version_id=artifact.document_version_id,
                artifact_path=str(artifact_target),
                fact_count=len(artifact.facts),
                requirement_count=len(artifact.requirements),
                store_responsibility="requirement_artifact_and_ledger_only",
            )
        artifact_path = ArtifactMirror(postgres).persist_committed_artifact(
            artifact=artifact,
            artifact_path=artifact_target,
            run_id=run_id,
        )
        write_requirement_identity_resolution_artifact(
            artifact,
            run_dir,
            logger=logger,
        )
        return artifact, artifact_path


def _apply_strictly_outdated_cascade(
    *,
    artifact: RequirementArtifact,
    prior_revisions: dict[str, RequirementRevisionSnapshot],
    postgres: PostgresStore,
    replace_version: bool,
) -> RequirementArtifact:
    """Apply strictly outdated cascade.

    Args:
        artifact (RequirementArtifact): Artifact required by the operation's typed contract.
        prior_revisions (dict[str, RequirementRevisionSnapshot]): Prior revisions required by the
                                                                  operation's typed contract.
        postgres (PostgresStore): Postgres required by the operation's typed contract.
        replace_version (bool): Replace version required by the operation's typed contract.

    Returns:
        RequirementArtifact: The typed result produced by the operation.
    """
    if not replace_version:
        return artifact
    strictly_outdated: dict[str, VerifiedRequirement] = {}
    strict_events: list[RequirementDeltaEvent] = []
    for requirement in artifact.requirements:
        prior = prior_revisions.get(requirement.requirement_id)
        if prior is None or not _is_explicit_removal(prior.statement, requirement.statement):
            continue
        strictly_outdated[requirement.requirement_id] = requirement
        postgres.hard_delete_requirement(requirement.requirement_id)
        strict_events.append(
            RequirementDeltaEvent(
                event_id=requirement_delta_event_id(
                    event_type="strictly_outdated",
                    requirement_identifier=requirement.requirement_id,
                    revision_identifier=requirement.revision_id,
                    previous_revision_identifier=prior.revision_id,
                    document_version_identifier=artifact.document_version_id,
                ),
                event_type="strictly_outdated",
                requirement_id=requirement.requirement_id,
                revision_id=requirement.revision_id,
                previous_revision_id=prior.revision_id,
                document_version_id=artifact.document_version_id,
                evidence_ids=[evidence.evidence_id for evidence in requirement.evidence],
                impacted_artifact_types=["user_story", "scenario", "test_case"],
            )
        )
    if not strictly_outdated:
        return artifact
    filtered_requirements = [
        requirement
        for requirement in artifact.requirements
        if requirement.requirement_id not in strictly_outdated
    ]
    filtered_events = [
        event for event in artifact.delta_events if event.requirement_id not in strictly_outdated
    ]
    return artifact.model_copy(
        update={
            "requirements": filtered_requirements,
            "delta_events": [*filtered_events, *strict_events],
        }
    )


def _is_explicit_removal(prior: str, incoming: str) -> bool:
    """Return whether explicit removal.

    Args:
        prior (str): Prior required by the operation's typed contract.
        incoming (str): Incoming required by the operation's typed contract.

    Returns:
        bool: The typed result produced by the operation.

    Side Effects:
        May create or atomically replace files in the configured artifact boundary.
    """
    incoming_norm = incoming.lower()
    markers = (
        "does not support",
        "do not support",
        "no longer supports",
        "must not",
        "shall not",
        "removed",
        "disabled",
        "not supported",
    )
    if not any(marker in incoming_norm for marker in markers):
        return False
    prior_tokens = {token for token in prior.lower().replace("-", " ").split() if len(token) > 3}
    incoming_tokens = {token for token in incoming_norm.replace("-", " ").split() if len(token) > 3}
    if not prior_tokens:
        return False
    return len(prior_tokens & incoming_tokens) / len(prior_tokens) >= 0.35


def run_ingestion(
    request: IngestionRequest,
    session: RunSession | None = None,
) -> IngestionResult:
    """Run ingestion.

    Args:
        request (IngestionRequest): Request required by the operation's typed contract.
        session (RunSession | None): Optional command session that owns run artifacts and
                                     diagnostics.

    Returns:
        IngestionResult: The typed result produced by the operation.

    Side Effects:
        May invoke configured model or workflow providers.
    """
    if session is None:
        generated_run_id = run_id(request.project, request.document, request.version)
        with command_session(
            project=request.project,
            version=request.version,
            command=RuntimeCommand.INGEST.value,
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
        ingestion_status=final_state.get("ingestion_status", "completed"),
        kg_status=final_state.get("kg_status"),
        kg_failure_reason=final_state.get("kg_failure_reason"),
        downstream_blocked=final_state.get("downstream_blocked", False),
        kg_rebuild_command=final_state.get("kg_rebuild_command"),
        warnings=final_state.get("warnings", []),
        errors=final_state.get("errors", []),
    )


def _build_knowledge_graph_best_effort(
    *,
    settings: AppSettings,
    manifest: Any,
    reasoning_model: Any,
    neo4j: Any,
    postgres: Any,
    run_id: str,
    logger: Any | None,
) -> dict[str, Any]:
    """Build + project the semantic knowledge graph without risking the run.

    Delegates the readiness state machine (building -> project -> active pointer ->
    ready, with bounded transient retry) to the shared guarded builder so this path
    and the standalone stage never drift. When the knowledge channel is disabled
    this is a no-op. On failure the requirement artifacts are already persisted, so
    the run still succeeds but is marked ``degraded`` with ``downstream_blocked`` so
    the operator sees that graph-primary generation is blocked until rebuild.
    """
    if not settings.knowledge_graph.enabled:
        return {"ingestion_status": "completed", "warnings": []}
    try:
        if logger is not None:
            logger.info(
                "Building semantic knowledge graph for {document_version_id}",
                step="build_knowledge_graph",
                document_version_id=manifest.document_version_id,
                chunk_count=len(manifest.chunks),
                store_responsibility="source_knowledge_graph",
            )
        result = run_guarded_knowledge_graph_build(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            doc_version=manifest.version,
            chunks=list(manifest.chunks),
            reasoning_model=reasoning_model,
            neo4j=neo4j,
            postgres=postgres,
            settings=settings,
            run_id=run_id,
            logger=logger,
        )
        return {
            "ingestion_status": "completed",
            "kg_status": result.state.status,
            "downstream_blocked": False,
            "warnings": [],
        }
    except Exception as exc:
        rebuild = knowledge_graph_rebuild_command(manifest.project, manifest.document_version_id)
        failure_summary = sanitized_exception_summary(exc)
        message = (
            f"knowledge-graph build failed for {manifest.document_version_id}: "
            f"{failure_summary}. "
            "Requirements were persisted; graph-primary story/scenario generation is "
            f"blocked for this version until you rebuild it with `{rebuild}`."
        )
        if logger is not None:
            logger.warning(
                "Knowledge-graph build failed; continuing (requirements already persisted)",
                step="build_knowledge_graph",
                document_version_id=manifest.document_version_id,
                exception_type=exc.__class__.__name__,
                error_summary=failure_summary,
                status="degraded",
            )
        return {
            "ingestion_status": "degraded",
            "kg_status": "failed",
            "kg_failure_reason": failure_summary,
            "downstream_blocked": True,
            "kg_rebuild_command": rebuild,
            "warnings": [message],
        }


def _ingest_run_dir(settings: AppSettings, project: str, run_identifier: str) -> Path:
    """Execute the ingest run dir operation within its declared architectural boundary.

    Args:
        settings (AppSettings): Validated settings that control this operation.
        project (str): Project scope that isolates persistence and retrieval.
        run_identifier (str): Canonical run identifier used as a safe operational anchor.

    Returns:
        Path: The typed result produced by the operation.
    """
    return settings.paths.generated_requirements_dir / project / "req" / run_identifier


def _validate_required_ingest_stack(settings: AppSettings) -> None:
    """Validate required ingest stack against the enforced runtime contract.

    Args:
        settings (AppSettings): Validated settings that control this operation.

    Raises:
        ConfigurationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    invalid_reasoning = {ProviderName.LOCAL_HEURISTIC.value}
    invalid_embedding = {ProviderName.LOCAL_HASH.value}
    invalid_reranker = {ProviderName.NONE.value}
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
    if settings.postgres.mode != ModeName.POSTGRES.value:
        raise ConfigurationError("POSTGRES_MODE=postgres is required for ingest")
    if settings.neo4j.mode != ModeName.NEO4J.value:
        raise ConfigurationError("NEO4J_MODE=neo4j is required for ingest")


def _warmup_reasoning_model(reasoning_model: Any) -> None:
    """Warm up reasoning model.

    Args:
        reasoning_model (Any): Provider-neutral model adapter used by the operation.

    Side Effects:
        May invoke configured model or workflow providers.
    """
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
    """Record failed run safely through the owning storage boundary.

    Args:
        postgres (PostgresStore): Postgres required by the operation's typed contract.
        run_id (str): Canonical run id used as a safe operational anchor.
        payload (dict[str, Any]): Validated structured data for the operation.
        logger (Any): Optional run-scoped logger used only for sanitized diagnostics.

    Side Effects:
        May write transactional or derivative state through the configured store.
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    try:
        postgres.record_run(run_id, "failed", payload)
    except Exception as record_error:
        if logger is not None:
            logger.warning(
                "Could not record failed ingest state in PostgreSQL; preserving original error",
                step="record_failed_run",
                run_id=run_id,
                error_type=record_error.__class__.__name__,
                error_summary=sanitized_exception_summary(record_error),
                status="warning",
            )
