"""Reusable semantic knowledge-graph build, shared by ingest and the standalone
``build-knowledge-graph`` stage.

This is the single implementation of "chunks -> entities + assertions -> Neo4j":
both the ingestion pipeline (auto-build) and the standalone workflow call
:func:`build_and_project_knowledge_graph` with an already-created reasoning model
and Neo4j store, so neither reloads the runtime stack or recursively invokes the
CLI. Neo4j writes stay idempotent MERGEs, so re-running a version is safe.
"""

from __future__ import annotations

from typing import Any

from multi_agentic_graph_rag.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    KnowledgeGraphArtifact,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.services.assertion_canonicalization import canonicalize_assertions
from multi_agentic_graph_rag.services.assertion_lifecycle import (
    LifecycleResult,
    reconcile_assertion_lifecycle,
)
from multi_agentic_graph_rag.services.entity_resolution import resolve_entities
from multi_agentic_graph_rag.services.text_unit_segmentation import (
    attach_evidence_text_units,
    segment_version_chunks,
)


def build_and_project_knowledge_graph(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    chunks: list[DocumentChunk],
    reasoning_model: ReasoningModel,
    neo4j: Any,
    logger: Any | None = None,
) -> KnowledgeGraphArtifact:
    """Extract, resolve, canonicalize and project one version's knowledge graph.

    Returns the projected :class:`KnowledgeGraphArtifact`. Raises on extraction or
    projection failure so the caller (ingest or the standalone stage) fails the run
    clearly rather than silently shipping an empty knowledge graph.
    """
    neo4j.ensure_knowledge_schema()

    text_units = segment_version_chunks(
        document_version_id=document_version_id,
        chunks=chunks,
    )
    if logger is not None:
        logger.info(
            "Segmented {chunk_count} chunks into {text_unit_count} text units",
            step="segment_text_units",
            chunk_count=len(chunks),
            text_unit_count=len(text_units),
            document_version_id=document_version_id,
        )
    neo4j.project_text_units(document_version_id, text_units)

    agent = KnowledgeExtractionAgent(reasoning_model, logger=logger)
    extraction = agent.run(project=project, version=doc_version, chunks=chunks)

    # Entity resolution reuses the entities participating in the document's prior
    # active knowledge version (never every stale entity ever created for the
    # project); see ``Neo4jStore.fetch_entities_for_resolution``.
    existing_entities = neo4j.fetch_entities_for_resolution(
        project=project, document_id=document_id
    )
    resolution = resolve_entities(
        project=project,
        extraction=extraction,
        existing_entities=existing_entities,
        chunk_text_by_id={chunk.chunk_id: chunk.text for chunk in chunks},
    )
    assertions, evidence = canonicalize_assertions(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        extraction=extraction,
        resolution=resolution,
    )
    evidence = attach_evidence_text_units(
        evidence,
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
        text_units=text_units,
    )

    # Cross-version lifecycle: diff this version's assertions against the prior
    # active knowledge version and mark supersede/retire/new. Skipped when there is
    # no prior version, or when re-running the same version (idempotent no-op).
    lifecycle = _reconcile_lifecycle(
        neo4j=neo4j,
        document_id=document_id,
        document_version_id=document_version_id,
        assertions=assertions,
        logger=logger,
    )
    assertions = lifecycle.assertions

    artifact = KnowledgeGraphArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
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
            document_version_id=document_version_id,
            entity_count=len(artifact.entities),
            assertion_count=len(artifact.assertions),
            evidence_count=len(artifact.evidence),
            store_responsibility="source_knowledge_graph",
        )
    neo4j.project_knowledge_graph(artifact)
    if lifecycle.prior_updates:
        neo4j.apply_assertion_lifecycle(lifecycle.prior_updates)
    return artifact


def _reconcile_lifecycle(
    *,
    neo4j: Any,
    document_id: str,
    document_version_id: str,
    assertions: list[Any],
    logger: Any | None,
) -> LifecycleResult:
    """Reconcile lifecycle deterministically within the active scope.

    Args:
        neo4j (Any): Neo4j required by the operation's typed contract.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        assertions (list[Any]): Assertions required by the operation's typed contract.
        logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        LifecycleResult: The typed result produced by the operation.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    prior_version = neo4j.active_knowledge_version(document_id)
    if prior_version is None or prior_version == document_version_id:
        return LifecycleResult(assertions=assertions)
    prior_assertions = neo4j.fetch_assertion_lineage(prior_version)
    result = reconcile_assertion_lifecycle(
        new_assertions=assertions, prior_assertions=prior_assertions
    )
    if logger is not None and (result.prior_updates or result.ambiguous_lineage_keys):
        superseded = sum(1 for u in result.prior_updates if u.status == "superseded")
        retired = sum(1 for u in result.prior_updates if u.status == "retired")
        logger.info(
            "Reconciled assertion lifecycle against prior knowledge version",
            step="assertion_lifecycle",
            document_version_id=document_version_id,
            prior_version=prior_version,
            superseded_count=superseded,
            retired_count=retired,
            ambiguous_count=len(result.ambiguous_lineage_keys),
        )
    return result
