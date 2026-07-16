"""Manifest-scoped hybrid graph/vector retrieval for Stages 2 and 3."""

from __future__ import annotations

from multi_agentic_graph_rag.config.settings import RetrievalSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirement,
    CanonicalUserStory,
    ChunkManifest,
    RetrievedEvidence,
    ScenarioContext,
    StoryContext,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, RerankerModel


class RetrievalService:
    """Combine authoritative anchors, semantic paths, and vector candidates."""

    def __init__(
        self,
        *,
        neo4j: Neo4jStore,
        chroma: ChromaStore,
        embedding: EmbeddingModel,
        reranker: RerankerModel,
        settings: RetrievalSettings,
    ) -> None:
        self.neo4j = neo4j
        self.chroma = chroma
        self.embedding = embedding
        self.reranker = reranker
        self.settings = settings

    def story_context(
        self,
        *,
        project: str,
        manifest: ChunkManifest,
        requirement: CanonicalRequirement,
    ) -> StoryContext:
        """Build the compact context package for one requirement."""
        allowed = {chunk.chunk_id for chunk in manifest.chunks}
        authoritative_ids = {item.chunk_id for item in requirement.evidence}
        _require_subset(authoritative_ids, allowed, "requirement evidence")
        ranked = self._retrieve(
            project=project,
            query=requirement.requirement_text,
            allowed_chunk_ids=allowed,
            authoritative_chunk_ids=authoritative_ids,
            entity_ids=set(requirement.entity_ids),
            relationship_ids=set(requirement.relationship_ids),
        )
        return StoryContext(
            requirement_id=requirement.requirement_id,
            requirement_text=requirement.requirement_text,
            source_req_id=requirement.source_req_id,
            source_req_id_type=requirement.source_req_id_type,
            authoritative_evidence_chunk_ids=sorted(authoritative_ids),
            mapped_entity_ids=requirement.entity_ids,
            mapped_relationship_ids=requirement.relationship_ids,
            ranked_evidence=ranked,
            retrieval_parameters=self._parameters(),
        )

    def scenario_context(
        self,
        *,
        project: str,
        manifest: ChunkManifest,
        story: CanonicalUserStory,
        requirements: list[CanonicalRequirement],
    ) -> ScenarioContext:
        """Build the compact context package for one user story."""
        allowed = {chunk.chunk_id for chunk in manifest.chunks}
        linked = {requirement.requirement_id: requirement for requirement in requirements}
        missing = set(story.requirement_ids) - set(linked)
        if missing:
            raise ValueError(f"story requirement IDs are unresolved: {sorted(missing)}")
        authoritative_ids = set(story.traceability.evidence_chunk_ids)
        entity_ids = set(story.traceability.entity_ids)
        relationship_ids = set(story.traceability.relationship_ids)
        for requirement_id in story.requirement_ids:
            requirement = linked[requirement_id]
            authoritative_ids.update(item.chunk_id for item in requirement.evidence)
            entity_ids.update(requirement.entity_ids)
            relationship_ids.update(requirement.relationship_ids)
        _require_subset(authoritative_ids, allowed, "story evidence")
        query = " ".join(
            (
                story.title,
                story.user_story.as_a,
                story.user_story.i_want,
                story.user_story.so_that,
                *(criterion.then for criterion in story.acceptance_criteria),
            )
        )
        ranked = self._retrieve(
            project=project,
            query=query,
            allowed_chunk_ids=allowed,
            authoritative_chunk_ids=authoritative_ids,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
        )
        return ScenarioContext(
            story_id=story.story_id,
            requirement_ids=story.requirement_ids,
            story_text=query,
            acceptance_criteria=story.acceptance_criteria,
            source_req_id=story.source_req_id,
            source_req_id_type=story.source_req_id_type,
            authoritative_evidence_chunk_ids=sorted(authoritative_ids),
            supporting_entity_ids=sorted(entity_ids),
            supporting_relationship_ids=sorted(relationship_ids),
            ranked_evidence=ranked,
            retrieval_parameters=self._parameters(),
        )

    def _retrieve(
        self,
        *,
        project: str,
        query: str,
        allowed_chunk_ids: set[str],
        authoritative_chunk_ids: set[str],
        entity_ids: set[str],
        relationship_ids: set[str],
    ) -> list[RetrievedEvidence]:
        rows: dict[str, RetrievedEvidence] = {}
        for chunk_id, text in self.neo4j.fetch_chunks(project, authoritative_chunk_ids):
            if chunk_id not in allowed_chunk_ids:
                continue
            rows[chunk_id] = RetrievedEvidence(
                chunk_id=chunk_id,
                text=text,
                source="authoritative",
                score=1.0,
                entity_ids=sorted(entity_ids),
                relationship_ids=sorted(relationship_ids),
            )
        for (
            chunk_id,
            text,
            score,
            graph_entities,
            graph_relationships,
        ) in self.neo4j.retrieve_semantic_chunks(
            project=project,
            anchor_entity_ids=entity_ids,
            anchor_relationship_ids=relationship_ids,
            allowed_chunk_ids=allowed_chunk_ids,
            max_hops=self.settings.max_hops,
            limit=self.settings.graph_k,
        ):
            if chunk_id not in allowed_chunk_ids:
                continue
            rows.setdefault(
                chunk_id,
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    text=text,
                    source="graph",
                    score=score,
                    entity_ids=graph_entities,
                    relationship_ids=graph_relationships,
                ),
            )
        vectors = self.embedding.embed_documents([query])
        if len(vectors) != 1:
            raise ValueError("query embedding returned an invalid vector count")
        for chunk_id, text, distance in self.chroma.query_chunks(
            project=project,
            query_embedding=vectors[0],
            allowed_chunk_ids=allowed_chunk_ids,
            n_results=self.settings.vector_k,
        ):
            if chunk_id not in allowed_chunk_ids:
                continue
            rows.setdefault(
                chunk_id,
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    text=text,
                    source="vector",
                    score=1.0 / (1.0 + max(distance, 0.0)),
                    entity_ids=[],
                    relationship_ids=[],
                ),
            )
        evidence = list(rows.values())
        mandatory = [item for item in evidence if item.source == "authoritative"]
        supplementary = [item for item in evidence if item.source != "authoritative"]
        if supplementary:
            order = self.reranker.rerank(query, [item.text for item in supplementary])
            supplementary = [
                supplementary[index] for index in order if 0 <= index < len(supplementary)
            ]
        selected = [*mandatory, *supplementary[: self.settings.top_k]]
        return _token_bound(selected, self.settings.token_budget)

    def _parameters(self) -> dict[str, int]:
        return {
            "top_k": self.settings.top_k,
            "vector_k": self.settings.vector_k,
            "graph_k": self.settings.graph_k,
            "max_hops": self.settings.max_hops,
            "token_budget": self.settings.token_budget,
        }


def _require_subset(values: set[str], allowed: set[str], label: str) -> None:
    invalid = values - allowed
    if invalid:
        raise ValueError(
            f"{label} references chunks outside the current manifest: {sorted(invalid)}"
        )


def _token_bound(
    evidence: list[RetrievedEvidence],
    token_budget: int,
) -> list[RetrievedEvidence]:
    selected: list[RetrievedEvidence] = []
    used = 0
    for item in evidence:
        estimated = max(1, len(item.text.split()))
        if selected and used + estimated > token_budget:
            continue
        selected.append(item)
        used += estimated
    return selected


__all__ = ["RetrievalService"]
