"""Resolve and project validated Stage 1.2 semantic candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from multi_agentic_graph_rag.agents.requirement_discovery_agent import quote_span
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.domain.identifiers import (
    make_entity_id,
    make_evidence_id,
    make_relationship_id,
    stable_token,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalEntity,
    CanonicalRelationship,
    EntityMention,
    Evidence,
    ManifestChunk,
    RequirementChunkResult,
    RequirementDiscoveryChunkResponse,
    RequirementMapEntry,
)


@dataclass(frozen=True)
class ChunkProjection:
    """Validated per-chunk map entry plus canonical graph records."""

    result: RequirementChunkResult
    entities: list[CanonicalEntity]
    relationships: list[CanonicalRelationship]


class KnowledgeGraphBuilder:
    """Project entities, mentions, and direct allowlisted semantic relationships."""

    def __init__(self, store: Neo4jStore) -> None:
        self.store = store

    def project(
        self,
        *,
        project: str,
        chunk: ManifestChunk,
        response: RequirementDiscoveryChunkResponse,
    ) -> ChunkProjection:
        """Resolve candidates, write only missing graph records, and validate full read-back."""
        entities_by_ref: dict[str, CanonicalEntity] = {}
        for candidate in response.entities:
            mention_quote = candidate.evidence_quotes[0]
            start, end = quote_span(chunk.chunk_text, mention_quote)
            entity_id = make_entity_id(
                project, candidate.normalized_name.strip().lower(), candidate.entity_type
            )
            entities_by_ref[candidate.entity_ref] = CanonicalEntity(
                entity_id=entity_id,
                name=candidate.name,
                normalized_name=candidate.normalized_name.strip().lower(),
                entity_type=candidate.entity_type,
                aliases=candidate.aliases,
                mentions=[
                    EntityMention(
                        chunk_id=chunk.chunk_id,
                        surface_text=mention_quote,
                        start_char=chunk.start_char + start,
                        end_char=chunk.start_char + end,
                    )
                ],
            )

        relationships_by_owner: dict[tuple[str, str], CanonicalRelationship] = {}
        relationship_hashes: dict[str, str] = {}
        relationships_by_ref = {
            relationship.relationship_ref: relationship for relationship in response.relationships
        }
        map_entries: list[RequirementMapEntry] = []
        for requirement in response.requirements:
            requirement_hash = hashlib.sha256(
                requirement.requirement_text.encode("utf-8")
            ).hexdigest()
            evidence: list[Evidence] = []
            for quote in requirement.evidence_quotes:
                start, end = quote_span(chunk.chunk_text, quote)
                evidence.append(
                    Evidence(
                        evidence_id=make_evidence_id(
                            stable_token(requirement.requirement_ref, requirement_hash),
                            chunk.chunk_id,
                            chunk.start_char + start,
                            chunk.start_char + end,
                        ),
                        chunk_id=chunk.chunk_id,
                        quote=quote,
                        start_char=chunk.start_char + start,
                        end_char=chunk.start_char + end,
                    )
                )
            entity_ids = [
                entities_by_ref[entity_ref].entity_id for entity_ref in requirement.entity_refs
            ]
            relationship_ids: list[str] = []
            for relationship_ref in requirement.relationship_refs:
                rel_candidate = relationships_by_ref[relationship_ref]
                relationship_start, relationship_end = quote_span(
                    chunk.chunk_text, rel_candidate.evidence_quote
                )
                source_id = entities_by_ref[rel_candidate.source_entity_ref].entity_id
                target_id = entities_by_ref[rel_candidate.target_entity_ref].entity_id
                evidence_hash = hashlib.sha256(
                    rel_candidate.evidence_quote.encode("utf-8")
                ).hexdigest()
                relationship_id = make_relationship_id(
                    project=project,
                    chunk_id=chunk.chunk_id,
                    requirement_text_hash=requirement_hash,
                    source_entity_id=source_id,
                    relationship_type=rel_candidate.relationship_type,
                    target_entity_id=target_id,
                    evidence_hash=evidence_hash,
                )
                relationship = CanonicalRelationship(
                    relationship_id=relationship_id,
                    chunk_id=chunk.chunk_id,
                    source_entity_id=source_id,
                    relationship_type=rel_candidate.relationship_type,
                    target_entity_id=target_id,
                    confidence=rel_candidate.confidence,
                    evidence=[
                        Evidence(
                            evidence_id=make_evidence_id(
                                relationship_id,
                                chunk.chunk_id,
                                chunk.start_char + relationship_start,
                                chunk.start_char + relationship_end,
                            ),
                            chunk_id=chunk.chunk_id,
                            quote=rel_candidate.evidence_quote,
                            start_char=chunk.start_char + relationship_start,
                            end_char=chunk.start_char + relationship_end,
                        )
                    ],
                )
                relationships_by_owner[(requirement.requirement_ref, relationship_ref)] = (
                    relationship
                )
                relationship_hashes[relationship_id] = requirement_hash
                relationship_ids.append(relationship_id)
            map_entries.append(
                RequirementMapEntry(
                    requirement_ref=requirement.requirement_ref,
                    source_req_id=requirement.source_req_id,
                    source_req_id_type=requirement.source_req_id_type,
                    requirement_text=requirement.requirement_text,
                    requirement_type=requirement.requirement_type,
                    priority=requirement.priority,
                    constraints=requirement.constraints,
                    evidence=evidence,
                    entity_ids=entity_ids,
                    relationship_ids=relationship_ids,
                    confidence=requirement.confidence,
                )
            )

        entities = list(entities_by_ref.values())
        relationships = list(relationships_by_owner.values())
        expected_entities = {item.entity_id for item in entities}
        expected_mentions = expected_entities
        expected_relationships = {item.relationship_id for item in relationships}
        self.store.prune_semantic_projection(
            project=project,
            chunk_id=chunk.chunk_id,
            entity_ids=expected_entities,
            relationship_ids=expected_relationships,
        )
        self.store.upsert_semantic_projection(
            project=project,
            chunk_id=chunk.chunk_id,
            entities=entities,
            mentioned_entity_ids=expected_mentions,
            relationships=relationships,
            requirement_text_hash_by_relationship=relationship_hashes,
        )
        validated = self.store.read_semantic_projection(
            project=project,
            chunk_id=chunk.chunk_id,
            entity_ids=expected_entities,
            relationship_ids=expected_relationships,
        )
        if (
            validated.entity_ids != expected_entities
            or validated.mentioned_entity_ids != expected_mentions
            or validated.relationship_ids != expected_relationships
        ):
            raise ValueError("Neo4j semantic projection read-back validation failed")
        self.store.validate_semantic_projection(
            project=project,
            chunk_id=chunk.chunk_id,
            entities=entities,
            relationships=relationships,
            requirement_text_hash_by_relationship=relationship_hashes,
        )
        status: Literal["completed", "no_requirements"] = (
            "completed" if map_entries else "no_requirements"
        )
        return ChunkProjection(
            result=RequirementChunkResult(
                chunk_id=chunk.chunk_id,
                sequence_index=chunk.sequence_index,
                status=status,
                requirements=map_entries,
                error=None,
            ),
            entities=entities,
            relationships=relationships,
        )


__all__ = ["ChunkProjection", "KnowledgeGraphBuilder"]
