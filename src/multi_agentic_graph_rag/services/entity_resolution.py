"""Layered, deterministic entity resolution for the knowledge graph.

Resolution order per candidate group (normalized name + entity type):

1. exact normalized name + same entity type against existing project entities
2. alias match (normalized) + same entity type
3. acronym / expansion match + same entity type
4. otherwise a new canonical entity with a deterministic project-scoped ID

Entities never merge across entity types or projects. Embedding recall,
reranking, and LLM tie-breaks are future layers; everything here is
deterministic so re-runs are idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multi_agentic_graph_rag.domain.identifiers import entity_id, entity_mention_id
from multi_agentic_graph_rag.domain.schemas import (
    EntityMentionRecord,
    EntityRecord,
    KnowledgeExtractionOutput,
)
from multi_agentic_graph_rag.services.ontology import acronym_of, normalize_entity_name


@dataclass
class EntityResolutionResult:
    """Coordinate entity resolution result behavior within the services boundary."""

    entities: list[EntityRecord] = field(default_factory=list)
    mentions: list[EntityMentionRecord] = field(default_factory=list)
    # (chunk_id, normalized_name) -> entity_id for chunk-scoped assertion linking.
    entity_id_by_chunk_name: dict[tuple[str, str], str] = field(default_factory=dict)
    # normalized_name -> entity_id fallback when a name is unambiguous project-wide.
    entity_id_by_name: dict[str, str] = field(default_factory=dict)

    def resolve_reference(self, chunk_id: str, name: str) -> str | None:
        """Resolve reference deterministically within the active scope.

        Args:
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            name (str): Name required by the operation's typed contract.

        Returns:
            str | None: The typed result produced by the operation.
        """
        normalized = normalize_entity_name(name)
        scoped = self.entity_id_by_chunk_name.get((chunk_id, normalized))
        if scoped is not None:
            return scoped
        return self.entity_id_by_name.get(normalized)

    def entity_names_by_id(self) -> dict[str, str]:
        """Execute the entity names by id operation within its declared architectural boundary.

        Returns:
            dict[str, str]: The typed result produced by the operation.
        """
        return {entity.entity_id: entity.canonical_name for entity in self.entities}


def resolve_entities(
    *,
    project: str,
    extraction: KnowledgeExtractionOutput,
    existing_entities: list[EntityRecord],
    chunk_text_by_id: dict[str, str],
) -> EntityResolutionResult:
    """Resolve entities deterministically within the active scope.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        extraction (KnowledgeExtractionOutput): Extraction required by the operation's typed
                                                contract.
        existing_entities (list[EntityRecord]): Existing entities required by the operation's typed
                                                contract.
        chunk_text_by_id (dict[str, str]): Canonical chunk text by id used as a safe operational
                                           anchor.

    Returns:
        EntityResolutionResult: The typed result produced by the operation.
    """
    result = EntityResolutionResult()
    resolved_by_id: dict[str, EntityRecord] = {}
    known: list[EntityRecord] = [entity.model_copy(deep=True) for entity in existing_entities]

    for chunk in extraction.chunks:
        for candidate in chunk.entities:
            entity = _resolve_candidate(
                project=project,
                normalized_name=candidate.normalized_name,
                surface_text=candidate.surface_text,
                entity_type=candidate.entity_type,
                known=known,
            )
            if entity.entity_id not in resolved_by_id:
                resolved_by_id[entity.entity_id] = entity
            entity = resolved_by_id[entity.entity_id]
            _record_alias(entity, candidate.surface_text)
            _register_reference(result, chunk.chunk_id, candidate.normalized_name, entity)
            mention = _build_mention(
                entity_id_value=entity.entity_id,
                chunk_id=chunk.chunk_id,
                surface_text=candidate.surface_text,
                chunk_text=chunk_text_by_id.get(chunk.chunk_id, ""),
            )
            if all(existing.mention_id != mention.mention_id for existing in result.mentions):
                result.mentions.append(mention)

    result.entities = list(resolved_by_id.values())
    return result


def _resolve_candidate(
    *,
    project: str,
    normalized_name: str,
    surface_text: str,
    entity_type: str,
    known: list[EntityRecord],
) -> EntityRecord:
    """Resolve candidate deterministically within the active scope.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        normalized_name (str): Normalized name required by the operation's typed contract.
        surface_text (str): Input text processed in memory and excluded from diagnostic logs.
        entity_type (str): Entity type required by the operation's typed contract.
        known (list[EntityRecord]): Known required by the operation's typed contract.

    Returns:
        EntityRecord: The typed result produced by the operation.
    """
    same_type = [entity for entity in known if entity.entity_type == entity_type]

    for entity in same_type:
        if entity.normalized_name == normalized_name:
            return entity

    for entity in same_type:
        aliases_normalized = {normalize_entity_name(alias) for alias in entity.aliases}
        if normalized_name in aliases_normalized:
            return entity

    candidate_acronym = acronym_of(normalized_name)
    for entity in same_type:
        if normalized_name == acronym_of(entity.normalized_name):
            return entity
        if candidate_acronym and candidate_acronym == entity.normalized_name:
            return entity

    entity = EntityRecord(
        entity_id=entity_id(project, normalized_name, entity_type),
        project=project,
        canonical_name=surface_text,
        normalized_name=normalized_name,
        entity_type=entity_type,
    )
    known.append(entity)
    return entity


def _record_alias(entity: EntityRecord, surface_text: str) -> None:
    """Record alias through the owning storage boundary.

    Args:
        entity (EntityRecord): Entity required by the operation's typed contract.
        surface_text (str): Input text processed in memory and excluded from diagnostic logs.
    """
    if surface_text == entity.canonical_name or surface_text in entity.aliases:
        return
    entity.aliases.append(surface_text)


def _register_reference(
    result: EntityResolutionResult,
    chunk_id: str,
    normalized_name: str,
    entity: EntityRecord,
) -> None:
    """Execute the register reference operation within its declared architectural boundary.

    Args:
        result (EntityResolutionResult): Result required by the operation's typed contract.
        chunk_id (str): Canonical chunk id used as a safe operational anchor.
        normalized_name (str): Normalized name required by the operation's typed contract.
        entity (EntityRecord): Entity required by the operation's typed contract.
    """
    result.entity_id_by_chunk_name[(chunk_id, normalized_name)] = entity.entity_id
    result.entity_id_by_name.setdefault(normalized_name, entity.entity_id)


def _build_mention(
    *,
    entity_id_value: str,
    chunk_id: str,
    surface_text: str,
    chunk_text: str,
) -> EntityMentionRecord:
    """Build mention.

    Args:
        entity_id_value (str): Entity id value required by the operation's typed contract.
        chunk_id (str): Canonical chunk id used as a safe operational anchor.
        surface_text (str): Input text processed in memory and excluded from diagnostic logs.
        chunk_text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        EntityMentionRecord: The typed result produced by the operation.
    """
    start_char: int | None = None
    end_char: int | None = None
    index = chunk_text.casefold().find(surface_text.casefold())
    if index >= 0:
        start_char = index
        end_char = index + len(surface_text)
    return EntityMentionRecord(
        mention_id=entity_mention_id(entity_id_value, chunk_id, surface_text),
        entity_id=entity_id_value,
        chunk_id=chunk_id,
        surface_text=surface_text,
        start_char=start_char,
        end_char=end_char,
    )
