"""Deterministic requirement canonicalization after complete chunk coverage."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from multi_agentic_graph_rag.domain.identifiers import new_requirement_id
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalEntity,
    CanonicalRelationship,
    CanonicalRequirement,
    RequirementEntityRelationshipMap,
    RequirementMapEntry,
    RequirementsArtifact,
    canonical_checksum,
)

_SPACE = re.compile(r"\s+")


def build_requirements_artifact(
    *,
    project: str,
    run_id: str,
    requirement_map: RequirementEntityRelationshipMap,
    entities: list[CanonicalEntity],
    relationships: list[CanonicalRelationship],
    existing: RequirementsArtifact | None = None,
) -> RequirementsArtifact:
    """Canonicalize occurrences, reuse project identities, and build the artifact."""
    failed = [
        result.chunk_id for result in requirement_map.chunk_results if result.status == "failed"
    ]
    if failed:
        raise ValueError(f"failed chunks block requirements publication: {failed}")
    existing_requirements = existing.requirements if existing is not None else []
    existing_by_source = {
        item.source_req_id: item
        for item in existing_requirements
        if item.source_req_id_type == "source"
    }
    existing_generated = {
        _normalized(item.requirement_text): item
        for item in existing_requirements
        if item.source_req_id_type == "generated"
    }

    occurrences = [
        requirement
        for result in sorted(requirement_map.chunk_results, key=lambda item: item.sequence_index)
        for requirement in result.requirements
    ]
    source_texts: dict[str, str] = {}
    for occurrence in occurrences:
        if occurrence.source_req_id_type != "source":
            continue
        source_id = occurrence.source_req_id or ""
        prior_text = source_texts.setdefault(source_id, occurrence.requirement_text)
        if prior_text != occurrence.requirement_text:
            raise ValueError(f"source ID {source_id} has conflicting exact requirement text")
        existing_row = existing_by_source.get(source_id)
        if (
            existing_row is not None
            and existing_row.requirement_text != occurrence.requirement_text
        ):
            raise ValueError(f"source ID {source_id} conflicts with the project requirement")

    grouped: dict[tuple[str, str], list[RequirementMapEntry]] = defaultdict(list)
    source_normalized = {
        _normalized(item.requirement_text)
        for item in occurrences
        if item.source_req_id_type == "source"
    }
    existing_source_by_text = {
        _normalized(item.requirement_text): item
        for item in existing_requirements
        if item.source_req_id_type == "source"
    }
    for occurrence in occurrences:
        normalized = _normalized(occurrence.requirement_text)
        if occurrence.source_req_id_type == "source":
            key = ("source", occurrence.source_req_id or "")
        elif normalized in source_normalized:
            source_match = next(
                item
                for item in occurrences
                if item.source_req_id_type == "source"
                and _normalized(item.requirement_text) == normalized
            )
            key = ("source", source_match.source_req_id or "")
        elif normalized in existing_source_by_text:
            key = ("source", existing_source_by_text[normalized].source_req_id or "")
        else:
            key = ("generated", normalized)
        grouped[key].append(occurrence)

    canonical: list[CanonicalRequirement] = []
    for key, group in grouped.items():
        representative = next(
            (item for item in group if item.source_req_id_type == "source"), group[0]
        )
        if key[0] == "source":
            prior = existing_by_source.get(key[1])
        else:
            prior = existing_generated.get(key[1])
        evidence = _unique_by(
            [item for occurrence in group for item in occurrence.evidence],
            lambda item: item.evidence_id,
        )
        canonical.append(
            CanonicalRequirement(
                requirement_id=prior.requirement_id if prior is not None else new_requirement_id(),
                source_req_id=(
                    prior.source_req_id if prior is not None else representative.source_req_id
                ),
                source_req_id_type=(
                    prior.source_req_id_type
                    if prior is not None
                    else representative.source_req_id_type
                ),
                requirement_text=(
                    prior.requirement_text
                    if prior is not None and prior.source_req_id_type == "source"
                    else representative.requirement_text
                ),
                requirement_type=(
                    prior.requirement_type if prior is not None else representative.requirement_type
                ),
                priority=prior.priority if prior is not None else representative.priority,
                confidence=max(item.confidence for item in group),
                constraints=list(
                    dict.fromkeys(value for item in group for value in item.constraints)
                ),
                entity_ids=list(
                    dict.fromkeys(value for item in group for value in item.entity_ids)
                ),
                relationship_ids=list(
                    dict.fromkeys(value for item in group for value in item.relationship_ids)
                ),
                evidence=evidence,
            )
        )

    merged_entities = _merge_entities(entities)
    merged_relationships = _unique_by(
        relationships,
        lambda item: item.relationship_id,
    )
    payload = RequirementsArtifact.model_construct(
        project=project,
        run_id=run_id,
        checksum="",
        requirements=canonical,
        entities=merged_entities,
        relationships=merged_relationships,
    )
    return RequirementsArtifact.model_validate(
        {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
    )


def _normalized(value: str) -> str:
    return _SPACE.sub(" ", value).strip().casefold()


def _unique_by[T](items: list[T], key: Callable[[T], object]) -> list[T]:
    result: list[T] = []
    seen: set[str] = set()
    for item in items:
        value = str(key(item))
        if value in seen:
            continue
        seen.add(value)
        result.append(item)
    return result


def _merge_entities(entities: list[CanonicalEntity]) -> list[CanonicalEntity]:
    grouped: dict[str, list[CanonicalEntity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.entity_id].append(entity)
    result: list[CanonicalEntity] = []
    for rows in grouped.values():
        first = rows[0]
        result.append(
            first.model_copy(
                update={
                    "aliases": list(dict.fromkeys(alias for row in rows for alias in row.aliases)),
                    "mentions": _unique_by(
                        [mention for row in rows for mention in row.mentions],
                        lambda mention: (
                            f"{mention.chunk_id}:{mention.start_char}:{mention.end_char}"
                        ),
                    ),
                }
            )
        )
    return result


__all__ = ["build_requirements_artifact"]
