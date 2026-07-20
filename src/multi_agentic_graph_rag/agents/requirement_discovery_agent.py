"""One-call-per-chunk requirement/entity/relationship discovery."""

from __future__ import annotations

import json
import re

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.domain.errors import SemanticValidationError
from multi_agentic_graph_rag.domain.schemas import (
    LLMEntityCandidate,
    LLMRequirementCandidate,
    ManifestChunk,
    RequirementDiscoveryChunkResponse,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel

_SPACE = re.compile(r"\s+")
_SOURCE_ID = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,}\b")
_WORD = re.compile(r"[A-Z0-9]+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)")
_SEMANTIC_MARKERS = re.compile(
    r"\b(?:shall|must|should|may|not|never|only|if|when|unless|within|before|after|"
    r"until|during|while|at least|at most|up to|every|each|all|remaining|"
    r"\d+(?:\.\d+)?(?:\s*(?:%|ms|s|sec(?:ond)?s?|min(?:ute)?s?|hours?|days?|"
    r"hz|°c|psi|mm/s))?)\b",
    re.IGNORECASE,
)


class RequirementDiscoveryAgent:
    """Run the only Stage 1.2 reasoning call and validate its grounding."""

    def __init__(self, reasoning_model: ReasoningModel) -> None:
        self.reasoning_model = reasoning_model

    def discover(self, chunk: ManifestChunk) -> RequirementDiscoveryChunkResponse:
        """Generate and validate one combined response without semantic repair."""
        prompt = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "evidence_source_text": _normalize(chunk.chunk_text),
                "evidence_quote_candidates": _evidence_quote_candidates(chunk.chunk_text),
                "source_requirement_rows": _explicit_requirement_rows(chunk.chunk_text),
                "layout": chunk.layout.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        response = self.reasoning_model.generate_structured(
            prompt=prompt,
            schema=RequirementDiscoveryChunkResponse,
            system_message=PromptRequirementDiscovery.SYSTEM.value,
            operation="requirement_discovery.combined",
            request_id=chunk.chunk_id,
            max_attempts=1,
        )
        response = self._prune_ungrounded_entities(response)
        try:
            self.validate_response(chunk, response)
        except ValueError as error:
            raise SemanticValidationError(str(error)) from error
        return response

    @staticmethod
    def _prune_ungrounded_entities(
        response: RequirementDiscoveryChunkResponse,
    ) -> RequirementDiscoveryChunkResponse:
        """Drop entity evidence quotes that are not visibly grounded in the entity's
        own name or aliases instead of failing the whole chunk.

        Every quote that survives is still grounded, so the storage guarantee is
        preserved; this only tolerates a reasoning model over-attaching spurious
        quotes to an entity. An entity left with no grounded quote is dropped, and
        references to it are cascaded out of requirements and relationships (and any
        entity that becomes an orphan is dropped in turn) so the result still
        satisfies RequirementDiscoveryChunkResponse's referential rules.
        """
        pruned_entities: list[LLMEntityCandidate] = []
        dropped_refs: set[str] = set()
        changed = False
        for entity in response.entities:
            grounded = [
                quote
                for quote in entity.evidence_quotes
                if _entity_grounded(entity.name, entity.aliases, quote)
            ]
            if not grounded:
                dropped_refs.add(entity.entity_ref)
                changed = True
                continue
            if len(grounded) != len(entity.evidence_quotes):
                changed = True
                entity = entity.model_copy(update={"evidence_quotes": grounded})
            pruned_entities.append(entity)

        # Cascade entity drops to a fixpoint: recompute surviving relationships and
        # the referenced-entity set, then drop any entity nothing references.
        relationships = list(response.relationships)
        while True:
            relationships = [
                relationship
                for relationship in response.relationships
                if relationship.source_entity_ref not in dropped_refs
                and relationship.target_entity_ref not in dropped_refs
            ]
            referenced: set[str] = set()
            for requirement in response.requirements:
                referenced.update(ref for ref in requirement.entity_refs if ref not in dropped_refs)
            for relationship in relationships:
                referenced.update((relationship.source_entity_ref, relationship.target_entity_ref))
            orphans = {
                entity.entity_ref
                for entity in pruned_entities
                if entity.entity_ref not in referenced
            }
            if not orphans:
                break
            dropped_refs |= orphans
            pruned_entities = [
                entity for entity in pruned_entities if entity.entity_ref not in orphans
            ]
            changed = True

        if not changed:
            return response

        surviving_relationship_refs = {
            relationship.relationship_ref for relationship in relationships
        }
        pruned_requirements = [
            requirement.model_copy(
                update={
                    "entity_refs": [
                        ref for ref in requirement.entity_refs if ref not in dropped_refs
                    ],
                    "relationship_refs": [
                        ref
                        for ref in requirement.relationship_refs
                        if ref in surviving_relationship_refs
                    ],
                }
            )
            for requirement in response.requirements
        ]
        return RequirementDiscoveryChunkResponse(
            chunk_id=response.chunk_id,
            requirements=pruned_requirements,
            entities=pruned_entities,
            relationships=relationships,
        )

    @staticmethod
    def validate_response(
        chunk: ManifestChunk,
        response: RequirementDiscoveryChunkResponse,
    ) -> None:
        """Apply deterministic completeness, provenance, quote, and association validation."""
        if response.chunk_id != chunk.chunk_id:
            raise ValueError("response chunk_id does not match the supplied manifest chunk")
        normalized_chunk = _normalize(chunk.chunk_text)
        explicit_rows = _explicit_requirement_rows(chunk.chunk_text)
        returned_source_ids = {
            item.source_req_id
            for item in response.requirements
            if item.source_req_id_type == "source" and item.source_req_id is not None
        }
        if explicit_rows and returned_source_ids != set(explicit_rows):
            missing = sorted(set(explicit_rows) - returned_source_ids)
            extra = sorted(returned_source_ids - set(explicit_rows))
            raise ValueError(
                f"explicit source requirement coverage mismatch missing={missing} extra={extra}"
            )
        requirements_by_ref = {
            requirement.requirement_ref: requirement for requirement in response.requirements
        }
        for requirement in response.requirements:
            for quote in requirement.evidence_quotes:
                _require_exact_quote(normalized_chunk, quote, "requirement")
            if requirement.source_req_id_type == "source":
                source_id = requirement.source_req_id or ""
                if source_id not in chunk.chunk_text:
                    raise ValueError("source_req_id is not visibly present in the chunk")
                expected_text = explicit_rows.get(source_id)
                if expected_text is not None:
                    if requirement.requirement_text != expected_text:
                        raise ValueError("source requirement text is not preserved exactly")
                    if expected_text not in requirement.evidence_quotes:
                        raise ValueError(
                            "source requirement evidence must contain the exact row text"
                        )
                elif _normalize(requirement.requirement_text) not in normalized_chunk:
                    raise ValueError("source requirement text is not preserved from the chunk")
            else:
                _require_atomic_generated_requirement(requirement.requirement_text)
            _require_semantic_markers(requirement)
        entities_by_ref = {entity.entity_ref: entity for entity in response.entities}
        for entity in response.entities:
            for quote in entity.evidence_quotes:
                _require_exact_quote(normalized_chunk, quote, "entity")
                _require_entity_grounding(entity.name, entity.aliases, quote, "entity")
        relationship_owners: dict[str, list[str]] = {
            relationship.relationship_ref: [] for relationship in response.relationships
        }
        for requirement in response.requirements:
            for relationship_ref in requirement.relationship_refs:
                relationship_owners[relationship_ref].append(requirement.requirement_ref)
        for relationship in response.relationships:
            _require_exact_quote(normalized_chunk, relationship.evidence_quote, "relationship")
            owners = relationship_owners[relationship.relationship_ref]
            if not owners:
                raise ValueError("relationship is not owned by a requirement")
            if len(owners) != 1:
                raise ValueError("relationship must be owned by exactly one requirement")
            requirement = requirements_by_ref[owners[0]]
            if not any(
                _normalize(relationship.evidence_quote) in _normalize(quote)
                or _normalize(quote) in _normalize(relationship.evidence_quote)
                for quote in requirement.evidence_quotes
            ):
                raise ValueError("relationship evidence is outside its requirement evidence")
            source = entities_by_ref[relationship.source_entity_ref]
            target = entities_by_ref[relationship.target_entity_ref]
            _require_entity_grounding(
                source.name,
                source.aliases,
                relationship.evidence_quote,
                "relationship source endpoint",
            )
            _require_entity_grounding(
                target.name,
                target.aliases,
                relationship.evidence_quote,
                "relationship target endpoint",
            )


def quote_span(chunk_text: str, quote: str) -> tuple[int, int]:
    """Resolve a quote to source offsets, permitting layout-only whitespace joining."""
    direct = chunk_text.find(quote)
    if direct >= 0:
        return direct, direct + len(quote)
    normalized_chunk, positions = _normalize_with_positions(chunk_text)
    normalized_quote = _normalize(quote)
    start = normalized_chunk.find(normalized_quote)
    if start < 0:
        raise ValueError("evidence quote is not present in the chunk")
    end_index = start + len(normalized_quote) - 1
    return positions[start], positions[end_index] + 1


def _require_exact_quote(normalized_chunk: str, quote: str, label: str) -> None:
    if not quote.strip() or _normalize(quote) not in normalized_chunk:
        raise ValueError(f"{label} evidence quote is not present in the chunk")


def _require_atomic_generated_requirement(requirement_text: str) -> None:
    """Reject headings, tables, or multiple prose items collapsed into one candidate."""
    normalized = _normalize(requirement_text)
    if requirement_text != normalized:
        raise ValueError("generated requirement must be one normalized atomic sentence")
    sentence_ends = list(_SENTENCE_END.finditer(normalized))
    if len(sentence_ends) != 1 or sentence_ends[0].end() != len(normalized):
        raise ValueError("generated requirement must be one complete atomic sentence")


def _entity_grounded(name: str, aliases: list[str], quote: str) -> bool:
    """Return True when the entity name or one alias is visibly present in the quote."""
    evidence_tokens = {_singular_token(token) for token in _WORD.findall(quote)}
    for candidate in (name, *aliases):
        candidate_tokens = {_singular_token(token) for token in _WORD.findall(candidate)}
        if candidate_tokens and candidate_tokens <= evidence_tokens:
            return True
    return False


def _require_entity_grounding(
    name: str,
    aliases: list[str],
    quote: str,
    label: str,
) -> None:
    """Require a visible name or alias in each entity-bearing evidence quote."""
    if not _entity_grounded(name, aliases, quote):
        raise ValueError(f"{label} is not visibly grounded in its evidence quote")


def _singular_token(token: str) -> str:
    normalized = token.casefold()
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        return normalized[:-1]
    return normalized


def _explicit_requirement_rows(chunk_text: str) -> dict[str, str]:
    """Return exact source-ID rows emitted by the layout-aware PDF parser."""
    result: dict[str, str] = {}
    for line in chunk_text.splitlines():
        normalized = _normalize(line)
        match = re.match(r"^([A-Z]{2,}(?:-[A-Z0-9]+){1,})\s+(.+)$", normalized)
        if match is None:
            continue
        source_id, requirement_text = match.groups()
        if _SOURCE_ID.fullmatch(source_id) and requirement_text:
            result[source_id] = requirement_text
    return result


def _evidence_quote_candidates(chunk_text: str) -> list[str]:
    """Build exact quote choices without modifying or repairing model output."""
    candidates: list[str] = []

    def add(value: str) -> None:
        normalized = _normalize(value)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for requirement_text in _explicit_requirement_rows(chunk_text).values():
        add(requirement_text)

    normalized_chunk = _normalize(chunk_text)
    start = 0
    for match in _SENTENCE_END.finditer(normalized_chunk):
        add(normalized_chunk[start : match.end()])
        start = match.end()
    add(normalized_chunk[start:])
    return candidates


def _require_semantic_markers(requirement: LLMRequirementCandidate) -> None:
    evidence_quotes = requirement.evidence_quotes
    requirement_text = _normalize(requirement.requirement_text).casefold()
    evidence_text = " ".join(_normalize(quote) for quote in evidence_quotes)
    missing = {
        _normalize(match.group(0)).casefold()
        for match in _SEMANTIC_MARKERS.finditer(evidence_text)
        if not _is_bare_decimal_marker(match.group(0))
        if _normalize(match.group(0)).casefold() not in requirement_text
    }
    if missing:
        raise ValueError(
            "requirement_text dropped modality, polarity, threshold, timing, or condition markers: "
            f"{sorted(missing)}"
        )


def _is_bare_decimal_marker(value: str) -> bool:
    """Exclude section/version identifiers while retaining decimal values with units."""
    return re.fullmatch(r"\d+\.\d+", _normalize(value)) is not None


def _normalize(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _normalize_with_positions(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    previous_space = False
    for index, char in enumerate(value):
        if char.isspace():
            if chars and not previous_space:
                chars.append(" ")
                positions.append(index)
            previous_space = True
        else:
            chars.append(char)
            positions.append(index)
            previous_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


__all__ = ["RequirementDiscoveryAgent", "quote_span"]
