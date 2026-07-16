"""One-call-per-chunk requirement/entity/relationship discovery."""

from __future__ import annotations

import json
import re

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.domain.schemas import (
    ManifestChunk,
    RequirementDiscoveryChunkResponse,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel

_SPACE = re.compile(r"\s+")


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
        self.validate_response(chunk, response)
        return response

    @staticmethod
    def validate_response(
        chunk: ManifestChunk,
        response: RequirementDiscoveryChunkResponse,
    ) -> None:
        """Apply deterministic quote, provenance, and association validation."""
        if response.chunk_id != chunk.chunk_id:
            raise ValueError("response chunk_id does not match the supplied manifest chunk")
        normalized_chunk = _normalize(chunk.chunk_text)
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
                if _normalize(requirement.requirement_text) not in normalized_chunk:
                    raise ValueError("source requirement text is not preserved from the chunk")
        for entity in response.entities:
            for quote in entity.evidence_quotes:
                _require_exact_quote(normalized_chunk, quote, "entity")
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
            for owner in owners:
                requirement = requirements_by_ref[owner]
                if not any(
                    _normalize(relationship.evidence_quote) in _normalize(quote)
                    or _normalize(quote) in _normalize(relationship.evidence_quote)
                    for quote in requirement.evidence_quotes
                ):
                    raise ValueError("relationship evidence is not associated with its requirement")


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
