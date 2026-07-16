"""Combined Stage 1.2 call and grounding tests."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    RequirementDiscoveryAgent,
)
from multi_agentic_graph_rag.domain.schemas import ChunkLayout, ManifestChunk


class _Model:
    provider_name = "test"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.max_attempts: list[int] = []

    def generate_structured(self, **kwargs: Any) -> Any:
        self.max_attempts.append(int(kwargs["max_attempts"]))
        return kwargs["schema"].model_validate(self.payload)


def _chunk() -> ManifestChunk:
    text = "BR-COM-001 The gateway shall communicate via TLS."
    return ManifestChunk(
        chunk_id="CHK-1",
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Communications",
            block_types=["table_row"],
            source_location="page=1,row=1",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _payload() -> dict[str, Any]:
    quote = "The gateway shall communicate via TLS."
    return {
        "chunk_id": "CHK-1",
        "requirements": [
            {
                "requirement_ref": "req_1",
                "source_req_id": "BR-COM-001",
                "source_req_id_type": "source",
                "requirement_text": quote,
                "requirement_type": "Functional Requirement",
                "priority": "Medium",
                "constraints": [],
                "entity_refs": ["entity_1", "entity_2"],
                "relationship_refs": ["relationship_1"],
                "evidence_quotes": [quote],
                "confidence": 0.98,
            }
        ],
        "entities": [
            {
                "entity_ref": "entity_1",
                "name": "gateway",
                "normalized_name": "gateway",
                "entity_type": "SYSTEM",
                "aliases": [],
                "evidence_quotes": ["gateway"],
                "confidence": 0.99,
            },
            {
                "entity_ref": "entity_2",
                "name": "TLS",
                "normalized_name": "tls",
                "entity_type": "INTERFACE",
                "aliases": [],
                "evidence_quotes": ["TLS"],
                "confidence": 0.99,
            },
        ],
        "relationships": [
            {
                "relationship_ref": "relationship_1",
                "source_entity_ref": "entity_1",
                "relationship_type": "COMMUNICATES_VIA",
                "target_entity_ref": "entity_2",
                "evidence_quote": quote,
                "confidence": 0.96,
            }
        ],
    }


def test_discovery_uses_one_provider_attempt_and_preserves_source_id() -> None:
    model = _Model(_payload())
    response = RequirementDiscoveryAgent(model).discover(_chunk())
    assert model.max_attempts == [1]
    assert response.requirements[0].source_req_id == "BR-COM-001"


def test_discovery_rejects_invented_source_prefix() -> None:
    payload = _payload()
    payload["requirements"][0]["source_req_id"] = "SYS-BR-COM-001"
    with pytest.raises(ValueError, match="not visibly present"):
        RequirementDiscoveryAgent(_Model(payload)).discover(_chunk())
