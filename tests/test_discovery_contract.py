"""Combined Stage 1.2 call and grounding tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    RequirementDiscoveryAgent,
)
from multi_agentic_graph_rag.domain.errors import SemanticValidationError
from multi_agentic_graph_rag.domain.schemas import ChunkLayout, ManifestChunk


class _Model:
    provider_name = "test"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.max_attempts: list[int] = []
        self.prompts: list[str] = []

    def generate_structured(self, **kwargs: Any) -> Any:
        self.max_attempts.append(int(kwargs["max_attempts"]))
        self.prompts.append(str(kwargs["prompt"]))
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
                "requirement_ref": "REQREF-1",
                "source_req_id": "BR-COM-001",
                "source_req_id_type": "source",
                "confidence": 0.98,
                "requirement_type": "Functional Requirement",
                "priority": "Medium",
                "requirement_text": quote,
                "constraints": [],
                "entity_refs": ["ENTREF-1", "ENTREF-2"],
                "relationship_refs": ["RELREF-1"],
                "evidence_quotes": [quote],
            }
        ],
        "entities": [
            {
                "entity_ref": "ENTREF-1",
                "name": "gateway",
                "normalized_name": "gateway",
                "entity_type": "COMPONENT",
                "aliases": [],
                "evidence_quotes": [quote],
                "confidence": 0.98,
            },
            {
                "entity_ref": "ENTREF-2",
                "name": "TLS",
                "normalized_name": "tls",
                "entity_type": "INTERFACE",
                "aliases": [],
                "evidence_quotes": [quote],
                "confidence": 0.98,
            },
        ],
        "relationships": [
            {
                "relationship_ref": "RELREF-1",
                "source_entity_ref": "ENTREF-1",
                "relationship_type": "COMMUNICATES_VIA",
                "target_entity_ref": "ENTREF-2",
                "evidence_quote": quote,
                "confidence": 0.98,
            }
        ],
    }


def test_discovery_uses_one_call_and_preserves_source_id() -> None:
    model = _Model(_payload())
    response = RequirementDiscoveryAgent(model).discover(_chunk())
    assert model.max_attempts == [1]
    assert response.requirements[0].source_req_id == "BR-COM-001"


def test_discovery_supplies_normalized_canonical_evidence_source() -> None:
    model = _Model(_payload())
    RequirementDiscoveryAgent(model).discover(_chunk())

    prompt = json.loads(model.prompts[0])
    assert prompt["evidence_source_text"] == ("BR-COM-001 The gateway shall communicate via TLS.")
    assert prompt["evidence_quote_candidates"] == [
        "The gateway shall communicate via TLS.",
        "BR-COM-001 The gateway shall communicate via TLS.",
    ]
    assert prompt["source_requirement_rows"] == {
        "BR-COM-001": "The gateway shall communicate via TLS."
    }


def test_discovery_keeps_allowlisted_relationship_type() -> None:
    response = RequirementDiscoveryAgent(_Model(_payload())).discover(_chunk())
    assert response.relationships[0].relationship_type == "COMMUNICATES_VIA"


def test_discovery_rejects_invented_source_prefix() -> None:
    payload = _payload()
    payload["requirements"][0]["source_req_id"] = "SYS-BR-COM-001"
    with pytest.raises(SemanticValidationError):
        RequirementDiscoveryAgent(_Model(payload)).discover(_chunk())


def test_source_row_evidence_with_id_prefix_is_repaired() -> None:
    """A source row quoted with its ID prefix validates and is repaired to the row text."""
    payload = _payload()
    payload["requirements"][0]["evidence_quotes"] = [
        "BR-COM-001 The gateway shall communicate via TLS."
    ]
    response = RequirementDiscoveryAgent(_Model(payload)).discover(_chunk())
    assert response.requirements[0].evidence_quotes == ["The gateway shall communicate via TLS."]


def test_source_row_prefix_repair_preserves_other_evidence_quotes() -> None:
    payload = _payload()
    payload["requirements"][0]["evidence_quotes"] = [
        "BR-COM-001 The gateway shall communicate via TLS.",
        "The gateway shall communicate via TLS.",
    ]
    response = RequirementDiscoveryAgent(_Model(payload)).discover(_chunk())
    # The prefixed quote collapses onto the bare row text without duplicating it.
    assert response.requirements[0].evidence_quotes == ["The gateway shall communicate via TLS."]


def _br_alt_rows() -> dict[str, str]:
    return {
        "BR-ALT-001": "Users shall be able to configure warning and critical thresholds.",
        "BR-ALT-002": "The system shall alert operators when a threshold is exceeded.",
        "BR-ALT-003": "Alerts shall be delivered within 5 seconds of detection.",
        "BR-ALT-004": "Operators shall be able to acknowledge active alerts.",
        "BR-ALT-005": "The system shall log every alert with a timestamp.",
    }


def _br_alt_chunk() -> ManifestChunk:
    rows = _br_alt_rows()
    text = "\n".join(f"{source_id} {row}" for source_id, row in rows.items())
    return ManifestChunk(
        chunk_id="CHK-ALT",
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Alerting",
            block_types=["table_row"],
            source_location="page=1,rows=1-5",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _br_alt_requirement(index: int, source_id: str, row: str, quote: str) -> dict[str, Any]:
    return {
        "requirement_ref": f"REQREF-{index}",
        "source_req_id": source_id,
        "source_req_id_type": "source",
        "confidence": 0.97,
        "requirement_type": "Functional Requirement",
        "priority": "Medium",
        "requirement_text": row,
        "constraints": [],
        "entity_refs": [],
        "relationship_refs": [],
        "evidence_quotes": [quote],
    }


def _br_alt_payload(prefixed: bool) -> dict[str, Any]:
    rows = _br_alt_rows()
    requirements = [
        _br_alt_requirement(
            index,
            source_id,
            row,
            f"{source_id} {row}" if prefixed else row,
        )
        for index, (source_id, row) in enumerate(rows.items(), start=1)
    ]
    return {
        "chunk_id": "CHK-ALT",
        "requirements": requirements,
        "entities": [],
        "relationships": [],
    }


def test_br_alt_rows_validate_with_id_prefixed_evidence() -> None:
    """Regression: all five BR-ALT rows quoted with their ID prefix validate and repair."""
    response = RequirementDiscoveryAgent(_Model(_br_alt_payload(prefixed=True))).discover(
        _br_alt_chunk()
    )
    rows = _br_alt_rows()
    assert {req.source_req_id for req in response.requirements} == set(rows)
    for requirement in response.requirements:
        assert requirement.evidence_quotes == [rows[requirement.source_req_id]]


def test_unrelated_longer_evidence_is_still_rejected() -> None:
    """A quote spanning two rows is not a bare-prefix match, so it stays rejected."""
    payload = _br_alt_payload(prefixed=True)
    # Replace one row's evidence with two concatenated rows: present in the chunk and
    # grounded, but not "<source_req_id> <expected_text>" for the claimed row.
    payload["requirements"][0]["evidence_quotes"] = [
        "BR-ALT-001 Users shall be able to configure warning and critical thresholds. "
        "BR-ALT-002 The system shall alert operators when a threshold is exceeded."
    ]
    with pytest.raises(SemanticValidationError):
        RequirementDiscoveryAgent(_Model(payload)).discover(_br_alt_chunk())


def _generated_chunk(text: str, chunk_id: str = "CHK-GEN") -> ManifestChunk:
    return ManifestChunk(
        chunk_id=chunk_id,
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Definition of Done",
            block_types=["paragraph"],
            source_location="page=1",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _generated_payload(
    requirement_text: str, evidence_quote: str, chunk_id: str = "CHK-GEN"
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "requirements": [
            {
                "requirement_ref": "REQREF-1",
                "source_req_id": None,
                "source_req_id_type": "generated",
                "confidence": 1.0,
                "requirement_type": "Validation Requirement",
                "priority": "High",
                "requirement_text": requirement_text,
                "constraints": [],
                "entity_refs": [],
                "relationship_refs": [],
                "evidence_quotes": [evidence_quote],
            }
        ],
        "entities": [],
        "relationships": [],
    }


def test_generated_requirement_ignores_definition_of_done_framing_when() -> None:
    """A framing-only 'when' inside a Definition of Done preamble is not a dropped marker."""
    evidence = (
        "Definition of Done The feature is complete when: "
        "Three configured sensors are monitored successfully."
    )
    requirement_text = "Three configured sensors must be monitored successfully."
    response = RequirementDiscoveryAgent(
        _Model(_generated_payload(requirement_text, evidence))
    ).discover(_generated_chunk(evidence))
    assert response.requirements[0].source_req_id_type == "generated"
    assert response.requirements[0].evidence_quotes == [evidence]


def test_generated_requirement_still_fails_on_real_dropped_when_condition() -> None:
    """A real behavioral 'when' condition dropped from requirement_text is still rejected."""
    evidence = "The system shall alert when sensor voltage exceeds the threshold."
    requirement_text = "The system shall alert on sensor voltage threshold breach."
    with pytest.raises(SemanticValidationError):
        RequirementDiscoveryAgent(_Model(_generated_payload(requirement_text, evidence))).discover(
            _generated_chunk(evidence)
        )


def test_semantic_marker_evidence_text_strips_framing_for_generated_only() -> None:
    """The framing strip is deterministic and applies only to generated requirements."""
    from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
        _semantic_marker_evidence_text,
    )
    from multi_agentic_graph_rag.domain.schemas import LLMRequirementCandidate

    framing = (
        "Definition of Done The feature is complete when: "
        "Three configured sensors are monitored successfully."
    )

    def _candidate(source_type: str, source_id: str | None) -> LLMRequirementCandidate:
        return LLMRequirementCandidate(
            requirement_ref="REQREF-1",
            source_req_id=source_id,
            source_req_id_type=source_type,
            requirement_text="Three configured sensors must be monitored successfully.",
            requirement_type="Validation Requirement",
            priority="High",
            constraints=[],
            entity_refs=[],
            relationship_refs=[],
            evidence_quotes=[framing],
            confidence=1.0,
        )

    generated = _semantic_marker_evidence_text(_candidate("generated", None))
    assert generated == "Three configured sensors are monitored successfully."
    assert "when" not in generated.casefold()

    source = _semantic_marker_evidence_text(_candidate("source", "BR-GEN-001"))
    assert source == framing  # source evidence is scanned verbatim, framing intact


def _cfg_chunk() -> ManifestChunk:
    text = (
        "BR-CFG-001 The controller shall validate configuration during startup.\n"
        "BR-CFG-004 Authorized users shall be able to update thresholds and polling settings."
    )
    return ManifestChunk(
        chunk_id="CHK-CFG",
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Configuration",
            block_types=["table_row"],
            source_location="page=1,rows=1-2",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _cfg_payload() -> dict[str, Any]:
    startup = "The controller shall validate configuration during startup."
    settings = "Authorized users shall be able to update thresholds and polling settings."
    return {
        "chunk_id": "CHK-CFG",
        "requirements": [
            {
                "requirement_ref": "REQREF-1",
                "source_req_id": "BR-CFG-001",
                "source_req_id_type": "source",
                "confidence": 1.0,
                "requirement_type": "Functional Requirement",
                "priority": "High",
                "requirement_text": startup,
                "constraints": [],
                "entity_refs": ["ENT-CONTROLLER"],
                "relationship_refs": [],
                "evidence_quotes": [startup],
            },
            {
                "requirement_ref": "REQREF-2",
                "source_req_id": "BR-CFG-004",
                "source_req_id_type": "source",
                "confidence": 1.0,
                "requirement_type": "Configuration Requirement",
                "priority": "High",
                "requirement_text": settings,
                "constraints": [],
                "entity_refs": ["ENT-USER", "ENT-CONTROLLER"],
                "relationship_refs": ["RELREF-1"],
                "evidence_quotes": [settings],
            },
        ],
        "entities": [
            {
                "entity_ref": "ENT-CONTROLLER",
                "name": "controller",
                "normalized_name": "controller",
                "entity_type": "COMPONENT",
                "aliases": [],
                "evidence_quotes": [startup],
                "confidence": 1.0,
            },
            {
                "entity_ref": "ENT-USER",
                "name": "Authorized users",
                "normalized_name": "authorized users",
                "entity_type": "ACTOR",
                "aliases": [],
                "evidence_quotes": [settings],
                "confidence": 1.0,
            },
        ],
        "relationships": [
            {
                "relationship_ref": "RELREF-1",
                "source_entity_ref": "ENT-USER",
                "relationship_type": "CONTROLS",
                "target_entity_ref": "ENT-CONTROLLER",
                # Grounds the source ("Authorized users") but not the target ("controller").
                "evidence_quote": settings,
                "confidence": 1.0,
            }
        ],
    }


def test_relationship_ungrounded_endpoint_is_pruned_not_failed() -> None:
    """A relationship whose quote misses an endpoint is dropped; the chunk still validates."""
    response = RequirementDiscoveryAgent(_Model(_cfg_payload())).discover(_cfg_chunk())
    assert response.relationships == []
    settings_req = next(req for req in response.requirements if req.source_req_id == "BR-CFG-004")
    assert settings_req.relationship_refs == []
    # Both requirements survive; entities stay because requirements still reference them.
    assert {req.source_req_id for req in response.requirements} == {"BR-CFG-001", "BR-CFG-004"}
    assert {ent.entity_ref for ent in response.entities} == {"ENT-CONTROLLER", "ENT-USER"}


def test_relationship_prune_drops_orphaned_entity() -> None:
    """An entity referenced only by a pruned relationship is cascaded out."""
    payload = _cfg_payload()
    # Add an entity that only the (to-be-pruned) relationship references.
    payload["entities"].append(
        {
            "entity_ref": "ENT-ORPHAN",
            "name": "startup",
            "normalized_name": "startup",
            "entity_type": "EVENT",
            "aliases": [],
            "evidence_quotes": ["The controller shall validate configuration during startup."],
            "confidence": 1.0,
        }
    )
    payload["relationships"][0]["source_entity_ref"] = "ENT-ORPHAN"
    response = RequirementDiscoveryAgent(_Model(payload)).discover(_cfg_chunk())
    assert response.relationships == []
    assert "ENT-ORPHAN" not in {ent.entity_ref for ent in response.entities}


def test_grounded_relationship_is_not_pruned() -> None:
    """A relationship grounded in both endpoints survives pruning unchanged."""
    response = RequirementDiscoveryAgent(_Model(_payload())).discover(_chunk())
    assert len(response.relationships) == 1
    assert response.relationships[0].relationship_ref == "RELREF-1"
