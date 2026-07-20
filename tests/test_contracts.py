"""Schema and checksum regression tests for the simplified contracts."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGenerationAgent
from multi_agentic_graph_rag.domain.schemas import (
    AcceptanceCriterion,
    CanonicalUserStory,
    ChunkLayout,
    ManifestChunk,
    SourceProvenance,
    SourceRequirementType,
    TestScenarioGenerationDraft,
    Traceability,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.manifest import build_chunk_manifest


def _chunk(index: int = 0) -> ManifestChunk:
    text = "BR-1 The service shall retain audit events."
    return ManifestChunk(
        chunk_id=f"CHK-{index}",
        sequence_index=index,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Audit",
            block_types=["paragraph"],
            source_location="page=1",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def test_source_provenance_pair_is_strict() -> None:
    assert (
        SourceProvenance(source_req_id="BR-1", source_req_id_type="source").source_req_id == "BR-1"
    )
    with pytest.raises(ValidationError):
        SourceProvenance(source_req_id=None, source_req_id_type="source")
    with pytest.raises(ValidationError):
        SourceProvenance(source_req_id="BR-1", source_req_id_type="generated")


@pytest.mark.parametrize(
    ("source_req_id", "source_req_id_type"),
    [(None, "generated"), ("BR-1", "source")],
)
def test_stage3_provenance_is_system_stamped(
    source_req_id: str | None,
    source_req_id_type: SourceRequirementType,
) -> None:
    story = CanonicalUserStory(
        source_req_id=source_req_id,
        source_req_id_type=source_req_id_type,
        story_id="US-1",
        requirement_ids=["REQ-1"],
        title="Retain events",
        priority="High",
        persona="Auditor",
        user_story=UserStoryStatement(
            as_a="auditor",
            i_want="events retained",
            so_that="I can investigate",
        ),
        acceptance_criteria=[
            AcceptanceCriterion(
                criterion_id="AC-1",
                title="Stored",
                given="An event exists",
                when="It is recorded",
                then="It remains available",
            )
        ],
        business_rules=[],
        confidence=1.0,
        traceability=Traceability(
            evidence_chunk_ids=["CHK-1"],
            entity_ids=[],
            relationship_ids=[],
        ),
    )
    payload: dict[str, Any] = {
        "story_id": "US-1",
        "requirement_ids": ["REQ-1"],
        "test_scenarios": [
            {
                "scenario_ref": "SC-1",
                "title": "Retain one event",
                "description": "Verify retention.",
                "scenario_type": "Positive",
                "priority": "High",
                "preconditions": ["An event exists"],
                "action": "Record the event.",
                "expected_result": "The event remains available.",
                "covered_acceptance_criterion_ids": ["AC-1"],
                "evidence_chunk_ids": ["CHK-1"],
                "supporting_entity_ids": [],
                "supporting_relationship_ids": [],
                "confidence": 1.0,
            }
        ],
    }

    draft = TestScenarioGenerationDraft.model_validate(payload)
    response = TestScenarioGenerationAgent._stamp_provenance(story, draft)

    assert response.test_scenarios[0].source_req_id == source_req_id
    assert response.test_scenarios[0].source_req_id_type == source_req_id_type
    rogue_payload = dict(payload)
    rogue_payload["test_scenarios"] = [
        {
            **payload["test_scenarios"][0],
            "source_req_id": "REQ-1",
            "source_req_id_type": "source",
        }
    ]
    with pytest.raises(ValidationError):
        TestScenarioGenerationDraft.model_validate(rogue_payload)


def test_manifest_requires_contiguous_persisted_chunks_and_checksum() -> None:
    manifest = build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[_chunk()])
    assert manifest.checksum.startswith("sha256:")
    pending = _chunk().model_copy(update={"chroma_status": "pending"})
    with pytest.raises(ValidationError):
        build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[pending])
    with pytest.raises(ValidationError):
        build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[_chunk(1)])
