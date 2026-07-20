"""Strict Stage 1.2, SIIMCS parsing, retry, reset, and boundary regressions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    RequirementDiscoveryAgent,
    _entity_grounded,
)
from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import ChunkingSettings
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import SemanticValidationError
from multi_agentic_graph_rag.domain.schemas import (
    ChunkLayout,
    ChunkManifest,
    LLMEntityCandidate,
    LLMRelationshipCandidate,
    LLMRequirementCandidate,
    ManifestChunk,
    RequirementDiscoveryChunkResponse,
)
from multi_agentic_graph_rag.llm_models.factory import create_reasoning_model
from multi_agentic_graph_rag.services.chunking import chunk_blocks
from multi_agentic_graph_rag.services.manifest import build_chunk_manifest
from multi_agentic_graph_rag.services.parsing import parse_document
from multi_agentic_graph_rag.services.project_reset import reset_project
from multi_agentic_graph_rag.services.retry import retry_transient_once
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    _call_combined_model,
    _checkpoint_validated_response,
)


def _chunk(text: str, *, block_type: str = "table_row") -> ManifestChunk:
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
            section="Requirements",
            block_types=[block_type],  # type: ignore[list-item]
            source_location="page=1,row=1",
        ),
        source_provenance={"source_path": "doc.pdf"},
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _response(text: str, *, source_id: str = "BR-COM-002") -> dict[str, Any]:
    return {
        "chunk_id": "CHK-1",
        "requirements": [
            {
                "requirement_ref": "REQREF-1",
                "source_req_id": source_id,
                "source_req_id_type": "source",
                "requirement_text": text,
                "requirement_type": "Functional Requirement",
                "priority": "High",
                "constraints": ["configurable intervals"],
                "entity_refs": ["ENTREF-1", "ENTREF-2"],
                "relationship_refs": ["RELREF-1"],
                "evidence_quotes": [text],
                "confidence": 0.99,
            }
        ],
        "entities": [
            {
                "entity_ref": "ENTREF-1",
                "name": "controller",
                "normalized_name": "controller",
                "entity_type": "COMPONENT",
                "aliases": [],
                "evidence_quotes": [text],
                "confidence": 0.99,
            },
            {
                "entity_ref": "ENTREF-2",
                "name": "configured sensors",
                "normalized_name": "configured sensors",
                "entity_type": "COMPONENT",
                "aliases": [],
                "evidence_quotes": [text],
                "confidence": 0.98,
            },
        ],
        "relationships": [
            {
                "relationship_ref": "RELREF-1",
                "source_entity_ref": "ENTREF-1",
                "relationship_type": "CONNECTS_TO",
                "target_entity_ref": "ENTREF-2",
                "evidence_quote": text,
                "confidence": 0.95,
            }
        ],
    }


class _Model:
    provider_name = "test"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.max_attempts: list[int] = []

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.max_attempts.append(int(kwargs["max_attempts"]))
        return kwargs["schema"].model_validate(self.payload)


def test_strict_schema_rejects_defaults_coercion_and_extra_fields() -> None:
    text = "The controller shall poll each configured sensor."
    payload = _response(text)
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        RequirementDiscoveryChunkResponse.model_validate(payload)
    with pytest.raises(ValidationError):
        LLMRequirementCandidate.model_validate(
            {
                "requirement_ref": "r1",
                "source_req_id": None,
                "source_req_id_type": "generated",
                "requirement_text": text,
            }
        )
    with pytest.raises(ValidationError):
        LLMEntityCandidate.model_validate(
            {
                "entity_ref": "e1",
                "name": "controller",
                "normalized_name": "controller",
                "entity_type": "DEVICE",
                "aliases": [],
                "evidence_quotes": [text],
                "confidence": 0.9,
            }
        )
    with pytest.raises(ValidationError):
        LLMRelationshipCandidate.model_validate(
            {
                "relationship_ref": "r1",
                "source_entity_ref": "e1",
                "relationship_type": "ARBITRARY",
                "target_entity_ref": "e2",
                "evidence_quote": text,
                "confidence": 0.9,
            }
        )


def test_explicit_rows_require_complete_exact_coverage_in_one_call() -> None:
    sentence = (
        "The controller shall continuously poll Sensor-1, Sensor-2, and Sensor-3 "
        "at configurable intervals."
    )
    chunk = _chunk(f"BR-COM-002 {sentence}\nBR-COM-003 Polling shall not block other sensors.")
    payload = _response(sentence)
    model = _Model(payload)
    with pytest.raises(SemanticValidationError, match="coverage mismatch"):
        RequirementDiscoveryAgent(model).discover(chunk)
    assert model.calls == 1
    assert model.max_attempts == [1]


def test_descriptive_chunk_may_return_all_empty_arrays() -> None:
    chunk = _chunk("Background and stakeholder context.", block_type="paragraph")
    payload = {"chunk_id": "CHK-1", "requirements": [], "entities": [], "relationships": []}
    assert RequirementDiscoveryAgent(_Model(payload)).discover(chunk).requirements == []


def test_generated_requirement_must_not_collapse_sections_into_one_candidate() -> None:
    text = (
        "Objective Description\n"
        "Real-Time Monitoring\n"
        "Collect and present current operating data.\n\n"
        "3. Business Scope\n"
        "Cloud-based storage and monitoring."
    )
    payload = {
        "chunk_id": "CHK-1",
        "requirements": [
            {
                "requirement_ref": "REQREF-1",
                "source_req_id": None,
                "source_req_id_type": "generated",
                "requirement_text": text,
                "requirement_type": "Business Requirement",
                "priority": "High",
                "constraints": [],
                "entity_refs": [],
                "relationship_refs": [],
                "evidence_quotes": [text],
                "confidence": 0.99,
            }
        ],
        "entities": [],
        "relationships": [],
    }
    with pytest.raises(SemanticValidationError, match="atomic sentence"):
        RequirementDiscoveryAgent(_Model(payload)).discover(_chunk(text, block_type="paragraph"))


def test_relationship_must_have_one_requirement_owner() -> None:
    first = "The controller shall collect data from configured sensors."
    second = "The controller shall poll configured sensors every second."
    payload = _response(first, source_id="BR-SEN-001")
    payload["requirements"].append(
        {
            "requirement_ref": "REQREF-2",
            "source_req_id": "BR-SEN-002",
            "source_req_id_type": "source",
            "requirement_text": second,
            "requirement_type": "Functional Requirement",
            "priority": "High",
            "constraints": [],
            "entity_refs": ["ENTREF-1", "ENTREF-2"],
            "relationship_refs": ["RELREF-1"],
            "evidence_quotes": [second],
            "confidence": 0.99,
        }
    )
    chunk = _chunk(f"BR-SEN-001 {first}\nBR-SEN-002 {second}")
    with pytest.raises(SemanticValidationError, match="exactly one requirement"):
        RequirementDiscoveryAgent(_Model(payload)).discover(chunk)


def test_ungrounded_entity_quote_is_pruned_not_fatal() -> None:
    # A model that over-attaches an ungrounded quote to an entity must not halt the
    # chunk. The ungrounded quote is pruned; an entity left with no grounded quote is
    # dropped and its references cascade out of requirements and relationships so
    # every stored evidence quote stays grounded.
    sentence = "The controller shall collect data from configured sensors."
    payload = _response(sentence, source_id="BR-SEN-001")
    payload["entities"][1]["name"] = "Modbus communication interface"
    payload["entities"][1]["normalized_name"] = "modbus communication interface"
    response = RequirementDiscoveryAgent(_Model(payload)).discover(_chunk(f"BR-SEN-001 {sentence}"))
    assert [entity.entity_ref for entity in response.entities] == ["ENTREF-1"]
    assert response.relationships == []
    requirement = response.requirements[0]
    assert requirement.entity_refs == ["ENTREF-1"]
    assert requirement.relationship_refs == []
    for entity in response.entities:
        for quote in entity.evidence_quotes:
            assert _entity_grounded(entity.name, entity.aliases, quote)


def test_relationship_quote_must_ground_both_endpoints() -> None:
    sentence = "The controller shall collect data from configured sensors."
    interface_evidence = "The Modbus communication interface is available."
    payload = _response(sentence, source_id="BR-SEN-001")
    payload["entities"][1]["name"] = "Modbus communication interface"
    payload["entities"][1]["normalized_name"] = "modbus communication interface"
    payload["entities"][1]["evidence_quotes"] = [interface_evidence]
    chunk_text = f"BR-SEN-001 {sentence}\n{interface_evidence}"
    with pytest.raises(SemanticValidationError, match="target endpoint"):
        RequirementDiscoveryAgent(_Model(payload)).discover(_chunk(chunk_text))


def test_valid_relationship_is_owned_and_grounded() -> None:
    sentence = "The controller shall collect data from configured sensors."
    response = RequirementDiscoveryAgent(
        _Model(_response(sentence, source_id="BR-SEN-001"))
    ).discover(_chunk(f"BR-SEN-001 {sentence}"))
    assert response.relationships[0].relationship_ref == "RELREF-1"


def test_siimcs_rows_reconstruct_split_ids_and_wrapped_sentences() -> None:
    blocks, fingerprint = parse_document(Path("documents/inbox/SIIMCS/SIIMCS_BRD_V1.pdf"))
    rows = {block.metadata.get("source_req_id"): block for block in blocks}
    assert fingerprint == "positioned-layout-v2"
    assert rows["BR-COM-002"].normalized_text.endswith("at configurable intervals.")
    assert rows["BR-CFG-002"].normalized_text.endswith("shall be rejected.")
    assert rows["BR-VAL-003"].normalized_text.endswith("trigger actions.")
    assert rows["BR-ALT-002"].normalized_text.endswith("notification channels.")
    assert all(block.block_type == "table_row" for key, block in rows.items() if key)
    assert not any(block.normalized_text.endswith("/ 7") for block in blocks)


def test_siimcs_requirement_subsections_are_separate_chunks() -> None:
    blocks, _ = parse_document(Path("documents/inbox/SIIMCS/SIIMCS_BRD_V1.pdf"))
    chunks, chunker_fingerprint = chunk_blocks(blocks, ChunkingSettings())

    communication = next(chunk for chunk in chunks if "BR-COM-001" in chunk.chunk_text)
    configuration = next(chunk for chunk in chunks if "BR-CFG-001" in chunk.chunk_text)

    assert chunker_fingerprint.startswith("block-pack-v4-structural-headings:")
    assert communication.chunk_id != configuration.chunk_id
    assert all(f"BR-COM-{index:03d}" in communication.chunk_text for index in range(1, 6))
    assert "BR-CFG-001" not in communication.chunk_text
    assert all(f"BR-CFG-{index:03d}" in configuration.chunk_text for index in range(1, 6))
    assert "BR-COM-005" not in configuration.chunk_text


def test_transient_failure_retries_exactly_once() -> None:
    calls = 0

    class TransientError(Exception):
        pass

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientError
        return "ok"

    assert retry_transient_once(operation) == "ok"
    assert calls == 2


def test_terminal_failure_is_not_retried() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise SemanticValidationError("invalid")

    with pytest.raises(SemanticValidationError):
        retry_transient_once(operation)
    assert calls == 1


def test_validated_checkpoint_prevents_repeated_model_call() -> None:
    sentence = "The controller shall poll each configured sensor."
    chunk = _chunk(sentence, block_type="paragraph")
    manifest: ChunkManifest = build_chunk_manifest(
        project="alpha",
        run_id="RUN-1",
        chunks=[chunk],
    )
    model = _Model(
        {
            "chunk_id": "CHK-1",
            "requirements": [],
            "entities": [],
            "relationships": [],
        }
    )
    agent = RequirementDiscoveryAgent(model)
    runtime = SimpleNamespace(agent=agent)
    state: dict[str, Any] = {
        "project": "alpha",
        "run_id": "RUN-1",
        "manifest": manifest.model_dump(mode="json"),
        "current_index": 0,
    }
    state.update(_call_combined_model(state, runtime))  # type: ignore[arg-type]
    state.update(_checkpoint_validated_response(state))
    _call_combined_model(state, runtime)  # type: ignore[arg-type]
    assert model.calls == 1


def test_stage12_huggingface_cap_does_not_lower_other_stages() -> None:
    settings = load_config()
    settings.reasoning_model.provider = "huggingface"
    regular = create_reasoning_model(settings)
    discovery = create_reasoning_model(settings, stage12=True)
    assert regular.settings.max_new_tokens == settings.huggingface.max_new_tokens  # type: ignore[attr-defined]
    assert discovery.settings.max_new_tokens == 1536  # type: ignore[attr-defined]


def test_reset_is_explicit_and_clears_local_project_state(tmp_path: Path) -> None:
    settings = load_config()
    settings.paths.generated_dir = tmp_path / "generated"
    settings.paths.chroma_persist_dir = tmp_path / "chroma"
    settings.postgres.mode = "local_json"
    settings.postgres.local_path = tmp_path / "postgres.jsonl"
    settings.neo4j.mode = "local_json"
    settings.neo4j.local_path = tmp_path / "neo4j.jsonl"
    project_dir = settings.paths.generated_dir / "alpha"
    project_dir.mkdir(parents=True)
    (project_dir / "artifact.json").write_text("{}", encoding="utf-8")
    Neo4jStore(settings).upsert_chunk(project="alpha", run_id="RUN-1", chunk=_chunk("Background."))
    PostgresStore(settings).record_run(
        run_id="RUN-1",
        project="alpha",
        stage="stage-1.1",
        status="started",
        payload={},
    )
    summary = reset_project("alpha", settings)
    assert summary["project"] == "alpha"
    assert not project_dir.exists()
    assert Neo4jStore(settings).read_chunk("alpha", "CHK-1") is None


def test_prompt_and_stage_boundary_document_strict_contract() -> None:
    prompt = PromptRequirementDiscovery.SYSTEM.value
    for token in (
        '"chunk_id"',
        '"requirements"',
        '"entities"',
        '"relationships"',
        "BR-COM-002",
        "genuinely descriptive",
        "COMMUNICATES_VIA",
        "exactly one requirement",
        "one atomic sentence",
    ):
        assert token in prompt
    workflow = Path(
        "src/multi_agentic_graph_rag/workflows/requirement_discovery_graph.py"
    ).read_text(encoding="utf-8")
    assert "create_embedding_model" not in workflow
    assert "ChromaStore" not in workflow
    assert "checkpoint_validated_response" in workflow
