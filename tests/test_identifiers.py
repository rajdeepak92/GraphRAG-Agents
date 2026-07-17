"""Determinism and run-independence of permanent/projection identities."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.identifiers import (
    make_chunk_id,
    make_entity_id,
    make_relationship_id,
    new_requirement_id,
    new_run_id,
)


def test_chunk_id_is_deterministic_and_run_independent() -> None:
    kwargs = {
        "chunk_text": "The service shall retain audit events.",
        "start_char": 0,
        "end_char": 38,
        "source_location": "page=1,row=1",
    }
    first = make_chunk_id(**kwargs)
    second = make_chunk_id(**kwargs)
    assert first == second
    # Distinct run IDs must never change the chunk identity.
    assert new_run_id("alpha") != new_run_id("alpha")
    assert make_chunk_id(**kwargs) == first


def test_entity_id_is_stable_and_project_scoped() -> None:
    assert make_entity_id("alpha", "gateway", "SYSTEM") == make_entity_id(
        "alpha", "gateway", "SYSTEM"
    )
    assert make_entity_id("alpha", "gateway", "SYSTEM") != make_entity_id(
        "beta", "gateway", "SYSTEM"
    )


def test_relationship_id_is_deterministic_and_provenance_scoped() -> None:
    kwargs = {
        "project": "alpha",
        "chunk_id": "CHK-1",
        "requirement_text_hash": "REQ-HASH",
        "source_entity_id": "ENT-A",
        "relationship_type": "CONTROLS",
        "target_entity_id": "ENT-B",
        "evidence_hash": "EVD-HASH",
    }
    assert make_relationship_id(**kwargs) == make_relationship_id(**kwargs)
    swapped = {**kwargs, "source_entity_id": "ENT-B", "target_entity_id": "ENT-A"}
    assert make_relationship_id(**swapped) != make_relationship_id(**kwargs)


def test_permanent_requirement_ids_are_unique() -> None:
    assert new_requirement_id() != new_requirement_id()
