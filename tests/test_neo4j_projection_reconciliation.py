"""Chunk-authoritative Stage 1.2 semantic projection regressions."""

from __future__ import annotations

from pathlib import Path

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalEntity,
    CanonicalRelationship,
    EntityMention,
    Evidence,
)


def _entity(entity_id: str) -> CanonicalEntity:
    return CanonicalEntity(
        entity_id=entity_id,
        name=entity_id,
        normalized_name=entity_id.casefold(),
        entity_type="COMPONENT",
        aliases=[],
        mentions=[
            EntityMention(
                chunk_id="CHK-1",
                surface_text=entity_id,
                start_char=0,
                end_char=len(entity_id),
            )
        ],
    )


def _relationship(relationship_id: str, confidence: float) -> CanonicalRelationship:
    return CanonicalRelationship(
        relationship_id=relationship_id,
        chunk_id="CHK-1",
        source_entity_id="ENT-1",
        relationship_type="USES",
        target_entity_id="ENT-2",
        confidence=confidence,
        evidence=[
            Evidence(
                evidence_id=f"EVD-{relationship_id}",
                chunk_id="CHK-1",
                quote="ENT-1 uses ENT-2.",
                start_char=0,
                end_char=17,
            )
        ],
    )


def test_replay_prunes_stale_chunk_edges_and_refreshes_confidence(tmp_path: Path) -> None:
    settings = load_config()
    settings.neo4j.mode = "local_json"
    settings.neo4j.local_path = tmp_path / "neo4j.jsonl"
    store = Neo4jStore(settings)
    entities = [_entity("ENT-1"), _entity("ENT-2")]
    original = _relationship("REL-KEEP", 0.5)
    stale = _relationship("REL-STALE", 0.4)
    store.upsert_semantic_projection(
        project="alpha",
        chunk_id="CHK-1",
        entities=entities,
        mentioned_entity_ids={"ENT-1", "ENT-2"},
        relationships=[original, stale],
        requirement_text_hash_by_relationship={
            "REL-KEEP": "REQ-HASH",
            "REL-STALE": "REQ-HASH",
        },
    )

    refreshed = _relationship("REL-KEEP", 0.9)
    store.prune_semantic_projection(
        project="alpha",
        chunk_id="CHK-1",
        entity_ids={"ENT-1"},
        relationship_ids={"REL-KEEP"},
    )
    store.upsert_semantic_projection(
        project="alpha",
        chunk_id="CHK-1",
        entities=entities,
        mentioned_entity_ids={"ENT-1"},
        relationships=[refreshed],
        requirement_text_hash_by_relationship={"REL-KEEP": "REQ-HASH"},
    )

    projection = store.read_semantic_projection(
        project="alpha",
        chunk_id="CHK-1",
        entity_ids={"ENT-1", "ENT-2"},
        relationship_ids={"REL-KEEP", "REL-STALE"},
    )
    assert projection.entity_ids == {"ENT-1", "ENT-2"}
    assert projection.mentioned_entity_ids == {"ENT-1"}
    assert projection.relationship_ids == {"REL-KEEP"}
    store.validate_semantic_projection(
        project="alpha",
        chunk_id="CHK-1",
        entities=entities,
        relationships=[refreshed],
        requirement_text_hash_by_relationship={"REL-KEEP": "REQ-HASH"},
    )
