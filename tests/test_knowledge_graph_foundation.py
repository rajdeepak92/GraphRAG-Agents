from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from multi_agentic_graph_rag.config.knowledge_graph import KnowledgeGraphFlags
from multi_agentic_graph_rag.db.neo4j_knowledge import (
    _project_lexical_knowledge_tx,
)
from multi_agentic_graph_rag.services.text_units import build_lexical_projection, segment_text_units


@dataclass(frozen=True)
class Block:
    block_id: str
    original_text: str
    start_char: int
    end_char: int
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    start_char: int
    end_char: int


class FakeTx:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **params: object) -> None:
        self.runs.append((query, params))


class KnowledgeGraphFoundationTests(unittest.TestCase):
    def test_flags_default_off(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            flags = KnowledgeGraphFlags.from_env()
        self.assertFalse(flags.enabled)
        self.assertFalse(flags.shadow_mode)
        self.assertFalse(flags.graph_primary_user_stories)
        self.assertFalse(flags.graph_primary_test_scenarios)

    def test_sentence_segmentation_is_deterministic(self) -> None:
        blocks = [Block("B1", "The system imports files. It validates every row.", 10, 58)]
        first = segment_text_units(document_version_id="DV-1", blocks=blocks)
        second = segment_text_units(document_version_id="DV-1", blocks=blocks)
        self.assertEqual(first, second)
        self.assertEqual(
            [item.text for item in first],
            ["The system imports files.", "It validates every row."],
        )

    def test_bullets_are_atomic_units(self) -> None:
        blocks = [Block("B1", "- Import CSV files\n- Reject malformed rows", 0, 41)]
        units = segment_text_units(document_version_id="DV-1", blocks=blocks)
        self.assertEqual([item.unit_type for item in units], ["bullet", "bullet"])
        self.assertEqual([item.text for item in units], ["Import CSV files", "Reject malformed rows"])

    def test_overlapping_chunks_reference_same_text_unit(self) -> None:
        blocks = [Block("B1", "The system imports files. It validates every row.", 0, 48)]
        chunks = [Chunk("C1", 0, 30), Chunk("C2", 20, 48)]
        projection = build_lexical_projection(
            project="P",
            document_id="D",
            document_version_id="DV-1",
            blocks=blocks,
            chunks=chunks,
        )
        linked_ids = [link.text_unit_id for link in projection.chunk_links]
        self.assertGreater(linked_ids.count(projection.text_units[0].text_unit_id), 1)

    def test_projection_cypher_is_source_only(self) -> None:
        projection = build_lexical_projection(
            project="P",
            document_id="D",
            document_version_id="DV-1",
            blocks=[Block("B1", "The system imports files.", 0, 25)],
            chunks=[Chunk("C1", 0, 25)],
        )
        tx = FakeTx()
        _project_lexical_knowledge_tx(tx, projection.model_dump(mode="json"))
        combined = "\n".join(query for query, _ in tx.runs)
        self.assertIn("TextUnit", combined)
        self.assertIn("CONTAINS_TEXT_UNIT", combined)
        self.assertIn("NEXT_TEXT_UNIT", combined)
        for forbidden in ("RequirementArtifact", "CanonicalFact", "UserStory", "TestScenario"):
            self.assertNotIn(forbidden, combined)

    def test_local_projection_can_be_serialized_idempotently(self) -> None:
        from multi_agentic_graph_rag.db.neo4j_knowledge import Neo4jKnowledgeStore

        projection = build_lexical_projection(
            project="P",
            document_id="D",
            document_version_id="DV-1",
            blocks=[Block("B1", "The system imports files.", 0, 25)],
            chunks=[Chunk("C1", 0, 25)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kg.jsonl"
            store = Neo4jKnowledgeStore(
                uri="bolt://unused",
                username="neo4j",
                password="",
                database="neo4j",
                local_json_path=path,
            )
            store.project_lexical_knowledge(projection)
            store.project_lexical_knowledge(projection)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_local_key"], "DV-1")


if __name__ == "__main__":
    unittest.main()
