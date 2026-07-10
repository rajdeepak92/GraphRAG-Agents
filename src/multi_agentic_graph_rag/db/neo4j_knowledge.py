"""Neo4j lexical source-knowledge projection.

This module intentionally imports no generated requirement, story, or scenario schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.domain.knowledge_models import LexicalKnowledgeProjection

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT text_unit_id_unique IF NOT EXISTS "
    "FOR (n:TextUnit) REQUIRE n.text_unit_id IS UNIQUE",
    "CREATE FULLTEXT INDEX text_unit_fulltext IF NOT EXISTS "
    "FOR (n:TextUnit) ON EACH [n.text, n.normalized_text]",
)


class Neo4jKnowledgeStore:
    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str,
        local_json_path: Path | None = None,
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.local_json_path = local_json_path

    def ensure_schema(self) -> None:
        if self.local_json_path is not None:
            return
        with self._driver() as driver, driver.session(database=self.database) as session:
            for statement in SCHEMA_STATEMENTS:
                session.run(statement).consume()

    def project_lexical_knowledge(self, projection: LexicalKnowledgeProjection) -> None:
        if self.local_json_path is not None:
            self._write_local(projection)
            return
        payload = projection.model_dump(mode="json")
        with self._driver() as driver, driver.session(database=self.database) as session:
            session.execute_write(_project_lexical_knowledge_tx, payload)

    def _driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

    def _write_local(self, projection: LexicalKnowledgeProjection) -> None:
        assert self.local_json_path is not None
        self.local_json_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if self.local_json_path.exists():
            rows = [
                json.loads(line)
                for line in self.local_json_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        payload = {
            "kind": "lexical_knowledge_projection",
            "_local_key": projection.document_version_id,
            "projection": projection.model_dump(mode="json"),
        }
        rows = [
            row
            for row in rows
            if not (
                row.get("kind") == payload["kind"]
                and row.get("_local_key") == payload["_local_key"]
            )
        ]
        rows.append(payload)
        self.local_json_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def _project_lexical_knowledge_tx(tx: Any, projection: dict[str, Any]) -> None:
    tx.run(
        """
        MATCH (v:DocumentVersion {document_version_id: $document_version_id})
        UNWIND $text_units AS unit
        MERGE (u:TextUnit {text_unit_id: unit.text_unit_id})
        SET u.document_version_id = $document_version_id,
            u.block_id = unit.block_id,
            u.ordinal = unit.ordinal,
            u.unit_type = unit.unit_type,
            u.text = unit.text,
            u.normalized_text = unit.normalized_text,
            u.page = unit.page,
            u.section = unit.section,
            u.start_char = unit.start_char,
            u.end_char = unit.end_char
        MERGE (v)-[:HAS_TEXT_UNIT]->(u)
        """,
        document_version_id=projection["document_version_id"],
        text_units=projection["text_units"],
    )
    tx.run(
        """
        UNWIND $chunk_links AS link
        MATCH (c:Chunk {chunk_id: link.chunk_id})
        MATCH (u:TextUnit {text_unit_id: link.text_unit_id})
        MERGE (c)-[r:CONTAINS_TEXT_UNIT]->(u)
        SET r.ordinal_in_chunk = link.ordinal_in_chunk
        """,
        chunk_links=projection["chunk_links"],
    )
    tx.run(
        """
        MATCH (v:DocumentVersion {document_version_id: $document_version_id})-[:HAS_TEXT_UNIT]->(u)
        WITH u ORDER BY u.ordinal
        WITH collect(u) AS units
        UNWIND range(0, size(units) - 2) AS index
        WITH units[index] AS current, units[index + 1] AS following
        MERGE (current)-[:NEXT_TEXT_UNIT]->(following)
        """,
        document_version_id=projection["document_version_id"],
    )
