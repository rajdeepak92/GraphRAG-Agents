"""Project-scoped Neo4j storage for chunks and validated semantic relationships."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalEntity,
    CanonicalRelationship,
    ManifestChunk,
    RelationshipType,
)

_RELATIONSHIP_TYPES: set[str] = set(get_args(RelationshipType))


@dataclass(frozen=True)
class SemanticProjectionReadback:
    """Existing deterministic semantic records for resume-safe projection."""

    entity_ids: set[str]
    mentioned_entity_ids: set[str]
    relationship_ids: set[str]


class Neo4jStore:
    """Keep the graph boundary limited to Chunk, Entity, MENTIONS, and semantic edges."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def check(self) -> str:
        """Validate connectivity."""
        if self.settings.neo4j.mode == "local_json":
            self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS neo4j local_json path={self.settings.neo4j.local_path}"
        with self._driver() as driver:
            driver.verify_connectivity()
        return "PASS neo4j connectivity"

    def ensure_schema(self) -> None:
        """Create the project-scoped uniqueness and retrieval indexes."""
        if self.settings.neo4j.mode == "local_json":
            return
        statements = (
            "CREATE CONSTRAINT chunk_project_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE (c.project, c.chunk_id) IS UNIQUE",
            "CREATE CONSTRAINT entity_project_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.project, e.entity_id) IS UNIQUE",
            "CREATE INDEX chunk_project_sequence IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.project, c.sequence_index)",
            "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
        )
        with self._session() as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert_chunk(self, *, project: str, run_id: str, chunk: ManifestChunk) -> None:
        """Idempotently persist one Stage 1.1 chunk."""
        payload = {
            "project": project,
            "run_id": run_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.chunk_text,
            "content_hash": chunk.content_hash,
            "sequence_index": chunk.sequence_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "layout": chunk.layout.model_dump(mode="json"),
            "source_provenance": chunk.source_provenance,
        }
        if self.settings.neo4j.mode == "local_json":
            self._upsert_local("chunk", f"{project}:{chunk.chunk_id}", payload)
            return
        with self._session() as session:
            session.run(
                """
                MERGE (c:Chunk {project: $project, chunk_id: $chunk_id})
                SET c.text = $text,
                    c.content_hash = $content_hash,
                    c.sequence_index = $sequence_index,
                    c.run_id = $run_id,
                    c.start_char = $start_char,
                    c.end_char = $end_char,
                    c.layout_json = $layout_json,
                    c.source_provenance_json = $source_provenance_json
                """,
                **payload,
                layout_json=json.dumps(payload["layout"], sort_keys=True),
                source_provenance_json=json.dumps(payload["source_provenance"], sort_keys=True),
            ).consume()

    def read_chunk(self, project: str, chunk_id: str) -> dict[str, Any] | None:
        """Read a chunk back for persistence validation."""
        if self.settings.neo4j.mode == "local_json":
            return self._read_local("chunk", f"{project}:{chunk_id}")
        with self._session() as session:
            record = session.run(
                """
                MATCH (c:Chunk {project: $project, chunk_id: $chunk_id})
                RETURN c.project AS project, c.chunk_id AS chunk_id, c.text AS text,
                       c.content_hash AS content_hash, c.sequence_index AS sequence_index,
                       c.run_id AS run_id
                """,
                project=project,
                chunk_id=chunk_id,
            ).single()
        return dict(record) if record is not None else None

    def upsert_semantic_projection(
        self,
        *,
        project: str,
        chunk_id: str,
        entities: list[CanonicalEntity],
        mentioned_entity_ids: set[str],
        relationships: list[CanonicalRelationship],
        requirement_text_hash_by_relationship: dict[str, str],
    ) -> None:
        """Idempotently project validated Stage 1.2 semantics."""
        if self.settings.neo4j.mode == "local_json":
            for entity in entities:
                self._upsert_local(
                    "entity",
                    f"{project}:{entity.entity_id}",
                    {"project": project, **entity.model_dump(mode="json")},
                )
            for entity_id in mentioned_entity_ids:
                self._upsert_local(
                    "mention",
                    f"{project}:{chunk_id}:{entity_id}",
                    {"project": project, "chunk_id": chunk_id, "entity_id": entity_id},
                )
            for relationship in relationships:
                self._upsert_local(
                    "relationship",
                    f"{project}:{relationship.relationship_id}",
                    {
                        "project": project,
                        "requirement_text_hash": requirement_text_hash_by_relationship[
                            relationship.relationship_id
                        ],
                        **relationship.model_dump(mode="json"),
                    },
                )
            return
        with self._session() as session:
            for entity in entities:
                session.run(
                    """
                    MERGE (e:Entity {project: $project, entity_id: $entity_id})
                    SET e.name = $name, e.normalized_name = $normalized_name,
                        e.entity_type = $entity_type, e.aliases = $aliases
                    """,
                    project=project,
                    **entity.model_dump(mode="json", exclude={"mentions"}),
                ).consume()
            for entity_id in mentioned_entity_ids:
                session.run(
                    """
                    MATCH (c:Chunk {project: $project, chunk_id: $chunk_id})
                    MATCH (e:Entity {project: $project, entity_id: $entity_id})
                    MERGE (c)-[m:MENTIONS {project: $project}]->(e)
                    SET m.chunk_id = $chunk_id
                    """,
                    project=project,
                    chunk_id=chunk_id,
                    entity_id=entity_id,
                ).consume()
            for relationship in relationships:
                relationship_type = relationship.relationship_type
                if relationship_type not in _RELATIONSHIP_TYPES:
                    raise ValueError(f"unsupported relationship type: {relationship_type}")
                evidence = relationship.evidence[0]
                session.run(
                    f"""
                    MATCH (s:Entity {{project: $project, entity_id: $source_entity_id}})
                    MATCH (t:Entity {{project: $project, entity_id: $target_entity_id}})
                    MERGE (s)-[r:{relationship_type} {{
                        project: $project, relationship_id: $relationship_id
                    }}]->(t)
                    SET r.chunk_id = $chunk_id,
                        r.requirement_text_hash = $requirement_text_hash,
                        r.evidence_quote = $evidence_quote,
                        r.evidence_start_char = $evidence_start_char,
                        r.evidence_end_char = $evidence_end_char,
                        r.confidence = $confidence,
                        r.extraction_method = 'llm'
                    """,
                    project=project,
                    relationship_id=relationship.relationship_id,
                    chunk_id=relationship.chunk_id,
                    source_entity_id=relationship.source_entity_id,
                    target_entity_id=relationship.target_entity_id,
                    requirement_text_hash=requirement_text_hash_by_relationship[
                        relationship.relationship_id
                    ],
                    evidence_quote=evidence.quote,
                    evidence_start_char=evidence.start_char,
                    evidence_end_char=evidence.end_char,
                    confidence=relationship.confidence,
                ).consume()

    def validate_relationships(self, project: str, relationship_ids: set[str]) -> set[str]:
        """Return relationship IDs that exist in the requested project."""
        if not relationship_ids:
            return set()
        if self.settings.neo4j.mode == "local_json":
            return {
                row["relationship_id"]
                for row in self._local_rows()
                if row.get("kind") == "relationship"
                and row.get("project") == project
                and row.get("relationship_id") in relationship_ids
            }
        with self._session() as session:
            records = session.run(
                """
                MATCH ()-[r]->()
                WHERE type(r) IN $relationship_types
                  AND r.project = $project
                  AND r.relationship_id IN $relationship_ids
                RETURN r.relationship_id AS relationship_id
                """,
                project=project,
                relationship_ids=list(relationship_ids),
                relationship_types=sorted(_RELATIONSHIP_TYPES),
            )
            return {str(record["relationship_id"]) for record in records}

    def read_semantic_projection(
        self,
        *,
        project: str,
        chunk_id: str,
        entity_ids: set[str],
        relationship_ids: set[str],
    ) -> SemanticProjectionReadback:
        """Read deterministic IDs without scanning MENTIONS as semantic relationships."""
        if self.settings.neo4j.mode == "local_json":
            rows = self._local_rows()
            return SemanticProjectionReadback(
                entity_ids={
                    str(row["entity_id"])
                    for row in rows
                    if row.get("kind") == "entity"
                    and row.get("project") == project
                    and row.get("entity_id") in entity_ids
                },
                mentioned_entity_ids={
                    str(row["entity_id"])
                    for row in rows
                    if row.get("kind") == "mention"
                    and row.get("project") == project
                    and row.get("chunk_id") == chunk_id
                    and row.get("entity_id") in entity_ids
                },
                relationship_ids={
                    str(row["relationship_id"])
                    for row in rows
                    if row.get("kind") == "relationship"
                    and row.get("project") == project
                    and row.get("relationship_type") in _RELATIONSHIP_TYPES
                    and row.get("relationship_id") in relationship_ids
                },
            )
        with self._session() as session:
            entity_records = session.run(
                """
                MATCH (e:Entity)
                WHERE e.project = $project AND e.entity_id IN $entity_ids
                OPTIONAL MATCH (c:Chunk {project: $project, chunk_id: $chunk_id})
                  -[m:MENTIONS {project: $project}]->(e)
                RETURN e.entity_id AS entity_id, m IS NOT NULL AS mentioned
                """,
                project=project,
                chunk_id=chunk_id,
                entity_ids=list(entity_ids),
            )
            found_entities: set[str] = set()
            mentioned_entities: set[str] = set()
            for record in entity_records:
                entity_id = str(record["entity_id"])
                found_entities.add(entity_id)
                if bool(record["mentioned"]):
                    mentioned_entities.add(entity_id)
            relationship_records = session.run(
                """
                MATCH ()-[r]->()
                WHERE type(r) IN $relationship_types
                  AND r.project = $project
                  AND r.relationship_id IN $relationship_ids
                RETURN r.relationship_id AS relationship_id
                """,
                project=project,
                relationship_ids=list(relationship_ids),
                relationship_types=sorted(_RELATIONSHIP_TYPES),
            )
            return SemanticProjectionReadback(
                entity_ids=found_entities,
                mentioned_entity_ids=mentioned_entities,
                relationship_ids={
                    str(record["relationship_id"]) for record in relationship_records
                },
            )

    def validate_semantic_projection(
        self,
        *,
        project: str,
        chunk_id: str,
        entities: list[CanonicalEntity],
        relationships: list[CanonicalRelationship],
        requirement_text_hash_by_relationship: dict[str, str],
    ) -> None:
        """Validate IDs, types, endpoints, provenance, evidence, confidence, and method."""
        if self.settings.neo4j.mode == "local_json":
            rows = self._local_rows()
            by_id = {
                str(row["relationship_id"]): row
                for row in rows
                if row.get("kind") == "relationship" and row.get("project") == project
            }
            for relationship in relationships:
                row = by_id.get(relationship.relationship_id)
                evidence = relationship.evidence[0]
                if row is None or any(
                    (
                        row.get("relationship_type") != relationship.relationship_type,
                        row.get("source_entity_id") != relationship.source_entity_id,
                        row.get("target_entity_id") != relationship.target_entity_id,
                        row.get("chunk_id") != chunk_id,
                        row.get("requirement_text_hash")
                        != requirement_text_hash_by_relationship[relationship.relationship_id],
                        (row.get("evidence") or [{}])[0].get("quote") != evidence.quote,
                        row.get("confidence") != relationship.confidence,
                        row.get("extraction_method") != "llm",
                    )
                ):
                    raise ValueError("local semantic relationship read-back mismatch")
            return
        for relationship in relationships:
            evidence = relationship.evidence[0]
            with self._session() as session:
                record = session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE type(r) IN $relationship_types
                      AND r.project = $project
                      AND r.relationship_id = $relationship_id
                    RETURN type(r) AS relationship_type,
                           s.entity_id AS source_entity_id,
                           t.entity_id AS target_entity_id,
                           r.project AS project,
                           r.chunk_id AS chunk_id,
                           r.requirement_text_hash AS requirement_text_hash,
                           r.evidence_quote AS evidence_quote,
                           r.evidence_start_char AS evidence_start_char,
                           r.evidence_end_char AS evidence_end_char,
                           r.confidence AS confidence,
                           r.extraction_method AS extraction_method
                    """,
                    project=project,
                    relationship_id=relationship.relationship_id,
                    relationship_types=sorted(_RELATIONSHIP_TYPES),
                ).single()
            expected = {
                "relationship_type": relationship.relationship_type,
                "source_entity_id": relationship.source_entity_id,
                "target_entity_id": relationship.target_entity_id,
                "project": project,
                "chunk_id": chunk_id,
                "requirement_text_hash": requirement_text_hash_by_relationship[
                    relationship.relationship_id
                ],
                "evidence_quote": evidence.quote,
                "evidence_start_char": evidence.start_char,
                "evidence_end_char": evidence.end_char,
                "confidence": relationship.confidence,
                "extraction_method": "llm",
            }
            if record is None or dict(record) != expected:
                raise ValueError("Neo4j semantic relationship read-back mismatch")

    def fetch_chunks(self, project: str, chunk_ids: set[str]) -> list[tuple[str, str]]:
        """Hydrate exact evidence directly from Chunk.text."""
        if not chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            rows = [
                row
                for row in self._local_rows()
                if row.get("kind") == "chunk"
                and row.get("project") == project
                and row.get("chunk_id") in chunk_ids
            ]
            return [(str(row["chunk_id"]), str(row["text"])) for row in rows]
        with self._session() as session:
            records = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.project = $project AND c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS chunk_id, c.text AS text
                ORDER BY c.sequence_index
                """,
                project=project,
                chunk_ids=list(chunk_ids),
            )
            return [(str(row["chunk_id"]), str(row["text"])) for row in records]

    def retrieve_semantic_chunks(
        self,
        *,
        project: str,
        anchor_entity_ids: set[str],
        anchor_relationship_ids: set[str],
        allowed_chunk_ids: set[str],
        max_hops: int,
        limit: int,
    ) -> list[tuple[str, str, float, list[str], list[str]]]:
        """Traverse anchored entities/relationships and enforce manifest scope."""
        if not allowed_chunk_ids or (not anchor_entity_ids and not anchor_relationship_ids):
            return []
        if self.settings.neo4j.mode == "local_json":
            relationships = [
                row
                for row in self._local_rows()
                if row.get("kind") == "relationship"
                and row.get("project") == project
                and row.get("chunk_id") in allowed_chunk_ids
                and (
                    row.get("relationship_id") in anchor_relationship_ids
                    or row.get("source_entity_id") in anchor_entity_ids
                    or row.get("target_entity_id") in anchor_entity_ids
                )
            ]
            chunks = dict(self.fetch_chunks(project, allowed_chunk_ids))
            return [
                (
                    str(row["chunk_id"]),
                    chunks.get(str(row["chunk_id"]), ""),
                    1.0,
                    [str(row["source_entity_id"]), str(row["target_entity_id"])],
                    [str(row["relationship_id"])],
                )
                for row in relationships[:limit]
            ]
        hop_limit = max(1, min(max_hops, 4))
        with self._session() as session:
            records = session.run(
                f"""
                MATCH (seed:Entity)
                WHERE seed.project = $project
                  AND (
                    seed.entity_id IN $anchor_entity_ids OR
                    EXISTS {{
                      MATCH (seed)-[ar]-()
                      WHERE ar.project = $project
                        AND ar.relationship_id IN $anchor_relationship_ids
                    }}
                  )
                MATCH path=(seed)-[rels*1..{hop_limit}]-(other:Entity)
                WHERE all(r IN rels WHERE r.project = $project)
                  AND all(r IN rels WHERE type(r) IN $relationship_types)
                  AND any(r IN rels WHERE r.chunk_id IN $allowed_chunk_ids)
                UNWIND rels AS r
                WITH DISTINCT r
                WHERE r.chunk_id IN $allowed_chunk_ids
                MATCH (c:Chunk {{project: $project, chunk_id: r.chunk_id}})
                RETURN c.chunk_id AS chunk_id, c.text AS text,
                       max(1.0 / $max_hops) AS score,
                       collect(DISTINCT startNode(r).entity_id) +
                         collect(DISTINCT endNode(r).entity_id) AS entity_ids,
                       collect(DISTINCT r.relationship_id) AS relationship_ids
                LIMIT $limit
                """,
                project=project,
                anchor_entity_ids=list(anchor_entity_ids),
                anchor_relationship_ids=list(anchor_relationship_ids),
                allowed_chunk_ids=list(allowed_chunk_ids),
                relationship_types=sorted(_RELATIONSHIP_TYPES),
                max_hops=max_hops,
                limit=limit,
            )
            return [
                (
                    str(row["chunk_id"]),
                    str(row["text"]),
                    float(row["score"]),
                    [str(value) for value in row["entity_ids"] if value],
                    [str(value) for value in row["relationship_ids"] if value],
                )
                for row in records
            ]

    def delete_project(self, project: str) -> int:
        """Delete all project-scoped nodes/relationships. Returns nodes removed."""
        if self.settings.neo4j.mode == "local_json":
            rows = [row for row in self._local_rows() if row.get("project") != project]
            removed = len(self._local_rows()) - len(rows)
            _write_rows(self.settings.neo4j.local_path, rows)
            return removed
        with self._session() as session:
            summary = session.run(
                "MATCH (n) WHERE n.project = $project DETACH DELETE n",
                project=project,
            ).consume()
            return int(summary.counters.nodes_deleted)

    def _driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            self.settings.neo4j.uri,
            auth=(self.settings.neo4j.username, self.settings.neo4j.password),
        )

    @contextmanager
    def _session(self) -> Iterator[Any]:
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            yield session

    def _local_rows(self) -> list[dict[str, Any]]:
        path = self.settings.neo4j.local_path
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        rows = self._local_rows()
        row = {"kind": kind, "key": key, **payload}
        updated = False
        for index, existing in enumerate(rows):
            if existing.get("kind") == kind and existing.get("key") == key:
                rows[index] = row
                updated = True
                break
        if not updated:
            rows.append(row)
        _write_rows(self.settings.neo4j.local_path, rows)

    def _read_local(self, kind: str, key: str) -> dict[str, Any] | None:
        for row in self._local_rows():
            if row.get("kind") == kind and row.get("key") == key:
                return row
        return None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["Neo4jStore", "SemanticProjectionReadback"]
