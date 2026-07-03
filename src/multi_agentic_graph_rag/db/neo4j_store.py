"""Neo4j document/chunk graph projection with a local JSON trace mode."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import (
    DocumentManifest,
    RequirementArtifact,
    TestScenarioArtifact,
    UserStoryArtifact,
)

_LUCENE_TOKEN = re.compile(r"[A-Za-z0-9]+")


class Neo4jStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def check(self) -> str:
        if self.settings.neo4j.mode == "local_json":
            self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS neo4j local_json path={self.settings.neo4j.local_path}"
        with self._driver() as driver:
            driver.verify_connectivity()
        return "PASS neo4j connectivity"

    def project_manifest(self, manifest: DocumentManifest) -> None:
        if self.settings.neo4j.mode == "local_json":
            self._upsert_local(
                "manifest_projection",
                manifest.document_version_id,
                {"kind": "manifest_projection", "manifest": manifest.model_dump(mode="json")},
            )
            return

        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(_project_manifest_tx, manifest.model_dump(mode="json"))

    def project_artifact(self, artifact: RequirementArtifact) -> None:
        """Compatibility no-op.

        Generated requirement artifacts and their ledger rows are intentionally
        stored in PostgreSQL, not projected into Neo4j.
        """
        _ = artifact

    def ensure_search_index(self) -> None:
        """Create the chunk full-text (BM25/Lucene) index used by hybrid retrieval."""
        if self.settings.neo4j.mode == "local_json":
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.run(
                "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text, c.normalized_text]"
            )

    def fulltext_search_chunks(
        self,
        query: str,
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """BM25/Lucene keyword search over chunks, scoped to one document version."""
        if limit <= 0:
            return []
        lucene_query = _lucene_keyword_query(query)
        if not lucene_query:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_fulltext_search(query, document_version_id, limit)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
                YIELD node, score
                WHERE node.document_version_id = $document_version_id
                RETURN node.chunk_id AS chunk_id, node.text AS text, score AS score
                LIMIT $limit
                """,
                query=lucene_query,
                document_version_id=document_version_id,
                limit=limit,
            )
            return [
                (str(record["chunk_id"]), str(record["text"] or ""), float(record["score"]))
                for record in records
            ]

    def neighbor_chunks(
        self,
        chunk_ids: list[str],
        document_version_id: str,
        window: int,
    ) -> list[tuple[str, str]]:
        """Multi-hop neighbours of the seed chunks within the same document version."""
        if not chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_neighbor_chunks(chunk_ids, document_version_id, window)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $document_version_id})
                      -[:HAS_CHUNK]->(seed:Chunk)
                WHERE seed.chunk_id IN $chunk_ids
                MATCH (v)-[:HAS_CHUNK]->(c:Chunk)
                WHERE abs(c.ordinal - seed.ordinal) <= $window
                   OR (seed.section IS NOT NULL AND c.section = seed.section)
                RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, c.ordinal AS ordinal
                ORDER BY ordinal
                """,
                document_version_id=document_version_id,
                chunk_ids=chunk_ids,
                window=window,
            )
            return [(str(record["chunk_id"]), str(record["text"] or "")) for record in records]

    def fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Fetch (chunk_id, text) for the requested chunks, preserving input order."""
        if not chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_fetch_chunks(chunk_ids)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS chunk_id, c.text AS text
                """,
                chunk_ids=chunk_ids,
            )
            by_id = {str(record["chunk_id"]): str(record["text"] or "") for record in records}
        return [(chunk_id, by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in by_id]

    def project_user_story_coverage(
        self,
        artifact: UserStoryArtifact,
        evidence_chunk_ids: Mapping[str, list[str]],
    ) -> None:
        """Project validated user-story coverage claim-nodes back into the graph.

        Writes UserStory -> Requirement -> Chunk traceability so a later stage can
        retrieve graph context, and marks each covered requirement.
        """
        evidence = {key: list(value) for key, value in evidence_chunk_ids.items()}
        if self.settings.neo4j.mode == "local_json":
            for story_id, record in artifact.stories.items():
                self._upsert_local(
                    "user_story_projection",
                    story_id,
                    {
                        "kind": "user_story_projection",
                        "story_id": story_id,
                        "requirement_id": record.requirement_id,
                        "project": record.project,
                        "document_version_id": record.document_version_id,
                        "title": record.title,
                        "covered": True,
                        "evidence_chunk_ids": evidence.get(record.requirement_id, []),
                    },
                )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_user_story_coverage_tx,
                artifact.model_dump(mode="json"),
                evidence,
            )

    def project_test_scenario_coverage(
        self,
        artifact: TestScenarioArtifact,
        evidence_chunk_ids: Mapping[str, list[str]],
    ) -> None:
        """Project validated test-scenario claim-nodes back into the graph."""
        evidence = {key: list(value) for key, value in evidence_chunk_ids.items()}
        if self.settings.neo4j.mode == "local_json":
            for scenario_id, record in artifact.scenarios.items():
                self._upsert_local(
                    "test_scenario_projection",
                    scenario_id,
                    {
                        "kind": "test_scenario_projection",
                        "scenario_id": scenario_id,
                        "story_id": record.story_id,
                        "requirement_id": record.requirement_id,
                        "project": record.project,
                        "document_version_id": record.document_version_id,
                        "title": record.title,
                        "scenario_type": record.scenario_type,
                        "priority": record.priority,
                        "confidence": record.confidence,
                        "evidence_chunk_ids": evidence.get(record.requirement_id, []),
                    },
                )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_test_scenario_coverage_tx,
                artifact.model_dump(mode="json"),
                evidence,
            )

    def _driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            self.settings.neo4j.uri,
            auth=(self.settings.neo4j.username, self.settings.neo4j.password),
        )

    def _append_local(self, payload: dict[str, Any]) -> None:
        self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
        payload["written_at"] = datetime.now(UTC).isoformat()
        with self.settings.neo4j.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if self.settings.neo4j.local_path.exists():
            rows = [
                json.loads(line)
                for line in self.settings.neo4j.local_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        payload["written_at"] = datetime.now(UTC).isoformat()
        payload["_local_key"] = key
        replaced = False
        for index, row in enumerate(rows):
            if row.get("kind") == kind and row.get("_local_key") == key:
                rows[index] = payload
                replaced = True
                break
        if not replaced:
            rows.append(payload)
        self.settings.neo4j.local_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_local_rows(self) -> list[dict[str, Any]]:
        if not self.settings.neo4j.local_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.settings.neo4j.local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _local_manifest_chunks(self, document_version_id: str) -> list[dict[str, Any]]:
        for row in reversed(self._read_local_rows()):
            if (
                row.get("kind") == "manifest_projection"
                and row.get("_local_key") == document_version_id
            ):
                manifest = row.get("manifest")
                if isinstance(manifest, dict):
                    chunks = manifest.get("chunks")
                    if isinstance(chunks, list):
                        return [chunk for chunk in chunks if isinstance(chunk, dict)]
        return []

    def _local_fulltext_search(
        self,
        query: str,
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        terms = {token.lower() for token in _LUCENE_TOKEN.findall(query)}
        if not terms:
            return []
        scored: list[tuple[str, str, float]] = []
        for chunk in self._local_manifest_chunks(document_version_id):
            text = str(chunk.get("text", ""))
            haystack = text.lower()
            score = float(sum(1 for term in terms if term in haystack))
            if score > 0:
                scored.append((str(chunk.get("chunk_id", "")), text, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:limit]

    def _local_neighbor_chunks(
        self,
        chunk_ids: list[str],
        document_version_id: str,
        window: int,
    ) -> list[tuple[str, str]]:
        chunks = self._local_manifest_chunks(document_version_id)
        if not chunks:
            return []
        by_id = {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}
        seeds = [by_id[cid] for cid in chunk_ids if cid in by_id]
        seed_ordinals = {int(seed.get("ordinal", 0)) for seed in seeds}
        seed_sections = {
            str(seed.get("section")) for seed in seeds if seed.get("section") is not None
        }
        selected: list[tuple[str, str]] = []
        for chunk in chunks:
            ordinal = int(chunk.get("ordinal", 0))
            section = chunk.get("section")
            near = any(abs(ordinal - seed) <= window for seed in seed_ordinals)
            same_section = section is not None and str(section) in seed_sections
            if near or same_section:
                selected.append((str(chunk.get("chunk_id", "")), str(chunk.get("text", ""))))
        return selected

    def _local_fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        by_id: dict[str, str] = {}
        for row in self._read_local_rows():
            if row.get("kind") != "manifest_projection":
                continue
            manifest = row.get("manifest")
            if not isinstance(manifest, dict):
                continue
            for chunk in manifest.get("chunks", []):
                if isinstance(chunk, dict):
                    by_id[str(chunk.get("chunk_id", ""))] = str(chunk.get("text", ""))
        return [(chunk_id, by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in by_id]


def _lucene_keyword_query(query: str) -> str:
    """Sanitize free text into a Lucene OR keyword query, dropping special chars."""
    tokens = _LUCENE_TOKEN.findall(query)
    return " OR ".join(tokens)


def _project_user_story_coverage_tx(
    tx: Any,
    artifact: dict[str, Any],
    evidence_chunk_ids: Mapping[str, list[str]],
) -> None:
    for story_id, record in artifact["stories"].items():
        requirement_id = record["requirement_id"]
        tx.run(
            """
            MERGE (r:Requirement {requirement_id: $requirement_id})
            SET r.covered = true,
                r.project = $project,
                r.document_id = $document_id,
                r.document_version_id = $document_version_id
            MERGE (s:UserStory {story_id: $story_id})
            SET s.title = $title,
                s.requirement_id = $requirement_id,
                s.project = $project,
                s.epic = $epic,
                s.persona = $persona,
                s.priority = $priority,
                s.document_version_id = $document_version_id
            MERGE (s)-[:COVERS_REQUIREMENT]->(r)
            """,
            story_id=story_id,
            requirement_id=requirement_id,
            project=record["project"],
            document_id=record["document_id"],
            document_version_id=record["document_version_id"],
            title=record["title"],
            epic=record["epic"],
            persona=record["persona"],
            priority=record["priority"],
        )
        tx.run(
            """
            MATCH (s:UserStory {story_id: $story_id})
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (v)-[:HAS_USER_STORY]->(s)
            """,
            story_id=story_id,
            document_version_id=record["document_version_id"],
        )
        for chunk_id in evidence_chunk_ids.get(requirement_id, []):
            tx.run(
                """
                MATCH (r:Requirement {requirement_id: $requirement_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (r)-[:EVIDENCED_BY_CHUNK]->(c)
                """,
                requirement_id=requirement_id,
                chunk_id=chunk_id,
            )


def _project_test_scenario_coverage_tx(
    tx: Any,
    artifact: dict[str, Any],
    evidence_chunk_ids: Mapping[str, list[str]],
) -> None:
    for scenario_id, record in artifact["scenarios"].items():
        requirement_id = record["requirement_id"]
        tx.run(
            """
            MERGE (t:TestScenario {scenario_id: $scenario_id})
            SET t.title = $title,
                t.scenario_type = $scenario_type,
                t.priority = $priority,
                t.confidence = $confidence,
                t.story_id = $story_id,
                t.requirement_id = $requirement_id,
                t.project = $project,
                t.document_version_id = $document_version_id
            MERGE (s:UserStory {story_id: $story_id})
            MERGE (t)-[:VALIDATES_STORY]->(s)
            MERGE (r:Requirement {requirement_id: $requirement_id})
            MERGE (t)-[:COVERS_REQUIREMENT]->(r)
            """,
            scenario_id=scenario_id,
            story_id=record["story_id"],
            requirement_id=requirement_id,
            project=record["project"],
            document_version_id=record["document_version_id"],
            title=record["title"],
            scenario_type=record["scenario_type"],
            priority=record["priority"],
            confidence=record["confidence"],
        )
        tx.run(
            """
            MATCH (t:TestScenario {scenario_id: $scenario_id})
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (v)-[:HAS_TEST_SCENARIO]->(t)
            """,
            scenario_id=scenario_id,
            document_version_id=record["document_version_id"],
        )
        for chunk_id in evidence_chunk_ids.get(requirement_id, []):
            tx.run(
                """
                MATCH (r:Requirement {requirement_id: $requirement_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (r)-[:EVIDENCED_BY_CHUNK]->(c)
                """,
                requirement_id=requirement_id,
                chunk_id=chunk_id,
            )


def _project_manifest_tx(tx: Any, manifest: dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (p:Project {project: $project})
        MERGE (d:Document {document_id: $document_id})
        SET d.project = $project,
            d.logical_name = $logical_name,
            d.source_path = $source_path
        MERGE (p)-[:OWNS_DOCUMENT]->(d)
        MERGE (v:DocumentVersion {document_version_id: $document_version_id})
        SET v.project = $project,
            v.document_id = $document_id,
            v.version = $version,
            v.checksum = $checksum,
            v.source_checksum = $checksum,
            v.source_path = $source_path,
            v.parser_fingerprint = $parser_fingerprint,
            v.chunker_fingerprint = $chunker_fingerprint,
            v.created_at = $created_at
        MERGE (d)-[:HAS_VERSION]->(v)
        """,
        project=manifest["project"],
        document_id=manifest["document_id"],
        document_version_id=manifest["document_version_id"],
        logical_name=manifest["logical_name"],
        version=manifest["version"],
        checksum=manifest["source_checksum"],
        source_path=manifest["source_path"],
        parser_fingerprint=manifest["parser_fingerprint"],
        chunker_fingerprint=manifest["chunker_fingerprint"],
        created_at=manifest["created_at"],
    )
    for chunk in manifest["chunks"]:
        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.project = $project,
                c.document_id = $document_id,
                c.document_version_id = $document_version_id,
                c.version = $version,
                c.source_checksum = $source_checksum,
                c.source_path = $source_path,
                c.chunk_id = $chunk_id,
                c.ordinal = $ordinal,
                c.text = $text,
                c.normalized_text = $normalized_text,
                c.page = $page,
                c.section = $section,
                c.start_char = $start_char,
                c.end_char = $end_char,
                c.source_block_ids = $source_block_ids
            MERGE (v)-[:HAS_CHUNK]->(c)
            """,
            project=manifest["project"],
            document_id=manifest["document_id"],
            document_version_id=manifest["document_version_id"],
            version=manifest["version"],
            source_checksum=manifest["source_checksum"],
            source_path=manifest["source_path"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            normalized_text=chunk["normalized_text"],
            ordinal=chunk["ordinal"],
            page=chunk.get("page"),
            section=chunk.get("section"),
            start_char=chunk["start_char"],
            end_char=chunk["end_char"],
            source_block_ids=chunk["source_block_ids"],
        )
