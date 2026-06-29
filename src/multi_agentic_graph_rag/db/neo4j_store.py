"""Neo4j graph projection with a local JSON trace mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import DocumentManifest, RequirementArtifact


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
        if self.settings.neo4j.mode == "local_json":
            self._upsert_local(
                "artifact_projection",
                artifact.document_version_id,
                {"kind": "artifact_projection", "artifact": artifact.model_dump(mode="json")},
            )
            return

        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(_project_artifact_tx, artifact.model_dump(mode="json"))

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


def _project_manifest_tx(tx: Any, manifest: dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (p:Project {project: $project})
        MERGE (d:Document {document_id: $document_id})
        MERGE (p)-[:OWNS_DOCUMENT]->(d)
        MERGE (v:DocumentVersion {document_version_id: $document_version_id})
        SET v.version = $version, v.checksum = $checksum
        MERGE (d)-[:HAS_VERSION]->(v)
        """,
        project=manifest["project"],
        document_id=manifest["document_id"],
        document_version_id=manifest["document_version_id"],
        version=manifest["version"],
        checksum=manifest["source_checksum"],
    )
    for chunk in manifest["chunks"]:
        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.text = $text, c.ordinal = $ordinal, c.page = $page, c.section = $section
            MERGE (v)-[:HAS_CHUNK]->(c)
            """,
            document_version_id=manifest["document_version_id"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            ordinal=chunk["ordinal"],
            page=chunk["page"],
            section=chunk["section"],
        )


def _project_artifact_tx(tx: Any, artifact: dict[str, Any]) -> None:
    for fact in artifact["facts"]:
        tx.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (f:Fact {fact_id: $fact_id})
            SET f.text = $text
            MERGE (f)-[:SUPPORTED_BY {quote: $quote}]->(c)
            """,
            fact_id=fact["fact_id"],
            text=fact["text"],
            chunk_id=fact["source_trace"]["chunk_id"],
            quote=fact["source_trace"]["quote"],
        )
    for req in artifact["requirements"]:
        tx.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (r:Requirement {requirement_id: $requirement_id})
            SET r.statement = $statement,
                r.priority = $priority,
                r.requirement_type = $requirement_type
            MERGE (r)-[:TRACED_TO {quote: $quote}]->(c)
            """,
            requirement_id=req["requirement_id"],
            statement=req["statement"],
            priority=req["priority"],
            requirement_type=req["requirement_type"],
            chunk_id=req["source_trace"]["chunk_id"],
            quote=req["source_trace"]["quote"],
        )
        for fact_id in req["fact_ids"]:
            tx.run(
                """
                MATCH (r:Requirement {requirement_id: $requirement_id})
                MATCH (f:Fact {fact_id: $fact_id})
                MERGE (r)-[:DERIVED_FROM]->(f)
                """,
                requirement_id=req["requirement_id"],
                fact_id=fact_id,
            )
