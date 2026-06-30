"""Neo4j document/chunk graph projection with a local JSON trace mode."""

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
        """Compatibility no-op.

        Generated requirement artifacts and their ledger rows are intentionally
        stored in PostgreSQL, not projected into Neo4j.
        """
        _ = artifact

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
