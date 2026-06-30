from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    RequirementArtifact,
    RequirementEvidence,
    SourceTrace,
    VerifiedFact,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.services.artifacts import write_requirement_artifact


class PostgresArtifactTests(unittest.TestCase):
    def test_persisted_artifact_payload_matches_local_requirements_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _artifact()
            artifact_path = write_requirement_artifact(artifact, root / "run")
            local_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

            store = PostgresStore(_settings(root))
            store.persist_artifact(artifact, str(artifact_path), "RUN-1")

            by_path = store.load_requirement_artifact_payload(artifact_path=str(artifact_path))
            by_version = store.load_requirement_artifact_payload(
                document_version_id=artifact.document_version_id
            )

        self.assertEqual(by_path, local_payload)
        self.assertEqual(by_version, local_payload)
        self.assertEqual(by_path, artifact.model_dump(mode="json"))

    def test_missing_artifact_payload_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PostgresStore(_settings(Path(temp_dir)))

            payload = store.load_requirement_artifact_payload(artifact_path="missing.json")

        self.assertIsNone(payload)


def _settings(root: Path) -> AppSettings:
    return AppSettings(
        paths=PathsSettings(
            project_root=root,
            global_cache_dir=root / ".global_cache",
            documents_inbox_dir=root / "documents" / "inbox",
            generated_requirements_dir=root / "generated",
            chroma_persist_dir=root / "runtime" / "databases" / "chroma",
            runtime_staging_dir=root / "runtime" / "staging",
            runtime_logs_dir=root / "runtime" / "logs",
            runtime_locks_dir=root / "runtime" / "locks",
        ),
        postgres=PostgresSettings(
            mode="local_json",
            local_path=root / "runtime" / "postgres.jsonl",
        ),
        neo4j=Neo4jSettings(
            mode="local_json",
            local_path=root / "runtime" / "neo4j.jsonl",
        ),
    )


def _artifact() -> RequirementArtifact:
    trace = SourceTrace(
        chunk_id="CHUNK-1",
        quote="The system shall import files.",
        start_char=0,
        end_char=30,
        page=1,
        section="Overview",
    )
    return RequirementArtifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        version="1.0",
        source_path="source.txt",
        source_checksum="abc",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        facts=[
            VerifiedFact(
                fact_id="FACT-1",
                canonical_fact_id="CF-1",
                text="The system shall import files.",
                source_trace=trace,
            )
        ],
        requirements=[
            VerifiedRequirement(
                requirement_id="REQ-1",
                revision_id="REQREV-1",
                requirement_key="import files",
                statement="The system shall import files.",
                normalized_statement="the system shall import files.",
                requirement_type="functional",
                priority="medium",
                fact_ids=["FACT-1"],
                source_trace=trace,
                evidence=[
                    RequirementEvidence(
                        evidence_id="REQEVID-1",
                        fact_ids=["FACT-1"],
                        source_trace=trace,
                    )
                ],
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
