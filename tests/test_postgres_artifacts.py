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
from multi_agentic_graph_rag.db.postgres import _MANAGED_TABLES, PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    AcceptanceCriterion,
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementEvidence,
    SourceTrace,
    UserStoryArtifact,
    UserStoryRecord,
    UserStoryStatement,
    VerifiedFact,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.services.artifacts import (
    write_compact_requirement_artifact,
    write_requirement_artifact,
)
from multi_agentic_graph_rag.services.requirement_builder import (
    build_compact_requirement_artifact,
)


class PostgresArtifactTests(unittest.TestCase):
    def test_persisted_artifact_payload_matches_local_requirements_full_json(self) -> None:
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

        self.assertEqual(artifact_path.name, "requirements_full.json")
        self.assertEqual(by_path, local_payload)
        self.assertEqual(by_version, local_payload)
        self.assertEqual(by_path, artifact.model_dump(mode="json"))

    def test_compact_requirements_json_is_agent_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _artifact(
                fact_ids=["FACT-1", "FACT-2"],
                delta_events=[
                    RequirementDeltaEvent(
                        event_id="EVENT-1",
                        event_type="superseded",
                        requirement_id="REQ-1",
                        revision_id="REQREV-1",
                        document_version_id="DOC-v1",
                        impacted_artifact_types=["user_story"],
                    )
                ],
            )
            compact = build_compact_requirement_artifact(artifact)
            compact_path = write_compact_requirement_artifact(compact, root / "run")
            payload = json.loads(compact_path.read_text(encoding="utf-8"))

        self.assertEqual(compact_path.name, "requirements.json")
        self.assertEqual(payload["artifact_schema_version"], "3.0-compact")
        self.assertEqual(payload["project"], "PROJECT")
        self.assertEqual(payload["document_id"], "DOC")
        self.assertEqual(payload["document_version_id"], "DOC-v1")
        self.assertEqual(payload["doc_version"], "1.0")
        self.assertIsInstance(payload["requirements"], dict)
        self.assertEqual(list(payload["requirements"]), ["REQ-1"])

        occurrences = payload["requirements"]["REQ-1"]
        self.assertEqual(len(occurrences), 2)
        self.assertEqual(
            [occurrence["fact_id"] for occurrence in occurrences], ["FACT-1", "FACT-2"]
        )
        self.assertEqual(
            set(occurrences[0]),
            {
                "chunk_id",
                "fact_id",
                "requirement_text",
                "requirement_type",
                "priority",
                "status",
                "doc_version",
            },
        )
        self.assertEqual(occurrences[0]["chunk_id"], "CHUNK-1")
        self.assertEqual(occurrences[0]["requirement_text"], "The system shall import files.")
        self.assertEqual(occurrences[0]["requirement_type"], "functional")
        self.assertEqual(occurrences[0]["priority"], "Medium")
        self.assertEqual(occurrences[0]["status"], "Superseded")
        self.assertEqual(occurrences[0]["doc_version"], "1.0")

        forbidden_keys = {
            "canonical_facts",
            "facts",
            "delta_events",
            "evidence",
            "quote",
            "start_char",
            "end_char",
            "page",
            "section",
            "source_checksum",
            "revision_id",
            "normalized_statement",
            "impacted_artifact_types",
        }
        self.assertTrue(forbidden_keys.isdisjoint(_json_keys(payload)))

    def test_user_story_artifact_round_trips_and_indexes_stories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _user_story_artifact()
            artifact_path = str(root / "run" / "user_stories.json")

            store = PostgresStore(_settings(root))
            store.persist_user_story_artifact(artifact, artifact_path, "RUN-1")

            by_path = store.load_user_story_artifact_payload(artifact_path=artifact_path)
            by_version = store.load_user_story_artifact_payload(
                document_version_id=artifact.document_version_id
            )
            rows = [
                json.loads(line)
                for line in (root / "runtime" / "postgres.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(by_path, artifact.model_dump(mode="json"))
        self.assertEqual(by_version, artifact.model_dump(mode="json"))
        story_rows = [row for row in rows if row.get("kind") == "user_story"]
        self.assertEqual(len(story_rows), 1)
        self.assertEqual(story_rows[0]["story_id"], "US-STORY-1")
        self.assertEqual(story_rows[0]["requirement_id"], "REQ-1")
        self.assertEqual(story_rows[0]["status"], "active")

    def test_user_story_tables_are_managed(self) -> None:
        self.assertIn("user_stories", _MANAGED_TABLES)
        self.assertIn("user_story_artifacts", _MANAGED_TABLES)

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


def _artifact(
    *,
    fact_ids: list[str] | None = None,
    delta_events: list[RequirementDeltaEvent] | None = None,
) -> RequirementArtifact:
    fact_ids = fact_ids or ["FACT-1"]
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
                fact_id=fact_id,
                canonical_fact_id="CF-1",
                text="The system shall import files.",
                source_trace=trace,
            )
            for fact_id in fact_ids
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
                fact_ids=fact_ids,
                source_trace=trace,
                evidence=[
                    RequirementEvidence(
                        evidence_id="REQEVID-1",
                        fact_ids=fact_ids,
                        source_trace=trace,
                    )
                ],
            )
        ],
        delta_events=delta_events or [],
    )


def _user_story_artifact() -> UserStoryArtifact:
    record = UserStoryRecord(
        story_id="US-STORY-1",
        requirement_id="REQ-1",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        title="Import files reliably",
        epic="Ingestion",
        priority="Medium",
        persona="Data Engineer",
        user_story=UserStoryStatement(
            as_a="data engineer",
            i_want="to import files",
            so_that="downstream reporting stays current",
        ),
        business_value="Keeps operational reporting current",
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                title="valid file imports",
                given="a valid source file",
                when="the import runs",
                then="records are available",
            )
        ],
    )
    return UserStoryArtifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        stories={record.story_id: record},
        coverage={"REQ-1": [record.story_id]},
    )


def _json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_json_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_json_keys(child))
        return keys
    return set()


if __name__ == "__main__":
    unittest.main()
