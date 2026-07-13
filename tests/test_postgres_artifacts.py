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
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementEvidence,
    SourceTrace,
    TestScenarioBuildResult,
    TestScenarioRecord,
    UserStoryBuildResult,
    UserStoryRecord,
    UserStoryStatement,
    VerifiedFact,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.artifacts import (
    verify_requirement_artifact,
    write_canonical_requirements_artifact,
)
from multi_agentic_graph_rag.services.requirement_builder import (
    build_canonical_requirements_artifact,
)
from multi_agentic_graph_rag.services.requirement_source import load_requirement_source_local
from multi_agentic_graph_rag.services.test_scenario_builder import project_test_scenario_artifact
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact


class PostgresArtifactTests(unittest.TestCase):
    def test_persisted_artifact_payload_matches_canonical_requirements_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _artifact()
            artifact_path = write_canonical_requirements_artifact(
                build_canonical_requirements_artifact(artifact), root / "run"
            )
            local_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

            store = PostgresStore(_settings(root))
            persisted_artifact = store.persist_artifact(artifact, str(artifact_path), "RUN-1")

            by_path = store.load_requirement_artifact_payload(artifact_path=str(artifact_path))
            by_version = store.load_requirement_artifact_payload(
                document_version_id=artifact.document_version_id
            )

        self.assertEqual(artifact_path.name, "requirements.json")
        self.assertEqual(by_path, local_payload)
        self.assertEqual(by_version, by_path)
        self.assertEqual(
            by_path,
            build_canonical_requirements_artifact(persisted_artifact).model_dump(mode="json"),
        )

    def test_canonical_requirements_round_trip_without_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _artifact(fact_ids=["FACT-1", "FACT-2"])
            public = build_canonical_requirements_artifact(artifact)
            path = write_canonical_requirements_artifact(public, root / "run")
            verified = verify_requirement_artifact(path)
            source = load_requirement_source_local(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["artifact_schema_version"], "5.0-requirements")
        self.assertEqual(len(payload["requirements"]), 1)
        self.assertEqual(verified.artifact_schema_version, "5.0-requirements")
        self.assertEqual(len(source.requirements), 1)
        self.assertEqual(source.requirements[0].requirement_id, "REQ-1")
        self.assertEqual(source.requirements[0].requirement_id, "REQ-1")
        self.assertEqual(source.requirements[0].evidence_chunk_ids, ["CHUNK-1"])

    def test_reconcile_regenerates_canonical_requirements_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _artifact()
            store = PostgresStore(_settings(root))
            canonical_path = root / "run" / "requirements.json"
            store.persist_artifact(artifact, str(canonical_path), "RUN-1")

            report = ArtifactMirror(store).reconcile(project="PROJECT")
            payload = json.loads(canonical_path.read_text(encoding="utf-8"))

        self.assertIn(str(canonical_path), report.repaired_paths)
        self.assertEqual(payload["artifact_schema_version"], "5.0-requirements")
        self.assertIn("requirement_id", payload["requirements"][0])
        self.assertEqual(payload["requirements"][0]["requirement_id"], "REQ-1")

    def test_user_story_artifact_round_trips_and_indexes_stories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _user_story_artifact()
            artifact_path = str(root / "run" / "user_stories.json")

            store = PostgresStore(_settings(root))
            store.persist_user_story_artifact(artifact, artifact_path, "RUN-1")

            by_path = store.load_user_story_artifact_payload(artifact_path=artifact_path)
            by_version = store.load_user_story_artifact_payload(
                document_version_id=artifact.artifact.document_version_id
            )
            rows = [
                json.loads(line)
                for line in (root / "runtime" / "postgres.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(by_path, artifact.artifact.model_dump(mode="json"))
        self.assertEqual(by_version, artifact.artifact.model_dump(mode="json"))
        self.assertEqual(by_path["artifact_schema_version"], "3.0-user-stories")
        self.assertTrue(by_path["stories"][0]["story_id"].startswith("US-"))
        self.assertTrue(by_path["stories"][0]["requirement_id"].startswith("REQ-"))
        self.assertEqual(by_path["stories"][0]["revision_id"], "REQREV-1")
        self.assertEqual(by_path["stories"][0]["source_req_id"], "SYS_REQ_001")
        self.assertEqual(by_path["traceability"][0]["source_req_id"], "SYS_REQ_001")
        story_rows = [row for row in rows if row.get("kind") == "user_story"]
        self.assertEqual(len(story_rows), 1)
        self.assertEqual(story_rows[0]["story_id"], "US-STORY-1")
        self.assertTrue(str(story_rows[0]["story_id"]).startswith("US-"))
        self.assertEqual(story_rows[0]["requirement_id"], "REQ-1")
        self.assertEqual(story_rows[0]["status"], "active")

    def test_user_story_tables_are_managed(self) -> None:
        self.assertIn("user_stories", _MANAGED_TABLES)
        self.assertIn("user_story_artifacts", _MANAGED_TABLES)

    def test_test_scenario_artifact_round_trips_and_indexes_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = _test_scenario_artifact()
            artifact_path = str(root / "run" / "test_scenarios.json")

            store = PostgresStore(_settings(root))
            store.persist_test_scenario_artifact(artifact, artifact_path, "RUN-1")

            by_path = store.load_test_scenario_artifact_payload(artifact_path=artifact_path)
            by_version = store.load_test_scenario_artifact_payload(
                document_version_id=artifact.artifact.document_version_id
            )
            rows = [
                json.loads(line)
                for line in (root / "runtime" / "postgres.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(by_path, artifact.artifact.model_dump(mode="json"))
        self.assertEqual(by_version, artifact.artifact.model_dump(mode="json"))
        self.assertEqual(by_path["artifact_schema_version"], "3.0-test-scenarios")
        self.assertTrue(by_path["scenarios"][0]["scenario_id"].startswith("SC-"))
        self.assertTrue(by_path["scenarios"][0]["story_id"].startswith("US-"))
        self.assertTrue(by_path["scenarios"][0]["requirement_id"].startswith("REQ-"))
        self.assertEqual(by_path["scenarios"][0]["revision_id"], "REQREV-1")
        self.assertEqual(by_path["scenarios"][0]["source_req_id"], "SYS_REQ_001")
        self.assertEqual(by_path["traceability"][0]["source_req_id"], "SYS_REQ_001")
        scenario_rows = [row for row in rows if row.get("kind") == "test_scenario"]
        self.assertEqual(len(scenario_rows), 1)
        self.assertEqual(scenario_rows[0]["scenario_id"], "SC-SCENARIO-1")
        self.assertTrue(str(scenario_rows[0]["scenario_id"]).startswith("SC-"))
        self.assertEqual(scenario_rows[0]["story_id"], "US-STORY-1")
        self.assertEqual(scenario_rows[0]["requirement_id"], "REQ-1")
        self.assertEqual(scenario_rows[0]["status"], "active")

    def test_test_scenario_tables_are_managed(self) -> None:
        self.assertIn("test_scenarios", _MANAGED_TABLES)
        self.assertIn("test_scenario_artifacts", _MANAGED_TABLES)

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


def _user_story_artifact() -> UserStoryBuildResult:
    record = UserStoryRecord(
        story_id="US-STORY-1",
        requirement_id="REQ-1",
        requirement_revision_id="REQREV-1",
        source_req_id="SYS_REQ_001",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        title="Import files reliably",
        priority="Medium",
        persona="Data Engineer",
        user_story=UserStoryStatement(
            as_a="data engineer",
            i_want="to import files",
            so_that="downstream reporting stays current",
        ),
        acceptance_criteria=["valid file imports: records are available after the import runs"],
        confidence=0.85,
    )
    artifact = project_user_story_artifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        records={record.story_id: record},
    )
    return UserStoryBuildResult(
        artifact=artifact,
        records={record.story_id: record},
        coverage={"REQ-1": [record.story_id]},
    )


def _test_scenario_artifact() -> TestScenarioBuildResult:
    record = TestScenarioRecord(
        scenario_id="SC-SCENARIO-1",
        story_id="US-STORY-1",
        requirement_id="REQ-1",
        requirement_revision_id="REQREV-1",
        source_req_id="SYS_REQ_001",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        title="Import valid file succeeds",
        description="Verify a valid source file is imported into the system",
        scenario_type="Positive",
        preconditions=["a valid source file is available"],
        expected_result="The file records are available for downstream reporting",
        priority="Medium",
        confidence=0.9,
    )
    artifact = project_test_scenario_artifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="1.0",
        records={record.scenario_id: record},
    )
    return TestScenarioBuildResult(
        artifact=artifact,
        records={record.scenario_id: record},
        coverage={"US-STORY-1": [record.scenario_id]},
        requirement_coverage={"REQ-1": [record.scenario_id]},
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
