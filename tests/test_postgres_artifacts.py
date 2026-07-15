from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.postgres import _MANAGED_TABLES, PostgresStore
from multi_agentic_graph_rag.domain.errors import IngestionError
from multi_agentic_graph_rag.domain.schemas import (
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementEvidence,
    SourceTrace,
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

    def test_missing_artifact_payload_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PostgresStore(_settings(Path(temp_dir)))

            payload = store.load_requirement_artifact_payload(artifact_path="missing.json")

        self.assertIsNone(payload)

    def test_local_requirement_lifecycle_is_order_independent_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = PostgresStore(_settings(root))
            artifact = _lifecycle_artifact(include_superseded_revision=True)
            artifact.requirements.reverse()

            store.persist_artifact(artifact, str(root / "requirements.json"), "RUN-1")
            first_rows = store._read_local_rows()
            store.persist_artifact(artifact, str(root / "requirements.json"), "RUN-1")
            replayed_rows = store._read_local_rows()

        parent = next(row for row in replayed_rows if row.get("kind") == "requirement")
        revisions = {
            row["_local_key"]: row
            for row in replayed_rows
            if row.get("kind") == "requirement_revision"
        }
        self.assertEqual(parent["active_revision_id"], "REQREV-NEW")
        self.assertEqual(revisions["REQREV-OLD"]["status"], "superseded")
        self.assertEqual(revisions["REQREV-NEW"]["status"], "active")
        self.assertEqual(len(replayed_rows), len(first_rows))

    def test_local_cross_version_replacement_supersedes_existing_active_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = PostgresStore(_settings(root))
            original = _artifact()
            replacement = _lifecycle_artifact(
                include_superseded_revision=False,
                old_revision_id="REQREV-1",
            )

            store.persist_artifact(original, str(root / "v1.json"), "RUN-1")
            store.persist_artifact(replacement, str(root / "v2.json"), "RUN-2")
            rows = store._read_local_rows()

        parent = next(row for row in rows if row.get("kind") == "requirement")
        revisions = {
            row["_local_key"]: row for row in rows if row.get("kind") == "requirement_revision"
        }
        self.assertEqual(parent["active_revision_id"], "REQREV-NEW")
        self.assertEqual(revisions["REQREV-1"]["status"], "superseded")
        self.assertEqual(revisions["REQREV-NEW"]["status"], "active")

    def test_invalid_revision_lifecycle_fails_before_local_writes(self) -> None:
        for statuses in (("superseded",), ("active", "active")):
            with self.subTest(statuses=statuses), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                artifact = _artifact()
                revisions = []
                for ordinal, status in enumerate(statuses, start=1):
                    revisions.append(
                        artifact.requirements[0].model_copy(
                            update={
                                "revision_id": f"REQREV-{ordinal}",
                                "status": status,
                            },
                            deep=True,
                        )
                    )
                invalid = artifact.model_copy(update={"requirements": revisions}, deep=True)
                store = PostgresStore(_settings(root))

                with self.assertRaisesRegex(
                    IngestionError,
                    "exactly one active revision",
                ):
                    store.persist_artifact(invalid, str(root / "requirements.json"), "RUN-1")

                self.assertFalse(store.settings.postgres.local_path.exists())

    def test_postgres_supersedes_before_active_insert_and_upserts_parent_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PostgresStore(_settings(Path(temp_dir)))
            artifact = _lifecycle_artifact(include_superseded_revision=True)
            artifact.requirements.reverse()
            active = next(row for row in artifact.requirements if row.status == "active")
            cursor = MagicMock()

            store._persist_requirement_ledger_postgres(
                cursor,
                artifact,
                {active.requirement_id: active},
            )

        calls = cursor.execute.call_args_list
        statements = [call.args[0] for call in calls]
        supersede_index = next(
            index for index, sql in enumerate(statements) if "update requirement_revisions" in sql
        )
        revision_insert_indexes = [
            index
            for index, sql in enumerate(statements)
            if "insert into requirement_revisions" in sql
        ]
        parent_calls = [call for call in calls if "insert into requirements" in call.args[0]]
        revision_calls = [
            call for call in calls if "insert into requirement_revisions" in call.args[0]
        ]

        self.assertLess(supersede_index, min(revision_insert_indexes))
        self.assertEqual(len(parent_calls), 1)
        self.assertEqual(parent_calls[0].args[1][-1], "REQREV-NEW")
        persisted_statuses = {call.args[1][0]: call.args[1][-2] for call in revision_calls}
        self.assertEqual(
            persisted_statuses,
            {"REQREV-NEW": "active", "REQREV-OLD": "superseded"},
        )


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


def _lifecycle_artifact(
    *,
    include_superseded_revision: bool,
    old_revision_id: str = "REQREV-OLD",
) -> RequirementArtifact:
    artifact = _artifact()
    template = artifact.requirements[0]
    old = template.model_copy(
        update={
            "revision_id": old_revision_id,
            "statement": "The controller shall trip at 70C.",
            "normalized_statement": "the controller shall trip at 70c.",
            "status": "superseded",
        },
        deep=True,
    )
    old.evidence[0].evidence_id = "REQEVID-OLD"
    new = template.model_copy(
        update={
            "revision_id": "REQREV-NEW",
            "statement": "The controller shall trip at 80C.",
            "normalized_statement": "the controller shall trip at 80c.",
            "status": "active",
        },
        deep=True,
    )
    new.evidence[0].evidence_id = "REQEVID-NEW"
    event = RequirementDeltaEvent(
        event_id=f"EVENT-{old_revision_id}",
        event_type="superseded",
        requirement_id=template.requirement_id,
        revision_id=old_revision_id,
        superseded_by_revision_id=new.revision_id,
        document_version_id="DOC-v2",
    )
    requirements = [old, new] if include_superseded_revision else [new]
    return artifact.model_copy(
        update={
            "document_version_id": "DOC-v2",
            "version": "2.0",
            "requirements": requirements,
            "delta_events": [event],
        },
        deep=True,
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
