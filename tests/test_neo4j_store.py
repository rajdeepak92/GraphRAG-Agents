from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import multi_agentic_graph_rag.db.neo4j_store as neo4j_store
from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore, _project_manifest_tx
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    RequirementArtifact,
    TestScenarioBuildResult,
    TestScenarioRecord,
    UserStoryBuildResult,
    UserStoryRecord,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.test_scenario_builder import project_test_scenario_artifact
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact


class Neo4jStoreTests(unittest.TestCase):
    def test_project_artifact_is_noop_in_local_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = Neo4jStore(_settings(root))
            store.project_manifest(_manifest())
            store.project_artifact(_artifact())

            rows = [
                json.loads(line)
                for line in (root / "runtime" / "neo4j.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual([row["kind"] for row in rows], ["manifest_projection"])

    def test_artifact_projection_cypher_is_absent(self) -> None:
        self.assertFalse(hasattr(neo4j_store, "_project_artifact_tx"))
        source = Path(neo4j_store.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "artifact_projection",
            "CanonicalFact",
            "FactOccurrence",
            "RequirementRevision",
            "TRACED_TO",
            "DERIVED_FROM",
            "HAS_FACT_OCCURRENCE",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_user_story_coverage_projection_local_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = Neo4jStore(_settings(root))
            store.project_manifest(_manifest())
            store.project_artifact(_artifact())
            store.project_user_story_coverage(
                _user_story_artifact(),
                {"REQ-1": ["CHUNK-1"]},
            )

            rows = [
                json.loads(line)
                for line in (root / "runtime" / "neo4j.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        kinds = [row["kind"] for row in rows]
        self.assertEqual(kinds, ["manifest_projection", "user_story_projection"])
        projection = rows[1]
        self.assertEqual(projection["story_id"], "US-STORY-1")
        self.assertEqual(projection["requirement_id"], "REQ-1")
        self.assertEqual(projection["revision_id"], "REQREV-1")
        self.assertTrue(projection["covered"])
        self.assertEqual(projection["evidence_chunk_ids"], ["CHUNK-1"])

    def test_test_scenario_coverage_projection_local_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = Neo4jStore(_settings(root))
            store.project_manifest(_manifest())
            store.project_test_scenario_coverage(
                _test_scenario_artifact(),
                {"REQ-1": ["CHUNK-1"]},
            )

            rows = [
                json.loads(line)
                for line in (root / "runtime" / "neo4j.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        kinds = [row["kind"] for row in rows]
        self.assertEqual(kinds, ["manifest_projection", "test_scenario_projection"])
        projection = rows[1]
        self.assertEqual(projection["scenario_id"], "SC-SCENARIO-1")
        self.assertEqual(projection["story_id"], "US-STORY-1")
        self.assertEqual(projection["requirement_id"], "REQ-1")
        self.assertEqual(projection["revision_id"], "REQREV-1")
        self.assertEqual(projection["scenario_type"], "Positive")
        self.assertEqual(projection["evidence_chunk_ids"], ["CHUNK-1"])

    def test_cleanup_identity_projections_preserves_source_graph_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = Neo4jStore(_settings(root))
            store.project_manifest(_manifest())
            store.project_user_story_coverage(_user_story_artifact(), {"REQ-1": ["CHUNK-1"]})
            store.project_test_scenario_coverage(_test_scenario_artifact(), {"REQ-1": ["CHUNK-1"]})
            store.cleanup_identity_projections("PROJECT")
            kinds = [row["kind"] for row in store._read_local_rows()]
        self.assertEqual(kinds, ["manifest_projection"])

    def test_manifest_projection_keeps_document_chunk_metadata(self) -> None:
        tx = _FakeTx()
        manifest = _manifest().model_dump(mode="json")

        _project_manifest_tx(tx, manifest)

        self.assertEqual(len(tx.runs), 2)
        root_query = tx.runs[0]["query"]
        self.assertIn("Project", root_query)
        self.assertIn("Document", root_query)
        self.assertIn("DocumentVersion", root_query)
        self.assertIn("OWNS_DOCUMENT", root_query)
        self.assertIn("HAS_VERSION", root_query)

        chunk_query = tx.runs[1]["query"]
        chunk_params = tx.runs[1]["params"]
        self.assertIn("Chunk", chunk_query)
        self.assertIn("HAS_CHUNK", chunk_query)
        self.assertEqual(chunk_params["chunk_id"], "CHUNK-1")
        self.assertEqual(chunk_params["ordinal"], 1)
        self.assertEqual(chunk_params["text"], "The system shall import files.")
        self.assertEqual(chunk_params["normalized_text"], "the system shall import files.")
        self.assertEqual(chunk_params["page"], 7)
        self.assertEqual(chunk_params["section"], "Business Rules")
        self.assertEqual(chunk_params["start_char"], 10)
        self.assertEqual(chunk_params["end_char"], 40)
        self.assertEqual(chunk_params["source_block_ids"], ["BLOCK-1"])
        self.assertEqual(chunk_params["source_checksum"], "abc")
        self.assertEqual(chunk_params["project"], "PROJECT")
        self.assertEqual(chunk_params["document_id"], "DOC")
        self.assertEqual(chunk_params["document_version_id"], "DOC-v1")


class _FakeTx:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def run(self, query: str, **params: Any) -> None:
        self.runs.append({"query": query, "params": params})


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


def _manifest() -> DocumentManifest:
    text = "The system shall import files."
    return DocumentManifest(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        logical_name="source",
        version="1.0",
        source_path="source.txt",
        source_checksum="abc",
        parser_fingerprint="parser",
        chunker_fingerprint="chunker",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunks=[
            DocumentChunk(
                chunk_id="CHUNK-1",
                ordinal=1,
                text=text,
                normalized_text=text.lower(),
                page=7,
                section="Business Rules",
                start_char=10,
                end_char=40,
                source_block_ids=["BLOCK-1"],
            )
        ],
    )


def _artifact() -> RequirementArtifact:
    return RequirementArtifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        version="1.0",
        source_path="source.txt",
        source_checksum="abc",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        facts=[],
        requirements=[],
    )


def _user_story_artifact() -> UserStoryBuildResult:
    record = UserStoryRecord(
        story_id="US-STORY-1",
        requirement_id="REQ-1",
        requirement_revision_id="REQREV-1",
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
        acceptance_criteria=[
            "Given a valid source file, when the import runs, then records are available."
        ],
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


if __name__ == "__main__":
    unittest.main()
