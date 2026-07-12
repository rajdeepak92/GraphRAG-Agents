from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    KnowledgeGraphSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    IngestionRequest,
    RequirementArtifact,
    RequirementEvidence,
    SourceTrace,
    VerifiedFact,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.workflows.ingestion_graph import _run_pipeline


class IngestionPersistenceBoundaryTests(unittest.TestCase):
    def test_ingestion_persists_chunks_and_requirements_to_separate_stores(self) -> None:
        events: list[str] = []
        artifact = _artifact()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            document = root / "source.txt"
            document.write_text("The system shall import files.", encoding="utf-8")
            settings = _settings(root, postgres_mode="postgres", neo4j_mode="neo4j")
            postgres = _FakePostgres(events)
            neo4j = _FakeNeo4j(events)
            chroma = _FakeChroma(events)

            class FakeRequirementDiscoveryAgent:
                def __init__(
                    self,
                    reasoning_model: object,
                    *,
                    logger: object = None,
                    coverage_ledger: object = None,
                ) -> None:
                    self.reasoning_model = reasoning_model
                    self.logger = logger
                    self.coverage_ledger = coverage_ledger

                def run(self, manifest: object) -> object:
                    events.append("requirements.discover")
                    return object()

            def fake_build_requirement_artifact(**_: object) -> RequirementArtifact:
                events.append("requirements.build_artifact")
                return artifact

            request = IngestionRequest(project="PROJECT", document=document, version="1.0")

            with (
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.load_config",
                    return_value=settings,
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.PostgresStore",
                    return_value=postgres,
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.Neo4jStore",
                    return_value=neo4j,
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.ChromaStore",
                    return_value=chroma,
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.create_reasoning_model",
                    return_value=_FakeModel("reasoning"),
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.create_embedding_model",
                    return_value=_FakeModel("embedding"),
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.create_reranker_model",
                    return_value=_FakeModel("reranker"),
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.parse_document",
                    return_value=([], "parser"),
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.chunk_blocks",
                    return_value=([_chunk()], "chunker"),
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.RequirementDiscoveryAgent",
                    FakeRequirementDiscoveryAgent,
                ),
                patch(
                    "multi_agentic_graph_rag.workflows.ingestion_graph.build_requirement_artifact",
                    side_effect=fake_build_requirement_artifact,
                ),
            ):
                result = _run_pipeline(
                    {"request": request.model_dump(mode="json"), "run_id": "RUN-1"}
                )

        self.assertIn("neo4j.project_manifest", events)
        self.assertIn("chroma.index_chunks", events)
        self.assertIn("postgres.persist_artifact", events)
        self.assertNotIn("neo4j.project_artifact", events)
        self.assertFalse(neo4j.project_artifact_called)
        self.assertLess(events.index("neo4j.project_manifest"), events.index("chroma.index_chunks"))
        self.assertLess(events.index("chroma.index_chunks"), events.index("requirements.discover"))
        self.assertIs(postgres.persisted_artifact, artifact)
        self.assertEqual(postgres.persisted_run_id, "RUN-1")
        self.assertTrue(str(postgres.persisted_artifact_path).endswith("requirements_full.json"))
        self.assertTrue(str(result["artifact_path"]).endswith("requirements.json"))
        self.assertTrue(str(result["full_artifact_path"]).endswith("requirements_full.json"))
        self.assertEqual(result["requirement_ids"], ["REQ-1"])


class _FakePostgres:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.persisted_artifact: RequirementArtifact | None = None
        self.persisted_artifact_path = ""
        self.persisted_run_id = ""

    def check(self) -> str:
        self.events.append("postgres.check")
        return "PASS postgres fake"

    def ensure_schema(self) -> None:
        self.events.append("postgres.ensure_schema")

    def assert_version_allowed(self, manifest: object, replace_version: bool) -> None:
        self.events.append("postgres.assert_version_allowed")

    def load_requirement_revision_snapshot(
        self,
        *,
        project: str,
        document_id: str,
    ) -> dict[str, object]:
        self.events.append("postgres.load_requirement_revision_snapshot")
        return {}

    def persist_manifest(self, manifest: object) -> None:
        self.events.append("postgres.persist_manifest")

    def persist_artifact(
        self,
        artifact: RequirementArtifact,
        artifact_path: str,
        run_id: str,
    ) -> RequirementArtifact:
        self.events.append("postgres.persist_artifact")
        self.persisted_artifact = artifact
        self.persisted_artifact_path = artifact_path
        self.persisted_run_id = run_id
        return artifact

    def load_requirement_artifact_payload(
        self,
        artifact_path: str | None = None,
        document_version_id: str | None = None,
    ) -> dict[str, object] | None:
        self.events.append("postgres.load_requirement_artifact_payload")
        if self.persisted_artifact is None:
            return None
        return self.persisted_artifact.model_dump(mode="json")

    def record_run(self, run_id: str, status: str, payload: dict[str, object]) -> None:
        self.events.append("postgres.record_run")


class _FakeNeo4j:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.project_artifact_called = False

    def check(self) -> str:
        self.events.append("neo4j.check")
        return "PASS neo4j fake"

    def project_manifest(self, manifest: object) -> None:
        self.events.append("neo4j.project_manifest")

    def project_artifact(self, artifact: RequirementArtifact) -> None:
        self.events.append("neo4j.project_artifact")
        self.project_artifact_called = True
        raise AssertionError("ingestion must not project requirement artifacts to Neo4j")


class _FakeChroma:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def check(self) -> str:
        self.events.append("chroma.check")
        return "PASS chroma fake"

    def index_chunks(self, manifest: object, embedding_model: object) -> None:
        self.events.append("chroma.index_chunks")


class _FakeModel:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name


def _settings(root: Path, *, postgres_mode: str, neo4j_mode: str) -> AppSettings:
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
            mode=postgres_mode,
            local_path=root / "runtime" / "postgres.jsonl",
        ),
        neo4j=Neo4jSettings(
            mode=neo4j_mode,
            local_path=root / "runtime" / "neo4j.jsonl",
        ),
        # Legacy chunk-only path (explicit GraphRAG opt-out): this suite exercises
        # ingestion persistence boundaries, not knowledge-graph construction.
        knowledge_graph=KnowledgeGraphSettings(enabled=False),
    )


def _chunk() -> DocumentChunk:
    text = "The system shall import files."
    return DocumentChunk(
        chunk_id="CHUNK-1",
        ordinal=1,
        text=text,
        normalized_text=text.lower(),
        page=1,
        section="Overview",
        start_char=0,
        end_char=len(text),
        source_block_ids=["BLOCK-1"],
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
