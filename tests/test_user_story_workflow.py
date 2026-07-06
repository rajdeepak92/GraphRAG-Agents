from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

from pydantic import BaseModel

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.domain.schemas import UserStoryArtifact, UserStoryRequest
from multi_agentic_graph_rag.observability.session import command_session
from multi_agentic_graph_rag.workflows import user_story_graph as usg

T = TypeVar("T", bound=BaseModel)


class UserStoryWorkflowTests(unittest.TestCase):
    def test_local_json_end_to_end_generates_persists_and_projects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            requirements_path = root / "in" / "requirement.json"
            requirements_path.parent.mkdir(parents=True, exist_ok=True)
            requirements_path.write_text(json.dumps(_compact_requirements()), encoding="utf-8")

            request = UserStoryRequest(requirements_path=requirements_path, project="SIIMCS")

            with ExitStack() as stack:
                stack.enter_context(patch.object(usg, "load_config", lambda: settings))
                stack.enter_context(
                    patch.object(usg, "_validate_required_user_story_stack", lambda _s: None)
                )
                stack.enter_context(
                    patch.object(
                        usg,
                        "create_reasoning_model",
                        lambda settings, logger=None, run_dir=None: _FakeReasoner(),
                    )
                )
                stack.enter_context(
                    patch.object(usg, "create_embedding_model", lambda settings: _FakeEmbedding())
                )
                stack.enter_context(
                    patch.object(usg, "create_reranker_model", lambda settings: _FakeReranker())
                )
                stack.enter_context(patch.object(usg, "ChromaStore", _FakeChromaStore))
                session = stack.enter_context(
                    command_session(
                        project="SIIMCS",
                        version="1.0",
                        command="user-stories",
                        run_id="RUN-TEST-USER-STORIES",
                        project_root=root,
                    )
                )
                result = usg.run_user_story_generation(request, session=session)

            self.assertEqual(result.requirement_count, 2)
            self.assertEqual(len(result.story_ids), 2)

            artifact_path = requirements_path.parent / "user_stories.json"
            self.assertTrue(artifact_path.exists())
            artifact = UserStoryArtifact.model_validate(
                json.loads(artifact_path.read_text(encoding="utf-8"))
            )
            self.assertEqual(len(artifact.stories), 2)
            self.assertTrue(all(story_id.startswith("US-") for story_id in artifact.stories))
            self.assertEqual(set(artifact.coverage), {"REQ-1", "REQ-2"})

            postgres_rows = _read_jsonl(settings.postgres.local_path)
            self.assertEqual(
                len([row for row in postgres_rows if row.get("kind") == "user_story"]), 2
            )
            self.assertEqual(
                len([row for row in postgres_rows if row.get("kind") == "user_story_artifact"]), 1
            )
            self.assertTrue(
                any(
                    row.get("kind") == "ingestion_run" and row.get("status") == "completed"
                    for row in postgres_rows
                )
            )

            neo4j_rows = _read_jsonl(settings.neo4j.local_path)
            self.assertEqual(
                len([row for row in neo4j_rows if row.get("kind") == "user_story_projection"]), 2
            )


class _FakeReasoner:
    provider_name = "huggingface"

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        return schema.model_validate({"user_stories": [_story_payload()]})


class _FakeEmbedding:
    provider_name = "huggingface"
    embedding_fingerprint = "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeReranker:
    provider_name = "huggingface"

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        return list(range(len(documents)))


class _FakeChromaStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def check(self) -> str:
        return "PASS chroma fake"

    def query_chunks(
        self, query_embedding: list[float], document_version_id: str, n_results: int
    ) -> list[tuple[str, str, float]]:
        return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _story_payload() -> dict[str, Any]:
    return {
        "title": "Configure warning thresholds",
        "epic": "Threshold Management",
        "priority": "Medium",
        "persona": "Operations Engineer",
        "user_story": {
            "as_a": "operations engineer",
            "i_want": "to configure warning thresholds",
            "so_that": "alerts fire before equipment is damaged",
        },
        "business_value": "Reduces unplanned downtime through timely alerting",
        "acceptance_criteria": [
            {
                "id": "AC1",
                "title": "threshold crossed",
                "given": "a configured sensor",
                "when": "a threshold is crossed",
                "then": "an alert is raised",
            }
        ],
        "business_rules": [{"id": "BR1", "rule": "only authorized users may configure"}],
        "test_scenarios": [{"id": "TS1", "scenario": "cross a warning threshold"}],
        "definition_of_done": ["code reviewed", "tests passing"],
    }


def _compact_requirements() -> dict[str, Any]:
    return {
        "artifact_schema_version": "3.0-compact",
        "project": "SIIMCS",
        "document_id": "DOC-SIIMCS-1",
        "document_version_id": "DV-1",
        "doc_version": "1.0",
        "generated_at": "2026-07-01T00:00:00Z",
        "requirements": {
            "REQ-1": [
                {
                    "chunk_id": "CHUNK-0001",
                    "fact_id": "FACT-1",
                    "requirement_text": "Users shall configure warning thresholds.",
                    "requirement_type": "Functional Requirement",
                    "priority": "Medium",
                    "status": "Active",
                    "doc_version": "1.0",
                }
            ],
            "REQ-2": [
                {
                    "chunk_id": "CHUNK-0002",
                    "fact_id": "FACT-2",
                    "requirement_text": "The system shall notify responsible teams on alerts.",
                    "requirement_type": "Functional Requirement",
                    "priority": "High",
                    "status": "Active",
                    "doc_version": "1.0",
                }
            ],
        },
    }


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


if __name__ == "__main__":
    unittest.main()
