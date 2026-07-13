"""Phase B: knowledge-graph readiness state machine + downstream blocking.

Covers the guarded build (`run_guarded_knowledge_graph_build`) and the
graph-primary gate (`require_knowledge_graph_when_primary`):

* a successful build reaches ``ready`` and moves the active pointer last;
* a failed build persists ``failed``, never moves the pointer, and re-raises;
* requirements are never touched when the KG build fails (inline degraded path);
* the gate blocks unless status is ``ready``, grandfathers pre-state versions on
  assertion presence, and honours the explicit graph-primary opt-out;
* transient failures retry within the bounded budget; deterministic failures do not;
* a rebuild transitions through ``rebuilding`` and increments the attempt counter.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.schemas import (
    AssertionEvidenceRecord,
    AssertionRecord,
    EntityRecord,
    KnowledgeGraphArtifact,
    KnowledgeGraphStateRecord,
    SourceTrace,
)
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.services.knowledge_graph_state import (
    run_guarded_knowledge_graph_build,
)
from multi_agentic_graph_rag.services.knowledge_retrieval import (
    require_knowledge_graph_when_primary,
)

_BUILD = "multi_agentic_graph_rag.services.knowledge_graph_state.build_and_project_knowledge_graph"


class _FakeNeo4j:
    def __init__(self, has_assertions: bool = False) -> None:
        self._has_assertions = has_assertions
        self.pointer_moves: list[str] = []

    def set_active_knowledge_version(self, *, document_id: str, document_version_id: str) -> None:
        self.pointer_moves.append(document_version_id)

    def has_knowledge_assertions(self, document_version_id: str) -> bool:
        return self._has_assertions


class KnowledgeGraphReadinessTests(unittest.TestCase):
    def _guarded(
        self,
        store: PostgresStore,
        neo4j: _FakeNeo4j,
        settings: AppSettings,
        logger: RunLogger | None = None,
    ) -> Any:
        return run_guarded_knowledge_graph_build(
            project="PROJECT",
            document_id="DOC",
            document_version_id="DOC-v1",
            doc_version="V1",
            chunks=[],
            reasoning_model=object(),
            neo4j=neo4j,
            postgres=store,
            settings=settings,
            run_id="RUN-1",
            logger=logger,
        )

    # 1. Success -> ready, counts recorded, pointer moved (once).
    def test_successful_build_marks_ready_and_moves_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            neo4j = _FakeNeo4j()
            with mock.patch(_BUILD, return_value=_artifact()):
                result = self._guarded(store, neo4j, settings)

            state = store.get_knowledge_graph_state("DOC-v1")

        self.assertEqual(result.state.status, "ready")
        self.assertEqual(neo4j.pointer_moves, ["DOC-v1"])
        assert state is not None
        self.assertEqual(state.status, "ready")
        self.assertEqual(state.assertion_count, 1)
        self.assertEqual(state.evidence_count, 1)
        self.assertEqual(state.graph_schema_version, "1.0-knowledge-graph")
        self.assertIsNotNone(state.completed_at)

    # 2. Failure -> failed persisted, pointer NOT moved, exception re-raised.
    def test_build_failure_marks_failed_and_does_not_move_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            neo4j = _FakeNeo4j()
            with (
                mock.patch(_BUILD, side_effect=ValueError("bad extraction")),
                self.assertRaises(ValueError),
            ):
                self._guarded(store, neo4j, settings)

            state = store.get_knowledge_graph_state("DOC-v1")

        self.assertEqual(neo4j.pointer_moves, [])
        assert state is not None
        self.assertEqual(state.status, "failed")
        self.assertRegex(state.failure_reason or "", r"^ValueError: <message redacted")

    # 3. Gate blocks when status is failed, naming the rebuild command.
    def test_gate_blocks_when_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            store.upsert_knowledge_graph_state(_state("failed", failure_reason="boom"))
            with self.assertRaises(ConfigurationError) as ctx:
                require_knowledge_graph_when_primary(
                    settings=settings,
                    neo4j=_FakeNeo4j(has_assertions=True),
                    document_version_id="DOC-v1",
                    primary=True,
                    stage="user-story",
                    postgres=store,
                    project="PROJECT",
                )
        message = str(ctx.exception)
        self.assertIn("failed", message)
        self.assertIn("build-knowledge-graph", message)

    # 4. Gate blocks when a build is still in progress (partial graph unauthorized).
    def test_gate_blocks_when_building(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            store.upsert_knowledge_graph_state(_state("building"))
            with self.assertRaises(ConfigurationError):
                require_knowledge_graph_when_primary(
                    settings=settings,
                    neo4j=_FakeNeo4j(has_assertions=True),
                    document_version_id="DOC-v1",
                    primary=True,
                    stage="test-scenario",
                    postgres=store,
                    project="PROJECT",
                )

    # 5. Gate allows when ready.
    def test_gate_allows_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            store.upsert_knowledge_graph_state(_state("ready"))
            require_knowledge_graph_when_primary(
                settings=settings,
                neo4j=_FakeNeo4j(has_assertions=False),
                document_version_id="DOC-v1",
                primary=True,
                stage="user-story",
                postgres=store,
                project="PROJECT",
            )

    # 6. Grandfather: no state row + assertions present -> allowed.
    def test_gate_grandfathers_preexisting_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            require_knowledge_graph_when_primary(
                settings=settings,
                neo4j=_FakeNeo4j(has_assertions=True),
                document_version_id="DOC-v1",
                primary=True,
                stage="user-story",
                postgres=store,
                project="PROJECT",
            )

    # 7. No state row + no assertions -> blocked.
    def test_gate_blocks_grandfather_without_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            with self.assertRaises(ConfigurationError):
                require_knowledge_graph_when_primary(
                    settings=settings,
                    neo4j=_FakeNeo4j(has_assertions=False),
                    document_version_id="DOC-v1",
                    primary=True,
                    stage="user-story",
                    postgres=store,
                    project="PROJECT",
                )

    # 8. Explicit opt-out (primary=False) allows legacy even with a failed KG.
    def test_explicit_optout_allows_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            store.upsert_knowledge_graph_state(_state("failed"))
            require_knowledge_graph_when_primary(
                settings=settings,
                neo4j=_FakeNeo4j(has_assertions=False),
                document_version_id="DOC-v1",
                primary=False,
                stage="user-story",
                postgres=store,
                project="PROJECT",
            )

    # 9. Transient failure is retried within budget, then succeeds -> ready.
    def test_transient_failure_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky(**_: Any) -> KnowledgeGraphArtifact:
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("temporary neo4j unavailable")
            return _artifact()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))  # build_max_attempts default 2
            store = PostgresStore(settings)
            neo4j = _FakeNeo4j()
            with mock.patch(_BUILD, side_effect=flaky):
                result = self._guarded(store, neo4j, settings)

        self.assertEqual(calls["n"], 2)
        self.assertEqual(result.state.status, "ready")

    # 10. Deterministic failure is NOT retried.
    def test_deterministic_failure_not_retried(self) -> None:
        calls = {"n": 0}

        def always_bad(**_: Any) -> KnowledgeGraphArtifact:
            calls["n"] += 1
            raise ValueError("schema violation")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            with (
                mock.patch(_BUILD, side_effect=always_bad),
                self.assertRaises(ValueError),
            ):
                self._guarded(store, _FakeNeo4j(), settings)

        self.assertEqual(calls["n"], 1)

    # 11. Rebuild transitions through 'rebuilding' and increments attempt.
    def test_rebuild_marks_rebuilding_and_increments_attempt(self) -> None:
        seen_status: list[str] = []
        real_artifact = _artifact()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            store = PostgresStore(settings)
            neo4j = _FakeNeo4j()
            with mock.patch(_BUILD, return_value=real_artifact):
                self._guarded(store, neo4j, settings)  # attempt 1, building

                original = store.upsert_knowledge_graph_state

                def spy(state: KnowledgeGraphStateRecord) -> None:
                    seen_status.append(state.status)
                    original(state)

                with mock.patch.object(store, "upsert_knowledge_graph_state", side_effect=spy):
                    result = self._guarded(store, neo4j, settings)  # attempt 2

        self.assertIn("rebuilding", seen_status)
        self.assertEqual(result.state.attempt, 2)

    def test_retry_logs_attempt_context_and_one_sanitized_terminal_failure(self) -> None:
        """Verify retry warnings, exhaustion ownership, and zero-secret sinks together."""
        secret = "SENTINEL-RETRY-SECRET"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = _settings(root)
            store = PostgresStore(settings)
            logger = RunLogger(
                root / "run.jsonl",
                root / "run.log",
                run_id="RUN-1",
                project="PROJECT",
                version="DOC-v1",
                level="DEBUG",
            )
            failure = TimeoutError(f"dsn=postgresql://user:{secret}@localhost/db")
            with (
                redirect_stderr(io.StringIO()),
                mock.patch(_BUILD, side_effect=failure),
                self.assertRaises(TimeoutError),
            ):
                self._guarded(store, _FakeNeo4j(), settings, logger=logger)
            logger.close()

            raw_jsonl = (root / "run.jsonl").read_text(encoding="utf-8")
            raw_text = (root / "run.log").read_text(encoding="utf-8")
            records = [json.loads(line) for line in raw_jsonl.splitlines()]

        attempts = [record for record in records if record["message"] == "retry.attempt_started"]
        warnings = [record for record in records if record["level"] == "WARNING"]
        terminals = [record for record in records if record["level"] == "EXCEPTION"]
        self.assertEqual([record["context"]["attempt"] for record in attempts], [1, 2])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["context"]["max_attempts"], 2)
        self.assertEqual(warnings[0]["context"]["retry_delay_seconds"], 0.0)
        self.assertEqual(len(terminals), 1)
        self.assertIs(terminals[0]["context"]["attempts_exhausted"], True)
        self.assertNotIn(secret, raw_jsonl + raw_text)


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


def _state(status: str, failure_reason: str | None = None) -> KnowledgeGraphStateRecord:
    return KnowledgeGraphStateRecord(
        document_version_id="DOC-v1",
        project="PROJECT",
        document_id="DOC",
        doc_version="V1",
        status=status,  # type: ignore[arg-type]
        failure_reason=failure_reason,
    )


def _artifact() -> KnowledgeGraphArtifact:
    return KnowledgeGraphArtifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        entities=[
            EntityRecord(
                entity_id="ENT-A",
                project="PROJECT",
                canonical_name="Gateway",
                normalized_name="gateway",
                entity_type="system",
            ),
            EntityRecord(
                entity_id="ENT-B",
                project="PROJECT",
                canonical_name="Data",
                normalized_name="data",
                entity_type="data_object",
            ),
        ],
        mentions=[],
        assertions=[
            AssertionRecord(
                assertion_id="AST-1",
                assertion_key="KEY-1",
                project="PROJECT",
                document_id="DOC",
                document_version_id="DOC-v1",
                subject_entity_id="ENT-A",
                predicate="COLLECTS",
                object_entity_id="ENT-B",
                modality="shall",
                polarity="positive",
                explicitness="explicit",
                confidence=0.9,
                display_text="Gateway COLLECTS Data",
            )
        ],
        evidence=[
            AssertionEvidenceRecord(
                evidence_id="ASTEVID-1",
                assertion_id="AST-1",
                source_trace=SourceTrace(
                    chunk_id="CHUNK-1",
                    quote="The gateway shall collect data",
                    start_char=0,
                    end_char=30,
                ),
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
