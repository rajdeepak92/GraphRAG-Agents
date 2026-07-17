"""Simplified PostgreSQL artifact, traceability, readiness, and run store."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.identifiers import normalize_project
from multi_agentic_graph_rag.domain.schemas import (
    CoverageSummary,
    KnowledgeGraphReadiness,
    RequirementsArtifact,
    TestScenariosArtifact,
    UserStoriesArtifact,
    canonical_checksum,
)

_PROJECT_DELETE_STEPS: tuple[tuple[str, str], ...] = (
    (
        "scenario_criteria",
        "DELETE FROM scenario_criteria c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "scenario_evidence",
        "DELETE FROM scenario_evidence c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "scenario_entities",
        "DELETE FROM scenario_entities c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "scenario_relationships",
        "DELETE FROM scenario_relationships c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "story_scenarios",
        "DELETE FROM story_scenarios c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "requirement_scenarios",
        "DELETE FROM requirement_scenarios c USING test_scenarios p"
        " WHERE c.scenario_id = p.scenario_id AND p.project = %s",
    ),
    (
        "acceptance_criteria",
        "DELETE FROM acceptance_criteria c USING user_stories p"
        " WHERE c.story_id = p.story_id AND p.project = %s",
    ),
    (
        "story_evidence",
        "DELETE FROM story_evidence c USING user_stories p"
        " WHERE c.story_id = p.story_id AND p.project = %s",
    ),
    (
        "story_entities",
        "DELETE FROM story_entities c USING user_stories p"
        " WHERE c.story_id = p.story_id AND p.project = %s",
    ),
    (
        "story_relationships",
        "DELETE FROM story_relationships c USING user_stories p"
        " WHERE c.story_id = p.story_id AND p.project = %s",
    ),
    (
        "requirement_stories",
        "DELETE FROM requirement_stories c USING user_stories p"
        " WHERE c.story_id = p.story_id AND p.project = %s",
    ),
    (
        "requirement_evidence",
        "DELETE FROM requirement_evidence c USING requirements p"
        " WHERE c.requirement_id = p.requirement_id AND p.project = %s",
    ),
    (
        "requirement_chunks",
        "DELETE FROM requirement_chunks c USING requirements p"
        " WHERE c.requirement_id = p.requirement_id AND p.project = %s",
    ),
    (
        "requirement_entities",
        "DELETE FROM requirement_entities c USING requirements p"
        " WHERE c.requirement_id = p.requirement_id AND p.project = %s",
    ),
    (
        "requirement_relationships",
        "DELETE FROM requirement_relationships c USING requirements p"
        " WHERE c.requirement_id = p.requirement_id AND p.project = %s",
    ),
    ("test_scenarios", "DELETE FROM test_scenarios WHERE project = %s"),
    ("user_stories", "DELETE FROM user_stories WHERE project = %s"),
    ("requirements", "DELETE FROM requirements WHERE project = %s"),
    ("generation_context", "DELETE FROM generation_context WHERE project = %s"),
    ("master_artifacts", "DELETE FROM master_artifacts WHERE project = %s"),
    ("knowledge_graph_readiness", "DELETE FROM knowledge_graph_readiness WHERE project = %s"),
    ("workflow_runs", "DELETE FROM workflow_runs WHERE project = %s"),
)
_CHECKPOINT_TABLES: tuple[str, ...] = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")

_BASELINE_REQUIRED_COLUMNS = {
    "requirements": {
        "requirement_id",
        "project",
        "run_id",
        "source_req_id",
        "source_req_id_type",
        "requirement_text",
        "requirement_type",
        "priority",
        "status",
        "confidence",
        "constraints_json",
    },
    "user_stories": {
        "story_id",
        "project",
        "run_id",
        "source_req_id",
        "source_req_id_type",
        "title",
        "priority",
        "persona",
        "user_story_json",
        "business_rules_json",
        "status",
        "confidence",
    },
    "test_scenarios": {
        "scenario_id",
        "project",
        "run_id",
        "source_req_id",
        "source_req_id_type",
        "title",
        "description",
        "scenario_type",
        "priority",
        "preconditions_json",
        "action",
        "expected_result",
        "status",
        "confidence",
    },
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id TEXT NOT NULL,
    project TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('started','completed','failed')),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS knowledge_graph_readiness (
    project TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('building','ready','failed')),
    build_run_id TEXT NOT NULL,
    failure_reason TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS master_artifacts (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('requirements','user_stories','test_scenarios')),
    artifact_schema_version TEXT NOT NULL,
    checksum TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, run_id, stage)
);

CREATE TABLE IF NOT EXISTS generation_context (
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    anchor_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project, run_id, stage, anchor_id)
);

CREATE TABLE IF NOT EXISTS requirements (
    requirement_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_req_id TEXT,
    source_req_id_type TEXT NOT NULL CHECK (source_req_id_type IN ('source','generated')),
    requirement_text TEXT NOT NULL,
    requirement_type TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('High','Medium','Low')),
    status TEXT NOT NULL CHECK (status = 'active'),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    constraints_json JSONB NOT NULL,
    CHECK (
      (source_req_id_type = 'source' AND source_req_id IS NOT NULL AND btrim(source_req_id) <> '')
      OR (source_req_id_type = 'generated' AND source_req_id IS NULL)
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS requirements_project_source_id
ON requirements(project, source_req_id)
WHERE source_req_id_type = 'source';

CREATE TABLE IF NOT EXISTS requirement_evidence (
    evidence_id TEXT PRIMARY KEY,
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    chunk_id TEXT NOT NULL,
    quote TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS requirement_chunks (
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (requirement_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS requirement_entities (
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    entity_id TEXT NOT NULL,
    PRIMARY KEY (requirement_id, entity_id)
);
CREATE TABLE IF NOT EXISTS requirement_relationships (
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    relationship_id TEXT NOT NULL,
    PRIMARY KEY (requirement_id, relationship_id)
);

CREATE TABLE IF NOT EXISTS user_stories (
    story_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_req_id TEXT,
    source_req_id_type TEXT NOT NULL CHECK (source_req_id_type IN ('source','generated')),
    title TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('High','Medium','Low')),
    persona TEXT NOT NULL,
    user_story_json JSONB NOT NULL,
    business_rules_json JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status = 'active'),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    CHECK (
      (source_req_id_type = 'source' AND source_req_id IS NOT NULL AND btrim(source_req_id) <> '')
      OR (source_req_id_type = 'generated' AND source_req_id IS NULL)
    )
);
CREATE TABLE IF NOT EXISTS requirement_stories (
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    PRIMARY KEY (requirement_id, story_id)
);
CREATE TABLE IF NOT EXISTS acceptance_criteria (
    criterion_id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    title TEXT NOT NULL,
    given_text TEXT NOT NULL,
    when_text TEXT NOT NULL,
    then_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS story_evidence (
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (story_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS story_entities (
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    entity_id TEXT NOT NULL,
    PRIMARY KEY (story_id, entity_id)
);
CREATE TABLE IF NOT EXISTS story_relationships (
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    relationship_id TEXT NOT NULL,
    PRIMARY KEY (story_id, relationship_id)
);

CREATE TABLE IF NOT EXISTS test_scenarios (
    scenario_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_req_id TEXT,
    source_req_id_type TEXT NOT NULL CHECK (source_req_id_type IN ('source','generated')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    scenario_type TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('High','Medium','Low')),
    preconditions_json JSONB NOT NULL,
    action TEXT NOT NULL,
    expected_result TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status = 'active'),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    CHECK (
      (source_req_id_type = 'source' AND source_req_id IS NOT NULL AND btrim(source_req_id) <> '')
      OR (source_req_id_type = 'generated' AND source_req_id IS NULL)
    )
);
CREATE TABLE IF NOT EXISTS story_scenarios (
    story_id TEXT NOT NULL REFERENCES user_stories(story_id),
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    PRIMARY KEY (story_id, scenario_id)
);
CREATE TABLE IF NOT EXISTS requirement_scenarios (
    requirement_id TEXT NOT NULL REFERENCES requirements(requirement_id),
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    PRIMARY KEY (requirement_id, scenario_id)
);
CREATE TABLE IF NOT EXISTS scenario_criteria (
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    criterion_id TEXT NOT NULL REFERENCES acceptance_criteria(criterion_id),
    PRIMARY KEY (scenario_id, criterion_id)
);
CREATE TABLE IF NOT EXISTS scenario_evidence (
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (scenario_id, chunk_id)
);
CREATE TABLE IF NOT EXISTS scenario_entities (
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    entity_id TEXT NOT NULL,
    PRIMARY KEY (scenario_id, entity_id)
);
CREATE TABLE IF NOT EXISTS scenario_relationships (
    scenario_id TEXT NOT NULL REFERENCES test_scenarios(scenario_id),
    relationship_id TEXT NOT NULL,
    PRIMARY KEY (scenario_id, relationship_id)
);
"""


class PostgresStore:
    """Persist canonical artifacts and their relational traceability."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def checkpoint_dsn(self) -> str:
        """Return the DSN used by langgraph-checkpoint-postgres."""
        return self.settings.postgres.dsn

    def check(self) -> str:
        """Validate PostgreSQL or local development storage."""
        if self.settings.postgres.mode == "local_json":
            self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS postgres local_json path={self.settings.postgres.local_path}"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return "PASS postgres connectivity"

    def ensure_schema(self) -> None:
        """Create the disposable-development baseline schema and checkpoint tables."""
        if self.settings.postgres.mode == "local_json":
            return
        with self._connect() as connection, connection.cursor() as cursor:
            self._assert_baseline_compatible(cursor)
            cursor.execute(_SCHEMA_SQL)
            connection.commit()
        self._setup_checkpoint_tables()

    def _setup_checkpoint_tables(self) -> None:
        """Idempotently provision the LangGraph checkpoint tables."""
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(self.settings.postgres.dsn) as saver:
            saver.setup()

    def delete_project(self, project: str) -> dict[str, int]:
        """Delete all project-scoped rows (FK-safe) plus its LangGraph checkpoints."""
        if self.settings.postgres.mode == "local_json":
            rows = self._local_rows()
            prefix = f"{project}:"
            kept = [
                row
                for row in rows
                if not str(row.get("key", "")).startswith(prefix)
                and (row.get("payload") or {}).get("project") != project
            ]
            _write_rows(self.settings.postgres.local_path, kept)
            return {"local_rows_deleted": len(rows) - len(kept)}
        removed: dict[str, int] = {}
        thread_prefix = f"{normalize_project(project)}:%"
        with self._connect() as connection, connection.cursor() as cursor:
            for table, statement in _PROJECT_DELETE_STEPS:
                cursor.execute(statement, (project,))
                if cursor.rowcount and cursor.rowcount > 0:
                    removed[table] = cursor.rowcount
            for table in _CHECKPOINT_TABLES:
                cursor.execute("SELECT to_regclass(%s)", (table,))
                if cursor.fetchone()[0] is None:
                    continue
                cursor.execute(
                    f"DELETE FROM {table} WHERE thread_id LIKE %s",
                    (thread_prefix,),
                )
                if cursor.rowcount and cursor.rowcount > 0:
                    removed[table] = cursor.rowcount
            connection.commit()
        return removed

    def reset_schema(self) -> str:
        """Explicitly drop and recreate managed development tables."""
        if self.settings.postgres.mode == "local_json":
            self.settings.postgres.local_path.unlink(missing_ok=True)
            return "PASS postgres local_json reset"
        with self._connect() as connection, connection.cursor() as cursor:
            constraint_count, table_count = self._drop_disposable_schema(cursor)
            cursor.execute(_SCHEMA_SQL)
            connection.commit()
        self._setup_checkpoint_tables()
        return (
            "PASS postgres simplified schema reset "
            f"constraints_dropped={constraint_count} tables_dropped={table_count}"
        )

    @contextmanager
    def discovery_lease(self, project: str) -> Iterator[None]:
        """Serialize requirement discovery for a project."""
        if self.settings.postgres.mode == "local_json":
            yield
            return
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(hashtext(%s))", (f"discovery:{project}",))
            yield
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(hashtext(%s))", (f"discovery:{project}",))
            connection.close()

    @contextmanager
    def project_maintenance_lease(self, project: str) -> Iterator[None]:
        """Acquire a non-blocking project reset lease or fail without waiting."""
        if self.settings.postgres.mode == "local_json":
            yield
            return
        connection = self._connect()
        lock_name = f"maintenance:{project}"
        acquired = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (lock_name,))
                row = cursor.fetchone()
                acquired = bool(row and row[0])
            if not acquired:
                raise RuntimeError(f"project maintenance lease is already held: {project}")
            yield
        finally:
            if acquired:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(hashtext(%s))", (lock_name,))
            connection.close()

    def record_run(
        self,
        *,
        run_id: str,
        project: str,
        stage: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        """Upsert operational run state."""
        row = {
            "run_id": run_id,
            "project": project,
            "stage": stage,
            "status": status,
            "payload": payload,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if self.settings.postgres.mode == "local_json":
            self._upsert_local("run", f"{run_id}:{stage}", row)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO workflow_runs(run_id, project, stage, status, payload)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (run_id, stage) DO UPDATE
                SET status=EXCLUDED.status, payload=EXCLUDED.payload, updated_at=now()
                """,
                (run_id, project, stage, status, json.dumps(payload)),
            )
            connection.commit()

    def set_readiness(self, readiness: KnowledgeGraphReadiness) -> None:
        """Upsert the project readiness gate."""
        payload = readiness.model_dump(mode="json")
        if self.settings.postgres.mode == "local_json":
            self._upsert_local("readiness", readiness.project, payload)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO knowledge_graph_readiness(
                    project,status,build_run_id,failure_reason,updated_at
                ) VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (project) DO UPDATE SET
                    status=EXCLUDED.status,
                    build_run_id=EXCLUDED.build_run_id,
                    failure_reason=EXCLUDED.failure_reason,
                    updated_at=EXCLUDED.updated_at
                """,
                (
                    readiness.project,
                    readiness.status,
                    readiness.build_run_id,
                    readiness.failure_reason,
                    readiness.updated_at,
                ),
            )
            connection.commit()

    def get_readiness(self, project: str) -> KnowledgeGraphReadiness | None:
        """Load the project readiness gate."""
        if self.settings.postgres.mode == "local_json":
            payload = self._read_local("readiness", project)
            return KnowledgeGraphReadiness.model_validate(payload) if payload else None
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT project,status,build_run_id,failure_reason,updated_at
                FROM knowledge_graph_readiness WHERE project=%s
                """,
                (project,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return KnowledgeGraphReadiness(
            project=row[0],
            status=row[1],
            build_run_id=row[2],
            failure_reason=row[3],
            updated_at=row[4],
        )

    def persist_requirements(self, artifact: RequirementsArtifact) -> None:
        """Transactionally persist canonical requirements and traceability."""
        if self.settings.postgres.mode == "local_json":
            self._persist_artifact_local("requirements", artifact)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            self._upsert_master(cursor, "requirements", artifact)
            for requirement in artifact.requirements:
                cursor.execute(
                    """
                    INSERT INTO requirements(
                      requirement_id,project,run_id,source_req_id,source_req_id_type,
                      requirement_text,requirement_type,priority,status,confidence,constraints_json
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (requirement_id) DO UPDATE SET
                      run_id=EXCLUDED.run_id, source_req_id=EXCLUDED.source_req_id,
                      source_req_id_type=EXCLUDED.source_req_id_type,
                      requirement_text=EXCLUDED.requirement_text,
                      requirement_type=EXCLUDED.requirement_type,
                      priority=EXCLUDED.priority, confidence=EXCLUDED.confidence,
                      constraints_json=EXCLUDED.constraints_json
                    """,
                    (
                        requirement.requirement_id,
                        artifact.project,
                        artifact.run_id,
                        requirement.source_req_id,
                        requirement.source_req_id_type,
                        requirement.requirement_text,
                        requirement.requirement_type,
                        requirement.priority,
                        requirement.status,
                        requirement.confidence,
                        json.dumps(requirement.constraints),
                    ),
                )
                for evidence in requirement.evidence:
                    cursor.execute(
                        """
                        INSERT INTO requirement_evidence(
                          evidence_id,requirement_id,chunk_id,quote,start_char,end_char
                        ) VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (evidence_id) DO UPDATE SET
                          quote=EXCLUDED.quote,start_char=EXCLUDED.start_char,
                          end_char=EXCLUDED.end_char
                        """,
                        (
                            evidence.evidence_id,
                            requirement.requirement_id,
                            evidence.chunk_id,
                            evidence.quote,
                            evidence.start_char,
                            evidence.end_char,
                        ),
                    )
                    cursor.execute(
                        """
                        INSERT INTO requirement_chunks(requirement_id,chunk_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (requirement.requirement_id, evidence.chunk_id),
                    )
                for entity_id in requirement.entity_ids:
                    cursor.execute(
                        """
                        INSERT INTO requirement_entities(requirement_id,entity_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (requirement.requirement_id, entity_id),
                    )
                for relationship_id in requirement.relationship_ids:
                    cursor.execute(
                        """
                        INSERT INTO requirement_relationships(requirement_id,relationship_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (requirement.requirement_id, relationship_id),
                    )
            connection.commit()

    def persist_user_stories(self, artifact: UserStoriesArtifact) -> None:
        """Transactionally persist stories, criteria, and traceability."""
        if self.settings.postgres.mode == "local_json":
            self._persist_artifact_local("user_stories", artifact)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            self._upsert_master(cursor, "user_stories", artifact)
            for story in artifact.stories:
                cursor.execute(
                    """
                    INSERT INTO user_stories(
                      story_id,project,run_id,source_req_id,source_req_id_type,title,priority,
                      persona,user_story_json,business_rules_json,status,confidence
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
                    ON CONFLICT (story_id) DO UPDATE SET
                      run_id=EXCLUDED.run_id,title=EXCLUDED.title,priority=EXCLUDED.priority,
                      persona=EXCLUDED.persona,user_story_json=EXCLUDED.user_story_json,
                      business_rules_json=EXCLUDED.business_rules_json,
                      confidence=EXCLUDED.confidence
                    """,
                    (
                        story.story_id,
                        artifact.project,
                        artifact.run_id,
                        story.source_req_id,
                        story.source_req_id_type,
                        story.title,
                        story.priority,
                        story.persona,
                        json.dumps(story.user_story.model_dump(mode="json")),
                        json.dumps(story.business_rules),
                        story.status,
                        story.confidence,
                    ),
                )
                for requirement_id in story.requirement_ids:
                    cursor.execute(
                        """
                        INSERT INTO requirement_stories(requirement_id,story_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (requirement_id, story.story_id),
                    )
                for criterion in story.acceptance_criteria:
                    cursor.execute(
                        """
                        INSERT INTO acceptance_criteria(
                          criterion_id,story_id,title,given_text,when_text,then_text
                        ) VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (criterion_id) DO UPDATE SET
                          title=EXCLUDED.title,given_text=EXCLUDED.given_text,
                          when_text=EXCLUDED.when_text,then_text=EXCLUDED.then_text
                        """,
                        (
                            criterion.criterion_id,
                            story.story_id,
                            criterion.title,
                            criterion.given,
                            criterion.when,
                            criterion.then,
                        ),
                    )
                self._persist_traceability(cursor, "story", story.story_id, story.traceability)
            connection.commit()

    def persist_test_scenarios(self, artifact: TestScenariosArtifact) -> None:
        """Transactionally persist scenarios and traceability."""
        if self.settings.postgres.mode == "local_json":
            self._persist_artifact_local("test_scenarios", artifact)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            self._upsert_master(cursor, "test_scenarios", artifact)
            for scenario in artifact.scenarios:
                cursor.execute(
                    """
                    INSERT INTO test_scenarios(
                      scenario_id,project,run_id,source_req_id,source_req_id_type,title,
                      description,scenario_type,priority,preconditions_json,action,
                      expected_result,status,confidence
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                    ON CONFLICT (scenario_id) DO UPDATE SET
                      run_id=EXCLUDED.run_id,title=EXCLUDED.title,description=EXCLUDED.description,
                      scenario_type=EXCLUDED.scenario_type,priority=EXCLUDED.priority,
                      preconditions_json=EXCLUDED.preconditions_json,action=EXCLUDED.action,
                      expected_result=EXCLUDED.expected_result,confidence=EXCLUDED.confidence
                    """,
                    (
                        scenario.scenario_id,
                        artifact.project,
                        artifact.run_id,
                        scenario.source_req_id,
                        scenario.source_req_id_type,
                        scenario.title,
                        scenario.description,
                        scenario.scenario_type,
                        scenario.priority,
                        json.dumps(scenario.preconditions),
                        scenario.action,
                        scenario.expected_result,
                        scenario.status,
                        scenario.confidence,
                    ),
                )
                for story_id in scenario.story_ids:
                    cursor.execute(
                        """
                        INSERT INTO story_scenarios(story_id,scenario_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (story_id, scenario.scenario_id),
                    )
                for requirement_id in scenario.requirement_ids:
                    cursor.execute(
                        """
                        INSERT INTO requirement_scenarios(requirement_id,scenario_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (requirement_id, scenario.scenario_id),
                    )
                for criterion_id in scenario.covered_acceptance_criterion_ids:
                    cursor.execute(
                        """
                        INSERT INTO scenario_criteria(scenario_id,criterion_id)
                        VALUES (%s,%s) ON CONFLICT DO NOTHING
                        """,
                        (scenario.scenario_id, criterion_id),
                    )
                self._persist_traceability(
                    cursor, "scenario", scenario.scenario_id, scenario.traceability
                )
            connection.commit()

    def load_requirements(self, project: str, run_id: str) -> RequirementsArtifact | None:
        """Reconstruct the canonical requirements artifact from PostgreSQL payload."""
        payload = self._load_master(project, run_id, "requirements")
        return RequirementsArtifact.model_validate(payload) if payload else None

    def load_project_requirements(self, project: str) -> RequirementsArtifact | None:
        """Load all project requirement identities for stable reuse across runs."""
        artifacts = [
            RequirementsArtifact.model_validate(payload)
            for payload in self._load_project_masters(project, "requirements")
        ]
        if not artifacts:
            return None
        requirements = {
            item.requirement_id: item for artifact in artifacts for item in artifact.requirements
        }
        entities = {item.entity_id: item for artifact in artifacts for item in artifact.entities}
        relationships = {
            item.relationship_id: item for artifact in artifacts for item in artifact.relationships
        }
        payload = RequirementsArtifact.model_construct(
            project=project,
            run_id=artifacts[-1].run_id,
            checksum="",
            requirements=list(requirements.values()),
            entities=list(entities.values()),
            relationships=list(relationships.values()),
        )
        return RequirementsArtifact.model_validate(
            {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
        )

    def load_user_stories(self, project: str, run_id: str) -> UserStoriesArtifact | None:
        """Reconstruct the canonical user-stories artifact."""
        payload = self._load_master(project, run_id, "user_stories")
        return UserStoriesArtifact.model_validate(payload) if payload else None

    def load_project_user_stories(self, project: str) -> UserStoriesArtifact | None:
        """Load all project story identities for permanent-ID reuse."""
        artifacts = [
            UserStoriesArtifact.model_validate(payload)
            for payload in self._load_project_masters(project, "user_stories")
        ]
        if not artifacts:
            return None
        stories = {item.story_id: item for artifact in artifacts for item in artifact.stories}
        payload = UserStoriesArtifact.model_construct(
            project=project,
            run_id=artifacts[-1].run_id,
            checksum="",
            stories=list(stories.values()),
        )
        return UserStoriesArtifact.model_validate(
            {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
        )

    def load_test_scenarios(self, project: str, run_id: str) -> TestScenariosArtifact | None:
        """Reconstruct the canonical test-scenarios artifact."""
        payload = self._load_master(project, run_id, "test_scenarios")
        return TestScenariosArtifact.model_validate(payload) if payload else None

    def load_project_test_scenarios(self, project: str) -> TestScenariosArtifact | None:
        """Load all project scenario identities for permanent-ID reuse."""
        artifacts = [
            TestScenariosArtifact.model_validate(payload)
            for payload in self._load_project_masters(project, "test_scenarios")
        ]
        if not artifacts:
            return None
        scenarios = {
            item.scenario_id: item for artifact in artifacts for item in artifact.scenarios
        }
        payload = TestScenariosArtifact.model_construct(
            project=project,
            run_id=artifacts[-1].run_id,
            checksum="",
            scenarios=list(scenarios.values()),
        )
        return TestScenariosArtifact.model_validate(
            {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
        )

    def save_generation_context(
        self,
        *,
        project: str,
        run_id: str,
        stage: str,
        anchor_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist compact context for audit and resume."""
        if self.settings.postgres.mode == "local_json":
            self._upsert_local("context", f"{project}:{run_id}:{stage}:{anchor_id}", payload)
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO generation_context(project,run_id,stage,anchor_id,payload)
                VALUES (%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (project,run_id,stage,anchor_id) DO UPDATE
                SET payload=EXCLUDED.payload,updated_at=now()
                """,
                (project, run_id, stage, anchor_id, json.dumps(payload)),
            )
            connection.commit()

    def coverage(self, project: str, run_id: str) -> CoverageSummary:
        """Return current project/run coverage without lifecycle masking."""
        requirements = self.load_requirements(project, run_id)
        stories = self.load_user_stories(project, run_id)
        scenarios = self.load_test_scenarios(project, run_id)
        requirement_ids = (
            {item.requirement_id for item in requirements.requirements} if requirements else set()
        )
        story_rows = stories.stories if stories else []
        scenario_rows = scenarios.scenarios if scenarios else []
        covered_requirements = {
            requirement_id for story in story_rows for requirement_id in story.requirement_ids
        }
        covered_stories = {
            story_id for scenario in scenario_rows for story_id in scenario.story_ids
        }
        return CoverageSummary(
            project=project,
            run_id=run_id,
            requirement_count=len(requirement_ids),
            story_count=len(story_rows),
            scenario_count=len(scenario_rows),
            requirements_with_stories=len(requirement_ids & covered_requirements),
            stories_with_scenarios=len({story.story_id for story in story_rows} & covered_stories),
        )

    def _persist_traceability(self, cursor: Any, prefix: str, item_id: str, trace: Any) -> None:
        id_column = f"{prefix}_id"
        for chunk_id in trace.evidence_chunk_ids:
            cursor.execute(
                f"INSERT INTO {prefix}_evidence({id_column},chunk_id) "
                "VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (item_id, chunk_id),
            )
        for entity_id in trace.entity_ids:
            cursor.execute(
                f"INSERT INTO {prefix}_entities({id_column},entity_id) "
                "VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (item_id, entity_id),
            )
        for relationship_id in trace.relationship_ids:
            cursor.execute(
                f"INSERT INTO {prefix}_relationships({id_column},relationship_id) "
                "VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (item_id, relationship_id),
            )

    def _upsert_master(self, cursor: Any, stage: str, artifact: Any) -> None:
        payload = artifact.model_dump(mode="json")
        cursor.execute(
            """
            INSERT INTO master_artifacts(
              project,run_id,stage,artifact_schema_version,checksum,payload
            ) VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (project,run_id,stage) DO UPDATE SET
              artifact_schema_version=EXCLUDED.artifact_schema_version,
              checksum=EXCLUDED.checksum,payload=EXCLUDED.payload,updated_at=now()
            """,
            (
                artifact.project,
                artifact.run_id,
                stage,
                artifact.artifact_schema_version,
                artifact.checksum,
                json.dumps(payload),
            ),
        )

    def _load_master(self, project: str, run_id: str, stage: str) -> dict[str, Any] | None:
        if self.settings.postgres.mode == "local_json":
            return self._read_local("artifact", f"{project}:{run_id}:{stage}")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM master_artifacts
                WHERE project=%s AND run_id=%s AND stage=%s
                """,
                (project, run_id, stage),
            )
            row = cursor.fetchone()
        return dict(row[0]) if row else None

    def _load_project_masters(self, project: str, stage: str) -> list[dict[str, Any]]:
        if self.settings.postgres.mode == "local_json":
            candidates = [
                row
                for row in self._local_rows()
                if row.get("kind") == "artifact"
                and str(row.get("key", "")).startswith(f"{project}:")
                and str(row.get("key", "")).endswith(f":{stage}")
            ]
            return [dict(row["payload"]) for row in candidates]
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM master_artifacts
                WHERE project=%s AND stage=%s
                ORDER BY updated_at
                """,
                (project, stage),
            )
            rows = cursor.fetchall()
        return [dict(row[0]) for row in rows]

    def _persist_artifact_local(self, stage: str, artifact: Any) -> None:
        self._upsert_local(
            "artifact",
            f"{artifact.project}:{artifact.run_id}:{stage}",
            artifact.model_dump(mode="json"),
        )

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self.settings.postgres.dsn)

    def _assert_baseline_compatible(self, cursor: Any) -> None:
        """Reject a legacy schema instead of attempting an implicit migration."""
        cursor.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            """,
            (list(_BASELINE_REQUIRED_COLUMNS),),
        )
        actual: dict[str, set[str]] = {}
        for table_name, column_name in cursor.fetchall():
            actual.setdefault(str(table_name), set()).add(str(column_name))
        incompatible = {
            table: sorted(required - actual[table])
            for table, required in _BASELINE_REQUIRED_COLUMNS.items()
            if table in actual and not required <= actual[table]
        }
        if incompatible:
            raise RuntimeError(
                "legacy PostgreSQL schema detected; no implicit migration is permitted. "
                "Confirm the database is disposable, back it up, then run "
                "`python -m multi_agentic_graph_rag postgres-reset --yes`. "
                f"missing simplified columns={incompatible}"
            )

    def _drop_disposable_schema(self, cursor: Any) -> tuple[int, int]:
        """Drop every table after explicitly removing its foreign-key constraints."""
        from psycopg import sql

        cursor.execute(
            """
            SELECT n.nspname, c.relname, constraint_row.conname
            FROM pg_constraint AS constraint_row
            JOIN pg_class AS c ON c.oid = constraint_row.conrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE constraint_row.contype = 'f'
              AND n.nspname = current_schema()
            ORDER BY c.relname, constraint_row.conname
            """
        )
        constraints = [
            (str(schema), str(table), str(constraint))
            for schema, table, constraint in cursor.fetchall()
        ]
        for schema, table, constraint in constraints:
            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} DROP CONSTRAINT {}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.Identifier(constraint),
                )
            )
        cursor.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )
        tables = [(str(schema), str(table)) for schema, table in cursor.fetchall()]
        for schema, table in tables:
            cursor.execute(
                sql.SQL("DROP TABLE {}.{}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                )
            )
        return len(constraints), len(tables)

    def _local_rows(self) -> list[dict[str, Any]]:
        path = self.settings.postgres.local_path
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        rows = self._local_rows()
        row = {"kind": kind, "key": key, "payload": payload}
        for index, existing in enumerate(rows):
            if existing.get("kind") == kind and existing.get("key") == key:
                rows[index] = row
                break
        else:
            rows.append(row)
        _write_rows(self.settings.postgres.local_path, rows)

    def _read_local(self, kind: str, key: str) -> dict[str, Any] | None:
        for row in self._local_rows():
            if row.get("kind") == kind and row.get("key") == key:
                return dict(row.get("payload") or {})
        return None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["PostgresStore"]
