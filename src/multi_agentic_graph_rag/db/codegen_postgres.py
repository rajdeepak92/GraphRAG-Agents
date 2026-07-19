"""Stage-4 PostgreSQL persistence for codegen runs, test cases, and evidence.

Kept separate from the Stage 1-3 ``PostgresStore`` (plan §17) and additive: it
introduces the Stage-4 tables (§15.4) without touching the baseline schema. A
``local_json`` fallback mirrors the existing stores so Stage-4 persistence is
testable without a live server, and ``rebuild_test_cases_artifact`` regenerates
``test-cases.json`` from the durable records (§15.6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    ContextManifest,
    TestCaseRecord,
    TestCasesArtifact,
    canonical_checksum,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS codegen_runs (
    codegen_run_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    run_id TEXT NOT NULL,
    framework_snapshot_id TEXT NOT NULL,
    test_data_snapshot_id TEXT NOT NULL,
    execution_profile_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS test_cases (
    test_case_id TEXT NOT NULL,
    test_case_revision INTEGER NOT NULL,
    project TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    execution_profile_id TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    payload JSONB NOT NULL,
    payload_checksum TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (test_case_id, test_case_revision)
);

CREATE TABLE IF NOT EXISTS codegen_context_manifests (
    manifest_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    checksum TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS codegen_blockers (
    blocker_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    blocker_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class CodegenPostgresStore:
    """Persist and rebuild Stage-4 artifacts with a local_json fallback."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def _local(self) -> bool:
        return self.settings.postgres.mode == "local_json"

    @property
    def _local_path(self) -> Path:
        return self.settings.stage4.codegen_local_path

    def ensure_schema(self) -> None:
        """Create the additive Stage-4 tables (no baseline changes)."""
        if self._local:
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(_SCHEMA_SQL)
            conn.commit()

    def save_codegen_run(
        self,
        *,
        codegen_run_id: str,
        project: str,
        run_id: str,
        framework_snapshot_id: str,
        test_data_snapshot_id: str,
        execution_profile_id: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Idempotently persist a codegen-run header row."""
        row = {
            "codegen_run_id": codegen_run_id,
            "project": project,
            "run_id": run_id,
            "framework_snapshot_id": framework_snapshot_id,
            "test_data_snapshot_id": test_data_snapshot_id,
            "execution_profile_id": execution_profile_id,
            "status": status,
            "payload": payload or {},
        }
        if self._local:
            self._upsert_local("codegen_run", codegen_run_id, row)
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO codegen_runs(codegen_run_id, project, run_id, framework_snapshot_id,
                    test_data_snapshot_id, execution_profile_id, status, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (codegen_run_id) DO UPDATE
                SET status = EXCLUDED.status, payload = EXCLUDED.payload, updated_at = now()
                """,
                (
                    codegen_run_id,
                    project,
                    run_id,
                    framework_snapshot_id,
                    test_data_snapshot_id,
                    execution_profile_id,
                    status,
                    json.dumps(payload or {}),
                ),
            )
            conn.commit()

    def save_test_case(self, project: str, test_case: TestCaseRecord) -> None:
        """Persist one test-case revision (reuses identity across regenerations)."""
        payload = test_case.model_dump(mode="json")
        checksum = canonical_checksum(payload)
        if self._local:
            key = f"{test_case.test_case_id}:{test_case.test_case_revision}"
            self._upsert_local(
                "test_case",
                key,
                {"project": project, "payload": payload, "payload_checksum": checksum},
            )
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO test_cases(test_case_id, test_case_revision, project, scenario_id,
                    execution_profile_id, validation_status, payload, payload_checksum)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (test_case_id, test_case_revision) DO UPDATE
                SET payload = EXCLUDED.payload, payload_checksum = EXCLUDED.payload_checksum,
                    validation_status = EXCLUDED.validation_status, updated_at = now()
                """,
                (
                    test_case.test_case_id,
                    test_case.test_case_revision,
                    project,
                    test_case.scenario_id,
                    test_case.execution_profile_id,
                    test_case.validation_status,
                    json.dumps(payload),
                    checksum,
                ),
            )
            conn.commit()

    def save_blocker(self, project: str, blocker: CodegenBlocker) -> None:
        """Persist a structured blocker (never silently dropped)."""
        payload = blocker.model_dump(mode="json")
        if self._local:
            self._upsert_local(
                "codegen_blocker", blocker.blocker_id, {"project": project, "payload": payload}
            )
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO codegen_blockers(
                    blocker_id, project, scenario_id, blocker_type, payload)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (blocker_id) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (
                    blocker.blocker_id,
                    project,
                    blocker.scenario_id,
                    blocker.blocker_type,
                    json.dumps(payload),
                ),
            )
            conn.commit()

    def save_context_manifest(self, project: str, manifest: ContextManifest) -> None:
        """Persist a context manifest for reproducibility (plan §10.4)."""
        payload = manifest.model_dump(mode="json")
        if self._local:
            self._upsert_local(
                "context_manifest",
                manifest.manifest_id,
                {"project": project, "payload": payload},
            )
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO codegen_context_manifests(manifest_id, project, scenario_id,
                    checksum, payload)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (manifest_id) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (
                    manifest.manifest_id,
                    project,
                    manifest.scenario_id,
                    manifest.checksum,
                    json.dumps(payload),
                ),
            )
            conn.commit()

    def rebuild_test_cases_artifact(self, project: str, run_id: str) -> TestCasesArtifact:
        """Regenerate test-cases.json from durable records with full traceability (§15.6)."""
        test_cases = self._load_test_cases(project)
        draft = TestCasesArtifact.model_construct(
            project=project, run_id=run_id, checksum="", test_cases=test_cases
        )
        return TestCasesArtifact.model_validate(
            {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
        )

    def _load_test_cases(self, project: str) -> list[TestCaseRecord]:
        if self._local:
            latest: dict[str, TestCaseRecord] = {}
            for row in self._local_rows():
                if row.get("kind") != "test_case" or row.get("project") != project:
                    continue
                record = TestCaseRecord.model_validate(row["payload"])
                existing = latest.get(record.test_case_id)
                if existing is None or record.test_case_revision > existing.test_case_revision:
                    latest[record.test_case_id] = record
            return sorted(latest.values(), key=lambda tc: tc.test_case_id)
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (test_case_id) payload
                FROM test_cases WHERE project = %s
                ORDER BY test_case_id, test_case_revision DESC
                """,
                (project,),
            )
            return [TestCaseRecord.model_validate(row[0]) for row in cursor.fetchall()]

    # --- connection + local helpers -----------------------------------------

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self.settings.postgres.dsn)

    def _local_rows(self) -> list[dict[str, Any]]:
        path = self._local_path
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        rows = self._local_rows()
        row = {"kind": kind, "key": key, **payload}
        for index, existing in enumerate(rows):
            if existing.get("kind") == kind and existing.get("key") == key:
                rows[index] = row
                break
        else:
            rows.append(row)
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_path.write_text(
            "".join(json.dumps(entry, sort_keys=True, default=str) + "\n" for entry in rows),
            encoding="utf-8",
        )


__all__ = ["CodegenPostgresStore"]
