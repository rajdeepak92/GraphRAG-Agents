"""Stage-4 PostgreSQL persistence for codegen runs, test cases, and evidence.

Kept separate from the Stage 1-3 ``PostgresStore`` (plan §17) and additive: it
introduces the Stage-4 tables (§15.4) without touching the baseline schema. A
``local_json`` fallback mirrors the existing stores so Stage-4 persistence is
testable without a live server, and ``rebuild_test_cases_artifact`` regenerates
``test-cases.json`` from the durable records (§15.6).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    ContextManifest,
    FrozenRunManifest,
    LogicalTestCaseKey,
    Stage4TestCaseRecord,
    TestCaseRecord,
    TestCasesArtifact,
    TestDataSnapshotRef,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.errors import (
    ConfigurationError,
    InputManifestChanged,
    RevisionRequired,
    Stage4PersistenceConflict,
    TcIdCapacityExhausted,
)
from multi_agentic_graph_rag.domain.identifiers import TC_ID_MAX, TC_ID_MIN, normalize_project

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

# Authoritative six-digit TC identity allocator (plan §8.2). The sequence is the
# single source of TC numbers; local JSON never invents or renumbers an ID.
_TC_SEQUENCE_SQL = f"""
CREATE SEQUENCE IF NOT EXISTS stage4_tc_id_seq
    AS integer
    START WITH {TC_ID_MIN}
    INCREMENT BY 1
    MINVALUE {TC_ID_MIN}
    MAXVALUE {TC_ID_MAX}
    NO CYCLE
    CACHE 1;

CREATE TABLE IF NOT EXISTS stage4_tc_reservations (
    tc_id INTEGER PRIMARY KEY,
    project_name TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    execution_profile_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    logical_key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_name, scenario_id, execution_profile_id, variant_id)
);

CREATE TABLE IF NOT EXISTS stage4_test_cases (
    tc_id INTEGER PRIMARY KEY REFERENCES stage4_tc_reservations(tc_id),
    project_name TEXT NOT NULL,
    logical_key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL,
    payload_checksum TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage4_tc_steps (
    tc_id INTEGER NOT NULL REFERENCES stage4_test_cases(tc_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (tc_id, step)
);

CREATE TABLE IF NOT EXISTS stage4_run_test_cases (
    project_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tc_id INTEGER NOT NULL REFERENCES stage4_test_cases(tc_id),
    input_fingerprint TEXT NOT NULL,
    generated_files_checksum TEXT NOT NULL,
    PRIMARY KEY (project_name, run_id, tc_id)
);

CREATE TABLE IF NOT EXISTS stage4_frozen_run_manifests (
    project_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    checksum TEXT NOT NULL,
    provider_fingerprint_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_name, run_id)
);

CREATE TABLE IF NOT EXISTS stage4_test_data_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    workbook_checksum TEXT NOT NULL,
    normalized_checksum TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage4_file_journals (
    project_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tc_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    journal_path TEXT NOT NULL,
    payload_checksum TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_name, run_id, tc_id)
);

CREATE TABLE IF NOT EXISTS stage4_kg_publication_attempts (
    attempt_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tc_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    previous_snapshot_id TEXT,
    proposed_snapshot_id TEXT,
    payload_checksum TEXT NOT NULL,
    payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage4_idempotency_records (
    idempotency_key TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    tc_id INTEGER NOT NULL,
    node_name TEXT NOT NULL,
    input_checksum TEXT NOT NULL,
    result_checksum TEXT NOT NULL,
    result_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# The exact sequence contract every migration must re-verify (plan §8.2): a bare
# ``CREATE SEQUENCE IF NOT EXISTS`` silently accepts a pre-existing sequence with
# the wrong bounds, so the start/min/max/cycle/type are inspected explicitly.
_TC_SEQUENCE_CONTRACT = {
    "data_type": "integer",
    "start_value": str(TC_ID_MIN),
    "minimum_value": str(TC_ID_MIN),
    "maximum_value": str(TC_ID_MAX),
    "increment": "1",
    "cycle_option": "NO",
    "cache_size": "1",
}


def _is_sequence_exhausted(exc: Exception) -> bool:
    """Detect PostgreSQL ``sequence_generator_limit_exceeded`` (SQLSTATE 2200H)."""
    sqlstate = getattr(exc, "sqlstate", None) or getattr(
        getattr(exc, "diag", None), "sqlstate", None
    )
    if sqlstate == "2200H":
        return True
    return "reached maximum value" in str(exc).lower()


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
        """Create the additive Stage-4 tables + TC sequence (no baseline changes)."""
        if self._local:
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(_SCHEMA_SQL)
            cursor.execute(_TC_SEQUENCE_SQL)
            self._verify_sequence_contract(cursor)
            conn.commit()

    def _verify_sequence_contract(self, cursor: Any) -> None:
        """Fail migration if an existing sequence violates the §8.2 contract."""
        cursor.execute(
            """
            SELECT data_type::text, start_value::text, min_value::text, max_value::text,
                   increment_by::text,
                   CASE WHEN cycle THEN 'YES' ELSE 'NO' END,
                   cache_size::text
            FROM pg_catalog.pg_sequences
            WHERE schemaname = current_schema()
              AND sequencename = 'stage4_tc_id_seq'
            """
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("stage4_tc_id_seq missing after migration")
        actual = dict(
            zip(
                (
                    "data_type",
                    "start_value",
                    "minimum_value",
                    "maximum_value",
                    "increment",
                    "cycle_option",
                    "cache_size",
                ),
                (str(value) for value in row),
                strict=True,
            )
        )
        mismatches = {
            key: (actual[key], expected)
            for key, expected in _TC_SEQUENCE_CONTRACT.items()
            if actual[key] != expected
        }
        if mismatches:
            raise RuntimeError(
                f"stage4_tc_id_seq contract violation: {mismatches}; "
                "IF NOT EXISTS is not sufficient validation (plan §8.2)"
            )

    def allocate_or_get_tc_id(self, key: LogicalTestCaseKey) -> tuple[int, bool]:
        """Return ``(tc_id, created)`` for a logical key; reuse an existing reservation.

        Transactional and idempotent (plan §8.2): the same logical key always maps
        to the same permanent TC ID. Exhausting ``999999`` raises
        ``TcIdCapacityExhausted`` — never wraparound or a random fallback.
        """
        if self._local:
            return self._allocate_local(key)
        return self._allocate_postgres(key)

    def _allocate_local(self, key: LogicalTestCaseKey) -> tuple[int, bool]:
        """Read a mirrored reservation; local JSON is never an allocator."""
        rows = self._local_rows()
        for row in rows:
            if row.get("kind") != "tc_reservation":
                continue
            payload = row.get("payload", {})
            if payload.get("logical_key_hash") == key.logical_key_hash:
                return int(payload["tc_id"]), False
        raise ConfigurationError(
            "PostgreSQL is authoritative for Stage-4 TC allocation; "
            "local_json contains no mirrored reservation for this logical key"
        )

    def mirror_tc_reservation(
        self, tc_id: int, key: LogicalTestCaseKey, *, status: str = "RESERVED"
    ) -> None:
        """Mirror an already-authoritative PostgreSQL reservation for local testing/review."""
        if not self._local:
            raise ConfigurationError("reservation mirroring is available only in local_json mode")
        if not TC_ID_MIN <= tc_id <= TC_ID_MAX:
            raise ValueError(f"tc_id {tc_id} outside {TC_ID_MIN}..{TC_ID_MAX}")
        for row in self._local_rows():
            if row.get("kind") != "tc_reservation":
                continue
            payload = row.get("payload", {})
            existing_tc_id = int(payload.get("tc_id", 0))
            existing_hash = str(payload.get("logical_key_hash", ""))
            if existing_tc_id == tc_id and existing_hash != key.logical_key_hash:
                raise Stage4PersistenceConflict(f"TC {tc_id} is mirrored for another logical key")
            if existing_hash == key.logical_key_hash and existing_tc_id != tc_id:
                raise Stage4PersistenceConflict(
                    "logical key is mirrored with a different authoritative TC ID"
                )
        self._upsert_local(
            "tc_reservation",
            str(tc_id),
            {
                "payload": {
                    "tc_id": tc_id,
                    "project_name": key.project_name,
                    "scenario_id": key.scenario_id,
                    "execution_profile_id": key.execution_profile_id,
                    "variant_id": key.variant_id,
                    "logical_key_hash": key.logical_key_hash,
                    "status": status,
                }
            },
        )

    def _allocate_postgres(self, key: LogicalTestCaseKey) -> tuple[int, bool]:
        with self._connect() as conn, conn.cursor() as cursor:
            existing = self._read_reservation(cursor, key.logical_key_hash)
            if existing is not None:
                return existing, False
            try:
                cursor.execute("SELECT nextval('stage4_tc_id_seq')")
                tc_id = int(cursor.fetchone()[0])
            except Exception as exc:  # inspect SQLSTATE below
                conn.rollback()
                if _is_sequence_exhausted(exc):
                    raise TcIdCapacityExhausted(f"TC id sequence exhausted at {TC_ID_MAX}") from exc
                raise
            try:
                cursor.execute(
                    """
                    INSERT INTO stage4_tc_reservations(tc_id, project_name, scenario_id,
                        execution_profile_id, variant_id, logical_key_hash, status)
                    VALUES (%s,%s,%s,%s,%s,%s,'RESERVED')
                    """,
                    (
                        tc_id,
                        key.project_name,
                        key.scenario_id,
                        key.execution_profile_id,
                        key.variant_id,
                        key.logical_key_hash,
                    ),
                )
                conn.commit()
                return tc_id, True
            except Exception:  # uniqueness race, read the winner
                conn.rollback()
                winner = self._read_reservation(cursor, key.logical_key_hash)
                if winner is None:
                    raise
                return winner, False

    @staticmethod
    def _read_reservation(cursor: Any, logical_key_hash: str) -> int | None:
        cursor.execute(
            "SELECT tc_id FROM stage4_tc_reservations WHERE logical_key_hash = %s",
            (logical_key_hash,),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else None

    def save_stage4_test_case(
        self, record: Stage4TestCaseRecord, *, run_id: str | None = None
    ) -> bool:
        """Persist a legal TC state transition without ever overwriting acceptance.

        Returns ``True`` when the master record changed.  Replaying the exact
        accepted record returns ``False`` (and may still associate it with a
        resumed run).  Any attempted mutation of an accepted record raises
        ``RevisionRequired``.
        """
        if self._local:
            return self._save_stage4_test_case_local(record, run_id=run_id)
        return self._save_stage4_test_case_postgres(record, run_id=run_id)

    def _save_stage4_test_case_local(
        self, record: Stage4TestCaseRecord, *, run_id: str | None
    ) -> bool:
        reservation = self._local_row("tc_reservation", str(record.tc_id))
        if reservation is None:
            raise Stage4PersistenceConflict(
                f"TC {record.tc_id} has no mirrored authoritative reservation"
            )
        reservation_payload = dict(reservation.get("payload", {}))
        self._validate_reservation(record, reservation_payload)

        existing_row = self._local_row("stage4_test_case", str(record.tc_id))
        existing = (
            Stage4TestCaseRecord.model_validate(existing_row["payload"])
            if existing_row is not None
            else None
        )
        changed = self._assert_test_case_mutation(existing, record)
        if changed:
            payload = record.model_dump(mode="json")
            self._upsert_local(
                "stage4_test_case",
                str(record.tc_id),
                {
                    "project": record.project_name,
                    "payload": payload,
                    "payload_checksum": canonical_checksum({"payload": payload}),
                },
            )
            reservation_payload["status"] = record.status
            reservation_payload["payload"] = payload
            self._upsert_local(
                "tc_reservation", str(record.tc_id), {"payload": reservation_payload}
            )
        if record.status == "ACCEPTED" and run_id is not None:
            self._associate_case_local(record, run_id)
        return changed

    def _save_stage4_test_case_postgres(
        self, record: Stage4TestCaseRecord, *, run_id: str | None
    ) -> bool:
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT project_name, scenario_id, execution_profile_id, variant_id,
                       logical_key_hash, status
                FROM stage4_tc_reservations WHERE tc_id = %s FOR UPDATE
                """,
                (record.tc_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise Stage4PersistenceConflict(
                    f"TC {record.tc_id} has no authoritative reservation"
                )
            reservation = dict(
                zip(
                    (
                        "project_name",
                        "scenario_id",
                        "execution_profile_id",
                        "variant_id",
                        "logical_key_hash",
                        "status",
                    ),
                    row,
                    strict=True,
                )
            )
            self._validate_reservation(record, reservation)

            cursor.execute(
                "SELECT payload FROM stage4_test_cases WHERE tc_id = %s FOR UPDATE",
                (record.tc_id,),
            )
            existing_row = cursor.fetchone()
            existing = None
            if existing_row is not None:
                existing_payload = existing_row[0]
                if not isinstance(existing_payload, dict):
                    existing_payload = json.loads(existing_payload)
                existing = Stage4TestCaseRecord.model_validate(existing_payload)
            changed = self._assert_test_case_mutation(existing, record)
            payload = record.model_dump(mode="json")
            if changed:
                cursor.execute(
                    """
                    INSERT INTO stage4_test_cases(tc_id, project_name, logical_key_hash,
                        status, input_fingerprint, payload, payload_checksum)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (tc_id) DO UPDATE
                    SET status = EXCLUDED.status,
                        input_fingerprint = EXCLUDED.input_fingerprint,
                        payload = EXCLUDED.payload,
                        payload_checksum = EXCLUDED.payload_checksum,
                        updated_at = now()
                    """,
                    (
                        record.tc_id,
                        record.project_name,
                        record.logical_key_hash,
                        record.status,
                        record.input_fingerprint,
                        json.dumps(payload),
                        canonical_checksum({"payload": payload}),
                    ),
                )
                cursor.execute("DELETE FROM stage4_tc_steps WHERE tc_id = %s", (record.tc_id,))
                for step in record.tc_steps:
                    cursor.execute(
                        "INSERT INTO stage4_tc_steps(tc_id, step, payload) VALUES (%s,%s,%s)",
                        (record.tc_id, step.step, json.dumps(step.model_dump(mode="json"))),
                    )
                cursor.execute(
                    """
                    UPDATE stage4_tc_reservations
                    SET status = %s, payload = %s, updated_at = now()
                    WHERE tc_id = %s AND logical_key_hash = %s
                    """,
                    (
                        record.status,
                        json.dumps(payload),
                        record.tc_id,
                        record.logical_key_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    raise Stage4PersistenceConflict(
                        f"TC {record.tc_id} reservation changed during persistence"
                    )
            if record.status == "ACCEPTED" and run_id is not None:
                self._associate_case_postgres(cursor, record, run_id)
            conn.commit()
            return changed

    @staticmethod
    def _validate_reservation(record: Stage4TestCaseRecord, reservation: dict[str, Any]) -> None:
        expected = {
            "project_name": record.project_name,
            "scenario_id": record.scenario_id,
            "execution_profile_id": record.execution_profile_id,
            "variant_id": record.variant_id,
            "logical_key_hash": record.logical_key_hash,
        }
        mismatches = {
            key: (reservation.get(key), value)
            for key, value in expected.items()
            if reservation.get(key) != value
        }
        if mismatches:
            raise Stage4PersistenceConflict(
                f"TC {record.tc_id} does not match its authoritative reservation: {mismatches}"
            )

    @staticmethod
    def _assert_test_case_mutation(
        existing: Stage4TestCaseRecord | None, incoming: Stage4TestCaseRecord
    ) -> bool:
        if existing is None:
            return True
        if existing.status != "ACCEPTED":
            return existing.model_dump(mode="json") != incoming.model_dump(mode="json")
        if existing.model_dump(mode="json") == incoming.model_dump(mode="json"):
            return False
        raise RevisionRequired(
            f"TC {incoming.tc_id} is already accepted; changed input or output requires revision"
        )

    def _associate_case_local(self, record: Stage4TestCaseRecord, run_id: str) -> None:
        self._upsert_local(
            "stage4_run_test_case",
            f"{record.project_name}:{run_id}:{record.tc_id}",
            {
                "project": record.project_name,
                "run_id": run_id,
                "tc_id": record.tc_id,
                "input_fingerprint": record.input_fingerprint,
                "generated_files_checksum": canonical_checksum(
                    {"files": record.generated_file_hashes}
                ),
            },
        )

    @staticmethod
    def _associate_case_postgres(cursor: Any, record: Stage4TestCaseRecord, run_id: str) -> None:
        cursor.execute(
            """
            INSERT INTO stage4_run_test_cases(project_name, run_id, tc_id,
                input_fingerprint, generated_files_checksum)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (project_name, run_id, tc_id) DO UPDATE
            SET input_fingerprint = EXCLUDED.input_fingerprint,
                generated_files_checksum = EXCLUDED.generated_files_checksum
            """,
            (
                record.project_name,
                run_id,
                record.tc_id,
                record.input_fingerprint,
                canonical_checksum({"files": record.generated_file_hashes}),
            ),
        )

    def get_stage4_test_case(self, tc_id: int) -> Stage4TestCaseRecord | None:
        """Return the persisted integer-identity master for ``tc_id`` if present."""
        if self._local:
            row = self._local_row("stage4_test_case", str(tc_id))
            if row is not None:
                return Stage4TestCaseRecord.model_validate(row["payload"])
            return None
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT payload FROM stage4_test_cases WHERE tc_id = %s", (tc_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                return None
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            if "tc_id" not in payload:
                return None
            return Stage4TestCaseRecord.model_validate(payload)

    # --- Frozen run inputs --------------------------------------------------

    def save_frozen_manifest(self, manifest: FrozenRunManifest) -> bool:
        """Create a frozen manifest or verify an identical replay.

        A provider/model/input change under the same project and run ID is a
        terminal reproducibility error, never an upsert.
        """
        existing = self.load_frozen_manifest(manifest.project_name, manifest.run_id)
        if existing is not None:
            self._assert_same_manifest(existing, manifest)
            return False
        payload = manifest.model_dump(mode="json")
        if self._local:
            self._upsert_local(
                "stage4_frozen_manifest",
                self._run_key(manifest.project_name, manifest.run_id),
                {"project": manifest.project_name, "run_id": manifest.run_id, "payload": payload},
            )
            return True
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO stage4_frozen_run_manifests(project_name, run_id, checksum,
                    provider_fingerprint_hash, payload)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (project_name, run_id) DO NOTHING
                """,
                (
                    manifest.project_name,
                    manifest.run_id,
                    manifest.checksum,
                    manifest.provider_fingerprint.fingerprint_hash,
                    json.dumps(payload),
                ),
            )
            created = int(cursor.rowcount) == 1
            if not created:
                cursor.execute(
                    """
                    SELECT payload FROM stage4_frozen_run_manifests
                    WHERE project_name = %s AND run_id = %s
                    """,
                    (manifest.project_name, manifest.run_id),
                )
                row = cursor.fetchone()
                if row is None:
                    raise Stage4PersistenceConflict("frozen manifest race could not be reconciled")
                stored_payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                self._assert_same_manifest(
                    FrozenRunManifest.model_validate(stored_payload), manifest
                )
            conn.commit()
            return created

    def save_or_validate_frozen_manifest(self, manifest: FrozenRunManifest) -> bool:
        """Compatibility alias with an explicit replay-safe name."""
        return self.save_frozen_manifest(manifest)

    def load_frozen_manifest(self, project_name: str, run_id: str) -> FrozenRunManifest | None:
        if self._local:
            row = self._local_row("stage4_frozen_manifest", self._run_key(project_name, run_id))
            return FrozenRunManifest.model_validate(row["payload"]) if row else None
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM stage4_frozen_run_manifests
                WHERE project_name = %s AND run_id = %s
                """,
                (project_name, run_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return FrozenRunManifest.model_validate(payload)

    @staticmethod
    def _assert_same_manifest(existing: FrozenRunManifest, incoming: FrozenRunManifest) -> None:
        if existing.model_dump(mode="json") != incoming.model_dump(mode="json"):
            raise InputManifestChanged(
                f"frozen input manifest changed for {incoming.project_name}/{incoming.run_id}"
            )

    # --- Immutable test-data snapshots ------------------------------------

    def save_test_data_snapshot(
        self, snapshot: TestDataSnapshotRef, normalized_payload: dict[str, Any]
    ) -> bool:
        """Persist an immutable normalized test-data snapshot and exact values."""
        envelope = {
            "snapshot": snapshot.model_dump(mode="json"),
            "normalized_payload": normalized_payload,
        }
        envelope_checksum = canonical_checksum({"payload": envelope})
        existing = self.load_test_data_snapshot(snapshot.snapshot_id)
        if existing is not None:
            existing_snapshot, existing_payload = existing
            if (
                existing_snapshot.model_dump(mode="json") != snapshot.model_dump(mode="json")
                or existing_payload != normalized_payload
            ):
                raise Stage4PersistenceConflict(
                    f"test-data snapshot {snapshot.snapshot_id} is immutable"
                )
            return False
        if self._local:
            self._upsert_local(
                "stage4_test_data_snapshot",
                snapshot.snapshot_id,
                {
                    "project": snapshot.project,
                    "payload": envelope,
                    "payload_checksum": envelope_checksum,
                },
            )
            return True
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO stage4_test_data_snapshots(snapshot_id, project_name,
                    workbook_checksum, normalized_checksum, payload_checksum, payload)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (snapshot_id) DO NOTHING
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.project,
                    snapshot.workbook_checksum,
                    snapshot.normalized_checksum,
                    envelope_checksum,
                    json.dumps(envelope),
                ),
            )
            created = int(cursor.rowcount) == 1
            if not created:
                cursor.execute(
                    "SELECT payload FROM stage4_test_data_snapshots WHERE snapshot_id = %s",
                    (snapshot.snapshot_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise Stage4PersistenceConflict(
                        "test-data snapshot race could not be reconciled"
                    )
                stored = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                if stored != envelope:
                    raise Stage4PersistenceConflict(
                        f"test-data snapshot {snapshot.snapshot_id} is immutable"
                    )
            conn.commit()
            return created

    def load_test_data_snapshot(
        self, snapshot_id: str
    ) -> tuple[TestDataSnapshotRef, dict[str, Any]] | None:
        if self._local:
            row = self._local_row("stage4_test_data_snapshot", snapshot_id)
            if row is None:
                return None
            envelope = row["payload"]
        else:
            with self._connect() as conn, conn.cursor() as cursor:
                cursor.execute(
                    "SELECT payload FROM stage4_test_data_snapshots WHERE snapshot_id = %s",
                    (snapshot_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                envelope = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return (
            TestDataSnapshotRef.model_validate(envelope["snapshot"]),
            dict(envelope["normalized_payload"]),
        )

    # --- Mutable coordination records -------------------------------------

    def save_file_journal_metadata(
        self,
        *,
        project_name: str,
        run_id: str,
        tc_id: int,
        status: str,
        journal_path: str,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "project_name": project_name,
            "run_id": run_id,
            "tc_id": tc_id,
            "status": status,
            "journal_path": journal_path,
            "payload": payload,
        }
        checksum = canonical_checksum({"payload": record})
        if self._local:
            self._upsert_local(
                "stage4_file_journal",
                f"{self._run_key(project_name, run_id)}:{tc_id}",
                {"project": project_name, "payload": record, "payload_checksum": checksum},
            )
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO stage4_file_journals(project_name, run_id, tc_id, status,
                    journal_path, payload_checksum, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (project_name, run_id, tc_id) DO UPDATE
                SET status = EXCLUDED.status, journal_path = EXCLUDED.journal_path,
                    payload_checksum = EXCLUDED.payload_checksum, payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (project_name, run_id, tc_id, status, journal_path, checksum, json.dumps(record)),
            )
            conn.commit()

    def load_file_journal_metadata(
        self, project_name: str, run_id: str, tc_id: int
    ) -> dict[str, Any] | None:
        if self._local:
            row = self._local_row(
                "stage4_file_journal", f"{self._run_key(project_name, run_id)}:{tc_id}"
            )
            return dict(row["payload"]) if row else None
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM stage4_file_journals
                WHERE project_name = %s AND run_id = %s AND tc_id = %s
                """,
                (project_name, run_id, tc_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return dict(payload)

    def save_kg_publication_attempt(
        self,
        *,
        attempt_id: str,
        project_name: str,
        run_id: str,
        tc_id: int,
        status: str,
        previous_snapshot_id: str | None,
        proposed_snapshot_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "attempt_id": attempt_id,
            "project_name": project_name,
            "run_id": run_id,
            "tc_id": tc_id,
            "status": status,
            "previous_snapshot_id": previous_snapshot_id,
            "proposed_snapshot_id": proposed_snapshot_id,
            "payload": payload,
        }
        checksum = canonical_checksum({"payload": record})
        if self._local:
            self._upsert_local(
                "stage4_kg_publication_attempt",
                attempt_id,
                {"project": project_name, "payload": record, "payload_checksum": checksum},
            )
            return
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO stage4_kg_publication_attempts(attempt_id, project_name, run_id,
                    tc_id, status, previous_snapshot_id, proposed_snapshot_id,
                    payload_checksum, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (attempt_id) DO UPDATE
                SET status = EXCLUDED.status,
                    previous_snapshot_id = EXCLUDED.previous_snapshot_id,
                    proposed_snapshot_id = EXCLUDED.proposed_snapshot_id,
                    payload_checksum = EXCLUDED.payload_checksum,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    attempt_id,
                    project_name,
                    run_id,
                    tc_id,
                    status,
                    previous_snapshot_id,
                    proposed_snapshot_id,
                    checksum,
                    json.dumps(record),
                ),
            )
            conn.commit()

    def load_kg_publication_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        if self._local:
            row = self._local_row("stage4_kg_publication_attempt", attempt_id)
            return dict(row["payload"]) if row else None
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM stage4_kg_publication_attempts WHERE attempt_id = %s",
                (attempt_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return dict(payload)

    @staticmethod
    def make_idempotency_key(
        *, run_id: str, tc_id: int, node_name: str, input_checksum: str
    ) -> str:
        value = "|".join((run_id, str(tc_id), node_name, input_checksum))
        return "S4IDEM-" + hashlib.sha256(value.encode("utf-8")).hexdigest()

    def save_idempotency_record(
        self,
        *,
        project_name: str,
        run_id: str,
        tc_id: int,
        node_name: str,
        input_checksum: str,
        result_payload: dict[str, Any],
    ) -> tuple[str, bool]:
        key = self.make_idempotency_key(
            run_id=run_id, tc_id=tc_id, node_name=node_name, input_checksum=input_checksum
        )
        result_checksum = canonical_checksum({"payload": result_payload})
        existing = self.get_idempotency_record(key)
        if existing is not None:
            if (
                existing.get("input_checksum") != input_checksum
                or existing.get("result_checksum") != result_checksum
            ):
                raise Stage4PersistenceConflict(f"idempotency key conflict: {key}")
            return key, False
        record = {
            "idempotency_key": key,
            "project_name": project_name,
            "run_id": run_id,
            "tc_id": tc_id,
            "node_name": node_name,
            "input_checksum": input_checksum,
            "result_checksum": result_checksum,
            "result_payload": result_payload,
        }
        if self._local:
            self._upsert_local(
                "stage4_idempotency_record", key, {"project": project_name, "payload": record}
            )
            return key, True
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO stage4_idempotency_records(idempotency_key, project_name, run_id,
                    tc_id, node_name, input_checksum, result_checksum, result_payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    key,
                    project_name,
                    run_id,
                    tc_id,
                    node_name,
                    input_checksum,
                    result_checksum,
                    json.dumps(result_payload),
                ),
            )
            created = int(cursor.rowcount) == 1
            if not created:
                cursor.execute(
                    """
                    SELECT input_checksum, result_checksum
                    FROM stage4_idempotency_records WHERE idempotency_key = %s
                    """,
                    (key,),
                )
                row = cursor.fetchone()
                if row is None or row[0] != input_checksum or row[1] != result_checksum:
                    raise Stage4PersistenceConflict(f"idempotency key conflict: {key}")
            conn.commit()
            return key, created

    def get_idempotency_record(self, idempotency_key: str) -> dict[str, Any] | None:
        if self._local:
            row = self._local_row("stage4_idempotency_record", idempotency_key)
            return dict(row["payload"]) if row else None
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT project_name, run_id, tc_id, node_name, input_checksum,
                       result_checksum, result_payload
                FROM stage4_idempotency_records WHERE idempotency_key = %s
                """,
                (idempotency_key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            result_payload = row[6] if isinstance(row[6], dict) else json.loads(row[6])
            return {
                "idempotency_key": idempotency_key,
                "project_name": row[0],
                "run_id": row[1],
                "tc_id": row[2],
                "node_name": row[3],
                "input_checksum": row[4],
                "result_checksum": row[5],
                "result_payload": result_payload,
            }

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

    def save_blocker(
        self,
        project: str,
        blocker: CodegenBlocker,
        *,
        run_id: str | None = None,
        tc_id: int | None = None,
    ) -> None:
        """Persist a structured blocker (never silently dropped)."""
        updates: dict[str, Any] = {}
        if run_id is not None:
            if blocker.run_id not in (None, run_id):
                raise Stage4PersistenceConflict("blocker run_id conflicts with persistence scope")
            updates["run_id"] = run_id
        if tc_id is not None:
            if blocker.tc_id not in (None, tc_id):
                raise Stage4PersistenceConflict("blocker tc_id conflicts with persistence scope")
            updates["tc_id"] = tc_id
        if updates:
            blocker = blocker.model_copy(update=updates)
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
        """Rebuild one run's accepted-case/blocker mirror from durable state."""
        test_cases = self._load_stage4_test_cases(project, run_id)
        blockers = self._load_run_blockers(project, run_id)
        draft = TestCasesArtifact.model_construct(
            project=project,
            run_id=run_id,
            checksum="",
            test_cases=test_cases,
            blockers=blockers,
        )
        return TestCasesArtifact.model_validate(
            {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
        )

    def write_test_cases_artifact(
        self,
        project: str,
        run_id: str,
        *,
        generated_dir: Path | None = None,
    ) -> Path:
        """Atomically write ``generated/<project>/<run>/test-cases/test_cases.json``."""
        safe_run_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        if not run_id or any(character not in safe_run_characters for character in run_id):
            raise ValueError("run_id contains unsafe path characters")
        root = (generated_dir or self.settings.paths.generated_dir).resolve()
        output_dir = (root / normalize_project(project) / run_id / "test-cases").resolve()
        if root != output_dir and root not in output_dir.parents:
            raise ValueError("artifact path escapes generated_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "test_cases.json"
        artifact = self.rebuild_test_cases_artifact(project, run_id)
        serialized = json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=".test_cases.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_name = handle.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return destination

    def _load_stage4_test_cases(self, project: str, run_id: str) -> list[Stage4TestCaseRecord]:
        if self._local:
            tc_ids = {
                int(row["tc_id"])
                for row in self._local_rows()
                if row.get("kind") == "stage4_run_test_case"
                and row.get("project") == project
                and row.get("run_id") == run_id
            }
            records: list[Stage4TestCaseRecord] = []
            for row in self._local_rows():
                if row.get("kind") != "stage4_test_case" or row.get("project") != project:
                    continue
                record = Stage4TestCaseRecord.model_validate(row["payload"])
                if record.tc_id in tc_ids and record.status == "ACCEPTED":
                    records.append(record)
            return sorted(records, key=lambda tc: tc.tc_id)
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT t.payload
                FROM stage4_test_cases t
                JOIN stage4_run_test_cases r ON r.tc_id = t.tc_id
                WHERE r.project_name = %s AND r.run_id = %s AND t.status = 'ACCEPTED'
                ORDER BY t.tc_id
                """,
                (project, run_id),
            )
            return [
                Stage4TestCaseRecord.model_validate(
                    row[0] if isinstance(row[0], dict) else json.loads(row[0])
                )
                for row in cursor.fetchall()
            ]

    def _load_run_blockers(self, project: str, run_id: str) -> list[CodegenBlocker]:
        if self._local:
            blockers = [
                CodegenBlocker.model_validate(row["payload"])
                for row in self._local_rows()
                if row.get("kind") == "codegen_blocker"
                and row.get("project") == project
                and (row.get("payload") or {}).get("run_id") == run_id
            ]
            return sorted(blockers, key=lambda blocker: blocker.blocker_id)
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload FROM codegen_blockers
                WHERE project = %s AND payload->>'run_id' = %s
                ORDER BY blocker_id
                """,
                (project, run_id),
            )
            return [
                CodegenBlocker.model_validate(
                    row[0] if isinstance(row[0], dict) else json.loads(row[0])
                )
                for row in cursor.fetchall()
            ]

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

    def _local_row(self, kind: str, key: str) -> dict[str, Any] | None:
        for row in self._local_rows():
            if row.get("kind") == kind and row.get("key") == key:
                return row
        return None

    @staticmethod
    def _run_key(project_name: str, run_id: str) -> str:
        return f"{project_name}\u241f{run_id}"

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
