"""PostgreSQL generated-content store with a local JSON trace mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.errors import VersionConflictError
from multi_agentic_graph_rag.domain.schemas import DocumentManifest, RequirementArtifact


class PostgresStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def check(self) -> str:
        if self.settings.postgres.mode == "local_json":
            self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS postgres local_json path={self.settings.postgres.local_path}"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("select 1")
            cursor.fetchone()
        return "PASS postgres connectivity"

    def ensure_schema(self) -> None:
        if self.settings.postgres.mode == "local_json":
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists document_versions (
                      document_version_id text primary key,
                      document_id text not null,
                      project text not null,
                      version text not null,
                      checksum text not null,
                      manifest jsonb not null,
                      created_at timestamptz not null default now(),
                      unique(document_id, version)
                    );
                    create table if not exists ingestion_runs (
                      run_id text primary key,
                      document_version_id text not null,
                      status text not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists requirement_artifacts (
                      artifact_path text primary key,
                      document_version_id text not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    """
                )
            connection.commit()

    def assert_version_allowed(self, manifest: DocumentManifest, replace_version: bool) -> None:
        if self.settings.postgres.mode == "local_json":
            existing = self._local_document_versions()
            key = (manifest.document_id, manifest.version)
            checksum = existing.get(key)
            if checksum and checksum != manifest.source_checksum and not replace_version:
                raise VersionConflictError(
                    "same document/version already exists with a different checksum; "
                    "use --replace-version to overwrite"
                )
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "select checksum from document_versions where document_id=%s and version=%s",
                (manifest.document_id, manifest.version),
            )
            row = cursor.fetchone()
        if row and row[0] != manifest.source_checksum and not replace_version:
            raise VersionConflictError(
                "same document/version already exists with a different checksum; "
                "use --replace-version to overwrite"
            )

    def persist_manifest(self, manifest: DocumentManifest) -> None:
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "document_version",
                manifest.document_version_id,
                {"kind": "document_version", "manifest": manifest.model_dump(mode="json")},
            )
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into document_versions
                      (document_version_id, document_id, project, version, checksum, manifest)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (document_version_id) do update set manifest=excluded.manifest
                    """,
                    (
                        manifest.document_version_id,
                        manifest.document_id,
                        manifest.project,
                        manifest.version,
                        manifest.source_checksum,
                        json.dumps(manifest.model_dump(mode="json")),
                    ),
                )
            connection.commit()

    def persist_artifact(
        self, artifact: RequirementArtifact, artifact_path: str, run_id: str
    ) -> None:
        payload = artifact.model_dump(mode="json")
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "requirement_artifact",
                artifact.document_version_id,
                {
                    "kind": "requirement_artifact",
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "artifact": payload,
                },
            )
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into requirement_artifacts
                      (artifact_path, document_version_id, payload)
                    values (%s, %s, %s)
                    on conflict (artifact_path) do update set payload=excluded.payload
                    """,
                    (artifact_path, artifact.document_version_id, json.dumps(payload)),
                )
            connection.commit()

    def record_run(self, run_id: str, status: str, payload: dict[str, Any]) -> None:
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "ingestion_run",
                run_id,
                {"kind": "ingestion_run", "run_id": run_id, "status": status, "payload": payload},
            )
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into ingestion_runs (run_id, document_version_id, status, payload)
                    values (%s, %s, %s, %s)
                    on conflict (run_id) do update
                    set status=excluded.status, payload=excluded.payload
                    """,
                    (
                        run_id,
                        payload.get("document_version_id", ""),
                        status,
                        json.dumps(payload),
                    ),
                )
            connection.commit()

    def _connect(self) -> Any:
        import psycopg

        return psycopg.connect(self.settings.postgres.dsn)

    def _append_local(self, payload: dict[str, Any]) -> None:
        self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
        payload["written_at"] = datetime.now(UTC).isoformat()
        with self.settings.postgres.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if self.settings.postgres.local_path.exists():
            rows = [
                json.loads(line)
                for line in self.settings.postgres.local_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        payload["written_at"] = datetime.now(UTC).isoformat()
        payload["_local_key"] = key
        replaced = False
        for index, row in enumerate(rows):
            if row.get("kind") == kind and row.get("_local_key") == key:
                rows[index] = payload
                replaced = True
                break
        if not replaced:
            rows.append(payload)
        self.settings.postgres.local_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _local_document_versions(self) -> dict[tuple[str, str], str]:
        versions: dict[tuple[str, str], str] = {}
        if not self.settings.postgres.local_path.exists():
            return versions
        for line in self.settings.postgres.local_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("kind") == "document_version":
                manifest = payload["manifest"]
                versions[(manifest["document_id"], manifest["version"])] = manifest[
                    "source_checksum"
                ]
        return versions
