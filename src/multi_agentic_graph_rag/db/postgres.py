"""PostgreSQL generated-content store with a local JSON trace mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.errors import SchemaMismatchError, VersionConflictError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentManifest,
    RequirementArtifact,
    RequirementRevisionSnapshot,
    UserStoryArtifact,
)

_MANAGED_TABLES = (
    "user_stories",
    "user_story_artifacts",
    "requirement_fact_links",
    "requirement_evidence",
    "requirement_delta_events",
    "requirement_revisions",
    "requirements",
    "fact_occurrences",
    "canonical_facts",
    "requirement_artifacts",
    "document_versions",
    "ingestion_runs",
)

_TEXT_TYPES = {"text", "character varying"}
_EXPECTED_COLUMNS = {
    "document_versions": {
        "document_version_id": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "version": _TEXT_TYPES,
        "checksum": _TEXT_TYPES,
        "manifest": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "ingestion_runs": {
        "run_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "status": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "requirement_artifacts": {
        "artifact_path": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "user_story_artifacts": {
        "artifact_path": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "user_stories": {
        "story_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "doc_version": _TEXT_TYPES,
        "title": _TEXT_TYPES,
        "priority": _TEXT_TYPES,
        "status": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "canonical_facts": {
        "canonical_fact_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "normalized_text": _TEXT_TYPES,
        "representative_text": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "fact_occurrences": {
        "fact_id": _TEXT_TYPES,
        "canonical_fact_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "chunk_id": _TEXT_TYPES,
        "page": {"integer"},
        "section": _TEXT_TYPES,
        "quote": _TEXT_TYPES,
        "start_char": {"integer"},
        "end_char": {"integer"},
        "source_path": _TEXT_TYPES,
        "text": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "requirements": {
        "requirement_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "requirement_key": _TEXT_TYPES,
        "status": _TEXT_TYPES,
        "first_seen_document_version_id": _TEXT_TYPES,
        "active_revision_id": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "requirement_revisions": {
        "revision_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "statement": _TEXT_TYPES,
        "normalized_statement": _TEXT_TYPES,
        "requirement_type": _TEXT_TYPES,
        "priority": _TEXT_TYPES,
        "status": _TEXT_TYPES,
        "first_seen_document_version_id": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "requirement_evidence": {
        "evidence_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "revision_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "chunk_id": _TEXT_TYPES,
        "page": {"integer"},
        "section": _TEXT_TYPES,
        "quote": _TEXT_TYPES,
        "start_char": {"integer"},
        "end_char": {"integer"},
        "source_path": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "requirement_fact_links": {
        "evidence_id": _TEXT_TYPES,
        "fact_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "revision_id": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "requirement_delta_events": {
        "event_id": _TEXT_TYPES,
        "event_type": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "revision_id": _TEXT_TYPES,
        "previous_revision_id": _TEXT_TYPES,
        "superseded_by_revision_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "evidence_ids": {"jsonb"},
        "impacted_artifact_types": {"jsonb"},
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
}


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
        self.ensure_schema()
        return "PASS postgres connectivity and schema"

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
                    create table if not exists user_story_artifacts (
                      artifact_path text primary key,
                      document_version_id text not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists user_stories (
                      story_id text primary key,
                      requirement_id text not null,
                      project text not null,
                      document_id text not null,
                      document_version_id text not null,
                      doc_version text not null,
                      title text not null,
                      priority text not null,
                      status text not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists canonical_facts (
                      canonical_fact_id text primary key,
                      project text not null,
                      document_id text not null,
                      normalized_text text not null,
                      representative_text text not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists fact_occurrences (
                      fact_id text primary key,
                      canonical_fact_id text not null,
                      project text not null,
                      document_id text not null,
                      document_version_id text not null,
                      chunk_id text not null,
                      page integer,
                      section text,
                      quote text not null,
                      start_char integer not null,
                      end_char integer not null,
                      source_path text not null,
                      text text not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists requirements (
                      requirement_id text primary key,
                      project text not null,
                      document_id text not null,
                      requirement_key text not null,
                      status text not null,
                      first_seen_document_version_id text not null,
                      active_revision_id text,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now(),
                      unique(project, document_id, requirement_key)
                    );
                    create table if not exists requirement_revisions (
                      revision_id text primary key,
                      requirement_id text not null,
                      statement text not null,
                      normalized_statement text not null,
                      requirement_type text not null,
                      priority text not null,
                      status text not null,
                      first_seen_document_version_id text not null,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
                    );
                    create table if not exists requirement_evidence (
                      evidence_id text primary key,
                      requirement_id text not null,
                      revision_id text not null,
                      document_version_id text not null,
                      chunk_id text not null,
                      page integer,
                      section text,
                      quote text not null,
                      start_char integer not null,
                      end_char integer not null,
                      source_path text not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists requirement_fact_links (
                      evidence_id text not null,
                      fact_id text not null,
                      requirement_id text not null,
                      revision_id text not null,
                      created_at timestamptz not null default now(),
                      primary key(evidence_id, fact_id)
                    );
                    create table if not exists requirement_delta_events (
                      event_id text primary key,
                      event_type text not null,
                      requirement_id text not null,
                      revision_id text,
                      previous_revision_id text,
                      superseded_by_revision_id text,
                      document_version_id text not null,
                      evidence_ids jsonb not null,
                      impacted_artifact_types jsonb not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create index if not exists requirement_revisions_requirement_idx
                      on requirement_revisions(requirement_id);
                    create index if not exists requirement_evidence_revision_idx
                      on requirement_evidence(revision_id);
                    create index if not exists fact_occurrences_chunk_idx
                      on fact_occurrences(chunk_id);
                    create index if not exists user_stories_requirement_idx
                      on user_stories(requirement_id);
                    create index if not exists user_stories_document_version_idx
                      on user_stories(document_version_id);
                    """
                )
                self._validate_schema(cursor)
            connection.commit()

    def reset_schema(self) -> str:
        if self.settings.postgres.mode == "local_json":
            if self.settings.postgres.local_path.exists():
                self.settings.postgres.local_path.unlink()
            return f"RESET postgres local_json path={self.settings.postgres.local_path}"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("drop table if exists " + ", ".join(_MANAGED_TABLES) + " cascade")
            connection.commit()
        self.ensure_schema()
        return f"RESET postgres schema tables={len(_MANAGED_TABLES)}"

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

    def load_requirement_revision_snapshot(
        self,
        *,
        project: str,
        document_id: str,
    ) -> dict[str, RequirementRevisionSnapshot]:
        if self.settings.postgres.mode == "local_json":
            return self._local_requirement_revision_snapshot(project, document_id)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                select r.requirement_id,
                       rr.revision_id,
                       rr.statement,
                       rr.normalized_statement
                from requirements r
                join requirement_revisions rr on rr.revision_id = r.active_revision_id
                where r.project = %s
                  and r.document_id = %s
                  and r.status = 'active'
                  and rr.status = 'active'
                """,
                (project, document_id),
            )
            rows = cursor.fetchall()
        return {
            str(row[0]): RequirementRevisionSnapshot(
                requirement_id=str(row[0]),
                revision_id=str(row[1]),
                statement=str(row[2]),
                normalized_statement=str(row[3]),
            )
            for row in rows
        }

    def persist_manifest(self, manifest: DocumentManifest) -> None:
        manifest_payload = _manifest_reference_payload(manifest)
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "document_version",
                manifest.document_version_id,
                {"kind": "document_version", "manifest": manifest_payload},
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
                        json.dumps(manifest_payload),
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
            self._persist_requirement_ledger_local(artifact)
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
                self._persist_requirement_ledger_postgres(cursor, artifact)
            connection.commit()

    def load_requirement_artifact_payload(
        self,
        artifact_path: str | None = None,
        document_version_id: str | None = None,
    ) -> dict[str, Any] | None:
        if artifact_path is None and document_version_id is None:
            raise ValueError("artifact_path or document_version_id is required")
        if self.settings.postgres.mode == "local_json":
            return self._local_requirement_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )

        with self._connect() as connection, connection.cursor() as cursor:
            if artifact_path is not None:
                cursor.execute(
                    """
                    select payload from requirement_artifacts
                    where artifact_path = %s
                    """,
                    (artifact_path,),
                )
                row = cursor.fetchone()
                if row:
                    return _artifact_payload_from_row(row[0])
            if document_version_id is not None:
                cursor.execute(
                    """
                    select payload from requirement_artifacts
                    where document_version_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (document_version_id,),
                )
                row = cursor.fetchone()
                if row:
                    return _artifact_payload_from_row(row[0])
        return None

    def persist_user_story_artifact(
        self,
        artifact: UserStoryArtifact,
        artifact_path: str,
        run_id: str,
    ) -> None:
        payload = artifact.model_dump(mode="json")
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "user_story_artifact",
                artifact.document_version_id,
                {
                    "kind": "user_story_artifact",
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "artifact": payload,
                },
            )
            self._persist_user_stories_local(artifact)
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into user_story_artifacts
                      (artifact_path, document_version_id, payload)
                    values (%s, %s, %s)
                    on conflict (artifact_path) do update set payload=excluded.payload
                    """,
                    (artifact_path, artifact.document_version_id, json.dumps(payload)),
                )
                self._persist_user_stories_postgres(cursor, artifact)
            connection.commit()

    def load_user_story_artifact_payload(
        self,
        artifact_path: str | None = None,
        document_version_id: str | None = None,
    ) -> dict[str, Any] | None:
        if artifact_path is None and document_version_id is None:
            raise ValueError("artifact_path or document_version_id is required")
        if self.settings.postgres.mode == "local_json":
            return self._local_user_story_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )
        with self._connect() as connection, connection.cursor() as cursor:
            if artifact_path is not None:
                cursor.execute(
                    "select payload from user_story_artifacts where artifact_path = %s",
                    (artifact_path,),
                )
                row = cursor.fetchone()
                if row:
                    return _artifact_payload_from_row(row[0])
            if document_version_id is not None:
                cursor.execute(
                    """
                    select payload from user_story_artifacts
                    where document_version_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (document_version_id,),
                )
                row = cursor.fetchone()
                if row:
                    return _artifact_payload_from_row(row[0])
        return None

    def _persist_user_stories_postgres(self, cursor: Any, artifact: UserStoryArtifact) -> None:
        for story_id, record in artifact.stories.items():
            cursor.execute(
                """
                insert into user_stories
                  (story_id, requirement_id, project, document_id, document_version_id,
                   doc_version, title, priority, status, payload)
                values (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s)
                on conflict (story_id) do update
                set title = excluded.title,
                    priority = excluded.priority,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    story_id,
                    record.requirement_id,
                    record.project,
                    record.document_id,
                    record.document_version_id,
                    record.doc_version,
                    record.title,
                    record.priority,
                    json.dumps(record.model_dump(mode="json")),
                ),
            )

    def _persist_user_stories_local(self, artifact: UserStoryArtifact) -> None:
        for story_id, record in artifact.stories.items():
            self._upsert_local(
                "user_story",
                story_id,
                {
                    "kind": "user_story",
                    "story_id": story_id,
                    "requirement_id": record.requirement_id,
                    "project": record.project,
                    "document_id": record.document_id,
                    "document_version_id": record.document_version_id,
                    "doc_version": record.doc_version,
                    "title": record.title,
                    "priority": record.priority,
                    "status": "active",
                    "user_story": record.model_dump(mode="json"),
                },
            )

    def _local_user_story_artifact_payload(
        self,
        *,
        artifact_path: str | None,
        document_version_id: str | None,
    ) -> dict[str, Any] | None:
        rows = [row for row in self._read_local_rows() if row.get("kind") == "user_story_artifact"]
        if artifact_path is not None:
            for row in reversed(rows):
                if row.get("artifact_path") == artifact_path:
                    artifact = row.get("artifact")
                    return dict(artifact) if isinstance(artifact, dict) else None
        if document_version_id is not None:
            for row in reversed(rows):
                if row.get("_local_key") == document_version_id:
                    artifact = row.get("artifact")
                    return dict(artifact) if isinstance(artifact, dict) else None
        return None

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

    def _persist_requirement_ledger_postgres(
        self,
        cursor: Any,
        artifact: RequirementArtifact,
    ) -> None:
        for canonical_fact in artifact.canonical_facts:
            cursor.execute(
                """
                insert into canonical_facts
                  (canonical_fact_id, project, document_id, normalized_text, representative_text)
                values (%s, %s, %s, %s, %s)
                on conflict (canonical_fact_id) do update
                set representative_text = excluded.representative_text
                """,
                (
                    canonical_fact.canonical_fact_id,
                    artifact.project,
                    artifact.document_id,
                    canonical_fact.normalized_text,
                    canonical_fact.representative_text,
                ),
            )

        for fact_occurrence in artifact.facts:
            trace = fact_occurrence.source_trace
            cursor.execute(
                """
                insert into fact_occurrences
                  (fact_id, canonical_fact_id, project, document_id, document_version_id,
                   chunk_id, page, section, quote, start_char, end_char, source_path, text)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (fact_id) do update
                set quote = excluded.quote,
                    page = excluded.page,
                    section = excluded.section
                """,
                (
                    fact_occurrence.fact_id,
                    fact_occurrence.canonical_fact_id,
                    artifact.project,
                    artifact.document_id,
                    artifact.document_version_id,
                    trace.chunk_id,
                    trace.page,
                    trace.section,
                    trace.quote,
                    trace.start_char,
                    trace.end_char,
                    artifact.source_path,
                    fact_occurrence.text,
                ),
            )

        for requirement in artifact.requirements:
            cursor.execute(
                """
                insert into requirements
                  (requirement_id, project, document_id, requirement_key, status,
                   first_seen_document_version_id, active_revision_id)
                values (%s, %s, %s, %s, 'active', %s, %s)
                on conflict (requirement_id) do update
                set status = 'active',
                    active_revision_id = excluded.active_revision_id,
                    updated_at = now()
                """,
                (
                    requirement.requirement_id,
                    artifact.project,
                    artifact.document_id,
                    requirement.requirement_key,
                    artifact.document_version_id,
                    requirement.revision_id,
                ),
            )
            cursor.execute(
                """
                insert into requirement_revisions
                  (revision_id, requirement_id, statement, normalized_statement,
                   requirement_type, priority, status, first_seen_document_version_id)
                values (%s, %s, %s, %s, %s, %s, 'active', %s)
                on conflict (revision_id) do update
                set statement = excluded.statement,
                    normalized_statement = excluded.normalized_statement,
                    requirement_type = excluded.requirement_type,
                    priority = excluded.priority,
                    status = 'active',
                    updated_at = now()
                """,
                (
                    requirement.revision_id,
                    requirement.requirement_id,
                    requirement.statement,
                    requirement.normalized_statement,
                    requirement.requirement_type,
                    requirement.priority,
                    artifact.document_version_id,
                ),
            )
            for evidence in requirement.evidence:
                trace = evidence.source_trace
                cursor.execute(
                    """
                    insert into requirement_evidence
                      (evidence_id, requirement_id, revision_id, document_version_id, chunk_id,
                       page, section, quote, start_char, end_char, source_path)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (evidence_id) do update
                    set quote = excluded.quote,
                        page = excluded.page,
                        section = excluded.section
                    """,
                    (
                        evidence.evidence_id,
                        requirement.requirement_id,
                        requirement.revision_id,
                        artifact.document_version_id,
                        trace.chunk_id,
                        trace.page,
                        trace.section,
                        trace.quote,
                        trace.start_char,
                        trace.end_char,
                        artifact.source_path,
                    ),
                )
                for fact_id in evidence.fact_ids:
                    cursor.execute(
                        """
                        insert into requirement_fact_links
                          (evidence_id, fact_id, requirement_id, revision_id)
                        values (%s, %s, %s, %s)
                        on conflict (evidence_id, fact_id) do nothing
                        """,
                        (
                            evidence.evidence_id,
                            fact_id,
                            requirement.requirement_id,
                            requirement.revision_id,
                        ),
                    )

        for event in artifact.delta_events:
            if event.event_type == "superseded" and event.revision_id:
                cursor.execute(
                    """
                    update requirement_revisions
                    set status = 'superseded', updated_at = now()
                    where revision_id = %s
                    """,
                    (event.revision_id,),
                )
            cursor.execute(
                """
                insert into requirement_delta_events
                  (event_id, event_type, requirement_id, revision_id, previous_revision_id,
                   superseded_by_revision_id, document_version_id, evidence_ids,
                   impacted_artifact_types, payload)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (event_id) do update
                set payload = excluded.payload
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.requirement_id,
                    event.revision_id,
                    event.previous_revision_id,
                    event.superseded_by_revision_id,
                    event.document_version_id,
                    json.dumps(event.evidence_ids),
                    json.dumps(event.impacted_artifact_types),
                    json.dumps(event.model_dump(mode="json")),
                ),
            )

    def _persist_requirement_ledger_local(self, artifact: RequirementArtifact) -> None:
        for canonical_fact in artifact.canonical_facts:
            self._upsert_local(
                "canonical_fact",
                canonical_fact.canonical_fact_id,
                {
                    "kind": "canonical_fact",
                    "project": artifact.project,
                    "document_id": artifact.document_id,
                    "canonical_fact": canonical_fact.model_dump(mode="json"),
                },
            )

        for fact_occurrence in artifact.facts:
            self._upsert_local(
                "fact_occurrence",
                fact_occurrence.fact_id,
                {
                    "kind": "fact_occurrence",
                    "project": artifact.project,
                    "document_id": artifact.document_id,
                    "document_version_id": artifact.document_version_id,
                    "source_path": artifact.source_path,
                    "fact": fact_occurrence.model_dump(mode="json"),
                },
            )

        for requirement in artifact.requirements:
            self._upsert_local(
                "requirement",
                requirement.requirement_id,
                {
                    "kind": "requirement",
                    "project": artifact.project,
                    "document_id": artifact.document_id,
                    "requirement_id": requirement.requirement_id,
                    "requirement_key": requirement.requirement_key,
                    "status": "active",
                    "active_revision_id": requirement.revision_id,
                    "first_seen_document_version_id": artifact.document_version_id,
                },
            )
            self._upsert_local(
                "requirement_revision",
                requirement.revision_id,
                {
                    "kind": "requirement_revision",
                    "project": artifact.project,
                    "document_id": artifact.document_id,
                    "document_version_id": artifact.document_version_id,
                    "requirement": requirement.model_dump(mode="json"),
                    "status": "active",
                },
            )
            for evidence in requirement.evidence:
                self._upsert_local(
                    "requirement_evidence",
                    evidence.evidence_id,
                    {
                        "kind": "requirement_evidence",
                        "project": artifact.project,
                        "document_id": artifact.document_id,
                        "document_version_id": artifact.document_version_id,
                        "requirement_id": requirement.requirement_id,
                        "revision_id": requirement.revision_id,
                        "source_path": artifact.source_path,
                        "evidence": evidence.model_dump(mode="json"),
                    },
                )
                for fact_id in evidence.fact_ids:
                    self._upsert_local(
                        "requirement_fact_link",
                        f"{evidence.evidence_id}:{fact_id}",
                        {
                            "kind": "requirement_fact_link",
                            "evidence_id": evidence.evidence_id,
                            "fact_id": fact_id,
                            "requirement_id": requirement.requirement_id,
                            "revision_id": requirement.revision_id,
                        },
                    )

        for event in artifact.delta_events:
            if event.event_type == "superseded" and event.revision_id:
                self._update_local_revision_status(event.revision_id, "superseded")
            self._upsert_local(
                "requirement_delta_event",
                event.event_id,
                {
                    "kind": "requirement_delta_event",
                    "event": event.model_dump(mode="json"),
                },
            )

    def _connect(self) -> Any:
        import psycopg

        dsn = self.settings.postgres.dsn
        scheme, separator, remainder = dsn.partition("://")
        if separator and "+" in scheme:
            dsn = f"{scheme.split('+', 1)[0]}://{remainder}"
        return psycopg.connect(dsn)

    def _validate_schema(self, cursor: Any) -> None:
        cursor.execute(
            """
            select table_name, column_name, data_type
            from information_schema.columns
            where table_schema = 'public'
              and table_name = any(%s)
            """,
            (list(_EXPECTED_COLUMNS),),
        )
        actual: dict[str, dict[str, str]] = {}
        for table_name, column_name, data_type in cursor.fetchall():
            actual.setdefault(str(table_name), {})[str(column_name)] = str(data_type)

        problems: list[str] = []
        for table_name, columns in _EXPECTED_COLUMNS.items():
            actual_columns = actual.get(table_name)
            if actual_columns is None:
                problems.append(f"{table_name}: missing table")
                continue
            for column_name, expected_types in columns.items():
                actual_type = actual_columns.get(column_name)
                if actual_type is None:
                    problems.append(f"{table_name}.{column_name}: missing column")
                    continue
                if actual_type not in expected_types:
                    expected = "/".join(sorted(expected_types))
                    problems.append(
                        f"{table_name}.{column_name}: expected {expected}, found {actual_type}"
                    )

        if problems:
            summary = "; ".join(problems[:8])
            if len(problems) > 8:
                summary += f"; ... {len(problems) - 8} more"
            raise SchemaMismatchError(
                "postgres schema does not match the current app contract. "
                f"{summary}. Run `python -m multi_agentic_graph_rag postgres-reset --yes` "
                "against a disposable/local database, then rerun db-check."
            )

    def _append_local(self, payload: dict[str, Any]) -> None:
        self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
        payload["written_at"] = datetime.now(UTC).isoformat()
        with self.settings.postgres.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        self.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._read_local_rows()
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
        self._write_local_rows(rows)

    def _read_local_rows(self) -> list[dict[str, Any]]:
        if not self.settings.postgres.local_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.settings.postgres.local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _write_local_rows(self, rows: list[dict[str, Any]]) -> None:
        self.settings.postgres.local_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _update_local_revision_status(self, revision_id: str, status: str) -> None:
        rows = self._read_local_rows()
        for row in rows:
            if row.get("kind") != "requirement_revision" or row.get("_local_key") != revision_id:
                continue
            row["status"] = status
            requirement = row.get("requirement")
            if isinstance(requirement, dict):
                requirement["status"] = status
            row["written_at"] = datetime.now(UTC).isoformat()
            break
        self._write_local_rows(rows)

    def _local_document_versions(self) -> dict[tuple[str, str], str]:
        versions: dict[tuple[str, str], str] = {}
        for payload in self._read_local_rows():
            if payload.get("kind") == "document_version":
                manifest = payload["manifest"]
                versions[(manifest["document_id"], manifest["version"])] = manifest[
                    "source_checksum"
                ]
        return versions

    def _local_requirement_artifact_payload(
        self,
        *,
        artifact_path: str | None,
        document_version_id: str | None,
    ) -> dict[str, Any] | None:
        rows = [row for row in self._read_local_rows() if row.get("kind") == "requirement_artifact"]
        if artifact_path is not None:
            for row in reversed(rows):
                if row.get("artifact_path") == artifact_path:
                    artifact = row.get("artifact")
                    return dict(artifact) if isinstance(artifact, dict) else None
        if document_version_id is not None:
            for row in reversed(rows):
                if row.get("_local_key") == document_version_id:
                    artifact = row.get("artifact")
                    return dict(artifact) if isinstance(artifact, dict) else None
        return None

    def _local_requirement_revision_snapshot(
        self,
        project: str,
        document_id: str,
    ) -> dict[str, RequirementRevisionSnapshot]:
        active_by_requirement: dict[str, str] = {}
        revisions: dict[str, dict[str, Any]] = {}
        for row in self._read_local_rows():
            if (
                row.get("kind") == "requirement"
                and row.get("project") == project
                and row.get("document_id") == document_id
                and row.get("status") == "active"
            ):
                active_by_requirement[str(row["requirement_id"])] = str(row["active_revision_id"])
            if (
                row.get("kind") == "requirement_revision"
                and row.get("project") == project
                and row.get("document_id") == document_id
                and row.get("status") == "active"
            ):
                requirement = row.get("requirement")
                if isinstance(requirement, dict):
                    revisions[str(row["_local_key"])] = requirement

        snapshot: dict[str, RequirementRevisionSnapshot] = {}
        for requirement_id, revision_id in active_by_requirement.items():
            revision = revisions.get(revision_id)
            if not revision:
                continue
            snapshot[requirement_id] = RequirementRevisionSnapshot(
                requirement_id=requirement_id,
                revision_id=revision_id,
                statement=str(revision["statement"]),
                normalized_statement=str(revision["normalized_statement"]),
            )
        return snapshot


def _manifest_reference_payload(manifest: DocumentManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload["chunks"] = [
        {key: value for key, value in chunk.items() if key not in {"text", "normalized_text"}}
        for chunk in payload["chunks"]
    ]
    return payload


def _artifact_payload_from_row(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value)
