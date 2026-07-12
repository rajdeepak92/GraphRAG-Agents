"""PostgreSQL generated-content store with a local JSON trace mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.errors import SchemaMismatchError, VersionConflictError
from multi_agentic_graph_rag.domain.identifiers import (
    test_scenario_evidence_id,
    user_story_evidence_id,
)
from multi_agentic_graph_rag.domain.schemas import (
    CoverageReport,
    CoverageRequirementRow,
    CoverageSummary,
    DocumentManifest,
    GenerationContextItem,
    GenerationContextRun,
    RequirementArtifact,
    RequirementInput,
    RequirementRevisionSnapshot,
    TestScenarioArtifact,
    TestScenarioBuildResult,
    TestScenarioRecord,
    UserStoryArtifact,
    UserStoryBuildResult,
    UserStoryRecord,
    normalize_priority_label,
)
from multi_agentic_graph_rag.services.id_registry import resolve_many
from multi_agentic_graph_rag.services.requirement_builder import apply_requirement_display_ids
from multi_agentic_graph_rag.services.test_scenario_builder import project_test_scenario_artifact
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact

_MANAGED_TABLES = (
    "schema_migrations",
    "test_scenario_evidence",
    "test_scenarios",
    "test_scenario_artifacts",
    "user_story_evidence",
    "user_stories",
    "user_story_artifacts",
    "artifact_display_ids",
    "display_id_counters",
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
    "generation_context_runs",
    "generation_context_items",
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
    "generation_context_runs": {
        "context_run_id": _TEXT_TYPES,
        "stage": _TEXT_TYPES,
        "anchor_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "source": _TEXT_TYPES,
        "metrics": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "generation_context_items": {
        "context_run_id": _TEXT_TYPES,
        "rank": {"integer"},
        "item_type": _TEXT_TYPES,
        "item_id": _TEXT_TYPES,
        "source": _TEXT_TYPES,
        "score": {"double precision"},
        "selected": {"boolean"},
        "assertion_id": _TEXT_TYPES,
        "text_unit_id": _TEXT_TYPES,
        "entity_id": _TEXT_TYPES,
        "predicate": _TEXT_TYPES,
        "hop_count": {"integer"},
        "normalized_score": {"double precision"},
        "reranker_score": {"double precision"},
        "mandatory": {"boolean"},
        "metadata_json": {"jsonb"},
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
        "display_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "requirement_display_id": _TEXT_TYPES,
        "requirement_revision_id": _TEXT_TYPES,
        "source_req_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "doc_version": _TEXT_TYPES,
        "origin_version": _TEXT_TYPES,
        "title": _TEXT_TYPES,
        "priority": _TEXT_TYPES,
        "confidence": {"double precision"},
        "status": _TEXT_TYPES,
        "origin": _TEXT_TYPES,
        "generation_context_run_id": _TEXT_TYPES,
        "run_id": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "user_story_evidence": {
        "evidence_id": _TEXT_TYPES,
        "story_id": _TEXT_TYPES,
        "story_display_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "requirement_display_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "chunk_id": _TEXT_TYPES,
        "run_id": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "test_scenario_artifacts": {
        "artifact_path": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
    },
    "test_scenarios": {
        "scenario_id": _TEXT_TYPES,
        "display_id": _TEXT_TYPES,
        "story_id": _TEXT_TYPES,
        "story_display_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "requirement_display_id": _TEXT_TYPES,
        "requirement_revision_id": _TEXT_TYPES,
        "source_req_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "doc_version": _TEXT_TYPES,
        "origin_version": _TEXT_TYPES,
        "title": _TEXT_TYPES,
        "scenario_type": _TEXT_TYPES,
        "priority": _TEXT_TYPES,
        "confidence": {"double precision"},
        "status": _TEXT_TYPES,
        "origin": _TEXT_TYPES,
        "generation_context_run_id": _TEXT_TYPES,
        "run_id": _TEXT_TYPES,
        "payload": {"jsonb"},
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "test_scenario_evidence": {
        "evidence_id": _TEXT_TYPES,
        "scenario_id": _TEXT_TYPES,
        "scenario_display_id": _TEXT_TYPES,
        "story_id": _TEXT_TYPES,
        "story_display_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "requirement_display_id": _TEXT_TYPES,
        "document_version_id": _TEXT_TYPES,
        "chunk_id": _TEXT_TYPES,
        "run_id": _TEXT_TYPES,
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
        "display_id": _TEXT_TYPES,
        "project": _TEXT_TYPES,
        "document_id": _TEXT_TYPES,
        "requirement_key": _TEXT_TYPES,
        "source_req_id": _TEXT_TYPES,
        "id_generation_type": _TEXT_TYPES,
        "confidence": {"double precision"},
        "status": _TEXT_TYPES,
        "first_seen_document_version_id": _TEXT_TYPES,
        "active_revision_id": _TEXT_TYPES,
        "superseded_by_version": _TEXT_TYPES,
        "superseded_at": {"timestamp with time zone"},
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "requirement_revisions": {
        "revision_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "display_id": _TEXT_TYPES,
        "statement": _TEXT_TYPES,
        "normalized_statement": _TEXT_TYPES,
        "requirement_type": _TEXT_TYPES,
        "priority": _TEXT_TYPES,
        "source_req_id": _TEXT_TYPES,
        "id_generation_type": _TEXT_TYPES,
        "confidence": {"double precision"},
        "status": _TEXT_TYPES,
        "first_seen_document_version_id": _TEXT_TYPES,
        "superseded_by_version": _TEXT_TYPES,
        "superseded_at": {"timestamp with time zone"},
        "created_at": {"timestamp with time zone"},
        "updated_at": {"timestamp with time zone"},
    },
    "requirement_evidence": {
        "evidence_id": _TEXT_TYPES,
        "requirement_id": _TEXT_TYPES,
        "requirement_display_id": _TEXT_TYPES,
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
        "requirement_display_id": _TEXT_TYPES,
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
    "schema_migrations": {
        "version": _TEXT_TYPES,
        "applied_at": {"timestamp with time zone"},
    },
    "artifact_display_ids": {
        "project": _TEXT_TYPES,
        "kind": _TEXT_TYPES,
        "internal_id": _TEXT_TYPES,
        "display_id": _TEXT_TYPES,
        "created_at": {"timestamp with time zone"},
    },
    "display_id_counters": {
        "project": _TEXT_TYPES,
        "kind": _TEXT_TYPES,
        "next_value": {"integer"},
        "updated_at": {"timestamp with time zone"},
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
                    create table if not exists schema_migrations (
                      version text primary key,
                      applied_at timestamptz not null default now()
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
                    create table if not exists test_scenario_artifacts (
                      artifact_path text primary key,
                      document_version_id text not null,
                      payload jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists artifact_display_ids (
                      project text not null,
                      kind text not null,
                      internal_id text primary key,
                      display_id text not null,
                      created_at timestamptz not null default now(),
                      unique(project, display_id)
                    );
                    create table if not exists display_id_counters (
                      project text not null,
                      kind text not null,
                      next_value integer not null,
                      updated_at timestamptz not null default now(),
                      primary key(project, kind)
                    );
                    create table if not exists user_stories (
                      story_id text primary key,
                      display_id text not null default '',
                      requirement_id text not null,
                      requirement_display_id text not null default '',
                      requirement_revision_id text not null default '',
                      source_req_id text,
                      project text not null,
                      document_id text not null,
                      document_version_id text not null,
                      doc_version text not null,
                      origin_version text not null default '',
                      title text not null,
                      priority text not null,
                      confidence double precision not null default 0,
                      status text not null,
                      origin text not null default 'generation',
                      generation_context_run_id text not null default '',
                      run_id text not null default '',
                      payload jsonb not null,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
                    );
                    create table if not exists user_story_evidence (
                      evidence_id text primary key,
                      story_id text not null,
                      story_display_id text not null default '',
                      requirement_id text not null,
                      requirement_display_id text not null default '',
                      document_version_id text not null,
                      chunk_id text not null,
                      run_id text not null default '',
                      created_at timestamptz not null default now()
                    );
                    create table if not exists test_scenarios (
                      scenario_id text primary key,
                      display_id text not null default '',
                      story_id text not null,
                      story_display_id text not null default '',
                      requirement_id text not null,
                      requirement_display_id text not null default '',
                      requirement_revision_id text not null default '',
                      source_req_id text,
                      project text not null,
                      document_id text not null,
                      document_version_id text not null,
                      doc_version text not null,
                      origin_version text not null default '',
                      title text not null,
                      scenario_type text not null,
                      priority text not null,
                      confidence double precision not null default 0,
                      status text not null,
                      origin text not null default 'generation',
                      generation_context_run_id text not null default '',
                      run_id text not null default '',
                      payload jsonb not null,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
                    );
                    create table if not exists test_scenario_evidence (
                      evidence_id text primary key,
                      scenario_id text not null,
                      scenario_display_id text not null default '',
                      story_id text not null,
                      story_display_id text not null default '',
                      requirement_id text not null,
                      requirement_display_id text not null default '',
                      document_version_id text not null,
                      chunk_id text not null,
                      run_id text not null default '',
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
                      display_id text not null default '',
                      project text not null,
                      document_id text not null,
                      requirement_key text not null,
                      source_req_id text,
                      id_generation_type text not null default 'generated',
                      confidence double precision not null default 0,
                      status text not null,
                      first_seen_document_version_id text not null,
                      active_revision_id text,
                      superseded_by_version text,
                      superseded_at timestamptz,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now(),
                      unique(project, document_id, requirement_key)
                    );
                    create table if not exists requirement_revisions (
                      revision_id text primary key,
                      requirement_id text not null,
                      display_id text not null default '',
                      statement text not null,
                      normalized_statement text not null,
                      requirement_type text not null,
                      priority text not null,
                      source_req_id text,
                      id_generation_type text not null default 'generated',
                      confidence double precision not null default 0,
                      status text not null,
                      first_seen_document_version_id text not null,
                      superseded_by_version text,
                      superseded_at timestamptz,
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
                    );
                    create table if not exists requirement_evidence (
                      evidence_id text primary key,
                      requirement_id text not null,
                      requirement_display_id text not null default '',
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
                      requirement_display_id text not null default '',
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
                    create table if not exists generation_context_runs (
                      context_run_id text primary key,
                      stage text not null,
                      anchor_id text not null default '',
                      project text not null default '',
                      document_version_id text not null,
                      source text not null default '',
                      metrics jsonb not null,
                      created_at timestamptz not null default now()
                    );
                    create table if not exists generation_context_items (
                      context_run_id text not null,
                      rank integer not null,
                      item_type text not null default 'chunk',
                      item_id text not null,
                      source text not null,
                      score double precision,
                      selected boolean not null default true,
                      created_at timestamptz not null default now(),
                      primary key(context_run_id, rank)
                    );
                    create index if not exists generation_context_runs_anchor_idx
                      on generation_context_runs(stage, anchor_id);
                    create index if not exists generation_context_items_item_idx
                      on generation_context_items(item_id);
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
                    create index if not exists test_scenarios_story_idx
                      on test_scenarios(story_id);
                    create index if not exists test_scenarios_requirement_idx
                      on test_scenarios(requirement_id);
                    create index if not exists test_scenarios_document_version_idx
                      on test_scenarios(document_version_id);
                    create index if not exists artifact_display_ids_project_kind_idx
                      on artifact_display_ids(project, kind);
                    create index if not exists user_story_evidence_story_idx
                      on user_story_evidence(story_id);
                    create index if not exists user_story_evidence_chunk_idx
                      on user_story_evidence(chunk_id);
                    create index if not exists user_story_evidence_document_version_idx
                      on user_story_evidence(document_version_id);
                    create index if not exists test_scenario_evidence_scenario_idx
                      on test_scenario_evidence(scenario_id);
                    create index if not exists test_scenario_evidence_chunk_idx
                      on test_scenario_evidence(chunk_id);
                    create index if not exists test_scenario_evidence_document_version_idx
                      on test_scenario_evidence(document_version_id);
                    alter table user_stories
                      add column if not exists display_id text not null default '',
                      add column if not exists requirement_display_id text not null default '',
                      add column if not exists requirement_revision_id text not null default '',
                      add column if not exists source_req_id text,
                      add column if not exists origin_version text not null default '',
                      add column if not exists confidence double precision not null default 0,
                      add column if not exists updated_at timestamptz not null default now();
                    alter table user_story_evidence
                      add column if not exists story_display_id text not null default '',
                      add column if not exists requirement_display_id text not null default '';
                    alter table test_scenarios
                      add column if not exists display_id text not null default '',
                      add column if not exists story_display_id text not null default '',
                      add column if not exists requirement_display_id text not null default '',
                      add column if not exists requirement_revision_id text not null default '',
                      add column if not exists source_req_id text,
                      add column if not exists origin_version text not null default '',
                      add column if not exists confidence double precision not null default 0,
                      add column if not exists updated_at timestamptz not null default now();
                    alter table test_scenario_evidence
                      add column if not exists scenario_display_id text not null default '',
                      add column if not exists story_display_id text not null default '',
                      add column if not exists requirement_display_id text not null default '';
                    alter table requirements
                      add column if not exists display_id text not null default '',
                      add column if not exists source_req_id text,
                      add column if not exists id_generation_type text not null default 'generated',
                      add column if not exists confidence double precision not null default 0,
                      add column if not exists superseded_by_version text,
                      add column if not exists superseded_at timestamptz;
                    alter table requirement_revisions
                      add column if not exists display_id text not null default '',
                      add column if not exists source_req_id text,
                      add column if not exists id_generation_type text not null default 'generated',
                      add column if not exists confidence double precision not null default 0,
                      add column if not exists superseded_by_version text,
                      add column if not exists superseded_at timestamptz;
                    alter table requirement_evidence
                      add column if not exists requirement_display_id text not null default '';
                    alter table requirement_fact_links
                      add column if not exists requirement_display_id text not null default '';
                    alter table generation_context_items
                      add column if not exists assertion_id text,
                      add column if not exists text_unit_id text,
                      add column if not exists entity_id text,
                      add column if not exists predicate text,
                      add column if not exists hop_count integer,
                      add column if not exists normalized_score double precision,
                      add column if not exists reranker_score double precision,
                      add column if not exists mandatory boolean not null default false,
                      add column if not exists metadata_json jsonb not null default '{}'::jsonb;
                    create index if not exists generation_context_items_assertion_idx
                      on generation_context_items(assertion_id);
                    alter table user_stories
                      add column if not exists generation_context_run_id text not null default '';
                    alter table test_scenarios
                      add column if not exists generation_context_run_id text not null default '';
                    create index if not exists user_stories_generation_context_idx
                      on user_stories(generation_context_run_id);
                    create index if not exists test_scenarios_generation_context_idx
                      on test_scenarios(generation_context_run_id);
                    insert into schema_migrations(version)
                    values ('0001_version_lineage_inline'),
                           ('0002_context_item_assertion_columns'),
                           ('0003_generation_context_run_id')
                    on conflict (version) do nothing;
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
    ) -> RequirementArtifact:
        internal_ids = [requirement.requirement_id for requirement in artifact.requirements]
        if self.settings.postgres.mode == "local_json":
            artifact = apply_requirement_display_ids(
                artifact,
                resolve_many(artifact.project, "requirement", internal_ids),
            )
            payload = artifact.model_dump(mode="json")
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
            return artifact
        with self._connect() as connection:
            with connection.cursor() as cursor:
                artifact = apply_requirement_display_ids(
                    artifact,
                    resolve_many(artifact.project, "requirement", internal_ids, cursor=cursor),
                )
                payload = artifact.model_dump(mode="json")
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
        return artifact

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
        artifact: UserStoryArtifact | UserStoryBuildResult,
        artifact_path: str,
        run_id: str,
    ) -> UserStoryBuildResult:
        result = _coerce_user_story_build_result(artifact)
        if self.settings.postgres.mode == "local_json":
            _apply_user_story_display_ids(
                result,
                story_display_ids=resolve_many(
                    result.artifact.project,
                    "user_story",
                    list(result.records),
                ),
                requirement_display_ids=resolve_many(
                    result.artifact.project,
                    "requirement",
                    [record.requirement_id for record in result.records.values()],
                ),
            )
            payload = result.artifact.model_dump(mode="json")
            self._upsert_local(
                "user_story_artifact",
                result.artifact.document_version_id,
                {
                    "kind": "user_story_artifact",
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "artifact": payload,
                },
            )
            self._persist_user_stories_local(result, run_id)
            return result
        with self._connect() as connection:
            with connection.cursor() as cursor:
                _apply_user_story_display_ids(
                    result,
                    story_display_ids=resolve_many(
                        result.artifact.project,
                        "user_story",
                        list(result.records),
                        cursor=cursor,
                    ),
                    requirement_display_ids=resolve_many(
                        result.artifact.project,
                        "requirement",
                        [record.requirement_id for record in result.records.values()],
                        cursor=cursor,
                    ),
                )
                payload = result.artifact.model_dump(mode="json")
                cursor.execute(
                    """
                    insert into user_story_artifacts
                      (artifact_path, document_version_id, payload)
                    values (%s, %s, %s)
                    on conflict (artifact_path) do update set payload=excluded.payload
                    """,
                    (artifact_path, result.artifact.document_version_id, json.dumps(payload)),
                )
                self._persist_user_stories_postgres(cursor, result, run_id)
            connection.commit()
        return result

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

    def _persist_user_stories_postgres(
        self, cursor: Any, artifact: UserStoryBuildResult, run_id: str
    ) -> None:
        for story_id, record in artifact.records.items():
            cursor.execute(
                """
                insert into user_stories
                  (story_id, display_id, requirement_id, requirement_display_id,
                   requirement_revision_id, source_req_id, project, document_id,
                   document_version_id, doc_version, origin_version, title, priority,
                   confidence, status, origin, generation_context_run_id, run_id, payload)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (story_id) do update
                set display_id = excluded.display_id,
                    requirement_display_id = excluded.requirement_display_id,
                    requirement_revision_id = excluded.requirement_revision_id,
                    source_req_id = excluded.source_req_id,
                    document_version_id = excluded.document_version_id,
                    doc_version = excluded.doc_version,
                    origin_version = excluded.origin_version,
                    title = excluded.title,
                    priority = excluded.priority,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    origin = excluded.origin,
                    generation_context_run_id = excluded.generation_context_run_id,
                    run_id = excluded.run_id,
                    payload = excluded.payload,
                    updated_at = now()
                """,
                (
                    story_id,
                    record.display_id,
                    record.requirement_id,
                    record.requirement_display_id,
                    record.requirement_revision_id,
                    record.source_req_id,
                    record.project,
                    record.document_id,
                    record.document_version_id,
                    record.doc_version,
                    record.origin_version,
                    record.title,
                    record.priority,
                    record.confidence,
                    record.status,
                    record.origin,
                    record.generation_context_run_id,
                    run_id,
                    json.dumps(record.model_dump(mode="json")),
                ),
            )
            for chunk_id in record.evidence_chunk_ids:
                cursor.execute(
                    """
                    insert into user_story_evidence
                      (evidence_id, story_id, story_display_id, requirement_id,
                       requirement_display_id, document_version_id, chunk_id, run_id)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (evidence_id) do nothing
                    """,
                    (
                        user_story_evidence_id(
                            story_identifier=story_id,
                            document_version_identifier=record.document_version_id,
                            chunk_identifier=chunk_id,
                        ),
                        story_id,
                        record.display_id,
                        record.requirement_id,
                        record.requirement_display_id,
                        record.document_version_id,
                        chunk_id,
                        run_id,
                    ),
                )

    def _persist_user_stories_local(self, artifact: UserStoryBuildResult, run_id: str) -> None:
        for story_id, record in artifact.records.items():
            self._upsert_local(
                "user_story",
                story_id,
                {
                    "kind": "user_story",
                    "story_id": story_id,
                    "display_id": record.display_id,
                    "requirement_id": record.requirement_id,
                    "requirement_display_id": record.requirement_display_id,
                    "requirement_revision_id": record.requirement_revision_id,
                    "source_req_id": record.source_req_id,
                    "project": record.project,
                    "document_id": record.document_id,
                    "document_version_id": record.document_version_id,
                    "doc_version": record.doc_version,
                    "origin_version": record.origin_version,
                    "title": record.title,
                    "priority": record.priority,
                    "confidence": record.confidence,
                    "status": record.status,
                    "origin": record.origin,
                    "generation_context_run_id": record.generation_context_run_id,
                    "run_id": run_id,
                    "user_story": record.model_dump(mode="json"),
                },
            )
            for chunk_id in record.evidence_chunk_ids:
                self._upsert_local(
                    "user_story_evidence",
                    user_story_evidence_id(
                        story_identifier=story_id,
                        document_version_identifier=record.document_version_id,
                        chunk_identifier=chunk_id,
                    ),
                    {
                        "kind": "user_story_evidence",
                        "story_id": story_id,
                        "story_display_id": record.display_id,
                        "requirement_id": record.requirement_id,
                        "requirement_display_id": record.requirement_display_id,
                        "document_version_id": record.document_version_id,
                        "chunk_id": chunk_id,
                        "run_id": run_id,
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

    def persist_test_scenario_artifact(
        self,
        artifact: TestScenarioArtifact | TestScenarioBuildResult,
        artifact_path: str,
        run_id: str,
    ) -> TestScenarioBuildResult:
        result = _coerce_test_scenario_build_result(artifact)
        if self.settings.postgres.mode == "local_json":
            _apply_test_scenario_display_ids(
                result,
                scenario_display_ids=resolve_many(
                    result.artifact.project,
                    "test_scenario",
                    list(result.records),
                ),
                story_display_ids=resolve_many(
                    result.artifact.project,
                    "user_story",
                    [record.story_id for record in result.records.values()],
                ),
                requirement_display_ids=resolve_many(
                    result.artifact.project,
                    "requirement",
                    [record.requirement_id for record in result.records.values()],
                ),
            )
            payload = result.artifact.model_dump(mode="json")
            self._upsert_local(
                "test_scenario_artifact",
                result.artifact.document_version_id,
                {
                    "kind": "test_scenario_artifact",
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "artifact": payload,
                },
            )
            self._persist_test_scenarios_local(result, run_id)
            return result
        with self._connect() as connection:
            with connection.cursor() as cursor:
                _apply_test_scenario_display_ids(
                    result,
                    scenario_display_ids=resolve_many(
                        result.artifact.project,
                        "test_scenario",
                        list(result.records),
                        cursor=cursor,
                    ),
                    story_display_ids=resolve_many(
                        result.artifact.project,
                        "user_story",
                        [record.story_id for record in result.records.values()],
                        cursor=cursor,
                    ),
                    requirement_display_ids=resolve_many(
                        result.artifact.project,
                        "requirement",
                        [record.requirement_id for record in result.records.values()],
                        cursor=cursor,
                    ),
                )
                payload = result.artifact.model_dump(mode="json")
                cursor.execute(
                    """
                    insert into test_scenario_artifacts
                      (artifact_path, document_version_id, payload)
                    values (%s, %s, %s)
                    on conflict (artifact_path) do update set payload=excluded.payload
                    """,
                    (artifact_path, result.artifact.document_version_id, json.dumps(payload)),
                )
                self._persist_test_scenarios_postgres(cursor, result, run_id)
            connection.commit()
        return result

    def load_test_scenario_artifact_payload(
        self,
        artifact_path: str | None = None,
        document_version_id: str | None = None,
    ) -> dict[str, Any] | None:
        if artifact_path is None and document_version_id is None:
            raise ValueError("artifact_path or document_version_id is required")
        if self.settings.postgres.mode == "local_json":
            return self._local_test_scenario_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )
        with self._connect() as connection, connection.cursor() as cursor:
            if artifact_path is not None:
                cursor.execute(
                    "select payload from test_scenario_artifacts where artifact_path = %s",
                    (artifact_path,),
                )
                row = cursor.fetchone()
                if row:
                    return _artifact_payload_from_row(row[0])
            if document_version_id is not None:
                cursor.execute(
                    """
                    select payload from test_scenario_artifacts
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

    def _persist_test_scenarios_postgres(
        self,
        cursor: Any,
        artifact: TestScenarioBuildResult,
        run_id: str,
    ) -> None:
        for scenario_id, record in artifact.records.items():
            cursor.execute(
                """
                insert into test_scenarios
                  (scenario_id, display_id, story_id, story_display_id, requirement_id,
                   requirement_display_id, project, document_id, document_version_id,
                   doc_version, requirement_revision_id, source_req_id, origin_version,
                   title, scenario_type, priority, confidence, status, origin,
                   generation_context_run_id, run_id, payload)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (scenario_id) do update
                set display_id = excluded.display_id,
                    story_display_id = excluded.story_display_id,
                    requirement_display_id = excluded.requirement_display_id,
                    requirement_revision_id = excluded.requirement_revision_id,
                    source_req_id = excluded.source_req_id,
                    document_version_id = excluded.document_version_id,
                    doc_version = excluded.doc_version,
                    origin_version = excluded.origin_version,
                    title = excluded.title,
                    scenario_type = excluded.scenario_type,
                    priority = excluded.priority,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    origin = excluded.origin,
                    generation_context_run_id = excluded.generation_context_run_id,
                    run_id = excluded.run_id,
                    payload = excluded.payload,
                    updated_at = now()
                """,
                (
                    scenario_id,
                    record.display_id,
                    record.story_id,
                    record.story_display_id,
                    record.requirement_id,
                    record.requirement_display_id,
                    record.project,
                    record.document_id,
                    record.document_version_id,
                    record.doc_version,
                    record.requirement_revision_id,
                    record.source_req_id,
                    record.origin_version,
                    record.title,
                    record.scenario_type,
                    record.priority,
                    record.confidence,
                    record.status,
                    record.origin,
                    record.generation_context_run_id,
                    run_id,
                    json.dumps(record.model_dump(mode="json")),
                ),
            )
            for chunk_id in record.evidence_chunk_ids:
                cursor.execute(
                    """
                    insert into test_scenario_evidence
                      (evidence_id, scenario_id, scenario_display_id, story_id,
                       story_display_id, requirement_id, requirement_display_id,
                       document_version_id, chunk_id, run_id)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (evidence_id) do nothing
                    """,
                    (
                        test_scenario_evidence_id(
                            scenario_identifier=scenario_id,
                            document_version_identifier=record.document_version_id,
                            chunk_identifier=chunk_id,
                        ),
                        scenario_id,
                        record.display_id,
                        record.story_id,
                        record.story_display_id,
                        record.requirement_id,
                        record.requirement_display_id,
                        record.document_version_id,
                        chunk_id,
                        run_id,
                    ),
                )

    def _persist_test_scenarios_local(self, artifact: TestScenarioBuildResult, run_id: str) -> None:
        for scenario_id, record in artifact.records.items():
            self._upsert_local(
                "test_scenario",
                scenario_id,
                {
                    "kind": "test_scenario",
                    "scenario_id": scenario_id,
                    "display_id": record.display_id,
                    "story_id": record.story_id,
                    "story_display_id": record.story_display_id,
                    "requirement_id": record.requirement_id,
                    "requirement_display_id": record.requirement_display_id,
                    "requirement_revision_id": record.requirement_revision_id,
                    "source_req_id": record.source_req_id,
                    "project": record.project,
                    "document_id": record.document_id,
                    "document_version_id": record.document_version_id,
                    "doc_version": record.doc_version,
                    "origin_version": record.origin_version,
                    "title": record.title,
                    "scenario_type": record.scenario_type,
                    "priority": record.priority,
                    "confidence": record.confidence,
                    "status": record.status,
                    "origin": record.origin,
                    "generation_context_run_id": record.generation_context_run_id,
                    "run_id": run_id,
                    "test_scenario": record.model_dump(mode="json"),
                },
            )
            for chunk_id in record.evidence_chunk_ids:
                self._upsert_local(
                    "test_scenario_evidence",
                    test_scenario_evidence_id(
                        scenario_identifier=scenario_id,
                        document_version_identifier=record.document_version_id,
                        chunk_identifier=chunk_id,
                    ),
                    {
                        "kind": "test_scenario_evidence",
                        "scenario_id": scenario_id,
                        "scenario_display_id": record.display_id,
                        "story_id": record.story_id,
                        "story_display_id": record.story_display_id,
                        "requirement_id": record.requirement_id,
                        "requirement_display_id": record.requirement_display_id,
                        "document_version_id": record.document_version_id,
                        "chunk_id": chunk_id,
                        "run_id": run_id,
                    },
                )

    def _local_test_scenario_artifact_payload(
        self,
        *,
        artifact_path: str | None,
        document_version_id: str | None,
    ) -> dict[str, Any] | None:
        rows = [
            row for row in self._read_local_rows() if row.get("kind") == "test_scenario_artifact"
        ]
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

    def load_coverage_status(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Per-requirement coverage: no_story / story_covered / scenario_covered.

        Computed purely from existing relational columns (`user_stories.requirement_id`,
        `test_scenarios.story_id`); no coverage table required.
        """
        if self.settings.postgres.mode == "local_json":
            return self._local_coverage_status(
                project=project, document_version_id=document_version_id
            )
        with self._connect() as connection, connection.cursor() as cursor:
            # Denominator = active requirement lineages for the project. When a
            # document version is requested, scope membership through evidence
            # *presence in that version* (distinct requirement_evidence rows), not
            # first_seen_document_version_id, because a lineage that first appeared
            # in an earlier version can legitimately recur in a later one.
            if document_version_id is not None:
                cursor.execute(
                    """
                    select r.requirement_id, r.status
                    from requirements r
                    where r.project = %s
                      and r.status = 'active'
                      and exists (
                          select 1 from requirement_evidence re
                          where re.requirement_id = r.requirement_id
                            and re.document_version_id = %s
                      )
                    """,
                    (project, document_version_id),
                )
            else:
                cursor.execute(
                    "select requirement_id, status from requirements "
                    "where project = %s and status = 'active'",
                    (project,),
                )
            requirements = [(str(r[0]), str(r[1])) for r in cursor.fetchall()]

            story_sql = (
                "select story_id, requirement_id from user_stories "
                "where project = %s and status = 'active'"
            )
            story_params: list[str] = [project]
            if document_version_id is not None:
                story_sql += " and document_version_id = %s"
                story_params.append(document_version_id)
            cursor.execute(story_sql, tuple(story_params))
            stories = [(str(r[0]), str(r[1])) for r in cursor.fetchall()]

            scenario_sql = (
                "select story_id from test_scenarios where project = %s and status = 'active'"
            )
            scenario_params: list[str] = [project]
            if document_version_id is not None:
                scenario_sql += " and document_version_id = %s"
                scenario_params.append(document_version_id)
            cursor.execute(scenario_sql, tuple(scenario_params))
            scenario_story_ids = [str(r[0]) for r in cursor.fetchall()]
        return _compute_coverage(requirements, stories, scenario_story_ids)

    def load_coverage_report(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> CoverageReport:
        """Strict per-requirement coverage with a deterministic summary rollup.

        Identical version-scoping semantics to :meth:`load_coverage_status` in
        both PostgreSQL and local-JSON modes.
        """
        rows = self.load_coverage_status(project=project, document_version_id=document_version_id)
        return build_coverage_report(
            project=project,
            document_version_id=document_version_id,
            rows=rows,
        )

    def _local_coverage_status(
        self,
        *,
        project: str,
        document_version_id: str | None,
    ) -> list[dict[str, Any]]:
        rows = self._read_local_rows()
        requirement_ids_in_version: set[str] | None = None
        if document_version_id is not None:
            requirement_ids_in_version = {
                str(row.get("requirement_id"))
                for row in rows
                if row.get("kind") == "requirement_evidence"
                and row.get("project") == project
                and row.get("document_version_id") == document_version_id
            }
        requirements = [
            (str(row.get("requirement_id")), str(row.get("status", "active")))
            for row in rows
            if row.get("kind") == "requirement"
            and row.get("project") == project
            and str(row.get("status", "active")) == "active"
            and (
                requirement_ids_in_version is None
                or str(row.get("requirement_id")) in requirement_ids_in_version
            )
        ]

        def _matches_version(row: dict[str, Any]) -> bool:
            return (
                document_version_id is None or row.get("document_version_id") == document_version_id
            )

        stories = [
            (str(row.get("story_id")), str(row.get("requirement_id")))
            for row in rows
            if row.get("kind") == "user_story"
            and row.get("project") == project
            and row.get("status", "active") == "active"
            and _matches_version(row)
        ]
        scenario_story_ids = [
            str(row.get("story_id"))
            for row in rows
            if row.get("kind") == "test_scenario"
            and row.get("project") == project
            and row.get("status", "active") == "active"
            and _matches_version(row)
        ]
        return _compute_coverage(requirements, stories, scenario_story_ids)

    def upsert_requirement(
        self,
        artifact: RequirementArtifact,
        artifact_path: str,
        run_id: str,
    ) -> None:
        self.persist_artifact(artifact, artifact_path, run_id)

    def upsert_user_story(
        self,
        artifact: UserStoryArtifact,
        artifact_path: str,
        run_id: str,
    ) -> None:
        self.persist_user_story_artifact(artifact, artifact_path, run_id)

    def upsert_test_scenario(
        self,
        artifact: TestScenarioArtifact,
        artifact_path: str,
        run_id: str,
    ) -> None:
        self.persist_test_scenario_artifact(artifact, artifact_path, run_id)

    def mark_requirement_superseded(
        self,
        *,
        requirement_id: str,
        revision_id: str,
        superseded_by_version: str,
    ) -> None:
        if self.settings.postgres.mode == "local_json":
            self._update_local_revision_status(revision_id, "superseded")
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    update requirement_revisions
                    set status = 'superseded',
                        superseded_by_version = %s,
                        superseded_at = now(),
                        updated_at = now()
                    where requirement_id = %s and revision_id = %s
                    """,
                    (superseded_by_version, requirement_id, revision_id),
                )
            connection.commit()

    def hard_delete_requirement(self, requirement_id: str) -> None:
        """
        Delete strictly outdated requirement.
        Child user stories and test scenarios are removed in the same transaction.
        TODO: Add approval gate before destructive cascade delete.
        """
        if self.settings.postgres.mode == "local_json":
            self._hard_delete_requirement_local(requirement_id)
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "delete from test_scenario_evidence where requirement_id = %s",
                    (requirement_id,),
                )
                delete_tables = (
                    "test_scenarios",
                    "user_story_evidence",
                    "user_stories",
                    "requirement_fact_links",
                    "requirement_evidence",
                    "requirement_delta_events",
                    "requirement_revisions",
                    "requirements",
                )
                for table_name in delete_tables:
                    cursor.execute(
                        f"delete from {table_name} where requirement_id = %s",
                        (requirement_id,),
                    )
            connection.commit()

    def load_requirements_for_generation(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> list[RequirementInput]:
        if self.settings.postgres.mode == "local_json":
            return self._load_requirements_for_generation_local(
                project=project,
                document_version_id=document_version_id,
            )
        sql = """
            select r.requirement_id,
                   rr.revision_id,
                   rr.display_id,
                   rr.source_req_id,
                   rr.id_generation_type,
                   rr.confidence,
                   rr.statement,
                   rr.requirement_type,
                   rr.priority,
                   coalesce(string_agg(distinct re.chunk_id, ','), '') as chunk_ids
            from requirements r
            join requirement_revisions rr on rr.revision_id = r.active_revision_id
            left join requirement_evidence re on re.revision_id = rr.revision_id
            where r.project = %s
              and r.status = 'active'
              and rr.status = 'active'
        """
        params: list[str] = [project]
        if document_version_id is not None:
            sql += " and rr.first_seen_document_version_id = %s"
            params.append(document_version_id)
        sql += """
            group by r.requirement_id, rr.revision_id, rr.display_id, rr.source_req_id,
                     rr.id_generation_type, rr.confidence, rr.statement, rr.requirement_type,
                     rr.priority
            order by r.requirement_id
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [
            RequirementInput(
                requirement_id=str(row[0]),
                revision_id=str(row[1]),
                display_id=str(row[2]),
                source_req_id=str(row[3]) if row[3] is not None else None,
                id_generation_type=_id_generation_type(row[4]),
                confidence=float(row[5]),
                requirement_text=str(row[6]),
                requirement_type=str(row[7]),
                priority=normalize_priority_label(row[8]),
                evidence_chunk_ids=[chunk for chunk in str(row[9]).split(",") if chunk],
            )
            for row in rows
        ]

    def load_user_stories_for_generation(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> list[UserStoryRecord]:
        if self.settings.postgres.mode == "local_json":
            return self._load_user_stories_for_generation_local(
                project=project,
                document_version_id=document_version_id,
            )
        sql = "select payload from user_stories where project = %s and status = 'active'"
        params: list[str] = [project]
        if document_version_id is not None:
            sql += " and document_version_id = %s"
            params.append(document_version_id)
        sql += " order by story_id"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [UserStoryRecord.model_validate(_artifact_payload_from_row(row[0])) for row in rows]

    def load_test_scenarios_for_generation(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> list[TestScenarioRecord]:
        if self.settings.postgres.mode == "local_json":
            return self._load_test_scenarios_for_generation_local(
                project=project,
                document_version_id=document_version_id,
            )
        sql = "select payload from test_scenarios where project = %s and status = 'active'"
        params: list[str] = [project]
        if document_version_id is not None:
            sql += " and document_version_id = %s"
            params.append(document_version_id)
        sql += " order by scenario_id"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [
            TestScenarioRecord.model_validate(_artifact_payload_from_row(row[0])) for row in rows
        ]

    def load_artifact_from_postgres(
        self,
        *,
        artifact_kind: str,
        artifact_path: str | None = None,
        document_version_id: str | None = None,
    ) -> dict[str, Any] | None:
        if artifact_kind in {"requirements", "requirements_full"}:
            return self.load_requirement_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )
        if artifact_kind == "user_stories":
            return self.load_user_story_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )
        if artifact_kind == "test_scenarios":
            return self.load_test_scenario_artifact_payload(
                artifact_path=artifact_path,
                document_version_id=document_version_id,
            )
        raise ValueError(f"unsupported artifact kind: {artifact_kind}")

    def load_artifact_payloads_for_project(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.settings.postgres.mode == "local_json":
            return self._load_artifact_payloads_for_project_local(
                project=project,
                document_version_id=document_version_id,
            )
        params: list[str] = [project]
        version_filter = ""
        if document_version_id is not None:
            version_filter = " and a.document_version_id = %s"
            params.append(document_version_id)
        queries = [
            (
                "requirements_full",
                "requirement_artifacts",
            ),
            ("user_stories", "user_story_artifacts"),
            ("test_scenarios", "test_scenario_artifacts"),
        ]
        rows: list[dict[str, Any]] = []
        with self._connect() as connection, connection.cursor() as cursor:
            for kind, table in queries:
                cursor.execute(
                    f"""
                    select %s as artifact_kind, a.artifact_path, a.document_version_id, a.payload
                    from {table} a
                    join document_versions dv
                      on dv.document_version_id = a.document_version_id
                    where dv.project = %s{version_filter}
                    order by a.created_at desc
                    """,
                    (kind, *params),
                )
                for row in cursor.fetchall():
                    rows.append(
                        {
                            "artifact_kind": str(row[0]),
                            "artifact_path": str(row[1]),
                            "document_version_id": str(row[2]),
                            "payload": _artifact_payload_from_row(row[3]),
                        }
                    )
        return rows

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

    def record_generation_context(
        self,
        run: GenerationContextRun,
        items: list[GenerationContextItem],
    ) -> None:
        """Persist one retrieval snapshot (shadow comparison / audit trail)."""
        if self.settings.postgres.mode == "local_json":
            self._upsert_local(
                "generation_context_run",
                run.context_run_id,
                {
                    "kind": "generation_context_run",
                    **run.model_dump(mode="json"),
                    "items": [item.model_dump(mode="json") for item in items],
                },
            )
            return
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into generation_context_runs
                      (context_run_id, stage, anchor_id, project,
                       document_version_id, source, metrics)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    on conflict (context_run_id) do update
                    set source=excluded.source, metrics=excluded.metrics
                    """,
                    (
                        run.context_run_id,
                        run.stage,
                        run.anchor_id,
                        run.project,
                        run.document_version_id,
                        run.source,
                        json.dumps(run.metrics),
                    ),
                )
                for item in items:
                    cursor.execute(
                        """
                        insert into generation_context_items
                          (context_run_id, rank, item_type, item_id, source, score, selected,
                           assertion_id, text_unit_id, entity_id, predicate, hop_count,
                           normalized_score, reranker_score, mandatory, metadata_json)
                        values (%s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        on conflict (context_run_id, rank) do update
                        set item_type=excluded.item_type, item_id=excluded.item_id,
                            source=excluded.source, score=excluded.score,
                            selected=excluded.selected, assertion_id=excluded.assertion_id,
                            text_unit_id=excluded.text_unit_id, entity_id=excluded.entity_id,
                            predicate=excluded.predicate, hop_count=excluded.hop_count,
                            normalized_score=excluded.normalized_score,
                            reranker_score=excluded.reranker_score,
                            mandatory=excluded.mandatory, metadata_json=excluded.metadata_json
                        """,
                        (
                            item.context_run_id,
                            item.rank,
                            item.item_type,
                            item.item_id,
                            item.source,
                            item.score,
                            item.selected,
                            item.assertion_id,
                            item.text_unit_id,
                            item.entity_id,
                            item.predicate,
                            item.hop_count,
                            item.normalized_score,
                            item.reranker_score,
                            item.mandatory,
                            json.dumps(item.metadata),
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
                  (requirement_id, display_id, project, document_id, requirement_key,
                   source_req_id, id_generation_type, confidence, status,
                   first_seen_document_version_id, active_revision_id)
                values (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
                on conflict (requirement_id) do update
                set display_id = excluded.display_id,
                    source_req_id = excluded.source_req_id,
                    id_generation_type = excluded.id_generation_type,
                    confidence = excluded.confidence,
                    status = 'active',
                    active_revision_id = excluded.active_revision_id,
                    updated_at = now()
                """,
                (
                    requirement.requirement_id,
                    requirement.display_id,
                    artifact.project,
                    artifact.document_id,
                    requirement.requirement_key,
                    requirement.source_req_id,
                    requirement.id_generation_type,
                    requirement.confidence,
                    artifact.document_version_id,
                    requirement.revision_id,
                ),
            )
            cursor.execute(
                """
                insert into requirement_revisions
                  (revision_id, requirement_id, display_id, statement, normalized_statement,
                   requirement_type, priority, source_req_id, id_generation_type, confidence,
                   status, first_seen_document_version_id)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s)
                on conflict (revision_id) do update
                set display_id = excluded.display_id,
                    statement = excluded.statement,
                    normalized_statement = excluded.normalized_statement,
                    requirement_type = excluded.requirement_type,
                    priority = excluded.priority,
                    source_req_id = excluded.source_req_id,
                    id_generation_type = excluded.id_generation_type,
                    confidence = excluded.confidence,
                    status = 'active',
                    updated_at = now()
                """,
                (
                    requirement.revision_id,
                    requirement.requirement_id,
                    requirement.display_id,
                    requirement.statement,
                    requirement.normalized_statement,
                    requirement.requirement_type,
                    requirement.priority,
                    requirement.source_req_id,
                    requirement.id_generation_type,
                    requirement.confidence,
                    artifact.document_version_id,
                ),
            )
            for evidence in requirement.evidence:
                trace = evidence.source_trace
                cursor.execute(
                    """
                    insert into requirement_evidence
                      (evidence_id, requirement_id, requirement_display_id, revision_id,
                       document_version_id, chunk_id, page, section, quote, start_char,
                       end_char, source_path)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (evidence_id) do update
                    set requirement_display_id = excluded.requirement_display_id,
                        quote = excluded.quote,
                        page = excluded.page,
                        section = excluded.section
                    """,
                    (
                        evidence.evidence_id,
                        requirement.requirement_id,
                        requirement.display_id,
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
                          (evidence_id, fact_id, requirement_id, requirement_display_id,
                           revision_id)
                        values (%s, %s, %s, %s, %s)
                        on conflict (evidence_id, fact_id) do nothing
                        """,
                        (
                            evidence.evidence_id,
                            fact_id,
                            requirement.requirement_id,
                            requirement.display_id,
                            requirement.revision_id,
                        ),
                    )

        for event in artifact.delta_events:
            if event.event_type == "superseded" and event.revision_id:
                cursor.execute(
                    """
                    update requirement_revisions
                    set status = 'superseded',
                        superseded_by_version = %s,
                        superseded_at = now(),
                        updated_at = now()
                    where revision_id = %s
                    """,
                    (event.document_version_id, event.revision_id),
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
                    "display_id": requirement.display_id,
                    "requirement_key": requirement.requirement_key,
                    "source_req_id": requirement.source_req_id,
                    "id_generation_type": requirement.id_generation_type,
                    "confidence": requirement.confidence,
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
                        "requirement_display_id": requirement.display_id,
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
                            "requirement_display_id": requirement.display_id,
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

    def _hard_delete_requirement_local(self, requirement_id: str) -> None:
        rows = [
            row
            for row in self._read_local_rows()
            if not _local_row_matches_requirement(row, requirement_id)
        ]
        self._write_local_rows(rows)

    def _load_requirements_for_generation_local(
        self,
        *,
        project: str,
        document_version_id: str | None,
    ) -> list[RequirementInput]:
        rows = self._read_local_rows()
        active_revision_by_requirement = {
            str(row["requirement_id"]): str(row["active_revision_id"])
            for row in rows
            if row.get("kind") == "requirement"
            and row.get("project") == project
            and row.get("status") == "active"
        }
        inputs: list[RequirementInput] = []
        for row in rows:
            if row.get("kind") != "requirement_revision" or row.get("project") != project:
                continue
            if row.get("status") != "active":
                continue
            if (
                document_version_id is not None
                and row.get("document_version_id") != document_version_id
            ):
                continue
            requirement = row.get("requirement")
            if not isinstance(requirement, dict):
                continue
            requirement_id = str(requirement["requirement_id"])
            revision_id = str(requirement["revision_id"])
            if active_revision_by_requirement.get(requirement_id) != revision_id:
                continue
            inputs.append(
                RequirementInput(
                    requirement_id=requirement_id,
                    revision_id=revision_id,
                    display_id=str(requirement.get("display_id", "")),
                    source_req_id=(
                        str(requirement.get("source_req_id"))
                        if requirement.get("source_req_id") is not None
                        else None
                    ),
                    id_generation_type=_id_generation_type(
                        requirement.get("id_generation_type", "generated")
                    ),
                    confidence=float(requirement.get("confidence", 0.0)),
                    requirement_text=str(requirement["statement"]),
                    requirement_type=str(requirement["requirement_type"]),
                    priority=normalize_priority_label(requirement["priority"]),
                    evidence_chunk_ids=[
                        str(evidence["source_trace"]["chunk_id"])
                        for evidence in requirement.get("evidence", [])
                        if isinstance(evidence, dict)
                        and isinstance(evidence.get("source_trace"), dict)
                    ],
                )
            )
        return inputs

    def _load_user_stories_for_generation_local(
        self,
        *,
        project: str,
        document_version_id: str | None,
    ) -> list[UserStoryRecord]:
        records: list[UserStoryRecord] = []
        for row in self._read_local_rows():
            if row.get("kind") != "user_story" or row.get("project") != project:
                continue
            if row.get("status", "active") != "active":
                continue
            if (
                document_version_id is not None
                and row.get("document_version_id") != document_version_id
            ):
                continue
            payload = row.get("user_story")
            if isinstance(payload, dict):
                records.append(UserStoryRecord.model_validate(payload))
        return records

    def _load_test_scenarios_for_generation_local(
        self,
        *,
        project: str,
        document_version_id: str | None,
    ) -> list[TestScenarioRecord]:
        records: list[TestScenarioRecord] = []
        for row in self._read_local_rows():
            if row.get("kind") != "test_scenario" or row.get("project") != project:
                continue
            if row.get("status", "active") != "active":
                continue
            if (
                document_version_id is not None
                and row.get("document_version_id") != document_version_id
            ):
                continue
            payload = row.get("test_scenario")
            if isinstance(payload, dict):
                records.append(TestScenarioRecord.model_validate(payload))
        return records

    def _load_artifact_payloads_for_project_local(
        self,
        *,
        project: str,
        document_version_id: str | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._read_local_rows():
            kind = row.get("kind")
            if kind not in {
                "requirement_artifact",
                "user_story_artifact",
                "test_scenario_artifact",
            }:
                continue
            artifact = row.get("artifact")
            if not isinstance(artifact, dict) or artifact.get("project") != project:
                continue
            if (
                document_version_id is not None
                and artifact.get("document_version_id") != document_version_id
            ):
                continue
            rows.append(
                {
                    "artifact_kind": _local_artifact_kind(str(kind)),
                    "artifact_path": str(row.get("artifact_path", "")),
                    "document_version_id": str(artifact.get("document_version_id", "")),
                    "payload": artifact,
                }
            )
        return rows

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


def _local_artifact_kind(kind: str) -> str:
    if kind == "requirement_artifact":
        return "requirements_full"
    if kind == "user_story_artifact":
        return "user_stories"
    if kind == "test_scenario_artifact":
        return "test_scenarios"
    return kind


def _id_generation_type(value: object) -> Literal["source", "generated"]:
    return "source" if str(value).strip().lower() == "source" else "generated"


def _coerce_user_story_build_result(
    artifact: UserStoryArtifact | UserStoryBuildResult,
) -> UserStoryBuildResult:
    if isinstance(artifact, UserStoryBuildResult):
        return artifact
    raise ValueError("persisting user stories requires internal UserStoryBuildResult records")


def _coerce_test_scenario_build_result(
    artifact: TestScenarioArtifact | TestScenarioBuildResult,
) -> TestScenarioBuildResult:
    if isinstance(artifact, TestScenarioBuildResult):
        return artifact
    raise ValueError("persisting test scenarios requires internal TestScenarioBuildResult records")


def _apply_user_story_display_ids(
    result: UserStoryBuildResult,
    *,
    story_display_ids: dict[str, str],
    requirement_display_ids: dict[str, str],
) -> None:
    records: dict[str, UserStoryRecord] = {}
    coverage: dict[str, list[str]] = {}
    for story_id, record in result.records.items():
        updated = record.model_copy(
            update={
                "display_id": story_display_ids.get(story_id, record.display_id),
                "requirement_display_id": requirement_display_ids.get(
                    record.requirement_id,
                    record.requirement_display_id,
                ),
            }
        )
        records[story_id] = updated
        coverage.setdefault(updated.requirement_id, []).append(story_id)
    result.records = records
    result.coverage = coverage
    result.artifact = project_user_story_artifact(
        project=result.artifact.project,
        document_id=result.artifact.document_id,
        document_version_id=result.artifact.document_version_id,
        doc_version=result.artifact.doc_version,
        records=records,
        requirement_display_ids=requirement_display_ids,
        story_display_ids=story_display_ids,
    )


def _apply_test_scenario_display_ids(
    result: TestScenarioBuildResult,
    *,
    scenario_display_ids: dict[str, str],
    story_display_ids: dict[str, str],
    requirement_display_ids: dict[str, str],
) -> None:
    records: dict[str, TestScenarioRecord] = {}
    coverage: dict[str, list[str]] = {}
    requirement_coverage: dict[str, list[str]] = {}
    for scenario_id, record in result.records.items():
        updated = record.model_copy(
            update={
                "display_id": scenario_display_ids.get(scenario_id, record.display_id),
                "story_display_id": story_display_ids.get(
                    record.story_id,
                    record.story_display_id,
                ),
                "requirement_display_id": requirement_display_ids.get(
                    record.requirement_id,
                    record.requirement_display_id,
                ),
            }
        )
        records[scenario_id] = updated
        coverage.setdefault(updated.story_id, []).append(scenario_id)
        requirement_coverage.setdefault(updated.requirement_id, []).append(scenario_id)
    result.records = records
    result.coverage = coverage
    result.requirement_coverage = requirement_coverage
    result.artifact = project_test_scenario_artifact(
        project=result.artifact.project,
        document_id=result.artifact.document_id,
        document_version_id=result.artifact.document_version_id,
        doc_version=result.artifact.doc_version,
        records=records,
        scenario_display_ids=scenario_display_ids,
    )


def _local_row_matches_requirement(row: dict[str, Any], requirement_id: str) -> bool:
    if row.get("requirement_id") == requirement_id:
        return True
    if row.get("kind") == "requirement" and row.get("_local_key") == requirement_id:
        return True
    for key in ("requirement", "user_story", "test_scenario", "event"):
        payload = row.get(key)
        if isinstance(payload, dict) and payload.get("requirement_id") == requirement_id:
            return True
    return False


def _compute_coverage(
    requirements: list[tuple[str, str]],
    stories: list[tuple[str, str]],
    scenario_story_ids: list[str],
) -> list[dict[str, Any]]:
    """Derive per-requirement coverage status from flat relational rows.

    ``requirements`` = (requirement_id, requirement_status);
    ``stories`` = (story_id, requirement_id); ``scenario_story_ids`` = parent story of
    each active scenario. Pure and deterministic for unit testing.
    """
    stories_by_requirement: dict[str, list[str]] = {}
    for story_id, requirement_id in stories:
        stories_by_requirement.setdefault(requirement_id, []).append(story_id)

    scenario_count_by_story: dict[str, int] = {}
    for story_id in scenario_story_ids:
        scenario_count_by_story[story_id] = scenario_count_by_story.get(story_id, 0) + 1

    result: list[dict[str, Any]] = []
    for requirement_id, requirement_status in sorted(requirements):
        story_ids = stories_by_requirement.get(requirement_id, [])
        if not story_ids:
            coverage_status = "no_story"
        elif all(scenario_count_by_story.get(story_id, 0) > 0 for story_id in story_ids):
            coverage_status = "scenario_covered"
        else:
            coverage_status = "story_covered"
        result.append(
            {
                "requirement_id": requirement_id,
                "requirement_status": requirement_status,
                "coverage_status": coverage_status,
                "story_ids": story_ids,
                "scenario_count": sum(
                    scenario_count_by_story.get(story_id, 0) for story_id in story_ids
                ),
            }
        )
    return result


def _safe_percentage(numerator: int, denominator: int) -> float:
    """Zero-safe percentage rounded deterministically to two decimals."""
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def build_coverage_report(
    *,
    project: str,
    document_version_id: str | None,
    rows: list[dict[str, Any]],
) -> CoverageReport:
    """Roll per-requirement coverage rows into a strict, zero-safe report."""
    requirement_rows = [
        CoverageRequirementRow(
            requirement_id=str(row["requirement_id"]),
            requirement_status=str(row["requirement_status"]),
            coverage_status=str(row["coverage_status"]),
            story_ids=[str(sid) for sid in row.get("story_ids", [])],
            scenario_count=int(row.get("scenario_count", 0)),
        )
        for row in rows
    ]
    total = len(requirement_rows)
    with_stories = sum(
        1
        for row in requirement_rows
        if row.coverage_status in ("story_covered", "scenario_covered")
    )
    scenario_covered = sum(
        1 for row in requirement_rows if row.coverage_status == "scenario_covered"
    )
    no_story = sum(1 for row in requirement_rows if row.coverage_status == "no_story")
    summary = CoverageSummary(
        total_requirements=total,
        requirements_with_stories=with_stories,
        requirements_scenario_covered=scenario_covered,
        no_story_count=no_story,
        story_coverage_pct=_safe_percentage(with_stories, total),
        scenario_coverage_pct=_safe_percentage(scenario_covered, total),
    )
    return CoverageReport(
        project=project,
        document_version_id=document_version_id,
        requirements=requirement_rows,
        summary=summary,
    )
