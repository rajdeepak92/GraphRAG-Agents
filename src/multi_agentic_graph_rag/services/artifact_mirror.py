"""Postgres-first generated artifact persistence with local JSON mirrors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    ArtifactReadResult,
    CompactRequirementArtifact,
    ReconcileReport,
    RequirementArtifact,
    RequirementsCatalogArtifact,
    TestScenarioArtifact,
    TestScenarioBuildResult,
    UserStoryArtifact,
    UserStoryBuildResult,
)
from multi_agentic_graph_rag.services.artifacts import (
    write_compact_requirement_artifact,
    write_requirement_artifact,
    write_requirements_catalog_artifact,
    write_test_scenario_artifact,
    write_user_story_artifact,
)
from multi_agentic_graph_rag.services.requirement_builder import build_requirements_catalog_artifact

ArtifactKind = Literal["requirements", "requirements_full", "user_stories", "test_scenarios"]


class ArtifactMirror:
    def __init__(self, postgres: PostgresStore) -> None:
        self.postgres = postgres

    def persist_committed_artifact(
        self,
        *,
        artifact: (
            RequirementArtifact
            | CompactRequirementArtifact
            | RequirementsCatalogArtifact
            | UserStoryArtifact
            | UserStoryBuildResult
            | TestScenarioArtifact
            | TestScenarioBuildResult
        ),
        artifact_path: Path,
        run_id: str,
    ) -> Path:
        """
        1. Persist to Postgres in a transaction.
        2. Commit.
        3. Write local JSON mirror after DB commit.
        """
        if isinstance(artifact, RequirementArtifact):
            persisted = self.postgres.persist_artifact(artifact, str(artifact_path), run_id)
            return write_requirement_artifact(persisted, artifact_path.parent)
        if isinstance(artifact, CompactRequirementArtifact):
            return write_compact_requirement_artifact(artifact, artifact_path.parent)
        if isinstance(artifact, RequirementsCatalogArtifact):
            return write_requirements_catalog_artifact(artifact, artifact_path.parent)
        if isinstance(artifact, UserStoryBuildResult):
            self.postgres.persist_user_story_artifact(artifact, str(artifact_path), run_id)
            return write_user_story_artifact(artifact.artifact, artifact_path.parent)
        if isinstance(artifact, UserStoryArtifact):
            self.postgres.persist_user_story_artifact(artifact, str(artifact_path), run_id)
            return write_user_story_artifact(artifact, artifact_path.parent)
        if isinstance(artifact, TestScenarioBuildResult):
            self.postgres.persist_test_scenario_artifact(artifact, str(artifact_path), run_id)
            return write_test_scenario_artifact(artifact.artifact, artifact_path.parent)
        self.postgres.persist_test_scenario_artifact(artifact, str(artifact_path), run_id)
        return write_test_scenario_artifact(artifact, artifact_path.parent)

    def read_preferring_local(
        self,
        *,
        artifact_path: Path,
        project: str,
        run_id: str | None,
        document_version_id: str | None,
    ) -> ArtifactReadResult:
        """
        1. Read local JSON first.
        2. Validate metadata/checksum against Postgres.
        3. If valid, return local.
        4. Else load from Postgres and re-materialize JSON.
        """
        kind = _kind_from_path(artifact_path)
        local_payload: dict[str, object] | None = None
        local_error = ""
        if artifact_path.exists():
            try:
                local_payload = _validate_payload(
                    kind,
                    json.loads(artifact_path.read_text("utf-8")),
                )
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                local_error = str(exc)

        postgres_payload = self._load_from_postgres(
            kind=kind,
            artifact_path=artifact_path,
            document_version_id=document_version_id,
        )
        if local_payload is not None and self._local_matches_postgres(
            local_payload=local_payload,
            postgres_payload=postgres_payload,
            project=project,
            document_version_id=document_version_id,
        ):
            return ArtifactReadResult(
                source="local_json",
                valid_local=True,
                artifact_path=str(artifact_path),
                payload=dict(local_payload),
                reason="local JSON matches PostgreSQL metadata and payload checksum",
            )
        if postgres_payload is None:
            reason = local_error or "local JSON missing/stale and PostgreSQL artifact unavailable"
            if local_payload is not None and self.postgres.settings.postgres.mode == "local_json":
                return ArtifactReadResult(
                    source="local_json",
                    valid_local=True,
                    artifact_path=str(artifact_path),
                    payload=dict(local_payload),
                    reason="PostgreSQL artifact unavailable; using validated local JSON",
                )
            raise FileNotFoundError(reason)

        _write_payload(artifact_path, postgres_payload)
        return ArtifactReadResult(
            source="postgres",
            valid_local=False,
            artifact_path=str(artifact_path),
            payload=postgres_payload,
            repaired=True,
            reason=local_error or "local JSON missing or mismatched; repaired from PostgreSQL",
        )

    def reconcile(
        self,
        *,
        project: str,
        document_version_id: str | None = None,
    ) -> ReconcileReport:
        """Rebuild local JSON artifacts from Postgres."""
        report = ReconcileReport(project=project, document_version_id=document_version_id)
        for row in self.postgres.load_artifact_payloads_for_project(
            project=project,
            document_version_id=document_version_id,
        ):
            path = Path(str(row["artifact_path"]))
            payload = row["payload"]
            if not isinstance(payload, dict):
                report.missing_artifacts.append(str(path))
                continue
            _write_payload(path, payload)
            report.repaired_paths.append(str(path))
            if row["artifact_kind"] == "requirements_full":
                catalog = build_requirements_catalog_artifact(
                    RequirementArtifact.model_validate(payload)
                )
                compact_path = path.parent / "requirements.json"
                _write_payload(compact_path, catalog.model_dump(mode="json"))
                report.repaired_paths.append(str(compact_path))
        return report

    def _load_from_postgres(
        self,
        *,
        kind: ArtifactKind,
        artifact_path: Path,
        document_version_id: str | None,
    ) -> dict[str, object] | None:
        if kind == "requirements":
            payload = self.postgres.load_requirement_artifact_payload(
                artifact_path=str(artifact_path),
                document_version_id=document_version_id,
            )
            if payload is None:
                return None
            catalog = build_requirements_catalog_artifact(
                RequirementArtifact.model_validate(payload)
            )
            return catalog.model_dump(mode="json")
        if kind == "requirements_full":
            payload = self.postgres.load_requirement_artifact_payload(
                artifact_path=str(artifact_path),
                document_version_id=document_version_id,
            )
        elif kind == "user_stories":
            payload = self.postgres.load_user_story_artifact_payload(
                artifact_path=str(artifact_path),
                document_version_id=document_version_id,
            )
        else:
            payload = self.postgres.load_test_scenario_artifact_payload(
                artifact_path=str(artifact_path),
                document_version_id=document_version_id,
            )
        return dict(payload) if payload is not None else None

    def _local_matches_postgres(
        self,
        *,
        local_payload: dict[str, object],
        postgres_payload: dict[str, object] | None,
        project: str,
        document_version_id: str | None,
    ) -> bool:
        if local_payload.get("project") != project:
            return False
        if document_version_id and local_payload.get("document_version_id") != document_version_id:
            return False
        if postgres_payload is None:
            return self.postgres.settings.postgres.mode == "local_json"
        return _checksum(local_payload) == _checksum(postgres_payload)


def _kind_from_path(path: Path) -> ArtifactKind:
    name = path.name
    if name in {"requirements.json", "requirement.json"}:
        return "requirements"
    if name == "requirements_full.json":
        return "requirements_full"
    if name == "user_stories.json":
        return "user_stories"
    if name == "test_scenarios.json":
        return "test_scenarios"
    raise ValueError(f"unsupported artifact filename: {name}")


def _validate_payload(kind: ArtifactKind, payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be a JSON object")
    if kind == "requirements":
        if payload.get("artifact_schema_version") == "4.0-catalog":
            return RequirementsCatalogArtifact.model_validate(payload).model_dump(mode="json")
        return CompactRequirementArtifact.model_validate(payload).model_dump(mode="json")
    if kind == "requirements_full":
        return RequirementArtifact.model_validate(payload).model_dump(mode="json")
    if kind == "user_stories":
        return UserStoryArtifact.model_validate(payload).model_dump(mode="json")
    return TestScenarioArtifact.model_validate(payload).model_dump(mode="json")


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _checksum(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
