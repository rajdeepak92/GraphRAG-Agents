"""Postgres-first generated artifact persistence with local JSON mirrors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    ArtifactReadResult,
    CanonicalRequirementsArtifact,
    ReconcileReport,
    RequirementArtifact,
    StageMasterArtifact,
    TestScenarioArtifact,
    TestScenarioBuildResult,
    UserStoryArtifact,
    UserStoryBuildResult,
)
from multi_agentic_graph_rag.services.artifacts import (
    write_canonical_requirements_artifact,
    write_test_scenario_artifact,
    write_user_story_artifact,
)
from multi_agentic_graph_rag.services.master_projection import (
    MASTER_FILENAMES,
    MASTER_STAGES,
    materialize_master,
)
from multi_agentic_graph_rag.services.requirement_builder import (
    build_canonical_requirements_artifact,
)

ArtifactKind = Literal["requirements", "user_stories", "test_scenarios"]


class ArtifactMirror:
    def __init__(self, postgres: PostgresStore) -> None:
        self.postgres = postgres

    def persist_committed_artifact(
        self,
        *,
        artifact: (
            RequirementArtifact
            | CanonicalRequirementsArtifact
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
            written = write_canonical_requirements_artifact(
                build_canonical_requirements_artifact(persisted), artifact_path.parent
            )
            self.sync_master_file(project=persisted.project, stage="requirements")
            return written
        if isinstance(artifact, CanonicalRequirementsArtifact):
            return write_canonical_requirements_artifact(artifact, artifact_path.parent)
        if isinstance(artifact, UserStoryBuildResult):
            self.postgres.persist_user_story_artifact(artifact, str(artifact_path), run_id)
            written = write_user_story_artifact(artifact.artifact, artifact_path.parent)
            self.sync_master_file(project=artifact.artifact.project, stage="user_stories")
            return written
        if isinstance(artifact, UserStoryArtifact):
            self.postgres.persist_user_story_artifact(artifact, str(artifact_path), run_id)
            written = write_user_story_artifact(artifact, artifact_path.parent)
            self.sync_master_file(project=artifact.project, stage="user_stories")
            return written
        if isinstance(artifact, TestScenarioBuildResult):
            self.postgres.persist_test_scenario_artifact(artifact, str(artifact_path), run_id)
            written = write_test_scenario_artifact(artifact.artifact, artifact_path.parent)
            self.sync_master_file(project=artifact.artifact.project, stage="test_scenarios")
            return written
        self.postgres.persist_test_scenario_artifact(artifact, str(artifact_path), run_id)
        written = write_test_scenario_artifact(artifact, artifact_path.parent)
        self.sync_master_file(project=artifact.project, stage="test_scenarios")
        return written

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
        """Rebuild local JSON artifacts from Postgres, then sync the master files."""
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
        self.reconcile_masters(project=project, report=report)
        return report

    # --- Cumulative per-project master files (Phase C) ---

    def master_file_path(self, project: str, stage: str) -> Path:
        """Stable per-project master file path, distinct from per-run artifacts."""
        return (
            self.postgres.settings.paths.generated_requirements_dir
            / project
            / MASTER_FILENAMES[stage]
        )

    def sync_master_file(self, *, project: str, stage: str) -> Path | None:
        """Write the committed master payload to its JSON file when missing/stale.

        The PostgreSQL master row is authoritative for the file: if the file is
        absent or its checksum differs, it is (re)written atomically from PG.
        """
        master = self.postgres.get_stage_master(project=project, stage=stage)
        if master is None:
            return None
        path = self.master_file_path(project, stage)
        if not _file_matches_master(path, master):
            _atomic_write_payload(path, master.model_dump(mode="json"))
        return path

    def reconcile_masters(
        self,
        *,
        project: str,
        report: ReconcileReport | None = None,
    ) -> ReconcileReport:
        """Rebuild master payloads/files from the authoritative normalized rows.

        For each stage: re-materialize from normalized rows. If the stored master
        payload disagrees (or is missing), the normalized rows win — the master is
        rebuilt in PG and a reconciliation warning is recorded. Then the JSON file
        is repaired from the (now-consistent) master if missing/stale/corrupt.
        """
        report = report or ReconcileReport(project=project)
        for stage in MASTER_STAGES:
            stored = self.postgres.get_stage_master(project=project, stage=stage)
            fresh = materialize_master(
                self.postgres,
                project=project,
                stage=stage,
                current_document_version_id=(
                    stored.current_document_version_id if stored is not None else ""
                ),
                run_id=stored.run_id if stored is not None else "",
            )
            if fresh.record_count == 0 and stored is None:
                continue
            if stored is None or stored.checksum != fresh.checksum:
                self.postgres.upsert_stage_master(fresh)
                stored = self.postgres.get_stage_master(project=project, stage=stage)
                if stored is None:
                    continue
                report.warnings.append(
                    f"{stage}_master rebuilt from normalized rows (checksum {fresh.checksum[:12]})"
                )
            path = self.master_file_path(project, stage)
            if not _file_matches_master(path, stored):
                _atomic_write_payload(path, stored.model_dump(mode="json"))
                report.repaired_paths.append(str(path))
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
            return CanonicalRequirementsArtifact.model_validate(payload).model_dump(mode="json")
        if kind == "user_stories":
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
    if name == "user_stories.json":
        return "user_stories"
    if name == "test_scenarios.json":
        return "test_scenarios"
    raise ValueError(f"unsupported artifact filename: {name}")


def _validate_payload(kind: ArtifactKind, payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("artifact payload must be a JSON object")
    if kind == "requirements":
        return CanonicalRequirementsArtifact.model_validate(payload).model_dump(mode="json")
    if kind == "user_stories":
        return UserStoryArtifact.model_validate(payload).model_dump(mode="json")
    return TestScenarioArtifact.model_validate(payload).model_dump(mode="json")


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _atomic_write_payload(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically (temp file + os.replace) so readers never see a
    partially written master file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _file_matches_master(path: Path, master: StageMasterArtifact) -> bool:
    """True when the on-disk master file exists and matches the master checksum."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(payload, dict) and payload.get("checksum") == master.checksum


def _checksum(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
