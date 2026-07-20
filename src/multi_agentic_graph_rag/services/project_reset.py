"""Explicit project maintenance resets with a Stage-4-only boundary."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import normalize_project
from multi_agentic_graph_rag.observability.logging import get_logger

_LOG = get_logger(__name__)

_STAGE4_ARTIFACT_DIRECTORIES = ("stage-4", "test-cases")
_STAGE123_ARTIFACT_DIRECTORIES: dict[str, tuple[str, ...]] = {
    "1": ("requirements", "user-stories", "test-scenario"),
    "2": ("user-stories", "test-scenario"),
    "3": ("test-scenario",),
}
_STAGE4_POSTGRES_DELETES = (
    (
        "idempotency_records",
        "DELETE FROM stage4_idempotency_records WHERE project_name = %s",
    ),
    (
        "kg_publication_attempts",
        "DELETE FROM stage4_kg_publication_attempts WHERE project_name = %s",
    ),
    ("file_journals", "DELETE FROM stage4_file_journals WHERE project_name = %s"),
    ("run_test_cases", "DELETE FROM stage4_run_test_cases WHERE project_name = %s"),
    (
        "frozen_run_manifests",
        "DELETE FROM stage4_frozen_run_manifests WHERE project_name = %s",
    ),
    ("context_manifests", "DELETE FROM codegen_context_manifests WHERE project = %s"),
    ("blockers", "DELETE FROM codegen_blockers WHERE project = %s"),
    ("legacy_test_cases", "DELETE FROM test_cases WHERE project = %s"),
    ("codegen_runs", "DELETE FROM codegen_runs WHERE project = %s"),
)


def reset_stage4_project(project: str, settings: AppSettings) -> dict[str, Any]:
    """Remove one project's Stage-4 orchestration state and artifacts only.

    This maintenance operation deliberately has no framework-path parameter. It
    never edits generated tests, Robot suites, ``test_lib`` files, or any other
    source in a user's framework. The permanent TC sequence is also retained so
    allocated six-digit identities are never recycled.
    """
    if not project.strip():
        raise ValueError("project must not be empty")

    summary: dict[str, Any] = {
        "project": project,
        "scope": "stage-4",
        "tc_sequence_retained": True,
        "permanent_tc_records_retained": True,
        "framework_files_removed": 0,
    }
    summary["postgres_rows_deleted"] = _reset_stage4_postgres(project, settings)
    summary["code_graph_rows_deleted"] = _reset_stage4_code_graph(project, settings)
    removed = _remove_stage4_artifacts(project, settings.paths.generated_dir)
    summary["artifact_directories_removed"] = removed
    _LOG.info(
        "reset.stage4 project=%s postgres=%s code_graph=%s artifact_dirs=%s",
        project,
        summary["postgres_rows_deleted"],
        summary["code_graph_rows_deleted"],
        len(removed),
    )
    return summary


def reset_stage_run(
    project: str,
    run_id: str,
    stage: str,
    settings: AppSettings,
) -> dict[str, Any]:
    """Clear one failed Stage 1-3 run while preserving valid upstream state.

    Reset scope cascades forward because regenerated upstream artifacts invalidate
    their downstream consumers: Stage 1 clears Stages 1-3, Stage 2 clears Stages
    2-3, and Stage 3 clears only Stage 3. Stage 4 and user-owned source documents
    are always retained.
    """
    if not project.strip():
        raise ValueError("project must not be empty")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if stage not in _STAGE123_ARTIFACT_DIRECTORIES:
        raise ValueError("stage must be one of: 1, 2, 3")

    postgres = PostgresStore(settings)
    postgres.ensure_schema()
    summary: dict[str, Any] = {
        "project": project,
        "run_id": run_id,
        "scope": f"stage-{stage}",
        "cleared_stages": list(range(int(stage), 4)),
        "stage4_retained": True,
        "source_documents_retained": True,
    }
    with postgres.project_maintenance_lease(project):
        if stage == "1":
            summary["neo4j_rows_deleted"] = Neo4jStore(settings).delete_run(project, run_id)
            summary["chroma_embeddings_deleted"] = ChromaStore(settings).delete_run(project, run_id)
        else:
            summary["neo4j_rows_deleted"] = {}
            summary["chroma_embeddings_deleted"] = 0

        summary["postgres_rows_deleted"] = postgres.delete_stage_run(project, run_id, stage)
        summary["artifact_directories_removed"] = _remove_stage_run_artifacts(
            project,
            run_id,
            stage,
            settings.paths.generated_dir,
        )

    _LOG.info(
        "reset.stage_run project=%s run_id=%s stage=%s postgres=%s neo4j=%s chroma=%s artifacts=%s",
        project,
        run_id,
        stage,
        summary["postgres_rows_deleted"],
        summary["neo4j_rows_deleted"],
        summary["chroma_embeddings_deleted"],
        len(summary["artifact_directories_removed"]),
    )
    return summary


def _reset_stage4_postgres(project: str, settings: AppSettings) -> int:
    store = CodegenPostgresStore(settings)
    store.ensure_schema()
    if settings.postgres.mode == "local_json":
        path = settings.stage4.codegen_local_path
        rows = _read_jsonl(path)
        permanent_kinds = {
            "tc_reservation",
            "stage4_test_case",
            "stage4_test_data_snapshot",
        }
        kept = [
            row
            for row in rows
            if _row_project(row) != project or row.get("kind") in permanent_kinds
        ]
        _write_jsonl_atomic(path, kept)
        return len(rows) - len(kept)

    deleted = 0
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"stage4:{project}",))
        for _, statement in _STAGE4_POSTGRES_DELETES:
            cursor.execute(statement, (project,))
            deleted += max(int(cursor.rowcount), 0)
        connection.commit()
    return deleted


def _reset_stage4_code_graph(project: str, settings: AppSettings) -> int:
    """Retain accepted semantic graph state referenced by permanent TCs.

    Framework snapshots, test-data snapshots, TestCase nodes, artifacts, and
    their traceability edges remain queryable. Pruning referenced READY state is
    a separate explicit maintenance operation, never part of run reset.
    """
    del project
    store = CodeGraphStore(settings)
    if settings.neo4j.mode == "local_json":
        return 0
    store.check()
    return 0


def _remove_stage4_artifacts(project: str, generated_dir: Path) -> list[str]:
    generated_root = generated_dir.resolve()
    project_root = generated_root / normalize_project(project)
    if not project_root.exists():
        return []
    if project_root.is_symlink():
        raise ValueError("generated project directory must not be a symlink")
    if not project_root.is_dir():
        raise ValueError("generated project path must be a directory")

    removed: list[str] = []
    for run_root in project_root.iterdir():
        if not run_root.is_dir() or run_root.is_symlink():
            continue
        for directory_name in _STAGE4_ARTIFACT_DIRECTORIES:
            target = run_root / directory_name
            _assert_within(target, generated_root)
            if target.is_symlink():
                target.unlink()
                removed.append(str(target))
            elif target.is_dir():
                shutil.rmtree(target)
                removed.append(str(target))
            elif target.exists():
                target.unlink()
                removed.append(str(target))
    return sorted(removed)


def _remove_stage_run_artifacts(
    project: str,
    run_id: str,
    stage: str,
    generated_dir: Path,
) -> list[str]:
    generated_root = generated_dir.resolve()
    run_component = Path(run_id)
    if run_id in {"", ".", ".."} or run_component.name != run_id:
        raise ValueError("run_id must be a single directory name")
    run_root = generated_root / normalize_project(project) / run_id
    _assert_within(run_root, generated_root)
    if not run_root.exists():
        return []
    if run_root.is_symlink():
        raise ValueError("generated run directory must not be a symlink")
    if not run_root.is_dir():
        raise ValueError("generated run path must be a directory")

    removed: list[str] = []
    for directory_name in _STAGE123_ARTIFACT_DIRECTORIES[stage]:
        target = run_root / directory_name
        _assert_within(target, generated_root)
        if target.is_symlink():
            target.unlink()
            removed.append(str(target))
        elif target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
        elif target.exists():
            target.unlink()
            removed.append(str(target))
    return sorted(removed)


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"generated artifact path escapes generated_dir: {path}") from exc


def _row_project(row: dict[str, Any]) -> str | None:
    candidates: list[Any] = [row]
    payload = row.get("payload")
    if isinstance(payload, dict):
        candidates.append(payload)
        nested = payload.get("payload")
        if isinstance(nested, dict):
            candidates.append(nested)
    for candidate in candidates:
        for key in ("project", "project_name"):
            value = candidate.get(key)
            if isinstance(value, str):
                return value
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path.exists() and not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reset_project(project: str, settings: AppSettings) -> dict[str, Any]:
    """Wipe one project across Stages 1-3 under an explicit maintenance lease.

    This legacy all-project reset remains separate from ``reset_stage4_project``
    so Stage 4 cleanup can never delete unrelated workflow data by accident.
    """
    summary: dict[str, Any] = {"project": project}
    postgres = PostgresStore(settings)
    postgres.ensure_schema()
    with postgres.project_maintenance_lease(project):
        nodes_deleted = Neo4jStore(settings).delete_project(project)
        summary["neo4j_nodes_deleted"] = nodes_deleted
        _LOG.info("reset.neo4j project=%s nodes_deleted=%s", project, nodes_deleted)

        collection_deleted = ChromaStore(settings).delete_project(project)
        summary["chroma_collection_deleted"] = collection_deleted
        _LOG.info("reset.chroma project=%s collection_deleted=%s", project, collection_deleted)

        rows_deleted = postgres.delete_project(project)
        summary["postgres_rows_deleted"] = rows_deleted
        _LOG.info("reset.postgres project=%s rows_deleted=%s", project, rows_deleted)

        generated = settings.paths.generated_dir / normalize_project(project)
        shutil.rmtree(generated, ignore_errors=True)
        summary["generated_dir_removed"] = str(generated)
        _LOG.info("reset.generated project=%s path=%s", project, generated)

    return summary


__all__ = ["reset_project", "reset_stage4_project", "reset_stage_run"]
