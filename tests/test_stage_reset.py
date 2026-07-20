"""Stage-scoped reset regression tests for Stages 1-3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import multi_agentic_graph_rag.cli as cli_module
from multi_agentic_graph_rag.cli import app
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.services.project_reset import reset_stage_run


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _settings(tmp_path: Path) -> Any:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.postgres.local_path = tmp_path / "runtime" / "postgres.jsonl"
    settings.neo4j.mode = "local_json"
    settings.neo4j.local_path = tmp_path / "runtime" / "neo4j.jsonl"
    settings.paths.generated_dir = tmp_path / "generated"
    return settings


def _postgres_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in ("stage-1.1", "stage-1.2", "stage-2", "stage-3"):
        rows.append(
            {
                "kind": "run",
                "key": f"RUN-1:{stage}",
                "payload": {
                    "project": "alpha",
                    "run_id": "RUN-1",
                    "stage": stage,
                },
            }
        )
    for artifact in ("requirements", "user_stories", "test_scenarios"):
        rows.append(
            {
                "kind": "artifact",
                "key": f"alpha:RUN-1:{artifact}",
                "payload": {"project": "alpha", "run_id": "RUN-1"},
            }
        )
    for context in ("user_story", "test_scenario"):
        rows.append(
            {
                "kind": "context",
                "key": f"alpha:RUN-1:{context}:anchor",
                "payload": {},
            }
        )
    rows.extend(
        [
            {
                "kind": "readiness",
                "key": "alpha",
                "payload": {"project": "alpha", "build_run_id": "RUN-1"},
            },
            {
                "kind": "run",
                "key": "RUN-2:stage-3",
                "payload": {
                    "project": "alpha",
                    "run_id": "RUN-2",
                    "stage": "stage-3",
                },
            },
        ]
    )
    return rows


def _create_run_artifacts(settings: Any) -> Path:
    run_root = settings.paths.generated_dir / "alpha" / "RUN-1"
    for name in ("requirements", "user-stories", "test-scenario", "stage-4"):
        path = run_root / name / "artifact.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return run_root


def test_stage2_reset_cascades_to_stage3_and_preserves_stage1(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_rows(settings.postgres.local_path, _postgres_rows())
    run_root = _create_run_artifacts(settings)

    summary = reset_stage_run("alpha", "RUN-1", "2", settings)

    assert summary["cleared_stages"] == [2, 3]
    assert summary["stage4_retained"] is True
    remaining_keys = {row["key"] for row in _read_rows(settings.postgres.local_path)}
    assert "RUN-1:stage-1.1" in remaining_keys
    assert "RUN-1:stage-1.2" in remaining_keys
    assert "alpha:RUN-1:requirements" in remaining_keys
    assert "RUN-1:stage-2" not in remaining_keys
    assert "RUN-1:stage-3" not in remaining_keys
    assert "RUN-2:stage-3" in remaining_keys
    assert (run_root / "requirements").exists()
    assert not (run_root / "user-stories").exists()
    assert not (run_root / "test-scenario").exists()
    assert (run_root / "stage-4").exists()


def test_stage3_reset_preserves_stage1_and_stage2(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_rows(settings.postgres.local_path, _postgres_rows())
    run_root = _create_run_artifacts(settings)

    summary = reset_stage_run("alpha", "RUN-1", "3", settings)

    assert summary["cleared_stages"] == [3]
    remaining_keys = {row["key"] for row in _read_rows(settings.postgres.local_path)}
    assert "RUN-1:stage-1.2" in remaining_keys
    assert "RUN-1:stage-2" in remaining_keys
    assert "alpha:RUN-1:user_stories" in remaining_keys
    assert "RUN-1:stage-3" not in remaining_keys
    assert "alpha:RUN-1:test_scenarios" not in remaining_keys
    assert (run_root / "requirements").exists()
    assert (run_root / "user-stories").exists()
    assert not (run_root / "test-scenario").exists()


def test_stage1_reset_clears_run_graph_vectors_and_stages_1_to_3(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    settings = _settings(tmp_path)
    _write_rows(settings.postgres.local_path, _postgres_rows())
    _write_rows(
        settings.neo4j.local_path,
        [
            {
                "kind": "chunk",
                "key": "alpha:C1",
                "project": "alpha",
                "run_id": "RUN-1",
                "chunk_id": "C1",
            },
            {"kind": "entity", "key": "alpha:E1", "project": "alpha", "entity_id": "E1"},
            {
                "kind": "mention",
                "key": "alpha:C1:E1",
                "project": "alpha",
                "chunk_id": "C1",
                "entity_id": "E1",
            },
            {
                "kind": "relationship",
                "key": "alpha:R1",
                "project": "alpha",
                "chunk_id": "C1",
                "relationship_id": "R1",
                "source_entity_id": "E1",
                "target_entity_id": "E1",
            },
            {
                "kind": "chunk",
                "key": "alpha:C2",
                "project": "alpha",
                "run_id": "RUN-2",
                "chunk_id": "C2",
            },
            {"kind": "entity", "key": "alpha:E2", "project": "alpha", "entity_id": "E2"},
            {
                "kind": "mention",
                "key": "alpha:C2:E2",
                "project": "alpha",
                "chunk_id": "C2",
                "entity_id": "E2",
            },
        ],
    )
    run_root = _create_run_artifacts(settings)
    calls: list[tuple[str, str]] = []

    def delete_run(_self: ChromaStore, project: str, run_id: str) -> int:
        calls.append((project, run_id))
        return 2

    monkeypatch.setattr(ChromaStore, "delete_run", delete_run)

    summary = reset_stage_run("alpha", "RUN-1", "1", settings)

    assert summary["cleared_stages"] == [1, 2, 3]
    assert summary["chroma_embeddings_deleted"] == 2
    assert calls == [("alpha", "RUN-1")]
    assert {row["key"] for row in _read_rows(settings.postgres.local_path)} == {"RUN-2:stage-3"}
    neo_keys = {row["key"] for row in _read_rows(settings.neo4j.local_path)}
    assert neo_keys == {"alpha:C2", "alpha:E2", "alpha:C2:E2"}
    assert not (run_root / "requirements").exists()
    assert not (run_root / "user-stories").exists()
    assert not (run_root / "test-scenario").exists()
    assert (run_root / "stage-4").exists()


def test_chroma_delete_run_removes_only_matching_ids(monkeypatch: Any, tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class Collection:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def get(self, **_kwargs: Any) -> dict[str, Any]:
            return {"ids": ["C1", "C2"]}

        def delete(self, *, ids: list[str]) -> None:
            self.deleted.extend(ids)

    collection = Collection()

    class Client:
        def get_collection(self, _name: str) -> Collection:
            return collection

    monkeypatch.setattr(ChromaStore, "_client", lambda _self: Client())

    assert ChromaStore(settings).delete_run("alpha", "RUN-1") == 2
    assert collection.deleted == ["C1", "C2"]


def test_postgres_stage2_reset_uses_run_and_downstream_checkpoint_prefixes(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.postgres.mode = "postgres"
    executed: list[tuple[str, tuple[Any, ...]]] = []

    class Cursor:
        rowcount = 0

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def execute(self, statement: str, params: tuple[Any, ...]) -> None:
            normalized = " ".join(statement.split())
            assert normalized.count("%s") == len(params)
            executed.append((normalized, params))
            self.rowcount = 1 if normalized.startswith("DELETE") else 0

        def fetchone(self) -> tuple[str]:
            return ("present",)

    class Connection:
        committed = False

        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def cursor(self) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            self.committed = True

    connection = Connection()
    store = PostgresStore(settings)
    monkeypatch.setattr(store, "_connect", lambda: connection)

    removed = store.delete_stage_run("Alpha", "RUN-1", "2")

    assert connection.committed is True
    assert removed["workflow_runs"] == 1
    workflow_params = next(
        params
        for statement, params in executed
        if statement.startswith("DELETE FROM workflow_runs")
    )
    assert workflow_params == ("Alpha", "RUN-1", "stage-2%", "stage-3%")
    checkpoint_params = [
        params for statement, params in executed if statement.startswith("DELETE FROM checkpoint")
    ]
    assert checkpoint_params == [
        ("alpha:RUN-1:stage-2%", "alpha:RUN-1:stage-3%"),
        ("alpha:RUN-1:stage-2%", "alpha:RUN-1:stage-3%"),
        ("alpha:RUN-1:stage-2%", "alpha:RUN-1:stage-3%"),
    ]


def test_stage_reset_cli_requires_confirmation_and_passes_scope(monkeypatch: Any) -> None:
    settings = load_config()
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(cli_module, "load_config", lambda: settings)

    def reset(project: str, run_id: str, stage: str, _settings: Any) -> dict[str, Any]:
        calls.append((project, run_id, stage))
        return {"project": project, "run_id": run_id, "scope": f"stage-{stage}"}

    monkeypatch.setattr(cli_module, "reset_stage_run", reset)
    runner = CliRunner()

    rejected = runner.invoke(
        app,
        ["stage-reset", "--project", "alpha", "--run-id", "RUN-1", "--stage", "2"],
    )
    assert rejected.exit_code != 0
    assert "--yes is required" in rejected.output

    accepted = runner.invoke(
        app,
        [
            "stage-reset",
            "--project",
            "alpha",
            "--run-id",
            "RUN-1",
            "--stage",
            "2",
            "--yes",
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert calls == [("alpha", "RUN-1", "2")]
