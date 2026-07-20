"""Stage-4 reset stays inside orchestration storage and generated artifacts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.services.project_reset import reset_stage4_project


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_stage4_reset_is_project_scoped_and_preserves_framework(tmp_path: Path) -> None:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.neo4j.mode = "local_json"
    settings.paths.generated_dir = tmp_path / "generated"
    settings.stage4.codegen_local_path = tmp_path / "runtime" / "codegen.jsonl"
    settings.stage4.code_graph_local_path = tmp_path / "runtime" / "code_graph.jsonl"

    run_root = settings.paths.generated_dir / "alpha" / "RUN-1"
    stage4_journal = run_root / "stage-4" / "journals" / "100001" / "journal.json"
    test_cases = run_root / "test-cases" / "test_cases.json"
    stage3_artifact = run_root / "test-scenario" / "test-scenarios.json"
    for path in (stage4_journal, test_cases, stage3_artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    framework_file = tmp_path / "user-framework" / "tests" / "sensor" / "Tc100001Demo.py"
    framework_file.parent.mkdir(parents=True)
    framework_file.write_text("class Tc100001Demo: pass\n", encoding="utf-8")

    _jsonl(
        settings.stage4.codegen_local_path,
        [
            {"kind": "codegen_run", "key": "alpha", "project": "alpha"},
            {
                "kind": "tc_reservation",
                "key": "100001",
                "payload": {"project_name": "alpha", "tc_id": 100001},
            },
            {"kind": "codegen_run", "key": "beta", "project": "beta"},
        ],
    )
    _jsonl(
        settings.stage4.code_graph_local_path,
        [
            {
                "_kind": "test_data_snapshot",
                "_key": "TD-alpha",
                "snapshot_id": "TD-alpha",
                "payload": {"snapshot_id": "TD-alpha", "project": "alpha"},
            },
            {
                "_kind": "test_data_record",
                "_key": "TD-alpha:R1",
                "snapshot_id": "TD-alpha",
                "payload": {"snapshot_id": "TD-alpha", "record_id": "R1"},
            },
            {
                "_kind": "test_case",
                "_key": "100001",
                "snapshot_id": None,
                "payload": {"tc_id": 100001, "project_name": "alpha"},
            },
            {
                "_kind": "relation",
                "_key": "100001:GENERATED_IN:FS-1",
                "snapshot_id": None,
                "payload": {
                    "source": "100001",
                    "relation": "GENERATED_IN",
                    "target": "FS-1",
                },
            },
            {
                "_kind": "test_case",
                "_key": "100002",
                "snapshot_id": None,
                "payload": {"tc_id": 100002, "project_name": "beta"},
            },
            {
                "_kind": "code_snapshot",
                "_key": "FS-1",
                "snapshot_id": "FS-1",
                "payload": {"snapshot_id": "FS-1", "repository_id": "framework"},
            },
        ],
    )

    summary = reset_stage4_project("alpha", settings)

    assert summary["scope"] == "stage-4"
    assert summary["tc_sequence_retained"] is True
    assert summary["permanent_tc_records_retained"] is True
    assert summary["framework_files_removed"] == 0
    assert not (run_root / "stage-4").exists()
    assert not (run_root / "test-cases").exists()
    assert stage3_artifact.exists()
    assert framework_file.read_text(encoding="utf-8") == "class Tc100001Demo: pass\n"
    codegen_rows = _rows(settings.stage4.codegen_local_path)
    assert {row["kind"] for row in codegen_rows} == {"tc_reservation", "codegen_run"}
    assert {row.get("project") for row in codegen_rows} == {None, "beta"}
    graph_rows = _rows(settings.stage4.code_graph_local_path)
    assert {row["_key"] for row in graph_rows} == {
        "TD-alpha",
        "TD-alpha:R1",
        "100001",
        "100001:GENERATED_IN:FS-1",
        "100002",
        "FS-1",
    }


def test_stage4_reset_does_not_expose_command_execution_or_source_control() -> None:
    source_path = Path("src/multi_agentic_graph_rag/services/project_reset.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "subprocess" not in imported_roots
    assert "git" not in imported_roots
    assert "gitpython" not in imported_roots
    assert "dulwich" not in imported_roots


def test_readme_documents_stage4_safety_and_provider_modes() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for required in (
        "generate-test-code",
        "--reasoning-provider azure_openai",
        "--reasoning-provider gemini",
        "--dry-run",
        "no Git operation",
        "reset_stage4_project(project, settings)",
        "never deletes generated framework code",
    ):
        assert required in readme


def test_stage4_reset_rejects_generated_project_symlink(tmp_path: Path) -> None:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.neo4j.mode = "local_json"
    settings.paths.generated_dir = tmp_path / "generated"
    settings.stage4.codegen_local_path = tmp_path / "runtime" / "codegen.jsonl"
    settings.stage4.code_graph_local_path = tmp_path / "runtime" / "code_graph.jsonl"
    outside = tmp_path / "outside"
    outside.mkdir()
    settings.paths.generated_dir.mkdir()
    project_link = settings.paths.generated_dir / "alpha"
    try:
        project_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        return

    try:
        reset_stage4_project("alpha", settings)
    except ValueError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("project symlink was not rejected")
