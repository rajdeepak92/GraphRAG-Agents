"""CLI and removed-surface regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import multi_agentic_graph_rag.cli as cli_module
from multi_agentic_graph_rag.cli import app
from multi_agentic_graph_rag.config.config_loader import load_config


def test_cli_exposes_only_current_workflow_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "ingest",
        "generate-user-stories",
        "generate-test-scenarios",
        "coverage",
        "postgres-reset",
        "project-reset",
    ):
        assert command in result.stdout
    for removed in (
        "build-knowledge-graph",
        "reconcile",
        "repair-identities",
        "artifact verify",
        "run resume",
    ):
        assert removed not in result.stdout


def test_ingestion_runtime_has_no_reasoning_model_call() -> None:
    source = Path("src/multi_agentic_graph_rag/workflows/ingestion_graph.py").read_text(
        encoding="utf-8"
    )
    assert "create_reasoning_model" not in source
    assert "RequirementDiscoveryAgent" not in source


def test_normal_ingest_does_not_reset_project_implicitly() -> None:
    source = Path("src/multi_agentic_graph_rag/cli.py").read_text(encoding="utf-8")
    ingest_body = source.split('@app.command("ingest")', 1)[1].split(
        '@app.command("project-reset")', 1
    )[0]
    assert "reset_project" not in ingest_body


def test_config_check_reports_stage4_provider_separately(monkeypatch: Any) -> None:
    settings = load_config()
    settings.reasoning_model.provider = "huggingface"
    settings.stage4.reasoning_provider = "azure_openai"
    monkeypatch.setattr(cli_module, "load_config", lambda: settings)

    result = CliRunner().invoke(app, ["config-check"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["reasoning_provider"] == "huggingface"
    assert payload["stage4_reasoning_provider"] == "azure_openai"


def test_doctor_checks_stage4_dependencies_without_initializing_models(
    monkeypatch: Any,
) -> None:
    settings = load_config()
    settings.stage4.graphify_command = "graphify-custom"
    monkeypatch.setattr(cli_module, "load_config", lambda: settings)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        cli_module.shutil,
        "which",
        lambda command: "C:/tools/graphify-custom.exe" if command == "graphify-custom" else None,
    )

    def unexpected_model_initialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("doctor must not initialize a model provider")

    monkeypatch.setattr(cli_module, "create_reasoning_model", unexpected_model_initialization)
    monkeypatch.setattr(cli_module, "create_embedding_model", unexpected_model_initialization)
    monkeypatch.setattr(cli_module, "create_reranker_model", unexpected_model_initialization)

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["stage4_dependencies"] == {"openpyxl": True, "robot": True}
    assert payload["graphify_command"] == "graphify-custom"
    assert payload["graphify_executable"] == "C:/tools/graphify-custom.exe"


def test_doctor_fails_when_graphify_executable_is_missing(monkeypatch: Any) -> None:
    settings = load_config()
    monkeypatch.setattr(cli_module, "load_config", lambda: settings)
    monkeypatch.setattr(cli_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(cli_module.shutil, "which", lambda _command: None)

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "FAIL"
    assert payload["graphify_executable"] is None
