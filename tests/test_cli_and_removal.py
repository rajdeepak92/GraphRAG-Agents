"""CLI and removed-surface regression tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from multi_agentic_graph_rag.cli import app


def test_cli_exposes_only_current_workflow_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "ingest",
        "generate-user-stories",
        "generate-test-scenarios",
        "coverage",
        "postgres-reset",
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
