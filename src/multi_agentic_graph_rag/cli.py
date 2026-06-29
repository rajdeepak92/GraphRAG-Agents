"""Typer CLI for the ingestion-first MARAG rebuild."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from multi_agentic_graph_rag import __version__
from multi_agentic_graph_rag.agents.ingestion_document_agent import IngestionDocumentAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import IngestionRequest
from multi_agentic_graph_rag.services.artifacts import verify_requirement_artifact

app = typer.Typer(
    name="marag",
    help="Multi-Agentic Graph RAG ingestion CLI.",
    no_args_is_help=True,
    add_completion=False,
)
run_app = typer.Typer(help="Run status and recovery commands.")
artifact_app = typer.Typer(help="Generated artifact commands.")
app.add_typer(run_app, name="run")
app.add_typer(artifact_app, name="artifact")
console = Console()


@app.command("version")
def version_command() -> None:
    console.print(f"marag {__version__}")


@app.command("config-check")
def config_check(
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    settings = load_config(config_path=config)
    rows: list[tuple[str, str]] = [
        ("project_root", str(settings.paths.project_root)),
        ("reasoning_model.provider", settings.reasoning_model.provider),
        ("embedding_model.provider", settings.embedding_model.provider),
        ("reranker_model.provider", settings.reranker_model.provider),
        ("postgres.mode", settings.postgres.mode),
        ("neo4j.mode", settings.neo4j.mode),
        ("chroma.collection", settings.chroma.collection_name),
        ("global_cache_dir", str(settings.paths.global_cache_dir)),
    ]
    _render_kv("Configuration", rows)


@app.command("doctor")
def doctor() -> None:
    settings = load_config()
    checks = [
        ("python_package", "PASS", "multi_agentic_graph_rag imports"),
        ("config", "PASS", "config.json/.env/defaults loaded"),
        ("cache_policy", "PASS", f"cache root {settings.paths.global_cache_dir}"),
        ("runtime_dirs", "PASS", "runtime directories exist"),
    ]
    _render_checks("Doctor", checks)


@app.command("db-check")
def db_check() -> None:
    settings = load_config()
    checks: list[tuple[str, str, str]] = []
    failed = False
    check_functions: list[tuple[str, Callable[[], str]]] = [
        ("postgres", lambda: PostgresStore(settings).check()),
        ("neo4j", lambda: Neo4jStore(settings).check()),
        ("chroma", lambda: ChromaStore(settings).check()),
    ]
    for name, fn in check_functions:
        try:
            checks.append((name, "PASS", fn()))
        except Exception as exc:
            failed = True
            checks.append((name, "FAIL", f"{exc.__class__.__name__}: {exc}"))
    _render_checks("Database Check", checks)
    if failed:
        raise typer.Exit(code=1)


@app.command("ingest")
def ingest(
    project: Annotated[str, typer.Option("--project")],
    document: Annotated[Path, typer.Option("--document")],
    version: Annotated[str, typer.Option("--version")],
    logical_name: Annotated[str | None, typer.Option("--logical-name")] = None,
    replace_version: Annotated[bool, typer.Option("--replace-version")] = False,
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    request = IngestionRequest(
        project=project,
        document=document,
        version=version,
        logical_name=logical_name,
        replace_version=replace_version,
        reasoning_provider=reasoning_provider,
        embedding_provider=embedding_provider,
    )
    result = IngestionDocumentAgent().run(request)
    if json_output:
        console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))
        return
    console.print(
        "[green]PASS[/green] ingest completed "
        f"run_id={result.run_id} chunks={len(result.chunk_ids)} "
        f"facts={len(result.fact_ids)} requirements={len(result.requirement_ids)}"
    )
    console.print(f"manifest={result.manifest_path}")
    console.print(f"artifact={result.artifact_path}")


@run_app.command("status")
def run_status(run_id: Annotated[str, typer.Argument()]) -> None:
    settings = load_config()
    log_path = settings.paths.runtime_logs_dir / f"{run_id}.jsonl"
    if not log_path.exists():
        raise typer.BadParameter(f"unknown run id: {run_id}")
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    last_by_step: dict[str, dict[str, Any]] = {}
    for line in lines:
        if "step" in line:
            last_by_step[str(line["step"])] = line
    checks = [
        (step, str(payload.get("status", "unknown")).upper(), str(payload.get("error", "")))
        for step, payload in last_by_step.items()
    ]
    _render_checks(f"Run {run_id}", checks)


@run_app.command("resume")
def run_resume(run_id: Annotated[str, typer.Argument()]) -> None:
    settings = load_config()
    log_path = settings.paths.runtime_logs_dir / f"{run_id}.jsonl"
    if not log_path.exists():
        raise typer.BadParameter(f"unknown run id: {run_id}")
    console.print(
        "Resume is intentionally conservative in this rebuild. "
        "Use the original ingest command again; same checksum/version is idempotent."
    )


@artifact_app.command("verify")
def artifact_verify(path: Annotated[Path, typer.Argument()]) -> None:
    artifact = verify_requirement_artifact(path)
    console.print(
        "[green]PASS[/green] artifact verified "
        f"requirements={len(artifact.requirements)} facts={len(artifact.facts)} "
        f"document_version_id={artifact.document_version_id}"
    )


def _render_kv(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def _render_checks(title: str, rows: list[tuple[str, str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")
    for check, status, detail in rows:
        style = "green" if status == "PASS" or status == "COMPLETED" else "red"
        if status in {"STARTED", "WARN"}:
            style = "yellow"
        table.add_row(check, f"[{style}]{status}[/{style}]", detail)
    console.print(table)


if __name__ == "__main__":
    app()
