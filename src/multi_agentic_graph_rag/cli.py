"""Command-line interface for the current project/run-scoped workflow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Annotated

import typer

from multi_agentic_graph_rag import __version__
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import IngestionRequest, StageRequest
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.observability.logging import configure_logging
from multi_agentic_graph_rag.services.project_reset import reset_project
from multi_agentic_graph_rag.workflows.ingestion_graph import run_ingestion
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    run_requirement_discovery,
)
from multi_agentic_graph_rag.workflows.test_scenario_graph import (
    run_test_scenario_generation,
)
from multi_agentic_graph_rag.workflows.user_story_graph import (
    run_user_story_generation,
)

app = typer.Typer(no_args_is_help=True, help="Multi-Agentic QA Knowledge GraphRAG")


@app.callback()
def _configure(
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Override the console log level (e.g. DEBUG, INFO)."),
    ] = None,
) -> None:
    """Configure runtime console logging before any command executes."""
    import os

    configure_logging(log_level or os.environ.get("LOG_LEVEL", "INFO"))


@app.command("version")
def version_command() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command("config-check")
def config_check() -> None:
    """Validate configuration without exposing credentials."""
    settings = load_config()
    _emit(
        {
            "status": "PASS",
            "app_env": settings.app_env,
            "reasoning_provider": settings.reasoning_model.provider,
            "embedding_provider": settings.embedding_model.provider,
            "reranker_provider": settings.reranker_model.provider,
            "postgres_mode": settings.postgres.mode,
            "neo4j_mode": settings.neo4j.mode,
            "generated_dir": str(settings.paths.generated_dir),
        }
    )


@app.command("hf-check")
def hf_check(
    load_model: Annotated[
        bool,
        typer.Option("--load-model", help="Instantiate configured private models."),
    ] = False,
) -> None:
    """Validate private Hugging Face model compatibility."""
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformers", "sentence_transformers", "torch")
    }
    if not all(dependencies.values()):
        _emit({"status": "FAIL", "dependencies": dependencies})
        raise typer.Exit(1)
    payload: dict[str, object] = {"status": "PASS", "dependencies": dependencies}
    if load_model:
        settings = load_config()
        payload["reasoning_provider"] = create_reasoning_model(settings).provider_name
        payload["embedding_provider"] = create_embedding_model(settings).provider_name
        payload["reranker_provider"] = create_reranker_model(settings).provider_name
    _emit(payload)


@app.command("doctor")
def doctor() -> None:
    """Run non-destructive configuration and dependency diagnostics."""
    settings = load_config()
    _emit(
        {
            "status": "PASS",
            "python_dependencies": {
                name: importlib.util.find_spec(name) is not None
                for name in ("langgraph", "psycopg", "neo4j", "chromadb", "pydantic")
            },
            "postgres_mode": settings.postgres.mode,
            "neo4j_mode": settings.neo4j.mode,
        }
    )


@app.command("db-check")
def db_check() -> None:
    """Validate stores and simplified schemas."""
    settings = load_config()
    results: dict[str, object] = {}
    failures = False
    try:
        postgres = PostgresStore(settings)
        connectivity = postgres.check()
        postgres.ensure_schema()
        results["postgres"] = f"{connectivity}; PASS simplified schema + checkpoint tables"
    except Exception as exc:
        failures = True
        results["postgres"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        neo4j = Neo4jStore(settings)
        connectivity = neo4j.check()
        neo4j.ensure_schema()
        results["neo4j"] = f"{connectivity}; PASS simplified schema"
    except Exception as exc:
        failures = True
        results["neo4j"] = f"FAIL {type(exc).__name__}: {exc}"
    try:
        results["chroma"] = ChromaStore(settings).check("diagnostic")
    except Exception as exc:
        failures = True
        results["chroma"] = f"FAIL {type(exc).__name__}: {exc}"
    _emit(
        {
            "status": "FAIL" if failures else "PASS",
            **results,
        }
    )
    if failures:
        raise typer.Exit(1)


@app.command("postgres-reset")
def postgres_reset(
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm the disposable-development reset."),
    ] = False,
) -> None:
    """Reset only the explicitly disposable development PostgreSQL store."""
    if not yes:
        raise typer.BadParameter("--yes is required for postgres-reset")
    _emit({"status": "PASS", "detail": PostgresStore(load_config()).reset_schema()})


@app.command("ingest")
def ingest(
    project: Annotated[str, typer.Option("--project")],
    document: Annotated[Path, typer.Option("--document")],
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
) -> None:
    """Run Stage 1.1 and joined Stage 1.2 for one project/source file."""
    settings = load_config()
    ingestion = run_ingestion(
        IngestionRequest(
            project_name=project,
            source_file=document,
            embedding_provider=embedding_provider,
        ),
        settings=settings,
    )
    discovery = run_requirement_discovery(
        StageRequest(
            project_name=project,
            run_id=ingestion.run_id,
            reasoning_provider=reasoning_provider,
        ),
        settings=settings,
    )
    _emit(
        {
            "status": "completed",
            "project": project,
            "run_id": ingestion.run_id,
            "chunk_manifest": str(ingestion.manifest_path),
            "requirements": str(discovery.artifact_path),
            "requirement_ids": discovery.item_ids,
        }
    )


@app.command("project-reset")
def project_reset(
    project: Annotated[str, typer.Option("--project")],
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm deletion of this project from every managed store."),
    ] = False,
) -> None:
    """Explicitly delete one project from Neo4j, Chroma, PostgreSQL, and generated files."""
    if not yes:
        raise typer.BadParameter("--yes is required for project-reset")
    _emit({"status": "PASS", "reset": reset_project(project, load_config())})


@app.command("generate-user-stories")
def generate_user_stories(
    project: Annotated[str, typer.Option("--project")],
    run_id: Annotated[str, typer.Option("--run-id")],
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
) -> None:
    """Generate Stage 2 canonical user stories."""
    result = run_user_story_generation(
        StageRequest(
            project_name=project,
            run_id=run_id,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
        )
    )
    _emit(
        {
            "status": "completed",
            "project": project,
            "run_id": run_id,
            "artifact": str(result.artifact_path),
            "story_ids": result.item_ids,
        }
    )


@app.command("generate-test-scenarios")
def generate_test_scenarios(
    project: Annotated[str, typer.Option("--project")],
    run_id: Annotated[str, typer.Option("--run-id")],
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
) -> None:
    """Generate Stage 3 canonical behavioral scenarios."""
    result = run_test_scenario_generation(
        StageRequest(
            project_name=project,
            run_id=run_id,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
        )
    )
    _emit(
        {
            "status": "completed",
            "project": project,
            "run_id": run_id,
            "artifact": str(result.artifact_path),
            "scenario_ids": result.item_ids,
        }
    )


@app.command("index-framework")
def index_framework_command(
    framework_path: Annotated[Path, typer.Option("--framework-path")],
    graphify_out: Annotated[
        Path | None,
        typer.Option("--graphify-out", help="Graphify output dir (default <fw>/graphify-out)"),
    ] = None,
    repository_id: Annotated[str | None, typer.Option("--repository-id")] = None,
) -> None:
    """Stage 4: index a framework revision into the code-property graph."""
    from multi_agentic_graph_rag.services.framework_indexer import index_framework

    settings = load_config()
    out_dir = graphify_out or (framework_path / "graphify-out")
    result = index_framework(
        settings=settings,
        framework_path=framework_path,
        graphify_out_dir=out_dir,
        repository_id=repository_id,
    )
    _emit(
        {
            "status": "completed",
            "snapshot_id": result.snapshot.snapshot_id,
            "commit": result.snapshot.commit,
            "tree_hash": result.snapshot.tree_hash,
            "files": result.file_count,
            "symbols": result.symbol_count,
            "edges": result.edge_count,
            "dependencies": result.dependency_count,
        }
    )


@app.command("ingest-test-data")
def ingest_test_data_command(
    project: Annotated[str, typer.Option("--project")],
    document: Annotated[Path, typer.Option("--document", help="normalized-test-data.json")],
    scenario_id: Annotated[
        list[str] | None,
        typer.Option("--scenario-id", help="Canonical scenario ID (repeatable)."),
    ] = None,
) -> None:
    """Stage 4A: validate a test-data document and publish an immutable snapshot."""
    from multi_agentic_graph_rag.services.test_data_document_reader import read_document
    from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document

    parsed = read_document(document)
    result = ingest_document(parsed, scenario_ids=set(scenario_id or []))
    payload: dict[str, object] = {
        "status": result.report.status,
        "project": parsed.project,
        "issues": [issue.model_dump(mode="json") for issue in result.report.issues],
    }
    if result.normalized is not None:
        payload["snapshot_id"] = result.normalized.snapshot_id
        payload["record_count"] = len(result.normalized.records)
        payload["binding_count"] = len(result.normalized.bindings)
    _emit(payload)
    if not result.is_ready:
        raise typer.Exit(1)


@app.command("codegen-readiness")
def codegen_readiness_command(
    readiness_input: Annotated[
        Path, typer.Option("--readiness-input", help="JSON of ReadinessInputs fields.")
    ],
    scenario_id: Annotated[str, typer.Option("--scenario-id")],
    execution_profile: Annotated[str, typer.Option("--execution-profile")],
) -> None:
    """Stage 4: evaluate the deterministic readiness gate for a scenario."""
    from multi_agentic_graph_rag.services.readiness_gate import (
        ReadinessInputs,
        evaluate_readiness,
    )

    data = json.loads(readiness_input.read_text(encoding="utf-8"))
    report = evaluate_readiness(
        scenario_id=scenario_id,
        execution_profile_id=execution_profile,
        inputs=ReadinessInputs(**data),
    )
    _emit(report.model_dump(mode="json"))
    if not report.is_ready:
        raise typer.Exit(1)


@app.command("coverage")
def coverage(
    project: Annotated[str, typer.Option("--project")],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    """Report current project/run requirement-story-scenario coverage."""
    _emit(PostgresStore(load_config()).coverage(project, run_id).model_dump(mode="json"))


def _emit(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def main() -> None:
    """Run the Typer application."""
    app()


__all__ = ["app", "main"]
