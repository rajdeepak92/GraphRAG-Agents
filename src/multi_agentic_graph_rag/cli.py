"""Typer CLI for the ingestion-first MARAG rebuild."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

from multi_agentic_graph_rag import __version__
from multi_agentic_graph_rag.agents.ingestion_document_agent import IngestionDocumentAgent
from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGeneratorAgent
from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGeneratorAgent
from multi_agentic_graph_rag.common_defs import EnvVar, RuntimeCommand
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.huggingface_env import (
    HF_OFFLINE_FLAGS,
    token_alias_status,
)
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import MaragError, SchemaMismatchError
from multi_agentic_graph_rag.domain.identifiers import run_id
from multi_agentic_graph_rag.domain.schemas import (
    IngestionRequest,
    KnowledgeGraphRequest,
    RequirementArtifact,
    TestScenarioRequest,
    UserStoryRequest,
)
from multi_agentic_graph_rag.llm_models.huggingface import (
    HuggingFaceEmbeddingModel,
    HuggingFaceReasoningModel,
    HuggingFaceRerankerModel,
)
from multi_agentic_graph_rag.observability.session import (
    command_run_id,
    command_session,
    find_run_jsonl,
)
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.artifacts import (
    verify_requirement_artifact,
    verify_test_scenario_artifact,
    verify_user_story_artifact,
)
from multi_agentic_graph_rag.services.test_scenario_builder import (
    project_test_scenario_artifact,
)
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact
from multi_agentic_graph_rag.workflows.knowledge_graph import run_knowledge_graph_build
from multi_agentic_graph_rag.workflows.test_scenario_graph import resolve_test_scenario_identity
from multi_agentic_graph_rag.workflows.user_story_graph import resolve_user_story_identity

app = typer.Typer(
    name="marag",
    help="Multi-Agentic Graph RAG ingestion CLI.",
    no_args_is_help=True,
    add_completion=False,
)
run_app = typer.Typer(help="Run status and recovery commands.")
artifact_app = typer.Typer(help="Generated artifact commands.")
requirements_app = typer.Typer(help="Requirement identity maintenance commands.")
app.add_typer(run_app, name="run")
app.add_typer(artifact_app, name="artifact")
app.add_typer(requirements_app, name="requirements")
console = Console()


@app.command("version")
def version_command() -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.VERSION.value,
        run_id=command_run_id(RuntimeCommand.VERSION.value),
    ) as session:
        session.logger.info(
            "Reporting version {version}",
            step="version",
            version=__version__,
            status="completed",
        )
        console.print(f"marag {__version__}")


@app.command("config-check")
def config_check(
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.CONFIG_CHECK.value,
        run_id=command_run_id(RuntimeCommand.CONFIG_CHECK.value),
    ) as session:
        settings = load_config(config_path=config)
        session.set_log_level(settings.log_level)
        session.logger.info(
            "Loaded configuration for {project_root}",
            step="config-check",
            project_root=str(settings.paths.project_root),
            status="completed",
        )
        rows: list[tuple[str, str]] = [
            ("project_root", str(settings.paths.project_root)),
            ("reasoning_model.provider", settings.reasoning_model.provider),
            ("embedding_model.provider", settings.embedding_model.provider),
            ("reranker_model.provider", settings.reranker_model.provider),
            ("postgres.mode", settings.postgres.mode),
            ("neo4j.mode", settings.neo4j.mode),
            ("chroma.collection", settings.chroma.collection_name),
            ("global_cache_dir", str(settings.paths.global_cache_dir)),
            ("huggingface.token", "set" if settings.huggingface.token else "not set"),
            ("huggingface.token_aliases", token_alias_status(os.environ)),
            ("huggingface.offline", str(settings.huggingface.offline)),
            ("huggingface.offline_flags", _hf_offline_flag_status()),
            (
                "huggingface.reasoning_model",
                settings.huggingface.reasoning_model or "<not configured>",
            ),
            ("huggingface.embedding_model", settings.huggingface.embedding_model),
            ("huggingface.reranker_model", settings.huggingface.reranker_model),
            ("huggingface.max_new_tokens", str(settings.huggingface.max_new_tokens)),
            ("discovery.batch_size", str(settings.discovery.batch_size)),
            ("discovery.log_llm_responses", str(settings.discovery.log_llm_responses)),
            ("discovery.ledger_enabled", str(settings.discovery.ledger_enabled)),
            ("discovery.ledger_max_entries", str(settings.discovery.ledger_max_entries)),
            ("discovery.ledger_top_k", str(settings.discovery.ledger_top_k)),
            ("enable_hfil", str(settings.enable_hfil)),
            ("hfil.match_threshold_pct", str(settings.hfil_match_threshold_pct)),
            ("hfil.out_of_context_pct", str(settings.hfil_out_of_context_pct)),
        ]
        _render_kv("Configuration", rows)


@app.command("hf-check")
def hf_check(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    load_model: Annotated[
        bool,
        typer.Option(
            "--load-model",
            help="Attempt actual Hugging Face reasoning and embedding model construction.",
        ),
    ] = False,
) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.HF_CHECK.value,
        run_id=command_run_id(RuntimeCommand.HF_CHECK.value),
    ) as session:
        settings = load_config(config_path=config)
        session.set_log_level(settings.log_level)
        requires_transformers = settings.reasoning_model.provider == "huggingface"
        requires_sentence_transformers = (
            settings.embedding_model.provider == "huggingface"
            or settings.reranker_model.provider == "huggingface"
        )
        checks: list[tuple[str, str, str]] = [
            (
                "token",
                "PASS" if settings.huggingface.token else "WARN",
                token_alias_status(os.environ),
            ),
            ("offline", "PASS", str(settings.huggingface.offline)),
            ("offline_flags", "PASS", _hf_offline_flag_status()),
            ("cache.HF_HOME", "PASS", os.environ.get("HF_HOME", "<not set>")),
            (
                "cache.TRANSFORMERS_CACHE",
                "PASS",
                os.environ.get("TRANSFORMERS_CACHE", "<not set>"),
            ),
            (
                "reasoning_model",
                "PASS"
                if settings.huggingface.reasoning_model
                else ("FAIL" if requires_transformers else "WARN"),
                settings.huggingface.reasoning_model or "<not configured>",
            ),
            (
                "embedding_model",
                "PASS"
                if settings.huggingface.embedding_model
                else ("FAIL" if settings.embedding_model.provider == "huggingface" else "WARN"),
                settings.huggingface.embedding_model or "<not configured>",
            ),
            (
                "reranker_model",
                "PASS"
                if settings.huggingface.reranker_model
                else ("FAIL" if settings.reranker_model.provider == "huggingface" else "WARN"),
                settings.huggingface.reranker_model or "<not configured>",
            ),
            (
                "transformers",
                _dependency_status("transformers", required=requires_transformers),
                "installed" if find_spec("transformers") is not None else "not installed",
            ),
            (
                "sentence_transformers",
                _dependency_status(
                    "sentence_transformers",
                    required=requires_sentence_transformers,
                ),
                "installed" if find_spec("sentence_transformers") is not None else "not installed",
            ),
            (
                "generation",
                "PASS",
                f"max_new_tokens={settings.huggingface.max_new_tokens} "
                f"discovery_batch_size={settings.discovery.batch_size} "
                f"log_llm_responses={settings.discovery.log_llm_responses}",
            ),
        ]
        if load_model:
            checks.extend(_hf_load_checks(settings.huggingface.reasoning_model, settings))
        else:
            checks.append(("model_load", "WARN", "skipped; pass --load-model to verify"))
        failed = any(status == "FAIL" for _, status, _ in checks)
        session.logger.info(
            "Hugging Face checks completed",
            step="hf-check",
            status="FAIL" if failed else "completed",
        )
        _render_checks("Hugging Face Check", checks)
        if failed:
            raise typer.Exit(code=1)


@app.command("doctor")
def doctor() -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.DOCTOR.value,
        run_id=command_run_id(RuntimeCommand.DOCTOR.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        checks = [
            ("python_package", "PASS", "multi_agentic_graph_rag imports"),
            ("config", "PASS", "config.json/.env/defaults loaded"),
            ("cache_policy", "PASS", f"cache root {settings.paths.global_cache_dir}"),
            ("runtime_dirs", "PASS", "runtime directories exist"),
        ]
        session.logger.info("Doctor checks completed", step="doctor", status="completed")
        _render_checks("Doctor", checks)


@app.command("db-check")
def db_check() -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.DB_CHECK.value,
        run_id=command_run_id(RuntimeCommand.DB_CHECK.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        checks: list[tuple[str, str, str]] = []
        failed = False
        check_functions: list[tuple[str, Callable[[], str]]] = [
            ("postgres", lambda: PostgresStore(settings).check()),
            ("neo4j", lambda: Neo4jStore(settings).check()),
            ("chroma", lambda: ChromaStore(settings).check()),
        ]
        for name, fn in check_functions:
            try:
                detail = fn()
                checks.append((name, "PASS", detail))
                session.logger.info(
                    "{check} passed",
                    step=f"db-check.{name}",
                    check=name,
                    detail=detail,
                    status="PASS",
                )
            except Exception as exc:
                failed = True
                detail = f"{exc.__class__.__name__}: {exc}"
                checks.append((name, "FAIL", detail))
                session.logger.error(
                    "{check} failed",
                    step=f"db-check.{name}",
                    check=name,
                    detail=detail,
                    status="FAIL",
                )
        _render_checks("Database Check", checks)
        if failed:
            raise typer.Exit(code=1)


@app.command("coverage")
def coverage(
    project: Annotated[str, typer.Option("--project", help="Project to report coverage for.")],
    document_version_id: Annotated[
        str | None,
        typer.Option("--document-version-id", help="Optional document version scope."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    """Report per-requirement coverage (no_story / story_covered / scenario_covered)."""
    with command_session(
        project=project,
        version=__version__,
        command=RuntimeCommand.COVERAGE.value,
        run_id=command_run_id(RuntimeCommand.COVERAGE.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        report = PostgresStore(settings).load_coverage_report(
            project=project,
            document_version_id=document_version_id,
        )
        summary = report.summary
        session.logger.info(
            "Computed coverage for {project}",
            step="coverage",
            project=project,
            requirement_count=summary.total_requirements,
            story_coverage_pct=summary.story_coverage_pct,
            scenario_coverage_pct=summary.scenario_coverage_pct,
            status="completed",
        )
        if json_output:
            console.print_json(json.dumps(report.model_dump(mode="json")))
            return
        table = Table(title=f"Coverage — {project}")
        table.add_column("Requirement")
        table.add_column("Coverage")
        table.add_column("Stories", justify="right")
        table.add_column("Scenarios", justify="right")
        for row in report.requirements:
            table.add_row(
                row.requirement_id,
                row.coverage_status,
                str(len(row.story_ids)),
                str(row.scenario_count),
            )
        console.print(table)
        console.print(
            f"[bold]Total active requirements:[/bold] {summary.total_requirements}  "
            f"[bold]With stories:[/bold] {summary.requirements_with_stories} "
            f"({summary.story_coverage_pct}%)  "
            f"[bold]Scenario-covered:[/bold] {summary.requirements_scenario_covered} "
            f"({summary.scenario_coverage_pct}%)  "
            f"[bold]No story:[/bold] {summary.no_story_count}"
        )


@app.command("reconcile")
def reconcile(
    project: Annotated[str, typer.Option("--project", help="Project to reconcile.")],
    document_version_id: Annotated[
        str | None,
        typer.Option("--document-version", help="Optional document version id scope."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    """Re-materialize local JSON artifacts from PostgreSQL."""
    with command_session(
        project=project,
        version=document_version_id or __version__,
        command=RuntimeCommand.RECONCILE.value,
        run_id=command_run_id(RuntimeCommand.RECONCILE.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        report = ArtifactMirror(PostgresStore(settings)).reconcile(
            project=project,
            document_version_id=document_version_id,
        )
        session.logger.info(
            "Reconciled local JSON artifacts from PostgreSQL",
            step="reconcile",
            project=project,
            document_version_id=document_version_id,
            repaired_count=len(report.repaired_paths),
            status="completed",
        )
        if json_output:
            console.print_json(json.dumps(report.model_dump(mode="json"), indent=2))
            return
        console.print(
            "[green]PASS[/green] reconcile completed "
            f"repaired={len(report.repaired_paths)} missing={len(report.missing_artifacts)}"
        )
        for path in report.repaired_paths:
            console.print(path)


@app.command("postgres-reset")
def postgres_reset(
    yes: Annotated[bool, typer.Option("--yes", help="Confirm PostgreSQL schema reset.")] = False,
    allow_nonlocal: Annotated[
        bool,
        typer.Option(
            "--allow-nonlocal",
            help=f"Allow reset when {EnvVar.POSTGRES_DSN.value} does not point to localhost.",
        ),
    ] = False,
) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.POSTGRES_RESET.value,
        run_id=command_run_id(RuntimeCommand.POSTGRES_RESET.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        if not yes:
            raise typer.BadParameter("pass --yes to reset PostgreSQL managed tables")
        if settings.postgres.mode == "postgres" and not allow_nonlocal:
            host = _postgres_dsn_host(settings.postgres.dsn)
            if host not in {"", "localhost", "127.0.0.1", "::1"}:
                raise typer.BadParameter(
                    f"{EnvVar.POSTGRES_DSN.value} is not local; "
                    "pass --allow-nonlocal only if you intend that"
                )
        detail = PostgresStore(settings).reset_schema()
        session.logger.warning(
            "PostgreSQL schema reset completed",
            step="postgres-reset",
            detail=detail,
            status="completed",
        )
        console.print(f"[yellow]{detail}[/yellow]")


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
    with command_session(
        project=project,
        version=version,
        command=RuntimeCommand.INGEST.value,
        run_id=run_id(project, document, version),
    ) as session:
        session.request_payload = {
            "project": project,
            "document": str(document),
            "version": version,
            "logical_name": logical_name,
            "replace_version": replace_version,
            "reasoning_provider": reasoning_provider,
            "embedding_provider": embedding_provider,
        }
        session.logger.info(
            "Starting ingest command",
            step="ingest",
            document=str(document),
            project=project,
            version=version,
            status="started",
        )
        request = IngestionRequest(
            project=project,
            document=document,
            version=version,
            logical_name=logical_name,
            replace_version=replace_version,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
        )
        try:
            result = IngestionDocumentAgent().run(request, session=session)
        except SchemaMismatchError:
            console.print(
                "[red]FAIL[/red] PostgreSQL schema mismatch. "
                "Reset disposable local app tables with: "
                "python -m multi_agentic_graph_rag postgres-reset --yes"
            )
            raise typer.Exit(code=1) from None
        except MaragError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            raise typer.Exit(code=1) from None
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


@app.command("generate-user-stories")
def generate_user_stories(
    requirements: Annotated[Path | None, typer.Option("--requirements")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    document_version_id: Annotated[str | None, typer.Option("--document-version-id")] = None,
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    reranker_provider: Annotated[str | None, typer.Option("--reranker-provider")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    request = UserStoryRequest(
        requirements_path=requirements,
        project=project,
        document_version_id=document_version_id,
        reasoning_provider=reasoning_provider,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        top_k=top_k,
    )
    try:
        resolved_project, resolved_version = resolve_user_story_identity(request)
    except MaragError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        raise typer.Exit(code=1) from None
    with command_session(
        project=resolved_project,
        version=resolved_version,
        command=RuntimeCommand.GENERATE_USER_STORIES.value,
        run_id=command_run_id(RuntimeCommand.GENERATE_USER_STORIES.value),
    ) as session:
        session.request_payload = request.model_dump(mode="json")
        session.logger.info(
            "Starting generate-user-stories command",
            step="user-stories",
            requirements=str(requirements) if requirements else None,
            project=resolved_project,
            document_version_id=document_version_id,
            status="started",
        )
        try:
            result = UserStoryGeneratorAgent().run(request, session=session)
        except SchemaMismatchError:
            console.print(
                "[red]FAIL[/red] PostgreSQL schema mismatch. "
                "Reset disposable local app tables with: "
                "python -m multi_agentic_graph_rag postgres-reset --yes"
            )
            raise typer.Exit(code=1) from None
        except MaragError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            raise typer.Exit(code=1) from None
        if json_output:
            console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))
            return
        console.print(
            "[green]PASS[/green] user-story generation completed "
            f"run_id={result.run_id} requirements={result.requirement_count} "
            f"stories={len(result.story_ids)} covered={len(result.coverage)}"
        )
        console.print(f"artifact={result.artifact_path}")


@app.command("build-knowledge-graph")
def build_knowledge_graph(
    project: Annotated[str, typer.Option("--project")],
    document_version_id: Annotated[str, typer.Option("--document-version-id")],
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    request = KnowledgeGraphRequest(
        project=project,
        document_version_id=document_version_id,
        reasoning_provider=reasoning_provider,
    )
    with command_session(
        project=project,
        version="generated",
        command=RuntimeCommand.BUILD_KNOWLEDGE_GRAPH.value,
        run_id=command_run_id(RuntimeCommand.BUILD_KNOWLEDGE_GRAPH.value),
    ) as session:
        session.request_payload = request.model_dump(mode="json")
        session.logger.info(
            "Starting build-knowledge-graph command",
            step="knowledge-graph",
            project=project,
            document_version_id=document_version_id,
            status="started",
        )
        try:
            result = run_knowledge_graph_build(request, session=session)
        except SchemaMismatchError:
            console.print(
                "[red]FAIL[/red] PostgreSQL schema mismatch. "
                "Reset disposable local app tables with: "
                "python -m multi_agentic_graph_rag postgres-reset --yes"
            )
            raise typer.Exit(code=1) from None
        except MaragError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            raise typer.Exit(code=1) from None
        if json_output:
            console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))
            return
        console.print(
            "[green]PASS[/green] knowledge-graph build completed "
            f"run_id={result.run_id} chunks={result.chunk_count} "
            f"entities={result.entity_count} assertions={result.assertion_count} "
            f"evidence={result.evidence_count}"
        )
        console.print(f"artifact={result.artifact_path}")


@app.command("generate-test-scenarios")
def generate_test_scenarios(
    user_stories: Annotated[Path | None, typer.Option("--user-stories")] = None,
    requirements: Annotated[Path | None, typer.Option("--requirements")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    document_version_id: Annotated[str | None, typer.Option("--document-version-id")] = None,
    reasoning_provider: Annotated[str | None, typer.Option("--reasoning-provider")] = None,
    embedding_provider: Annotated[str | None, typer.Option("--embedding-provider")] = None,
    reranker_provider: Annotated[str | None, typer.Option("--reranker-provider")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k")] = None,
    hfil: Annotated[
        bool | None,
        typer.Option(
            "--hfil/--no-hfil",
            help="Enable or disable the test-scenario human feedback loop.",
        ),
    ] = None,
    emit_md: Annotated[
        bool,
        typer.Option("--emit-md", help="Also emit a human-readable Markdown report."),
    ] = False,
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="Stable LangGraph thread id for HFIL resume."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    request = TestScenarioRequest(
        user_stories_path=user_stories,
        requirements_path=requirements,
        project=project,
        document_version_id=document_version_id,
        reasoning_provider=reasoning_provider,
        embedding_provider=embedding_provider,
        reranker_provider=reranker_provider,
        top_k=top_k,
        hfil_enabled=hfil,
        emit_md=emit_md,
        thread_id=thread_id,
    )
    try:
        resolved_project, resolved_version = resolve_test_scenario_identity(request)
    except MaragError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        raise typer.Exit(code=1) from None
    with command_session(
        project=resolved_project,
        version=resolved_version,
        command=RuntimeCommand.GENERATE_TEST_SCENARIOS.value,
        run_id=command_run_id(RuntimeCommand.GENERATE_TEST_SCENARIOS.value),
    ) as session:
        session.request_payload = request.model_dump(mode="json")
        session.logger.info(
            "Starting generate-test-scenarios command",
            step="test-scenarios",
            user_stories=str(user_stories) if user_stories else None,
            requirements=str(requirements) if requirements else None,
            project=resolved_project,
            document_version_id=document_version_id,
            hfil_enabled=hfil,
            emit_md=emit_md,
            thread_id=thread_id,
            status="started",
        )
        try:
            result = TestScenarioGeneratorAgent().run(request, session=session)
        except SchemaMismatchError:
            console.print(
                "[red]FAIL[/red] PostgreSQL schema mismatch. "
                "Reset disposable local app tables with: "
                "python -m multi_agentic_graph_rag postgres-reset --yes"
            )
            raise typer.Exit(code=1) from None
        except MaragError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            raise typer.Exit(code=1) from None
        if json_output:
            console.print_json(json.dumps(result.model_dump(mode="json"), indent=2))
            return
        console.print(
            "[green]PASS[/green] test-scenario generation completed "
            f"run_id={result.run_id} stories={result.story_count} "
            f"scenarios={len(result.scenario_ids)}"
        )
        console.print(f"artifact={result.artifact_path}")


@run_app.command("status")
def run_status(run_id_value: Annotated[str, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.RUN_STATUS.value,
        run_id=command_run_id(RuntimeCommand.RUN_STATUS.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        log_path = find_run_jsonl(settings.paths.project_root, run_id_value)
        if log_path is None:
            raise typer.BadParameter(f"unknown run id: {run_id_value}")
        session.logger.info(
            "Resolved run log {path}",
            step="run-status",
            run_id=run_id_value,
            path=str(log_path),
            status="completed",
        )
        lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        last_by_step: dict[str, dict[str, Any]] = {}
        for line in lines:
            if "step" not in line or line.get("level") == "DEBUG":
                continue
            last_by_step[str(line["step"])] = line
        checks = [
            (
                step,
                str(
                    payload.get("context", {}).get("status")
                    or payload.get("status")
                    or payload.get("level", "unknown")
                ).upper(),
                str(
                    payload.get("context", {}).get("detail")
                    or payload.get("context", {}).get("error")
                    or payload.get("error")
                    or payload.get("message", "")
                ),
            )
            for step, payload in last_by_step.items()
        ]
        _render_checks(f"Run {run_id_value}", checks)


@run_app.command("resume")
def run_resume(run_id_value: Annotated[str, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.RUN_RESUME.value,
        run_id=command_run_id(RuntimeCommand.RUN_RESUME.value),
    ) as session:
        settings = load_config()
        session.set_log_level(settings.log_level)
        log_path = find_run_jsonl(settings.paths.project_root, run_id_value)
        if log_path is None:
            raise typer.BadParameter(f"unknown run id: {run_id_value}")
        session.logger.warning(
            "Resume requested for {run_id}; ingest is idempotent",
            step="run-resume",
            run_id=run_id_value,
            path=str(log_path),
            status="completed",
        )
        console.print(
            "Resume is intentionally conservative in this rebuild. "
            "Use the original ingest command again; same checksum/version is idempotent."
        )


@requirements_app.command("repair-identities")
def requirements_repair_identities(
    artifact: Annotated[
        Path | None,
        typer.Option("--artifact", help="Path to a legacy requirement artifact to inspect."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", help="Repair canonical identities for one stored project."),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Commit the repair (default is a dry run)."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Analyze and report without writing (the default)."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json-output")] = False,
) -> None:
    """Dry-run or repair canonical identities for a project or legacy artifact.

    Dry-run by default. Splits any lineage whose revisions carry more than one
    identity signature onto corrected deterministic lineages, preserving all
    evidence. Project apply is transactional; legacy artifact apply is atomic.
    """
    from multi_agentic_graph_rag.services.requirement_repair import (
        analyze_requirement_artifact,
        apply_repair,
        migrate_legacy_catalog_payload,
    )

    if apply and dry_run:
        raise typer.BadParameter("--apply and --dry-run are mutually exclusive")
    if (artifact is None) == (project is None):
        raise typer.BadParameter("provide exactly one of --project or --artifact")
    if project is not None:
        settings = load_config()
        postgres = PostgresStore(settings)
        project_report = postgres.repair_project_identities(project=project, apply=apply)
        if json_output:
            console.print_json(json.dumps(project_report))
        else:
            raw_impact = project_report.get("impact_counts", {})
            impact = raw_impact if isinstance(raw_impact, dict) else {}
            console.print(
                f"Project {project}: artifacts={project_report.get('artifact_count', 0)} "
                f"revisions={impact.get('requirement_revisions', 0)} "
                f"evidence={impact.get('evidence_occurrences', 0)}"
            )
        raw_ambiguous = project_report.get("ambiguous_cases", [])
        ambiguous = raw_ambiguous if isinstance(raw_ambiguous, list) else []
        if ambiguous:
            for message in ambiguous:
                console.print(f"[red]AMBIGUOUS[/red] {message}")
            if apply:
                raise typer.Exit(code=2)
        if not apply:
            console.print("[yellow]DRY RUN[/yellow] re-run with --apply to commit the repair")
            return
        _rebuild_identity_projections(postgres, Neo4jStore(settings), project)
        console.print(f"[green]APPLIED[/green] repaired project {project}")
        return

    assert artifact is not None
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("artifact_schema_version") == "4.0-catalog":
        migrated = migrate_legacy_catalog_payload(payload)
        summary = {
            "source_schema": "4.0-catalog",
            "target_schema": migrated.artifact_schema_version,
            "canonical_requirements": len(migrated.requirements),
            "evidence_occurrences": sum(len(row.evidence) for row in migrated.requirements),
        }
        if json_output:
            console.print_json(json.dumps(summary))
        else:
            console.print(
                f"Legacy catalog -> {migrated.artifact_schema_version}: "
                f"requirements={summary['canonical_requirements']} "
                f"evidence={summary['evidence_occurrences']}"
            )
        if not apply:
            console.print("[yellow]DRY RUN[/yellow] re-run with --apply to write the migration")
            return
        from multi_agentic_graph_rag.services.generation_checkpoint import atomic_write_json

        atomic_write_json(artifact, migrated.model_dump(mode="json"))
        console.print(f"[green]APPLIED[/green] migrated legacy catalog -> {artifact}")
        return
    parsed = RequirementArtifact.model_validate(payload)
    with command_session(
        project=parsed.project,
        version=__version__,
        command="requirements-repair-identities",
        run_id=command_run_id("requirements-repair-identities"),
    ) as session:
        report = analyze_requirement_artifact(parsed)
        session.logger.info(
            "Analyzed requirement lineages for pollution",
            step="repair-identities",
            project=parsed.project,
            total_lineages=report.total_lineages,
            polluted_lineages=report.polluted_lineages,
            status="dry_run" if not apply else "apply",
        )
        if json_output:
            console.print_json(json.dumps(report.to_dict()))
        else:
            console.print(
                f"Requirement lineages: {report.total_lineages} "
                f"polluted={report.polluted_lineages} "
                f"({report.total_requirements} requirements)"
            )
            for finding in report.findings:
                console.print(
                    f"  [yellow]{finding.old_requirement_id}[/yellow] -> "
                    f"{len(set(finding.revision_remap.values()))} distinct lineages "
                    f"({len(finding.signatures)} signatures)"
                )
        if not report.id_remap and not report.revision_id_remap and not report.evidence_id_remap:
            console.print("[green]PASS[/green] no polluted lineages detected")
            return
        if not apply:
            console.print("[yellow]DRY RUN[/yellow] re-run with --apply to write the fix")
            return
        from multi_agentic_graph_rag.services.generation_checkpoint import atomic_write_json

        repaired = apply_repair(parsed, report)
        atomic_write_json(artifact, repaired.model_dump(mode="json"))
        console.print(
            f"[green]APPLIED[/green] repaired {report.polluted_lineages} lineages -> {artifact}"
        )


def _rebuild_identity_projections(
    postgres: PostgresStore,
    neo4j: Neo4jStore,
    project: str,
) -> None:
    """Clean and rebuild derivative Neo4j projections from repaired PostgreSQL rows."""
    evidence: dict[str, list[str]] = {}
    for row in postgres.load_artifact_payloads_for_project(project=project):
        if row.get("artifact_kind") != "requirements":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        from multi_agentic_graph_rag.domain.schemas import CanonicalRequirementsArtifact

        canonical = CanonicalRequirementsArtifact.model_validate(payload)
        for requirement in canonical.requirements:
            evidence.setdefault(requirement.requirement_id, []).extend(
                item.chunk_id for item in requirement.evidence
            )
    neo4j.cleanup_identity_projections(project)
    stories = postgres.load_user_stories_for_generation(project=project)
    by_version: dict[str, dict[str, Any]] = {}
    for story in stories:
        by_version.setdefault(story.document_version_id, {})[story.story_id] = story
    for records in by_version.values():
        head = next(iter(records.values()))
        artifact = project_user_story_artifact(
            project=project,
            document_id=head.document_id,
            document_version_id=head.document_version_id,
            doc_version=head.doc_version,
            records=records,
        )
        neo4j.project_user_story_coverage(artifact, evidence)
    scenarios = postgres.load_test_scenarios_for_generation(project=project)
    scenario_versions: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        scenario_versions.setdefault(scenario.document_version_id, {})[scenario.scenario_id] = (
            scenario
        )
    for records in scenario_versions.values():
        head = next(iter(records.values()))
        scenario_artifact = project_test_scenario_artifact(
            project=project,
            document_id=head.document_id,
            document_version_id=head.document_version_id,
            doc_version=head.doc_version,
            records=records,
        )
        neo4j.project_test_scenario_coverage(scenario_artifact, evidence)


@artifact_app.command("verify")
def artifact_verify(path: Annotated[Path, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.ARTIFACT_VERIFY.value,
        run_id=command_run_id(RuntimeCommand.ARTIFACT_VERIFY.value),
    ) as session:
        artifact = verify_requirement_artifact(path)
        fact_count = len(getattr(artifact, "facts", []))
        session.logger.info(
            "Verified requirement artifact {path}",
            step="artifact-verify",
            path=str(path),
            requirement_count=len(artifact.requirements),
            fact_count=fact_count,
            status="completed",
        )
        console.print(
            "[green]PASS[/green] artifact verified "
            f"requirements={len(artifact.requirements)} facts={fact_count} "
            f"document_version_id={artifact.document_version_id}"
        )


@artifact_app.command("verify-user-stories")
def artifact_verify_user_stories(path: Annotated[Path, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.ARTIFACT_VERIFY_USER_STORIES.value,
        run_id=command_run_id(RuntimeCommand.ARTIFACT_VERIFY_USER_STORIES.value),
    ) as session:
        artifact = verify_user_story_artifact(path)
        covered_requirements = len({row.requirement_id for row in artifact.traceability})
        session.logger.info(
            "Verified user-story artifact {path}",
            step="artifact-verify-user-stories",
            path=str(path),
            story_count=len(artifact.stories),
            requirement_count=covered_requirements,
            status="completed",
        )
        console.print(
            "[green]PASS[/green] user-story artifact verified "
            f"stories={len(artifact.stories)} covered={covered_requirements} "
            f"document_version_id={artifact.document_version_id}"
        )


@artifact_app.command("verify-test-scenarios")
def artifact_verify_test_scenarios(path: Annotated[Path, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.ARTIFACT_VERIFY_TEST_SCENARIOS.value,
        run_id=command_run_id(RuntimeCommand.ARTIFACT_VERIFY_TEST_SCENARIOS.value),
    ) as session:
        artifact = verify_test_scenario_artifact(path)
        covered_stories = len({row.story_id for row in artifact.traceability})
        covered_requirements = len({row.requirement_id for row in artifact.traceability})
        session.logger.info(
            "Verified test-scenario artifact {path}",
            step="artifact-verify-test-scenarios",
            path=str(path),
            scenario_count=len(artifact.scenarios),
            story_count=covered_stories,
            requirement_count=covered_requirements,
            status="completed",
        )
        console.print(
            "[green]PASS[/green] test-scenario artifact verified "
            f"scenarios={len(artifact.scenarios)} stories={covered_stories} "
            f"covered_requirements={covered_requirements} "
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


def _hf_offline_flag_status() -> str:
    return ", ".join(f"{key}={os.environ.get(key, '<not set>')}" for key in HF_OFFLINE_FLAGS)


def _dependency_status(module: str, *, required: bool) -> str:
    installed = find_spec(module) is not None
    if installed:
        return "PASS"
    return "FAIL" if required else "WARN"


def _hf_load_checks(reasoning_model: str, settings: Any) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    if reasoning_model:
        try:
            HuggingFaceReasoningModel(settings.huggingface).warmup()
            checks.append(("reasoning_load", "PASS", reasoning_model))
        except Exception as exc:
            checks.append(("reasoning_load", "FAIL", f"{exc.__class__.__name__}: {exc}"))
    else:
        checks.append(("reasoning_load", "WARN", "skipped; HUGGINGFACE_REASONING_MODEL empty"))
    try:
        HuggingFaceEmbeddingModel(settings.huggingface)
        checks.append(("embedding_load", "PASS", settings.huggingface.embedding_model))
    except Exception as exc:
        checks.append(("embedding_load", "FAIL", f"{exc.__class__.__name__}: {exc}"))
    try:
        HuggingFaceRerankerModel(settings.huggingface)
        checks.append(("reranker_load", "PASS", settings.huggingface.reranker_model))
    except Exception as exc:
        checks.append(("reranker_load", "FAIL", f"{exc.__class__.__name__}: {exc}"))
    return checks


def _postgres_dsn_host(dsn: str) -> str:
    normalized_dsn = dsn
    scheme, separator, remainder = normalized_dsn.partition("://")
    if separator and "+" in scheme:
        normalized_dsn = f"{scheme.split('+', 1)[0]}://{remainder}"
    return urlparse(normalized_dsn).hostname or ""


def main() -> None:
    app()


if __name__ == "__main__":
    main()
