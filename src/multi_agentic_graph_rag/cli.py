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

from common_defs import EnvVar, RuntimeCommand
from multi_agentic_graph_rag import __version__
from multi_agentic_graph_rag.agents.ingestion_document_agent import IngestionDocumentAgent
from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGeneratorAgent
from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGeneratorAgent
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
from multi_agentic_graph_rag.services.artifacts import (
    verify_requirement_artifact,
    verify_test_scenario_artifact,
    verify_user_story_artifact,
)
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
app.add_typer(run_app, name="run")
app.add_typer(artifact_app, name="artifact")
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
        rows = PostgresStore(settings).load_coverage_status(
            project=project,
            document_version_id=document_version_id,
        )
        session.logger.info(
            "Computed coverage for {project}",
            step="coverage",
            project=project,
            requirement_count=len(rows),
            status="completed",
        )
        if json_output:
            console.print_json(json.dumps(rows))
            return
        table = Table(title=f"Coverage — {project}")
        table.add_column("Requirement")
        table.add_column("Coverage")
        table.add_column("Stories", justify="right")
        table.add_column("Scenarios", justify="right")
        for row in rows:
            table.add_row(
                str(row["requirement_id"]),
                str(row["coverage_status"]),
                str(len(row["story_ids"])),
                str(row["scenario_count"]),
            )
        console.print(table)


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


@artifact_app.command("verify")
def artifact_verify(path: Annotated[Path, typer.Argument()]) -> None:
    with command_session(
        project="_system",
        version=__version__,
        command=RuntimeCommand.ARTIFACT_VERIFY.value,
        run_id=command_run_id(RuntimeCommand.ARTIFACT_VERIFY.value),
    ) as session:
        artifact = verify_requirement_artifact(path)
        session.logger.info(
            "Verified requirement artifact {path}",
            step="artifact-verify",
            path=str(path),
            requirement_count=len(artifact.requirements),
            fact_count=len(artifact.facts),
            status="completed",
        )
        console.print(
            "[green]PASS[/green] artifact verified "
            f"requirements={len(artifact.requirements)} facts={len(artifact.facts)} "
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
        session.logger.info(
            "Verified user-story artifact {path}",
            step="artifact-verify-user-stories",
            path=str(path),
            story_count=len(artifact.stories),
            requirement_count=len(artifact.coverage),
            status="completed",
        )
        console.print(
            "[green]PASS[/green] user-story artifact verified "
            f"stories={len(artifact.stories)} covered={len(artifact.coverage)} "
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
        session.logger.info(
            "Verified test-scenario artifact {path}",
            step="artifact-verify-test-scenarios",
            path=str(path),
            scenario_count=len(artifact.scenarios),
            story_count=len(artifact.coverage),
            requirement_count=len(artifact.requirement_coverage),
            status="completed",
        )
        console.print(
            "[green]PASS[/green] test-scenario artifact verified "
            f"scenarios={len(artifact.scenarios)} stories={len(artifact.coverage)} "
            f"covered_requirements={len(artifact.requirement_coverage)} "
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
