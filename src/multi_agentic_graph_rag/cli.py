"""Command-line interface for the MARAG platform."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import async_sessionmaker

from multi_agentic_graph_rag.application.workflows.ingestion.runner import (
    run_ingestion_skeleton,
)
from multi_agentic_graph_rag.config.providers import (
    EmbeddingProvider,
    GraphStoreProvider,
    ReasoningLLMProvider,
    VectorStoreProvider,
)
from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.domain.commands import (
    IngestDocumentCommand,
    ProviderOverrides,
)
from multi_agentic_graph_rag.domain.enums import ReplacePolicy
from multi_agentic_graph_rag.infrastructure.neo4j.client import neo4j_driver_scope
from multi_agentic_graph_rag.infrastructure.neo4j.schema import ensure_neo4j_schema
from multi_agentic_graph_rag.infrastructure.postgres.health import (
    format_postgres_health_report,
    run_postgres_health_check,
)
from multi_agentic_graph_rag.infrastructure.postgres.session import create_postgres_engine

from . import __version__
from .bootstrap import CheckResult, CheckStatus, configuration_checks, doctor_checks

app = typer.Typer(
    name="marag",
    help="Multi-Agentic Knowledge-Graph RAG command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)

db_check_app = typer.Typer(help="Database health checks.")
vector_app = typer.Typer(help="Vector index commands.")

app.add_typer(db_check_app, name="db-check")
app.add_typer(vector_app, name="vectors")

console = Console()

_STATUS_STYLE: dict[CheckStatus, str] = {
    "PASS": "green",
    "WARN": "yellow",
    "FAIL": "red",
}


def _render_results(
    title: str,
    results: Sequence[CheckResult],
) -> None:
    """Render diagnostic results in a table."""

    table = Table(title=title)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    for result in results:
        style = _STATUS_STYLE[result.status]

        table.add_row(
            result.name,
            f"[{style}]{result.status}[/{style}]",
            result.detail,
        )

    console.print(table)


def _execute_checks(
    title: str,
    results: Sequence[CheckResult],
) -> None:
    """Render checks and return a nonzero exit code on failure."""

    _render_results(title, results)

    if any(result.status == "FAIL" for result in results):
        raise typer.Exit(code=1)


@app.command("version")
def version_command() -> None:
    """Print the installed application version."""

    console.print(f"marag {__version__}")


@app.command("config-check")
def config_check_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to config.json override."),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option("--project-root", help="Override resolved project root."),
    ] = None,
    reasoning_provider: Annotated[
        ReasoningLLMProvider | None,
        typer.Option("--reasoning-provider", help="Override reasoning LLM provider."),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider | None,
        typer.Option("--embedding-provider", help="Override embedding provider."),
    ] = None,
    vector_store_provider: Annotated[
        VectorStoreProvider | None,
        typer.Option("--vector-store-provider", help="Override vector store provider."),
    ] = None,
    graph_store_provider: Annotated[
        GraphStoreProvider | None,
        typer.Option("--graph-store-provider", help="Override graph store provider."),
    ] = None,
) -> None:
    """Validate Phase 2 configuration and create approved runtime directories."""

    overrides: dict[str, object] = {}

    if project_root is not None:
        overrides.setdefault("paths", {})
        assert isinstance(overrides["paths"], dict)
        overrides["paths"]["project_root"] = project_root

    if reasoning_provider is not None:
        overrides.setdefault("requirement_discovery", {})
        assert isinstance(overrides["requirement_discovery"], dict)
        overrides["requirement_discovery"]["reasoning_provider"] = reasoning_provider.value

    if embedding_provider is not None:
        overrides.setdefault("embedding", {})
        assert isinstance(overrides["embedding"], dict)
        overrides["embedding"]["provider"] = embedding_provider.value

    if vector_store_provider is not None:
        overrides.setdefault("providers", {})
        assert isinstance(overrides["providers"], dict)
        overrides["providers"]["vector_store_provider"] = vector_store_provider.value

    if graph_store_provider is not None:
        overrides.setdefault("providers", {})
        assert isinstance(overrides["providers"], dict)
        overrides["providers"]["graph_store_provider"] = graph_store_provider.value

    # Force-load once so enum/provider errors are raised before rendering.
    load_settings(config_path=config, overrides=overrides)

    _execute_checks(
        title="MARAG Configuration Check",
        results=configuration_checks(config_path=config, overrides=overrides),
    )


@app.command("doctor")
def doctor_command() -> None:
    """Validate the Phase 1 local development environment."""

    _execute_checks(
        title="MARAG Environment Doctor",
        results=doctor_checks(),
    )


@db_check_app.command("postgres")
def db_check_postgres() -> None:
    """Run PostgreSQL Phase 4 health checks."""

    asyncio.run(_db_check_postgres())


async def _db_check_postgres() -> None:
    settings = load_settings()
    engine = create_postgres_engine(settings)

    try:
        report = await run_postgres_health_check(engine)

        for line in format_postgres_health_report(report):
            typer.echo(line)

        if not report.passed:
            raise typer.Exit(code=1)

    finally:
        await engine.dispose()


def _secret_value(value: Any) -> str:
    """Return a plain string from pydantic SecretStr or normal string."""

    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())

    return str(value)


@app.command("neo4j-schema")
def neo4j_schema_command() -> None:
    """Create or verify Neo4j constraints."""

    asyncio.run(_neo4j_schema())


async def _neo4j_schema() -> None:
    settings = load_settings()
    neo4j_settings = settings.neo4j

    async with neo4j_driver_scope(
        uri=str(neo4j_settings.uri),
        username=str(neo4j_settings.username),
        password=_secret_value(neo4j_settings.password),
        database=str(neo4j_settings.database),
    ) as driver:
        await driver.verify_connectivity()
        await ensure_neo4j_schema(driver, database=str(neo4j_settings.database))

    typer.echo("PASS Neo4j schema constraints are ready.")


@vector_app.command("check")
def vector_check_command() -> None:
    """Verify local Chroma vector store."""

    from multi_agentic_graph_rag.infrastructure.chroma.client import (
        create_persistent_chroma_client,
    )
    from multi_agentic_graph_rag.infrastructure.chroma.vector_store import ChromaVectorStore

    settings = load_settings()

    client = create_persistent_chroma_client(settings.paths.chroma_persist_dir)
    store = ChromaVectorStore(
        client=client,
        collection_name=settings.chroma.collection_name,
    )

    store.verify_connection()

    typer.echo("PASS Chroma vector store is ready.")


@vector_app.command("index-document-version")
def vector_index_document_version_command(
    document_version_id: Annotated[
        UUID,
        typer.Argument(help="Document version UUID to index into Chroma."),
    ],
) -> None:
    """Index active chunks for one document version into Chroma."""

    asyncio.run(_vector_index_document_version(document_version_id))


async def _vector_index_document_version(document_version_id: UUID) -> None:
    from multi_agentic_graph_rag.application.services.index_chunk_vectors import (
        IndexChunkVectorsService,
    )
    from multi_agentic_graph_rag.infrastructure.chroma.client import (
        create_persistent_chroma_client,
    )
    from multi_agentic_graph_rag.infrastructure.chroma.vector_store import ChromaVectorStore
    from multi_agentic_graph_rag.infrastructure.embeddings.huggingface import (
        HuggingFaceEmbeddingAdapter,
    )
    from multi_agentic_graph_rag.infrastructure.postgres.repositories import (
        SqlAlchemyChunkRepository,
    )

    settings = load_settings()

    engine = create_postgres_engine(settings)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    client = create_persistent_chroma_client(settings.paths.chroma_persist_dir)
    vector_store = ChromaVectorStore(
        client=client,
        collection_name=settings.chroma.collection_name,
    )

    embedding_provider = HuggingFaceEmbeddingAdapter(
        model_name=settings.huggingface.embedding_model,
        normalize_embeddings=settings.embedding.normalize,
    )

    try:
        async with session_factory() as session:
            service = IndexChunkVectorsService(
                chunk_repository=SqlAlchemyChunkRepository(session),
                embedding_provider=embedding_provider,
                vector_store=vector_store,
            )

            result = await service.index_document_version(
                document_version_id=document_version_id,
            )

            typer.echo(
                "PASS indexed "
                f"{result.indexed_count} chunks for document_version_id="
                f"{result.document_version_id} using {result.embedding_fingerprint}"
            )

    finally:
        await engine.dispose()


@app.command("ingest")
def ingest_command(
    project: Annotated[
        str,
        typer.Option("--project", help="Project key, for example PROJECT_1."),
    ],
    document: Annotated[
        Path,
        typer.Option("--document", help="Path to the document to ingest."),
    ],
    version: Annotated[
        str,
        typer.Option("--version", help="Logical document version, for example 1.0."),
    ],
    logical_name: Annotated[
        str | None,
        typer.Option("--logical-name", help="Stable logical document name."),
    ] = None,
    reasoning_provider: Annotated[
        ReasoningLLMProvider | None,
        typer.Option("--reasoning-provider", help="Override reasoning provider."),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider | None,
        typer.Option("--embedding-provider", help="Override embedding provider."),
    ] = None,
    replace_version: Annotated[
        bool,
        typer.Option("--replace-version", help="Explicitly replace same document/version."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Run orchestration skeleton without real ingestion side effects.",
        ),
    ] = False,
    resume_run: Annotated[
        str | None,
        typer.Option("--resume-run", help="Resume/checkpoint thread id."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json-output", help="Print final graph state as JSON."),
    ] = False,
    checkpoint: Annotated[
        Literal["memory", "postgres"],
        typer.Option("--checkpoint", help="Checkpoint backend for Phase 7 skeleton."),
    ] = "memory",
    setup_checkpointer: Annotated[
        bool,
        typer.Option(
            "--setup-checkpointer",
            help="Create LangGraph checkpoint tables. Use once for postgres mode.",
        ),
    ] = False,
) -> None:
    """Run the Phase 7 LangGraph ingestion skeleton."""

    command = IngestDocumentCommand(
        project_key=project,
        document_path=document,
        document_version=version,
        logical_document_name=logical_name,
        provider_overrides=ProviderOverrides(
            reasoning_provider=reasoning_provider.value if reasoning_provider else None,
            embedding_provider=embedding_provider.value if embedding_provider else None,
        ),
        replace_policy=(ReplacePolicy.REPLACE_VERSION if replace_version else ReplacePolicy.REJECT),
        dry_run=dry_run,
    )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    final_state = asyncio.run(
        run_ingestion_skeleton(
            command=command,
            resume_run=resume_run,
            checkpoint_mode=checkpoint,
            setup_checkpointer=setup_checkpointer,
        )
    )

    if json_output:
        console.print_json(json.dumps(final_state, indent=2, default=str))
        return

    console.print(
        "[green]PASS[/green] ingestion skeleton completed "
        f"run_id={final_state.get('run_id')} "
        f"current_step={final_state.get('current_step')}"
    )


if __name__ == "__main__":
    app()
