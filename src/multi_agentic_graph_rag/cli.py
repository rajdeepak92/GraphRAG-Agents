"""Command-line interface for the MARAG platform."""

from __future__ import annotations

from collections.abc import Sequence

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .bootstrap import CheckResult, CheckStatus, configuration_checks, doctor_checks

app = typer.Typer(
    name="marag",
    help="Multi-Agentic Knowledge-Graph RAG command-line interface.",
    no_args_is_help=True,
    add_completion=False,
)

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
def config_check_command() -> None:
    """Validate the Phase 1 repository configuration."""

    _execute_checks(
        title="MARAG Configuration Check",
        results=configuration_checks(),
    )


@app.command("doctor")
def doctor_command() -> None:
    """Validate the Phase 1 local development environment."""

    _execute_checks(
        title="MARAG Environment Doctor",
        results=doctor_checks(),
    )


if __name__ == "__main__":
    app()
