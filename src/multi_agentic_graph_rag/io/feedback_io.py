"""Frontend-agnostic human feedback I/O for HFIL."""

from __future__ import annotations

from typing import Protocol

from rich.console import Console
from rich.table import Table


class FeedbackIO(Protocol):
    def show(self, message: str) -> None: ...

    def prompt(self, message: str) -> str: ...

    def show_scenarios(self, scenarios: list[dict[str, object]]) -> None: ...


class CLIFeedbackIO:
    """CLI-backed feedback I/O using Rich rendering and stdin."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def show(self, message: str) -> None:
        self.console.print(message)

    def prompt(self, message: str) -> str:
        return self.console.input(message)

    def show_scenarios(self, scenarios: list[dict[str, object]]) -> None:
        table = Table(title="Test Scenarios")
        table.add_column("Scenario ID")
        table.add_column("Story ID")
        table.add_column("Scenario Text")
        for scenario in scenarios:
            table.add_row(
                str(scenario.get("scenario_id", "")),
                str(scenario.get("story_id", "")),
                str(scenario.get("scenario_text", "")),
            )
        self.console.print(table)
