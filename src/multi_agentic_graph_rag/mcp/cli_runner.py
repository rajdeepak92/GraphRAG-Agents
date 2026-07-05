"""Safe subprocess wrapper for existing MARAG CLI commands."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.mcp.contracts import CliRunResult


class CliExecutionError(RuntimeError):
    """Raised when command output cannot be interpreted as requested."""


def run_marag_command(
    args: list[str],
    *,
    project_root: Path,
    timeout_seconds: int = 1800,
    expect_json: bool = False,
) -> CliRunResult:
    command = ["uv", "run", "marag", *args]

    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CliRunResult(
            command=command,
            exit_code=-1,
            stdout=_decode_timeout_stream(exc.stdout),
            stderr=f"command timed out after {timeout_seconds} seconds",
            parsed_json=None,
        )

    parsed_json: dict[str, Any] | None = None
    if expect_json and completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CliExecutionError(f"expected JSON output from marag command: {exc}") from exc
        if not isinstance(parsed, dict):
            raise CliExecutionError("expected JSON object output from marag command")
        parsed_json = parsed

    return CliRunResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        parsed_json=parsed_json,
    )


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
