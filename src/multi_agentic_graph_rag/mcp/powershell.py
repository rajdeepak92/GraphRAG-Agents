"""Controlled PowerShell script execution for repo-owned MCP scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

from multi_agentic_graph_rag.mcp.contracts import CliRunResult


def run_powershell_script(
    script_path: Path,
    *,
    project_root: Path,
    args: list[str] | None = None,
    timeout_seconds: int = 120,
) -> CliRunResult:
    root = project_root.resolve()
    scripts_root = (root / "scripts" / "mcp").resolve()
    resolved_script = script_path.resolve()

    if scripts_root != resolved_script.parent:
        return CliRunResult(
            command=[],
            exit_code=1,
            stdout="",
            stderr=f"script is not under scripts/mcp: {script_path}",
            parsed_json=None,
        )
    if not resolved_script.exists():
        return CliRunResult(
            command=[],
            exit_code=1,
            stdout="",
            stderr=f"script not found: {resolved_script}",
            parsed_json=None,
        )

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(resolved_script),
        *(args or []),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        return CliRunResult(
            command=command,
            exit_code=-1,
            stdout=stdout,
            stderr=f"PowerShell script timed out after {timeout_seconds} seconds",
            parsed_json=None,
        )

    return CliRunResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        parsed_json=None,
    )
