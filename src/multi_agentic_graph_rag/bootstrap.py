"""Phase 1 repository and environment diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Literal

from multi_agentic_graph_rag.config.paths import create_directories, set_cache_environment
from multi_agentic_graph_rag.config.settings import SettingsError, load_settings

from . import DISTRIBUTION_NAME

CheckStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one deterministic diagnostic check."""

    name: str
    status: CheckStatus
    detail: str


def find_project_root(start: Path | None = None) -> Path | None:
    """Locate the nearest parent directory containing pyproject.toml."""

    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    return None


def _is_git_ignored(root: Path, relative_path: str) -> bool:
    """Return whether Git ignores a repository-relative path."""

    git_executable = shutil.which("git")

    if git_executable is None:
        return False

    completed = subprocess.run(
        [
            git_executable,
            "-C",
            str(root),
            "check-ignore",
            "--quiet",
            relative_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    return completed.returncode == 0


def configuration_checks(
    *,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[CheckResult, ...]:
    """Validate Phase 2 configuration and runtime paths."""

    try:
        settings = load_settings(config_path=config_path, overrides=overrides)
    except SettingsError as exc:
        return (
            CheckResult(
                name="Settings",
                status="FAIL",
                detail=str(exc),
            ),
        )

    create_directories(settings.paths.approved_runtime_directories())
    create_directories(tuple(settings.paths.cache_environment().values()))
    set_cache_environment(settings.paths.cache_environment())

    return tuple(
        CheckResult(
            name=check.name,
            status=check.status,
            detail=check.detail,
        )
        for check in settings.diagnostics()
    )


def doctor_checks() -> tuple[CheckResult, ...]:
    """Validate the Phase 1 development environment."""

    results: list[CheckResult] = []

    python_version = sys.version_info
    python_supported = python_version[:2] == (3, 12)

    results.append(
        CheckResult(
            name="Python",
            status="PASS" if python_supported else "FAIL",
            detail=(f"{python_version.major}.{python_version.minor}.{python_version.micro}"),
        ),
    )

    for executable_name in ("uv", "git"):
        executable_path = shutil.which(executable_name)

        results.append(
            CheckResult(
                name=executable_name,
                status="PASS" if executable_path else "FAIL",
                detail=executable_path or "Executable not found on PATH.",
            ),
        )

    try:
        installed_version = distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        results.append(
            CheckResult(
                name="Installed package",
                status="FAIL",
                detail=f"{DISTRIBUTION_NAME} is not installed.",
            ),
        )
    else:
        results.append(
            CheckResult(
                name="Installed package",
                status="PASS",
                detail=f"{DISTRIBUTION_NAME} {installed_version}",
            ),
        )

    return tuple(results)
