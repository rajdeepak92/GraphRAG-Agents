"""Phase 1 repository and environment diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Literal

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


def configuration_checks() -> tuple[CheckResult, ...]:
    """Validate the Phase 1 repository configuration."""

    root = find_project_root()

    if root is None:
        return (
            CheckResult(
                name="Project root",
                status="FAIL",
                detail="No pyproject.toml was found.",
            ),
        )

    results: list[CheckResult] = [
        CheckResult(
            name="Project root",
            status="PASS",
            detail=str(root),
        ),
    ]

    required_files = (
        ".python-version",
        ".env.example",
        ".gitignore",
        ".pre-commit-config.yaml",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "src/multi_agentic_graph_rag/__init__.py",
        "src/multi_agentic_graph_rag/bootstrap.py",
        "src/multi_agentic_graph_rag/cli.py",
    )

    for relative_path in required_files:
        exists = (root / relative_path).is_file()

        results.append(
            CheckResult(
                name=relative_path,
                status="PASS" if exists else "FAIL",
                detail="Present" if exists else "Missing",
            ),
        )

    python_pin_path = root / ".python-version"
    python_pin = (
        python_pin_path.read_text(encoding="utf-8").strip() if python_pin_path.is_file() else ""
    )
    python_pin_valid = python_pin == "3.12" or python_pin.startswith("3.12.")

    results.append(
        CheckResult(
            name="Python pin",
            status="PASS" if python_pin_valid else "FAIL",
            detail=python_pin or "Missing",
        ),
    )

    env_is_ignored = _is_git_ignored(root, ".env")

    results.append(
        CheckResult(
            name=".env ignore rule",
            status="PASS" if env_is_ignored else "FAIL",
            detail=(".env is ignored by Git." if env_is_ignored else ".env is not ignored by Git."),
        ),
    )

    git_repository_exists = (root / ".git").is_dir()

    results.append(
        CheckResult(
            name="Git repository",
            status="PASS" if git_repository_exists else "FAIL",
            detail=(str(root / ".git") if git_repository_exists else ".git directory is missing."),
        ),
    )

    return tuple(results)


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
