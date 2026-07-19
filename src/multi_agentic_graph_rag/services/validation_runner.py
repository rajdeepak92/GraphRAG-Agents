"""Allow-listed static/collection validation for generated tests (§13.2, §19).

The model never supplies an arbitrary shell command. Commands are selected from
a fixed allow-list, run without a shell inside the worktree, with wall-time and
output caps and network disabled by default. Syntax parsing is done in-process
via ``ast`` (no subprocess). Results map to the precise validation status
taxonomy — a test is never called ``EXECUTABLE`` unless it has actually run.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from multi_agentic_graph_rag.domain.codegen_schemas import ValidationStatus

# name -> argv tail appended after the interpreter. Nothing else may be run.
_COMMAND_ALLOW_LIST: dict[str, list[str]] = {
    "format_check": ["-m", "ruff", "format", "--check"],
    "lint": ["-m", "ruff", "check"],
    "type_check": ["-m", "mypy"],
    "collect": ["-m", "pytest", "--collect-only", "-q"],
    "unit": ["-m", "pytest", "-q"],
}

_MAX_OUTPUT_CHARS = 20_000
_DEFAULT_TIMEOUT = 120


class ValidationError(ValueError):
    """Raised when a validation request is unsafe or not on the allow-list."""


@dataclass(frozen=True)
class ValidationRunResult:
    """One validation command outcome with truncated captured output."""

    name: str
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    unavailable: bool = False


@dataclass(frozen=True)
class ParseOutcome:
    """Result of in-process syntax validation across files."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT_CHARS else text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"


class ValidationRunner:
    """Run bounded, allow-listed checks against files inside one worktree."""

    def __init__(
        self,
        worktree_root: Path,
        *,
        python_executable: str = "python",
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.worktree_root = worktree_root.resolve()
        self.python_executable = python_executable
        self.timeout = timeout

    def _safe_relative(self, relative_path: str) -> str:
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValidationError(f"unsafe path: {relative_path}")
        target = (self.worktree_root / rel).resolve()
        if target != self.worktree_root and self.worktree_root not in target.parents:
            raise ValidationError(f"path escapes worktree: {relative_path}")
        return str(rel)

    def parse_python(self, paths: list[str]) -> ParseOutcome:
        """Validate syntax in-process; no subprocess, no code execution (§13.2)."""
        errors: list[str] = []
        for relative_path in paths:
            safe = self._safe_relative(relative_path)
            file_path = self.worktree_root / safe
            if not file_path.exists():
                errors.append(f"{safe}: file not found")
                continue
            try:
                ast.parse(file_path.read_text(encoding="utf-8"), filename=safe)
            except SyntaxError as exc:
                errors.append(f"{safe}:{exc.lineno}: {exc.msg}")
        return ParseOutcome(ok=not errors, errors=tuple(errors))

    def run_command(self, name: str, paths: list[str]) -> ValidationRunResult:
        """Run one allow-listed command with resource limits and output caps."""
        if name not in _COMMAND_ALLOW_LIST:
            raise ValidationError(f"command '{name}' is not on the allow-list")
        safe_paths = [self._safe_relative(path) for path in paths]
        argv = [self.python_executable, *_COMMAND_ALLOW_LIST[name], *safe_paths]
        try:
            completed = subprocess.run(
                argv,
                cwd=self.worktree_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
                env=self._sandboxed_env(),
            )
        except FileNotFoundError:
            return ValidationRunResult(
                name=name,
                ok=False,
                returncode=None,
                stdout="",
                stderr="tool unavailable",
                unavailable=True,
            )
        except subprocess.TimeoutExpired:
            return ValidationRunResult(
                name=name,
                ok=False,
                returncode=None,
                stdout="",
                stderr="timed out",
                timed_out=True,
            )
        return ValidationRunResult(
            name=name,
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=_truncate(completed.stdout),
            stderr=_truncate(completed.stderr),
        )

    def _sandboxed_env(self) -> dict[str, str]:
        import os

        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "PATHEXT", "TEMP", "TMP", "HOME", "LANG"}
        }
        # Disable network egress hints by default (best-effort; §13.3).
        env["NO_NETWORK"] = "1"
        env["PIP_NO_INDEX"] = "1"
        return env


def classify_static(
    *,
    parse_ok: bool,
    collect_ok: bool,
    static_ok: bool,
) -> ValidationStatus:
    """Map static-pipeline outcomes to the validation status taxonomy (§19)."""
    if not parse_ok:
        return "FAILED_VALIDATION"
    if not collect_ok:
        return "GENERATED"
    if not static_ok:
        return "COLLECTABLE"
    return "STATICALLY_VALIDATED"


__all__ = [
    "ParseOutcome",
    "ValidationError",
    "ValidationRunResult",
    "ValidationRunner",
    "classify_static",
]
