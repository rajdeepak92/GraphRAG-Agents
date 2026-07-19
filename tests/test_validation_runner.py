"""Tests for the allow-listed validation runner and status classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from multi_agentic_graph_rag.services.validation_runner import (
    ValidationError,
    ValidationRunner,
    classify_static,
)


def _runner(tmp_path: Path) -> ValidationRunner:
    return ValidationRunner(tmp_path, python_executable=sys.executable, timeout=60)


def test_parse_python_detects_syntax_errors(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    runner = _runner(tmp_path)
    assert runner.parse_python(["good.py"]).ok
    bad = runner.parse_python(["bad.py"])
    assert not bad.ok
    assert bad.errors


def test_run_command_rejects_non_allowlisted(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(ValidationError):
        runner.run_command("rm", ["good.py"])


def test_paths_outside_worktree_are_rejected(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(ValidationError):
        runner.parse_python(["../escape.py"])
    with pytest.raises(ValidationError):
        runner.run_command("collect", ["../../etc/passwd"])


def test_collect_runs_pytest_on_a_generated_test(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    runner = _runner(tmp_path)
    result = runner.run_command("collect", ["test_sample.py"])
    if result.unavailable:  # pragma: no cover - environment without pytest on the interpreter
        pytest.skip("pytest not available on the target interpreter")
    assert result.ok, result.stderr


def test_classify_static_taxonomy() -> None:
    assert classify_static(parse_ok=False, collect_ok=False, static_ok=False) == "FAILED_VALIDATION"
    assert classify_static(parse_ok=True, collect_ok=False, static_ok=False) == "GENERATED"
    assert classify_static(parse_ok=True, collect_ok=True, static_ok=False) == "COLLECTABLE"
    assert classify_static(parse_ok=True, collect_ok=True, static_ok=True) == "STATICALLY_VALIDATED"
