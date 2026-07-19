"""Focused tests for the non-executing Stage-4 generated-case validator."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from multi_agentic_graph_rag.services.validation_runner import (
    GeneratedCaseValidationRequest,
    ValidationError,
    ValidationRunner,
    classify_static,
)

PYTHON_PATH = "tests/sensor/Tc100001ValidateTemperatureSensorThreshold.py"
ROBOT_PATH = "tests_robot/sensor/Tc100001ValidateTemperatureSensorThreshold.robot"
WRAPPER_PATH = "test_lib/sensor/sensor_wrappers.py"
STEM = "Tc100001ValidateTemperatureSensorThreshold"

VALID_PYTHON = """from __future__ import annotations

import logging


class Tc100001ValidateTemperatureSensorThreshold:
    test_variables: dict[str, object] = {}

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.test_variables = {}

    def test_setup(self) -> bool:
        self.logger.info("Setup")
        return True

    def execute_test(self) -> bool:
        step_results: list[bool] = []
        step_results.append(bool(True))
        return bool(step_results) and all(step_results)

    def test_teardown(self) -> bool:
        self.logger.info("Teardown")
        return True

    def run_test(self) -> bool:
        if not self.test_setup():
            return False

        execution_ok = False
        teardown_ok = False
        try:
            execution_ok = bool(self.execute_test())
        except Exception:
            self.logger.exception("Execution stage failed")
            execution_ok = False
        finally:
            try:
                teardown_ok = bool(self.test_teardown())
            except Exception:
                self.logger.exception("Teardown stage failed")
                teardown_ok = False

        return execution_ok and teardown_ok


if __name__ == "__main__":
    result = Tc100001ValidateTemperatureSensorThreshold().run_test()
    raise SystemExit(0 if result else 1)
"""

VALID_ROBOT = """*** Settings ***
Library    tests.sensor.Tc100001ValidateTemperatureSensorThreshold

*** Test Cases ***
TC100001 Validate Temperature Sensor Threshold
    ${result}=    Run Test
    Should Be True    ${result}
"""


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case(tmp_path: Path, *, python: str = VALID_PYTHON, robot: str = VALID_ROBOT) -> None:
    python_path = tmp_path / PYTHON_PATH
    robot_path = tmp_path / ROBOT_PATH
    python_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text(python, encoding="utf-8")
    robot_path.write_text(robot, encoding="utf-8")


def _request(tmp_path: Path, **overrides: Any) -> GeneratedCaseValidationRequest:
    values: dict[str, Any] = {
        "python_file": PYTHON_PATH,
        "robot_file": ROBOT_PATH,
        "expected_hashes": {
            PYTHON_PATH: _hash(tmp_path / PYTHON_PATH),
            ROBOT_PATH: _hash(tmp_path / ROBOT_PATH),
        },
    }
    values.update(overrides)
    return GeneratedCaseValidationRequest(**values)


def _runner(tmp_path: Path, **kwargs: Any) -> ValidationRunner:
    return ValidationRunner(tmp_path, python_executable=sys.executable, timeout=60, **kwargs)


def _successful_process(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_parse_python_parses_and_compiles_without_import(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def f(:\n", encoding="utf-8")
    runner = _runner(tmp_path)
    assert runner.parse_python(["good.py"]).ok
    bad = runner.parse_python(["bad.py"])
    assert not bad.ok and bad.errors
    assert not runner.parse_python(["suite.robot"]).ok


def test_generic_commands_reject_unsafe_and_disable_target_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(ValidationError):
        runner.run_command("rm", ["good.py"])
    with pytest.raises(ValidationError):
        runner.run_command("lint", ["../escape.py"])

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("disabled target discovery launched a process"),
    )
    result = runner.run_command("collect", ["tests/sensor/Tc100001X.py"])
    assert not result.ok and result.unavailable


def test_complete_generated_case_pipeline_and_exact_robot_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    calls: list[list[str]] = []

    def _run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        assert kwargs["cwd"] == tmp_path.resolve()
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        return _successful_process(argv)

    monkeypatch.setattr(subprocess, "run", _run)
    seen_hooks: list[str] = []
    result = _runner(tmp_path).validate_generated_case(
        _request(tmp_path),
        traceability_hook=lambda context: seen_hooks.append(context.request.python_file),
        exact_test_data_hook=lambda _context: True,
    )
    assert result.ok, result.diagnostics
    assert seen_hooks == [PYTHON_PATH]
    assert len(calls) == 2
    assert calls[1] == [
        sys.executable,
        "-m",
        "robot",
        "--dryrun",
        "--pythonpath",
        str(tmp_path.resolve()),
        "--output",
        "NONE",
        "--report",
        "NONE",
        "--log",
        "NONE",
        ROBOT_PATH,
    ]
    assert all(
        not (part == "-m" and index + 1 < len(argv) and argv[index + 1].casefold() == "pytest")
        for argv in calls
        for index, part in enumerate(argv)
    )


def test_real_isolated_import_constructs_class_without_running_lifecycle(tmp_path: Path) -> None:
    _write_case(tmp_path)
    result = _runner(tmp_path, robot_dryrun=False).validate_generated_case(_request(tmp_path))
    assert result.ok, result.diagnostics
    assert [command.name for command in result.commands] == ["import_smoke"]
    assert not list(tmp_path.rglob("__pycache__"))


def test_pipeline_rejects_production_and_mismatched_paths(tmp_path: Path) -> None:
    _write_case(tmp_path)
    production = tmp_path / "src" / "sensor.py"
    production.parent.mkdir()
    production.write_text("pass\n", encoding="utf-8")
    request = _request(
        tmp_path,
        support_files=("src/sensor.py",),
        expected_hashes={
            PYTHON_PATH: _hash(tmp_path / PYTHON_PATH),
            ROBOT_PATH: _hash(tmp_path / ROBOT_PATH),
            "src/sensor.py": _hash(production),
        },
    )
    result = _runner(tmp_path).validate_generated_case(request)
    assert not result.ok
    assert "WRITE_PATH_FORBIDDEN" in {issue.code for issue in result.issues}


def test_additive_ast_guard_accepts_append_only_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    original = "def existing(value: int) -> bool:\n    return value > 0\n"
    current = (
        original
        + "\n\ndef generated_wrapper(value: int) -> bool:\n"
        + "    import math\n"
        + "    return math.isfinite(value)\n"
    )
    wrapper = tmp_path / WRAPPER_PATH
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(current, encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _successful_process)
    expected = dict(_request(tmp_path).expected_hashes)
    expected[WRAPPER_PATH] = _hash(wrapper)
    result = _runner(tmp_path).validate_generated_case(
        _request(
            tmp_path,
            support_files=(WRAPPER_PATH,),
            shared_preimages={WRAPPER_PATH: original},
            expected_hashes=expected,
        )
    )
    assert result.ok, result.diagnostics


def test_shared_support_requires_journal_origin_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    wrapper = tmp_path / WRAPPER_PATH
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("def generated_wrapper() -> bool:\n    return True\n", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _successful_process)
    expected = dict(_request(tmp_path).expected_hashes)
    expected[WRAPPER_PATH] = _hash(wrapper)
    result = _runner(tmp_path).validate_generated_case(
        _request(tmp_path, support_files=(WRAPPER_PATH,), expected_hashes=expected)
    )
    assert "SHARED_ORIGIN_EVIDENCE_MISSING" in {issue.code for issue in result.issues}

    created_result = _runner(tmp_path).validate_generated_case(
        _request(
            tmp_path,
            support_files=(WRAPPER_PATH,),
            created_files=frozenset({WRAPPER_PATH}),
            expected_hashes=expected,
        )
    )
    assert created_result.ok, created_result.diagnostics


@pytest.mark.parametrize(
    "changed",
    [
        "import os\n\ndef existing(value: int) -> bool:\n    return value > 0\n",
        "def existing(value: str) -> bool:\n    return bool(value)\n",
        "def existing(value: int) -> bool:\n    return value >= 0\n",
        "def existing(value: int) -> bool:\n    return value > 0\n\nVALUE = 1\n",
    ],
)
def test_additive_ast_guard_rejects_existing_behavior_changes(
    tmp_path: Path, changed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    original = "def existing(value: int) -> bool:\n    return value > 0\n"
    wrapper = tmp_path / WRAPPER_PATH
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _successful_process)
    expected = dict(_request(tmp_path).expected_hashes)
    expected[WRAPPER_PATH] = _hash(wrapper)
    result = _runner(tmp_path).validate_generated_case(
        _request(
            tmp_path,
            support_files=(WRAPPER_PATH,),
            shared_preimages={WRAPPER_PATH: original},
            expected_hashes=expected,
        )
    )
    assert "EXISTING_BEHAVIOR_CHANGE_FORBIDDEN" in {issue.code for issue in result.issues}


def test_lifecycle_and_top_level_side_effects_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace(
        "import logging\n", "import logging\n\nopen('hardware', 'w')\n"
    ).replace("def test_teardown(self) -> bool:", "def teardown(self) -> bool:")
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)
    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))
    codes = {issue.code for issue in result.issues}
    assert "TOP_LEVEL_EXECUTION_FORBIDDEN" in codes
    assert "LIFECYCLE_METHOD_MISSING" in codes
    assert result.commands == ()


def test_coordinator_cannot_turn_execution_exception_into_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace(
        'self.logger.exception("Execution stage failed")\n            execution_ok = False',
        'self.logger.exception("Execution stage failed")\n            execution_ok = True',
    )
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)

    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))

    assert "RUN_TEST_COORDINATOR_INVALID" in {issue.code for issue in result.issues}


def test_main_entrypoint_requires_zero_for_true_and_one_for_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace("0 if result else 1", "1 if result else 0")
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)

    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))

    assert "MAIN_ENTRYPOINT_INVALID" in {issue.code for issue in result.issues}


def test_source_control_import_and_executable_call_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace(
        "import logging\n",
        "import logging\n"
        "import git\n"
        "import subprocess\n\n\n"
        "def forbidden():\n"
        "    subprocess.run(['git', 'status'])\n",
    )
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)
    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))
    codes = {issue.code for issue in result.issues}
    assert {"GIT_LIBRARY_FORBIDDEN", "GIT_EXECUTABLE_FORBIDDEN"} <= codes
    assert result.commands == ()


def test_source_control_policy_resolves_aliases_and_dynamic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace(
        "import logging\n",
        "import logging\n"
        "import builtins\n"
        "import importlib as loader\n"
        "import subprocess as process\n"
        "from os import popen as launch\n"
        "runner = process.run\n"
        "dynamic_import = builtins.__import__\n",
    ).replace(
        "step_results: list[bool] = []",
        "command = ['git', 'status']\n"
        "        process.run(['git.exe', 'status'])\n"
        "        process.run(command)\n"
        "        runner(['git.cmd', 'status'])\n"
        "        launch('git status')\n"
        "        loader.import_module(command[0])\n"
        "        dynamic_import(command[0])\n"
        "        step_results: list[bool] = []",
    )
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)

    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))

    codes = {issue.code for issue in result.issues}
    assert {
        "GIT_EXECUTABLE_FORBIDDEN",
        "DYNAMIC_PROCESS_EXECUTABLE_FORBIDDEN",
        "DYNAMIC_IMPORT_FORBIDDEN",
    } <= codes
    assert result.commands == ()


def test_method_default_expression_is_rejected_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace(
        "def test_setup(self) -> bool:",
        "def test_setup(self=logging.getLogger('eager')) -> bool:",
    )
    _write_case(tmp_path, python=unsafe)
    monkeypatch.setattr(subprocess, "run", _successful_process)

    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))

    assert "CLASS_CONSTRUCTION_SIDE_EFFECT" in {issue.code for issue in result.issues}
    assert result.commands == ()


def test_domain_import_must_be_function_local_before_import_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsafe = VALID_PYTHON.replace("import logging\n", "import logging\nimport hardware_sdk\n")
    _write_case(tmp_path, python=unsafe)
    calls: list[list[str]] = []

    def should_not_run(argv: list[str], **_kwargs: Any) -> Any:
        calls.append(argv)
        raise AssertionError("unsafe module reached a subprocess")

    monkeypatch.setattr(subprocess, "run", should_not_run)

    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))

    assert "TOP_LEVEL_IMPORT_FORBIDDEN" in {issue.code for issue in result.issues}
    assert calls == []


@pytest.mark.parametrize(
    "support_source",
    [
        "def marker(function):\n"
        "    return function\n\n"
        "@marker\n"
        "def generated():\n"
        "    return True\n",
        "def generated(value=dangerous()):\n    return bool(value)\n",
        "def generated(value: dangerous()) -> bool:\n    return bool(value)\n",
        "generated = dangerous()\n",
    ],
)
def test_created_support_import_time_execution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    support_source: str,
) -> None:
    _write_case(tmp_path)
    wrapper = tmp_path / WRAPPER_PATH
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(support_source, encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", _successful_process)
    expected = dict(_request(tmp_path).expected_hashes)
    expected[WRAPPER_PATH] = _hash(wrapper)

    result = _runner(tmp_path).validate_generated_case(
        _request(
            tmp_path,
            support_files=(WRAPPER_PATH,),
            created_files=frozenset({WRAPPER_PATH}),
            expected_hashes=expected,
        )
    )

    assert "TOP_LEVEL_EXECUTION_FORBIDDEN" in {issue.code for issue in result.issues}
    assert result.commands == ()


def test_robot_may_call_only_python_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_robot = VALID_ROBOT.replace(
        "    ${result}=    Run Test\n",
        "    Setup Test\n    ${result}=    Run Test\n",
    )
    _write_case(tmp_path, robot=bad_robot)
    calls: list[list[str]] = []

    def _run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _successful_process(argv)

    monkeypatch.setattr(subprocess, "run", _run)
    result = _runner(tmp_path).validate_generated_case(_request(tmp_path))
    assert "ROBOT_COORDINATOR_ONLY" in {issue.code for issue in result.issues}
    assert len(calls) == 1  # import smoke only; unsafe Robot is never handed to Robot


def test_import_timeout_stops_before_robot_dryrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    calls = 0

    def _timeout(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    result = _runner(tmp_path, import_timeout=1).validate_generated_case(_request(tmp_path))
    assert "ISOLATED_IMPORT_FAILED" in {issue.code for issue in result.issues}
    assert result.commands[0].timed_out
    assert calls == 1


def test_traceability_exact_data_and_checksum_failures_are_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_case(tmp_path)
    monkeypatch.setattr(subprocess, "run", _successful_process)
    expected = dict(_request(tmp_path).expected_hashes)
    expected[PYTHON_PATH] = "sha256:" + "0" * 64
    result = _runner(tmp_path).validate_generated_case(
        _request(tmp_path, expected_hashes=expected),
        traceability_hook=lambda _context: ["REQ-2 missing"],
        exact_test_data_hook=lambda _context: False,
    )
    codes = {issue.code for issue in result.issues}
    assert {
        "TRACEABILITY_CHECK_FAILED",
        "EXACT_TEST_DATA_CHECK_FAILED",
        "GENERATED_CHECKSUM_MISMATCH",
    } <= codes


def test_classify_static_taxonomy() -> None:
    assert classify_static(parse_ok=False, collect_ok=False, static_ok=False) == "FAILED_VALIDATION"
    assert classify_static(parse_ok=True, collect_ok=False, static_ok=False) == "GENERATED"
    assert classify_static(parse_ok=True, collect_ok=True, static_ok=False) == "COLLECTABLE"
    assert classify_static(parse_ok=True, collect_ok=True, static_ok=True) == "STATICALLY_VALIDATED"
