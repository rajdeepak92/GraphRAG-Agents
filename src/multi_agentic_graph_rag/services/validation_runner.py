"""Static and dry-run validation for generated Stage-4 test cases.

Generated target tests are plain Python/Robot artifacts, not test-runner test
modules.  This validator therefore never discovers or executes them through a
Python test runner.  It performs the fixed Stage-4 policy, AST, import-smoke,
Robot dry-run, traceability/data-hook, and checksum pipeline instead.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from multi_agentic_graph_rag.domain.codegen_schemas import ValidationStatus
from multi_agentic_graph_rag.services.direct_file_transaction import (
    ClassifiedPath,
    PathPolicyError,
    classify_path,
)

_COMMAND_ALLOW_LIST: dict[str, list[str]] = {
    "format_check": ["-m", "ruff", "format", "--check"],
    "lint": ["-m", "ruff", "check"],
    "type_check": ["-m", "mypy"],
}
_LEGACY_DISABLED_COMMANDS = frozenset({"collect", "unit"})
_MAX_OUTPUT_CHARS = 20_000
_DEFAULT_TIMEOUT = 120
_DEFAULT_IMPORT_TIMEOUT = 10
_FORBIDDEN_GIT_MODULES = frozenset({"git", "gitpython", "dulwich", "pygit2"})
_SAFE_TOP_LEVEL_IMPORTS = frozenset(
    {
        "__future__",
        "abc",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "itertools",
        "json",
        "logging",
        "math",
        "pathlib",
        "re",
        "sys",
        "typing",
    }
)
_PROCESS_CALL_ARGUMENTS = {
    "asyncio.create_subprocess_exec": 0,
    "asyncio.create_subprocess_shell": 0,
    "os.execl": 0,
    "os.execle": 0,
    "os.execlp": 0,
    "os.execlpe": 0,
    "os.execv": 0,
    "os.execve": 0,
    "os.execvp": 0,
    "os.execvpe": 0,
    "os.popen": 0,
    "os.spawnl": 1,
    "os.spawnle": 1,
    "os.spawnlp": 1,
    "os.spawnlpe": 1,
    "os.spawnv": 1,
    "os.spawnve": 1,
    "os.spawnvp": 1,
    "os.spawnvpe": 1,
    "os.startfile": 0,
    "os.system": 0,
    "subprocess.Popen": 0,
    "subprocess.call": 0,
    "subprocess.check_call": 0,
    "subprocess.check_output": 0,
    "subprocess.getoutput": 0,
    "subprocess.getstatusoutput": 0,
    "subprocess.run": 0,
}
_DYNAMIC_IMPORT_CALLS = frozenset({"__import__", "builtins.__import__", "importlib.import_module"})
_LIFECYCLE_METHODS = ("test_setup", "execute_test", "test_teardown", "run_test")
_ROBOT_SPLIT = re.compile(r"(?: {2,}|\t+)")


class ValidationError(ValueError):
    """A validation request itself is unsafe or malformed."""


@dataclass(frozen=True)
class ValidationRunResult:
    """One bounded subprocess outcome with capped output."""

    name: str
    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    unavailable: bool = False
    argv: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParseOutcome:
    """In-process Python parsing/compilation result."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GeneratedValidationIssue:
    """One stable, bounded diagnostic from the generated-case pipeline."""

    stage: str
    code: str
    message: str
    relative_path: str | None = None

    def render(self) -> str:
        location = f" [{self.relative_path}]" if self.relative_path else ""
        return f"{self.code}{location}: {self.message}"


@dataclass(frozen=True)
class GeneratedCaseValidationRequest:
    """All deterministic inputs needed to validate one generated case."""

    python_file: str
    robot_file: str
    support_files: tuple[str, ...] = ()
    expected_hashes: Mapping[str, str] = field(default_factory=dict)
    # Exact pre-generation contents for shared wrapper/helper modules only.
    shared_preimages: Mapping[str, str | bytes] = field(default_factory=dict)
    # Journal evidence for support modules absent before this case began.
    created_files: frozenset[str] = frozenset()
    traceability_context: Any = None
    exact_test_data_context: Any = None
    expected_step_count: int | None = None
    critical_step_count: int | None = None

    @property
    def all_files(self) -> tuple[str, ...]:
        return (self.python_file, self.robot_file, *self.support_files)


@dataclass(frozen=True)
class GeneratedValidationContext:
    """Read-only context supplied to traceability and exact-data hooks."""

    root: Path
    request: GeneratedCaseValidationRequest
    sources: Mapping[str, str]


ValidationHookResult = bool | str | Iterable[str] | None
ValidationHook = Callable[[GeneratedValidationContext], ValidationHookResult]


@dataclass(frozen=True)
class GeneratedCaseValidationResult:
    """Complete ordered validation evidence for one case."""

    ok: bool
    issues: tuple[GeneratedValidationIssue, ...] = ()
    commands: tuple[ValidationRunResult, ...] = ()

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(issue.render() for issue in self.issues)


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT_CHARS else text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _call_name(node: ast.Call) -> str:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _expression_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                aliases[bound] = alias.name if alias.asname else alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        resolved = _resolved_expression_name(value, aliases)
        if resolved not in _PROCESS_CALL_ARGUMENTS and resolved not in _DYNAMIC_IMPORT_CALLS:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = resolved
    return aliases


def _resolved_expression_name(node: ast.AST, aliases: Mapping[str, str]) -> str:
    name = _expression_name(node)
    if name:
        head, *tail = name.split(".")
        resolved = aliases.get(head, head)
        return ".".join((resolved, *tail))
    if (
        isinstance(node, ast.Call)
        and _expression_name(node.func) == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        owner = _resolved_expression_name(node.args[0], aliases)
        return f"{owner}.{node.args[1].value}" if owner else ""
    return ""


def _literal_module_root(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    return node.value.split(".", 1)[0].casefold()


def _is_git_executable(value: str) -> bool:
    name = PurePosixPath(value.strip().strip("\"'").replace("\\", "/")).name.casefold()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name == "git"


def _literal_command_mentions_git(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return any(_is_git_executable(token) for token in re.findall(r"[^\s|;&]+", node.value))
    if isinstance(node, ast.List | ast.Tuple):
        return any(_literal_command_mentions_git(item) for item in node.elts)
    return False


def _literal_command_head(node: ast.expr) -> str | None:
    candidate = node
    if isinstance(candidate, ast.Starred):
        candidate = candidate.value
    if isinstance(candidate, ast.List | ast.Tuple) and candidate.elts:
        candidate = candidate.elts[0]
    if not isinstance(candidate, ast.Constant) or not isinstance(candidate.value, str):
        return None
    first = candidate.value.strip().split(maxsplit=1)[0] if candidate.value.strip() else ""
    return PurePosixPath(first.replace("\\", "/")).name.casefold()


def _git_policy_issues(tree: ast.Module, relative_path: str) -> list[GeneratedValidationIssue]:
    issues: list[GeneratedValidationIssue] = []
    aliases = _import_aliases(tree)
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        for module_name in imported:
            if module_name.split(".", 1)[0].casefold() in _FORBIDDEN_GIT_MODULES:
                issues.append(
                    GeneratedValidationIssue(
                        "policy",
                        "GIT_LIBRARY_FORBIDDEN",
                        f"forbidden source-control library import: {module_name}",
                        relative_path,
                    )
                )
        if not isinstance(node, ast.Call):
            continue

        call_name = _resolved_expression_name(node.func, aliases)
        if call_name in _DYNAMIC_IMPORT_CALLS:
            module_root = _literal_module_root(node.args[0]) if node.args else None
            if module_root in _FORBIDDEN_GIT_MODULES:
                issues.append(
                    GeneratedValidationIssue(
                        "policy",
                        "GIT_LIBRARY_FORBIDDEN",
                        f"forbidden dynamic source-control import through {call_name}",
                        relative_path,
                    )
                )
            else:
                issues.append(
                    GeneratedValidationIssue(
                        "policy",
                        "DYNAMIC_IMPORT_FORBIDDEN",
                        f"dynamic import through {call_name} cannot prove the no-Git policy",
                        relative_path,
                    )
                )
            continue

        command_index = _PROCESS_CALL_ARGUMENTS.get(call_name)
        if command_index is None:
            continue
        command: ast.expr | None = (
            node.args[command_index] if len(node.args) > command_index else None
        )
        if command is None:
            command = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"args", "cmd", "command"}
                ),
                None,
            )
        if command is None or _literal_command_head(command) is None:
            issues.append(
                GeneratedValidationIssue(
                    "policy",
                    "DYNAMIC_PROCESS_EXECUTABLE_FORBIDDEN",
                    f"dynamic executable through {call_name} cannot prove the no-Git policy",
                    relative_path,
                )
            )
        elif _literal_command_mentions_git(command):
            issues.append(
                GeneratedValidationIssue(
                    "policy",
                    "GIT_EXECUTABLE_FORBIDDEN",
                    f"forbidden source-control executable call through {call_name}",
                    relative_path,
                )
            )
    return issues


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    names: set[str] = set()
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _additive_ast_issues(
    *, relative_path: str, original: str, current: str
) -> list[GeneratedValidationIssue]:
    try:
        before = ast.parse(original, filename=f"{relative_path}:preimage")
        after = ast.parse(current, filename=relative_path)
    except SyntaxError as exc:
        return [
            GeneratedValidationIssue(
                "additive_ast",
                "ADDITIVE_AST_PARSE_ERROR",
                f"cannot compare shared module AST: line {exc.lineno}: {exc.msg}",
                relative_path,
            )
        ]

    if len(after.body) < len(before.body):
        return [
            GeneratedValidationIssue(
                "additive_ast",
                "EXISTING_BEHAVIOR_CHANGE_FORBIDDEN",
                "pre-existing top-level statements were removed",
                relative_path,
            )
        ]
    for index, old_node in enumerate(before.body):
        if ast.dump(old_node, include_attributes=False) != ast.dump(
            after.body[index], include_attributes=False
        ):
            return [
                GeneratedValidationIssue(
                    "additive_ast",
                    "EXISTING_BEHAVIOR_CHANGE_FORBIDDEN",
                    (
                        "a pre-existing import, assignment, decorator, signature, "
                        "class, or function body changed"
                    ),
                    relative_path,
                )
            ]

    known_names: set[str] = set()
    for node in before.body:
        known_names.update(_bound_names(node))
    for node in after.body[len(before.body) :]:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return [
                GeneratedValidationIssue(
                    "additive_ast",
                    "EXISTING_BEHAVIOR_CHANGE_FORBIDDEN",
                    "shared test_lib modules may append only uniquely named functions",
                    relative_path,
                )
            ]
        if node.name in known_names:
            return [
                GeneratedValidationIssue(
                    "additive_ast",
                    "EXISTING_BEHAVIOR_CHANGE_FORBIDDEN",
                    f"appended function name is not unique: {node.name}",
                    relative_path,
                )
            ]
        known_names.add(node.name)
    return []


def _is_safe_value(node: ast.expr | None) -> bool:
    if node is None or isinstance(node, ast.Constant | ast.Name | ast.Attribute):
        return True
    if isinstance(node, ast.Dict):
        return all(
            _is_safe_value(key) and _is_safe_value(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_safe_value(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        return _is_safe_value(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_safe_value(node.left) and _is_safe_value(node.right)
    if isinstance(node, ast.Call):
        return _call_name(node) == "logging.getLogger" and all(
            _is_safe_value(argument) for argument in node.args
        )
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _subprocess_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _definition_import_time_issues(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative_path: str,
    *,
    code: str,
) -> list[GeneratedValidationIssue]:
    issues: list[GeneratedValidationIssue] = []
    if node.decorator_list:
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                code,
                f"decorators on {node.name} execute while the module is imported",
                relative_path,
            )
        )

    defaults: list[ast.expr] = [*node.args.defaults]
    defaults.extend(value for value in node.args.kw_defaults if value is not None)
    annotations: list[ast.expr] = [
        argument.annotation
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.annotation is not None
    ]
    if node.args.vararg is not None and node.args.vararg.annotation is not None:
        annotations.append(node.args.vararg.annotation)
    if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
        annotations.append(node.args.kwarg.annotation)
    if node.returns is not None:
        annotations.append(node.returns)

    executable = (
        ast.Call,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.NamedExpr,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    if any(isinstance(child, executable) for value in defaults for child in ast.walk(value)):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                code,
                f"default expressions on {node.name} may execute while importing",
                relative_path,
            )
        )
    if any(isinstance(child, executable) for value in annotations for child in ast.walk(value)):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                code,
                f"annotation expressions on {node.name} may execute while importing",
                relative_path,
            )
        )
    return issues


def _top_level_guard_issues(
    tree: ast.Module,
    relative_path: str,
    generated_class_name: str,
    *,
    require_main_guard: bool = True,
) -> list[GeneratedValidationIssue]:
    issues: list[GeneratedValidationIssue] = []
    main_guards = 0
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node):
            main_guards += 1
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            unsafe = [
                module
                for module in modules
                if module.split(".", 1)[0] not in _SAFE_TOP_LEVEL_IMPORTS
            ]
            if unsafe:
                issues.append(
                    GeneratedValidationIssue(
                        "lifecycle",
                        "TOP_LEVEL_IMPORT_FORBIDDEN",
                        "framework, test_lib, and third-party imports must be function-local: "
                        + ", ".join(unsafe),
                        relative_path,
                    )
                )
            continue
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            issues.extend(
                _definition_import_time_issues(
                    node,
                    relative_path,
                    code="TOP_LEVEL_EXECUTION_FORBIDDEN",
                )
            )
            continue
        if isinstance(node, ast.ClassDef):
            unsafe_header = bool(node.decorator_list) or any(
                isinstance(child, ast.Call)
                for expression in [*node.bases, *(keyword.value for keyword in node.keywords)]
                for child in ast.walk(expression)
            )
            if unsafe_header:
                issues.append(
                    GeneratedValidationIssue(
                        "lifecycle",
                        "CLASS_CONSTRUCTION_SIDE_EFFECT",
                        f"class {node.name} has executable bases, keywords, or decorators",
                        relative_path,
                    )
                )
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    issues.extend(
                        _definition_import_time_issues(
                            child,
                            relative_path,
                            code="CLASS_CONSTRUCTION_SIDE_EFFECT",
                        )
                    )
                    continue
                if isinstance(child, ast.Pass):
                    continue
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                    continue
                value = child.value if isinstance(child, ast.Assign | ast.AnnAssign) else None
                if isinstance(child, ast.Assign | ast.AnnAssign) and _is_safe_value(value):
                    continue
                issues.append(
                    GeneratedValidationIssue(
                        "lifecycle",
                        "CLASS_CONSTRUCTION_SIDE_EFFECT",
                        f"class {node.name} contains executable construction-time behavior",
                        relative_path,
                    )
                )
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # module docstring
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if isinstance(node, ast.Assign | ast.AnnAssign) and _is_safe_value(value):
            continue
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "TOP_LEVEL_EXECUTION_FORBIDDEN",
                "only declarations and the __main__ guard may execute at module level",
                relative_path,
            )
        )
    if require_main_guard and main_guards != 1:
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "MAIN_GUARD_REQUIRED",
                "generated module must contain exactly one __main__ guard",
                relative_path,
            )
        )
    return issues


def _is_bool_expression(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, bool)
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id in {"bool", "all", "any"}
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.Not)
    return isinstance(node, ast.BoolOp | ast.Compare)


def _self_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
    )


def _assigns_boolean(stmts: Iterable[ast.stmt], name: str, value: bool) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
        and isinstance(node.value, ast.Constant)
        and node.value.value is value
        for stmt in stmts
        for node in ast.walk(stmt)
    )


def _assigns_bool_self_call(stmts: Iterable[ast.stmt], name: str, method: str) -> bool:
    for stmt in stmts:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value
            if not isinstance(target, ast.Name) or target.id != name:
                continue
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "bool"
                and len(value.args) == 1
                and _self_call(value.args[0], method)
            ):
                return True
    return False


def _run_test_shape_ok(method: ast.FunctionDef) -> bool:
    body = list(method.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if not body or not isinstance(body[0], ast.If):
        return False
    setup_if = body[0]
    if not (
        isinstance(setup_if.test, ast.UnaryOp)
        and isinstance(setup_if.test.op, ast.Not)
        and _self_call(setup_if.test.operand, "test_setup")
        and setup_if.body
        and isinstance(setup_if.body[0], ast.Return)
        and isinstance(setup_if.body[0].value, ast.Constant)
        and setup_if.body[0].value.value is False
    ):
        return False
    try_nodes = [node for node in body[1:] if isinstance(node, ast.Try)]
    if len(try_nodes) != 1:
        return False
    coordinator = try_nodes[0]
    prefix = body[1 : body.index(coordinator)]
    if not (
        _assigns_boolean(prefix, "execution_ok", False)
        and _assigns_boolean(prefix, "teardown_ok", False)
        and _assigns_bool_self_call(coordinator.body, "execution_ok", "execute_test")
    ):
        return False
    if not coordinator.handlers or not all(
        _assigns_boolean(handler.body, "execution_ok", False) for handler in coordinator.handlers
    ):
        return False
    teardown_tries = [node for node in coordinator.finalbody if isinstance(node, ast.Try)]
    if (
        len(teardown_tries) != 1
        or not teardown_tries[0].handlers
        or not _assigns_bool_self_call(teardown_tries[0].body, "teardown_ok", "test_teardown")
        or not all(
            _assigns_boolean(handler.body, "teardown_ok", False)
            for handler in teardown_tries[0].handlers
        )
    ):
        return False
    final_return = body[-1] if body else None
    if not isinstance(final_return, ast.Return) or not isinstance(final_return.value, ast.BoolOp):
        return False
    values = final_return.value.values
    return (
        isinstance(final_return.value.op, ast.And)
        and len(values) == 2
        and isinstance(values[0], ast.Name)
        and values[0].id == "execution_ok"
        and isinstance(values[1], ast.Name)
        and values[1].id == "teardown_ok"
    )


def _execute_completion_ok(method: ast.FunctionDef) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.BoolOp):
            continue
        if not isinstance(node.value.op, ast.And) or len(node.value.values) != 2:
            continue
        first, second = node.value.values
        if not (
            isinstance(first, ast.Call)
            and isinstance(first.func, ast.Name)
            and first.func.id == "bool"
            and len(first.args) == 1
            and isinstance(first.args[0], ast.Name)
            and first.args[0].id == "step_results"
        ):
            continue
        if (
            isinstance(second, ast.Call)
            and isinstance(second.func, ast.Name)
            and second.func.id == "all"
            and len(second.args) == 1
            and isinstance(second.args[0], ast.Name)
            and second.args[0].id == "step_results"
        ):
            return True
    return False


def _main_entrypoint_ok(guard: ast.If, stem: str) -> bool:
    instance_names: set[str] = set()
    for statement in guard.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        constructor = statement.value
        if (
            isinstance(target, ast.Name)
            and isinstance(constructor, ast.Call)
            and isinstance(constructor.func, ast.Name)
            and constructor.func.id == stem
        ):
            instance_names.add(target.id)
    result_names: set[str] = set()
    for statement in guard.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        runner_call = statement.value
        if not isinstance(target, ast.Name) or not isinstance(runner_call, ast.Call):
            continue
        if not isinstance(runner_call.func, ast.Attribute) or runner_call.func.attr != "run_test":
            continue
        receiver = runner_call.func.value
        direct_constructor = (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == stem
        )
        named_instance = isinstance(receiver, ast.Name) and receiver.id in instance_names
        if direct_constructor or named_instance:
            result_names.add(target.id)
    for child in ast.walk(guard):
        exit_call: ast.Call | None = None
        if isinstance(child, ast.Call) and _call_name(child) == "sys.exit":
            exit_call = child
        elif (
            isinstance(child, ast.Raise)
            and isinstance(child.exc, ast.Call)
            and _call_name(child.exc) == "SystemExit"
        ):
            exit_call = child.exc
        if exit_call is None or len(exit_call.args) != 1:
            continue
        code = exit_call.args[0]
        if (
            isinstance(code, ast.IfExp)
            and isinstance(code.test, ast.Name)
            and code.test.id in result_names
            and isinstance(code.body, ast.Constant)
            and code.body.value == 0
            and isinstance(code.orelse, ast.Constant)
            and code.orelse.value == 1
        ):
            return True
    return False


def _constructor_is_safe(class_node: ast.ClassDef) -> bool:
    constructor = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        return True
    allowed_calls = {"logging.getLogger", "dict", "list", "set", "tuple"}
    return all(
        not isinstance(node, ast.Call) or _call_name(node) in allowed_calls
        for node in ast.walk(constructor)
    )


def _handler_returns_false(handler: ast.ExceptHandler) -> bool:
    return any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(handler)
    )


def _logger_call_count(method: ast.FunctionDef) -> int:
    return sum(
        1
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"debug", "info", "warning", "error", "exception", "critical"}
    )


def _plan_lifecycle_issues(
    methods: Mapping[str, ast.FunctionDef],
    *,
    expected_step_count: int,
    critical_step_count: int,
    path: str,
) -> list[GeneratedValidationIssue]:
    issues: list[GeneratedValidationIssue] = []
    setup = methods.get("test_setup")
    execute = methods.get("execute_test")
    teardown = methods.get("test_teardown")
    for stage_name, method in (("setup", setup), ("teardown", teardown)):
        if method is None:
            continue
        handlers = [node for node in ast.walk(method) if isinstance(node, ast.ExceptHandler)]
        if (
            _logger_call_count(method) < 2
            or not handlers
            or not all(_handler_returns_false(handler) for handler in handlers)
        ):
            issues.append(
                GeneratedValidationIssue(
                    "lifecycle",
                    "STAGE_ERROR_HANDLING_INVALID",
                    f"{stage_name} must log start/result and convert every exception to False",
                    path,
                )
            )
    if execute is None:
        return issues
    append_count = sum(
        1
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "step_results"
        and node.func.attr == "append"
    )
    normalized_count = sum(
        1
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        and node.args
        and not (isinstance(node.args[0], ast.Name) and node.args[0].id == "step_results")
    )
    handlers = [node for node in ast.walk(execute) if isinstance(node, ast.ExceptHandler)]
    critical_exits = sum(
        1
        for node in ast.walk(execute)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and any(
            isinstance(child, ast.Return)
            and isinstance(child.value, ast.Constant)
            and child.value.value is False
            for statement in node.body
            for child in ast.walk(statement)
        )
    )
    if (
        _logger_call_count(execute) < expected_step_count
        or append_count < expected_step_count
        or normalized_count < expected_step_count
        or not handlers
        or not all(_handler_returns_false(handler) for handler in handlers)
        or critical_exits != critical_step_count
    ):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "STEP_LIFECYCLE_INVALID",
                "execution must log, Boolean-normalize, and append every planned step; "
                "exceptions and critical failures must return False",
                path,
            )
        )
    return issues


def _lifecycle_issues(
    tree: ast.Module,
    classified: ClassifiedPath,
    *,
    expected_step_count: int | None = None,
    critical_step_count: int | None = None,
) -> list[GeneratedValidationIssue]:
    path = classified.relative_path
    stem = classified.stem or PurePosixPath(path).stem
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == stem]
    issues = _top_level_guard_issues(tree, path, stem)
    if len(classes) != 1:
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "EXACT_TEST_CLASS_REQUIRED",
                f"module must expose exactly one top-level class named {stem}",
                path,
            )
        )
        return issues
    class_node = classes[0]
    if not _constructor_is_safe(class_node):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "CLASS_CONSTRUCTION_SIDE_EFFECT",
                "__init__ may assign local state but must not initialize domain/hardware resources",
                path,
            )
        )

    variables = [
        node
        for node in class_node.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "test_variables"
    ]
    annotation = ast.unparse(variables[0].annotation).replace(" ", "") if variables else ""
    if len(variables) != 1 or annotation != "dict[str,object]":
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "TEST_VARIABLES_CONTRACT",
                "class must declare test_variables: dict[str, object] exactly once",
                path,
            )
        )

    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in _LIFECYCLE_METHODS
    }
    for method_name in _LIFECYCLE_METHODS:
        method = methods.get(method_name)
        if method is None:
            issues.append(
                GeneratedValidationIssue(
                    "lifecycle",
                    "LIFECYCLE_METHOD_MISSING",
                    f"missing {method_name}(self) -> bool",
                    path,
                )
            )
            continue
        arguments = method.args
        exact_self = (
            not arguments.posonlyargs
            and len(arguments.args) == 1
            and arguments.args[0].arg == "self"
            and arguments.vararg is None
            and arguments.kwarg is None
            and not arguments.kwonlyargs
            and not method.decorator_list
        )
        returns_bool = isinstance(method.returns, ast.Name) and method.returns.id == "bool"
        if not exact_self or not returns_bool:
            issues.append(
                GeneratedValidationIssue(
                    "lifecycle",
                    "LIFECYCLE_SIGNATURE_INVALID",
                    f"{method_name} must have exact signature {method_name}(self) -> bool",
                    path,
                )
            )
        invalid_returns = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Return) and not _is_bool_expression(node.value)
        ]
        if invalid_returns:
            issues.append(
                GeneratedValidationIssue(
                    "lifecycle",
                    "LIFECYCLE_NON_BOOLEAN_RETURN",
                    f"{method_name} contains a return that is not statically Boolean",
                    path,
                )
            )

    run_method = methods.get("run_test")
    if run_method is not None and not _run_test_shape_ok(run_method):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "RUN_TEST_COORDINATOR_INVALID",
                "run_test must exit on setup failure and run teardown in execution's finally block",
                path,
            )
        )
    execute_method = methods.get("execute_test")
    if execute_method is not None and not _execute_completion_ok(execute_method):
        issues.append(
            GeneratedValidationIssue(
                "lifecycle",
                "EXECUTION_AGGREGATION_INVALID",
                "normal execution must return bool(step_results) and all(step_results)",
                path,
            )
        )

    if expected_step_count is not None:
        issues.extend(
            _plan_lifecycle_issues(
                methods,
                expected_step_count=expected_step_count,
                critical_step_count=critical_step_count or 0,
                path=path,
            )
        )

    main_guards = [node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)]
    if main_guards:
        guard = main_guards[0]
        if not _main_entrypoint_ok(guard, stem):
            issues.append(
                GeneratedValidationIssue(
                    "lifecycle",
                    "MAIN_ENTRYPOINT_INVALID",
                    "__main__ must instantiate the matching class, call run_test(), and exit 0/1",
                    path,
                )
            )
    return issues


def _normalize_robot_keyword(value: str) -> str:
    return value.replace("_", " ").replace(" ", "").casefold()


def _robot_contract_issues(
    source: str,
    *,
    relative_path: str,
    module: str,
    stem: str,
    tc_id: int,
) -> list[GeneratedValidationIssue]:
    sections: dict[str, list[tuple[bool, str]]] = {}
    current: str | None = None
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("***") and stripped.endswith("***"):
            current = stripped.strip("* ").casefold()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append((bool(raw_line[:1].isspace()), stripped))

    issues: list[GeneratedValidationIssue] = []
    unexpected = set(sections) - {"settings", "test cases"}
    if unexpected:
        issues.append(
            GeneratedValidationIssue(
                "robot_contract",
                "ROBOT_SECTION_FORBIDDEN",
                f"unexpected Robot sections: {sorted(unexpected)}",
                relative_path,
            )
        )
    settings = sections.get("settings", [])
    settings_tokens = [_ROBOT_SPLIT.split(line) for _, line in settings]
    expected_library = f"tests.{module}.{stem}"
    libraries = [
        tokens for tokens in settings_tokens if tokens and tokens[0].casefold() == "library"
    ]
    if libraries != [["Library", expected_library]] and not (
        len(libraries) == 1 and len(libraries[0]) == 2 and libraries[0][1] == expected_library
    ):
        issues.append(
            GeneratedValidationIssue(
                "robot_contract",
                "ROBOT_LIBRARY_INVALID",
                f"Robot must import only matching library {expected_library}",
                relative_path,
            )
        )
    if len(settings_tokens) != 1:
        issues.append(
            GeneratedValidationIssue(
                "robot_contract",
                "ROBOT_SETTINGS_INVALID",
                "Robot Settings must contain only the matching Library import",
                relative_path,
            )
        )

    test_lines = sections.get("test cases", [])
    titles = [line for indented, line in test_lines if not indented]
    keyword_lines = [line for indented, line in test_lines if indented]
    if len(titles) != 1 or not titles[0].startswith(f"TC{tc_id} "):
        issues.append(
            GeneratedValidationIssue(
                "robot_contract",
                "ROBOT_TEST_NAME_INVALID",
                f"Robot must contain one test named with prefix 'TC{tc_id} '",
                relative_path,
            )
        )
    parsed = [_ROBOT_SPLIT.split(line) for line in keyword_lines]
    first_ok = (
        len(parsed) == 2
        and len(parsed[0]) == 2
        and parsed[0][0].casefold() == "${result}="
        and _normalize_robot_keyword(parsed[0][1]) == "runtest"
    )
    second_ok = (
        len(parsed) == 2
        and len(parsed[1]) == 2
        and _normalize_robot_keyword(parsed[1][0]) == "shouldbetrue"
        and parsed[1][1].casefold() == "${result}"
    )
    if not (first_ok and second_ok):
        issues.append(
            GeneratedValidationIssue(
                "robot_contract",
                "ROBOT_COORDINATOR_ONLY",
                "Robot must call only Run Test and then Should Be True on its result",
                relative_path,
            )
        )
    return issues


def _hook_messages(result: ValidationHookResult) -> list[str]:
    if result is None or result is True:
        return []
    if result is False:
        return ["validation hook returned False"]
    if isinstance(result, str):
        return [result]
    return [str(value) for value in result if str(value)]


class ValidationRunner:
    """Run the fixed generated-case validation pipeline inside a framework root."""

    def __init__(
        self,
        worktree_root: Path,
        *,
        python_executable: str = "python",
        timeout: int = _DEFAULT_TIMEOUT,
        import_timeout: int = _DEFAULT_IMPORT_TIMEOUT,
        robot_dryrun: bool = True,
    ) -> None:
        self.worktree_root = worktree_root.resolve()
        self.python_executable = python_executable
        self.timeout = timeout
        self.import_timeout = import_timeout
        self.robot_dryrun = robot_dryrun

    def _safe_relative(self, relative_path: str) -> str:
        if not relative_path or relative_path.startswith("-") or "\\" in relative_path:
            raise ValidationError(f"unsafe path: {relative_path}")
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValidationError(f"unsafe path: {relative_path}")
        cursor = self.worktree_root
        for part in rel.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValidationError(f"path contains a symlink component: {relative_path}")
        target = (self.worktree_root / rel).resolve()
        if target != self.worktree_root and self.worktree_root not in target.parents:
            raise ValidationError(f"path escapes framework root: {relative_path}")
        return PurePosixPath(*rel.parts).as_posix()

    def parse_python(self, paths: list[str]) -> ParseOutcome:
        """Parse and compile Python paths without importing or executing them."""
        errors: list[str] = []
        for relative_path in paths:
            safe = self._safe_relative(relative_path)
            if Path(safe).suffix != ".py":
                errors.append(f"{safe}: not a Python source file")
                continue
            file_path = self.worktree_root / safe
            if not file_path.is_file():
                errors.append(f"{safe}: file not found")
                continue
            try:
                source = file_path.read_text(encoding="utf-8")
                ast.parse(source, filename=safe)
                compile(source, safe, "exec", dont_inherit=True)
            except (SyntaxError, UnicodeError) as exc:
                line = getattr(exc, "lineno", None)
                errors.append(f"{safe}:{line or '?'}: {exc}")
        return ParseOutcome(ok=not errors, errors=tuple(errors))

    def run_command(self, name: str, paths: list[str]) -> ValidationRunResult:
        """Run only harmless generic static tools; target discovery is disabled."""
        if name in _LEGACY_DISABLED_COMMANDS:
            return ValidationRunResult(
                name=name,
                ok=False,
                returncode=None,
                stdout="",
                stderr=(
                    "generated target discovery/execution is disabled; use validate_generated_case"
                ),
                unavailable=True,
            )
        if name not in _COMMAND_ALLOW_LIST:
            raise ValidationError(f"command '{name}' is not on the allow-list")
        safe_paths = [self._safe_relative(path) for path in paths]
        if not safe_paths:
            raise ValidationError("at least one explicit path is required")
        argv = [self.python_executable, *_COMMAND_ALLOW_LIST[name], *safe_paths]
        return self._run_subprocess(name, argv, timeout=self.timeout)

    def validate_generated_case(
        self,
        request: GeneratedCaseValidationRequest,
        *,
        traceability_hook: ValidationHook | None = None,
        exact_test_data_hook: ValidationHook | None = None,
    ) -> GeneratedCaseValidationResult:
        issues: list[GeneratedValidationIssue] = []
        commands: list[ValidationRunResult] = []
        classifications: dict[str, ClassifiedPath] = {}
        sources: dict[str, str] = {}

        # 1. Exact allowlist, symlink, module, and TC/stem agreement.
        if len(set(request.all_files)) != len(request.all_files):
            issues.append(
                GeneratedValidationIssue(
                    "path_policy", "DUPLICATE_GENERATED_PATH", "file list contains duplicates"
                )
            )
        for relative_path in request.all_files:
            try:
                classified = classify_path(relative_path)
                safe = self._safe_relative(relative_path)
                target = self.worktree_root / safe
                if not target.is_file():
                    raise ValidationError("generated file is missing")
                classifications[relative_path] = classified
                if target.suffix in {".py", ".robot"}:
                    sources[relative_path] = target.read_text(encoding="utf-8")
            except (PathPolicyError, ValidationError, OSError, UnicodeError) as exc:
                issues.append(
                    GeneratedValidationIssue(
                        "path_policy", "WRITE_PATH_FORBIDDEN", str(exc), relative_path
                    )
                )
        python_classified = classifications.get(request.python_file)
        robot_classified = classifications.get(request.robot_file)
        if python_classified is not None and python_classified.role != "test":
            issues.append(
                GeneratedValidationIssue(
                    "path_policy",
                    "PYTHON_TEST_PATH_REQUIRED",
                    "python_file must use tests/<module>/Tc....py",
                    request.python_file,
                )
            )
        if robot_classified is not None and robot_classified.role != "robot":
            issues.append(
                GeneratedValidationIssue(
                    "path_policy",
                    "ROBOT_TEST_PATH_REQUIRED",
                    "robot_file must use tests_robot/<module>/Tc....robot",
                    request.robot_file,
                )
            )
        if (
            python_classified is not None
            and robot_classified is not None
            and (
                python_classified.module != robot_classified.module
                or python_classified.stem != robot_classified.stem
                or python_classified.tc_id != robot_classified.tc_id
            )
        ):
            issues.append(
                GeneratedValidationIssue(
                    "path_policy",
                    "PYTHON_ROBOT_IDENTITY_MISMATCH",
                    "Python and Robot module/stem/TC identity must match exactly",
                )
            )
        for support_path in request.support_files:
            support = classifications.get(support_path)
            if support is not None and support.role not in {"wrapper", "helper", "init"}:
                issues.append(
                    GeneratedValidationIssue(
                        "path_policy",
                        "SUPPORT_PATH_FORBIDDEN",
                        "support file is not module-specific test_lib or an allowed __init__.py",
                        support_path,
                    )
                )
            if (
                support is not None
                and python_classified is not None
                and support.module != python_classified.module
            ):
                issues.append(
                    GeneratedValidationIssue(
                        "path_policy",
                        "SUPPORT_MODULE_MISMATCH",
                        "support file module differs from the generated case module",
                        support_path,
                    )
                )
            if (
                support is not None
                and support.role in {"wrapper", "helper"}
                and support_path not in request.shared_preimages
                and support_path not in request.created_files
            ):
                issues.append(
                    GeneratedValidationIssue(
                        "additive_ast",
                        "SHARED_ORIGIN_EVIDENCE_MISSING",
                        "support module requires a preimage or journal-created evidence",
                        support_path,
                    )
                )
        unknown_created = request.created_files.difference(request.all_files)
        for relative_path in sorted(unknown_created):
            issues.append(
                GeneratedValidationIssue(
                    "additive_ast",
                    "CREATED_FILE_EVIDENCE_INVALID",
                    "journal-created evidence names a file outside this generated bundle",
                    relative_path,
                )
            )

        path_failed = any(issue.stage == "path_policy" for issue in issues)
        python_trees: dict[str, ast.Module] = {}
        if not path_failed:
            # 2. No source-control interaction policy.
            for relative_path, source in sources.items():
                if not relative_path.endswith(".py"):
                    continue
                try:
                    policy_tree = ast.parse(source, filename=relative_path)
                except SyntaxError:
                    continue
                issues.extend(_git_policy_issues(policy_tree, relative_path))

            # 3. Shared test_lib files are append-only at the AST level.
            for relative_path, preimage in request.shared_preimages.items():
                shared_classified = classifications.get(relative_path)
                if shared_classified is None or shared_classified.role not in {
                    "wrapper",
                    "helper",
                }:
                    issues.append(
                        GeneratedValidationIssue(
                            "additive_ast",
                            "SHARED_PREIMAGE_PATH_INVALID",
                            (
                                "preimages are accepted only for generated support "
                                "wrapper/helper paths"
                            ),
                            relative_path,
                        )
                    )
                    continue
                original = preimage.decode("utf-8") if isinstance(preimage, bytes) else preimage
                issues.extend(
                    _additive_ast_issues(
                        relative_path=relative_path,
                        original=original,
                        current=sources.get(relative_path, ""),
                    )
                )

            # 4. Parse and syntax compile every generated Python file.
            for relative_path, source in sources.items():
                if not relative_path.endswith(".py"):
                    continue
                try:
                    tree = ast.parse(source, filename=relative_path)
                    compile(source, relative_path, "exec", dont_inherit=True)
                    python_trees[relative_path] = tree
                except (SyntaxError, UnicodeError) as exc:
                    issues.append(
                        GeneratedValidationIssue(
                            "python_syntax", "PYTHON_SYNTAX_INVALID", str(exc), relative_path
                        )
                    )

            # 5. Import-time safety across every generated Python file, followed
            # by the primary class/lifecycle/coordinator contract.
            for relative_path, tree in python_trees.items():
                if relative_path == request.python_file:
                    continue
                guarded_nodes = list(tree.body)
                support_preimage = request.shared_preimages.get(relative_path)
                if support_preimage is not None:
                    original = (
                        support_preimage.decode("utf-8")
                        if isinstance(support_preimage, bytes)
                        else support_preimage
                    )
                    try:
                        before = ast.parse(original, filename=f"{relative_path}:preimage")
                    except SyntaxError:
                        before = None
                    if before is not None:
                        guarded_nodes = guarded_nodes[len(before.body) :]
                issues.extend(
                    _top_level_guard_issues(
                        ast.Module(body=guarded_nodes, type_ignores=[]),
                        relative_path,
                        "",
                        require_main_guard=False,
                    )
                )

            main_tree = python_trees.get(request.python_file)
            if main_tree is not None and python_classified is not None:
                issues.extend(
                    _lifecycle_issues(
                        main_tree,
                        python_classified,
                        expected_step_count=request.expected_step_count,
                        critical_step_count=request.critical_step_count,
                    )
                )

            unsafe_python = any(
                issue.stage in {"policy", "additive_ast", "python_syntax", "lifecycle"}
                for issue in issues
            )
            import_ok = not unsafe_python
            if import_ok and python_classified is not None:
                # 6. Isolated import and side-effect-free class construction smoke check.
                import_result = self._import_smoke(
                    request.python_file, python_classified.stem or ""
                )
                commands.append(import_result)
                if not import_result.ok:
                    import_ok = False
                    issues.append(
                        GeneratedValidationIssue(
                            "import_smoke",
                            "ISOLATED_IMPORT_FAILED",
                            import_result.stderr or import_result.stdout or "import failed",
                            request.python_file,
                        )
                    )

            # 7. Robot caller contract, followed by exact dry-run invocation.
            robot_source = sources.get(request.robot_file)
            if robot_source is not None and python_classified is not None:
                robot_issues = _robot_contract_issues(
                    robot_source,
                    relative_path=request.robot_file,
                    module=python_classified.module,
                    stem=python_classified.stem or "",
                    tc_id=python_classified.tc_id or 0,
                )
                issues.extend(robot_issues)
                if not robot_issues and import_ok and self.robot_dryrun:
                    robot_result = self._robot_dryrun(request.robot_file)
                    commands.append(robot_result)
                    if not robot_result.ok:
                        issues.append(
                            GeneratedValidationIssue(
                                "robot_dryrun",
                                "ROBOT_DRYRUN_FAILED",
                                robot_result.stderr
                                or robot_result.stdout
                                or "Robot dry-run failed",
                                request.robot_file,
                            )
                        )

        # 8. Caller-provided deterministic traceability and exact-data checks.
        context = GeneratedValidationContext(self.worktree_root, request, sources)
        for stage, code, hook in (
            ("traceability", "TRACEABILITY_CHECK_FAILED", traceability_hook),
            ("exact_test_data", "EXACT_TEST_DATA_CHECK_FAILED", exact_test_data_hook),
        ):
            if hook is None:
                continue
            try:
                messages = _hook_messages(hook(context))
            except (
                Exception
            ) as exc:  # hooks are validation evidence, never allowed to crash the run
                messages = [f"hook raised {type(exc).__name__}: {exc}"]
            issues.extend(GeneratedValidationIssue(stage, code, message) for message in messages)

        # 9. Final byte-for-byte checksum verification for every generated file.
        for relative_path in request.all_files:
            expected = request.expected_hashes.get(relative_path)
            if expected is None:
                issues.append(
                    GeneratedValidationIssue(
                        "checksums",
                        "GENERATED_CHECKSUM_MISSING",
                        "no expected SHA-256 was supplied",
                        relative_path,
                    )
                )
                continue
            try:
                actual = _sha256(
                    (self.worktree_root / self._safe_relative(relative_path)).read_bytes()
                )
            except (OSError, ValidationError) as exc:
                issues.append(
                    GeneratedValidationIssue(
                        "checksums", "GENERATED_CHECKSUM_UNREADABLE", str(exc), relative_path
                    )
                )
                continue
            if actual != expected:
                issues.append(
                    GeneratedValidationIssue(
                        "checksums",
                        "GENERATED_CHECKSUM_MISMATCH",
                        f"{actual} != {expected}",
                        relative_path,
                    )
                )

        return GeneratedCaseValidationResult(
            ok=not issues,
            issues=tuple(issues),
            commands=tuple(commands),
        )

    def _import_smoke(self, relative_path: str, class_name: str) -> ValidationRunResult:
        script = (
            "import importlib.util, pathlib, sys; "
            "root=pathlib.Path(sys.argv[1]); path=root / sys.argv[2]; "
            "sys.path.insert(0, str(root)); "
            "spec=importlib.util.spec_from_file_location('_stage4_generated_case', path); "
            "module=importlib.util.module_from_spec(spec); "
            "assert spec.loader is not None; spec.loader.exec_module(module); "
            "getattr(module, sys.argv[3])()"
        )
        argv = [
            self.python_executable,
            "-B",
            "-I",
            "-c",
            script,
            str(self.worktree_root),
            relative_path,
            class_name,
        ]
        return self._run_subprocess("import_smoke", argv, timeout=self.import_timeout)

    def _robot_dryrun(self, relative_path: str) -> ValidationRunResult:
        argv = [
            self.python_executable,
            "-m",
            "robot",
            "--dryrun",
            "--pythonpath",
            str(self.worktree_root),
            "--output",
            "NONE",
            "--report",
            "NONE",
            "--log",
            "NONE",
            relative_path,
        ]
        return self._run_subprocess("robot_dryrun", argv, timeout=self.timeout)

    def _run_subprocess(self, name: str, argv: list[str], *, timeout: int) -> ValidationRunResult:
        try:
            completed = subprocess.run(
                argv,
                cwd=self.worktree_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
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
                argv=tuple(argv),
            )
        except subprocess.TimeoutExpired as exc:
            return ValidationRunResult(
                name=name,
                ok=False,
                returncode=None,
                stdout=_truncate(_subprocess_text(exc.stdout)),
                stderr="timed out",
                timed_out=True,
                argv=tuple(argv),
            )
        return ValidationRunResult(
            name=name,
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            stdout=_truncate(completed.stdout),
            stderr=_truncate(completed.stderr),
            argv=tuple(argv),
        )

    def _sandboxed_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "PATHEXT", "TEMP", "TMP", "HOME", "LANG"}
        }
        env["NO_NETWORK"] = "1"
        env["PIP_NO_INDEX"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env


def classify_static(
    *,
    parse_ok: bool,
    collect_ok: bool,
    static_ok: bool,
) -> ValidationStatus:
    """Legacy status mapping retained for non-target callers during migration."""
    if not parse_ok:
        return "FAILED_VALIDATION"
    if not collect_ok:
        return "GENERATED"
    if not static_ok:
        return "COLLECTABLE"
    return "STATICALLY_VALIDATED"


__all__ = [
    "GeneratedCaseValidationRequest",
    "GeneratedCaseValidationResult",
    "GeneratedValidationContext",
    "GeneratedValidationIssue",
    "ParseOutcome",
    "ValidationError",
    "ValidationHook",
    "ValidationRunResult",
    "ValidationRunner",
    "classify_static",
]
