"""Stage-4 provider isolation and zero-Git boundary proofs."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from multi_agentic_graph_rag.llm_models import factory
from multi_agentic_graph_rag.services.framework_indexer import (
    GraphifyCommandError,
    detect_graphify_capabilities,
)


class _SelectedProviderSettings:
    def __init__(self, *, selected: str) -> None:
        self.selected = selected
        self._azure = SimpleNamespace(
            endpoint="https://example.openai.azure.com",
            api_key="secret",
            reasoning_deployment="deployment",
            log_llm_responses=False,
        )
        self._huggingface = SimpleNamespace(reasoning_model="private/model")

    @property
    def azure_openai(self) -> Any:
        if self.selected != "azure_openai":
            raise AssertionError("Hugging Face selection inspected Azure configuration")
        return self._azure

    @property
    def huggingface(self) -> Any:
        if self.selected != "huggingface":
            raise AssertionError("Azure selection inspected Hugging Face configuration")
        return self._huggingface


@pytest.mark.parametrize("provider", ["azure_openai", "huggingface"])
def test_stage4_factory_reads_and_initializes_only_selected_provider(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[str] = []
    dependency_checks: list[str] = []

    def azure_constructor(*_args: Any, **_kwargs: Any) -> object:
        if provider != "azure_openai":
            raise AssertionError("Hugging Face selection initialized Azure")
        constructed.append("azure_openai")
        return object()

    def huggingface_constructor(*_args: Any, **_kwargs: Any) -> object:
        if provider != "huggingface":
            raise AssertionError("Azure selection initialized Hugging Face")
        constructed.append("huggingface")
        return object()

    def find_selected_dependency(name: str) -> object:
        dependency_checks.append(name)
        return object()

    monkeypatch.setattr(factory, "find_spec", find_selected_dependency)
    monkeypatch.setattr(factory, "AzureOpenAIReasoningModel", azure_constructor)
    monkeypatch.setattr(factory, "HuggingFaceReasoningModel", huggingface_constructor)

    result = factory.create_stage4_reasoning_model(
        _SelectedProviderSettings(selected=provider),  # type: ignore[arg-type]
        provider=provider,
    )

    assert result is not None
    assert constructed == [provider]
    assert dependency_checks == ["openai" if provider == "azure_openai" else "transformers"]


@pytest.mark.parametrize("provider", ["azure_openai", "huggingface"])
def test_stage4_factory_never_cross_provider_falls_back_after_selected_failure(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls: list[str] = []

    def azure_constructor(*_args: Any, **_kwargs: Any) -> object:
        constructor_calls.append("azure_openai")
        if provider != "azure_openai":
            raise AssertionError("Hugging Face failure fell back to Azure")
        raise RuntimeError("selected Azure constructor failed")

    def huggingface_constructor(*_args: Any, **_kwargs: Any) -> object:
        constructor_calls.append("huggingface")
        if provider != "huggingface":
            raise AssertionError("Azure failure fell back to Hugging Face")
        raise RuntimeError("selected Hugging Face constructor failed")

    monkeypatch.setattr(factory, "find_spec", lambda _name: object())
    monkeypatch.setattr(factory, "AzureOpenAIReasoningModel", azure_constructor)
    monkeypatch.setattr(factory, "HuggingFaceReasoningModel", huggingface_constructor)

    with pytest.raises(RuntimeError, match="selected"):
        factory.create_stage4_reasoning_model(
            _SelectedProviderSettings(selected=provider),  # type: ignore[arg-type]
            provider=provider,
        )

    assert constructor_calls == [provider]


_STAGE4_SOURCE_FILES = (
    "workflows/codegen_run_graph.py",
    "workflows/codegen_apply_graph.py",
    "services/direct_file_transaction.py",
    "services/framework_indexer.py",
    "services/framework_snapshot.py",
    "services/validation_runner.py",
    "services/codegen_context_retriever.py",
    "services/scenario_action_mapper.py",
    "services/llm_patch_producer.py",
    "db/codegen_postgres.py",
    "db/code_graph_store.py",
)
_FORBIDDEN_IMPORT_ROOTS = {"git", "gitpython", "dulwich", "pygit2"}
_FORBIDDEN_STAGE4_NAMES = {"WorktreeManager", "RepoWriter"}
_SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output"}


def _stage4_source_root() -> Path:
    return Path("src/multi_agentic_graph_rag")


def _literal_executable(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value).name.lower()
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        return _literal_executable(node.elts[0])
    return None


def test_active_stage4_source_has_no_git_library_or_legacy_writer_dependency() -> None:
    violations: list[str] = []
    for relative_path in _STAGE4_SOURCE_FILES:
        path = _stage4_source_root() / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0].lower() in _FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{relative_path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0].lower() in _FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"{relative_path}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_STAGE4_NAMES:
                violations.append(f"{relative_path}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_STAGE4_NAMES:
                violations.append(f"{relative_path}:{node.lineno}: .{node.attr}")
    assert violations == []


def test_active_stage4_subprocess_calls_never_name_git_executable() -> None:
    violations: list[str] = []
    for relative_path in _STAGE4_SOURCE_FILES:
        path = _stage4_source_root() / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr not in _SUBPROCESS_CALLS:
                continue
            if _literal_executable(node.args[0]) == "git":
                violations.append(f"{relative_path}:{node.lineno}")
    assert violations == []


def test_graphify_capability_probe_launches_no_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def guarded_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        executable = Path(command[0]).name.lower()
        if executable in {"git", "git.exe"}:
            raise AssertionError(f"Stage 4 launched forbidden executable: {command!r}")
        commands.append(command)
        stdout = "usage: graphify update [--no-cluster] path" if "--help" in command else "1.0"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "multi_agentic_graph_rag.services.framework_indexer.subprocess.run",
        guarded_run,
    )

    capabilities = detect_graphify_capabilities("graphify")

    assert capabilities.supports_no_cluster is True
    assert commands == [
        ["graphify", "--version"],
        ["graphify", "update", "--help"],
    ]


@pytest.mark.parametrize("command", ["git", "GIT.EXE", "C:/tools/Git.exe"])
def test_graphify_command_cannot_be_reconfigured_to_launch_git(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("subprocess must not run for a forbidden executable")

    monkeypatch.setattr(
        "multi_agentic_graph_rag.services.framework_indexer.subprocess.run",
        fail_if_called,
    )

    with pytest.raises(GraphifyCommandError, match="forbids launching"):
        detect_graphify_capabilities(command)
