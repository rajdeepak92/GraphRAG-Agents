"""MCP resource registration for MARAG state and artifacts."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.mcp import service_manager
from multi_agentic_graph_rag.mcp.artifact_index import find_latest_artifacts
from multi_agentic_graph_rag.mcp.service_manager import read_project_env
from multi_agentic_graph_rag.observability.session import find_run_jsonl

ProjectRootFactory = Callable[[], Path]

SECRET_KEYS = {
    "POSTGRES_DSN",
    "NEO4J_PASSWORD",
    "AZURE_OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
}


def register_resources(mcp: Any, *, project_root: ProjectRootFactory) -> None:
    @mcp.resource("marag://config/current")
    def current_config() -> str:
        return json.dumps(sanitized_config(project_root()), indent=2)

    @mcp.resource("marag://health/latest")
    def latest_health() -> str:
        latest_report = service_manager.LAST_HEALTH_REPORT
        report = latest_report.model_dump(mode="json") if latest_report else None
        return json.dumps({"health": report}, indent=2)

    @mcp.resource("marag://artifacts/{project}/latest/requirements")
    def latest_requirements(project: str) -> str:
        return _latest_artifact_payload(project_root(), project, "requirements")

    @mcp.resource("marag://artifacts/{project}/latest/user-stories")
    def latest_user_stories(project: str) -> str:
        return _latest_artifact_payload(project_root(), project, "user_stories")

    @mcp.resource("marag://artifacts/{project}/latest/test-scenarios")
    def latest_test_scenarios(project: str) -> str:
        return _latest_artifact_payload(project_root(), project, "test_scenarios")

    @mcp.resource("marag://logs/{run_id}")
    def run_logs(run_id: str) -> str:
        return json.dumps(find_run_logs(project_root(), run_id), indent=2)


def sanitized_config(project_root: Path) -> dict[str, Any]:
    previous = os.environ.get("PROJECT_ROOT")
    os.environ["PROJECT_ROOT"] = str(project_root.resolve())
    try:
        settings = load_config()
    finally:
        if previous is None:
            os.environ.pop("PROJECT_ROOT", None)
        else:
            os.environ["PROJECT_ROOT"] = previous

    env = read_project_env(project_root)
    mcp_env_keys = sorted(
        key for key in env if key.startswith("MARAG_") or key.startswith("NEO4J_")
    )
    return {
        "project_root": str(settings.paths.project_root),
        "providers": {
            "reasoning": settings.reasoning_model.provider,
            "embedding": settings.embedding_model.provider,
            "reranker": settings.reranker_model.provider,
        },
        "postgres": {
            "mode": settings.postgres.mode,
            "dsn": _mask_dsn(settings.postgres.dsn),
        },
        "neo4j": {
            "mode": settings.neo4j.mode,
            "uri": settings.neo4j.uri,
            "username": settings.neo4j.username,
            "password": _mask_secret(settings.neo4j.password),
            "database": settings.neo4j.database,
        },
        "paths": {
            "chroma_persist_dir": str(settings.paths.chroma_persist_dir),
            "runtime_staging_dir": str(settings.paths.runtime_staging_dir),
            "runtime_logs_dir": str(settings.paths.runtime_logs_dir),
            "runtime_locks_dir": str(settings.paths.runtime_locks_dir),
        },
        "mcp_env": {key: _mask_env_value(key, env[key]) for key in mcp_env_keys},
    }


def find_run_logs(project_root: Path, run_id: str) -> dict[str, Any]:
    jsonl_path = find_run_jsonl(project_root, run_id)
    if jsonl_path is None:
        return {"run_id": run_id, "found": False}
    log_path = jsonl_path.with_suffix(".log")
    return {
        "run_id": run_id,
        "found": True,
        "jsonl_path": str(jsonl_path),
        "jsonl": _read_text_cap(jsonl_path),
        "log_path": str(log_path) if log_path.exists() else None,
        "log": _read_text_cap(log_path) if log_path.exists() else None,
    }


def _latest_artifact_payload(project_root: Path, project: str, key: str) -> str:
    lookup = find_latest_artifacts(project_root, project)
    path_value = getattr(lookup, key)
    if path_value is None:
        return json.dumps({"project": project, "artifact": key, "found": False}, indent=2)
    path = Path(path_value)
    return path.read_text(encoding="utf-8")


def _mask_env_value(key: str, value: str) -> str:
    if key == "POSTGRES_DSN":
        return _mask_dsn(value)
    if key in SECRET_KEYS:
        return _mask_secret(value)
    return value


def _mask_secret(value: str) -> str:
    return "***" if value else ""


def _mask_dsn(value: str) -> str:
    if not value:
        return ""
    parts = urlsplit(value)
    if not parts.password:
        return value
    username = parts.username or ""
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _read_text_cap(path: Path, *, limit: int = 200_000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]
