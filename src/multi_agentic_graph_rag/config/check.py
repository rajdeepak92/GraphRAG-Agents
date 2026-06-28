"""Configuration diagnostic helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import SecretStr

from multi_agentic_graph_rag.config.settings import Settings


def runtime_directory_status(settings: Settings) -> dict[str, bool]:
    """Return existence status for approved runtime directories."""

    return {
        "global_cache_dir": settings.paths.global_cache_dir.exists(),
        "documents_inbox_dir": settings.paths.documents_inbox_dir.exists(),
        "generated_artifacts_dir": settings.paths.generated_artifacts_dir.exists(),
        "chroma_persist_dir": settings.paths.chroma_persist_dir.exists(),
        "runtime_staging_dir": settings.paths.runtime_staging_dir.exists(),
        "runtime_logs_dir": settings.paths.runtime_logs_dir.exists(),
        "runtime_locks_dir": settings.paths.runtime_locks_dir.exists(),
    }


def settings_to_redacted_dict(settings: Settings) -> dict[str, Any]:
    """Convert settings to a serializable dictionary with secrets redacted."""

    return cast(dict[str, Any], _redact(settings.model_dump(mode="python")))


def _redact(value: object) -> Any:
    if isinstance(value, SecretStr):
        return "********"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _redact(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_redact(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)

    return value
