"""Configuration loader with CLI/env/.env/config/default precedence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    AzureOpenAISettings,
    ChromaSettings,
    ChunkingSettings,
    HuggingFaceSettings,
    ModelSection,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _cfg_path(root: Path, data: dict[str, Any], key: str, default: str) -> Path:
    raw = data.get("paths", {}).get(key, default)
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppSettings:
    root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    config_file = config_path or root / "config.json"
    config_data: dict[str, Any] = {}
    if config_file.exists():
        config_data = json.loads(config_file.read_text(encoding="utf-8"))

    env_file_values = _read_dotenv(root / ".env")
    env = {**env_file_values, **os.environ}

    if cli_overrides:
        config_data = _deep_update(config_data, cli_overrides)

    paths = PathsSettings(
        project_root=root,
        global_cache_dir=Path(
            env.get(
                "GLOBAL_CACHE_DIR",
                _cfg_path(root, config_data, "global_cache_dir", ".global_cache"),
            )
        ),
        documents_inbox_dir=Path(
            env.get(
                "DOCUMENTS_INBOX_DIR",
                _cfg_path(root, config_data, "documents_inbox_dir", "documents/inbox"),
            )
        ),
        generated_requirements_dir=Path(
            env.get(
                "GENERATED_REQUIREMENTS_DIR",
                _cfg_path(root, config_data, "generated_requirements_dir", "generated/inbox"),
            )
        ),
        chroma_persist_dir=Path(
            env.get(
                "CHROMA_PERSIST_DIR",
                _cfg_path(root, config_data, "chroma_persist_dir", "runtime/databases/chroma"),
            )
        ),
        runtime_staging_dir=Path(
            env.get(
                "RUNTIME_STAGING_DIR",
                _cfg_path(root, config_data, "runtime_staging_dir", "runtime/staging"),
            )
        ),
        runtime_logs_dir=Path(
            env.get(
                "RUNTIME_LOGS_DIR", _cfg_path(root, config_data, "runtime_logs_dir", "runtime/logs")
            )
        ),
        runtime_locks_dir=Path(
            env.get(
                "RUNTIME_LOCKS_DIR",
                _cfg_path(root, config_data, "runtime_locks_dir", "runtime/locks"),
            )
        ),
    )

    for cache_var in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "UV_CACHE_DIR"):
        os.environ.setdefault(cache_var, str(paths.global_cache_dir / cache_var.lower()))

    reasoning_cfg = config_data.get("reasoning_model", {})
    embedding_cfg = config_data.get("embedding_model", {})
    reranker_cfg = config_data.get("reranker_model", {})
    postgres_cfg = config_data.get("postgres", {})
    neo4j_cfg = config_data.get("neo4j", {})

    settings = AppSettings(
        app_env=env.get(
            "APP_ENV", config_data.get("application", {}).get("app_env", "development")
        ),
        log_level=env.get("LOG_LEVEL", config_data.get("application", {}).get("log_level", "INFO")),
        paths=paths,
        reasoning_model=ModelSection(
            provider=env.get(
                "REASONING_MODEL_PROVIDER",
                env.get("REASONING_LLM_PROVIDER", reasoning_cfg.get("provider", "local_heuristic")),
            ),
            model=reasoning_cfg.get("model"),
            deployment=env.get(
                "AZURE_OPENAI_REASONING_DEPLOYMENT", reasoning_cfg.get("deployment")
            ),
        ),
        embedding_model=ModelSection(
            provider=env.get(
                "EMBEDDING_MODEL_PROVIDER",
                env.get("EMBEDDING_PROVIDER", embedding_cfg.get("provider", "local_hash")),
            ),
            model=env.get("HUGGINGFACE_EMBEDDING_MODEL", embedding_cfg.get("model", "hash-384")),
            deployment=env.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", embedding_cfg.get("deployment")
            ),
        ),
        reranker_model=ModelSection(
            provider=env.get("RERANKER_MODEL_PROVIDER", reranker_cfg.get("provider", "none")),
            model=env.get("HUGGINGFACE_RERANKER_MODEL", reranker_cfg.get("model")),
        ),
        chunking=ChunkingSettings(**config_data.get("chunking", {})),
        postgres=PostgresSettings(
            mode=env.get("POSTGRES_MODE", postgres_cfg.get("mode", "local_json")),
            dsn=env.get(
                "POSTGRES_DSN",
                postgres_cfg.get("dsn", "postgresql://marag:marag@127.0.0.1:5432/marag"),
            ),
            local_path=Path(
                postgres_cfg.get("local_path", paths.runtime_staging_dir / "postgres_records.jsonl")
            ),
        ),
        neo4j=Neo4jSettings(
            mode=env.get("NEO4J_MODE", neo4j_cfg.get("mode", "local_json")),
            uri=env.get("NEO4J_URI", neo4j_cfg.get("uri", "bolt://127.0.0.1:7687")),
            username=env.get("NEO4J_USERNAME", neo4j_cfg.get("username", "neo4j")),
            password=env.get("NEO4J_PASSWORD", neo4j_cfg.get("password", "")),
            database=env.get("NEO4J_DATABASE", neo4j_cfg.get("database", "neo4j")),
            local_path=Path(
                neo4j_cfg.get("local_path", paths.runtime_staging_dir / "neo4j_projection.jsonl")
            ),
        ),
        chroma=ChromaSettings(**config_data.get("chroma", {})),
        azure_openai=AzureOpenAISettings(
            endpoint=env.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            reasoning_deployment=env.get("AZURE_OPENAI_REASONING_DEPLOYMENT", ""),
            embedding_deployment=env.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        ),
        huggingface=HuggingFaceSettings(
            token=env.get("HUGGINGFACE_TOKEN", ""),
            reasoning_model=env.get("HUGGINGFACE_REASONING_MODEL", ""),
            embedding_model=env.get(
                "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker_model=env.get(
                "HUGGINGFACE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            offline=env.get("HUGGINGFACE_OFFLINE", "false").lower() == "true",
        ),
        raw_config=config_data,
    )

    ensure_runtime_dirs(settings)
    return settings


def ensure_runtime_dirs(settings: AppSettings) -> None:
    dirs = [
        settings.paths.global_cache_dir,
        settings.paths.documents_inbox_dir,
        settings.paths.generated_requirements_dir,
        settings.paths.chroma_persist_dir,
        settings.paths.runtime_staging_dir,
        settings.paths.runtime_logs_dir,
        settings.paths.runtime_locks_dir,
        settings.postgres.local_path.parent,
        settings.neo4j.local_path.parent,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
