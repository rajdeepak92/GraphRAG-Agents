"""Configuration loader with CLI/env/.env/config/default precedence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.huggingface_env import (
    env_bool,
    env_positive_int,
    export_huggingface_environment,
    first_huggingface_token,
)
from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    AzureOpenAISettings,
    ChromaSettings,
    ChunkingSettings,
    DiscoverySettings,
    HuggingFaceSettings,
    ModelSection,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
    TestScenarioSettings,
    UserStorySettings,
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


def _positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _optional_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
    huggingface_token = first_huggingface_token(env)
    huggingface_offline = env_bool(env.get("HUGGINGFACE_OFFLINE"), default=False)

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
                _cfg_path(root, config_data, "generated_requirements_dir", "generated"),
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
    export_huggingface_environment(token=huggingface_token, offline=huggingface_offline)

    reasoning_cfg = config_data.get("reasoning_model", {})
    embedding_cfg = config_data.get("embedding_model", {})
    reranker_cfg = config_data.get("reranker_model", {})
    postgres_cfg = config_data.get("postgres", {})
    neo4j_cfg = config_data.get("neo4j", {})
    discovery_cfg = config_data.get("discovery", {})
    configured_discovery_batch_size = _positive_int(
        discovery_cfg.get("batch_size"),
        default=1,
    )
    discovery_batch_size = env_positive_int(
        env.get("DISCOVERY_BATCH_SIZE") or env.get("HUGGINGFACE_DISCOVERY_BATCH_SIZE"),
        default=configured_discovery_batch_size,
    )
    log_llm_responses = env_bool(
        env.get("LOG_LLM_RESPONSES"),
        default=bool(discovery_cfg.get("log_llm_responses", False)),
    )

    user_story_cfg = config_data.get("user_story", {})
    user_story_max_new_tokens = _optional_positive_int(
        env.get("USER_STORY_MAX_NEW_TOKENS", user_story_cfg.get("max_new_tokens"))
    )
    user_story = UserStorySettings(
        top_k=_positive_int(env.get("USER_STORY_TOP_K", user_story_cfg.get("top_k")), default=4),
        dense_k=_positive_int(
            env.get("USER_STORY_DENSE_K", user_story_cfg.get("dense_k")), default=8
        ),
        sparse_k=_positive_int(
            env.get("USER_STORY_SPARSE_K", user_story_cfg.get("sparse_k")), default=8
        ),
        neighbor_window=_positive_int(
            env.get("USER_STORY_NEIGHBOR_WINDOW", user_story_cfg.get("neighbor_window")),
            default=1,
        ),
        max_new_tokens=user_story_max_new_tokens,
    )

    test_scenario_cfg = config_data.get("test_scenario", {})
    test_scenario_max_new_tokens = _optional_positive_int(
        env.get("TEST_SCENARIO_MAX_NEW_TOKENS", test_scenario_cfg.get("max_new_tokens"))
    )
    test_scenario = TestScenarioSettings(
        top_k=_positive_int(
            env.get("TEST_SCENARIO_TOP_K", test_scenario_cfg.get("top_k")), default=4
        ),
        dense_k=_positive_int(
            env.get("TEST_SCENARIO_DENSE_K", test_scenario_cfg.get("dense_k")), default=8
        ),
        sparse_k=_positive_int(
            env.get("TEST_SCENARIO_SPARSE_K", test_scenario_cfg.get("sparse_k")), default=8
        ),
        neighbor_window=_positive_int(
            env.get("TEST_SCENARIO_NEIGHBOR_WINDOW", test_scenario_cfg.get("neighbor_window")),
            default=1,
        ),
        max_new_tokens=test_scenario_max_new_tokens,
    )

    settings = AppSettings(
        app_env=env.get(
            "APP_ENV", config_data.get("application", {}).get("app_env", "development")
        ),
        log_level=env.get("LOG_LEVEL", config_data.get("application", {}).get("log_level", "INFO")),
        paths=paths,
        reasoning_model=ModelSection(
            provider=env.get(
                "REASONING_MODEL_PROVIDER",
                env.get("REASONING_LLM_PROVIDER", reasoning_cfg.get("provider", "huggingface")),
            ),
            model=env.get(
                "HUGGINGFACE_REASONING_MODEL",
                reasoning_cfg.get("model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ),
            deployment=env.get(
                "AZURE_OPENAI_REASONING_DEPLOYMENT", reasoning_cfg.get("deployment")
            ),
        ),
        embedding_model=ModelSection(
            provider=env.get(
                "EMBEDDING_MODEL_PROVIDER",
                env.get("EMBEDDING_PROVIDER", embedding_cfg.get("provider", "huggingface")),
            ),
            model=env.get("HUGGINGFACE_EMBEDDING_MODEL", embedding_cfg.get("model", "BAAI/bge-m3")),
            deployment=env.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", embedding_cfg.get("deployment")
            ),
        ),
        reranker_model=ModelSection(
            provider=env.get(
                "RERANKER_MODEL_PROVIDER",
                reranker_cfg.get("provider", "huggingface"),
            ),
            model=env.get(
                "HUGGINGFACE_RERANKER_MODEL",
                reranker_cfg.get("model", "BAAI/bge-reranker-base"),
            ),
        ),
        chunking=ChunkingSettings(**config_data.get("chunking", {})),
        discovery=DiscoverySettings(
            batch_size=discovery_batch_size,
            log_llm_responses=log_llm_responses,
        ),
        postgres=PostgresSettings(
            mode=env.get("POSTGRES_MODE", postgres_cfg.get("mode", "postgres")),
            dsn=env.get(
                "POSTGRES_DSN",
                postgres_cfg.get("dsn", "postgresql://marag:marag@127.0.0.1:5432/marag"),
            ),
            local_path=Path(
                postgres_cfg.get("local_path", paths.runtime_staging_dir / "postgres_records.jsonl")
            ),
        ),
        neo4j=Neo4jSettings(
            mode=env.get("NEO4J_MODE", neo4j_cfg.get("mode", "neo4j")),
            uri=env.get("NEO4J_URI", neo4j_cfg.get("uri", "bolt://127.0.0.1:7687")),
            username=env.get("NEO4J_USERNAME", neo4j_cfg.get("username", "neo4j")),
            password=env.get("NEO4J_PASSWORD", neo4j_cfg.get("password", "")),
            database=env.get("NEO4J_DATABASE", neo4j_cfg.get("database", "neo4j")),
            local_path=Path(
                neo4j_cfg.get("local_path", paths.runtime_staging_dir / "neo4j_projection.jsonl")
            ),
        ),
        chroma=ChromaSettings(**config_data.get("chroma", {})),
        user_story=user_story,
        test_scenario=test_scenario,
        azure_openai=AzureOpenAISettings(
            endpoint=env.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=env.get("AZURE_OPENAI_API_KEY", ""),
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            reasoning_deployment=env.get("AZURE_OPENAI_REASONING_DEPLOYMENT", ""),
            embedding_deployment=env.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
        ),
        huggingface=HuggingFaceSettings(
            token=huggingface_token,
            reasoning_model=env.get(
                "HUGGINGFACE_REASONING_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
            ),
            embedding_model=env.get("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=env.get("HUGGINGFACE_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            offline=huggingface_offline,
            max_new_tokens=env_positive_int(env.get("HUGGINGFACE_MAX_NEW_TOKENS"), default=4096),
            discovery_batch_size=discovery_batch_size,
            log_llm_responses=log_llm_responses,
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
