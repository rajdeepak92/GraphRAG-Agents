"""Configuration loader with CLI/env/.env/config/default precedence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from multi_agentic_graph_rag.common_defs import EnvVar, ModeName, PathDef, ProviderName
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
    KnowledgeGraphSettings,
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


def _float(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hfil_checkpointer(value: Any) -> Literal["postgres", "memory"]:
    text = str(value or "postgres").strip().lower()
    return "memory" if text == "memory" else "postgres"


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
    huggingface_offline = env_bool(env.get(EnvVar.HUGGINGFACE_OFFLINE.value), default=False)

    if cli_overrides:
        config_data = _deep_update(config_data, cli_overrides)

    paths = PathsSettings(
        project_root=root,
        global_cache_dir=Path(
            env.get(
                EnvVar.GLOBAL_CACHE_DIR.value,
                _cfg_path(root, config_data, "global_cache_dir", PathDef.GLOBAL_CACHE_DIR.value),
            )
        ),
        documents_inbox_dir=Path(
            env.get(
                EnvVar.DOCUMENTS_INBOX_DIR.value,
                _cfg_path(
                    root,
                    config_data,
                    "documents_inbox_dir",
                    PathDef.DOCUMENTS_INBOX_DIR.value,
                ),
            )
        ),
        generated_requirements_dir=Path(
            env.get(
                EnvVar.GENERATED_REQUIREMENTS_DIR.value,
                _cfg_path(
                    root,
                    config_data,
                    "generated_requirements_dir",
                    PathDef.GENERATED_REQUIREMENTS_DIR.value,
                ),
            )
        ),
        chroma_persist_dir=Path(
            env.get(
                EnvVar.CHROMA_PERSIST_DIR.value,
                _cfg_path(
                    root,
                    config_data,
                    "chroma_persist_dir",
                    PathDef.CHROMA_DB_PATH.value,
                ),
            )
        ),
        runtime_staging_dir=Path(
            env.get(
                EnvVar.RUNTIME_STAGING_DIR.value,
                _cfg_path(
                    root,
                    config_data,
                    "runtime_staging_dir",
                    PathDef.RUNTIME_STAGING_DIR.value,
                ),
            )
        ),
        runtime_logs_dir=Path(
            env.get(
                EnvVar.RUNTIME_LOGS_DIR.value,
                _cfg_path(root, config_data, "runtime_logs_dir", PathDef.RUNTIME_LOGS_DIR.value),
            )
        ),
        runtime_locks_dir=Path(
            env.get(
                EnvVar.RUNTIME_LOCKS_DIR.value,
                _cfg_path(root, config_data, "runtime_locks_dir", PathDef.RUNTIME_LOCKS_DIR.value),
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
        env.get(EnvVar.DISCOVERY_BATCH_SIZE.value)
        or env.get(EnvVar.HUGGINGFACE_DISCOVERY_BATCH_SIZE.value),
        default=configured_discovery_batch_size,
    )
    log_llm_responses = env_bool(
        env.get(EnvVar.LOG_LLM_RESPONSES.value),
        default=bool(discovery_cfg.get("log_llm_responses", False)),
    )

    user_story_cfg = config_data.get("user_story", {})
    user_story_max_new_tokens = _optional_positive_int(
        env.get(EnvVar.USER_STORY_MAX_NEW_TOKENS.value, user_story_cfg.get("max_new_tokens"))
    )
    user_story = UserStorySettings(
        top_k=_positive_int(
            env.get(EnvVar.USER_STORY_TOP_K.value, user_story_cfg.get("top_k")), default=4
        ),
        dense_k=_positive_int(
            env.get(EnvVar.USER_STORY_DENSE_K.value, user_story_cfg.get("dense_k")), default=8
        ),
        sparse_k=_positive_int(
            env.get(EnvVar.USER_STORY_SPARSE_K.value, user_story_cfg.get("sparse_k")), default=8
        ),
        neighbor_window=_positive_int(
            env.get(EnvVar.USER_STORY_NEIGHBOR_WINDOW.value, user_story_cfg.get("neighbor_window")),
            default=1,
        ),
        max_new_tokens=user_story_max_new_tokens,
    )

    test_scenario_cfg = config_data.get("test_scenario", {})
    test_scenario_max_new_tokens = _optional_positive_int(
        env.get(
            EnvVar.TEST_SCENARIO_MAX_NEW_TOKENS.value,
            test_scenario_cfg.get("max_new_tokens"),
        )
    )
    test_scenario = TestScenarioSettings(
        top_k=_positive_int(
            env.get(EnvVar.TEST_SCENARIO_TOP_K.value, test_scenario_cfg.get("top_k")),
            default=4,
        ),
        dense_k=_positive_int(
            env.get(EnvVar.TEST_SCENARIO_DENSE_K.value, test_scenario_cfg.get("dense_k")),
            default=8,
        ),
        sparse_k=_positive_int(
            env.get(EnvVar.TEST_SCENARIO_SPARSE_K.value, test_scenario_cfg.get("sparse_k")),
            default=8,
        ),
        neighbor_window=_positive_int(
            env.get(
                EnvVar.TEST_SCENARIO_NEIGHBOR_WINDOW.value,
                test_scenario_cfg.get("neighbor_window"),
            ),
            default=1,
        ),
        max_new_tokens=test_scenario_max_new_tokens,
    )
    hfil_cfg = config_data.get("hfil", {})

    knowledge_graph_cfg = config_data.get("knowledge_graph", {})
    knowledge_graph = KnowledgeGraphSettings(
        enabled=env_bool(
            env.get(EnvVar.KNOWLEDGE_GRAPH_ENABLED.value),
            default=bool(knowledge_graph_cfg.get("enabled", False)),
        ),
        shadow_mode=env_bool(
            env.get(EnvVar.KNOWLEDGE_GRAPH_SHADOW_MODE.value),
            default=bool(knowledge_graph_cfg.get("shadow_mode", True)),
        ),
        graph_primary_story=env_bool(
            env.get(EnvVar.GRAPH_PRIMARY_STORY.value),
            default=bool(knowledge_graph_cfg.get("graph_primary_story", False)),
        ),
        graph_primary_scenario=env_bool(
            env.get(EnvVar.GRAPH_PRIMARY_SCENARIO.value),
            default=bool(knowledge_graph_cfg.get("graph_primary_scenario", False)),
        ),
        expansion_k=_positive_int(
            env.get(
                EnvVar.KNOWLEDGE_GRAPH_EXPANSION_K.value,
                knowledge_graph_cfg.get("expansion_k"),
            ),
            default=6,
        ),
    )

    settings = AppSettings(
        app_env=env.get(
            EnvVar.APP_ENV.value, config_data.get("application", {}).get("app_env", "development")
        ),
        log_level=env.get(
            EnvVar.LOG_LEVEL.value, config_data.get("application", {}).get("log_level", "INFO")
        ),
        paths=paths,
        reasoning_model=ModelSection(
            provider=env.get(
                EnvVar.REASONING_MODEL_PROVIDER.value,
                env.get(
                    EnvVar.REASONING_LLM_PROVIDER.value,
                    reasoning_cfg.get("provider", ProviderName.HUGGINGFACE.value),
                ),
            ),
            model=env.get(
                EnvVar.HUGGINGFACE_REASONING_MODEL.value,
                reasoning_cfg.get("model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ),
            deployment=env.get(
                EnvVar.AZURE_OPENAI_REASONING_DEPLOYMENT.value, reasoning_cfg.get("deployment")
            ),
        ),
        embedding_model=ModelSection(
            provider=env.get(
                EnvVar.EMBEDDING_MODEL_PROVIDER.value,
                env.get(
                    EnvVar.EMBEDDING_PROVIDER.value,
                    embedding_cfg.get("provider", ProviderName.HUGGINGFACE.value),
                ),
            ),
            model=env.get(
                EnvVar.HUGGINGFACE_EMBEDDING_MODEL.value, embedding_cfg.get("model", "BAAI/bge-m3")
            ),
            deployment=env.get(
                EnvVar.AZURE_OPENAI_EMBEDDING_DEPLOYMENT.value, embedding_cfg.get("deployment")
            ),
        ),
        reranker_model=ModelSection(
            provider=env.get(
                EnvVar.RERANKER_MODEL_PROVIDER.value,
                reranker_cfg.get("provider", ProviderName.HUGGINGFACE.value),
            ),
            model=env.get(
                EnvVar.HUGGINGFACE_RERANKER_MODEL.value,
                reranker_cfg.get("model", "BAAI/bge-reranker-base"),
            ),
        ),
        chunking=ChunkingSettings(**config_data.get("chunking", {})),
        discovery=DiscoverySettings(
            batch_size=discovery_batch_size,
            log_llm_responses=log_llm_responses,
            ledger_enabled=env_bool(
                env.get(EnvVar.DISCOVERY_LEDGER_ENABLED.value),
                default=bool(discovery_cfg.get("ledger_enabled", True)),
            ),
            ledger_max_entries=_positive_int(
                env.get(
                    EnvVar.DISCOVERY_LEDGER_MAX_ENTRIES.value,
                    discovery_cfg.get("ledger_max_entries"),
                ),
                default=500,
            ),
            ledger_top_k=_positive_int(
                env.get(
                    EnvVar.DISCOVERY_LEDGER_TOP_K.value,
                    discovery_cfg.get("ledger_top_k"),
                ),
                default=40,
            ),
        ),
        postgres=PostgresSettings(
            mode=env.get(
                EnvVar.POSTGRES_MODE.value, postgres_cfg.get("mode", ModeName.POSTGRES.value)
            ),
            dsn=env.get(
                EnvVar.POSTGRES_DSN.value,
                postgres_cfg.get("dsn", "postgresql://marag:marag@127.0.0.1:5432/marag"),
            ),
            local_path=Path(
                postgres_cfg.get("local_path", paths.runtime_staging_dir / "postgres_records.jsonl")
            ),
        ),
        neo4j=Neo4jSettings(
            mode=env.get(EnvVar.NEO4J_MODE.value, neo4j_cfg.get("mode", ModeName.NEO4J.value)),
            uri=env.get(EnvVar.NEO4J_URI.value, neo4j_cfg.get("uri", "bolt://127.0.0.1:7687")),
            username=env.get(EnvVar.NEO4J_USERNAME.value, neo4j_cfg.get("username", "neo4j")),
            password=env.get(EnvVar.NEO4J_PASSWORD.value, neo4j_cfg.get("password", "")),
            database=env.get(EnvVar.NEO4J_DATABASE.value, neo4j_cfg.get("database", "neo4j")),
            local_path=Path(
                neo4j_cfg.get("local_path", paths.runtime_staging_dir / "neo4j_projection.jsonl")
            ),
        ),
        chroma=ChromaSettings(**config_data.get("chroma", {})),
        user_story=user_story,
        test_scenario=test_scenario,
        knowledge_graph=knowledge_graph,
        azure_openai=AzureOpenAISettings(
            endpoint=env.get(EnvVar.AZURE_OPENAI_ENDPOINT.value, ""),
            api_key=env.get(EnvVar.AZURE_OPENAI_API_KEY.value, ""),
            api_version=env.get(EnvVar.AZURE_OPENAI_API_VERSION.value, "2024-10-21"),
            reasoning_deployment=env.get(EnvVar.AZURE_OPENAI_REASONING_DEPLOYMENT.value, ""),
            embedding_deployment=env.get(EnvVar.AZURE_OPENAI_EMBEDDING_DEPLOYMENT.value, ""),
        ),
        huggingface=HuggingFaceSettings(
            token=huggingface_token,
            reasoning_model=env.get(
                EnvVar.HUGGINGFACE_REASONING_MODEL.value, "Qwen/Qwen2.5-Coder-7B-Instruct"
            ),
            embedding_model=env.get(EnvVar.HUGGINGFACE_EMBEDDING_MODEL.value, "BAAI/bge-m3"),
            reranker_model=env.get(
                EnvVar.HUGGINGFACE_RERANKER_MODEL.value, "BAAI/bge-reranker-base"
            ),
            offline=huggingface_offline,
            max_new_tokens=env_positive_int(
                env.get(EnvVar.HUGGINGFACE_MAX_NEW_TOKENS.value), default=4096
            ),
            discovery_batch_size=discovery_batch_size,
            log_llm_responses=log_llm_responses,
        ),
        enable_hfil=env_bool(
            env.get(EnvVar.ENABLE_HFIL.value),
            default=bool(hfil_cfg.get("enable", hfil_cfg.get("enable_hfil", False))),
        ),
        hfil_match_threshold_pct=_float(
            env.get(
                EnvVar.HFIL_MATCH_THRESHOLD_PCT.value,
                hfil_cfg.get("match_threshold_pct"),
            ),
            default=60.0,
        ),
        hfil_out_of_context_pct=_float(
            env.get(
                EnvVar.HFIL_OUT_OF_CONTEXT_PCT.value,
                hfil_cfg.get("out_of_context_pct"),
            ),
            default=5.0,
        ),
        hfil_cos_floor=_float(
            env.get(EnvVar.HFIL_COS_FLOOR.value, hfil_cfg.get("cos_floor")),
            default=0.30,
        ),
        hfil_cos_ceil=_float(
            env.get(EnvVar.HFIL_COS_CEIL.value, hfil_cfg.get("cos_ceil")),
            default=0.80,
        ),
        dedup_recall_cosine=_float(
            env.get(EnvVar.DEDUP_RECALL_COSINE.value, hfil_cfg.get("dedup_recall_cosine")),
            default=0.55,
        ),
        hfil_emit_md=env_bool(
            env.get(EnvVar.HFIL_EMIT_MD.value),
            default=bool(hfil_cfg.get("emit_md", False)),
        ),
        hfil_checkpointer=_hfil_checkpointer(
            env.get(
                EnvVar.HFIL_CHECKPOINTER.value,
                hfil_cfg.get("checkpointer", "postgres"),
            )
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
