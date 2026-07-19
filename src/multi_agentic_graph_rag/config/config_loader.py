"""Load operational configuration from config.json, .env, and environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from multi_agentic_graph_rag.config.huggingface_env import (
    env_bool,
    export_huggingface_environment,
    first_huggingface_token,
)
from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    AzureOpenAISettings,
    ChromaSettings,
    ChunkingSettings,
    HuggingFaceDevice,
    HuggingFaceQuantization,
    HuggingFaceSettings,
    ModelSection,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
    RetrievalSettings,
    RuntimeSettings,
    Stage4ReasoningProvider,
    Stage4Settings,
)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return dict(value) if isinstance(value, dict) else {}


def _path(root: Path, value: object, default: str) -> Path:
    candidate = Path(str(value or default))
    return candidate if candidate.is_absolute() else root / candidate


def _int(value: object, default: int) -> int:
    try:
        return int(str(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _build_stage4(root: Path, cfg: dict[str, Any], env: dict[str, str]) -> Stage4Settings:
    raw_roots = cfg.get("framework_allowed_roots") or [str(root)]
    if isinstance(raw_roots, str):
        raw_roots = [raw_roots]
    allowed_roots = [_path(root, entry, str(root)) for entry in raw_roots]
    raw_allowlist: object = env.get("STAGE4_WRITE_ALLOWLIST", cfg.get("write_allowlist"))
    if isinstance(raw_allowlist, str):
        write_allowlist = [item.strip() for item in raw_allowlist.split(",") if item.strip()]
    elif isinstance(raw_allowlist, list):
        write_allowlist = [str(item) for item in raw_allowlist]
    else:
        write_allowlist = Stage4Settings().write_allowlist
    return Stage4Settings(
        code_graph_local_path=_path(
            root, cfg.get("code_graph_local_path"), "runtime/staging/code_graph.jsonl"
        ),
        test_data_local_path=_path(
            root, cfg.get("test_data_local_path"), "runtime/staging/test_data.jsonl"
        ),
        codegen_local_path=_path(
            root, cfg.get("codegen_local_path"), "runtime/staging/codegen_records.jsonl"
        ),
        framework_allowed_roots=allowed_roots,
        write_allowlist=write_allowlist,
        rollback_failed_case=cast(
            Literal[True],
            env_bool(
                env.get("STAGE4_ROLLBACK_FAILED_CASE"),
                default=bool(cfg.get("rollback_failed_case", True)),
            ),
        ),
        robot_dryrun=cast(
            Literal[True],
            env_bool(env.get("STAGE4_ROBOT_DRYRUN"), default=bool(cfg.get("robot_dryrun", True))),
        ),
        model_transport_retries=_int(
            env.get("STAGE4_MODEL_TRANSPORT_RETRIES", cfg.get("model_transport_retries")),
            1,
        ),
        reindex_after_each_case=cast(
            Literal[True],
            env_bool(
                env.get("STAGE4_REINDEX_AFTER_EACH_CASE"),
                default=bool(cfg.get("reindex_after_each_case", True)),
            ),
        ),
        retain_referenced_snapshots=cast(
            Literal[True],
            env_bool(
                env.get("STAGE4_RETAIN_REFERENCED_SNAPSHOTS"),
                default=bool(cfg.get("retain_referenced_snapshots", True)),
            ),
        ),
        graphify_command=str(
            env.get("STAGE4_GRAPHIFY_COMMAND", cfg.get("graphify_command", "graphify"))
        ),
        graphify_no_cluster=env_bool(
            env.get("STAGE4_GRAPHIFY_NO_CLUSTER"),
            default=bool(cfg.get("graphify_no_cluster", True)),
        ),
        extractor_version=env.get(
            "STAGE4_EXTRACTOR_VERSION", cfg.get("extractor_version", "graphify-adapter-1")
        ),
        max_repair_attempts=_int(
            env.get("STAGE4_MAX_REPAIR_ATTEMPTS", cfg.get("max_repair_attempts")), 2
        ),
        context_token_budget=_int(
            env.get("STAGE4_CONTEXT_TOKEN_BUDGET", cfg.get("context_token_budget")), 24000
        ),
        symbol_search_limit=_int(
            env.get("STAGE4_SYMBOL_SEARCH_LIMIT", cfg.get("symbol_search_limit")), 20
        ),
        max_context_symbols=_int(
            env.get("STAGE4_MAX_CONTEXT_SYMBOLS", cfg.get("max_context_symbols")), 80
        ),
        reasoning_provider=cast(
            Stage4ReasoningProvider,
            str(
                env.get(
                    "STAGE4_REASONING_PROVIDER",
                    cfg.get("reasoning_provider", "azure_openai"),
                )
            ).lower(),
        ),
        label_prefix=cfg.get("label_prefix", "Code"),
    )


def load_config(config_path: Path | None = None) -> AppSettings:
    """Load and validate the current operational configuration."""
    root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    path = config_path or root / "config.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    env = {**_read_dotenv(root / ".env"), **os.environ}

    paths_cfg = _section(data, "paths")
    generated_dir = _path(
        root,
        env.get("GENERATED_DIR", paths_cfg.get("generated_requirements_dir")),
        "generated",
    )
    staging_dir = _path(
        root,
        env.get("RUNTIME_STAGING_DIR", paths_cfg.get("runtime_staging_dir")),
        "runtime/staging",
    )
    paths = PathsSettings(
        project_root=root,
        generated_dir=generated_dir,
        chroma_persist_dir=_path(
            root,
            env.get("CHROMA_PERSIST_DIR", paths_cfg.get("chroma_persist_dir")),
            "runtime/databases/chroma",
        ),
        runtime_staging_dir=staging_dir,
        runtime_logs_dir=_path(
            root,
            env.get("RUNTIME_LOGS_DIR", paths_cfg.get("runtime_logs_dir")),
            "runtime/logs",
        ),
    )

    reasoning_cfg = _section(data, "reasoning_model")
    embedding_cfg = _section(data, "embedding_model")
    reranker_cfg = _section(data, "reranker_model")
    postgres_cfg = _section(data, "postgres")
    neo4j_cfg = _section(data, "neo4j")
    azure_cfg = _section(data, "azure_openai")
    hf_cfg = _section(data, "huggingface")
    runtime_cfg = _section(data, "runtime")
    retrieval_cfg = _section(data, "retrieval")

    hf_token = first_huggingface_token(env)
    hf_offline = env_bool(
        env.get("HUGGINGFACE_OFFLINE"),
        default=bool(hf_cfg.get("offline", False)),
    )
    export_huggingface_environment(token=hf_token, offline=hf_offline)

    settings = AppSettings(
        app_env=env.get("APP_ENV", _section(data, "application").get("app_env", "development")),
        log_level=env.get("LOG_LEVEL", _section(data, "application").get("log_level", "INFO")),
        paths=paths,
        reasoning_model=ModelSection(
            provider=env.get(
                "REASONING_MODEL_PROVIDER",
                reasoning_cfg.get("provider", "huggingface"),
            ),
            model=env.get(
                "HUGGINGFACE_REASONING_MODEL",
                reasoning_cfg.get("model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ),
            deployment=env.get(
                "AZURE_OPENAI_REASONING_DEPLOYMENT",
                reasoning_cfg.get("deployment"),
            ),
        ),
        embedding_model=ModelSection(
            provider=env.get(
                "EMBEDDING_MODEL_PROVIDER",
                embedding_cfg.get("provider", "huggingface"),
            ),
            model=env.get(
                "HUGGINGFACE_EMBEDDING_MODEL",
                embedding_cfg.get("model", "BAAI/bge-m3"),
            ),
            deployment=env.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                embedding_cfg.get("deployment"),
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
        chunking=ChunkingSettings(**_section(data, "chunking")),
        runtime=RuntimeSettings(
            concurrency=_int(env.get("WORKFLOW_CONCURRENCY", runtime_cfg.get("concurrency")), 1),
            timeout_seconds=_int(
                env.get("WORKFLOW_TIMEOUT_SECONDS", runtime_cfg.get("timeout_seconds")), 120
            ),
            transient_max_attempts=_int(
                env.get(
                    "TRANSIENT_MAX_ATTEMPTS",
                    runtime_cfg.get("transient_max_attempts"),
                ),
                2,
            ),
        ),
        retrieval=RetrievalSettings(
            top_k=_int(env.get("RETRIEVAL_TOP_K", retrieval_cfg.get("top_k")), 6),
            vector_k=_int(env.get("RETRIEVAL_VECTOR_K", retrieval_cfg.get("vector_k")), 12),
            graph_k=_int(env.get("RETRIEVAL_GRAPH_K", retrieval_cfg.get("graph_k")), 12),
            max_hops=_int(env.get("RETRIEVAL_MAX_HOPS", retrieval_cfg.get("max_hops")), 2),
            token_budget=_int(
                env.get("RETRIEVAL_TOKEN_BUDGET", retrieval_cfg.get("token_budget")), 6000
            ),
        ),
        postgres=PostgresSettings(
            mode=env.get("POSTGRES_MODE", postgres_cfg.get("mode", "postgres")),
            dsn=env.get(
                "POSTGRES_DSN",
                postgres_cfg.get(
                    "dsn",
                    "postgresql://marag:marag@127.0.0.1:5432/marag",
                ),
            ),
            local_path=_path(
                root,
                postgres_cfg.get("local_path"),
                "runtime/staging/postgres_records.jsonl",
            ),
        ),
        neo4j=Neo4jSettings(
            mode=env.get("NEO4J_MODE", neo4j_cfg.get("mode", "neo4j")),
            uri=env.get("NEO4J_URI", neo4j_cfg.get("uri", "bolt://127.0.0.1:7687")),
            username=env.get("NEO4J_USERNAME", neo4j_cfg.get("username", "neo4j")),
            password=env.get("NEO4J_PASSWORD", neo4j_cfg.get("password", "")),
            database=env.get("NEO4J_DATABASE", neo4j_cfg.get("database", "neo4j")),
            local_path=_path(
                root,
                neo4j_cfg.get("local_path"),
                "runtime/staging/neo4j_projection.jsonl",
            ),
        ),
        chroma=ChromaSettings(
            collection_prefix=_section(data, "chroma").get("collection_prefix", "marag")
        ),
        stage4=_build_stage4(root, _section(data, "stage4"), env),
        azure_openai=AzureOpenAISettings(
            endpoint=env.get("AZURE_OPENAI_ENDPOINT", azure_cfg.get("endpoint", "")),
            api_key=env.get("AZURE_OPENAI_API_KEY", azure_cfg.get("api_key", "")),
            api_version=env.get(
                "AZURE_OPENAI_API_VERSION",
                azure_cfg.get("api_version", "2024-10-21"),
            ),
            reasoning_deployment=env.get(
                "AZURE_OPENAI_REASONING_DEPLOYMENT",
                azure_cfg.get("reasoning_deployment", ""),
            ),
            embedding_deployment=env.get(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                azure_cfg.get("embedding_deployment", ""),
            ),
            log_llm_responses=env_bool(
                env.get("AZURE_OPENAI_LOG_LLM_RESPONSES"),
                default=bool(azure_cfg.get("log_llm_responses", False)),
            ),
        ),
        huggingface=HuggingFaceSettings(
            token=hf_token,
            reasoning_model=env.get(
                "HUGGINGFACE_REASONING_MODEL",
                hf_cfg.get("reasoning_model", "Qwen/Qwen2.5-Coder-7B-Instruct"),
            ),
            model_revision=env.get(
                "HUGGINGFACE_MODEL_REVISION",
                hf_cfg.get("model_revision"),
            ),
            embedding_model=env.get(
                "HUGGINGFACE_EMBEDDING_MODEL",
                hf_cfg.get("embedding_model", "BAAI/bge-m3"),
            ),
            reranker_model=env.get(
                "HUGGINGFACE_RERANKER_MODEL",
                hf_cfg.get("reranker_model", "BAAI/bge-reranker-base"),
            ),
            device=cast(
                HuggingFaceDevice,
                str(
                    env.get(
                        "HUGGINGFACE_DEVICE",
                        hf_cfg.get("device", "auto"),
                    )
                ).lower(),
            ),
            quantization=cast(
                HuggingFaceQuantization,
                str(
                    env.get(
                        "HUGGINGFACE_QUANTIZATION",
                        hf_cfg.get("quantization", "none"),
                    )
                ).lower(),
            ),
            disable_thinking=env_bool(
                env.get("HUGGINGFACE_DISABLE_THINKING"),
                default=bool(hf_cfg.get("disable_thinking", True)),
            ),
            offline=hf_offline,
            max_new_tokens=_int(
                env.get("HUGGINGFACE_MAX_NEW_TOKENS", hf_cfg.get("max_new_tokens")), 4096
            ),
            stage12_max_new_tokens=_int(
                env.get(
                    "HUGGINGFACE_STAGE12_MAX_NEW_TOKENS",
                    hf_cfg.get("stage12_max_new_tokens"),
                ),
                1536,
            ),
            log_llm_responses=env_bool(
                env.get("LOG_LLM_RESPONSES"),
                default=bool(hf_cfg.get("log_llm_responses", False)),
            ),
        ),
    )
    return settings


__all__ = ["load_config"]
