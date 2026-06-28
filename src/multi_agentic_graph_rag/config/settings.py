"""Validated application settings for Phase 2."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from multi_agentic_graph_rag.config.paths import (
    find_project_root,
    require_under,
    resolve_under_root,
)
from multi_agentic_graph_rag.config.providers import (
    EmbeddingProvider,
    GraphStoreProvider,
    ReasoningLLMProvider,
    VectorStoreProvider,
)

ConfigStatus = Literal["PASS", "WARN", "FAIL"]


class SettingsError(RuntimeError):
    """Raised when settings cannot be loaded or validated."""


class ConfigCheck(BaseModel):
    """One config-check diagnostic row."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: ConfigStatus
    detail: str


class ApplicationSettings(BaseModel):
    """Application-level non-secret settings."""

    model_config = ConfigDict(extra="forbid")

    app_env: str = "development"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()

        if normalized not in allowed:
            msg = f"Unsupported LOG_LEVEL={value!r}. Allowed values: {sorted(allowed)}"
            raise ValueError(msg)

        return normalized


class PathSettings(BaseModel):
    """Filesystem path settings."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    project_root: Path = Field(default_factory=find_project_root)

    global_cache_dir: Path = Path(".global_cache")
    documents_inbox_dir: Path = Path("documents/inbox")
    generated_artifacts_dir: Path = Path("generated_artifacts")
    chroma_persist_dir: Path = Path("runtime/databases/chroma")
    runtime_staging_dir: Path = Path("runtime/staging")
    runtime_logs_dir: Path = Path("runtime/logs")
    runtime_locks_dir: Path = Path("runtime/locks")

    @model_validator(mode="after")
    def resolve_and_validate(self) -> PathSettings:
        root = self.project_root.resolve()
        self.project_root = root

        path_fields = (
            "global_cache_dir",
            "documents_inbox_dir",
            "generated_artifacts_dir",
            "chroma_persist_dir",
            "runtime_staging_dir",
            "runtime_logs_dir",
            "runtime_locks_dir",
        )

        for field_name in path_fields:
            resolved = resolve_under_root(root, getattr(self, field_name))
            require_under(root, resolved, label=field_name)
            setattr(self, field_name, resolved)

        return self

    def approved_runtime_directories(self) -> tuple[Path, ...]:
        """Directories that Phase 2 may create automatically."""

        return (
            self.global_cache_dir,
            self.documents_inbox_dir,
            self.generated_artifacts_dir,
            self.chroma_persist_dir,
            self.runtime_staging_dir,
            self.runtime_logs_dir,
            self.runtime_locks_dir,
        )

    def cache_environment(self) -> dict[str, Path]:
        """Cache environment variables constrained under .global_cache."""

        return {
            "HF_HOME": self.global_cache_dir / "huggingface",
            "HF_HUB_CACHE": self.global_cache_dir / "huggingface" / "hub",
            "TRANSFORMERS_CACHE": self.global_cache_dir / "transformers",
            "TORCH_HOME": self.global_cache_dir / "torch",
            "XDG_CACHE_HOME": self.global_cache_dir,
            "UV_CACHE_DIR": self.global_cache_dir / "uv",
        }


class PostgresSettings(BaseModel):
    """PostgreSQL configuration."""

    model_config = ConfigDict(extra="forbid")

    dsn: SecretStr | None = None
    pool_size: int = 5
    max_overflow: int = 10


class Neo4jSettings(BaseModel):
    """Neo4j configuration."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = None
    username: str | None = None
    password: SecretStr | None = None
    database: str = "neo4j"


class ChromaSettings(BaseModel):
    """Chroma vector-store configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: VectorStoreProvider = VectorStoreProvider.CHROMA_LOCAL
    collection_name: str = "knowledge_chunks_v1"


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI configuration."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    api_key: SecretStr | None = None
    api_version: str | None = None
    reasoning_deployment: str | None = None
    embedding_deployment: str | None = None


class HuggingFaceSettings(BaseModel):
    """Hugging Face configuration."""

    model_config = ConfigDict(extra="forbid")

    token: SecretStr | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reasoning_model: str | None = None
    offline: bool = False


class EmbeddingSettings(BaseModel):
    """Embedding selection and validation policy."""

    model_config = ConfigDict(extra="forbid")

    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    normalize: bool = True


class RequirementDiscoverySettings(BaseModel):
    """Requirement discovery runtime policy."""

    model_config = ConfigDict(extra="forbid")

    reasoning_provider: ReasoningLLMProvider = ReasoningLLMProvider.AZURE_OPENAI
    max_correction_retries: int = 2
    temperature: float = 0.0


class ChunkingSettings(BaseModel):
    """Chunking configuration."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 1200
    chunk_overlap: int = 150
    minimum_chunk_size: int = 200
    maximum_chunk_size: int = 2000
    preserve_page_boundaries: bool = True
    preserve_heading_context: bool = True


class ObservabilitySettings(BaseModel):
    """Logging and diagnostics configuration."""

    model_config = ConfigDict(extra="forbid")

    structured_logging: bool = True
    log_full_prompts: bool = False
    log_full_documents: bool = False


class StoreProviderSettings(BaseModel):
    """Store provider selection."""

    model_config = ConfigDict(extra="forbid")

    vector_store_provider: VectorStoreProvider = VectorStoreProvider.CHROMA_LOCAL
    graph_store_provider: GraphStoreProvider = GraphStoreProvider.NEO4J_LOCAL


class Settings(BaseModel):
    """Complete application settings."""

    model_config = ConfigDict(extra="forbid")

    application: ApplicationSettings = Field(default_factory=ApplicationSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    requirement_discovery: RequirementDiscoverySettings = Field(
        default_factory=RequirementDiscoverySettings,
    )
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    providers: StoreProviderSettings = Field(default_factory=StoreProviderSettings)

    def diagnostics(self) -> tuple[ConfigCheck, ...]:
        """Return Phase 2 deterministic configuration diagnostics."""

        checks: list[ConfigCheck] = []

        checks.append(
            ConfigCheck(
                name="Project root",
                status="PASS",
                detail=str(self.paths.project_root),
            ),
        )

        for path in self.paths.approved_runtime_directories():
            checks.append(
                ConfigCheck(
                    name=f"Path: {path.name}",
                    status="PASS",
                    detail=str(path),
                ),
            )

        for env_name, path in self.paths.cache_environment().items():
            try:
                require_under(self.paths.global_cache_dir, path, label=env_name)
            except ValueError as exc:
                checks.append(ConfigCheck(name=env_name, status="FAIL", detail=str(exc)))
            else:
                checks.append(
                    ConfigCheck(
                        name=env_name,
                        status="PASS",
                        detail=str(path),
                    ),
                )

        required = {
            "POSTGRES_DSN": self.postgres.dsn,
            "NEO4J_URI": self.neo4j.uri,
            "NEO4J_USERNAME": self.neo4j.username,
            "NEO4J_PASSWORD": self.neo4j.password,
        }

        if self.requirement_discovery.reasoning_provider == ReasoningLLMProvider.AZURE_OPENAI:
            required |= {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai.endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai.api_key,
                "AZURE_OPENAI_REASONING_DEPLOYMENT": self.azure_openai.reasoning_deployment,
            }

        if self.embedding.provider == EmbeddingProvider.AZURE_OPENAI:
            required |= {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai.endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai.api_key,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai.embedding_deployment,
            }

        for name, value in required.items():
            present = _has_value(value)
            checks.append(
                ConfigCheck(
                    name=name,
                    status="PASS" if present else "FAIL",
                    detail=_masked_value(name, value) if present else "Missing",
                ),
            )

        checks.append(
            ConfigCheck(
                name="Reasoning provider",
                status="PASS",
                detail=self.requirement_discovery.reasoning_provider.value,
            ),
        )
        checks.append(
            ConfigCheck(
                name="Embedding provider",
                status="PASS",
                detail=self.embedding.provider.value,
            ),
        )
        checks.append(
            ConfigCheck(
                name="Vector store provider",
                status="PASS",
                detail=self.providers.vector_store_provider.value,
            ),
        )
        checks.append(
            ConfigCheck(
                name="Graph store provider",
                status="PASS",
                detail=self.providers.graph_store_provider.value,
            ),
        )

        return tuple(checks)


ENV_KEY_MAP: dict[str, tuple[str, ...]] = {
    "APP_ENV": ("application", "app_env"),
    "LOG_LEVEL": ("application", "log_level"),
    "PROJECT_ROOT": ("paths", "project_root"),
    "GLOBAL_CACHE_DIR": ("paths", "global_cache_dir"),
    "DOCUMENTS_INBOX_DIR": ("paths", "documents_inbox_dir"),
    "GENERATED_ARTIFACTS_DIR": ("paths", "generated_artifacts_dir"),
    "CHROMA_PERSIST_DIR": ("paths", "chroma_persist_dir"),
    "RUNTIME_STAGING_DIR": ("paths", "runtime_staging_dir"),
    "RUNTIME_LOGS_DIR": ("paths", "runtime_logs_dir"),
    "RUNTIME_LOCKS_DIR": ("paths", "runtime_locks_dir"),
    "POSTGRES_DSN": ("postgres", "dsn"),
    "NEO4J_URI": ("neo4j", "uri"),
    "NEO4J_USERNAME": ("neo4j", "username"),
    "NEO4J_PASSWORD": ("neo4j", "password"),
    "NEO4J_DATABASE": ("neo4j", "database"),
    "REASONING_LLM_PROVIDER": ("requirement_discovery", "reasoning_provider"),
    "EMBEDDING_PROVIDER": ("embedding", "provider"),
    "VECTOR_STORE_PROVIDER": ("providers", "vector_store_provider"),
    "GRAPH_STORE_PROVIDER": ("providers", "graph_store_provider"),
    "AZURE_OPENAI_ENDPOINT": ("azure_openai", "endpoint"),
    "AZURE_OPENAI_API_KEY": ("azure_openai", "api_key"),
    "AZURE_OPENAI_API_VERSION": ("azure_openai", "api_version"),
    "AZURE_OPENAI_REASONING_DEPLOYMENT": ("azure_openai", "reasoning_deployment"),
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": ("azure_openai", "embedding_deployment"),
    "HUGGINGFACE_TOKEN": ("huggingface", "token"),
    "HUGGINGFACE_EMBEDDING_MODEL": ("huggingface", "embedding_model"),
    "HUGGINGFACE_REASONING_MODEL": ("huggingface", "reasoning_model"),
    "HUGGINGFACE_OFFLINE": ("huggingface", "offline"),
}


def load_settings(
    *,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load settings using CLI > env/.env > config.json > defaults precedence."""

    project_root = find_project_root()
    resolved_config_path = config_path or project_root / "config.json"

    data: dict[str, Any] = {}

    if resolved_config_path.is_file():
        data = _deep_merge(data, _read_json_object(resolved_config_path))

    env_values = _read_dotenv(project_root / ".env")
    env_values.update(os.environ)

    data = _deep_merge(data, _env_to_nested_dict(env_values))

    if overrides:
        data = _deep_merge(data, dict(overrides))

    try:
        return Settings.model_validate(data)
    except Exception as exc:
        raise SettingsError(str(exc)) from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        msg = f"{path} must contain a JSON object."
        raise SettingsError(msg)

    return raw


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    parsed: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")

    return parsed


def _env_to_nested_dict(env_values: Mapping[str, str]) -> dict[str, Any]:
    nested: dict[str, Any] = {}

    for env_key, path in ENV_KEY_MAP.items():
        if env_key not in env_values:
            continue

        raw_value = env_values[env_key]

        if raw_value == "":
            continue

        current = nested

        for part in path[:-1]:
            current = current.setdefault(part, {})

        current[path[-1]] = _coerce_env_value(raw_value)

    return nested


def _coerce_env_value(value: str) -> Any:
    lowered = value.lower()

    if lowered in {"true", "false"}:
        return lowered == "true"

    if re.fullmatch(r"\d+", value):
        return int(value)

    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _has_value(value: object) -> bool:
    if value is None:
        return False

    if isinstance(value, SecretStr):
        return bool(value.get_secret_value())

    if isinstance(value, str):
        return bool(value.strip())

    return True


def _masked_value(name: str, value: object) -> str:
    raw_value = value.get_secret_value() if isinstance(value, SecretStr) else str(value)

    if "DSN" in name:
        return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1********\2", raw_value)

    if any(token in name for token in ("PASSWORD", "TOKEN", "KEY", "SECRET")):
        return "********"

    return raw_value
