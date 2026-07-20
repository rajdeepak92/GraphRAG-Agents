"""Validated operational settings with stage capability boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HuggingFaceDevice = Literal["auto", "cpu", "cuda"]
Stage4ReasoningProvider = Literal["azure_openai", "gemini"]

_STAGE4_WRITE_ALLOWLIST = (
    "tests/<module>/Tc<id><PascalTitle>.py",
    "tests/<module>/__init__.py",
    "tests_robot/<module>/Tc<id><PascalTitle>.robot",
    "test_lib/<module>/<module>_wrappers.py",
    "test_lib/<module>/<module>_helpers.py",
    "test_lib/<module>/__init__.py",
)


class ModelSection(BaseModel):
    """Provider and deployment/model selection."""

    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str | None = None
    deployment: str | None = None


class PathsSettings(BaseModel):
    """Runtime filesystem locations."""

    project_root: Path
    generated_dir: Path
    chroma_persist_dir: Path
    runtime_staging_dir: Path
    runtime_logs_dir: Path


class ChunkingSettings(BaseModel):
    """Deterministic chunking limits."""

    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    maximum_chunk_size: int = Field(default=2000, gt=0)


class RuntimeSettings(BaseModel):
    """Bounded execution controls."""

    concurrency: int = Field(default=1, ge=1, le=32)
    timeout_seconds: int = Field(default=120, ge=1)
    transient_max_attempts: int = Field(default=2, ge=1, le=2)


class RetrievalSettings(BaseModel):
    """Hybrid retrieval and context limits."""

    top_k: int = Field(default=6, ge=1)
    vector_k: int = Field(default=12, ge=1)
    graph_k: int = Field(default=12, ge=1)
    max_hops: int = Field(default=2, ge=1, le=4)
    token_budget: int = Field(default=6000, ge=512)


class PostgresSettings(BaseModel):
    """PostgreSQL application/checkpoint store."""

    mode: str = "postgres"
    dsn: str = "postgresql://marag:marag@127.0.0.1:5432/marag"
    local_path: Path


class Neo4jSettings(BaseModel):
    """Neo4j project-scoped graph store."""

    mode: str = "neo4j"
    uri: str = "bolt://127.0.0.1:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    local_path: Path


class ChromaSettings(BaseModel):
    """Chroma project-collection configuration."""

    collection_prefix: str = "marag"


class Stage4Settings(BaseModel):
    """Optimized no-Git Stage-4 test-code generation controls."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    code_graph_local_path: Path = Path("runtime/staging/code_graph.jsonl")
    test_data_local_path: Path = Path("runtime/staging/test_data.jsonl")
    codegen_local_path: Path = Path("runtime/staging/codegen_records.jsonl")
    framework_allowed_roots: list[Path] = Field(default_factory=list)
    write_allowlist: list[str] = Field(default_factory=lambda: list(_STAGE4_WRITE_ALLOWLIST))
    rollback_failed_case: Literal[True] = True
    robot_dryrun: Literal[True] = True
    model_transport_retries: int = Field(default=1, ge=1, le=1)
    reindex_after_each_case: Literal[True] = True
    retain_referenced_snapshots: Literal[True] = True
    graphify_command: str = "graphify"
    graphify_no_cluster: bool = True
    extractor_version: str = "graphify-adapter-1"
    max_repair_attempts: int = Field(default=2, ge=1, le=2)
    context_token_budget: int = Field(default=24000, ge=512)
    symbol_search_limit: int = Field(default=20, ge=1)
    max_context_symbols: int = Field(default=80, ge=1)
    reasoning_provider: Stage4ReasoningProvider = "azure_openai"
    label_prefix: str = "Code"


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI credentials and deployments."""

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-10-21"
    reasoning_deployment: str = ""
    embedding_deployment: str = ""
    log_llm_responses: bool = False


class GeminiSettings(BaseModel):
    """Google Gemini (Developer API) credentials and model selection."""

    api_key: str = ""
    reasoning_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    log_llm_responses: bool = False


class HuggingFaceSettings(BaseModel):
    """Private Hugging Face reranker model configuration."""

    token: str = ""
    reranker_model: str = "BAAI/bge-reranker-base"
    device: HuggingFaceDevice = "auto"
    offline: bool = False


class AppSettings(BaseModel):
    """Complete operational configuration."""

    model_config = ConfigDict(extra="forbid")
    app_env: str = "development"
    log_level: str = "INFO"
    paths: PathsSettings
    reasoning_model: ModelSection
    embedding_model: ModelSection
    reranker_model: ModelSection
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    postgres: PostgresSettings
    neo4j: Neo4jSettings
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    stage4: Stage4Settings = Field(default_factory=Stage4Settings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
