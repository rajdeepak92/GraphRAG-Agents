"""Validated operational settings with stage capability boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HuggingFaceDevice = Literal["auto", "cpu", "cuda"]
HuggingFaceQuantization = Literal["none", "bitsandbytes_4bit"]


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


class AzureOpenAISettings(BaseModel):
    """Azure OpenAI credentials and deployments."""

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-10-21"
    reasoning_deployment: str = ""
    embedding_deployment: str = ""


class HuggingFaceSettings(BaseModel):
    """Private Hugging Face model configuration."""

    token: str = ""
    reasoning_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: HuggingFaceDevice = "auto"
    quantization: HuggingFaceQuantization = "none"
    disable_thinking: bool = True
    offline: bool = False
    max_new_tokens: int = 4096
    stage12_max_new_tokens: int = Field(default=1536, ge=1)
    discovery_batch_size: int = 1
    log_llm_responses: bool = False


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
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
