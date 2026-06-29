"""Application settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model: str | None = None
    deployment: str | None = None


class PathsSettings(BaseModel):
    project_root: Path
    global_cache_dir: Path
    documents_inbox_dir: Path
    generated_requirements_dir: Path
    chroma_persist_dir: Path
    runtime_staging_dir: Path
    runtime_logs_dir: Path
    runtime_locks_dir: Path


class ChunkingSettings(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 150
    minimum_chunk_size: int = 80
    maximum_chunk_size: int = 2000


class PostgresSettings(BaseModel):
    mode: str = "local_json"
    dsn: str = "postgresql://marag:marag@127.0.0.1:5432/marag"
    local_path: Path


class Neo4jSettings(BaseModel):
    mode: str = "local_json"
    uri: str = "bolt://127.0.0.1:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    local_path: Path


class ChromaSettings(BaseModel):
    collection_name: str = "marag_chunks"


class AzureOpenAISettings(BaseModel):
    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-10-21"
    reasoning_deployment: str = ""
    embedding_deployment: str = ""


class HuggingFaceSettings(BaseModel):
    token: str = ""
    reasoning_model: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    offline: bool = False


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_env: str = "development"
    log_level: str = "INFO"
    paths: PathsSettings
    reasoning_model: ModelSection = Field(
        default_factory=lambda: ModelSection(provider="local_heuristic", model="rules")
    )
    embedding_model: ModelSection = Field(
        default_factory=lambda: ModelSection(provider="local_hash", model="hash-384")
    )
    reranker_model: ModelSection = Field(
        default_factory=lambda: ModelSection(provider="none", model=None)
    )
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    postgres: PostgresSettings
    neo4j: Neo4jSettings
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    raw_config: dict[str, Any] = Field(default_factory=dict)
