"""Application settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from common_defs import ModeName, ProviderName


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
    mode: str = ModeName.POSTGRES.value
    dsn: str = "postgresql://marag:marag@127.0.0.1:5432/marag"
    local_path: Path


class Neo4jSettings(BaseModel):
    mode: str = ModeName.NEO4J.value
    uri: str = "bolt://127.0.0.1:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    local_path: Path


class ChromaSettings(BaseModel):
    collection_name: str = "marag_chunks"


class DiscoverySettings(BaseModel):
    batch_size: int = 1
    log_llm_responses: bool = False
    ledger_enabled: bool = True
    ledger_max_entries: int = 500
    ledger_top_k: int = 40


class UserStorySettings(BaseModel):
    top_k: int = 4
    dense_k: int = 8
    sparse_k: int = 8
    neighbor_window: int = 1
    max_new_tokens: int | None = None


class TestScenarioSettings(UserStorySettings):
    """Stage-4 retrieval/generation settings, type-compatible with RetrievalService."""


class AzureOpenAISettings(BaseModel):
    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-10-21"
    reasoning_deployment: str = ""
    embedding_deployment: str = ""


class HuggingFaceSettings(BaseModel):
    token: str = ""
    reasoning_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    offline: bool = False
    max_new_tokens: int = 4096
    discovery_batch_size: int = 1
    log_llm_responses: bool = False


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_env: str = "development"
    log_level: str = "INFO"
    paths: PathsSettings
    reasoning_model: ModelSection = Field(
        default_factory=lambda: ModelSection(
            provider=ProviderName.HUGGINGFACE.value, model="Qwen/Qwen2.5-Coder-7B-Instruct"
        )
    )
    embedding_model: ModelSection = Field(
        default_factory=lambda: ModelSection(
            provider=ProviderName.HUGGINGFACE.value, model="BAAI/bge-m3"
        )
    )
    reranker_model: ModelSection = Field(
        default_factory=lambda: ModelSection(
            provider=ProviderName.HUGGINGFACE.value, model="BAAI/bge-reranker-base"
        )
    )
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    postgres: PostgresSettings
    neo4j: Neo4jSettings
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    user_story: UserStorySettings = Field(default_factory=UserStorySettings)
    test_scenario: TestScenarioSettings = Field(default_factory=TestScenarioSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    enable_hfil: bool = False
    # CALIBRATION RATIONALE — DO NOT REMOVE:
    # BGE-family embedding cosine similarities are compressed into a narrow band
    # (roughly [0.6, 1.0] for related text per BAAI guidance); raw cosine does NOT
    # map to an intuitive 0-100% match. cos_floor/cos_ceil linearly rescale the
    # usable cosine band into a calibrated percentage so the 60% / 5% business
    # thresholds behave as intended.
    # RECALIBRATE cos_floor / cos_ceil IF THE EMBEDDING PROVIDER OR MODEL CHANGES.
    hfil_match_threshold_pct: float = 60.0
    hfil_out_of_context_pct: float = 5.0
    hfil_cos_floor: float = 0.30
    hfil_cos_ceil: float = 0.80
    dedup_recall_cosine: float = 0.55
    hfil_emit_md: bool = False
    hfil_checkpointer: Literal["postgres", "memory"] = "postgres"
    raw_config: dict[str, Any] = Field(default_factory=dict)
