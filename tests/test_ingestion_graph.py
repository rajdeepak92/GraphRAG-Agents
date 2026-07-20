"""Stage 1.1 graph integration with in-memory external adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.schemas import ChunkManifest, IngestionRequest
from multi_agentic_graph_rag.services.manifest import load_model
from multi_agentic_graph_rag.workflows.ingestion_graph import (
    _Runtime,
    _validate_ingest_tech_stack,
    build_ingestion_graph,
)


class _Postgres:
    def check(self) -> str:
        return "PASS"

    def ensure_schema(self) -> None:
        return None


class _Neo4j:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def check(self) -> str:
        return "PASS"

    def ensure_schema(self) -> None:
        return None

    def upsert_chunk(self, *, project: str, run_id: str, chunk: Any) -> None:
        self.rows[chunk.chunk_id] = {
            "project": project,
            "run_id": run_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.chunk_text,
            "content_hash": chunk.content_hash,
            "sequence_index": chunk.sequence_index,
        }

    def read_chunk(self, project: str, chunk_id: str) -> dict[str, Any] | None:
        return self.rows.get(chunk_id)


class _Chroma:
    def __init__(self, embedding_metadata: dict[str, Any] | None = None) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self._embedding_metadata = embedding_metadata

    def check(self, project: str) -> str:
        return "PASS"

    def embedding_metadata(self, project: str) -> dict[str, Any] | None:
        return self._embedding_metadata

    def upsert_chunk(
        self,
        *,
        project: str,
        run_id: str,
        chunk: Any,
        embedding: list[float],
        embedding_fingerprint: str,
    ) -> None:
        self.rows[chunk.chunk_id] = {
            "id": chunk.chunk_id,
            "document": chunk.chunk_text,
            "metadata": {
                "project": project,
                "run_id": run_id,
                "content_hash": chunk.content_hash,
                "embedding_dimension": len(embedding),
                "embedding_fingerprint": embedding_fingerprint,
            },
            "embedding": embedding,
        }

    def read_chunk(self, project: str, chunk_id: str) -> dict[str, Any] | None:
        return self.rows.get(chunk_id)


class _Embedding:
    provider_name = "test"
    embedding_fingerprint = "test:3"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _GeminiEmbedding:
    provider_name = "gemini"
    embedding_fingerprint = "gemini:gemini-embedding-001"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 3072 for _ in texts]


def test_chroma_store_reads_existing_embedding_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "embedding_fingerprint": "hf:BAAI/bge-m3",
        "embedding_dimension": 1024,
    }

    class _Collection:
        def get(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"limit": 1, "include": ["metadatas"]}
            return {"ids": ["CHK-1"], "metadatas": [expected]}

    class _Client:
        def get_or_create_collection(self, name: str) -> _Collection:
            assert name.endswith("-siimcs")
            return _Collection()

    store = ChromaStore(load_config())
    monkeypatch.setattr(store, "_client", lambda: _Client())

    assert store.embedding_metadata("SIIMCS") == expected


def _runtime_with(
    *,
    chroma: _Chroma,
    embedding: _Embedding | _GeminiEmbedding,
) -> _Runtime:
    return _Runtime(
        settings=load_config(),
        postgres=_Postgres(),  # type: ignore[arg-type]
        neo4j=_Neo4j(),  # type: ignore[arg-type]
        chroma=chroma,  # type: ignore[arg-type]
        embedding=embedding,
        checkpointer=InMemorySaver(),
    )


def test_ingest_tech_stack_rejects_existing_embedding_contract_mismatch() -> None:
    runtime = _runtime_with(
        chroma=_Chroma(
            {
                "embedding_fingerprint": "hf:BAAI/bge-m3",
                "embedding_dimension": 1024,
            }
        ),
        embedding=_GeminiEmbedding(),
    )

    with pytest.raises(ConfigurationError) as error:
        _validate_ingest_tech_stack({"project": "SIIMCS"}, runtime)

    assert str(error.value) == (
        "project 'SIIMCS' was built with hf:BAAI/bge-m3 (1024-dim); "
        "current model is gemini:gemini-embedding-001 (3072-dim). "
        "Run project-reset before re-ingesting."
    )


def test_ingest_tech_stack_rejects_same_dimension_fingerprint_mismatch() -> None:
    runtime = _runtime_with(
        chroma=_Chroma(
            {
                "embedding_fingerprint": "other:model",
                "embedding_dimension": 3,
            }
        ),
        embedding=_Embedding(),
    )

    with pytest.raises(ConfigurationError, match="Run project-reset before re-ingesting"):
        _validate_ingest_tech_stack({"project": "alpha"}, runtime)


def test_ingest_tech_stack_accepts_matching_embedding_contract() -> None:
    runtime = _runtime_with(
        chroma=_Chroma(
            {
                "embedding_fingerprint": "test:3",
                "embedding_dimension": 3,
            }
        ),
        embedding=_Embedding(),
    )

    assert _validate_ingest_tech_stack({"project": "alpha"}, runtime) == {}


def test_ingestion_graph_publishes_only_after_both_stores_validate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "requirements.md"
    source.write_text("# Security\n\nThe service shall retain audit events.", encoding="utf-8")
    settings = load_config()
    settings.paths.generated_dir = tmp_path / "generated"
    runtime = _Runtime(
        settings=settings,
        postgres=_Postgres(),  # type: ignore[arg-type]
        neo4j=_Neo4j(),  # type: ignore[arg-type]
        chroma=_Chroma(),  # type: ignore[arg-type]
        embedding=_Embedding(),
        checkpointer=InMemorySaver(),
    )
    graph = build_ingestion_graph(runtime)
    final = graph.invoke(
        {
            "request": IngestionRequest(
                project_name="Alpha Project",
                source_file=source,
            ).model_dump(mode="json"),
            "run_id": "RUN-1",
        },
        config={"configurable": {"thread_id": "alpha:RUN-1:stage-1.1"}},
    )
    manifest = load_model(Path(final["manifest_path"]), ChunkManifest)
    assert manifest.project == "Alpha Project"
    assert manifest.chunks
    assert all(chunk.neo4j_status == "persisted" for chunk in manifest.chunks)
    assert all(chunk.chroma_status == "persisted" for chunk in manifest.chunks)
