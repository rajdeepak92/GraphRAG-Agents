"""Stage 1.1 graph integration with in-memory external adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.domain.schemas import ChunkManifest, IngestionRequest
from multi_agentic_graph_rag.services.manifest import load_model
from multi_agentic_graph_rag.workflows.ingestion_graph import (
    _Runtime,
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
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def check(self, project: str) -> str:
        return "PASS"

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
