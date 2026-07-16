"""Stage 1.1 LangGraph: parse, chunk, persist, validate, and publish."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import (
    make_checkpoint_thread_id,
    new_run_id,
    normalize_project,
)
from multi_agentic_graph_rag.domain.schemas import (
    ChunkManifest,
    IngestionRequest,
    IngestionResult,
    ManifestChunk,
)
from multi_agentic_graph_rag.llm_models.factory import create_embedding_model
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel
from multi_agentic_graph_rag.services.checkpointing import workflow_checkpointer
from multi_agentic_graph_rag.services.chunking import chunk_blocks
from multi_agentic_graph_rag.services.manifest import (
    atomic_write_model,
    build_chunk_manifest,
    load_model,
)
from multi_agentic_graph_rag.services.parsing import parse_document
from multi_agentic_graph_rag.services.retry import retry_transient_once


class IngestionState(TypedDict, total=False):
    """Parent Stage 1.1 state."""

    request: dict[str, Any]
    project: str
    run_id: str
    artifact_dir: str
    manifest_path: str
    chunk_ids: list[str]
    errors: list[str]


class ChunkPipelineState(TypedDict, total=False):
    """Checkpointed per-chunk subgraph state."""

    project: str
    run_id: str
    source_file: str
    artifact_dir: str
    chunks: list[dict[str, Any]]
    current_index: int
    current_embedding: list[float]
    current_chroma_written: bool
    embedding_dimension: int
    parser_fingerprint: str
    chunker_fingerprint: str
    manifest_path: str


@dataclass
class _Runtime:
    settings: AppSettings
    postgres: PostgresStore
    neo4j: Neo4jStore
    chroma: ChromaStore
    embedding: EmbeddingModel
    checkpointer: Any


def build_ingestion_graph(runtime: _Runtime) -> Any:
    """Build the four-node IngestProjectRun parent graph."""
    graph = StateGraph(IngestionState)
    graph.add_node("validate_user_ingest_request", _validate_user_ingest_request)
    graph.add_node(
        "validate_ingest_tech_stack",
        lambda state: _validate_ingest_tech_stack(state, runtime),
    )
    graph.add_node(
        "run_ingest_pipeline",
        lambda state: _run_ingest_pipeline(state, runtime),
    )
    graph.add_node("validate_session_exit_gate", _validate_session_exit_gate)
    graph.set_entry_point("validate_user_ingest_request")
    graph.add_edge("validate_user_ingest_request", "validate_ingest_tech_stack")
    graph.add_edge("validate_ingest_tech_stack", "run_ingest_pipeline")
    graph.add_edge("run_ingest_pipeline", "validate_session_exit_gate")
    graph.add_edge("validate_session_exit_gate", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _validate_user_ingest_request(state: IngestionState) -> IngestionState:
    request = IngestionRequest.model_validate(state["request"])
    if not request.source_file.exists() or not request.source_file.is_file():
        raise FileNotFoundError(request.source_file)
    if request.source_file.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
        raise ValueError("source_file must be .txt, .md, .pdf, or .docx")
    run_id = state.get("run_id") or new_run_id(request.project_name)
    return {
        "request": request.model_dump(mode="json"),
        "project": request.project_name,
        "run_id": run_id,
        "errors": [],
    }


def _validate_ingest_tech_stack(
    state: IngestionState,
    runtime: _Runtime,
) -> IngestionState:
    project = state["project"]
    runtime.postgres.check()
    runtime.postgres.ensure_schema()
    runtime.neo4j.check()
    runtime.neo4j.ensure_schema()
    runtime.chroma.check(project)
    probe = runtime.embedding.embed_documents(["embedding compatibility probe"])
    if len(probe) != 1 or not probe[0] or any(not math.isfinite(value) for value in probe[0]):
        raise ValueError("embedding model compatibility probe failed")
    return {}


def _run_ingest_pipeline(
    state: IngestionState,
    runtime: _Runtime,
) -> IngestionState:
    artifact_dir = (
        runtime.settings.paths.generated_dir
        / normalize_project(state["project"])
        / state["run_id"]
        / "requirements"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    subgraph = _build_chunk_subgraph(runtime)
    child_thread = make_checkpoint_thread_id(state["project"], state["run_id"], "stage-1.1-chunks")
    final = subgraph.invoke(
        {
            "project": state["project"],
            "run_id": state["run_id"],
            "source_file": state["request"]["source_file"],
            "artifact_dir": str(artifact_dir),
            "current_index": 0,
        },
        config={"configurable": {"thread_id": child_thread}},
    )
    return {
        "artifact_dir": str(artifact_dir),
        "manifest_path": final["manifest_path"],
        "chunk_ids": [ManifestChunk.model_validate(chunk).chunk_id for chunk in final["chunks"]],
    }


def _validate_session_exit_gate(state: IngestionState) -> IngestionState:
    manifest = load_model(Path(state["manifest_path"]), ChunkManifest)
    if manifest.project != state["project"] or manifest.run_id != state["run_id"]:
        raise ValueError("manifest project/run scope mismatch")
    if [chunk.chunk_id for chunk in manifest.chunks] != state["chunk_ids"]:
        raise ValueError("manifest chunk IDs do not match completed checkpoint state")
    return {}


def _build_chunk_subgraph(runtime: _Runtime) -> Any:
    graph = StateGraph(ChunkPipelineState)
    graph.add_node(
        "parse_and_create_chunks",
        lambda state: _parse_and_create_chunks(state, runtime),
    )
    graph.add_node("persist_chunk_neo4j", lambda state: _persist_chunk_neo4j(state, runtime))
    graph.add_node("generate_chunk_embedding", lambda state: _generate_embedding(state, runtime))
    graph.add_node("persist_chunk_chroma", lambda state: _persist_chunk_chroma(state, runtime))
    graph.add_node("validate_chunk_persistence", lambda state: _validate_chunk(state, runtime))
    graph.add_node("publish_chunk_manifest", _publish_manifest)
    graph.set_entry_point("parse_and_create_chunks")
    graph.add_edge("parse_and_create_chunks", "persist_chunk_neo4j")
    graph.add_edge("persist_chunk_neo4j", "generate_chunk_embedding")
    graph.add_edge("generate_chunk_embedding", "persist_chunk_chroma")
    graph.add_edge("persist_chunk_chroma", "validate_chunk_persistence")
    graph.add_conditional_edges(
        "validate_chunk_persistence",
        _chunk_route,
        {
            "next": "persist_chunk_neo4j",
            "publish": "publish_chunk_manifest",
        },
    )
    graph.add_edge("publish_chunk_manifest", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _parse_and_create_chunks(
    state: ChunkPipelineState,
    runtime: _Runtime,
) -> ChunkPipelineState:
    if state.get("chunks"):
        return {}
    blocks, parser_fingerprint = parse_document(Path(state["source_file"]))
    chunks, chunker_fingerprint = chunk_blocks(blocks, runtime.settings.chunking)
    if not chunks:
        raise ValueError("source document produced no chunks")
    return {
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "current_index": state.get("current_index", 0),
        "parser_fingerprint": parser_fingerprint,
        "chunker_fingerprint": chunker_fingerprint,
    }


def _current_chunk(state: ChunkPipelineState) -> ManifestChunk:
    return ManifestChunk.model_validate(state["chunks"][state["current_index"]])


def _replace_current(
    state: ChunkPipelineState,
    chunk: ManifestChunk,
) -> list[dict[str, Any]]:
    chunks = list(state["chunks"])
    chunks[state["current_index"]] = chunk.model_dump(mode="json")
    return chunks


def _persist_chunk_neo4j(
    state: ChunkPipelineState,
    runtime: _Runtime,
) -> ChunkPipelineState:
    chunk = _current_chunk(state)
    if chunk.neo4j_status == "persisted":
        return {}
    retry_transient_once(
        lambda: runtime.neo4j.upsert_chunk(
            project=state["project"],
            run_id=state["run_id"],
            chunk=chunk,
        )
    )
    stored = retry_transient_once(
        lambda: runtime.neo4j.read_chunk(state["project"], chunk.chunk_id)
    )
    if stored is None or any(
        (
            stored.get("project") != state["project"],
            stored.get("chunk_id") != chunk.chunk_id,
            stored.get("text") != chunk.chunk_text,
            stored.get("content_hash") != chunk.content_hash,
            int(stored.get("sequence_index", -1)) != chunk.sequence_index,
        )
    ):
        raise ValueError("Neo4j chunk read-back validation failed")
    return {
        "chunks": _replace_current(state, chunk.model_copy(update={"neo4j_status": "persisted"}))
    }


def _generate_embedding(
    state: ChunkPipelineState,
    runtime: _Runtime,
) -> ChunkPipelineState:
    if state.get("current_embedding"):
        return {}
    chunk = _current_chunk(state)
    vectors = retry_transient_once(lambda: runtime.embedding.embed_documents([chunk.chunk_text]))
    if len(vectors) != 1 or not vectors[0]:
        raise ValueError("embedding model returned an invalid vector count")
    vector = [float(value) for value in vectors[0]]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding contains a non-finite value")
    expected = state.get("embedding_dimension")
    if expected is not None and expected != len(vector):
        raise ValueError("embedding dimension changed within the run")
    return {"current_embedding": vector, "embedding_dimension": len(vector)}


def _persist_chunk_chroma(
    state: ChunkPipelineState,
    runtime: _Runtime,
) -> ChunkPipelineState:
    chunk = _current_chunk(state)
    if chunk.chroma_status == "persisted" or state.get("current_chroma_written", False):
        return {}
    embedding = state["current_embedding"]
    retry_transient_once(
        lambda: runtime.chroma.upsert_chunk(
            project=state["project"],
            run_id=state["run_id"],
            chunk=chunk,
            embedding=embedding,
            embedding_fingerprint=runtime.embedding.embedding_fingerprint,
        )
    )
    return {"current_chroma_written": True}


def _validate_chunk(
    state: ChunkPipelineState,
    runtime: _Runtime,
) -> ChunkPipelineState:
    chunk = _current_chunk(state)
    record = retry_transient_once(
        lambda: runtime.chroma.read_chunk(state["project"], chunk.chunk_id)
    )
    if record is None:
        raise ValueError("Chroma chunk record is missing")
    metadata = record["metadata"]
    if any(
        (
            record["id"] != chunk.chunk_id,
            record["document"] != chunk.chunk_text,
            metadata.get("project") != state["project"],
            metadata.get("content_hash") != chunk.content_hash,
            int(metadata.get("embedding_dimension", -1)) != state["embedding_dimension"],
        )
    ):
        raise ValueError("Chroma chunk read-back validation failed")
    updated = chunk.model_copy(update={"chroma_status": "persisted"})
    return {
        "chunks": _replace_current(state, updated),
        "current_index": state["current_index"] + 1,
        "current_embedding": [],
        "current_chroma_written": False,
    }


def _chunk_route(state: ChunkPipelineState) -> Literal["next", "publish"]:
    return "publish" if state["current_index"] >= len(state["chunks"]) else "next"


def _publish_manifest(state: ChunkPipelineState) -> ChunkPipelineState:
    chunks = [ManifestChunk.model_validate(item) for item in state["chunks"]]
    manifest = build_chunk_manifest(
        project=state["project"],
        run_id=state["run_id"],
        chunks=chunks,
    )
    path = atomic_write_model(
        manifest,
        Path(state["artifact_dir"]) / "chunk_manifest.json",
    )
    return {"manifest_path": str(path)}


def run_ingestion(
    request: IngestionRequest,
    *,
    settings: AppSettings | None = None,
) -> IngestionResult:
    """Execute Stage 1.1 with durable run and per-chunk checkpoints."""
    resolved = settings or load_config()
    if request.embedding_provider is not None:
        resolved.embedding_model.provider = request.embedding_provider
    run_id = new_run_id(request.project_name)
    postgres = PostgresStore(resolved)
    postgres.record_run(
        run_id=run_id,
        project=request.project_name,
        stage="stage-1.1",
        status="started",
        payload={},
    )
    with workflow_checkpointer(resolved) as checkpointer:
        runtime = _Runtime(
            settings=resolved,
            postgres=postgres,
            neo4j=Neo4jStore(resolved),
            chroma=ChromaStore(resolved),
            embedding=create_embedding_model(resolved),
            checkpointer=checkpointer,
        )
        graph = build_ingestion_graph(runtime)
        try:
            final = graph.invoke(
                {"request": request.model_dump(mode="json"), "run_id": run_id},
                config={
                    "configurable": {
                        "thread_id": make_checkpoint_thread_id(
                            request.project_name, run_id, "stage-1.1"
                        )
                    }
                },
            )
        except Exception as exc:
            postgres.record_run(
                run_id=run_id,
                project=request.project_name,
                stage="stage-1.1",
                status="failed",
                payload={"error_type": exc.__class__.__name__},
            )
            raise
    result = IngestionResult(
        project=request.project_name,
        run_id=run_id,
        artifact_dir=Path(final["artifact_dir"]),
        manifest_path=Path(final["manifest_path"]),
        chunk_ids=final["chunk_ids"],
    )
    postgres.record_run(
        run_id=run_id,
        project=request.project_name,
        stage="stage-1.1",
        status="completed",
        payload=result.model_dump(mode="json"),
    )
    return result


__all__ = ["build_ingestion_graph", "run_ingestion"]
