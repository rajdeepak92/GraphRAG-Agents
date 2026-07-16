"""Stage 1.2 per-chunk failure isolation and durable checkpoint resume."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import (
    make_checkpoint_thread_id,
    normalize_project,
)
from multi_agentic_graph_rag.domain.schemas import (
    ChunkLayout,
    LLMRequirementCandidate,
    ManifestChunk,
    RequirementDiscoveryChunkResponse,
    StageRequest,
)
from multi_agentic_graph_rag.services.manifest import atomic_write_model, build_chunk_manifest
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    _build_discovery_subgraph,
    build_requirement_discovery_graph,
)
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    _Runtime as DiscoveryRuntime,
)

_PROJECT = "alpha"
_RUN = "RUN-FAIL"


class _Interrupt(BaseException):
    """Simulates a hard process interruption not caught by Exception handlers."""


def _chunk(chunk_id: str, sequence_index: int, text: str) -> ManifestChunk:
    return ManifestChunk(
        chunk_id=chunk_id,
        sequence_index=sequence_index,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section=None,
            block_types=["paragraph"],
            source_location=None,
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )


def _success(chunk: ManifestChunk) -> RequirementDiscoveryChunkResponse:
    return RequirementDiscoveryChunkResponse(
        chunk_id=chunk.chunk_id,
        requirements=[
            LLMRequirementCandidate(
                requirement_ref="req_1",
                source_req_id=None,
                source_req_id_type="generated",
                requirement_text=chunk.chunk_text,
                requirement_type="Functional Requirement",
                priority="Medium",
                constraints=[],
                entity_refs=[],
                relationship_refs=[],
                evidence_quotes=[chunk.chunk_text],
                confidence=0.95,
            )
        ],
        entities=[],
        relationships=[],
    )


class _Agent:
    def __init__(
        self,
        *,
        fail_on: set[str] | None = None,
        interrupt_once_on: set[str] | None = None,
    ) -> None:
        self.fail_on = fail_on or set()
        self.interrupt_once_on = interrupt_once_on or set()
        self.calls: list[str] = []
        self._interrupted: set[str] = set()

    def discover(self, chunk: ManifestChunk) -> RequirementDiscoveryChunkResponse:
        self.calls.append(chunk.chunk_id)
        if chunk.chunk_id in self.interrupt_once_on and chunk.chunk_id not in self._interrupted:
            self._interrupted.add(chunk.chunk_id)
            raise _Interrupt
        if chunk.chunk_id in self.fail_on:
            raise ValueError("invalid combined response")
        return _success(chunk)


def _write_manifest(settings: AppSettings, chunks: list[ManifestChunk]) -> Path:
    artifact_dir = (
        settings.paths.generated_dir / normalize_project(_PROJECT) / _RUN / "requirements"
    )
    manifest = build_chunk_manifest(project=_PROJECT, run_id=_RUN, chunks=chunks)
    atomic_write_model(manifest, artifact_dir / "chunk_manifest.json")
    return artifact_dir


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.paths.generated_dir = tmp_path / "generated"
    settings.postgres.mode = "local_json"
    settings.postgres.local_path = tmp_path / "postgres.jsonl"
    settings.neo4j.mode = "local_json"
    settings.neo4j.local_path = tmp_path / "neo4j.jsonl"
    return settings


def _runtime(settings: AppSettings, agent: _Agent, checkpointer: Any) -> DiscoveryRuntime:
    return DiscoveryRuntime(
        settings=settings,
        postgres=PostgresStore(settings),
        neo4j=Neo4jStore(settings),
        agent=agent,  # type: ignore[arg-type]
        checkpointer=checkpointer,
    )


def test_failed_chunk_is_isolated_and_blocks_publication(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    chunk_a = _chunk("CHK-A", 0, "The service shall retain audit events.")
    chunk_b = _chunk("CHK-B", 1, "The gateway shall reject malformed requests.")
    artifact_dir = _write_manifest(settings, [chunk_a, chunk_b])
    agent = _Agent(fail_on={"CHK-B"})
    graph = build_requirement_discovery_graph(_runtime(settings, agent, InMemorySaver()))

    with pytest.raises(ValueError, match="CHK-B"):
        graph.invoke(
            {"request": StageRequest(project_name=_PROJECT, run_id=_RUN).model_dump(mode="json")},
            config={"configurable": {"thread_id": "fail-isolation"}},
        )

    # The sibling chunk was still processed, and publication is withheld.
    assert agent.calls == ["CHK-A", "CHK-B"]
    assert not (artifact_dir / "requirements.json").exists()


def test_interrupted_discovery_resumes_without_reprocessing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    chunk_a = _chunk("CHK-A", 0, "The service shall retain audit events.")
    chunk_b = _chunk("CHK-B", 1, "The gateway shall reject malformed requests.")
    manifest = build_chunk_manifest(project=_PROJECT, run_id=_RUN, chunks=[chunk_a, chunk_b])
    agent = _Agent(interrupt_once_on={"CHK-B"})
    checkpointer = InMemorySaver()
    subgraph = _build_discovery_subgraph(_runtime(settings, agent, checkpointer))
    thread = make_checkpoint_thread_id(_PROJECT, _RUN, "stage-1.2-chunks")
    initial = {
        "project": _PROJECT,
        "run_id": _RUN,
        "manifest": manifest.model_dump(mode="json"),
        "current_index": 0,
        "chunk_results": [],
        "entities": [],
        "relationships": [],
    }

    with pytest.raises(_Interrupt):
        subgraph.invoke(initial, config={"configurable": {"thread_id": thread}})

    # Resume from the durable checkpoint (input=None) after the simulated interruption.
    final = subgraph.invoke(None, config={"configurable": {"thread_id": thread}})

    assert len(final["chunk_results"]) == 2
    assert {result["status"] for result in final["chunk_results"]} == {"completed"}
    # CHK-A completed once and was not reprocessed; CHK-B was retried on resume.
    assert agent.calls.count("CHK-A") == 1
    assert agent.calls.count("CHK-B") == 2
