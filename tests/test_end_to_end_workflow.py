"""Local-adapter end-to-end test through behavioral scenario generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    LLMAcceptanceCriterion,
    LLMRequirementCandidate,
    LLMTestScenarioCandidate,
    LLMUserStoryCandidate,
    ProgressReport,
    RequirementDiscoveryChunkResponse,
    ScenarioContext,
    StageRequest,
    StoryContext,
    TestScenarioGenerationResponse,
    UserStoryGenerationResponse,
    UserStoryStatement,
)
from multi_agentic_graph_rag.workflows.ingestion_graph import (
    _Runtime as IngestionRuntime,
)
from multi_agentic_graph_rag.workflows.ingestion_graph import build_ingestion_graph
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    _Runtime as DiscoveryRuntime,
)
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    build_requirement_discovery_graph,
)
from multi_agentic_graph_rag.workflows.test_scenario_graph import (
    _Runtime as ScenarioRuntime,
)
from multi_agentic_graph_rag.workflows.test_scenario_graph import (
    build_test_scenario_graph,
)
from multi_agentic_graph_rag.workflows.user_story_graph import (
    _Runtime as StoryRuntime,
)
from multi_agentic_graph_rag.workflows.user_story_graph import build_user_story_graph


class _Embedding:
    provider_name = "test"
    embedding_fingerprint = "test:3"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


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


class _DiscoveryAgent:
    def discover(self, chunk: Any) -> RequirementDiscoveryChunkResponse:
        return RequirementDiscoveryChunkResponse(
            chunk_id=chunk.chunk_id,
            requirements=[
                LLMRequirementCandidate(
                    requirement_ref="req_1",
                    source_req_id=None,
                    source_req_id_type="generated",
                    confidence=0.95,
                    requirement_type="Functional Requirement",
                    priority="Medium",
                    requirement_text=chunk.chunk_text,
                    constraints=[],
                    entity_refs=[],
                    relationship_refs=[],
                    evidence_quotes=[chunk.chunk_text],
                )
            ],
            entities=[],
            relationships=[],
        )


class _Retrieval:
    def __init__(self, neo4j: Neo4jStore, chroma: _Chroma) -> None:
        self.neo4j = neo4j
        self.chroma = chroma

    def story_context(self, *, project: str, manifest: Any, requirement: Any) -> StoryContext:
        return StoryContext(
            requirement_id=requirement.requirement_id,
            requirement_text=requirement.requirement_text,
            source_req_id=requirement.source_req_id,
            source_req_id_type=requirement.source_req_id_type,
            authoritative_evidence_chunk_ids=[item.chunk_id for item in requirement.evidence],
            mapped_entity_ids=requirement.entity_ids,
            mapped_relationship_ids=requirement.relationship_ids,
            ranked_evidence=[],
            retrieval_parameters={"project": project, "manifest_run_id": manifest.run_id},
        )

    def scenario_context(
        self,
        *,
        project: str,
        manifest: Any,
        story: Any,
        requirements: list[Any],
    ) -> ScenarioContext:
        return ScenarioContext(
            story_id=story.story_id,
            requirement_ids=story.requirement_ids,
            story_text=story.user_story.i_want,
            acceptance_criteria=story.acceptance_criteria,
            source_req_id=story.source_req_id,
            source_req_id_type=story.source_req_id_type,
            authoritative_evidence_chunk_ids=story.traceability.evidence_chunk_ids,
            supporting_entity_ids=story.traceability.entity_ids,
            supporting_relationship_ids=story.traceability.relationship_ids,
            ranked_evidence=[],
            retrieval_parameters={"project": project, "manifest_run_id": manifest.run_id},
        )


class _StoryAgent:
    def generate(self, requirement: Any, context: StoryContext) -> UserStoryGenerationResponse:
        return UserStoryGenerationResponse(
            requirement_id=requirement.requirement_id,
            user_stories=[
                LLMUserStoryCandidate(
                    story_ref="story_1",
                    source_req_id=requirement.source_req_id,
                    source_req_id_type=requirement.source_req_id_type,
                    title="Retain audit events",
                    priority=requirement.priority,
                    persona="Auditor",
                    user_story=UserStoryStatement(
                        as_a="auditor",
                        i_want="audit events retained",
                        so_that="activity can be investigated",
                    ),
                    acceptance_criteria=[
                        LLMAcceptanceCriterion(
                            title="Event retained",
                            given="An audit event is produced",
                            when="The event is recorded",
                            then="The event remains available",
                        )
                    ],
                    business_rules=[],
                    evidence_chunk_ids=context.authoritative_evidence_chunk_ids,
                    supporting_entity_ids=[],
                    supporting_relationship_ids=[],
                    confidence=0.94,
                )
            ],
        )


class _ScenarioAgent:
    def generate(self, story: Any, context: ScenarioContext) -> TestScenarioGenerationResponse:
        return TestScenarioGenerationResponse(
            story_id=story.story_id,
            requirement_ids=story.requirement_ids,
            test_scenarios=[
                LLMTestScenarioCandidate(
                    scenario_ref="scenario_1",
                    source_req_id=story.source_req_id,
                    source_req_id_type=story.source_req_id_type,
                    title="Retain one audit event",
                    description="Verify the retention behavior.",
                    scenario_type="Positive",
                    priority=story.priority,
                    preconditions=["An audit event can be produced"],
                    action="Record an audit event",
                    expected_result="The audit event remains available",
                    covered_acceptance_criterion_ids=[story.acceptance_criteria[0].criterion_id],
                    evidence_chunk_ids=context.authoritative_evidence_chunk_ids,
                    supporting_entity_ids=[],
                    supporting_relationship_ids=[],
                    confidence=0.96,
                )
            ],
        )


def test_all_four_graphs_publish_traceable_current_run_artifacts(tmp_path: Path) -> None:
    settings = load_config()
    settings.paths.generated_dir = tmp_path / "generated"
    settings.postgres.mode = "local_json"
    settings.postgres.local_path = tmp_path / "postgres.jsonl"
    settings.neo4j.mode = "local_json"
    settings.neo4j.local_path = tmp_path / "neo4j.jsonl"
    postgres = PostgresStore(settings)
    neo4j = Neo4jStore(settings)
    chroma = _Chroma()
    checkpointer = InMemorySaver()
    source = tmp_path / "requirements.md"
    source.write_text("The service shall retain audit events.", encoding="utf-8")
    project = "alpha"
    run_id = "RUN-E2E"

    ingestion = build_ingestion_graph(
        IngestionRuntime(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            chroma=chroma,  # type: ignore[arg-type]
            embedding=_Embedding(),
            checkpointer=checkpointer,
        )
    ).invoke(
        {
            "request": {
                "project_name": project,
                "source_file": str(source),
                "embedding_provider": None,
            },
            "run_id": run_id,
        },
        config={"configurable": {"thread_id": "e2e-stage-1.1"}},
    )
    assert Path(ingestion["manifest_path"]).exists()

    request = StageRequest(project_name=project, run_id=run_id)
    discovery = build_requirement_discovery_graph(
        DiscoveryRuntime(
            settings=settings,
            postgres=postgres,
            neo4j=neo4j,
            agent=_DiscoveryAgent(),  # type: ignore[arg-type]
            checkpointer=checkpointer,
        )
    ).invoke(
        {"request": request.model_dump(mode="json")},
        config={"configurable": {"thread_id": "e2e-stage-1.2"}},
    )
    assert discovery["requirement_ids"]

    retrieval = _Retrieval(neo4j, chroma)
    stories = build_user_story_graph(
        StoryRuntime(
            settings=settings,
            postgres=postgres,
            retrieval=retrieval,  # type: ignore[arg-type]
            agent=_StoryAgent(),  # type: ignore[arg-type]
            checkpointer=checkpointer,
        )
    ).invoke(
        {"request": request.model_dump(mode="json")},
        config={"configurable": {"thread_id": "e2e-stage-2"}},
    )
    assert stories["story_ids"]

    scenarios = build_test_scenario_graph(
        ScenarioRuntime(
            settings=settings,
            postgres=postgres,
            retrieval=retrieval,  # type: ignore[arg-type]
            agent=_ScenarioAgent(),  # type: ignore[arg-type]
            checkpointer=checkpointer,
        )
    ).invoke(
        {"request": request.model_dump(mode="json")},
        config={"configurable": {"thread_id": "e2e-stage-3"}},
    )
    assert scenarios["scenario_ids"]

    story_progress = Path(stories["artifact_path"]).parent / "progress_story.json"
    scenario_progress = Path(scenarios["artifact_path"]).parent / "progress_scenario.json"
    assert story_progress.exists()
    assert scenario_progress.exists()
    story_report = ProgressReport.model_validate_json(story_progress.read_text(encoding="utf-8"))
    scenario_report = ProgressReport.model_validate_json(
        scenario_progress.read_text(encoding="utf-8")
    )
    assert story_report.stage == "user_story"
    assert [item.status for item in story_report.items] == ["generated"]
    assert scenario_report.stage == "test_scenario"
    assert [item.status for item in scenario_report.items] == ["generated"]
    # Diagnostic-only: deleting them must not affect the durable artifacts.
    story_progress.unlink()
    scenario_progress.unlink()

    coverage = postgres.coverage(project, run_id)
    assert coverage.requirement_count == 1
    assert coverage.story_count == 1
    assert coverage.scenario_count == 1
    assert coverage.requirements_with_stories == 1
    assert coverage.stories_with_scenarios == 1
