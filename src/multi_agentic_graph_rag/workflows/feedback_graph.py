"""LangGraph orchestration for the standalone human-feedback (HFIL) stage.

Add-only v1: a reviewer comment yields either additional user-story/test-scenario
records (grounded in retrieved chunk evidence) appended to the target artifact, or a
structured, recorded decline. Both outcomes land in the ``feedback_events`` ledger; a
decline is a valid business outcome (exit 0), not an error.

This stage never mutates the batch generation graphs; it reuses their generation agents
via an optional ``reviewer_directive`` prompt parameter.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.agents.feedback_agent import FeedbackGateAgent
from multi_agentic_graph_rag.agents.test_scenario_agent import TestScenarioGenerationAgent
from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGenerationAgent
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.identifiers import feedback_id as make_feedback_id
from multi_agentic_graph_rag.domain.schemas import (
    FeedbackRequest,
    FeedbackResult,
    RequirementInput,
    TestScenarioArtifact,
    UserStoryArtifact,
    UserStoryRecord,
)
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)
from multi_agentic_graph_rag.observability.session import (
    RunSession,
    command_run_id,
    command_session,
)
from multi_agentic_graph_rag.services.feedback_anchor import (
    REQUIREMENT_ID_RE,
    STORY_ID_RE,
    resolve_anchor,
)
from multi_agentic_graph_rag.services.feedback_builder import (
    atomic_rewrite_json,
    build_test_scenario_feedback_records,
    build_user_story_feedback_records,
    existing_test_scenario_titles,
    existing_user_story_titles,
    find_duplicate_titles,
    merge_test_scenarios,
    merge_user_stories,
    normalize_title,
    test_scenario_delta_artifact,
    user_story_delta_artifact,
)
from multi_agentic_graph_rag.services.feedback_intent import classify_destructive_intent
from multi_agentic_graph_rag.services.requirement_source import (
    RequirementSource,
    load_requirement_source_from_full_payload,
    load_requirement_source_local,
)
from multi_agentic_graph_rag.services.retrieval import RetrievalService, RetrievedContext


class FeedbackState(TypedDict, total=False):
    route: str  # "continue" | "decline"


def run_feedback(
    request: FeedbackRequest,
    session: RunSession | None = None,
) -> FeedbackResult:
    if session is None:
        project, version = resolve_feedback_identity(request)
        with command_session(
            project=project,
            version=version,
            command="feedback",
            run_id=command_run_id("feedback"),
        ) as managed_session:
            return run_feedback(request, session=managed_session)
    session.request_payload = request.model_dump(mode="json")
    pipeline = _FeedbackPipeline(request, session)
    graph = _build_feedback_graph(pipeline)
    try:
        graph.invoke({"route": "continue"})
    except Exception as exc:
        if session.logger is not None:
            session.logger.exception(
                "Feedback pipeline failed", step="feedback", exc=exc, status="failed"
            )
        session.write_failure_envelope(error=exc)
        raise
    assert pipeline.result is not None
    return pipeline.result


def _build_feedback_graph(pipeline: _FeedbackPipeline) -> Any:
    graph = StateGraph(FeedbackState)
    graph.add_node("validate_request", pipeline.validate_request)
    graph.add_node("load_artifacts", pipeline.load_artifacts)
    graph.add_node("guard_intent", pipeline.guard_intent)
    graph.add_node("resolve_anchor", pipeline.resolve_anchor)
    graph.add_node("retrieve_context", pipeline.retrieve_context)
    graph.add_node("gate_validate", pipeline.gate_validate)
    graph.add_node("generate", pipeline.generate)
    graph.add_node("merge_and_persist", pipeline.merge_and_persist)
    graph.add_node("project_delta", pipeline.project_delta)
    graph.add_node("record_decline", pipeline.record_decline)

    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "load_artifacts")
    graph.add_edge("load_artifacts", "guard_intent")
    graph.add_conditional_edges(
        "guard_intent", _route, {"continue": "resolve_anchor", "decline": "record_decline"}
    )
    graph.add_conditional_edges(
        "resolve_anchor", _route, {"continue": "retrieve_context", "decline": "record_decline"}
    )
    graph.add_edge("retrieve_context", "gate_validate")
    graph.add_conditional_edges(
        "gate_validate", _route, {"continue": "generate", "decline": "record_decline"}
    )
    graph.add_conditional_edges(
        "generate", _route, {"continue": "merge_and_persist", "decline": "record_decline"}
    )
    graph.add_edge("merge_and_persist", "project_delta")
    graph.add_edge("project_delta", END)
    graph.add_edge("record_decline", END)
    return graph.compile()


def _route(state: FeedbackState) -> str:
    return state.get("route", "continue")


class _FeedbackPipeline:
    """Holds all per-run state; nodes mutate instance attributes, state carries routing."""

    def __init__(self, request: FeedbackRequest, session: RunSession) -> None:
        self.request = request
        self.session = session
        self.logger = session.logger
        self.settings: AppSettings = load_config()
        session.set_log_level(self.settings.log_level)
        _apply_overrides(self.settings, request)
        self.postgres = PostgresStore(self.settings)
        self.neo4j = Neo4jStore(self.settings)
        self.chroma = ChromaStore(self.settings)
        self.run_id = session.run_id
        self.result: FeedbackResult | None = None

        # Populated across nodes.
        self.user_story_artifact: UserStoryArtifact | None = None
        self.scenario_artifact: TestScenarioArtifact | None = None
        self.requirement_source: RequirementSource | None = None
        self.project = ""
        self.document_version_id = ""
        self.anchor_requirement_id: str | None = None
        self.anchor_story_id: str | None = None
        self.anchor_requirement: RequirementInput | None = None
        self.anchor_story: UserStoryRecord | None = None
        self.feedback_id = ""
        self.gate_context: RetrievedContext | None = None
        self.approved_chunk_ids: list[str] = []
        self.generated_records: list[Any] = []
        self.decline_reason = ""
        self._gate_output_payload: dict[str, Any] | None = None

    # -- nodes ------------------------------------------------------------------

    def validate_request(self, state: FeedbackState) -> FeedbackState:
        _validate_required_feedback_stack(self.settings)
        if not self.request.artifact_path.exists():
            raise FileNotFoundError(self.request.artifact_path)
        self._check_store("check_postgres", self.postgres.check)
        self._check_store("check_neo4j", self.neo4j.check)
        self._check_store("check_chroma", self.chroma.check)
        self.postgres.ensure_schema()
        self.neo4j.ensure_search_index()
        return {"route": "continue"}

    def load_artifacts(self, state: FeedbackState) -> FeedbackState:
        data = json.loads(self.request.artifact_path.read_text(encoding="utf-8"))
        if self.request.stage == "user_story":
            self.user_story_artifact = UserStoryArtifact.model_validate(data)
            self.project = self.user_story_artifact.project
            self.document_version_id = self.user_story_artifact.document_version_id
        else:
            self.scenario_artifact = TestScenarioArtifact.model_validate(data)
            self.project = self.scenario_artifact.project
            self.document_version_id = self.scenario_artifact.document_version_id
            self.user_story_artifact = self._load_user_stories_for_scenarios()
        self.requirement_source = self._load_requirement_source()
        return {"route": "continue"}

    def guard_intent(self, state: FeedbackState) -> FeedbackState:
        reason = classify_destructive_intent(self.request.comment)
        if reason is not None:
            return self._decline(reason)
        return {"route": "continue"}

    def resolve_anchor(self, state: FeedbackState) -> FeedbackState:
        if self.request.stage == "user_story":
            resolution = self._resolve_requirement_anchor()
        else:
            resolution = self._resolve_story_anchor()
        if resolution is None:
            return {"route": self._current_route()}
        # feedback_id is deterministic on the resolved anchor (idempotency key).
        anchor_id = self.anchor_requirement_id or self.anchor_story_id or ""
        self.feedback_id = make_feedback_id(
            self.project,
            self.document_version_id,
            self.request.stage,
            normalize_title(self.request.comment),
            anchor_id,
        )
        replay = self._maybe_replay_existing()
        if replay is not None:
            self.result = replay
            return {"route": "decline"}  # short-circuit to END via record_decline no-op
        return {"route": "continue"}

    def retrieve_context(self, state: FeedbackState) -> FeedbackState:
        assert self.anchor_requirement is not None
        seed = list(self.anchor_requirement.evidence_chunk_ids)
        query = _gate_query(
            self.request.comment,
            self.anchor_requirement.requirement_text,
            self.anchor_story.title if self.anchor_story is not None else None,
        )
        self.gate_context = self._retrieval().retrieve_context(
            requirement_text=query,
            document_version_id=self.document_version_id,
            evidence_chunk_ids=seed,
        )
        return {"route": "continue"}

    def gate_validate(self, state: FeedbackState) -> FeedbackState:
        assert self.anchor_requirement is not None and self.gate_context is not None
        gate = FeedbackGateAgent(self._reasoning_model(), logger=self.logger)
        output = gate.gate(
            comment=self.request.comment,
            anchor_requirement_text=self.anchor_requirement.requirement_text,
            anchor_story_text=self.anchor_story.title if self.anchor_story else None,
            context=self.gate_context,
        )
        self._gate_output_payload = output.model_dump(mode="json")
        if output.verdict == "decline":
            return self._decline(f"gate declined: {output.reason}")
        self.approved_chunk_ids = list(output.supporting_chunk_ids)
        return {"route": "continue"}

    def generate(self, state: FeedbackState) -> FeedbackState:
        assert self.anchor_requirement is not None and self.gate_context is not None
        filtered = _filter_context(self.gate_context, set(self.approved_chunk_ids))
        if self.request.stage == "user_story":
            records = self._generate_user_stories(filtered)
        else:
            records = self._generate_test_scenarios(filtered)
        if not records:
            return self._decline("all generated items duplicated existing records")
        self.generated_records = records
        return {"route": "continue"}

    def merge_and_persist(self, state: FeedbackState) -> FeedbackState:
        if self.request.stage == "user_story":
            self._merge_and_persist_user_stories()
        else:
            self._merge_and_persist_test_scenarios()
        created = [self._record_id(record) for record in self.generated_records]
        self.postgres.persist_feedback_event(
            feedback_id=self.feedback_id,
            project=self.project,
            document_version_id=self.document_version_id,
            stage=self.request.stage,
            anchor_requirement_id=self.anchor_requirement_id,
            anchor_story_id=self.anchor_story_id,
            comment_text=self.request.comment,
            verdict="applied",
            reason="grounded feedback applied",
            created_ids=created,
            run_id=self.run_id,
        )
        self.result = FeedbackResult(
            run_id=self.run_id,
            feedback_id=self.feedback_id,
            status="applied",
            stage=self.request.stage,
            project=self.project,
            document_version_id=self.document_version_id,
            verdict_reason="grounded feedback applied",
            anchor_requirement_id=self.anchor_requirement_id,
            anchor_story_id=self.anchor_story_id,
            created_ids=created,
            artifact_path=self.request.artifact_path,
        )
        self._write_delta(created)
        return {"route": "continue"}

    def project_delta(self, state: FeedbackState) -> FeedbackState:
        evidence = self._delta_evidence()
        if self.request.stage == "user_story":
            assert self.user_story_artifact is not None
            us_delta = user_story_delta_artifact(self.user_story_artifact, self.generated_records)
            self.neo4j.project_user_story_coverage(us_delta, evidence)
        else:
            assert self.scenario_artifact is not None
            ts_delta = test_scenario_delta_artifact(self.scenario_artifact, self.generated_records)
            self.neo4j.project_test_scenario_coverage(ts_delta, evidence)
        if self.logger is not None:
            self.logger.info(
                "Projected feedback delta into Neo4j",
                step="feedback.project_delta",
                stage=self.request.stage,
                created_count=len(self.generated_records),
                status="completed",
            )
        return {"route": "continue"}

    def record_decline(self, state: FeedbackState) -> FeedbackState:
        if self.result is not None:
            # Idempotent-replay short-circuit already produced a result.
            return {"route": "decline"}
        self.postgres.persist_feedback_event(
            feedback_id=self.feedback_id or self._pre_anchor_feedback_id(),
            project=self.project,
            document_version_id=self.document_version_id,
            stage=self.request.stage,
            anchor_requirement_id=self.anchor_requirement_id,
            anchor_story_id=self.anchor_story_id,
            comment_text=self.request.comment,
            verdict="declined",
            reason=self.decline_reason,
            created_ids=[],
            run_id=self.run_id,
        )
        self.result = FeedbackResult(
            run_id=self.run_id,
            feedback_id=self.feedback_id or self._pre_anchor_feedback_id(),
            status="declined",
            stage=self.request.stage,
            project=self.project,
            document_version_id=self.document_version_id,
            verdict_reason=self.decline_reason,
            anchor_requirement_id=self.anchor_requirement_id,
            anchor_story_id=self.anchor_story_id,
            created_ids=[],
            artifact_path=self.request.artifact_path,
        )
        self._write_delta([])
        return {"route": "decline"}

    # -- helpers ----------------------------------------------------------------

    def _decline(self, reason: str) -> FeedbackState:
        self.decline_reason = reason
        if self.logger is not None:
            self.logger.info(
                "Feedback declined", step="feedback.decline", reason=reason, status="declined"
            )
        return {"route": "decline"}

    def _current_route(self) -> str:
        return "decline" if self.decline_reason or self.result is not None else "continue"

    def _resolve_requirement_anchor(self) -> str | None:
        assert self.requirement_source is not None
        known = {
            requirement.requirement_id: requirement.requirement_text
            for requirement in self.requirement_source.requirements
        }
        candidates = self._retrieval_anchor_candidates("user_story")
        resolution = resolve_anchor(
            explicit_id=self.request.requirement_id,
            comment=self.request.comment,
            id_pattern=REQUIREMENT_ID_RE,
            known_titles=known,
            retrieval_candidates=candidates,
        )
        if resolution.anchor_id is None:
            self._decline(resolution.decline_reason or "could not resolve requirement anchor")
            return None
        self.anchor_requirement_id = resolution.anchor_id
        self.anchor_requirement = next(
            requirement
            for requirement in self.requirement_source.requirements
            if requirement.requirement_id == resolution.anchor_id
        )
        return resolution.anchor_id

    def _resolve_story_anchor(self) -> str | None:
        assert self.user_story_artifact is not None and self.requirement_source is not None
        known = {
            story_id: record.title for story_id, record in self.user_story_artifact.stories.items()
        }
        candidates = self._retrieval_anchor_candidates("test_scenario")
        resolution = resolve_anchor(
            explicit_id=self.request.story_id,
            comment=self.request.comment,
            id_pattern=STORY_ID_RE,
            known_titles=known,
            retrieval_candidates=candidates,
        )
        if resolution.anchor_id is None:
            self._decline(resolution.decline_reason or "could not resolve story anchor")
            return None
        self.anchor_story_id = resolution.anchor_id
        self.anchor_story = self.user_story_artifact.stories[resolution.anchor_id]
        self.anchor_requirement_id = self.anchor_story.requirement_id
        by_id = {
            requirement.requirement_id: requirement
            for requirement in self.requirement_source.requirements
        }
        anchor = by_id.get(self.anchor_story.requirement_id)
        if anchor is None:
            self._decline(
                f"could not load requirement {self.anchor_story.requirement_id} for story anchor"
            )
            return None
        self.anchor_requirement = anchor
        return resolution.anchor_id

    def _retrieval_anchor_candidates(self, stage: str) -> list[tuple[str, int]]:
        # Only needed for the retrieval rung; skip the model call when a flag/comment id
        # will resolve the anchor, but running it is cheap and deterministic here.
        context = self._retrieval().retrieve_context(
            requirement_text=self.request.comment,
            document_version_id=self.document_version_id,
            evidence_chunk_ids=[],
        )
        chunk_ids = [chunk.chunk_id for chunk in context.chunks]
        if not chunk_ids:
            return []
        return self.neo4j.resolve_feedback_anchors(chunk_ids, self.document_version_id, stage)

    def _generate_user_stories(self, context: RetrievedContext) -> list[Any]:
        assert self.anchor_requirement is not None and self.user_story_artifact is not None
        agent = UserStoryGenerationAgent(self._reasoning_model(), logger=self.logger)
        output = agent.generate(
            self.anchor_requirement, context, reviewer_directive=self.request.comment
        )
        existing = existing_user_story_titles(
            self.user_story_artifact, self.anchor_requirement.requirement_id
        )
        generated = list(output.user_stories)
        duplicates = find_duplicate_titles(existing, [story.title for story in generated])
        if duplicates:
            self._decline(f"duplicate user story title(s): {', '.join(duplicates)}")
            return []
        ordinal_start = len(
            self.user_story_artifact.coverage.get(self.anchor_requirement.requirement_id, [])
        )
        return build_user_story_feedback_records(
            project=self.project,
            document_id=self.user_story_artifact.document_id,
            document_version_id=self.document_version_id,
            doc_version=self.user_story_artifact.doc_version,
            requirement=self.anchor_requirement,
            ordinal_start=ordinal_start,
            generated=generated,
            feedback_id_value=self.feedback_id,
            evidence_chunk_ids=self.approved_chunk_ids,
        )

    def _generate_test_scenarios(self, context: RetrievedContext) -> list[Any]:
        assert (
            self.anchor_story is not None
            and self.anchor_requirement is not None
            and self.scenario_artifact is not None
        )
        agent = TestScenarioGenerationAgent(self._reasoning_model(), logger=self.logger)
        output = agent.generate(
            self.anchor_story,
            context,
            requirement_text=self.anchor_requirement.requirement_text,
            reviewer_directive=self.request.comment,
        )
        existing = existing_test_scenario_titles(self.scenario_artifact, self.anchor_story.story_id)
        generated = list(output.test_scenarios)
        duplicates = find_duplicate_titles(existing, [scenario.title for scenario in generated])
        if duplicates:
            self._decline(f"duplicate test scenario title(s): {', '.join(duplicates)}")
            return []
        ordinal_start = len(self.scenario_artifact.coverage.get(self.anchor_story.story_id, []))
        return build_test_scenario_feedback_records(
            project=self.project,
            document_id=self.scenario_artifact.document_id,
            document_version_id=self.document_version_id,
            doc_version=self.scenario_artifact.doc_version,
            story=self.anchor_story,
            ordinal_start=ordinal_start,
            generated=generated,
            feedback_id_value=self.feedback_id,
            evidence_chunk_ids=self.approved_chunk_ids,
        )

    def _merge_and_persist_user_stories(self) -> None:
        assert self.user_story_artifact is not None
        merged = merge_user_stories(self.user_story_artifact, self.generated_records)
        atomic_rewrite_json(self.request.artifact_path, merged.model_dump(mode="json"))
        self.postgres.persist_user_story_artifact(
            merged, str(self.request.artifact_path), self.run_id
        )
        self.user_story_artifact = merged

    def _merge_and_persist_test_scenarios(self) -> None:
        assert self.scenario_artifact is not None
        merged = merge_test_scenarios(self.scenario_artifact, self.generated_records)
        atomic_rewrite_json(self.request.artifact_path, merged.model_dump(mode="json"))
        self.postgres.persist_test_scenario_artifact(
            merged, str(self.request.artifact_path), self.run_id
        )
        self.scenario_artifact = merged

    def _delta_evidence(self) -> dict[str, list[str]]:
        seed = list(self.anchor_requirement.evidence_chunk_ids) if self.anchor_requirement else []
        combined: list[str] = []
        for chunk_id in [*seed, *self.approved_chunk_ids]:
            if chunk_id and chunk_id not in combined:
                combined.append(chunk_id)
        assert self.anchor_requirement_id is not None
        return {self.anchor_requirement_id: combined}

    def _maybe_replay_existing(self) -> FeedbackResult | None:
        existing = self.postgres.load_feedback_event(self.feedback_id)
        if existing is None:
            return None
        if self.logger is not None:
            self.logger.info(
                "Idempotent feedback replay; returning prior outcome",
                step="feedback.replay",
                feedback_id=self.feedback_id,
                verdict=existing.get("verdict"),
                status="replayed",
            )
        status = "applied" if existing.get("verdict") == "applied" else "declined"
        return FeedbackResult(
            run_id=self.run_id,
            feedback_id=self.feedback_id,
            status=status,  # type: ignore[arg-type]
            stage=self.request.stage,
            project=self.project,
            document_version_id=self.document_version_id,
            verdict_reason=str(existing.get("reason", "")),
            anchor_requirement_id=self.anchor_requirement_id,
            anchor_story_id=self.anchor_story_id,
            created_ids=list(existing.get("created_ids", []) or []),
            artifact_path=self.request.artifact_path,
            warnings=["idempotent replay: no new records created"],
        )

    def _write_delta(self, created_ids: list[str]) -> None:
        payload: dict[str, Any] = {
            "feedback_id": self.feedback_id or self._pre_anchor_feedback_id(),
            "stage": self.request.stage,
            "status": self.result.status if self.result else "declined",
            "anchor_requirement_id": self.anchor_requirement_id,
            "anchor_story_id": self.anchor_story_id,
            "created_ids": created_ids,
            "gate_output": getattr(self, "_gate_output_payload", None),
            "decline_reason": self.decline_reason or None,
            "records": [record.model_dump(mode="json") for record in self.generated_records],
        }
        (self.session.run_dir / "feedback_delta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _pre_anchor_feedback_id(self) -> str:
        return make_feedback_id(
            self.project,
            self.document_version_id,
            self.request.stage,
            normalize_title(self.request.comment),
            "",
        )

    def _load_user_stories_for_scenarios(self) -> UserStoryArtifact:
        candidate = self.request.user_stories_path or (
            self.request.artifact_path.parent / "user_stories.json"
        )
        if not candidate.exists():
            raise ConfigurationError(
                "test-scenario feedback requires the user-story artifact; pass --user-stories "
                f"or place user_stories.json beside the scenario artifact (looked at {candidate})"
            )
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return UserStoryArtifact.model_validate(data)

    def _load_requirement_source(self) -> RequirementSource:
        directories = [self.request.artifact_path.parent]
        if self.request.user_stories_path is not None:
            directories.append(self.request.user_stories_path.parent)
        for directory in directories:
            for name in ("requirements.json", "requirements_full.json"):
                candidate = directory / name
                if candidate.exists():
                    if name == "requirements.json":
                        return load_requirement_source_local(candidate)
                    return load_requirement_source_from_full_payload(
                        json.loads(candidate.read_text(encoding="utf-8"))
                    )
        payload = self.postgres.load_requirement_artifact_payload(
            document_version_id=self.document_version_id
        )
        if payload is None:
            raise ConfigurationError(
                "could not locate requirements for feedback grounding "
                f"(document_version_id={self.document_version_id})"
            )
        return load_requirement_source_from_full_payload(payload)

    def _retrieval(self) -> RetrievalService:
        return RetrievalService(
            chroma=self.chroma,
            neo4j=self.neo4j,
            embedding_model=create_embedding_model(self.settings),
            reranker_model=create_reranker_model(self.settings),
            settings=self.settings.user_story,
            logger=self.logger,
        )

    def _reasoning_model(self) -> Any:
        model = create_reasoning_model(
            self.settings, logger=self.logger, run_dir=self.session.run_dir
        )
        warmup = getattr(model, "warmup", None)
        if callable(warmup):
            warmup()
        return model

    def _record_id(self, record: Any) -> str:
        return str(getattr(record, "story_id", None) or record.scenario_id)

    def _check_store(self, step: str, check: Any) -> None:
        detail = check()
        if self.logger is not None:
            self.logger.info(detail, step=step, status="PASS", detail=detail)


def _gate_query(comment: str, requirement_text: str, story_title: str | None) -> str:
    parts = [comment, requirement_text]
    if story_title:
        parts.append(story_title)
    return "\n".join(parts)


def _filter_context(context: RetrievedContext, allowed: set[str]) -> RetrievedContext:
    return RetrievedContext(
        chunks=[chunk for chunk in context.chunks if chunk.chunk_id in allowed],
        source=context.source,
    )


def _apply_overrides(settings: AppSettings, request: FeedbackRequest) -> None:
    if request.reasoning_provider:
        settings.reasoning_model.provider = request.reasoning_provider
    if request.embedding_provider:
        settings.embedding_model.provider = request.embedding_provider
    if request.reranker_provider:
        settings.reranker_model.provider = request.reranker_provider
    if request.top_k is not None and request.top_k > 0:
        settings.user_story.top_k = request.top_k


def _validate_required_feedback_stack(settings: AppSettings) -> None:
    if settings.reasoning_model.provider in {"local_heuristic"}:
        raise ConfigurationError(
            f"REASONING_MODEL_PROVIDER={settings.reasoning_model.provider} "
            "is not valid for feedback generation"
        )
    if settings.embedding_model.provider in {"local_hash"}:
        raise ConfigurationError(
            f"EMBEDDING_MODEL_PROVIDER={settings.embedding_model.provider} "
            "is not valid for feedback generation"
        )
    if settings.reranker_model.provider in {"none"}:
        raise ConfigurationError(
            f"RERANKER_MODEL_PROVIDER={settings.reranker_model.provider} "
            "is not valid for feedback generation"
        )
    if settings.postgres.mode != "postgres":
        raise ConfigurationError("POSTGRES_MODE=postgres is required for feedback generation")
    if settings.neo4j.mode != "neo4j":
        raise ConfigurationError("NEO4J_MODE=neo4j is required for feedback generation")


def resolve_feedback_identity(request: FeedbackRequest) -> tuple[str, str]:
    if not request.artifact_path.exists():
        raise FileNotFoundError(request.artifact_path)
    data = json.loads(request.artifact_path.read_text(encoding="utf-8"))
    project = str(data.get("project") or "")
    version = str(data.get("doc_version") or "generated")
    if not project:
        raise ConfigurationError("target artifact is missing a project field")
    return project, version
