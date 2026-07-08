"""HFIL review node for test-scenario generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import interrupt

from multi_agentic_graph_rag.common_prompt_defs import FEEDBACK_STRICT_SCENARIO_PROMPT
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.identifiers import test_scenario_id
from multi_agentic_graph_rag.domain.schemas import (
    DuplicateGroup,
    TestScenarioGenerationOutput,
    TestScenarioRecord,
    UserStoryRecord,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.services.scenario_dedup import DedupEngine
from multi_agentic_graph_rag.services.semantic_matcher import SemanticMatcher

_STORY_ID_RE = re.compile(r"(?:story[-_ ]?id\s*=\s*[\"']?)?(US-[A-Z0-9-]+)", re.I)
_WORD_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class HFILRuntime:
    settings: AppSettings
    matcher: SemanticMatcher
    dedup: DedupEngine
    reasoner: ReasoningModel


def hfil_review_node(state: dict[str, Any], *, runtime: HFILRuntime) -> dict[str, Any]:
    """Run one HFIL turn.

    The node re-executes from the top on every resume. Keep code before interrupt
    deterministic and side-effect free. Do not wrap interrupt in a loop.
    """
    if state.get("hfil_done") or not state.get("hfil_enabled", False):
        return state
    payload = _interrupt_payload(state)
    user_line = interrupt(payload)
    return _process_user_line(state, str(user_line or "").strip(), runtime=runtime)


def route_hfil(state: dict[str, Any]) -> Literal["loop", "done"]:
    return "done" if state.get("hfil_done") else "loop"


def initialize_hfil_state(
    *,
    state: dict[str, Any],
    scenarios: list[TestScenarioRecord],
    user_stories: list[UserStoryRecord],
    enabled: bool,
) -> dict[str, Any]:
    next_state = dict(state)
    next_state.update(
        {
            "hfil_enabled": enabled,
            "hfil_done": not enabled,
            "hfil_phase": "review",
            "hfil_pending_prompt": _review_prompt(),
            "hfil_scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
            "hfil_user_stories": [story.model_dump(mode="json") for story in user_stories],
            "hfil_last_duplicate_groups": [],
            "hfil_messages": [],
        }
    )
    return next_state


def hfil_scenarios_from_state(state: dict[str, Any]) -> list[TestScenarioRecord]:
    return [
        TestScenarioRecord.model_validate(item)
        for item in state.get("hfil_scenarios", [])
        if isinstance(item, dict)
    ]


def _interrupt_payload(state: dict[str, Any]) -> dict[str, Any]:
    scenarios = hfil_scenarios_from_state(state)
    phase = str(state.get("hfil_phase", "review"))
    prompt = str(state.get("hfil_pending_prompt") or _review_prompt())
    return {
        "kind": "hfil_review",
        "phase": phase,
        "prompt": prompt,
        "messages": list(state.get("hfil_messages", [])),
        "scenarios": [_scenario_payload(scenario) for scenario in scenarios],
        "duplicate_groups": list(state.get("hfil_last_duplicate_groups", [])),
    }


def _process_user_line(
    state: dict[str, Any],
    user_line: str,
    *,
    runtime: HFILRuntime,
) -> dict[str, Any]:
    phase = str(state.get("hfil_phase", "review"))
    if phase == "await_duplicate_delete":
        return _apply_duplicate_delete(state, user_line)
    if phase == "await_candidate_approval":
        return _apply_candidate_approval(state, user_line, runtime=runtime)
    normalized = user_line.strip().lower()
    if normalized == "exit":
        return _with_messages(
            state,
            ["HFIL exit received; finalizing current scenario list."],
            hfil_done=True,
            hfil_phase="done",
            hfil_pending_prompt="",
        )
    if normalized == "remove duplicates":
        return _run_duplicate_detection(state, runtime=runtime)
    if not user_line.strip():
        return _with_messages(state, ["Enter feedback, `remove duplicates`, or `exit`."])
    return _handle_feedback_comment(state, user_line, runtime=runtime)


def _run_duplicate_detection(state: dict[str, Any], *, runtime: HFILRuntime) -> dict[str, Any]:
    scenarios = hfil_scenarios_from_state(state)
    groups = runtime.dedup.find_duplicates(scenarios)
    if not groups:
        return _with_messages(
            state,
            ["No duplicates found."],
            hfil_last_duplicate_groups=[],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    return _with_messages(
        state,
        ["Duplicate groups found. Reply with the scenario IDs to delete, or `none`."],
        hfil_last_duplicate_groups=[group.model_dump(mode="json") for group in groups],
        hfil_phase="await_duplicate_delete",
        hfil_pending_prompt=(
            "Which scenario IDs should be deleted? Example: "
            "`only delete SC-..., SC-...`; reply `none` to keep all."
        ),
    )


def _apply_duplicate_delete(state: dict[str, Any], user_line: str) -> dict[str, Any]:
    if user_line.strip().lower() in {"none", "cancel", "skip"}:
        return _with_messages(
            state,
            ["No duplicate deletions applied."],
            hfil_last_duplicate_groups=[],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    scenarios = hfil_scenarios_from_state(state)
    known_ids = {scenario.scenario_id for scenario in scenarios}
    delete_ids = sorted({scenario_id for scenario_id in known_ids if scenario_id in user_line})
    if not delete_ids:
        return _with_messages(
            state,
            ["No known scenario IDs were found in that reply; no deletions applied."],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    remaining = [scenario for scenario in scenarios if scenario.scenario_id not in delete_ids]
    return _with_messages(
        state,
        [f"Deleted approved duplicate IDs: {', '.join(delete_ids)}."],
        hfil_scenarios=[scenario.model_dump(mode="json") for scenario in remaining],
        hfil_last_duplicate_groups=[],
        hfil_phase="review",
        hfil_pending_prompt=_review_prompt(),
    )


def _handle_feedback_comment(
    state: dict[str, Any],
    comment: str,
    *,
    runtime: HFILRuntime,
) -> dict[str, Any]:
    stories = _stories_from_state(state)
    story_by_id = {story.story_id: story for story in stories}
    explicit_story_id = _extract_story_id(comment)
    explicit_story = story_by_id.get(explicit_story_id) if explicit_story_id else None
    best_story, raw_cosine, match_pct = runtime.matcher.match_comment_to_story(comment, stories)
    if explicit_story is not None:
        best_story = explicit_story
        match_pct = max(match_pct, runtime.settings.hfil_match_threshold_pct)
    if best_story is None:
        return _with_messages(
            state,
            ["No user stories are available for HFIL matching."],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    if match_pct < runtime.settings.hfil_out_of_context_pct:
        return _with_messages(
            state,
            [
                "The comment appears out-of-context for the available user stories "
                f"(match={match_pct:.1f}%, cosine={raw_cosine:.3f}). "
                "Provide clearer scenario feedback or `exit`."
            ],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    if match_pct < runtime.settings.hfil_match_threshold_pct:
        return _with_messages(
            state,
            [
                "The comment is related but not clear enough "
                f"(match={match_pct:.1f}%, cosine={raw_cosine:.3f}). "
                "Rewrite a clearer sentence."
            ],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    candidate = _generate_feedback_scenario(
        state,
        story=best_story,
        comment=comment,
        runtime=runtime,
    )
    duplicate_group = _candidate_duplicate_group(state, candidate, runtime=runtime)
    if duplicate_group is not None:
        return _with_messages(
            state,
            [
                "Generated feedback scenario is already covered by existing scenarios; "
                f"duplicate group: {', '.join(duplicate_group.scenario_ids)}."
            ],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    return _with_messages(
        state,
        [
            "Generated one candidate scenario. Reply `ok` to append it, or provide a "
            "refinement sentence."
        ],
        hfil_phase="await_candidate_approval",
        hfil_pending_prompt="Reply `ok` to append this scenario, or provide a refinement.",
        hfil_pending_candidate=candidate.model_dump(mode="json"),
        hfil_pending_story_id=best_story.story_id,
        hfil_pending_comment=comment,
    )


def _apply_candidate_approval(
    state: dict[str, Any],
    user_line: str,
    *,
    runtime: HFILRuntime,
) -> dict[str, Any]:
    candidate_payload = state.get("hfil_pending_candidate")
    if not isinstance(candidate_payload, dict):
        return _with_messages(
            state,
            ["No pending scenario candidate was found."],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
        )
    candidate = TestScenarioRecord.model_validate(candidate_payload)
    if user_line.strip().lower() == "ok":
        scenarios = hfil_scenarios_from_state(state)
        scenarios.append(candidate)
        return _with_messages(
            state,
            [f"Appended scenario {candidate.scenario_id}."],
            hfil_scenarios=[scenario.model_dump(mode="json") for scenario in scenarios],
            hfil_phase="review",
            hfil_pending_prompt=_review_prompt(),
            hfil_pending_candidate=None,
            hfil_pending_story_id=None,
            hfil_pending_comment=None,
        )
    story = next(
        (
            item
            for item in _stories_from_state(state)
            if item.story_id == state.get("hfil_pending_story_id")
        ),
        None,
    )
    if story is None:
        return _with_messages(state, ["Pending candidate story no longer exists."])
    refined_comment = f"{state.get('hfil_pending_comment', '')}\nRefinement: {user_line}"
    refined_candidate = _generate_feedback_scenario(
        state,
        story=story,
        comment=refined_comment,
        runtime=runtime,
    )
    return _with_messages(
        state,
        ["Regenerated candidate scenario from refinement. Reply `ok` to append it."],
        hfil_phase="await_candidate_approval",
        hfil_pending_prompt="Reply `ok` to append this scenario, or provide another refinement.",
        hfil_pending_candidate=refined_candidate.model_dump(mode="json"),
        hfil_pending_comment=refined_comment,
    )


def _generate_feedback_scenario(
    state: dict[str, Any],
    *,
    story: UserStoryRecord,
    comment: str,
    runtime: HFILRuntime,
) -> TestScenarioRecord:
    prompt = (
        f"{FEEDBACK_STRICT_SCENARIO_PROMPT}\n\n"
        f"Feedback comment:\n{comment}\n\n"
        f"Matched user_story:\n{json.dumps(story.model_dump(mode='json'), indent=2)}\n"
    )
    output = runtime.reasoner.generate_structured(
        prompt=prompt,
        schema=TestScenarioGenerationOutput,
    )
    scenario = output.test_scenarios[0]
    ordinal = sum(
        1 for record in hfil_scenarios_from_state(state) if record.story_id == story.story_id
    )
    scenario_id = test_scenario_id(
        story.project,
        story.story_id,
        _normalize_title(scenario.title),
        ordinal,
    )
    return TestScenarioRecord(
        scenario_id=scenario_id,
        story_id=story.story_id,
        story_display_id=story.display_id,
        requirement_id=story.requirement_id,
        requirement_display_id=story.requirement_display_id,
        requirement_revision_id=story.requirement_revision_id,
        source_req_id=story.source_req_id,
        project=story.project,
        document_id=story.document_id,
        document_version_id=story.document_version_id,
        doc_version=story.doc_version,
        origin_version=story.origin_version or story.doc_version,
        origin="feedback",
        title=scenario.title,
        description=scenario.description,
        scenario_type=scenario.scenario_type,
        preconditions=list(scenario.preconditions),
        expected_result=scenario.expected_result,
        priority=scenario.priority,
        confidence=scenario.confidence,
        evidence_chunk_ids=list(story.evidence_chunk_ids),
    )


def _candidate_duplicate_group(
    state: dict[str, Any],
    candidate: TestScenarioRecord,
    *,
    runtime: HFILRuntime,
) -> DuplicateGroup | None:
    scenarios = [*hfil_scenarios_from_state(state), candidate]
    for group in runtime.dedup.find_duplicates(scenarios):
        if candidate.scenario_id in group.scenario_ids:
            return group
    return None


def _with_messages(
    state: dict[str, Any],
    messages: list[str],
    **updates: object,
) -> dict[str, Any]:
    next_state = dict(state)
    next_state["hfil_messages"] = [*list(state.get("hfil_messages", [])), *messages]
    for key, value in updates.items():
        if value is None:
            next_state.pop(key, None)
        else:
            next_state[key] = value
    return next_state


def _stories_from_state(state: dict[str, Any]) -> list[UserStoryRecord]:
    return [
        UserStoryRecord.model_validate(item)
        for item in state.get("hfil_user_stories", [])
        if isinstance(item, dict)
    ]


def _scenario_payload(scenario: TestScenarioRecord) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "story_id": scenario.story_id,
        "scenario_text": _scenario_text(scenario),
    }


def _scenario_text(scenario: TestScenarioRecord) -> str:
    return f"{scenario.title}: {scenario.description} Expected: {scenario.expected_result}"


def _extract_story_id(comment: str) -> str | None:
    match = _STORY_ID_RE.search(comment)
    return match.group(1).upper() if match else None


def _normalize_title(title: str) -> str:
    return _WORD_SPACE.sub(" ", title.strip().lower())


def _review_prompt() -> str:
    return "Enter feedback, `remove duplicates`, or `exit`."
