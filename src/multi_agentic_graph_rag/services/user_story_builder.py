"""Canonicalize Stage 2 candidates within provenance partitions."""

from __future__ import annotations

import re
from collections import defaultdict

from multi_agentic_graph_rag.domain.identifiers import (
    make_criterion_id,
    new_story_id,
)
from multi_agentic_graph_rag.domain.schemas import (
    AcceptanceCriterion,
    CanonicalUserStory,
    LLMAcceptanceCriterion,
    LLMUserStoryCandidate,
    Traceability,
    UserStoriesArtifact,
    canonical_checksum,
)

_SPACE = re.compile(r"\s+")


def build_user_stories_artifact(
    *,
    project: str,
    run_id: str,
    candidates: list[tuple[str, LLMUserStoryCandidate]],
    existing: UserStoriesArtifact | None = None,
) -> UserStoriesArtifact:
    """Deduplicate only inside a source-provenance partition and assign IDs."""
    existing_by_key: dict[tuple[str | None, str, str], CanonicalUserStory] = {
        (
            story.source_req_id,
            story.source_req_id_type,
            _canonical_key(story),
        ): story
        for story in (existing.stories if existing is not None else [])
    }
    grouped: dict[tuple[str | None, str, str], list[tuple[str, LLMUserStoryCandidate]]] = (
        defaultdict(list)
    )
    for requirement_id, candidate in candidates:
        candidate_key = (
            candidate.source_req_id,
            candidate.source_req_id_type,
            _canonical_key(candidate),
        )
        grouped[candidate_key].append((requirement_id, candidate))
    stories: list[CanonicalUserStory] = []
    for key, rows in grouped.items():
        representative = rows[0][1]
        prior = existing_by_key.get(key)
        story_id = prior.story_id if prior is not None else new_story_id()
        prior_criteria = {
            _criterion_key(criterion): criterion
            for criterion in (prior.acceptance_criteria if prior is not None else [])
        }
        criteria: list[AcceptanceCriterion] = []
        for index, criterion in enumerate(representative.acceptance_criteria, start=1):
            criterion_key = _criterion_key(criterion)
            criteria.append(
                prior_criteria.get(criterion_key)
                or AcceptanceCriterion(
                    criterion_id=make_criterion_id(
                        story_id,
                        index,
                        " ".join(
                            (criterion.title, criterion.given, criterion.when, criterion.then)
                        ),
                    ),
                    **criterion.model_dump(mode="json"),
                )
            )
        stories.append(
            CanonicalUserStory(
                story_id=story_id,
                requirement_ids=list(dict.fromkeys(requirement_id for requirement_id, _ in rows)),
                source_req_id=representative.source_req_id,
                source_req_id_type=representative.source_req_id_type,
                title=representative.title,
                priority=representative.priority,
                persona=representative.persona,
                user_story=representative.user_story,
                acceptance_criteria=criteria,
                business_rules=list(
                    dict.fromkeys(
                        value for _, candidate in rows for value in candidate.business_rules
                    )
                ),
                confidence=max(candidate.confidence for _, candidate in rows),
                traceability=Traceability(
                    evidence_chunk_ids=list(
                        dict.fromkeys(
                            value for _, candidate in rows for value in candidate.evidence_chunk_ids
                        )
                    ),
                    entity_ids=list(
                        dict.fromkeys(
                            value
                            for _, candidate in rows
                            for value in candidate.supporting_entity_ids
                        )
                    ),
                    relationship_ids=list(
                        dict.fromkeys(
                            value
                            for _, candidate in rows
                            for value in candidate.supporting_relationship_ids
                        )
                    ),
                ),
            )
        )
    payload = UserStoriesArtifact.model_construct(
        project=project,
        run_id=run_id,
        checksum="",
        stories=stories,
    )
    return UserStoriesArtifact.model_validate(
        {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
    )


def _canonical_key(candidate: LLMUserStoryCandidate | CanonicalUserStory) -> str:
    values = (
        candidate.title,
        candidate.persona,
        candidate.user_story.as_a,
        candidate.user_story.i_want,
        candidate.user_story.so_that,
    )
    return "|".join(_SPACE.sub(" ", value).strip().casefold() for value in values)


def _criterion_key(
    criterion: LLMAcceptanceCriterion | AcceptanceCriterion,
) -> str:
    values = (criterion.title, criterion.given, criterion.when, criterion.then)
    return "|".join(_SPACE.sub(" ", value).strip().casefold() for value in values)


__all__ = ["build_user_stories_artifact"]
