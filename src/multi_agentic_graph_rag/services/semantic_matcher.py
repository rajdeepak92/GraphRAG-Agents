"""Semantic matching for HFIL comments against generated user stories."""

from __future__ import annotations

import math

from multi_agentic_graph_rag.domain.schemas import UserStoryRecord
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel


class SemanticMatcher:
    def __init__(self, embedder: EmbeddingModel, cos_floor: float, cos_ceil: float) -> None:
        if cos_ceil <= cos_floor:
            raise ValueError("cos_ceil must be greater than cos_floor")
        self.embedder = embedder
        self.cos_floor = cos_floor
        self.cos_ceil = cos_ceil

    def match_comment_to_story(
        self,
        comment_text: str,
        user_stories: list[UserStoryRecord],
    ) -> tuple[UserStoryRecord | None, float, float]:
        """
        Return:
            best_story
            raw_cosine
            calibrated_match_pct
        """
        if not comment_text.strip() or not user_stories:
            return None, 0.0, 0.0
        documents = [comment_text, *[_story_text(story) for story in user_stories]]
        vectors = self.embedder.embed_documents(documents)
        if len(vectors) != len(documents):
            raise ValueError("embedding provider returned an unexpected vector count")
        query = vectors[0]
        best_story: UserStoryRecord | None = None
        best_cosine = -1.0
        for story, vector in zip(user_stories, vectors[1:], strict=True):
            score = _cosine(query, vector)
            if score > best_cosine:
                best_cosine = score
                best_story = story
        return best_story, best_cosine, self.calibrated_match_pct(best_cosine)

    def calibrated_match_pct(self, cosine: float) -> float:
        # CALIBRATION RATIONALE — DO NOT REMOVE:
        # BGE-family embedding cosine similarities are compressed into a narrow band
        # (roughly [0.6, 1.0] for related text per BAAI guidance); raw cosine does NOT
        # map to an intuitive 0-100% match. cos_floor/cos_ceil linearly rescale the
        # usable cosine band into a calibrated percentage so the 60% / 5% business
        # thresholds behave as intended.
        # RECALIBRATE cos_floor / cos_ceil IF THE EMBEDDING PROVIDER OR MODEL CHANGES.
        scaled = (cosine - self.cos_floor) / (self.cos_ceil - self.cos_floor)
        return max(0.0, min(1.0, scaled)) * 100.0


def _story_text(story: UserStoryRecord) -> str:
    parts = [
        story.story_id,
        story.title,
        story.persona,
        story.user_story.as_a,
        story.user_story.i_want,
        story.user_story.so_that,
        story.business_value,
    ]
    parts.extend(
        f"Given {criterion.given}, when {criterion.when}, then {criterion.then}."
        for criterion in story.acceptance_criteria
    )
    parts.extend(rule.rule for rule in story.business_rules)
    parts.extend(scenario.scenario for scenario in story.test_scenarios)
    return "\n".join(part.strip() for part in parts if part.strip())


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
