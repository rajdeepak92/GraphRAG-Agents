"""Semantic duplicate detection for generated test scenarios."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from multi_agentic_graph_rag.common_prompt_defs import (
    DUPLICATE_JUDGE_PROMPT,
    SCENARIO_CANONICALIZATION_PROMPT,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalScenario,
    DuplicateCandidate,
    DuplicateGroup,
    DuplicateJudgeResult,
    TestScenarioRecord,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, ReasoningModel, RerankerModel


@dataclass(frozen=True)
class DedupConfig:
    recall_cosine: float = 0.55


class DedupEngine:
    def __init__(
        self,
        embedder: EmbeddingModel,
        reranker: RerankerModel | None,
        reasoner: ReasoningModel,
        cfg: DedupConfig,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.reasoner = reasoner
        self.cfg = cfg
        self._canonical_cache: dict[str, CanonicalScenario] = {}

    def canonicalize(self, scenario_text: str) -> CanonicalScenario:
        prompt = f"{SCENARIO_CANONICALIZATION_PROMPT}\n\nScenario:\n{scenario_text.strip()}\n"
        return self.reasoner.generate_structured(prompt=prompt, schema=CanonicalScenario)

    def candidate_pairs(self, scenarios: list[TestScenarioRecord]) -> list[DuplicateCandidate]:
        if len(scenarios) < 2:
            return []
        canonical = [self._canonical_for(record) for record in scenarios]
        vectors = self.embedder.embed_documents([item.canonical_text for item in canonical])
        candidates: list[DuplicateCandidate] = []
        for left_index, left in enumerate(scenarios):
            for right_index in range(left_index + 1, len(scenarios)):
                cosine = _cosine(vectors[left_index], vectors[right_index])
                if cosine >= self.cfg.recall_cosine:
                    candidates.append(
                        DuplicateCandidate(
                            left_id=left.scenario_id,
                            right_id=scenarios[right_index].scenario_id,
                            cosine=cosine,
                        )
                    )
        return self._rerank_candidates(candidates, scenarios)

    def verify_duplicate(
        self,
        left: TestScenarioRecord,
        right: TestScenarioRecord,
    ) -> bool:
        left_canonical = self._canonical_for(left)
        right_canonical = self._canonical_for(right)
        first = self._judge(left_canonical, right_canonical)
        second = self._judge(right_canonical, left_canonical)
        return (
            first.verdict == "DUPLICATE"
            and second.verdict == "DUPLICATE"
            and first.a_entails_b
            and first.b_entails_a
            and second.a_entails_b
            and second.b_entails_a
        )

    def find_duplicates(
        self,
        scenarios: list[TestScenarioRecord],
    ) -> list[DuplicateGroup]:
        by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        parent: dict[str, str] = {
            scenario.scenario_id: scenario.scenario_id for scenario in scenarios
        }

        def find(item: str) -> str:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        verified_pairs: list[DuplicateCandidate] = []
        for candidate in self.candidate_pairs(scenarios):
            left = by_id[candidate.left_id]
            right = by_id[candidate.right_id]
            if self.verify_duplicate(left, right):
                union(left.scenario_id, right.scenario_id)
                verified_pairs.append(candidate)

        groups_by_root: dict[str, list[TestScenarioRecord]] = {}
        for scenario in scenarios:
            groups_by_root.setdefault(find(scenario.scenario_id), []).append(scenario)

        groups: list[DuplicateGroup] = []
        confidence_by_pair = {
            frozenset((candidate.left_id, candidate.right_id)): candidate.cosine
            for candidate in verified_pairs
        }
        for members in groups_by_root.values():
            if len(members) < 2:
                continue
            member_ids = [member.scenario_id for member in members]
            pair_confidences = [
                score
                for pair, score in confidence_by_pair.items()
                if pair.issubset(set(member_ids))
            ]
            groups.append(
                DuplicateGroup(
                    scenario_ids=member_ids,
                    scenarios=members,
                    story_ids=sorted({member.story_id for member in members}),
                    reason="bidirectional_entailment_duplicate",
                    confidence=max(pair_confidences) if pair_confidences else 1.0,
                    verification_method="embedding_recall+llm_bidirectional_entailment",
                )
            )
        return groups

    def _canonical_for(self, scenario: TestScenarioRecord) -> CanonicalScenario:
        cached = self._canonical_cache.get(scenario.scenario_id)
        if cached is not None:
            return cached
        canonical = self.canonicalize(_scenario_text(scenario))
        self._canonical_cache[scenario.scenario_id] = canonical
        return canonical

    def _judge(
        self,
        left: CanonicalScenario,
        right: CanonicalScenario,
    ) -> DuplicateJudgeResult:
        prompt = (
            f"{DUPLICATE_JUDGE_PROMPT}\n\n"
            "Use deterministic judging. Temperature must be 0.0.\n\n"
            f"Scenario A:\n{json.dumps(left.model_dump(mode='json'), indent=2)}\n\n"
            f"Scenario B:\n{json.dumps(right.model_dump(mode='json'), indent=2)}\n"
        )
        return self.reasoner.generate_structured(prompt=prompt, schema=DuplicateJudgeResult)

    def _rerank_candidates(
        self,
        candidates: list[DuplicateCandidate],
        scenarios: list[TestScenarioRecord],
    ) -> list[DuplicateCandidate]:
        if self.reranker is None or len(candidates) < 2:
            return candidates
        by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        documents = [
            f"{_scenario_text(by_id[candidate.left_id])}\n---\n"
            f"{_scenario_text(by_id[candidate.right_id])}"
            for candidate in candidates
        ]
        order = self.reranker.rerank("duplicate test scenario pairs", documents)
        return [candidates[index] for index in order if 0 <= index < len(candidates)]


def _scenario_text(scenario: TestScenarioRecord) -> str:
    preconditions = "; ".join(scenario.preconditions)
    return (
        f"{scenario.title}\n"
        f"{scenario.description}\n"
        f"type={scenario.scenario_type}\n"
        f"preconditions={preconditions}\n"
        f"expected_result={scenario.expected_result}"
    )


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
