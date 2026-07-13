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
    """Coordinate dedup config behavior within the services boundary."""

    recall_cosine: float = 0.55


class DedupEngine:
    """Coordinate dedup engine behavior within the services boundary."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        reranker: RerankerModel | None,
        reasoner: ReasoningModel,
        cfg: DedupConfig,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            embedder (EmbeddingModel): Provider-neutral model adapter used by the operation.
            reranker (RerankerModel | None): Provider-neutral model adapter used by the operation.
            reasoner (ReasoningModel): Provider-neutral model adapter used by the operation.
            cfg (DedupConfig): Cfg required by the operation's typed contract.
        """
        self.embedder = embedder
        self.reranker = reranker
        self.reasoner = reasoner
        self.cfg = cfg
        self._canonical_cache: dict[str, CanonicalScenario] = {}

    def canonicalize(self, scenario_text: str) -> CanonicalScenario:
        """Execute the canonicalize operation within its declared architectural boundary.

        Args:
            scenario_text (str): Input text processed in memory and excluded from diagnostic logs.

        Returns:
            CanonicalScenario: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
        prompt = f"{SCENARIO_CANONICALIZATION_PROMPT}\n\nScenario:\n{scenario_text.strip()}\n"
        return self.reasoner.generate_structured(prompt=prompt, schema=CanonicalScenario)

    def candidate_pairs(self, scenarios: list[TestScenarioRecord]) -> list[DuplicateCandidate]:
        """Execute the candidate pairs operation within its declared architectural boundary.

        Args:
            scenarios (list[TestScenarioRecord]): Ordered scenarios processed without changing their
                                                  identities.

        Returns:
            list[DuplicateCandidate]: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
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
        """Verify duplicate against the enforced runtime contract.

        Args:
            left (TestScenarioRecord): Left required by the operation's typed contract.
            right (TestScenarioRecord): Right required by the operation's typed contract.

        Returns:
            bool: The typed result produced by the operation.
        """
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
        """Find duplicates.

        Args:
            scenarios (list[TestScenarioRecord]): Ordered scenarios processed without changing their
                                                  identities.

        Returns:
            list[DuplicateGroup]: The typed result produced by the operation.
        """
        by_id = {scenario.scenario_id: scenario for scenario in scenarios}
        parent: dict[str, str] = {
            scenario.scenario_id: scenario.scenario_id for scenario in scenarios
        }

        def find(item: str) -> str:
            """Find find.

            Args:
                item (str): Item required by the operation's typed contract.

            Returns:
                str: The typed result produced by the operation.
            """
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: str, right: str) -> None:
            """Execute the union operation within its declared architectural boundary.

            Args:
                left (str): Left required by the operation's typed contract.
                right (str): Right required by the operation's typed contract.
            """
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
        """Execute the canonical for operation within its declared architectural boundary.

        Args:
            scenario (TestScenarioRecord): Scenario required by the operation's typed contract.

        Returns:
            CanonicalScenario: The typed result produced by the operation.
        """
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
        """Execute the judge operation within its declared architectural boundary.

        Args:
            left (CanonicalScenario): Left required by the operation's typed contract.
            right (CanonicalScenario): Right required by the operation's typed contract.

        Returns:
            DuplicateJudgeResult: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
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
        """Execute the rerank candidates operation within its declared architectural boundary.

        Args:
            candidates (list[DuplicateCandidate]): Candidates required by the operation's typed
                                                   contract.
            scenarios (list[TestScenarioRecord]): Ordered scenarios processed without changing their
                                                  identities.

        Returns:
            list[DuplicateCandidate]: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
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
    """Execute the scenario text operation within its declared architectural boundary.

    Args:
        scenario (TestScenarioRecord): Scenario required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    preconditions = "; ".join(scenario.preconditions)
    return (
        f"{scenario.title}\n"
        f"{scenario.description}\n"
        f"type={scenario.scenario_type}\n"
        f"preconditions={preconditions}\n"
        f"expected_result={scenario.expected_result}"
    )


def _cosine(left: list[float], right: list[float]) -> float:
    """Execute the cosine operation within its declared architectural boundary.

    Args:
        left (list[float]): Left required by the operation's typed contract.
        right (list[float]): Right required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
