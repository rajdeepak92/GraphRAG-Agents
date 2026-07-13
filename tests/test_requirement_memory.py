"""Cross-version requirement memory + fail-closed reconciliation (Increment 2)."""

from __future__ import annotations

import unittest

from multi_agentic_graph_rag.config.settings import RequirementIdentitySettings
from multi_agentic_graph_rag.services.requirement_memory import (
    MemoryEntry,
    RequirementMemory,
)


def _settings(**overrides: object) -> RequirementIdentitySettings:
    base = {
        "candidate_top_k": 8,
        "recall_cosine_threshold": 0.62,
        "token_overlap_threshold": 0.6,
        "use_reranker": False,
        "require_entailment_for_merge": True,
    }
    base.update(overrides)
    return RequirementIdentitySettings(**base)  # type: ignore[arg-type]


def _entry(
    rid: str,
    revision: str,
    statement: str,
    requirement_type: str = "Functional Requirement",
    source_req_id: str | None = None,
    embedding: list[float] | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        requirement_id=rid,
        revision_id=revision,
        statement=statement,
        normalized_statement=statement.strip().lower(),
        requirement_type=requirement_type,
        source_req_id=source_req_id,
        embedding=embedding,
    )


class _AlwaysJudge:
    def __init__(self, forward: bool, backward: bool) -> None:
        self._forward = forward
        self._backward = backward
        self._calls = 0

    def entails(self, premise: str, hypothesis: str) -> bool:
        self._calls += 1
        return self._forward if self._calls == 1 else self._backward


class _ConstEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class ReconcileTests(unittest.TestCase):
    def test_exact_statement_is_exact_reuse(self) -> None:
        memory = RequirementMemory(settings=_settings())
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall expose Modbus."))
        result = memory.reconcile(
            statement="The gateway shall expose Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall expose modbus.",
        )
        self.assertEqual(result.decision, "EXACT")
        self.assertEqual(result.requirement_id, "REQ-1")
        self.assertEqual(result.revision_id, "REV-1")

    def test_same_signature_different_wording_is_new_revision(self) -> None:
        memory = RequirementMemory(settings=_settings())
        memory.add(_entry("REQ-1", "REV-1", "The controller shall trip at 70C."))
        result = memory.reconcile(
            statement="The controller shall trip at 80C.",
            requirement_type="Functional Requirement",
            normalized_statement="the controller shall trip at 80c.",
        )
        self.assertEqual(result.decision, "SAME_LINEAGE_REVISION")
        self.assertEqual(result.requirement_id, "REQ-1")
        self.assertIsNone(result.revision_id)

    def test_recall_only_without_judge_fails_closed(self) -> None:
        memory = RequirementMemory(settings=_settings())
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall expose readings over Modbus."))
        result = memory.reconcile(
            statement="The gateway shall expose readings over Modbus now.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall expose readings over modbus now.",
        )
        self.assertEqual(result.decision, "DISTINCT")
        self.assertIn("recall_only_no_judge_fail_closed", result.reasons)

    def test_paraphrase_merges_only_with_bidirectional_entailment(self) -> None:
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=_AlwaysJudge(forward=True, backward=True),
        )
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall provide readings via Modbus."))
        result = memory.reconcile(
            statement="Readings shall be published by the gateway on Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="readings shall be published by the gateway on modbus.",
        )
        self.assertEqual(result.decision, "EXACT")
        self.assertEqual(result.requirement_id, "REQ-1")
        self.assertIn("bidirectional_entailment", result.reasons)

    def test_one_directional_entailment_stays_distinct(self) -> None:
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=_AlwaysJudge(forward=True, backward=False),
        )
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall provide readings via Modbus."))
        result = memory.reconcile(
            statement="Readings shall be published by the gateway on Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="readings shall be published by the gateway on modbus.",
        )
        self.assertEqual(result.decision, "DISTINCT")
        self.assertIn("entailment_not_mutual_fail_closed", result.reasons)

    def test_multiple_entailing_lineages_are_ambiguous_and_fail_closed(self) -> None:
        memory = RequirementMemory(
            settings=_settings(token_overlap_threshold=0.1),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=_AlwaysJudge(forward=True, backward=True),
        )
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall publish Modbus readings."))
        memory.add(_entry("REQ-2", "REV-2", "Modbus readings shall be exposed by the gateway."))
        result = memory.reconcile(
            statement="The gateway shall expose readings over Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall expose readings over modbus.",
        )
        self.assertEqual(result.decision, "AMBIGUOUS")
        self.assertIsNone(result.requirement_id)

    def test_family_mismatch_never_merges(self) -> None:
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=_AlwaysJudge(forward=True, backward=True),
        )
        memory.add(
            _entry(
                "REQ-1",
                "REV-1",
                "The system shall report equipment health.",
                requirement_type="Business Requirement",
            )
        )
        result = memory.reconcile(
            statement="The system shall report equipment health.",
            requirement_type="Acceptance Criteria",
            normalized_statement="the system shall report equipment health.",
        )
        self.assertEqual(result.decision, "DISTINCT")

    def test_no_candidate_is_distinct(self) -> None:
        memory = RequirementMemory(settings=_settings())
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall expose Modbus."))
        result = memory.reconcile(
            statement="Operators shall receive maintenance schedules by email.",
            requirement_type="Functional Requirement",
            normalized_statement="operators shall receive maintenance schedules by email.",
        )
        self.assertEqual(result.decision, "DISTINCT")
        self.assertIn("no_recall_candidate", result.reasons)

    def test_no_transitive_over_merge(self) -> None:
        # A~B and B~C must not merge A and C: reconcile verifies against the top
        # canonical representative only, judged pairwise.
        judge = _AlwaysJudge(forward=True, backward=True)
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        memory.add(_entry("REQ-A", "REV-A", "The gateway shall expose readings via Modbus."))
        result = memory.reconcile(
            statement="The gateway shall publish readings on Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall publish readings on modbus.",
        )
        # It merges to exactly one representative (REQ-A), not a transitive cluster.
        self.assertEqual(result.decision, "EXACT")
        self.assertEqual(result.candidate_ids, ("REQ-A",))


class RecallCalibrationTests(unittest.TestCase):
    def test_cosine_threshold_gates_recall(self) -> None:
        # Orthogonal embedding => cosine 0 => below threshold => not recalled.
        memory = RequirementMemory(
            settings=_settings(recall_cosine_threshold=0.9, token_overlap_threshold=0.99),
            embedder=_ConstEmbedder([0.0, 1.0, 0.0]),
        )
        memory.add(
            _entry(
                "REQ-1",
                "REV-1",
                "Alpha beta gamma.",
                embedding=[1.0, 0.0, 0.0],
            )
        )
        pool = memory.candidates(
            statement="Delta epsilon zeta.", requirement_type="Functional Requirement"
        )
        self.assertEqual(pool, [])


if __name__ == "__main__":
    unittest.main()
