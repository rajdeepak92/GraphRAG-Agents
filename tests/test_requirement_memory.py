"""Cross-version requirement memory + fail-closed reconciliation (Increment 2)."""

from __future__ import annotations

import json
import unittest
from typing import Any

from pydantic import ValidationError

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementIdentity
from multi_agentic_graph_rag.config.settings import RequirementIdentitySettings
from multi_agentic_graph_rag.services.requirement_memory import (
    MemoryEntry,
    ModelEntailmentJudge,
    RequirementMemory,
    _BidirectionalEntailmentOutput,
)


def _settings(**overrides: object) -> RequirementIdentitySettings:
    base = {
        "candidate_top_k": 2,
        "max_entailment_calls": 200,
        "max_structured_attempts": 2,
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
    semantic_recall_enabled: bool = True,
) -> MemoryEntry:
    return MemoryEntry(
        requirement_id=rid,
        revision_id=revision,
        statement=statement,
        normalized_statement=statement.strip().lower(),
        requirement_type=requirement_type,
        source_req_id=source_req_id,
        embedding=embedding,
        semantic_recall_enabled=semantic_recall_enabled,
    )


class _AlwaysJudge:
    def __init__(self, forward: bool, backward: bool) -> None:
        self._forward = forward
        self._backward = backward
        self._calls = 0

    def equivalent(self, premise: str, hypothesis: str) -> bool:
        self._calls += 1
        return self._forward and self._backward

    @property
    def calls(self) -> int:
        return self._calls


class _CapturingReasoner:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(
            {
                "system_message": kwargs["system_message"],
                "operation": kwargs["operation"],
                "request_id": kwargs["request_id"],
                "prompt": kwargs["prompt"],
                "schema": kwargs["schema"],
                "max_attempts": kwargs["max_attempts"],
            }
        )
        return kwargs["schema"].model_validate(
            {
                "premise_entails_hypothesis": False,
                "hypothesis_entails_premise": False,
            }
        )


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[dict[str, Any]] = []
        self.debugs: list[dict[str, Any]] = []

    def warning(self, message: str, **kwargs: Any) -> None:
        self.warnings.append({"message": message, **kwargs})

    def debug(self, message: str, **kwargs: Any) -> None:
        self.debugs.append({"message": message, **kwargs})


class _ConstEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class ReconcileTests(unittest.TestCase):
    def test_discovery_facts_payload_cannot_validate_as_identity_output(self) -> None:
        with self.assertRaises(ValidationError):
            _BidirectionalEntailmentOutput.model_validate({"facts": []})

    def test_model_judge_uses_identity_system_message_and_unique_request(self) -> None:
        reasoner = _CapturingReasoner()
        judge = ModelEntailmentJudge(reasoner)

        self.assertFalse(judge.equivalent("premise", "hypothesis"))

        self.assertEqual(
            reasoner.calls[0]["system_message"],
            PromptRequirementIdentity.SYS_PROMPT_REQUIREMENT_IDENTITY.value,
        )
        self.assertEqual(reasoner.calls[0]["operation"], "requirement_identity.entailment")
        self.assertTrue(reasoner.calls[0]["request_id"].startswith("pair-000001-"))
        self.assertEqual(
            json.loads(reasoner.calls[0]["prompt"]),
            {"premise": "premise", "hypothesis": "hypothesis"},
        )
        self.assertEqual(
            set(reasoner.calls[0]["schema"].model_fields),
            {"premise_entails_hypothesis", "hypothesis_entails_premise"},
        )
        self.assertEqual(reasoner.calls[0]["max_attempts"], 2)

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

    def test_disabled_entailment_never_invokes_available_judge(self) -> None:
        judge = _AlwaysJudge(forward=True, backward=True)
        memory = RequirementMemory(
            settings=_settings(require_entailment_for_merge=False),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        memory.add(_entry("REQ-1", "REV-1", "The gateway shall expose readings over Modbus."))

        result = memory.reconcile(
            statement="Readings shall be published by the gateway on Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="readings shall be published by the gateway on modbus.",
        )

        self.assertEqual(result.decision, "DISTINCT")
        self.assertIn("semantic_merge_disabled", result.reasons)
        self.assertEqual(judge.calls, 0)

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

    def test_same_run_entries_are_excluded_from_semantic_recall(self) -> None:
        judge = _AlwaysJudge(forward=True, backward=True)
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        memory.add(
            _entry(
                "REQ-1",
                "REV-1",
                "The gateway shall expose readings via Modbus.",
                semantic_recall_enabled=False,
            )
        )

        result = memory.reconcile(
            statement="Readings shall be published by the gateway on Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="readings shall be published by the gateway on modbus.",
        )

        self.assertEqual(result.decision, "DISTINCT")
        self.assertEqual(judge.calls, 0)

    def test_entailment_candidates_are_limited(self) -> None:
        judge = _AlwaysJudge(forward=False, backward=False)
        memory = RequirementMemory(
            settings=_settings(candidate_top_k=2),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        for index in range(5):
            memory.add(_entry(f"REQ-{index}", f"REV-{index}", f"Prior statement {index}."))

        memory.reconcile(
            statement="A completely reworded candidate requirement.",
            requirement_type="Functional Requirement",
            normalized_statement="a completely reworded candidate requirement.",
        )

        self.assertEqual(judge.calls, 2)

    def test_entailment_budget_fails_closed(self) -> None:
        judge = _AlwaysJudge(forward=False, backward=False)
        logger = _RecordingLogger()
        memory = RequirementMemory(
            settings=_settings(max_entailment_calls=1, candidate_top_k=2),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
            logger=logger,
        )
        memory.add(_entry("REQ-1", "REV-1", "Prior statement one."))
        memory.add(_entry("REQ-2", "REV-2", "Prior statement two."))

        result = memory.reconcile(
            statement="A completely reworded candidate requirement.",
            requirement_type="Functional Requirement",
            normalized_statement="a completely reworded candidate requirement.",
        )

        self.assertEqual(result.decision, "DISTINCT")
        self.assertIn("entailment_budget_exhausted", result.reasons)
        self.assertEqual(memory.entailment_calls_used, 1)
        self.assertEqual(len(logger.warnings), 1)

        memory.reconcile(
            statement="Another reworded candidate requirement.",
            requirement_type="Functional Requirement",
            normalized_statement="another reworded candidate requirement.",
        )
        self.assertEqual(len(logger.warnings), 1)

    def test_default_global_budget_never_exceeds_two_hundred_model_calls(self) -> None:
        judge = _AlwaysJudge(forward=False, backward=False)
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        memory.add(_entry("REQ-PRIOR", "REV-PRIOR", "Prior requirement statement."))

        final = None
        for index in range(201):
            final = memory.reconcile(
                statement=f"Distinct candidate requirement number {index}.",
                requirement_type="Functional Requirement",
                normalized_statement=f"distinct candidate requirement number {index}.",
            )

        self.assertIsNotNone(final)
        assert final is not None
        self.assertIn("entailment_budget_exhausted", final.reasons)
        self.assertEqual(judge.calls, 200)
        self.assertEqual(memory.entailment_calls_used, 200)

    def test_progress_logging_reports_safe_counts(self) -> None:
        logger = _RecordingLogger()
        memory = RequirementMemory(settings=_settings(), logger=logger)

        memory.reconcile(
            statement="A new requirement.",
            requirement_type="Functional Requirement",
            normalized_statement="a new requirement.",
            identity_index=3,
            identity_total=10,
        )

        self.assertEqual(len(logger.debugs), 1)
        self.assertEqual(logger.debugs[0]["identity_index"], 3)
        self.assertEqual(logger.debugs[0]["identity_total"], 10)
        self.assertEqual(logger.debugs[0]["candidate_count"], 0)
        self.assertEqual(logger.debugs[0]["cache_hits"], 0)
        self.assertEqual(logger.debugs[0]["model_call_count"], 0)
        self.assertEqual(logger.debugs[0]["budget_remaining"], 200)
        self.assertEqual(logger.debugs[0]["decision"], "DISTINCT")
        self.assertEqual(logger.debugs[0]["reason"], "no_recall_candidate")
        self.assertIn("elapsed_ms", logger.debugs[0])
        self.assertEqual(logger.debugs[0]["status"], "completed")
        self.assertNotIn("A new requirement", str(logger.debugs[0]))

    def test_bidirectional_entailment_results_are_cached(self) -> None:
        judge = _AlwaysJudge(forward=False, backward=False)
        memory = RequirementMemory(
            settings=_settings(),
            embedder=_ConstEmbedder([1.0, 0.0, 0.0]),
            judge=judge,
        )
        memory.add(_entry("REQ-1", "REV-1", "Prior requirement statement."))
        kwargs = {
            "statement": "A completely reworded candidate requirement.",
            "requirement_type": "Functional Requirement",
            "normalized_statement": "a completely reworded candidate requirement.",
        }

        memory.reconcile(**kwargs)
        memory.reconcile(**kwargs)

        self.assertEqual(judge.calls, 1)
        self.assertEqual(memory.entailment_calls_used, 1)
        self.assertEqual(memory.entailment_cache_hits, 1)


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
