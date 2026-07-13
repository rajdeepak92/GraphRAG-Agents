from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    RequirementDiscoveryAgent,
    _collect_semantic_diagnostics,
    _has_multiple_actors,
)
from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    RequirementDiscoveryChunkOutput,
    RequirementRevisionSnapshot,
)
from multi_agentic_graph_rag.services.coverage_ledger import CoverageLedger
from multi_agentic_graph_rag.services.requirement_builder import build_requirement_artifact

T = TypeVar("T", bound=BaseModel)


class RequirementDiscoveryAgentTests(unittest.TestCase):
    def test_chunk_local_discovery_calls_each_chunk_individually(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        reasoner = _ChunkReasoner(discovery_batch_size=2)

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)
        self.assertEqual([chunk.chunk_id for chunk in output.chunks], ["CHUNK-1", "CHUNK-2"])
        self.assertEqual(
            reasoner.system_messages,
            [PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value] * 2,
        )

    def test_missing_provider_batch_size_still_uses_one_chunk_per_call(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        reasoner = _ChunkReasoner(discovery_batch_size=None)

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)
        self.assertEqual([chunk.chunk_id for chunk in output.chunks], ["CHUNK-1", "CHUNK-2"])

    def test_many_requirements_under_one_fact_are_preserved(self) -> None:
        manifest = _manifest(["BR-CFG-01: The system shall import and export files."])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "BR-CFG-01: The system shall import and export files.",
                    requirements=[
                        _requirement(
                            "The system shall import files.", requirement_key="import_files"
                        ),
                        _requirement(
                            "The system shall export files.", requirement_key="export_files"
                        ),
                    ],
                )
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        fact = output.chunks[0].facts[0]
        self.assertEqual(len(fact.requirements), 2)
        self.assertEqual(fact.requirements[0].temp_id, "R1")
        self.assertEqual(fact.requirements[1].temp_id, "R2")

    def test_equivalent_observations_may_reuse_one_requirement_key(self) -> None:
        first_quote = "Context A: The system shall import files."
        second_quote = "Context B: The system shall import files."
        manifest = _manifest([f"{first_quote} {second_quote}"])
        reasoner = _StaticReasoner(
            [
                _fact(first_quote, requirements=[_requirement("The system shall import files.")]),
                _fact(second_quote, requirements=[_requirement("The system shall import files.")]),
            ]
        )

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = _build_artifact(manifest, discovery)

        self.assertEqual(reasoner.prompts, 1)
        self.assertEqual(len(artifact.requirements), 1)
        self.assertEqual(len(artifact.requirements[0].evidence), 2)

    def test_incompatible_key_is_precisely_retried_and_split_into_stable_keys(self) -> None:
        quote = "The system shall import and export files."
        manifest = _manifest([quote])
        reasoner = _CollisionRepairReasoner(quote)

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(len(reasoner.prompts), 2)
        retry_prompt = reasoner.prompts[1]
        self.assertIn('"system_files"', retry_prompt)
        self.assertIn("The system shall import files.", retry_prompt)
        self.assertIn("The system shall export files.", retry_prompt)
        self.assertIn("normalized_atomic_signature", retry_prompt)
        self.assertIn("source_quote", retry_prompt)
        self.assertIn("distinct stable keys", retry_prompt)
        keys = [
            requirement.requirement_key for requirement in output.chunks[0].facts[0].requirements
        ]
        self.assertEqual(keys, ["system_import_files", "system_export_files"])

    def test_compound_source_split_uses_distinct_semantic_keys(self) -> None:
        quote = "The system shall import and export files."
        manifest = _manifest([quote])
        reasoner = _StaticReasoner(
            [
                _fact(
                    quote,
                    requirements=[
                        _requirement(
                            "The system shall import files.",
                            requirement_key="system_import_files",
                        ),
                        _requirement(
                            "The system shall export files.",
                            requirement_key="system_export_files",
                        ),
                    ],
                )
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        keys = {
            requirement.requirement_key for requirement in output.chunks[0].facts[0].requirements
        }
        self.assertEqual(keys, {"system_import_files", "system_export_files"})

    def test_same_source_req_id_with_incompatible_semantics_is_not_merged(self) -> None:
        quote = "BR-CFG-01: The system shall import and export files."
        manifest = _manifest([quote])
        reasoner = _StaticReasoner(
            [
                _fact(
                    quote,
                    requirements=[
                        _requirement(
                            "The system shall import files.",
                            source_req_id="BR-CFG-01",
                            requirement_key="system_import_files",
                        ),
                        _requirement(
                            "The system shall export files.",
                            source_req_id="BR-CFG-01",
                            requirement_key="system_export_files",
                        ),
                    ],
                )
            ]
        )

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = _build_artifact(manifest, discovery)

        self.assertEqual(len(artifact.requirements), 2)
        self.assertEqual(
            {requirement.source_req_id for requirement in artifact.requirements},
            {"BR-CFG-01"},
        )
        self.assertEqual(
            len({requirement.requirement_id for requirement in artifact.requirements}),
            2,
        )

    def test_pure_key_collision_surviving_retry_is_repaired_and_proceeds(self) -> None:
        # A requirement_key is a non-authoritative hint; Python owns permanent identity.
        # A pure collision the model never fixes must be repaired deterministically, not
        # fail the chunk and drop supported requirements.
        quote = "The system shall import and export files."
        manifest = _manifest([quote])
        with tempfile.TemporaryDirectory() as temp_dir:
            reasoner = _PersistingCollisionReasoner(Path(temp_dir), quote)

            discovery = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)
        keys = [
            requirement.requirement_key for requirement in discovery.chunks[0].facts[0].requirements
        ]
        # Repaired to distinct, stable keys derived from the semantic signature.
        self.assertTrue(all(key.startswith("system_files:") for key in keys), keys)
        self.assertEqual(len(set(keys)), 2, keys)

        artifact = _build_artifact(manifest, discovery)
        self.assertEqual(len(artifact.requirements), 2)
        self.assertEqual(
            len({requirement.requirement_id for requirement in artifact.requirements}), 2
        )

    def test_pure_key_collision_repair_is_stable_across_reruns(self) -> None:
        quote = "The system shall import and export files."
        manifest = _manifest([quote])
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = RequirementDiscoveryAgent(
                _PersistingCollisionReasoner(Path(first_dir), quote)
            ).run(manifest)
            second = RequirementDiscoveryAgent(
                _PersistingCollisionReasoner(Path(second_dir), quote)
            ).run(manifest)

        first_keys = [r.requirement_key for r in first.chunks[0].facts[0].requirements]
        second_keys = [r.requirement_key for r in second.chunks[0].facts[0].requirements]
        self.assertEqual(first_keys, second_keys)

    def test_source_support_failure_fails_closed_with_diagnostics(self) -> None:
        quote = "Embedded controller that reads sensor data using Modbus."
        req_text = (
            "SIIMCS uses an embedded controller to collect operating data from industrial "
            "sensors through a Modbus communication interface."
        )
        manifest = _manifest([f"Overview. {quote}"])
        with tempfile.TemporaryDirectory() as temp_dir:
            reasoner = _PersistingReasoner(
                Path(temp_dir),
                [_fact(quote, requirements=[_requirement(req_text)])],
            )

            with self.assertRaises(ModelOutputError) as raised:
                RequirementDiscoveryAgent(reasoner).run(manifest)

        message = str(raised.exception)
        self.assertIn("CHUNK-1", message)
        self.assertIn("attempt=2", message)
        self.assertIn("_SemanticValidationError: <message redacted", message)
        self.assertNotIn("missing_tokens", message)
        self.assertNotIn("siimcs", message)
        self.assertIn("llm_response_1_2.txt", message)
        self.assertEqual(reasoner.prompts, 2)

    def test_embedded_controller_quote_cannot_support_longer_statement(self) -> None:
        quote = "Embedded controller that reads sensor data using Modbus."
        req_text = (
            "SIIMCS uses an embedded controller to collect operating data from industrial "
            "sensors through a Modbus communication interface."
        )
        output = RequirementDiscoveryChunkOutput.model_validate(
            {"facts": [_fact(quote, requirements=[_requirement(req_text)])]}
        )

        diagnostics = _collect_semantic_diagnostics(output, chunk_id="CHUNK-1")

        support = [d for d in diagnostics if d.category == "source_support"]
        self.assertEqual(len(support), 1)
        self.assertIsNotNone(support[0].support_ratio)
        assert support[0].support_ratio is not None
        self.assertLess(support[0].support_ratio, 0.5)
        # Tokens borrowed from outside the quote are reported as missing.
        self.assertIn("siimcs", support[0].missing_tokens)
        self.assertIn("industrial", support[0].missing_tokens)
        self.assertIn("interface", support[0].missing_tokens)

    def test_noun_phrase_bullet_with_template_scaffolding_is_supported(self) -> None:
        # "The system shall provide monitoring." is grounded in the bullet
        # "Cloud-based storage and monitoring." — only the injected actor "system"
        # and the empty verb "provide" are absent, and those are template scaffolding.
        output = RequirementDiscoveryChunkOutput.model_validate(
            {
                "facts": [
                    _fact(
                        "Cloud-based storage and monitoring.",
                        requirements=[
                            _requirement(
                                "The system shall provide monitoring.",
                                requirement_key="provide_monitoring",
                            ),
                            _requirement(
                                "The system shall provide cloud-based storage.",
                                requirement_key="cloud_based_storage",
                            ),
                        ],
                    )
                ]
            }
        )

        diagnostics = _collect_semantic_diagnostics(output, chunk_id="CHUNK-1")

        self.assertEqual([d for d in diagnostics if d.category == "source_support"], [])

    def test_template_scaffolding_does_not_mask_unsupported_content(self) -> None:
        # An empty obligation verb must not smuggle in an unsupported domain noun.
        output = RequirementDiscoveryChunkOutput.model_validate(
            {
                "facts": [
                    _fact(
                        "Cloud-based storage and monitoring.",
                        requirements=[
                            _requirement(
                                "The system shall provide encryption.",
                                requirement_key="provide_encryption",
                            )
                        ],
                    )
                ]
            }
        )

        diagnostics = _collect_semantic_diagnostics(output, chunk_id="CHUNK-1")

        support = [d for d in diagnostics if d.category == "source_support"]
        self.assertEqual(len(support), 1)
        self.assertIn("encryption", support[0].missing_tokens)

    def test_source_support_corrective_retry_produces_atomic_statement(self) -> None:
        quote = "Embedded controller that reads sensor data using Modbus."
        unsupported = (
            "SIIMCS uses an embedded controller to collect operating data from industrial "
            "sensors through a Modbus communication interface."
        )
        supported = "The embedded controller reads sensor data using Modbus."
        manifest = _manifest([f"Overview. {quote}"])
        reasoner = _RetryReasoner(
            [
                [_fact(quote, requirements=[_requirement(unsupported)])],
                [_fact(quote, requirements=[_requirement(supported)])],
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(len(reasoner.prompts), 2)
        retry_prompt = reasoner.prompts[1]
        self.assertIn("source_support", retry_prompt)
        self.assertIn("missing_tokens", retry_prompt)
        self.assertIn("previous structured output", retry_prompt)
        self.assertIn("fact_index", retry_prompt)
        statements = [r.statement for r in output.chunks[0].facts[0].requirements]
        self.assertEqual(statements, [supported])

    def test_multiple_semantic_errors_reported_and_repaired_in_one_retry(self) -> None:
        quote = "Sensor-1 and Sensor-2 report data. Embedded controller that reads sensor data."
        bad_actor = "Sensor-1 and Sensor-2 shall report data."
        bad_support = (
            "SIIMCS uses an embedded controller to collect operating data from industrial "
            "sensors through Modbus."
        )
        good_split = [
            _requirement("Sensor-1 shall report data.", requirement_key="sensor_1_report"),
            _requirement("Sensor-2 shall report data.", requirement_key="sensor_2_report"),
            _requirement(
                "The embedded controller reads sensor data.", requirement_key="controller_reads"
            ),
        ]
        manifest = _manifest([quote])
        reasoner = _RetryReasoner(
            [
                [
                    _fact(
                        quote,
                        requirements=[
                            _requirement(bad_actor, requirement_key="sensors_report"),
                            _requirement(bad_support, requirement_key="controller_reads"),
                        ],
                    )
                ],
                [_fact(quote, requirements=good_split)],
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(len(reasoner.prompts), 2)
        retry_prompt = reasoner.prompts[1]
        self.assertIn("multiple_actors", retry_prompt)
        self.assertIn("source_support", retry_prompt)
        self.assertEqual(len(output.chunks[0].facts[0].requirements), 3)

    def test_qa_and_validation_teams_is_single_actor(self) -> None:
        self.assertFalse(_has_multiple_actors("QA and Validation Teams"))
        quote = "QA and Validation Teams shall validate SIIMCS outputs."
        manifest = _manifest([quote])
        reasoner = _StaticReasoner([_fact(quote, requirements=[_requirement(quote)])])

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 1)
        self.assertEqual(len(output.chunks[0].facts[0].requirements), 1)

    def test_genuine_independent_numbered_actors_are_flagged(self) -> None:
        self.assertTrue(_has_multiple_actors("Sensor-1 and Sensor-2"))
        self.assertTrue(_has_multiple_actors("Sensor-1 or Sensor-2"))
        self.assertFalse(_has_multiple_actors("the operator and the administrator"))
        output = RequirementDiscoveryChunkOutput.model_validate(
            {
                "facts": [
                    _fact(
                        "Sensor-1 and Sensor-2 report data.",
                        requirements=[_requirement("Sensor-1 and Sensor-2 shall report data.")],
                    )
                ]
            }
        )

        diagnostics = _collect_semantic_diagnostics(output, chunk_id="CHUNK-1")

        self.assertTrue(any(d.category == "multiple_actors" for d in diagnostics))

    def test_corrected_chunk_reuses_prior_uuid_ids_without_duplicates(self) -> None:
        quote = "The system shall import and export files."
        manifest = _manifest([quote])
        facts = [
            _fact(
                quote,
                requirements=[
                    _requirement(
                        "The system shall import files.",
                        requirement_key="system_import_files",
                    ),
                    _requirement(
                        "The system shall export files.",
                        requirement_key="system_export_files",
                    ),
                ],
            )
        ]
        first_discovery = RequirementDiscoveryAgent(_StaticReasoner(facts)).run(manifest)
        first = _build_artifact(manifest, first_discovery)
        prior = {
            requirement.requirement_id: RequirementRevisionSnapshot(
                requirement_id=requirement.requirement_id,
                revision_id=requirement.revision_id,
                statement=requirement.statement,
                normalized_statement=requirement.normalized_statement,
                requirement_type=requirement.requirement_type,
                semantic_signature=requirement.semantic_signature,
            )
            for requirement in first.requirements
        }

        second_discovery = RequirementDiscoveryAgent(_StaticReasoner(facts)).run(manifest)
        second = _build_artifact(manifest, second_discovery, prior_revisions=prior)

        self.assertEqual(len(first.requirements), 2)
        self.assertEqual(len(second.requirements), 2)
        self.assertEqual(
            {requirement.requirement_id for requirement in first.requirements},
            {requirement.requirement_id for requirement in second.requirements},
        )
        self.assertEqual(
            {requirement.revision_id for requirement in first.requirements},
            {requirement.revision_id for requirement in second.requirements},
        )

    def test_normalized_quote_maps_back_to_exact_raw_pdf_newline_span(self) -> None:
        raw_text = (
            "Overview text before the requirement. The system shall support industrial "
            "monitoring\nand control telemetry."
        )
        manifest = _manifest([raw_text])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "industrial monitoring and control telemetry.",
                    requirements=[
                        _requirement(
                            "The system shall support industrial monitoring and control telemetry."
                        )
                    ],
                )
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        trace = output.chunks[0].facts[0].source_trace
        expected_quote = "industrial monitoring\nand control telemetry."
        self.assertEqual(trace.quote, expected_quote)
        self.assertEqual(trace.start_char, raw_text.index("industrial"))
        self.assertEqual(raw_text[trace.start_char : trace.end_char], expected_quote)

    def test_split_requirement_id_quote_maps_back_to_raw_pdf_span(self) -> None:
        raw_text = "BR-ALT-\n001\nUsers shall be able to configure warning thresholds."
        manifest = _manifest([raw_text])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "BR-ALT-001 Users shall be able to configure warning thresholds.",
                    requirements=[
                        _requirement("Users shall be able to configure warning thresholds.")
                    ],
                )
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        trace = output.chunks[0].facts[0].source_trace
        self.assertEqual(trace.quote, raw_text)
        self.assertEqual(raw_text[trace.start_char : trace.end_char], raw_text)

    def test_identifier_requirement_text_is_rejected_by_schema(self) -> None:
        invalid_texts = [
            "BR-SEN-001",
            "BR-SEN-001 The controller shall collect data from configured sensors.",
            "BR-COM- 002",
            "Requirement",
        ]

        for req_text in invalid_texts:
            with self.subTest(req_text=req_text), self.assertRaises(ValidationError):
                RequirementDiscoveryChunkOutput.model_validate(
                    {
                        "facts": [
                            _fact(
                                "The controller shall collect data from configured sensors.",
                                requirements=[_requirement(req_text)],
                            )
                        ]
                    }
                )

    def test_prompt_requires_meaningful_requirement_text_not_source_ids(self) -> None:
        manifest = _manifest(["The controller shall collect data from three configured sensors."])
        reasoner = _ChunkReasoner(discovery_batch_size=1)

        RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertIn("Analyze the entire input chunk_text", reasoner.last_prompt)
        self.assertIn("CHUNK-07", reasoner.last_prompt)
        self.assertIn("user stories, scenarios, and test cases", reasoner.last_prompt)
        self.assertIn(
            "req_text must be a complete, meaningful business requirement sentence",
            reasoner.last_prompt,
        )
        self.assertIn("must never be only an identifier", reasoner.last_prompt)
        self.assertIn(
            "do not copy the source identifier into req_text",
            reasoner.last_prompt,
        )

    def test_unlocatable_quote_is_retried_once(self) -> None:
        manifest = _manifest(["Intro sentence. The system shall import files."])
        reasoner = _QuoteRetryReasoner()

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)
        self.assertEqual(
            output.chunks[0].facts[0].source_trace.quote,
            "The system shall import files.",
        )

    def test_fact_only_record_is_accepted(self) -> None:
        manifest = _manifest(["The imported source file is a monthly operations extract."])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "monthly operations extract",
                    fact_text="The imported source file is a monthly operations extract.",
                    requirements=[],
                )
            ]
        )

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        self.assertEqual(len(discovery.chunks[0].facts[0].requirements), 0)
        self.assertEqual(len(artifact.facts), 1)
        self.assertEqual(len(artifact.requirements), 0)

    def test_empty_facts_are_accepted_for_non_requirement_chunk(self) -> None:
        manifest = _manifest(["Background context only."])
        reasoner = _EmptyReasoner()

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(output.chunks, [])

    def test_empty_facts_are_rejected_for_requirement_bearing_chunk(self) -> None:
        manifest = _manifest(["The system shall import files."])
        reasoner = _EmptyReasoner()

        with self.assertRaises(ModelOutputError):
            RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)

    def test_temporary_llm_ids_are_replaced_by_permanent_python_ids(self) -> None:
        manifest = _manifest(["The system shall import files."])
        reasoner = _ChunkReasoner(discovery_batch_size=1)

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        self.assertEqual(len(artifact.facts), 1)
        self.assertEqual(len(artifact.requirements), 1)
        self.assertTrue(artifact.facts[0].fact_id.startswith("FACT-"))
        self.assertNotEqual(artifact.facts[0].fact_id, "F1")
        self.assertTrue(artifact.requirements[0].requirement_id.startswith("REQ-"))
        self.assertTrue(artifact.requirements[0].revision_id.startswith("REQREV-"))
        self.assertTrue(artifact.requirements[0].evidence[0].evidence_id.startswith("REQEVID-"))

    def test_source_req_id_is_normalized_when_present_in_source_quote(self) -> None:
        manifest = _manifest(["SYS_REQ_001: The system shall import files."])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "SYS_REQ_001: The system shall import files.",
                    requirements=[
                        _requirement(
                            "The system shall import files.",
                            source_req_id="SYS REQ 001",
                        )
                    ],
                )
            ]
        )

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        requirement = artifact.requirements[0]
        self.assertEqual(requirement.source_req_id, "SYS_REQ_001")
        self.assertEqual(requirement.id_generation_type, "source")
        self.assertEqual(requirement.confidence, 0.85)

    def test_source_req_id_falls_back_to_null_when_not_in_source_quote(self) -> None:
        manifest = _manifest(["The system shall import files."])
        reasoner = _StaticReasoner(
            [
                _fact(
                    "The system shall import files.",
                    requirements=[
                        _requirement(
                            "The system shall import files.",
                            source_req_id="SYS_REQ_001",
                        )
                    ],
                )
            ]
        )

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        requirement = artifact.requirements[0]
        self.assertIsNone(requirement.source_req_id)
        self.assertEqual(requirement.id_generation_type, "generated")

    def test_failed_chunk_saves_raw_response_for_each_validation_attempt(self) -> None:
        manifest = _manifest(["Intro. The system shall import files."])
        with tempfile.TemporaryDirectory() as temp_dir:
            reasoner = _PersistingWrongQuoteReasoner(Path(temp_dir))

            with self.assertRaises(ModelOutputError):
                RequirementDiscoveryAgent(reasoner).run(manifest)

            first = Path(temp_dir) / "llm_response_1_1.txt"
            second = Path(temp_dir) / "llm_response_1_2.txt"
            self.assertEqual(first.read_text(encoding="utf-8"), "raw response 1")
            self.assertEqual(second.read_text(encoding="utf-8"), "raw response 2")

    def test_duplicate_overlap_chunks_preserve_fact_occurrences(
        self,
    ) -> None:
        manifest = _manifest(
            [
                "Context A. The system shall import files.",
                "Context B. The system shall import files.",
            ]
        )
        reasoner = _ChunkReasoner(discovery_batch_size=1)

        discovery = RequirementDiscoveryAgent(reasoner).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        self.assertEqual(len(artifact.facts), 2)
        self.assertEqual(len(artifact.canonical_facts), 1)
        self.assertEqual(len(artifact.requirements), 1)
        self.assertEqual(len(artifact.requirements[0].evidence), 2)

    def test_ledger_disabled_first_chunk_prompt_has_no_ledger_section(self) -> None:
        manifest = _manifest(["The system shall import files."])
        reasoner = _PromptCapturingReasoner()

        RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertNotIn("PREVIOUSLY DISCOVERED REQUIREMENTS", reasoner.prompts[0])

    def test_ledger_enabled_first_chunk_prompt_has_no_ledger_section(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        reasoner = _PromptCapturingReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)

        self.assertNotIn("PREVIOUSLY DISCOVERED REQUIREMENTS", reasoner.prompts[0])

    def test_ledger_enabled_second_chunk_prompt_includes_prior_requirement(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        reasoner = _PromptCapturingReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)

        second_prompt = reasoner.prompts[1]
        self.assertIn("PREVIOUSLY DISCOVERED REQUIREMENTS", second_prompt)
        self.assertIn("system behavior", second_prompt)
        self.assertIn("The system shall import files.", second_prompt)

    def test_ledger_snapshot_is_reused_across_retry_attempts(self) -> None:
        manifest = _manifest(
            ["The system shall import files.", "Intro. The system shall export files."]
        )
        reasoner = _LedgerRetryReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)

        ledger_json = '"statement":"The system shall import files."'
        # prompts: chunk1, chunk2 attempt1 (bad quote), chunk2 attempt2 (repaired).
        self.assertEqual(len(reasoner.prompts), 3)
        self.assertIn(ledger_json, reasoner.prompts[1])
        self.assertIn(ledger_json, reasoner.prompts[2])

    def test_ledger_disabled_prompt_matches_enabled_first_chunk_prompt(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        enabled_reasoner = _PromptCapturingReasoner()
        disabled_reasoner = _PromptCapturingReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        RequirementDiscoveryAgent(enabled_reasoner, coverage_ledger=ledger).run(manifest)
        RequirementDiscoveryAgent(disabled_reasoner).run(manifest)

        self.assertEqual(enabled_reasoner.prompts[0], disabled_reasoner.prompts[0])
        self.assertNotIn("PREVIOUSLY DISCOVERED REQUIREMENTS", disabled_reasoner.prompts[1])

    def test_ledger_records_only_after_validated_output(self) -> None:
        manifest = _manifest(["Intro. The system shall import files."])
        reasoner = _AlwaysBadQuoteReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        with self.assertRaises(ModelOutputError):
            RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)

        self.assertEqual(ledger.size, 0)

    def test_ledger_does_not_bypass_trace_validation(self) -> None:
        manifest = _manifest(["Intro. The system shall import files."])
        reasoner = _AlwaysBadQuoteReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        with self.assertRaises(ModelOutputError):
            RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)

        self.assertEqual(len(reasoner.prompts), 2)

    def test_ledger_logging_emits_counts_without_requirement_text(self) -> None:
        manifest = _manifest(["The system shall import files."])
        reasoner = _ChunkReasoner(discovery_batch_size=1)
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)
        logger = _RecordingLogger()

        RequirementDiscoveryAgent(reasoner, logger=logger, coverage_ledger=ledger).run(manifest)

        ledger_logs = [
            kwargs
            for _level, _message, kwargs in logger.records
            if kwargs.get("step") == "discover_requirements.ledger"
        ]
        self.assertEqual(len(ledger_logs), 1)
        entry = ledger_logs[0]
        self.assertEqual(entry["ledger_size"], 1)
        self.assertEqual(entry["injected_count"], 0)
        self.assertEqual(entry["exact_converged_count"], 0)
        self.assertEqual(entry["new_entries"], 1)
        self.assertEqual(entry["status"], "completed")
        self.assertNotIn("The system shall import files.", str(entry))

    def test_ledger_enabled_duplicate_chunks_collapse_with_two_evidence(self) -> None:
        manifest = _manifest(
            [
                "Context A. The system shall import files.",
                "Context B. The system shall import files.",
            ]
        )
        reasoner = _ChunkReasoner(discovery_batch_size=1)
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        discovery = RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        self.assertEqual(len(artifact.facts), 2)
        self.assertEqual(len(artifact.canonical_facts), 1)
        self.assertEqual(len(artifact.requirements), 1)
        self.assertEqual(len(artifact.requirements[0].evidence), 2)
        self.assertEqual(ledger.size, 1)

    def test_ledger_preserves_changed_revision_for_same_functional_key(self) -> None:
        manifest = _manifest(
            [
                "The alarm shall trigger above 70 degrees.",
                "The alarm shall trigger above 80 degrees.",
            ]
        )
        reasoner = _AlarmThresholdReasoner()
        ledger = CoverageLedger(max_entries=100, injection_top_k=10)

        discovery = RequirementDiscoveryAgent(reasoner, coverage_ledger=ledger).run(manifest)
        artifact = build_requirement_artifact(
            project=manifest.project,
            document_id=manifest.document_id,
            document_version_id=manifest.document_version_id,
            version=manifest.version,
            source_path=manifest.source_path,
            source_checksum=manifest.source_checksum,
            discovery=discovery,
        )

        self.assertEqual(len(artifact.requirements), 2)
        lineage_ids = {requirement.requirement_id for requirement in artifact.requirements}
        revision_ids = {requirement.revision_id for requirement in artifact.requirements}
        self.assertEqual(len(lineage_ids), 1)
        self.assertEqual(len(revision_ids), 2)
        self.assertTrue(any(event.event_type == "changed" for event in artifact.delta_events))
        self.assertEqual(ledger.size, 2)


class _ChunkReasoner:
    provider_name = "huggingface"

    def __init__(self, *, discovery_batch_size: int | None) -> None:
        if discovery_batch_size is not None:
            self.discovery_batch_size = discovery_batch_size
        self.prompts = 0
        self.last_prompt = ""
        self.system_messages: list[str] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[T],
        system_message: str,
        **_: object,
    ) -> T:
        self.prompts += 1
        self.last_prompt = prompt
        self.system_messages.append(system_message)
        chunk_text = _chunk_text(prompt)
        quote = _requirement_quote(chunk_text)
        return schema.model_validate({"facts": [_fact(quote)]})


class _StaticReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        return schema.model_validate({"facts": self.facts})


class _CollisionRepairReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        keys = (
            ("system_files", "system_files")
            if len(self.prompts) == 1
            else ("system_import_files", "system_export_files")
        )
        return schema.model_validate({"facts": _collision_facts(self.quote, keys)})


class _PersistingCollisionReasoner:
    provider_name = "azure_openai"
    discovery_batch_size = 1

    def __init__(self, run_dir: Path, quote: str) -> None:
        self.run_dir = run_dir
        self.quote = quote
        self.prompts = 0
        self.last_response_path: Path | None = None

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        return schema.model_validate(
            {"facts": _collision_facts(self.quote, ("system_files", "system_files"))}
        )

    def persist_last_response(self, *, filename: str) -> Path:
        path = self.run_dir / filename
        path.write_text(f"raw collision response {self.prompts}", encoding="utf-8")
        self.last_response_path = path
        return path


class _RetryReasoner:
    """Returns a fixed facts payload per attempt, capturing the prompts it saw."""

    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self, facts_per_attempt: list[list[dict[str, Any]]]) -> None:
        self.facts_per_attempt = facts_per_attempt
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.facts_per_attempt) - 1)
        return schema.model_validate({"facts": self.facts_per_attempt[index]})


class _PersistingReasoner:
    """Always returns the same facts and persists a raw response file per attempt."""

    provider_name = "azure_openai"
    discovery_batch_size = 1

    def __init__(self, run_dir: Path, facts: list[dict[str, Any]]) -> None:
        self.run_dir = run_dir
        self.facts = facts
        self.prompts = 0
        self.last_response_path: Path | None = None

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        return schema.model_validate({"facts": self.facts})

    def persist_last_response(self, *, filename: str) -> Path:
        path = self.run_dir / filename
        path.write_text(f"raw response {self.prompts}", encoding="utf-8")
        self.last_response_path = path
        return path


class _QuoteRetryReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        quote = "not present in the source chunk"
        if self.prompts == 2:
            quote = "The system shall import files."
        return schema.model_validate({"facts": [_fact(quote)]})


class _EmptyReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        return schema.model_validate({"facts": []})


class _PersistingWrongQuoteReasoner:
    provider_name = "azure_openai"
    discovery_batch_size = 1

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.prompts = 0
        self.last_response_path: Path | None = None
        self._last_response = ""

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts += 1
        self._last_response = f"raw response {self.prompts}"
        return schema.model_validate({"facts": [_fact("not present in the source chunk")]})

    def persist_last_response(self, *, filename: str) -> Path:
        path = self.run_dir / filename
        path.write_text(self._last_response, encoding="utf-8")
        self.last_response_path = path
        return path


class _PromptCapturingReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        chunk_text = _chunk_text(prompt)
        quote = _requirement_quote(chunk_text)
        return schema.model_validate({"facts": [_fact(quote)]})


class _LedgerRetryReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._export_attempts = 0

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        chunk_text = _chunk_text(prompt)
        if "export" in chunk_text:
            self._export_attempts += 1
            quote = (
                "not present in the source chunk"
                if self._export_attempts == 1
                else _requirement_quote(chunk_text)
            )
        else:
            quote = _requirement_quote(chunk_text)
        return schema.model_validate({"facts": [_fact(quote)]})


class _AlwaysBadQuoteReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        return schema.model_validate({"facts": [_fact("not present in the source chunk")]})


class _AlarmThresholdReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_structured(self, *, prompt: str, schema: type[T], **_: object) -> T:
        self.prompts.append(prompt)
        chunk_text = _chunk_text(prompt)
        return schema.model_validate(
            {
                "facts": [
                    {
                        "fact_text": chunk_text,
                        "quote": chunk_text,
                        "requirements": [
                            {
                                "req_text": chunk_text,
                                "requirement_type": "functional",
                                "priority": "medium",
                                "requirement_key": "alarm_threshold",
                                "source_req_id": "",
                                "confidence": 0.85,
                            }
                        ],
                    }
                ]
            }
        )


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def info(self, message: str, **kwargs: Any) -> None:
        self.records.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.records.append(("warning", message, kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        self.records.append(("debug", message, kwargs))


def _chunk_text(prompt: str) -> str:
    marker = "Input chunk JSON:\n"
    return str(json.loads(prompt.split(marker, 1)[1])["chunk_text"])


def _requirement_quote(chunk_text: str) -> str:
    marker = "The system shall"
    start = chunk_text.index(marker) if marker in chunk_text else 0
    return chunk_text[start:]


def _fact(
    quote: str,
    *,
    fact_text: str | None = None,
    requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "fact_text": fact_text or quote,
        "quote": quote,
        "requirements": requirements if requirements is not None else [_requirement(quote)],
    }


def _requirement(
    req_text: str,
    *,
    source_req_id: str = "",
    requirement_key: str = "system behavior",
) -> dict[str, object]:
    return {
        "req_text": req_text,
        "requirement_type": "functional",
        "priority": "medium",
        "requirement_key": requirement_key,
        "source_req_id": source_req_id,
        "confidence": 0.85,
    }


def _collision_facts(quote: str, keys: tuple[str, str]) -> list[dict[str, Any]]:
    return [
        _fact(
            quote,
            requirements=[
                _requirement("The system shall import files.", requirement_key=keys[0]),
                _requirement("The system shall export files.", requirement_key=keys[1]),
            ],
        )
    ]


def _build_artifact(
    manifest: DocumentManifest,
    discovery: Any,
    *,
    prior_revisions: dict[str, RequirementRevisionSnapshot] | None = None,
) -> Any:
    return build_requirement_artifact(
        project=manifest.project,
        document_id=manifest.document_id,
        document_version_id=manifest.document_version_id,
        version=manifest.version,
        source_path=manifest.source_path,
        source_checksum=manifest.source_checksum,
        discovery=discovery,
        prior_revisions=prior_revisions,
    )


def _manifest(texts: list[str]) -> DocumentManifest:
    chunks = [
        DocumentChunk(
            chunk_id=f"CHUNK-{index}",
            ordinal=index,
            text=text,
            normalized_text=text.lower(),
            start_char=0,
            end_char=len(text),
            source_block_ids=[f"BLOCK-{index}"],
        )
        for index, text in enumerate(texts, start=1)
    ]
    return DocumentManifest(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        logical_name="doc",
        version="1.0",
        source_path="doc.txt",
        source_checksum="abc",
        parser_fingerprint="parser",
        chunker_fingerprint="chunker",
        created_at=datetime.now(UTC),
        chunks=chunks,
    )


if __name__ == "__main__":
    unittest.main()
