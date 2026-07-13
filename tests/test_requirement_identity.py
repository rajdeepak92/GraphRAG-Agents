"""SIIMCS requirement identity + active-revision regression fixtures (Increment 1).

Encodes the non-negotiable identity semantics:
- exact requirement across chunks -> one canonical requirement, many evidence;
- Sensor-1/2/3 stay three distinct requirements;
- Business Requirement vs Acceptance Criteria never share a lineage;
- unchanged requirement across versions reuses lineage + revision;
- a mutable threshold change reuses lineage but gets a new revision;
- only the active revision reaches user-story generation.
"""

from __future__ import annotations

import unittest

from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.identifiers import (
    is_requirement_uuid7,
    requirement_evidence_occurrence_key,
)
from multi_agentic_graph_rag.domain.schemas import (
    LLMChunkExtraction,
    LLMFactCandidate,
    LLMRequirementCandidate,
    RequirementArtifact,
    RequirementDiscoveryOutput,
    RequirementRevisionSnapshot,
    SourceTrace,
)
from multi_agentic_graph_rag.services.requirement_builder import (
    build_canonical_requirements_artifact,
    build_requirement_artifact,
)
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    requirement_lineage_signature,
    resolve_requirement_identity,
)
from multi_agentic_graph_rag.services.requirement_source import (
    load_requirement_source_from_full_payload,
)

PROJECT = "SIIMCS"
DOC = "DOC-SIIMCS"


def _req(
    *,
    statement: str,
    requirement_type: str = "Functional Requirement",
    priority: str = "Medium",
    source_req_id: str | None = None,
    chunk_id: str,
    quote: str,
) -> LLMRequirementCandidate:
    trace = SourceTrace(chunk_id=chunk_id, quote=quote, start_char=0, end_char=len(quote))
    return LLMRequirementCandidate(
        temp_id="R1",
        statement=statement,
        requirement_type=requirement_type,
        priority=priority,
        source_req_id=source_req_id,
        confidence=0.9,
        source_trace=trace,
    )


def _chunk(chunk_id: str, *requirements: LLMRequirementCandidate) -> LLMChunkExtraction:
    quote = requirements[0].source_trace.quote
    fact = LLMFactCandidate(
        temp_id="F1",
        text=f"fact for {chunk_id}",
        source_trace=SourceTrace(chunk_id=chunk_id, quote=quote, start_char=0, end_char=len(quote)),
        requirements=list(requirements),
    )
    return LLMChunkExtraction(chunk_id=chunk_id, facts=[fact])


def _build(
    *requirements_per_chunk: LLMChunkExtraction,
    version: str = "1.0",
    prior: RequirementArtifact | None = None,
) -> RequirementArtifact:
    prior_revisions = None
    if prior is not None:
        prior_revisions = {
            requirement.requirement_id: RequirementRevisionSnapshot(
                requirement_id=requirement.requirement_id,
                revision_id=requirement.revision_id,
                statement=requirement.statement,
                normalized_statement=requirement.normalized_statement,
                requirement_type=requirement.requirement_type,
                evidence_ids={
                    requirement_evidence_occurrence_key(
                        document_version_identifier=prior.document_version_id,
                        chunk_identifier=evidence.source_trace.chunk_id,
                        quote=evidence.source_trace.quote,
                        start_char=evidence.source_trace.start_char,
                        end_char=evidence.source_trace.end_char,
                    ): evidence.evidence_id
                    for evidence in requirement.evidence
                },
            )
            for requirement in prior.requirements
        }
    return build_requirement_artifact(
        project=PROJECT,
        document_id=DOC,
        document_version_id=f"DV-{version}",
        version=version,
        source_path="brd.pdf",
        source_checksum="chk",
        discovery=RequirementDiscoveryOutput(chunks=list(requirements_per_chunk)),
        prior_revisions=prior_revisions,
    )


class SignatureTests(unittest.TestCase):
    def test_entity_discriminators_are_preserved(self) -> None:
        s1 = requirement_lineage_signature(
            "Sensor-1 shall poll every 5 seconds.", "Functional Requirement"
        )
        s2 = requirement_lineage_signature(
            "Sensor-2 shall poll every 5 seconds.", "Functional Requirement"
        )
        self.assertNotEqual(s1, s2)
        self.assertIn("sensor-1", s1)
        self.assertIn("sensor-2", s2)

    def test_mutable_parameters_are_masked(self) -> None:
        s70 = requirement_lineage_signature(
            "The controller shall trip at 70°C.", "Functional Requirement"
        )
        s80 = requirement_lineage_signature(
            "The controller shall trip at 80°C.", "Functional Requirement"
        )
        self.assertEqual(s70, s80)
        self.assertNotIn("70", s70)

    def test_family_scopes_lineage(self) -> None:
        br = requirement_lineage_signature(
            "The system shall report health.", "Business Requirement"
        )
        ac = requirement_lineage_signature("The system shall report health.", "Acceptance Criteria")
        self.assertNotEqual(br, ac)

    def test_identity_is_version_independent(self) -> None:
        a = resolve_requirement_identity(
            project=PROJECT,
            document_id=DOC,
            statement="The gateway shall expose Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall expose modbus.",
        )
        b = resolve_requirement_identity(
            project=PROJECT,
            document_id=DOC,
            statement="The gateway shall expose Modbus.",
            requirement_type="Functional Requirement",
            normalized_statement="the gateway shall expose modbus.",
        )
        self.assertEqual(a.requirement_id, b.requirement_id)
        self.assertEqual(a.revision_id, b.revision_id)


class BuilderIdentityTests(unittest.TestCase):
    def test_new_ids_are_uuid7_and_public_artifact_nests_occurrences(self) -> None:
        statement = "The gateway shall expose readings over Modbus."
        artifact = _build(
            _chunk("CHUNK-1", _req(statement=statement, chunk_id="CHUNK-1", quote="v1")),
            _chunk("CHUNK-2", _req(statement=statement, chunk_id="CHUNK-2", quote="v2")),
        )
        requirement = artifact.requirements[0]
        self.assertTrue(is_requirement_uuid7(requirement.requirement_id))
        self.assertTrue(is_requirement_uuid7(requirement.revision_id))
        self.assertTrue(all(is_requirement_uuid7(row.evidence_id) for row in requirement.evidence))
        public = build_canonical_requirements_artifact(artifact)
        self.assertEqual(public.artifact_schema_version, "5.0-requirements")
        self.assertEqual(len(public.requirements), 1)
        self.assertEqual(len(public.requirements[0].evidence), 2)

    def test_exact_requirement_in_two_chunks_is_one_requirement_two_evidence(self) -> None:
        stmt = "The system shall generate an alarm when a threshold is crossed."
        artifact = _build(
            _chunk("CHUNK-1", _req(statement=stmt, chunk_id="CHUNK-1", quote="alarm one")),
            _chunk("CHUNK-2", _req(statement=stmt, chunk_id="CHUNK-2", quote="alarm two")),
        )
        self.assertEqual(len(artifact.requirements), 1)
        requirement = artifact.requirements[0]
        chunks = {ev.source_trace.chunk_id for ev in requirement.evidence}
        self.assertEqual(chunks, {"CHUNK-1", "CHUNK-2"})

    def test_sensor_1_2_3_are_three_distinct_requirements(self) -> None:
        artifact = _build(
            _chunk(
                "CHUNK-1",
                _req(
                    statement="Sensor-1 shall poll every 5 seconds.", chunk_id="CHUNK-1", quote="s1"
                ),
                _req(
                    statement="Sensor-2 shall poll every 5 seconds.", chunk_id="CHUNK-1", quote="s2"
                ),
                _req(
                    statement="Sensor-3 shall poll every 5 seconds.", chunk_id="CHUNK-1", quote="s3"
                ),
            )
        )
        ids = {r.requirement_id for r in artifact.requirements}
        self.assertEqual(len(ids), 3)

    def test_business_requirement_and_acceptance_criteria_do_not_share_lineage(self) -> None:
        artifact = _build(
            _chunk(
                "CHUNK-1",
                _req(
                    statement="The system shall report equipment health.",
                    requirement_type="Business Requirement",
                    chunk_id="CHUNK-1",
                    quote="br",
                ),
                _req(
                    statement="The system shall report equipment health.",
                    requirement_type="Acceptance Criteria",
                    chunk_id="CHUNK-1",
                    quote="ac",
                ),
            )
        )
        ids = {r.requirement_id for r in artifact.requirements}
        self.assertEqual(len(ids), 2)

    def test_unchanged_requirement_across_versions_reuses_lineage_and_revision(self) -> None:
        stmt = "The gateway shall expose readings over Modbus."
        v1 = _build(
            _chunk("CHUNK-1", _req(statement=stmt, chunk_id="CHUNK-1", quote="v1")), version="1.0"
        )
        v2 = _build(
            _chunk("CHUNK-9", _req(statement=stmt, chunk_id="CHUNK-9", quote="v2")),
            version="2.0",
            prior=v1,
        )
        self.assertEqual(v1.requirements[0].requirement_id, v2.requirements[0].requirement_id)
        self.assertEqual(v1.requirements[0].revision_id, v2.requirements[0].revision_id)

    def test_reingesting_same_occurrence_reuses_evidence_id(self) -> None:
        statement = "The gateway shall expose readings over Modbus."
        chunk = _chunk("CHUNK-1", _req(statement=statement, chunk_id="CHUNK-1", quote="same quote"))
        first = _build(chunk, version="1.0")
        second = _build(chunk, version="1.0", prior=first)
        self.assertEqual(
            first.requirements[0].evidence[0].evidence_id,
            second.requirements[0].evidence[0].evidence_id,
        )

    def test_threshold_change_reuses_lineage_new_revision(self) -> None:
        v1 = _build(
            _chunk(
                "CHUNK-1",
                _req(
                    statement="The controller shall trip at 70°C.", chunk_id="CHUNK-1", quote="v1"
                ),
            ),
            version="1.0",
        )
        v2 = _build(
            _chunk(
                "CHUNK-9",
                _req(
                    statement="The controller shall trip at 80°C.", chunk_id="CHUNK-9", quote="v2"
                ),
            ),
            version="2.0",
            prior=v1,
        )
        self.assertEqual(v1.requirements[0].requirement_id, v2.requirements[0].requirement_id)
        self.assertNotEqual(v1.requirements[0].revision_id, v2.requirements[0].revision_id)


class ActiveRevisionSelectionTests(unittest.TestCase):
    def _payload(self, requirements: list[dict]) -> dict:
        base = {
            "artifact_schema_version": "2.1",
            "project": PROJECT,
            "document_id": DOC,
            "document_version_id": "DV-2.0",
            "version": "2.0",
            "source_checksum": "chk",
            "facts": [],
            "requirements": requirements,
        }
        return base

    def _requirement(self, revision_id: str, status: str, statement: str) -> dict:
        return {
            "requirement_id": "REQ-LINEAGE",
            "revision_id": revision_id,
            "statement": statement,
            "requirement_type": "Functional Requirement",
            "priority": "Medium",
            "status": status,
            "fact_ids": ["FACT-1"],
            "source_trace": {
                "chunk_id": "CHUNK-A",
                "quote": statement,
                "start_char": 0,
                "end_char": len(statement),
            },
            "evidence": [],
        }

    def test_active_revision_selected_even_when_superseded_is_first(self) -> None:
        payload = self._payload(
            [
                self._requirement("REQREV-OLD", "superseded", "trip at 70C"),
                self._requirement("REQREV-NEW", "active", "trip at 80C"),
            ]
        )
        source = load_requirement_source_from_full_payload(payload)
        self.assertEqual(len(source.requirements), 1)
        self.assertEqual(source.requirements[0].revision_id, "REQREV-NEW")
        self.assertEqual(source.requirements[0].requirement_text, "trip at 80C")

    def test_no_active_revision_fails_loudly(self) -> None:
        payload = self._payload([self._requirement("REQREV-OLD", "superseded", "trip at 70C")])
        with self.assertRaises(ConfigurationError):
            load_requirement_source_from_full_payload(payload)

    def test_multiple_active_revisions_fail_loudly(self) -> None:
        payload = self._payload(
            [
                self._requirement("REQREV-A", "active", "trip at 70C"),
                self._requirement("REQREV-B", "active", "trip at 80C"),
            ]
        )
        with self.assertRaises(ConfigurationError):
            load_requirement_source_from_full_payload(payload)


if __name__ == "__main__":
    unittest.main()
