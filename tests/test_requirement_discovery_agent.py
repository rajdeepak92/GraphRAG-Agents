from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.agents.requirement_discovery_agent import RequirementDiscoveryAgent
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    RequirementDiscoveryChunkOutput,
)
from multi_agentic_graph_rag.services.requirement_builder import build_requirement_artifact

T = TypeVar("T", bound=BaseModel)


class RequirementDiscoveryAgentTests(unittest.TestCase):
    def test_chunk_local_discovery_calls_each_chunk_individually(self) -> None:
        manifest = _manifest(["The system shall import files.", "The system shall export files."])
        reasoner = _ChunkReasoner(discovery_batch_size=2)

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        self.assertEqual(reasoner.prompts, 2)
        self.assertEqual([chunk.chunk_id for chunk in output.chunks], ["CHUNK-1", "CHUNK-2"])

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
                        _requirement("The system shall import files."),
                        _requirement("The system shall export files."),
                    ],
                )
            ]
        )

        output = RequirementDiscoveryAgent(reasoner).run(manifest)

        fact = output.chunks[0].facts[0]
        self.assertEqual(len(fact.requirements), 2)
        self.assertEqual(fact.requirements[0].temp_id, "R1")
        self.assertEqual(fact.requirements[1].temp_id, "R2")

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


class _ChunkReasoner:
    provider_name = "huggingface"

    def __init__(self, *, discovery_batch_size: int | None) -> None:
        if discovery_batch_size is not None:
            self.discovery_batch_size = discovery_batch_size
        self.prompts = 0
        self.last_prompt = ""

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        self.last_prompt = prompt
        chunk_text = _chunk_text(prompt)
        quote = _requirement_quote(chunk_text)
        return schema.model_validate({"facts": [_fact(quote)]})


class _StaticReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        return schema.model_validate({"facts": self.facts})


class _QuoteRetryReasoner:
    provider_name = "huggingface"
    discovery_batch_size = 1

    def __init__(self) -> None:
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
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

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
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

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        self._last_response = f"raw response {self.prompts}"
        return schema.model_validate({"facts": [_fact("not present in the source chunk")]})

    def persist_last_response(self, *, filename: str) -> Path:
        path = self.run_dir / filename
        path.write_text(self._last_response, encoding="utf-8")
        self.last_response_path = path
        return path


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


def _requirement(req_text: str) -> dict[str, str]:
    return {
        "req_text": req_text,
        "requirement_type": "functional",
        "priority": "medium",
        "requirement_key": "system behavior",
    }


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
