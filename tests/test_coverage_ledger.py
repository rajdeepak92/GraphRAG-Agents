from __future__ import annotations

import unittest

from multi_agentic_graph_rag.domain.schemas import (
    LLMChunkExtraction,
    LLMFactCandidate,
    LLMRequirementCandidate,
    RequirementDiscoveryOutput,
    SourceTrace,
)
from multi_agentic_graph_rag.services.coverage_ledger import CoverageLedger, LedgerEntry
from multi_agentic_graph_rag.services.requirement_builder import (
    derive_requirement_key,
    normalize_requirement_statement,
)


class CoverageLedgerTests(unittest.TestCase):
    def test_record_adds_entries_from_output(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)

        added = ledger.record(
            _output(
                _requirement("The system shall import files."),
                _requirement("The system shall export files."),
            )
        )

        self.assertEqual(added, 2)
        self.assertEqual(ledger.size, 2)

    def test_record_deduplicates_by_key_and_statement_not_key_alone(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)

        first = ledger.record(
            _output(_requirement("The system shall import files.", key="import_files"))
        )
        second = ledger.record(
            _output(_requirement("The system shall import files.", key="import_files"))
        )

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(ledger.size, 1)

    def test_same_key_changed_statement_is_retained_separately(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)

        ledger.record(
            _output(
                _requirement("The alarm shall trigger above 70 degrees.", key="alarm_threshold"),
                _requirement("The alarm shall trigger above 80 degrees.", key="alarm_threshold"),
            )
        )

        entries = ledger.select_for_chunk("chunk")
        self.assertEqual(ledger.size, 2)
        self.assertEqual(entries[0].requirement_key, entries[1].requirement_key)
        self.assertNotEqual(entries[0].normalized_statement, entries[1].normalized_statement)

    def test_fifo_eviction_respects_max_entries(self) -> None:
        ledger = CoverageLedger(max_entries=2, injection_top_k=2)

        ledger.record(_output(_requirement("First requirement statement.")))
        ledger.record(_output(_requirement("Second requirement statement.")))
        ledger.record(_output(_requirement("Third requirement statement.")))

        statements = [entry.statement for entry in ledger.select_for_chunk("chunk")]
        self.assertEqual(ledger.size, 2)
        self.assertEqual(
            statements,
            ["Second requirement statement.", "Third requirement statement."],
        )

    def test_select_returns_all_entries_when_within_top_k(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=5)
        ledger.record(
            _output(
                _requirement("Alpha requirement."),
                _requirement("Beta requirement."),
                _requirement("Gamma requirement."),
            )
        )

        entries = ledger.select_for_chunk("chunk")

        self.assertEqual(
            [entry.statement for entry in entries],
            ["Alpha requirement.", "Beta requirement.", "Gamma requirement."],
        )

    def test_select_without_embedder_returns_most_recent_k_in_order(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=2, embedder=None)
        ledger.record(
            _output(
                _requirement("Alpha requirement."),
                _requirement("Beta requirement."),
                _requirement("Gamma requirement."),
                _requirement("Delta requirement."),
            )
        )

        entries = ledger.select_for_chunk("chunk")

        self.assertEqual(
            [entry.statement for entry in entries],
            ["Gamma requirement.", "Delta requirement."],
        )

    def test_select_with_embedder_returns_top_k_by_cosine(self) -> None:
        table = {
            "Alpha requirement.": [1.0, 0.0, 0.0],
            "Beta requirement.": [0.0, 1.0, 0.0],
            "Gamma requirement.": [0.0, 0.0, 1.0],
            "find gamma then beta": [0.0, 0.3, 1.0],
        }
        embedder = _StubEmbedder(table)
        ledger = CoverageLedger(max_entries=10, injection_top_k=2, embedder=embedder)
        ledger.record(
            _output(
                _requirement("Alpha requirement."),
                _requirement("Beta requirement."),
                _requirement("Gamma requirement."),
            )
        )

        entries = ledger.select_for_chunk("find gamma then beta")

        self.assertEqual(
            [entry.statement for entry in entries],
            ["Gamma requirement.", "Beta requirement."],
        )

    def test_ledger_statement_embeddings_are_cached(self) -> None:
        table = {
            "Alpha requirement.": [1.0, 0.0, 0.0],
            "Beta requirement.": [0.0, 1.0, 0.0],
            "Gamma requirement.": [0.0, 0.0, 1.0],
            "chunk text": [0.2, 0.4, 1.0],
        }
        embedder = _StubEmbedder(table)
        ledger = CoverageLedger(max_entries=10, injection_top_k=2, embedder=embedder)
        ledger.record(
            _output(
                _requirement("Alpha requirement."),
                _requirement("Beta requirement."),
                _requirement("Gamma requirement."),
            )
        )

        ledger.select_for_chunk("chunk text")
        ledger.select_for_chunk("chunk text")

        for statement in ("Alpha requirement.", "Beta requirement.", "Gamma requirement."):
            self.assertEqual(embedder.embedded.count(statement), 1)
        # The chunk text is embedded once per call and is never cached.
        self.assertEqual(embedder.embedded.count("chunk text"), 2)

    def test_render_prompt_section_is_empty_for_no_entries(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)
        self.assertEqual(ledger.render_prompt_section([]), "")

    def test_render_prompt_section_omits_normalized_statement(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)
        entry = LedgerEntry(
            requirement_key="import_files",
            statement="The System Shall Import Files.",
            normalized_statement="the system shall import files.",
        )

        rendered = ledger.render_prompt_section([entry])

        self.assertIn("PREVIOUSLY DISCOVERED REQUIREMENTS", rendered)
        self.assertIn('"requirement_key":"import_files"', rendered)
        self.assertIn('"statement":"The System Shall Import Files."', rendered)
        self.assertNotIn("normalized_statement", rendered)

    def test_count_exact_converged_matches_only_exact_identity(self) -> None:
        ledger = CoverageLedger(max_entries=10, injection_top_k=10)
        recorded = _output(
            _requirement("The alarm shall trigger above 70 degrees.", key="alarm_threshold")
        )
        ledger.record(recorded)

        same = _output(
            _requirement("The alarm shall trigger above 70 degrees.", key="alarm_threshold")
        )
        changed = _output(
            _requirement("The alarm shall trigger above 80 degrees.", key="alarm_threshold")
        )

        self.assertEqual(ledger.count_exact_converged(same), 1)
        self.assertEqual(ledger.count_exact_converged(changed), 0)

    def test_injection_top_k_is_clamped_to_max_entries(self) -> None:
        ledger = CoverageLedger(max_entries=2, injection_top_k=50)
        ledger.record(
            _output(
                _requirement("Alpha requirement."),
                _requirement("Beta requirement."),
                _requirement("Gamma requirement."),
            )
        )

        entries = ledger.select_for_chunk("chunk")
        self.assertEqual(ledger.size, 2)
        self.assertEqual(len(entries), 2)

    def test_identity_matches_builder_normalizers(self) -> None:
        # Ledger identity must be derived with the same helpers the builder uses.
        statement = "The System Shall Import   Files."
        expected_key = derive_requirement_key(statement, "import_files")
        expected_norm = normalize_requirement_statement(statement)

        ledger = CoverageLedger(max_entries=10, injection_top_k=10)
        ledger.record(_output(_requirement(statement, key="import_files")))
        entry = ledger.select_for_chunk("chunk")[0]

        self.assertEqual(entry.requirement_key, expected_key)
        self.assertEqual(entry.normalized_statement, expected_norm)


class _StubEmbedder:
    provider_name = "stub"
    embedding_fingerprint = "stub"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.calls = 0
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded.extend(texts)
        return [self.table[text] for text in texts]


def _trace() -> SourceTrace:
    return SourceTrace(chunk_id="CHUNK-1", quote="quote", start_char=0, end_char=5)


def _requirement(statement: str, *, key: str | None = None) -> LLMRequirementCandidate:
    return LLMRequirementCandidate(
        temp_id="R1",
        statement=statement,
        requirement_key=key,
        source_trace=_trace(),
    )


def _output(*requirements: LLMRequirementCandidate) -> RequirementDiscoveryOutput:
    fact = LLMFactCandidate(
        temp_id="F1",
        text="fact text",
        source_trace=_trace(),
        requirements=list(requirements),
    )
    return RequirementDiscoveryOutput(chunks=[LLMChunkExtraction(chunk_id="CHUNK-1", facts=[fact])])


if __name__ == "__main__":
    unittest.main()
