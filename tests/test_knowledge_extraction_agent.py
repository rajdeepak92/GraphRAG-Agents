from __future__ import annotations

import unittest

from multi_agentic_graph_rag.agents.knowledge_extraction_agent import KnowledgeExtractionAgent
from multi_agentic_graph_rag.common_prompt_defs import PromptKnowledgeExtraction
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    KnowledgeExtractionChunkOutput,
    LLMExtractedAssertion,
    LLMExtractedEntity,
)

_CHUNK_TEXT = (
    "The gateway shall collect operating data from industrial equipment. "
    "The gateway must not exceed a polling interval of 5 seconds."
)
_ARTICLE_CHUNK_TEXT = (
    "The Smart Industrial IoT Monitoring & Control System supports Cloud Services."
)


class KnowledgeExtractionAgentTests(unittest.TestCase):
    def test_valid_output_is_grounded_and_normalized(self) -> None:
        reasoner = _FakeReasoner([_valid_output()])
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

        self.assertEqual(len(output.chunks), 1)
        candidates = output.chunks[0]
        self.assertEqual(
            [entity.normalized_name for entity in candidates.entities],
            ["gateway", "operating data"],
        )
        self.assertEqual(len(candidates.assertions), 2)
        collect = candidates.assertions[0]
        self.assertEqual(collect.subject_name, "gateway")
        self.assertEqual(collect.predicate, "COLLECTS")
        self.assertEqual(collect.object_name, "operating data")
        self.assertIsNone(collect.object_literal)
        self.assertEqual(collect.modality, "shall")
        trace = collect.source_trace
        self.assertEqual(trace.chunk_id, "CHUNK-1")
        self.assertIn(trace.quote, _CHUNK_TEXT)
        self.assertEqual(_CHUNK_TEXT[trace.start_char : trace.end_char], trace.quote)

        limit = candidates.assertions[1]
        self.assertEqual(limit.polarity, "negative")
        self.assertEqual(limit.modality, "must_not")
        self.assertEqual(limit.object_literal, "5 seconds")
        self.assertIsNone(limit.object_name)
        self.assertEqual(
            reasoner.system_messages,
            [PromptKnowledgeExtraction.SYS_PROMPT_KNOWLEDGE_EXTRACTION.value],
        )

    def test_ungrounded_quote_retries_once_with_feedback(self) -> None:
        bad = _valid_output()
        bad.assertions[0] = bad.assertions[0].model_copy(
            update={"quote": "This sentence is not in the chunk."}
        )
        reasoner = _FakeReasoner([bad, _valid_output()])
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

        self.assertEqual(len(reasoner.prompts), 2)
        self.assertIn("Validation error:", reasoner.prompts[1])
        self.assertIn("do not add or remove the, a, or an", reasoner.prompts[1])
        self.assertEqual(len(output.chunks[0].assertions), 2)

    def test_second_failure_raises_model_output_error(self) -> None:
        bad = _valid_output()
        bad.assertions[0] = bad.assertions[0].model_copy(
            update={"quote": "This sentence is not in the chunk."}
        )
        reasoner = _FakeReasoner([bad, bad])
        agent = KnowledgeExtractionAgent(reasoner)

        with self.assertRaises(ModelOutputError):
            agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

    def test_unknown_subject_is_rejected(self) -> None:
        bad = _valid_output()
        bad.assertions[0] = bad.assertions[0].model_copy(update={"subject": "cloud service"})
        reasoner = _FakeReasoner([bad, bad])
        agent = KnowledgeExtractionAgent(reasoner)

        with self.assertRaises(ModelOutputError) as caught:
            agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])
        self.assertIn("TraceValidationError: <message redacted", str(caught.exception))

    def test_entity_missing_from_chunk_is_rejected(self) -> None:
        bad = _valid_output()
        bad.entities.append(LLMExtractedEntity(name="billing engine", entity_type="system"))
        reasoner = _FakeReasoner([bad, bad])
        agent = KnowledgeExtractionAgent(reasoner)

        with self.assertRaises(ModelOutputError) as caught:
            agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])
        self.assertIn("TraceValidationError: <message redacted", str(caught.exception))

    def test_self_loop_assertion_is_rejected(self) -> None:
        bad = _valid_output()
        bad.assertions[0] = bad.assertions[0].model_copy(update={"object_name": "gateway"})
        reasoner = _FakeReasoner([bad, bad])
        agent = KnowledgeExtractionAgent(reasoner)

        with self.assertRaises(ModelOutputError) as caught:
            agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])
        self.assertIn("TraceValidationError: <message redacted", str(caught.exception))

    def test_undeclared_object_is_demoted_to_literal(self) -> None:
        out = _valid_output()
        # "cloud service" is used as an object but never declared in entities[].
        out.assertions.append(
            LLMExtractedAssertion(
                subject="gateway",
                predicate="reports to",
                object_name="cloud service",
                modality="shall",
                quote="The gateway shall collect operating data from industrial equipment.",
                confidence=0.9,
            )
        )
        reasoner = _FakeReasoner([out])
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

        demoted = [a for a in output.chunks[0].assertions if a.object_literal == "cloud service"]
        self.assertEqual(len(demoted), 1)
        self.assertIsNone(demoted[0].object_name)
        # No retry needed: the undeclared object no longer fails the chunk.
        self.assertEqual(len(reasoner.prompts), 1)

    def test_unknown_subject_still_rejected_after_leniency(self) -> None:
        bad = _valid_output()
        bad.assertions[0] = bad.assertions[0].model_copy(update={"subject": "cloud service"})
        reasoner = _FakeReasoner([bad, bad])
        agent = KnowledgeExtractionAgent(reasoner)
        with self.assertRaises(ModelOutputError):
            agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

    def test_leading_the_subject_resolves_to_declared_entity(self) -> None:
        reasoner = _FakeReasoner(
            [
                _article_output(
                    subject="The Smart Industrial IoT Monitoring & Control System",
                    object_name="Cloud Services",
                )
            ]
        )
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_article_chunk()])

        assertion = output.chunks[0].assertions[0]
        self.assertEqual(
            assertion.subject_name,
            "smart industrial iot monitoring & control system",
        )
        self.assertEqual(assertion.object_name, "cloud services")
        self.assertEqual(len(reasoner.prompts), 1)

    def test_leading_the_object_resolves_to_declared_entity(self) -> None:
        reasoner = _FakeReasoner(
            [
                _article_output(
                    subject="Smart Industrial IoT Monitoring & Control System",
                    object_name="The Cloud Services",
                )
            ]
        )
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_article_chunk()])

        assertion = output.chunks[0].assertions[0]
        self.assertEqual(assertion.object_name, "cloud services")
        self.assertIsNone(assertion.object_literal)

    def test_exact_entity_reference_precedes_leading_the_compatibility(self) -> None:
        text = "The Control System coordinates Control System."
        chunk = _chunk_with_text(text)
        output = KnowledgeExtractionChunkOutput(
            entities=[
                LLMExtractedEntity(name="The Control System", entity_type="system"),
                LLMExtractedEntity(name="Control System", entity_type="system"),
            ],
            assertions=[
                LLMExtractedAssertion(
                    subject="The Control System",
                    predicate="coordinates",
                    object_name="Control System",
                    quote=text,
                    confidence=0.9,
                )
            ],
        )
        agent = KnowledgeExtractionAgent(_FakeReasoner([output]))

        result = agent.run(project="PROJECT", version="1.0", chunks=[chunk])

        assertion = result.chunks[0].assertions[0]
        self.assertEqual(assertion.subject_name, "the control system")
        self.assertEqual(assertion.object_name, "control system")

    def test_indefinite_articles_are_not_relaxed(self) -> None:
        for article in ("A", "An"):
            with self.subTest(article=article):
                bad = _valid_output()
                bad.assertions[0] = bad.assertions[0].model_copy(
                    update={"subject": f"{article} gateway"}
                )
                agent = KnowledgeExtractionAgent(_FakeReasoner([bad, bad]))

                with self.assertRaises(ModelOutputError):
                    agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

    def test_empty_chunk_output_is_allowed(self) -> None:
        reasoner = _FakeReasoner([KnowledgeExtractionChunkOutput()])
        agent = KnowledgeExtractionAgent(reasoner)

        output = agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

        self.assertEqual(output.chunks[0].entities, [])
        self.assertEqual(output.chunks[0].assertions, [])

    def test_objectless_assertion_is_allowed(self) -> None:
        # An intransitive/objectless claim (neither object_name nor object_literal)
        # must be accepted end-to-end: the schema no longer enforces "exactly one
        # object" (Azure strict outputs cannot), and the agent reconciles it to an
        # objectless candidate rather than failing the whole chunk.
        output = KnowledgeExtractionChunkOutput(
            entities=[LLMExtractedEntity(name="gateway", entity_type="system")],
            assertions=[
                LLMExtractedAssertion(
                    subject="gateway",
                    predicate="restarts",
                    modality="shall",
                    quote="The gateway shall collect operating data from industrial equipment.",
                    confidence=0.9,
                )
            ],
        )
        reasoner = _FakeReasoner([output])
        agent = KnowledgeExtractionAgent(reasoner)

        result = agent.run(project="PROJECT", version="1.0", chunks=[_chunk()])

        assertion = result.chunks[0].assertions[0]
        self.assertEqual(assertion.subject_name, "gateway")
        self.assertEqual(assertion.predicate, "RESTARTS")
        self.assertIsNone(assertion.object_name)
        self.assertIsNone(assertion.object_literal)


class _FakeReasoner:
    provider_name = "fake"

    def __init__(self, outputs: list[KnowledgeExtractionChunkOutput]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.system_messages: list[str] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[KnowledgeExtractionChunkOutput],
        system_message: str,
        **_: object,
    ) -> KnowledgeExtractionChunkOutput:
        self.prompts.append(prompt)
        self.system_messages.append(system_message)
        return self.outputs.pop(0)


def _chunk() -> DocumentChunk:
    return _chunk_with_text(_CHUNK_TEXT)


def _article_chunk() -> DocumentChunk:
    return _chunk_with_text(_ARTICLE_CHUNK_TEXT)


def _chunk_with_text(text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="CHUNK-1",
        ordinal=1,
        text=text,
        normalized_text=text.lower(),
        page=3,
        section="Monitoring",
        start_char=0,
        end_char=len(text),
        source_block_ids=["BLOCK-1"],
    )


def _article_output(*, subject: str, object_name: str) -> KnowledgeExtractionChunkOutput:
    return KnowledgeExtractionChunkOutput(
        entities=[
            LLMExtractedEntity(
                name="Smart Industrial IoT Monitoring & Control System",
                entity_type="system",
            ),
            LLMExtractedEntity(name="Cloud Services", entity_type="system"),
        ],
        assertions=[
            LLMExtractedAssertion(
                subject=subject,
                predicate="supports",
                object_name=object_name,
                quote=_ARTICLE_CHUNK_TEXT,
                confidence=0.99,
            )
        ],
    )


def _valid_output() -> KnowledgeExtractionChunkOutput:
    return KnowledgeExtractionChunkOutput(
        entities=[
            LLMExtractedEntity(name="gateway", entity_type="system"),
            LLMExtractedEntity(name="operating data", entity_type="data_object"),
        ],
        assertions=[
            LLMExtractedAssertion(
                subject="gateway",
                predicate="collects",
                object_name="operating data",
                modality="shall",
                quote="The gateway shall collect operating data from industrial equipment.",
                confidence=0.95,
            ),
            LLMExtractedAssertion(
                subject="gateway",
                predicate="exceeds polling interval",
                object_literal="5 seconds",
                modality="must_not",
                polarity="negative",
                quote="The gateway must not exceed a polling interval of 5 seconds.",
                confidence=0.9,
            ),
        ],
    )


if __name__ == "__main__":
    unittest.main()
