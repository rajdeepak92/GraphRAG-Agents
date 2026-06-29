"""Bounded requirement discovery component."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.errors import TraceValidationError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    RequirementDiscoveryOutput,
    SourceTrace,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel


class RequirementDiscoveryAgent:
    def __init__(self, reasoning_model: ReasoningModel) -> None:
        self.reasoning_model = reasoning_model

    def run(self, manifest: DocumentManifest) -> RequirementDiscoveryOutput:
        prompt = _build_prompt(manifest)
        output = self.reasoning_model.generate_structured(
            prompt=prompt,
            schema=RequirementDiscoveryOutput,
        )
        _verify_traces(manifest, output)
        return output


def _build_prompt(manifest: DocumentManifest) -> str:
    chunks = "\n\n".join(
        f'<chunk id="{chunk.chunk_id}">\n{chunk.text}\n</chunk>' for chunk in manifest.chunks
    )
    return (
        "You are extracting requirement traceability data from document chunks.\n"
        "Return only one JSON object. Do not include markdown, commentary, or extra keys.\n"
        'The root object must be exactly {"chunks": [...]}.\n'
        "Each chunks[] item must have chunk_id and facts[]. Each fact must stay under the "
        "chunk where it was found, even if the same fact text appears in another chunk. "
        "Each fact must include requirements[] derived from that fact. Do not merge fact "
        "occurrences across chunks.\n"
        "Every source_trace.chunk_id must match the parent chunk_id. Every quote must be an "
        "exact substring of the chunk text, and start_char/end_char must be offsets within "
        "that chunk. Use null for page and section if they are unknown.\n"
        "Use temporary IDs that are unique within their parent chunk, such as F1 and R1. "
        "Set requirement_key to a stable functional identity that excludes revision values "
        "such as thresholds, dates, amounts, or temperatures when possible.\n"
        "Sample output shape:\n"
        "{\n"
        '  "chunks": [\n'
        "    {\n"
        '      "chunk_id": "CHUNK-0001-EXAMPLE",\n'
        '      "facts": [\n'
        "        {\n"
        '          "temp_id": "F1",\n'
        '          "text": "The system shall keep the room below 70 F.",\n'
        '          "source_trace": {\n'
        '            "chunk_id": "CHUNK-0001-EXAMPLE",\n'
        '            "quote": "The system shall keep the room below 70 F.",\n'
        '            "start_char": 0,\n'
        '            "end_char": 42,\n'
        '            "page": null,\n'
        '            "section": null\n'
        "          },\n"
        '          "requirements": [\n'
        "            {\n"
        '              "temp_id": "R1",\n'
        '              "statement": "The system shall keep the room below 70 F.",\n'
        '              "requirement_type": "functional",\n'
        '              "priority": "medium",\n'
        '              "requirement_key": "system keep room below temperature threshold",\n'
        '              "source_trace": {\n'
        '                "chunk_id": "CHUNK-0001-EXAMPLE",\n'
        '                "quote": "The system shall keep the room below 70 F.",\n'
        '                "start_char": 0,\n'
        '                "end_char": 42,\n'
        '                "page": null,\n'
        '                "section": null\n'
        "              }\n"
        "            }\n"
        "          ]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Project: {manifest.project}\n"
        f"Document version: {manifest.version}\n"
        f"{chunks}"
    )


def _verify_traces(
    manifest: DocumentManifest,
    output: RequirementDiscoveryOutput,
) -> None:
    chunks = {chunk.chunk_id: chunk for chunk in manifest.chunks}
    for chunk_output in output.chunks:
        chunk = chunks.get(chunk_output.chunk_id)
        if chunk is None:
            raise TraceValidationError(f"unknown chunk_id in output: {chunk_output.chunk_id}")
        for fact in chunk_output.facts:
            _verify_trace(chunk, fact.source_trace, f"fact {fact.temp_id}")
            for requirement in fact.requirements:
                _verify_trace(chunk, requirement.source_trace, f"requirement {requirement.temp_id}")


def _verify_trace(chunk: DocumentChunk, trace: SourceTrace, label: str) -> None:
    if trace.chunk_id != chunk.chunk_id:
        raise TraceValidationError(
            f"{label} source trace points to {trace.chunk_id}, expected {chunk.chunk_id}"
        )
    if trace.quote not in chunk.text:
        raise TraceValidationError(f"source quote for {label} is not present in chunk")
    if trace.end_char > len(chunk.text):
        raise TraceValidationError(f"source span for {label} exceeds chunk length")
    if chunk.text[trace.start_char : trace.end_char] != trace.quote:
        raise TraceValidationError(f"source span for {label} does not match quote")
    if trace.page is None:
        trace.page = chunk.page
    if trace.section is None:
        trace.section = chunk.section
