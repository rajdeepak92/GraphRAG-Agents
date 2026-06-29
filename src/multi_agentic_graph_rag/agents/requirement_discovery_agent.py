"""Bounded requirement discovery component."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.errors import TraceValidationError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentManifest,
    RequirementDiscoveryOutput,
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
        "Extract factual statements and requirements from the chunks below. "
        "Use temporary fact keys F1.. and requirement keys R1... Every quote must be exact.\n"
        f"Project: {manifest.project}\n"
        f"Document version: {manifest.version}\n"
        f"{chunks}"
    )


def _verify_traces(
    manifest: DocumentManifest,
    output: RequirementDiscoveryOutput,
) -> None:
    chunks = {chunk.chunk_id: chunk.text for chunk in manifest.chunks}
    for fact in output.facts:
        trace = fact.source_trace
        chunk_text = chunks.get(trace.chunk_id)
        if chunk_text is None:
            raise TraceValidationError(f"unknown chunk_id in source trace: {trace.chunk_id}")
        if trace.quote not in chunk_text:
            raise TraceValidationError(f"source quote for {fact.temp_id} is not present in chunk")

    for requirement in output.requirements:
        trace = requirement.source_trace
        chunk_text = chunks.get(trace.chunk_id)
        if chunk_text is None:
            raise TraceValidationError(f"unknown chunk_id in source trace: {trace.chunk_id}")
        if trace.quote not in chunk_text:
            raise TraceValidationError(
                f"source quote for {requirement.temp_id} is not present in chunk"
            )
