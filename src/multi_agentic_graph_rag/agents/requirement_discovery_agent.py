"""Bounded requirement discovery component."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from multi_agentic_graph_rag.domain.errors import ModelOutputError, TraceValidationError
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    LLMChunkExtraction,
    LLMDiscoveredFact,
    LLMFactCandidate,
    LLMRequirementCandidate,
    RequirementDiscoveryChunkOutput,
    RequirementDiscoveryOutput,
    SourceTrace,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel

_REQUIREMENT_HINT = re.compile(
    r"\b(?:shall|must|required|requires|requirement|acceptance criteria|"
    r"non-functional requirements|BR-[A-Z0-9/-]+|AC-[A-Z0-9/-]+|"
    r"FR-[A-Z0-9/-]+|NFR(?:-[A-Z0-9/-]+)?)\b",
    re.I,
)


@dataclass(frozen=True)
class _NormalizedChunk:
    chunk: DocumentChunk
    text: str
    char_map: list[int]


class RequirementDiscoveryAgent:
    def __init__(self, reasoning_model: ReasoningModel, *, logger: Any | None = None) -> None:
        self.reasoning_model = reasoning_model
        self.logger = logger

    def run(self, manifest: DocumentManifest) -> RequirementDiscoveryOutput:
        outputs: list[RequirementDiscoveryOutput] = []
        for chunk_index, chunk in enumerate(manifest.chunks, start=1):
            chunk_manifest = manifest.model_copy(update={"chunks": [chunk]})
            outputs.append(
                self._run_chunk(
                    chunk_manifest,
                    chunk=chunk,
                    chunk_index=chunk_index,
                )
            )
        return _merge_outputs(outputs)

    def _run_chunk(
        self,
        manifest: DocumentManifest,
        *,
        chunk: DocumentChunk,
        chunk_index: int,
    ) -> RequirementDiscoveryOutput:
        context = _normalized_context(chunk)
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                prompt = _build_prompt(
                    manifest,
                    context,
                    validation_error=validation_error,
                )
                _set_response_context(
                    self.reasoning_model,
                    batch_index=chunk_index,
                    attempt=attempt,
                    chunk_ids=[chunk.chunk_id],
                )
                chunk_output = self.reasoning_model.generate_structured(
                    prompt=prompt,
                    schema=RequirementDiscoveryChunkOutput,
                )
                try:
                    output = _chunk_to_nested(context, chunk_output)
                    _verify_traces(manifest, output)
                except TraceValidationError as error:
                    response_path = _persist_last_response(
                        self.reasoning_model,
                        batch_index=chunk_index,
                        attempt=attempt,
                    )
                    self._log_validation_failure(
                        chunk_index=chunk_index,
                        chunk_id=chunk.chunk_id,
                        attempt=attempt,
                        error=error,
                        response_path=response_path,
                    )
                    if attempt == 2:
                        raise ModelOutputError(
                            "Requirement discovery chunk "
                            f"{chunk_index} failed validation after retry "
                            f"for {chunk.chunk_id}: {error}; raw_response_path={response_path}"
                        ) from error
                    validation_error = str(error)
                    continue

                response_path = _last_response_path(self.reasoning_model)
                self._log_chunk_completed(
                    chunk_index=chunk_index,
                    chunk_id=chunk.chunk_id,
                    fact_count=len(chunk_output.facts),
                    requirement_count=_output_requirement_count(output),
                    retry_count=attempt - 1,
                    response_path=response_path,
                )
                return output
        finally:
            _clear_response_context(self.reasoning_model)

        raise ModelOutputError(
            f"Requirement discovery chunk {chunk_index} did not produce a result"
        )

    def _log_chunk_completed(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        fact_count: int,
        requirement_count: int,
        retry_count: int,
        response_path: str | None,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "Requirement discovery chunk completed",
            step="discover_requirements.chunk",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            fact_count=fact_count,
            requirement_count=requirement_count,
            retry_count=retry_count,
            raw_response_path=response_path,
            status="completed",
        )

    def _log_validation_failure(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        attempt: int,
        error: TraceValidationError,
        response_path: str | None,
    ) -> None:
        if self.logger is None:
            return
        self.logger.warning(
            "Requirement discovery chunk failed validation",
            step="discover_requirements.chunk",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            attempt=attempt,
            retry_count=attempt - 1,
            error=str(error),
            raw_response_path=response_path,
            status="failed" if attempt == 2 else "retrying",
        )


def _build_prompt(
    manifest: DocumentManifest,
    chunk: _NormalizedChunk,
    validation_error: str | None = None,
) -> str:
    chunk_json = json.dumps(
        {
            "chunk_id": chunk.chunk.chunk_id,
            "chunk_text": chunk.text,
        },
        ensure_ascii=False,
        indent=2,
    )

    feedback = ""
    if validation_error:
        feedback = (
            "Previous output failed validation. Return one corrected JSON object for "
            "the same chunk only.\n"
            "Repair the invalid output by using quotes copied exactly from the normalized "
            "chunk_text.\n"
            "If a quote failed because it merged a heading with a bullet, table cell, "
            "or nearby line, do not synthesize a combined quote. Instead, choose the "
            "smallest exact contiguous source span that exists in chunk_text.\n"
            "If no exact quote can be copied for a fact, remove that fact from the output.\n"
            f"Validation error: {validation_error}\n\n"
        )

    return (
        "You are extracting requirement traceability data from exactly one document chunk.\n"
        "Return exactly one valid JSON object and no other text. Do not include markdown, "
        "code fences, commentary, XML tags, hidden reasoning, or explanations.\n\n"
        f"{feedback}"
        "Input is one JSON object with chunk_id and normalized chunk_text. The chunk ID, "
        "page, section, source offsets, and permanent IDs are owned by Python and must "
        "not be returned.\n\n"
        "Output schema:\n"
        "{\n"
        '  "facts": [\n'
        "    {\n"
        '      "fact_id": "F1",\n'
        '      "fact_text": "...",\n'
        '      "quote": "...",\n'
        '      "requirements": [\n'
        "        {\n"
        '          "req_id": "R1",\n'
        '          "req_text": "...",\n'
        '          "requirement_type": "...",\n'
        '          "priority": "Medium",\n'
        '          "requirement_key": "..."\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Required root field: facts.\n"
        "Each fact entry must contain exactly these fields: fact_id, fact_text, quote, "
        "requirements.\n"
        "Each requirement entry must contain exactly these fields: req_id, req_text, "
        "requirement_type, priority, requirement_key.\n"
        "All returned field values must be JSON strings except facts and requirements, "
        "which must be JSON arrays.\n"
        "Never return null for fact_id, fact_text, quote, requirements, req_id, req_text, "
        "requirement_type, priority, or requirement_key.\n"
        "Use temporary fact_id and req_id values only, such as F1, F2, R1, and R2. "
        "Python will replace them with permanent IDs.\n\n"
        "Primary task:\n"
        "Analyze the entire input chunk_text for the provided chunk_id. Extract complete, "
        "meaningful business requirement traceability facts that can later be used to "
        "create requirements, user stories, scenarios, and test cases.\n\n"
        "For example, if chunk_id is CHUNK-07, use CHUNK-07 only to scope analysis; "
        "never return it as a generated identifier or requirement text.\n\n"
        "Relevant source items include requirements, constraints, business rules, "
        "acceptance criteria, non-functional requirements, scope items, out-of-scope "
        "items, capabilities, system behavior, configuration rules, validation rules, "
        "alerting rules, health rules, data-quality rules, offline behavior, and "
        "application behavior.\n\n"
        "Hard traceability rule for quote:\n"
        "quote must be copied exactly from normalized chunk_text.\n"
        "quote must be an exact contiguous substring that can be found in chunk_text after "
        "whitespace normalization.\n"
        "Do not paraphrase quote.\n"
        "Do not improve quote.\n"
        "Do not add words to quote.\n"
        "Do not remove words from the middle of quote.\n"
        "Do not merge a heading with a bullet, table row, clause, or nearby line unless "
        "that exact merged text appears contiguously in chunk_text.\n"
        "Do not create artificial quote text such as 'Heading: bullet text' unless that "
        "exact text appears in chunk_text.\n"
        "If the source is a bullet item, quote the bullet item body exactly as it appears, "
        "excluding only the bullet marker when the marker is not needed.\n"
        "If the source is a table row, quote the smallest exact useful text from that row.\n"
        "If the source is a heading-body pair, quote the smallest exact useful body text "
        "or exact contiguous heading-body span that appears in chunk_text. Never invent "
        "a colon-joined heading-body quote.\n"
        "If no exact quote can be copied for a fact, do not return that fact.\n\n"
        "Before returning the final JSON, internally verify every quote against chunk_text. "
        "If any quote is not locatable, repair it using an exact shorter quote. If it "
        "still cannot be repaired, remove that fact. Do not describe this verification.\n\n"
        "fact_text rules:\n"
        "fact_text must preserve the smallest meaningful source text.\n"
        "fact_text may remove only leading source identifiers, row codes, numbering, "
        "bullets, or labels such as BR-SEN-001, AC-001, FR-001, NFR-001, or section "
        "numbers when they are not part of the requirement meaning.\n"
        "Preserve the remaining source wording.\n"
        "Do not add meaning, domain nouns, actors, conditions, limits, purposes, causes, "
        "consequences, implementation details, or test details that are not present in "
        "the source text.\n\n"
        "req_text rules:\n"
        "req_text must be a complete, meaningful business requirement sentence.\n"
        "req_text must never be only an identifier, label, heading, placeholder, or source "
        "row code.\n"
        "req_text must not contain source identifiers, row codes, bullets, numbering, "
        "markdown, labels, headings, placeholders, or unnecessary symbols.\n"
        "do not copy the source identifier into req_text.\n"
        "Do not copy source identifiers such as BR-SEN-001, AC-001, FR-001, or NFR-001 "
        "into req_text.\n"
        "If the source text is a valid single requirement and is grammatically correct, "
        "return it as req_text without changing it.\n"
        "If the source text is a valid single requirement but is not grammatically "
        "correct, make only the smallest grammar correction required. Do not alter the "
        "meaning.\n"
        "If a relevant fact does not contain a separate derived requirement, return one "
        "requirement whose req_text preserves fact_text as much as possible.\n"
        "If grammar is incomplete, add only the minimum words needed to make the sentence "
        "valid. Do not add words that change the meaning.\n\n"
        "Splitting rules:\n"
        "If one source text contains multiple requirements in a list or coordinated "
        "sentence, split it into separate requirement records.\n"
        "For each split requirement, reuse the shared source subject and shared source "
        "predicate, then attach exactly one listed item. Preserve the listed item wording.\n"
        "Do not rewrite verbs unless required for minimal grammar correction.\n"
        "Example: if the source says 'The system supports real-time monitoring, "
        "threshold-based alerts, rule-based control, cloud reporting, and maintenance "
        "planning.', emit these req_text values: 'The system supports real-time "
        "monitoring.', 'The system supports threshold-based alerts.', 'The system supports "
        "rule-based control.', 'The system supports cloud reporting.', and 'The system "
        "supports maintenance planning.'. Do not rewrite 'supports' as 'shall support'.\n\n"
        "Explicit row rules:\n"
        "Do not merge explicit BR-*, AC-*, FR-*, NFR-*, or similar rows into summaries.\n"
        "Emit each explicit row as its own requirement record.\n"
        "Remove the source identifier from req_text, but preserve the remaining row body "
        "as closely as possible.\n\n"
        "requirement_type rules:\n"
        "Set requirement_type from the source category when clear.\n"
        "Use exactly one JSON string. Allowed values include: Functional Requirement, "
        "Business Requirement, Acceptance Criteria, Non-Functional Requirement, Security "
        "Requirement, Configuration Requirement, Validation Requirement, Alerting "
        "Requirement, Health Requirement, Data Quality Requirement, Application "
        "Requirement, Offline Requirement, Scope Requirement, or Out-of-Scope Requirement.\n"
        "If unclear, use Functional Requirement.\n\n"
        "priority rules:\n"
        "priority must always be exactly one of: High, Medium, Low.\n"
        "Use High only for explicit critical, mandatory, safety, security, reliability, "
        "or data-loss language.\n"
        "Use Low only for explicit optional, future, nice-to-have, or out-of-scope "
        "language.\n"
        "Otherwise use Medium.\n\n"
        "requirement_key rules:\n"
        "requirement_key must be a stable functional identity.\n"
        "requirement_key must not be null.\n"
        "requirement_key must not be a source identifier.\n"
        "Exclude revision values such as thresholds, dates, amounts, counts, and "
        "temperatures when possible.\n"
        "Use lowercase snake_case when possible.\n\n"
        "Return facts as an empty list only when the chunk has no relevant facts, "
        "requirements, constraints, business rules, acceptance criteria, scope items, "
        "out-of-scope items, capabilities, system behaviors, or non-functional "
        "requirements.\n\n"
        f"Project: {manifest.project}\n"
        f"Document version: {manifest.version}\n"
        f"Input chunk JSON:\n{chunk_json}"
    )


def _normalized_context(chunk: DocumentChunk) -> _NormalizedChunk:
    normalized_text, char_map = _normalize_for_prompt(chunk.text)
    return _NormalizedChunk(chunk=chunk, text=normalized_text, char_map=char_map)


def _normalize_for_prompt(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    char_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if not chars:
                continue
            if _is_split_identifier_boundary(chars, text, index):
                continue
            if not in_whitespace:
                chars.append(" ")
                char_map.append(index)
                in_whitespace = True
            continue
        chars.append(char)
        char_map.append(index)
        in_whitespace = False
    if chars and chars[-1] == " ":
        chars.pop()
        char_map.pop()
    return "".join(chars), char_map


def _is_split_identifier_boundary(chars: list[str], text: str, whitespace_index: int) -> bool:
    next_char = _next_non_whitespace(text, whitespace_index)
    if next_char is None or not next_char.isalnum():
        return False
    prefix = "".join(chars[-24:])
    return bool(re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*-$", prefix, re.I))


def _next_non_whitespace(text: str, start_index: int) -> str | None:
    for char in text[start_index + 1 :]:
        if not char.isspace():
            return char
    return None


def _chunk_to_nested(
    context: _NormalizedChunk,
    output: RequirementDiscoveryChunkOutput,
) -> RequirementDiscoveryOutput:
    if not output.facts:
        if _chunk_looks_requirement_bearing(context.text):
            raise TraceValidationError(
                f"chunk returned facts=[] but appears requirement-bearing: {context.chunk.chunk_id}"
            )
        return RequirementDiscoveryOutput(chunks=[])

    facts = [_fact_to_candidate(context, fact) for fact in output.facts]
    return RequirementDiscoveryOutput(
        chunks=[LLMChunkExtraction(chunk_id=context.chunk.chunk_id, facts=facts)]
    )


def _fact_to_candidate(
    context: _NormalizedChunk,
    fact: LLMDiscoveredFact,
) -> LLMFactCandidate:
    trace = _source_trace_from_quote(fact, context)
    requirements = [
        LLMRequirementCandidate(
            temp_id=requirement.req_id,
            statement=requirement.req_text,
            requirement_type=requirement.requirement_type,
            priority=requirement.priority,
            requirement_key=requirement.requirement_key,
            source_trace=trace.model_copy(),
        )
        for requirement in fact.requirements
    ]
    return LLMFactCandidate(
        temp_id=fact.fact_id,
        text=fact.fact_text,
        source_trace=trace,
        requirements=requirements,
    )


def _source_trace_from_quote(fact: LLMDiscoveredFact, context: _NormalizedChunk) -> SourceTrace:
    normalized_quote, _ = _normalize_for_prompt(fact.quote)
    if not normalized_quote:
        raise TraceValidationError(f"empty source quote for fact {fact.fact_id}")

    normalized_start = context.text.find(normalized_quote)
    if normalized_start < 0:
        raise TraceValidationError(
            f"source quote for fact {fact.fact_id} cannot be located in "
            f"{context.chunk.chunk_id} after normalization"
        )

    normalized_end = normalized_start + len(normalized_quote)
    if normalized_start >= len(context.char_map) or normalized_end > len(context.char_map):
        raise TraceValidationError(
            f"source quote for fact {fact.fact_id} cannot be mapped to raw text"
        )

    raw_start = context.char_map[normalized_start]
    raw_end = context.char_map[normalized_end - 1] + 1
    raw_quote = context.chunk.text[raw_start:raw_end]
    return SourceTrace(
        chunk_id=context.chunk.chunk_id,
        quote=raw_quote,
        start_char=raw_start,
        end_char=raw_end,
        page=context.chunk.page,
        section=context.chunk.section,
    )


def _chunk_looks_requirement_bearing(text: str) -> bool:
    return bool(_REQUIREMENT_HINT.search(text))


def _set_response_context(
    reasoning_model: ReasoningModel,
    *,
    batch_index: int,
    attempt: int,
    chunk_ids: list[str],
) -> None:
    setter = getattr(reasoning_model, "set_response_context", None)
    if callable(setter):
        setter(batch_index=batch_index, attempt=attempt, chunk_ids=chunk_ids)


def _clear_response_context(reasoning_model: ReasoningModel) -> None:
    clearer = getattr(reasoning_model, "clear_response_context", None)
    if callable(clearer):
        clearer()


def _persist_last_response(
    reasoning_model: ReasoningModel,
    *,
    batch_index: int,
    attempt: int,
) -> str | None:
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        return _last_response_path(reasoning_model)
    path = persister(filename=f"llm_response_{batch_index}_{attempt}.txt")
    return str(path) if path is not None else None


def _last_response_path(reasoning_model: ReasoningModel) -> str | None:
    response_path = getattr(reasoning_model, "last_response_path", None)
    return str(response_path) if response_path else None


def _merge_outputs(outputs: Iterable[RequirementDiscoveryOutput]) -> RequirementDiscoveryOutput:
    chunk_outputs: list[LLMChunkExtraction] = []
    for output in outputs:
        chunk_outputs.extend(output.chunks)
    return RequirementDiscoveryOutput(chunks=chunk_outputs)


def _output_requirement_count(output: RequirementDiscoveryOutput) -> int:
    return sum(
        len(fact.requirements) for chunk_output in output.chunks for fact in chunk_output.facts
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
