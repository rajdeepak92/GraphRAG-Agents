"""Bounded requirement discovery component."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from multi_agentic_graph_rag.common_prompt_defs import (
    PromptRequirementDiscovery,
    PromptSharedFragments,
)
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
from multi_agentic_graph_rag.services.coverage_ledger import CoverageLedger, LedgerEntry

_REQUIREMENT_HINT = re.compile(
    r"\b(?:shall|must|required|requires|requirement|acceptance criteria|"
    r"non-functional requirements|BR-[A-Z0-9/-]+|AC-[A-Z0-9/-]+|"
    r"FR-[A-Z0-9/-]+|NFR(?:-[A-Z0-9/-]+)?|SYS[-_ ]REQ[-_ ][A-Z0-9/-]+)\b",
    re.I,
)
_QUOTE_LEADING_LABELS = re.compile(
    r"^(?:"
    r"acceptance criteria|business requirement|functional requirement|"
    r"non-functional requirement|application requirement|security requirement|"
    r"area requirement|"
    r"reliability|availability|performance|scalability|maintainability|"
    r"security|embedded safety|alerts and notifications|rule-based automation|"
    r"startup and configuration|request and response validation|"
    r"offline storage and recovery|health and data status"
    r")\s+",
    re.I,
)


@dataclass(frozen=True)
class _NormalizedChunk:
    chunk: DocumentChunk
    text: str
    char_map: list[int]


class RequirementDiscoveryAgent:
    def __init__(
        self,
        reasoning_model: ReasoningModel,
        *,
        logger: Any | None = None,
        coverage_ledger: CoverageLedger | None = None,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.logger = logger
        self.coverage_ledger = coverage_ledger

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
        ledger_entries: list[LedgerEntry] = []
        ledger_section = ""
        if self.coverage_ledger is not None:
            ledger_entries = self.coverage_ledger.select_for_chunk(context.text)
            ledger_section = self.coverage_ledger.render_prompt_section(ledger_entries)
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                prompt = _build_prompt(
                    manifest,
                    context,
                    validation_error=validation_error,
                    ledger_section=ledger_section,
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
                    _validate_candidate_semantics(chunk_output)
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
                if self.coverage_ledger is not None:
                    converged_count = self.coverage_ledger.count_exact_converged(output)
                    new_ledger_entries = self.coverage_ledger.record(output)
                    self._log_ledger_state(
                        chunk_index=chunk_index,
                        chunk_id=chunk.chunk_id,
                        ledger_size=self.coverage_ledger.size,
                        injected_count=len(ledger_entries),
                        exact_converged_count=converged_count,
                        new_entries=new_ledger_entries,
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

    def _log_ledger_state(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        ledger_size: int,
        injected_count: int,
        exact_converged_count: int,
        new_entries: int,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "Coverage ledger updated after chunk",
            step="discover_requirements.ledger",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            ledger_size=ledger_size,
            injected_count=injected_count,
            exact_converged_count=exact_converged_count,
            new_entries=new_entries,
            status="completed",
        )


def _build_prompt(
    manifest: DocumentManifest,
    chunk: _NormalizedChunk,
    validation_error: str | None = None,
    ledger_section: str = "",
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
            f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
            "Repair the invalid output by using quotes copied exactly from the normalized "
            "chunk_text.\n"
            "If a quote includes a section heading, category label, row title, or source "
            "identifier before the actual requirement text, remove that leading label and keep "
            "only the smallest exact contiguous body span that exists in chunk_text.\n"
            "If a quote failed because it merged a heading with a bullet, table cell, "
            "or nearby line, do not synthesize a combined quote. Instead, choose the "
            "smallest exact contiguous source span that exists in chunk_text.\n"
            "If no exact quote can be copied for a fact, remove that fact from the output.\n"
            f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
        )

    return (
        f"{PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value}"
        f"{ledger_section}"
        f"{feedback}"
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

    facts = [
        _fact_to_candidate(context, fact, fact_index)
        for fact_index, fact in enumerate(output.facts, start=1)
    ]
    return RequirementDiscoveryOutput(
        chunks=[LLMChunkExtraction(chunk_id=context.chunk.chunk_id, facts=facts)]
    )


def _fact_to_candidate(
    context: _NormalizedChunk,
    fact: LLMDiscoveredFact,
    fact_index: int,
) -> LLMFactCandidate:
    # Temp ids are synthesized by ordinal (not returned by the model); they are used
    # only as human-readable labels in validation error messages. Permanent ids are
    # assigned later by requirement_builder.
    fact_label = f"F{fact_index}"
    trace = _source_trace_from_quote(fact, context, fact_label)
    requirements = [
        LLMRequirementCandidate(
            temp_id=f"R{requirement_index}",
            statement=requirement.req_text,
            requirement_type=requirement.requirement_type,
            priority=requirement.priority,
            requirement_key=requirement.requirement_key,
            source_req_id=requirement.source_req_id,
            confidence=requirement.confidence,
            source_trace=trace.model_copy(),
            actor=requirement.actor,
            modality=requirement.modality,
            action=requirement.action,
            object=requirement.object,
            condition=requirement.condition,
            polarity=requirement.polarity,
            requirement_family=requirement.requirement_family,
            entity_discriminators=list(requirement.entity_discriminators),
            mutable_parameters=list(requirement.mutable_parameters),
        )
        for requirement_index, requirement in enumerate(fact.requirements, start=1)
    ]
    return LLMFactCandidate(
        temp_id=fact_label,
        text=fact.fact_text,
        source_trace=trace,
        requirements=requirements,
    )


def _validate_candidate_semantics(output: RequirementDiscoveryChunkOutput) -> None:
    signatures_by_key: dict[str, tuple[str, ...]] = {}
    for fact in output.facts:
        for requirement in fact.requirements:
            if " and " in requirement.action.casefold() or " or " in requirement.action.casefold():
                raise TraceValidationError("one candidate contains multiple independent actions")
            if requirement.object.casefold().startswith(("and ", "or ")):
                raise TraceValidationError("one candidate contains multiple independent actions")
            if (
                " and " in requirement.actor.casefold()
                and len(requirement.entity_discriminators) > 1
            ):
                raise TraceValidationError("one candidate contains multiple independent actors")
            statement_tokens = _semantic_support_tokens(requirement.req_text)
            quote_tokens = _semantic_support_tokens(fact.quote)
            if statement_tokens and (
                len(statement_tokens & quote_tokens) / len(statement_tokens) < 0.5
            ):
                raise TraceValidationError(
                    "source trace does not support the extracted requirement statement"
                )
            signature = (
                requirement.requirement_family.casefold(),
                requirement.actor.casefold(),
                requirement.action.casefold(),
                requirement.object.casefold(),
                requirement.condition.casefold(),
                requirement.polarity,
                *sorted(value.casefold() for value in requirement.entity_discriminators),
            )
            hint = (requirement.requirement_key or "").strip().casefold()
            previous = signatures_by_key.get(hint) if hint else None
            if previous is not None and previous != signature:
                raise TraceValidationError(
                    "one requirement_key hint maps to incompatible atomic semantic signatures"
                )
            if hint:
                signatures_by_key[hint] = signature


def _semantic_support_tokens(value: str) -> set[str]:
    ignored = {
        "a",
        "an",
        "and",
        "be",
        "is",
        "of",
        "or",
        "shall",
        "should",
        "the",
        "to",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in ignored
    }


def _source_trace_from_quote(
    fact: LLMDiscoveredFact, context: _NormalizedChunk, fact_label: str
) -> SourceTrace:
    normalized_quote, _ = _normalize_for_prompt(fact.quote)
    if not normalized_quote:
        raise TraceValidationError(f"empty source quote for fact {fact_label}")

    normalized_start = -1
    matched_quote = normalized_quote
    for candidate in _quote_search_candidates(normalized_quote):
        normalized_start = context.text.find(candidate)
        if normalized_start >= 0:
            matched_quote = candidate
            break
    if normalized_start < 0:
        raise TraceValidationError(
            f"source quote for fact {fact_label} cannot be located in "
            f"{context.chunk.chunk_id} after normalization"
        )

    normalized_end = normalized_start + len(matched_quote)
    if normalized_start >= len(context.char_map) or normalized_end > len(context.char_map):
        raise TraceValidationError(
            f"source quote for fact {fact_label} cannot be mapped to raw text"
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


def _quote_search_candidates(normalized_quote: str) -> list[str]:
    candidates = [normalized_quote]
    stripped = _QUOTE_LEADING_LABELS.sub("", normalized_quote).strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    tokens = normalized_quote.split()
    for index in range(1, len(tokens)):
        candidate = " ".join(tokens[index:]).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


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
