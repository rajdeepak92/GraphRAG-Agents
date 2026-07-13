"""Per-chunk LLM extraction of source-knowledge entities and assertions.

Follows the same discipline as ``RequirementDiscoveryAgent``: one strict prompt
per chunk, structured output validated against Pydantic schemas, exact-quote
grounding back to the chunk text, and a single retry with the validation error
fed back before the raw response is persisted and the run fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Reuse the discovery agent's prompt-side whitespace normalization and
# response-context hooks so quote grounding behaves identically across stages.
from multi_agentic_graph_rag.agents.requirement_discovery_agent import (
    _clear_response_context,
    _last_response_path,
    _normalize_for_prompt,
    _persist_last_response,
    _set_response_context,
)
from multi_agentic_graph_rag.common_prompt_defs import (
    PromptKnowledgeExtraction,
    PromptSharedFragments,
)
from multi_agentic_graph_rag.domain.errors import ModelOutputError, TraceValidationError
from multi_agentic_graph_rag.domain.schemas import (
    AssertionCandidate,
    ChunkKnowledgeCandidates,
    DocumentChunk,
    EntityCandidate,
    KnowledgeExtractionChunkOutput,
    KnowledgeExtractionOutput,
    LLMExtractedAssertion,
    SourceTrace,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary
from multi_agentic_graph_rag.services.ontology import (
    normalize_entity_name,
    normalize_entity_type,
    normalize_predicate,
)


@dataclass(frozen=True)
class _NormalizedChunk:
    """Coordinate normalized chunk behavior within the agents boundary."""

    chunk: DocumentChunk
    text: str
    char_map: list[int]


class KnowledgeExtractionAgent:
    """Coordinate knowledge extraction agent behavior within the agents boundary."""

    def __init__(
        self,
        reasoning_model: ReasoningModel,
        *,
        logger: Any | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
        """
        self.reasoning_model = reasoning_model
        self.logger = logger

    def run(
        self,
        *,
        project: str,
        version: str,
        chunks: list[DocumentChunk],
    ) -> KnowledgeExtractionOutput:
        """Run run.

        Args:
            project (str): Project scope that isolates persistence and retrieval.
            version (str): Document version label within the project scope.
            chunks (list[DocumentChunk]): Ordered chunks whose identities remain unchanged.

        Returns:
            KnowledgeExtractionOutput: The typed result produced by the operation.
        """
        outputs: list[ChunkKnowledgeCandidates] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            outputs.append(
                self._run_chunk(
                    project=project,
                    version=version,
                    chunk=chunk,
                    chunk_index=chunk_index,
                )
            )
        return KnowledgeExtractionOutput(chunks=outputs)

    def _run_chunk(
        self,
        *,
        project: str,
        version: str,
        chunk: DocumentChunk,
        chunk_index: int,
    ) -> ChunkKnowledgeCandidates:
        """Run chunk.

        Args:
            project (str): Project scope that isolates persistence and retrieval.
            version (str): Document version label within the project scope.
            chunk (DocumentChunk): Chunk required by the operation's typed contract.
            chunk_index (int): Chunk index required by the operation's typed contract.

        Returns:
            ChunkKnowledgeCandidates: The typed result produced by the operation.

        Raises:
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.

        Side Effects:
            May invoke configured model or workflow providers.
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        normalized_text, char_map = _normalize_for_prompt(chunk.text)
        context = _NormalizedChunk(chunk=chunk, text=normalized_text, char_map=char_map)
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                if self.logger is not None:
                    self.logger.debug(
                        "retry.attempt_started",
                        step="extract_knowledge.chunk",
                        operation="knowledge_extraction.chunk",
                        chunk_id=chunk.chunk_id,
                        attempt=attempt,
                        max_attempts=2,
                        status="attempting",
                    )
                prompt = _build_knowledge_prompt(
                    project=project,
                    version=version,
                    context=context,
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
                    schema=KnowledgeExtractionChunkOutput,
                )
                try:
                    candidates = _validate_chunk_output(context, chunk_output)
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
                            f"Knowledge extraction chunk {chunk_index} failed validation "
                            f"after retry for {chunk.chunk_id}: "
                            f"{sanitized_exception_summary(error)}; "
                            f"raw_response_path={response_path}"
                        ) from error
                    validation_error = str(error)
                    continue

                self._log_chunk_completed(
                    chunk_index=chunk_index,
                    chunk_id=chunk.chunk_id,
                    entity_count=len(candidates.entities),
                    assertion_count=len(candidates.assertions),
                    retry_count=attempt - 1,
                    response_path=_last_response_path(self.reasoning_model),
                )
                return candidates
        finally:
            _clear_response_context(self.reasoning_model)

        raise ModelOutputError(f"Knowledge extraction chunk {chunk_index} did not produce a result")

    def _log_chunk_completed(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        entity_count: int,
        assertion_count: int,
        retry_count: int,
        response_path: str | None,
    ) -> None:
        """Execute the log chunk completed operation within its declared architectural boundary.

        Args:
            chunk_index (int): Chunk index required by the operation's typed contract.
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            entity_count (int): Bounded entity count used for deterministic processing.
            assertion_count (int): Bounded assertion count used for deterministic processing.
            retry_count (int): Bounded retry count used for deterministic processing.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        self.logger.info(
            "Knowledge extraction chunk completed",
            step="extract_knowledge.chunk",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            entity_count=entity_count,
            assertion_count=assertion_count,
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
        """Execute the log validation failure operation within its declared architectural boundary.

        Args:
            chunk_index (int): Chunk index required by the operation's typed contract.
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            attempt (int): Bounded attempt used for deterministic processing.
            error (TraceValidationError): Failure being classified or converted without exposing its
                                          message.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        self.logger.warning(
            "Knowledge extraction chunk failed validation",
            step="extract_knowledge.chunk",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            attempt=attempt,
            max_attempts=2,
            retry_delay_seconds=0.0,
            exception_type=error.__class__.__name__,
            error_summary=sanitized_exception_summary(error),
            raw_response_path=response_path,
            status="failed" if attempt == 2 else "retrying",
        )


def _build_knowledge_prompt(
    *,
    project: str,
    version: str,
    context: _NormalizedChunk,
    validation_error: str | None = None,
) -> str:
    """Build knowledge prompt.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        version (str): Document version label within the project scope.
        context (_NormalizedChunk): Context required by the operation's typed contract.
        validation_error (str | None): Validation error required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    chunk_json = json.dumps(
        {
            "chunk_id": context.chunk.chunk_id,
            "chunk_text": context.text,
        },
        ensure_ascii=False,
        indent=2,
    )

    feedback = ""
    if validation_error:
        feedback = (
            f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
            "Repair the invalid output. Every quote must be copied exactly from the "
            "normalized chunk_text; every subject and object_name must exactly match "
            "one entities[].name; every entity name must appear in chunk_text.\n"
            "If an entry cannot be repaired, remove it from the output.\n"
            f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
        )

    return (
        f"{PromptKnowledgeExtraction.SYS_PROMPT_KNOWLEDGE_EXTRACTION.value}"
        f"{feedback}"
        f"Project: {project}\n"
        f"Document version: {version}\n"
        f"Input chunk JSON:\n{chunk_json}"
    )


def _validate_chunk_output(
    context: _NormalizedChunk,
    output: KnowledgeExtractionChunkOutput,
) -> ChunkKnowledgeCandidates:
    """Validate chunk output against the enforced runtime contract.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        output (KnowledgeExtractionChunkOutput): Output required by the operation's typed contract.

    Returns:
        ChunkKnowledgeCandidates: The typed result produced by the operation.
    """
    entities = _validate_entities(context, output)
    names = {entity.normalized_name for entity in entities}
    assertions = [
        _validate_assertion(context, assertion, names, assertion_index)
        for assertion_index, assertion in enumerate(output.assertions, start=1)
    ]
    return ChunkKnowledgeCandidates(
        chunk_id=context.chunk.chunk_id,
        entities=entities,
        assertions=assertions,
    )


def _validate_entities(
    context: _NormalizedChunk,
    output: KnowledgeExtractionChunkOutput,
) -> list[EntityCandidate]:
    """Validate entities against the enforced runtime contract.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        output (KnowledgeExtractionChunkOutput): Output required by the operation's typed contract.

    Returns:
        list[EntityCandidate]: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    haystack = context.text.casefold()
    entities: list[EntityCandidate] = []
    seen: set[tuple[str, str]] = set()
    for entity in output.entities:
        surface = entity.name
        normalized_surface, _ = _normalize_for_prompt(surface)
        if not normalized_surface or normalized_surface.casefold() not in haystack:
            raise TraceValidationError(
                f"entity {surface!r} does not appear in chunk_text of {context.chunk.chunk_id}"
            )
        normalized_name = normalize_entity_name(surface)
        entity_type = normalize_entity_type(entity.entity_type)
        key = (normalized_name, entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            EntityCandidate(
                chunk_id=context.chunk.chunk_id,
                surface_text=surface,
                normalized_name=normalized_name,
                entity_type=entity_type,
            )
        )
    return entities


def _validate_assertion(
    context: _NormalizedChunk,
    assertion: LLMExtractedAssertion,
    entity_names: set[str],
    assertion_index: int,
) -> AssertionCandidate:
    """Validate assertion against the enforced runtime contract.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        assertion (LLMExtractedAssertion): Assertion required by the operation's typed contract.
        entity_names (set[str]): Entity names required by the operation's typed contract.
        assertion_index (int): Assertion index required by the operation's typed contract.

    Returns:
        AssertionCandidate: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    label = f"assertion {assertion_index}"
    subject_name = normalize_entity_name(assertion.subject)
    if subject_name not in entity_names:
        raise TraceValidationError(
            f"{label} subject {assertion.subject!r} does not match any entities[].name"
        )
    # Object resolution is lenient: an object counts as an ENTITY only when it was
    # declared. An *undeclared* object is demoted to a literal value instead of
    # failing the whole chunk — models frequently reference an object concept they
    # forgot to add to entities[] (e.g. "operating data"). The subject stays strict
    # (it is the assertion's anchor), and a declared self-loop is still rejected.
    object_name: str | None = None
    demoted_object_literal: str | None = None
    raw_object = assertion.object_name.strip()
    if raw_object:
        normalized_object = normalize_entity_name(raw_object)
        if normalized_object not in entity_names:
            demoted_object_literal = raw_object
        elif normalized_object == subject_name:
            raise TraceValidationError(
                f"{label} subject and object_name must reference different entities"
            )
        else:
            object_name = normalized_object

    predicate = normalize_predicate(assertion.predicate)
    if not predicate:
        raise TraceValidationError(f"{label} predicate {assertion.predicate!r} is not usable")

    trace = _source_trace_from_quote(context, assertion.quote, label)
    literal = assertion.object_literal.strip() or (demoted_object_literal or "")
    condition = assertion.condition.strip()
    return AssertionCandidate(
        chunk_id=context.chunk.chunk_id,
        subject_name=subject_name,
        predicate=predicate,
        object_name=object_name,
        object_literal=literal or None,
        modality=assertion.modality,
        polarity=assertion.polarity,
        explicitness=assertion.explicitness,
        condition=condition or None,
        confidence=assertion.confidence,
        source_trace=trace,
    )


def _source_trace_from_quote(
    context: _NormalizedChunk,
    quote: str,
    label: str,
) -> SourceTrace:
    """Execute the source trace from quote operation within its declared architectural boundary.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        quote (str): Quote required by the operation's typed contract.
        label (str): Label required by the operation's typed contract.

    Returns:
        SourceTrace: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    normalized_quote, _ = _normalize_for_prompt(quote)
    if not normalized_quote:
        raise TraceValidationError(f"empty source quote for {label}")

    normalized_start = -1
    matched_quote = normalized_quote
    for candidate in _quote_candidates(normalized_quote):
        normalized_start = context.text.find(candidate)
        if normalized_start >= 0:
            matched_quote = candidate
            break
    if normalized_start < 0:
        raise TraceValidationError(
            f"source quote for {label} cannot be located in "
            f"{context.chunk.chunk_id} after normalization"
        )

    normalized_end = normalized_start + len(matched_quote)
    if normalized_start >= len(context.char_map) or normalized_end > len(context.char_map):
        raise TraceValidationError(f"source quote for {label} cannot be mapped to raw text")

    raw_start = context.char_map[normalized_start]
    raw_end = context.char_map[normalized_end - 1] + 1
    return SourceTrace(
        chunk_id=context.chunk.chunk_id,
        quote=context.chunk.text[raw_start:raw_end],
        start_char=raw_start,
        end_char=raw_end,
        page=context.chunk.page,
        section=context.chunk.section,
    )


def _quote_candidates(normalized_quote: str) -> list[str]:
    """The exact quote first, then progressively label-stripped suffixes."""
    candidates = [normalized_quote]
    tokens = normalized_quote.split()
    for index in range(1, len(tokens)):
        candidate = " ".join(tokens[index:]).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates
