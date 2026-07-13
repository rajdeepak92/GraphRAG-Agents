"""Bounded requirement discovery component."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from multi_agentic_graph_rag.common_prompt_defs import (
    PromptRequirementDiscovery,
    PromptSharedFragments,
)
from multi_agentic_graph_rag.domain.errors import ModelOutputError, TraceValidationError
from multi_agentic_graph_rag.domain.identifiers import stable_token
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    LLMChunkExtraction,
    LLMDiscoveredFact,
    LLMDiscoveredRequirement,
    LLMFactCandidate,
    LLMRequirementCandidate,
    RequirementDiscoveryChunkOutput,
    RequirementDiscoveryOutput,
    SourceTrace,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary
from multi_agentic_graph_rag.services.coverage_ledger import CoverageLedger, LedgerEntry
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    structured_requirement_signature,
)

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
    """Coordinate normalized chunk behavior within the agents boundary."""

    chunk: DocumentChunk
    text: str
    char_map: list[int]


# Semantic-validation category slugs. ``key_collision`` is the only non-terminal
# category: because Python (never the LLM key) owns permanent identity, a surviving
# key collision is repaired deterministically instead of failing the chunk.
_CATEGORY_KEY_COLLISION = "key_collision"
_CATEGORY_SOURCE_SUPPORT = "source_support"
_CATEGORY_MULTIPLE_ACTIONS = "multiple_actions"
_CATEGORY_MULTIPLE_ACTORS = "multiple_actors"

_SUPPORT_RATIO_FLOOR = 0.5
_DIAGNOSTIC_TEXT_LIMIT = 400
_DIAGNOSTICS_JSON_LIMIT = 6000
_PREVIOUS_OUTPUT_JSON_LIMIT = 6000
# A numbered instance discriminator (Sensor-1, reg_3, api-2) names one concrete
# entity; two such distinct instances joined by a conjunction are genuinely
# independent actors. Reuses the first branch of the schema discriminator regex.
_NUMBERED_INSTANCE_RE = re.compile(r"[A-Za-z]+[-_]\d+[A-Za-z0-9_-]*")
_ACTOR_CONJUNCTION_RE = re.compile(r"\band\b|\bor\b|&|,|;", re.I)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
# Requirement-template scaffolding used only by the source-support check: the
# default actor injected for a bare noun-phrase source, plus empty obligation verbs
# that turn a copied noun phrase into an obligation. These carry no domain meaning,
# so they are not required to appear in the exact quote. Meaning-bearing verbs
# (store, log, display, generate, send, delete, encrypt, synchronize, collect, …)
# are intentionally excluded so genuine over-reach and inversions still fail.
_REQUIREMENT_SCAFFOLDING = frozenset(
    {
        "system",
        "provide",
        "provides",
        "support",
        "supports",
        "include",
        "includes",
        "allow",
        "allows",
        "enable",
        "enables",
        "offer",
        "offers",
        "have",
        "has",
    }
)


@dataclass
class _CandidateDiagnostic:
    """One actionable semantic-validation finding for a single requirement candidate."""

    category: str
    chunk_id: str
    fact_index: int
    requirement_index: int
    requirement_key: str
    source_req_id: str | None
    req_text: str
    source_quote: str
    normalized_atomic_signature: str
    semantic_field: str = ""
    support_ratio: float | None = None
    missing_tokens: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        """Convert the value to payload without mutating its source.

        Returns:
            dict[str, object]: The typed result produced by the operation.
        """
        payload: dict[str, object] = {
            "category": self.category,
            "chunk_id": self.chunk_id,
            "fact_index": self.fact_index,
            "requirement_index": self.requirement_index,
            "requirement_key": self.requirement_key,
            "source_req_id": self.source_req_id,
            "req_text": _sanitize_diagnostic_text(self.req_text),
            "source_quote": _sanitize_diagnostic_text(self.source_quote),
            "normalized_atomic_signature": _sanitize_diagnostic_text(
                self.normalized_atomic_signature
            ),
        }
        if self.semantic_field:
            payload["semantic_field"] = self.semantic_field
        if self.support_ratio is not None:
            payload["support_ratio"] = self.support_ratio
        if self.missing_tokens:
            payload["missing_tokens"] = self.missing_tokens[:20]
        return payload


class _SemanticValidationError(TraceValidationError):
    """Aggregated, actionable semantic-validation failures for one chunk.

    Carries every independent candidate error found in a single pass so one
    corrective retry can address all of them. ``requirement_key`` is a
    non-authoritative hint, so a pure key collision is *not* terminal.
    """

    def __init__(self, diagnostics: list[_CandidateDiagnostic]) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            diagnostics (list[_CandidateDiagnostic]): Diagnostics required by the operation's typed
                                                      contract.
        """
        self.diagnostics = diagnostics
        self.categories = sorted({diagnostic.category for diagnostic in diagnostics})
        self.diagnostics_json = _bounded_json([d.to_payload() for d in diagnostics])
        chunk_ids = sorted({diagnostic.chunk_id for diagnostic in diagnostics})
        super().__init__(
            "requirement discovery semantic validation failed for "
            f"{','.join(chunk_ids)}: categories={','.join(self.categories)}; "
            f"diagnostics={self.diagnostics_json}"
        )

    @property
    def terminal_diagnostics(self) -> list[_CandidateDiagnostic]:
        """Execute the terminal diagnostics operation within its declared architectural boundary.

        Returns:
            list[_CandidateDiagnostic]: The typed result produced by the operation.
        """
        return [d for d in self.diagnostics if d.category != _CATEGORY_KEY_COLLISION]

    @property
    def has_terminal(self) -> bool:
        """Return whether terminal.

        Returns:
            bool: The typed result produced by the operation.
        """
        return bool(self.terminal_diagnostics)


class RequirementDiscoveryAgent:
    """Coordinate requirement discovery agent behavior within the agents boundary."""

    def __init__(
        self,
        reasoning_model: ReasoningModel,
        *,
        logger: Any | None = None,
        coverage_ledger: CoverageLedger | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
            coverage_ledger (CoverageLedger | None): Optional cross-chunk coverage memory.
        """
        self.reasoning_model = reasoning_model
        self.logger = logger
        self.coverage_ledger = coverage_ledger

    def run(self, manifest: DocumentManifest) -> RequirementDiscoveryOutput:
        """Run run.

        Args:
            manifest (DocumentManifest): Manifest required by the operation's typed contract.

        Returns:
            RequirementDiscoveryOutput: The typed result produced by the operation.
        """
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
        """Run chunk.

        Args:
            manifest (DocumentManifest): Manifest required by the operation's typed contract.
            chunk (DocumentChunk): Chunk required by the operation's typed contract.
            chunk_index (int): Chunk index required by the operation's typed contract.

        Returns:
            RequirementDiscoveryOutput: The typed result produced by the operation.

        Raises:
            semantic_error: If validated inputs or required dependencies cannot satisfy the
            contract.
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.

        Side Effects:
            May invoke configured model or workflow providers.
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        context = _normalized_context(chunk)
        ledger_entries: list[LedgerEntry] = []
        ledger_section = ""
        if self.coverage_ledger is not None:
            ledger_entries = self.coverage_ledger.select_for_chunk(context.text)
            ledger_section = self.coverage_ledger.render_prompt_section(ledger_entries)
        validation_error: TraceValidationError | None = None
        previous_output: RequirementDiscoveryChunkOutput | None = None
        try:
            for attempt in (1, 2):
                if self.logger is not None:
                    self.logger.debug(
                        "retry.attempt_started",
                        step="discover_requirements.chunk",
                        operation="requirement_discovery.chunk",
                        chunk_id=chunk.chunk_id,
                        attempt=attempt,
                        max_attempts=2,
                        status="attempting",
                    )
                prompt = _build_prompt(
                    manifest,
                    context,
                    validation_error=validation_error,
                    previous_output=previous_output,
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
                    system_message=(
                        PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value
                    ),
                    operation="requirement_discovery.chunk",
                    request_id=chunk.chunk_id,
                )
                try:
                    diagnostics = _collect_semantic_diagnostics(
                        chunk_output, chunk_id=chunk.chunk_id
                    )
                    if diagnostics:
                        semantic_error = _SemanticValidationError(diagnostics)
                        if attempt < 2 or semantic_error.has_terminal:
                            raise semantic_error
                        # Final attempt, only key collisions remain: the LLM key is a
                        # non-authoritative hint, so repair it deterministically from
                        # validated semantic content rather than failing the chunk.
                        _canonicalize_conflicting_keys(chunk_output, diagnostics)
                        self._log_key_canonicalization(
                            chunk_index=chunk_index,
                            chunk_id=chunk.chunk_id,
                            diagnostics=diagnostics,
                        )
                    output = _chunk_to_nested(context, chunk_output)
                    _verify_traces(manifest, output)
                except TraceValidationError as error:
                    response_path = _persist_last_response(self.reasoning_model)
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
                            f"for {chunk.chunk_id}: attempt={attempt}; "
                            f"{sanitized_exception_summary(error)}"
                            f"; raw_response_path={response_path}"
                        ) from error
                    validation_error = error
                    previous_output = chunk_output
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

    def _log_key_canonicalization(
        self,
        *,
        chunk_index: int,
        chunk_id: str,
        diagnostics: list[_CandidateDiagnostic],
    ) -> None:
        """Log deterministic repair of a non-authoritative key collision.

        Args:
            chunk_index (int): Chunk index required by the operation's typed contract.
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            diagnostics (list[_CandidateDiagnostic]): Diagnostics required by the operation's typed
                                                      contract.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        conflicting_keys = sorted(
            {
                diagnostic.requirement_key
                for diagnostic in diagnostics
                if diagnostic.category == _CATEGORY_KEY_COLLISION
            }
        )
        self.logger.info(
            "Repaired non-authoritative requirement_key collision deterministically",
            step="discover_requirements.chunk",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            conflicting_keys=conflicting_keys,
            status="repaired",
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
        """Execute the log chunk completed operation within its declared architectural boundary.

        Args:
            chunk_index (int): Chunk index required by the operation's typed contract.
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            fact_count (int): Bounded fact count used for deterministic processing.
            requirement_count (int): Bounded requirement count used for deterministic processing.
            retry_count (int): Bounded retry count used for deterministic processing.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
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
            "Requirement discovery chunk failed validation",
            step="discover_requirements.chunk",
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
        """Execute the log ledger state operation within its declared architectural boundary.

        Args:
            chunk_index (int): Chunk index required by the operation's typed contract.
            chunk_id (str): Canonical chunk id used as a safe operational anchor.
            ledger_size (int): Ledger size required by the operation's typed contract.
            injected_count (int): Bounded injected count used for deterministic processing.
            exact_converged_count (int): Bounded exact converged count used for deterministic
                                         processing.
            new_entries (int): New entries required by the operation's typed contract.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
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
    validation_error: TraceValidationError | None = None,
    previous_output: RequirementDiscoveryChunkOutput | None = None,
    ledger_section: str = "",
) -> str:
    """Build prompt.

    Args:
        manifest (DocumentManifest): Manifest required by the operation's typed contract.
        chunk (_NormalizedChunk): Chunk required by the operation's typed contract.
        validation_error (TraceValidationError | None): Validation error required by the operation's
                                                        typed contract.
        previous_output (RequirementDiscoveryChunkOutput | None): Previous output required by the
                                                                  operation's typed contract.
        ledger_section (str): Ledger section required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
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
        if isinstance(validation_error, _SemanticValidationError):
            feedback = _semantic_retry_feedback(validation_error, previous_output)
        else:
            feedback = _trace_retry_feedback(validation_error, previous_output)

    return (
        f"{PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value}"
        f"{ledger_section}"
        f"{feedback}"
        f"Project: {manifest.project}\n"
        f"Document version: {manifest.version}\n"
        f"Input chunk JSON:\n{chunk_json}"
    )


def _previous_output_section(
    previous_output: RequirementDiscoveryChunkOutput | None,
) -> str:
    """Execute the previous output section operation within its declared architectural boundary.

    Args:
        previous_output (RequirementDiscoveryChunkOutput | None): Previous output required by the
                                                                  operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    if previous_output is None:
        return ""
    serialized = _bounded_json(
        previous_output.model_dump(mode="json"),
        limit=_PREVIOUS_OUTPUT_JSON_LIMIT,
    )
    return (
        "Your previous structured output that failed validation (JSON), for reference "
        "while you correct it — do not repeat its mistakes:\n"
        f"{serialized}\n\n"
    )


def _semantic_retry_feedback(
    error: _SemanticValidationError,
    previous_output: RequirementDiscoveryChunkOutput | None,
) -> str:
    """Execute the semantic retry feedback operation within its declared architectural boundary.

    Args:
        error (_SemanticValidationError): Failure being classified or converted without exposing its
                                          message.
        previous_output (RequirementDiscoveryChunkOutput | None): Previous output required by the
                                                                  operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    categories = set(error.categories)
    instructions: list[str] = []
    if _CATEGORY_SOURCE_SUPPORT in categories:
        instructions.append(
            "- source_support: rewrite req_text so it expresses ONLY meaning that is present "
            "in its own exact quote (source_quote). Do not borrow details from adjacent facts, "
            "other chunks, section headings, or ledger entries. Only minimal grammar completion "
            "is allowed and it must add no new meaning. The listed missing_tokens do not appear "
            "in the quote and must not appear in req_text unless the quote supports them. If a "
            "supported atomic requirement cannot be derived from the quote, drop only that "
            "requirement and keep every other valid requirement."
        )
    if _CATEGORY_MULTIPLE_ACTIONS in categories:
        instructions.append(
            "- multiple_actions: a single requirement must describe one action. Split a "
            "requirement whose action joins independent actions into separate atomic "
            "requirements, each with its own exact quote."
        )
    if _CATEGORY_MULTIPLE_ACTORS in categories:
        instructions.append(
            "- multiple_actors: split only genuinely independent numbered actors (for example "
            "'Sensor-1 and Sensor-2') into separate requirements. Do NOT split an official "
            "stakeholder, organization, department, team, or role name that merely contains a "
            "conjunction — 'QA and Validation Teams' is ONE actor and must stay one requirement."
        )
    if _CATEGORY_KEY_COLLISION in categories:
        instructions.append(
            "- key_collision: you reused one requirement_key for incompatible atomic "
            "requirements. Give incompatible requirements distinct stable keys derived from "
            "semantic content and source provenance, not list position. Reuse a key only for an "
            "equivalent atomic signature. Do not merge or delete a supported requirement."
        )
    return (
        f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
        "The previous response failed requirement discovery semantic validation. Return the "
        "COMPLETE corrected chunk output (every fact and requirement), not a patch. Preserve "
        "every source-supported atomic requirement and its exact contiguous quote. Keep a "
        "document identifier in source_req_id only when it occurs in that same exact quote; "
        "otherwise use an empty string.\n"
        "Full per-candidate diagnostics (category, fact_index, requirement_index, "
        "requirement_key, source_req_id, req_text, source_quote, semantic_field, support_ratio, "
        "missing_tokens, normalized_atomic_signature):\n"
        f"{error.diagnostics_json}\n"
        "Apply these corrections:\n" + "\n".join(instructions) + "\n"
        f"{_previous_output_section(previous_output)}"
    )


def _trace_retry_feedback(
    validation_error: TraceValidationError,
    previous_output: RequirementDiscoveryChunkOutput | None,
) -> str:
    """Execute the trace retry feedback operation within its declared architectural boundary.

    Args:
        validation_error (TraceValidationError): Validation error required by the operation's typed
                                                 contract.
        previous_output (RequirementDiscoveryChunkOutput | None): Previous output required by the
                                                                  operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
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
        f"{_previous_output_section(previous_output)}"
        f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
    )


def _sanitize_diagnostic_text(value: str) -> str:
    """Sanitize diagnostic text.

    Args:
        value (str): Value required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    cleaned = _CONTROL_CHARS_RE.sub(" ", value).strip()
    if len(cleaned) > _DIAGNOSTIC_TEXT_LIMIT:
        return cleaned[:_DIAGNOSTIC_TEXT_LIMIT] + "…"
    return cleaned


def _bounded_json(payload: object, *, limit: int = _DIAGNOSTICS_JSON_LIMIT) -> str:
    """Execute the bounded json operation within its declared architectural boundary.

    Args:
        payload (object): Validated structured data for the operation.
        limit (int): Bounded limit used for deterministic processing.

    Returns:
        str: The typed result produced by the operation.
    """
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(serialized) > limit:
        return serialized[:limit] + "…(truncated)"
    return serialized


def _normalized_context(chunk: DocumentChunk) -> _NormalizedChunk:
    """Execute the normalized context operation within its declared architectural boundary.

    Args:
        chunk (DocumentChunk): Chunk required by the operation's typed contract.

    Returns:
        _NormalizedChunk: The typed result produced by the operation.
    """
    normalized_text, char_map = _normalize_for_prompt(chunk.text)
    return _NormalizedChunk(chunk=chunk, text=normalized_text, char_map=char_map)


def _normalize_for_prompt(text: str) -> tuple[str, list[int]]:
    """Normalize for prompt deterministically within the active scope.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        tuple[str, list[int]]: The typed result produced by the operation.
    """
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
    """Return whether split identifier boundary.

    Args:
        chars (list[str]): Chars required by the operation's typed contract.
        text (str): Input text processed in memory and excluded from diagnostic logs.
        whitespace_index (int): Whitespace index required by the operation's typed contract.

    Returns:
        bool: The typed result produced by the operation.
    """
    next_char = _next_non_whitespace(text, whitespace_index)
    if next_char is None or not next_char.isalnum():
        return False
    prefix = "".join(chars[-24:])
    return bool(re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*-$", prefix, re.I))


def _next_non_whitespace(text: str, start_index: int) -> str | None:
    """Execute the next non whitespace operation within its declared architectural boundary.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.
        start_index (int): Start index required by the operation's typed contract.

    Returns:
        str | None: The typed result produced by the operation.
    """
    for char in text[start_index + 1 :]:
        if not char.isspace():
            return char
    return None


def _chunk_to_nested(
    context: _NormalizedChunk,
    output: RequirementDiscoveryChunkOutput,
) -> RequirementDiscoveryOutput:
    """Execute the chunk to nested operation within its declared architectural boundary.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        output (RequirementDiscoveryChunkOutput): Output required by the operation's typed contract.

    Returns:
        RequirementDiscoveryOutput: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
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
    """Execute the fact to candidate operation within its declared architectural boundary.

    Args:
        context (_NormalizedChunk): Context required by the operation's typed contract.
        fact (LLMDiscoveredFact): Fact required by the operation's typed contract.
        fact_index (int): Fact index required by the operation's typed contract.

    Returns:
        LLMFactCandidate: The typed result produced by the operation.
    """
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


def _candidate_signature(requirement: LLMDiscoveredRequirement) -> str:
    """Execute the candidate signature operation within its declared architectural boundary.

    Args:
        requirement (LLMDiscoveredRequirement): Requirement required by the operation's typed
                                                contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return structured_requirement_signature(
        statement=requirement.req_text,
        requirement_type=requirement.requirement_type,
        actor=requirement.actor,
        modality=requirement.modality,
        action=requirement.action,
        object_text=requirement.object,
        condition=requirement.condition,
        polarity=requirement.polarity,
        requirement_family=requirement.requirement_family,
        entity_discriminators=requirement.entity_discriminators,
        # A model-provided mutable list must not be able to erase a real
        # actor/action/object/condition difference from collision validation.
        mutable_parameters=[],
    )


def _has_multiple_actors(actor: str) -> bool:
    """Return True only for genuinely independent numbered actors.

    Splits the actor string on conjunctions and flags multiple actors only when at
    least two conjuncts each name a *distinct numbered instance* discriminator
    (``Sensor-1``/``Sensor-2``). Official stakeholder, organization, department,
    team, or role names that merely contain a conjunction (``QA and Validation
    Teams``) contain no numbered instances and stay a single actor. This never keys
    off the enriched ``entity_discriminators`` list, which may include acronym parts
    of a single proper name.
    """
    conjuncts = [part.strip() for part in _ACTOR_CONJUNCTION_RE.split(actor) if part.strip()]
    if len(conjuncts) < 2:
        return False
    instances = {
        match.group(0).casefold()
        for part in conjuncts
        if (match := _NUMBERED_INSTANCE_RE.search(part)) is not None
    }
    numbered_conjuncts = sum(
        1 for part in conjuncts if _NUMBERED_INSTANCE_RE.search(part) is not None
    )
    return numbered_conjuncts >= 2 and len(instances) >= 2


def _source_support(req_text: str, quote: str) -> tuple[float | None, list[str]]:
    # A source bullet is often a bare noun phrase ("Cloud-based storage and
    # monitoring."). The atomic requirement wraps its copied domain content in a
    # fixed template — the injected default actor plus an empty obligation verb —
    # that carries no meaning of its own and need not appear in the quote. Only the
    # DOMAIN CONTENT (nouns, entities, meaning-bearing verbs) must be supported, so
    # requiring the scaffolding to be quoted would falsely reject a valid, grounded
    # requirement. Meaning-bearing verbs are deliberately excluded from this set so a
    # real inversion ("shall delete data" vs a quote that says "store data") is still
    # caught.
    """Execute the source support operation within its declared architectural boundary.

    Args:
        req_text (str): Input text processed in memory and excluded from diagnostic logs.
        quote (str): Quote required by the operation's typed contract.

    Returns:
        tuple[float | None, list[str]]: The typed result produced by the operation.
    """
    statement_tokens = _semantic_support_tokens(req_text) - _REQUIREMENT_SCAFFOLDING
    if not statement_tokens:
        return None, []
    quote_tokens = _semantic_support_tokens(quote)
    ratio = len(statement_tokens & quote_tokens) / len(statement_tokens)
    missing = sorted(statement_tokens - quote_tokens)
    return ratio, missing


def _collect_semantic_diagnostics(
    output: RequirementDiscoveryChunkOutput, *, chunk_id: str
) -> list[_CandidateDiagnostic]:
    """Aggregate every independent semantic error in one chunk (never fail-fast).

    Returns actionable per-candidate diagnostics so a single corrective retry can
    address all findings. Key collisions are reported per member of the conflicting
    group. Diagnostics are returned in a deterministic order.
    """
    diagnostics: list[_CandidateDiagnostic] = []
    observations_by_key: dict[str, list[_CandidateDiagnostic]] = {}
    for fact_index, fact in enumerate(output.facts, start=1):
        for requirement_index, requirement in enumerate(fact.requirements, start=1):
            signature = _candidate_signature(requirement)
            base: dict[str, Any] = {
                "chunk_id": chunk_id,
                "fact_index": fact_index,
                "requirement_index": requirement_index,
                "requirement_key": requirement.requirement_key or "",
                "source_req_id": requirement.source_req_id,
                "req_text": requirement.req_text,
                "source_quote": fact.quote,
                "normalized_atomic_signature": signature,
            }

            action_cf = requirement.action.casefold()
            if (
                " and " in action_cf
                or " or " in action_cf
                or requirement.object.casefold().startswith(("and ", "or "))
            ):
                diagnostics.append(
                    _CandidateDiagnostic(
                        category=_CATEGORY_MULTIPLE_ACTIONS, semantic_field="action", **base
                    )
                )
            if _has_multiple_actors(requirement.actor):
                diagnostics.append(
                    _CandidateDiagnostic(
                        category=_CATEGORY_MULTIPLE_ACTORS, semantic_field="actor", **base
                    )
                )
            ratio, missing = _source_support(requirement.req_text, fact.quote)
            if ratio is not None and ratio < _SUPPORT_RATIO_FLOOR:
                diagnostics.append(
                    _CandidateDiagnostic(
                        category=_CATEGORY_SOURCE_SUPPORT,
                        semantic_field="req_text",
                        support_ratio=round(ratio, 3),
                        missing_tokens=missing,
                        **base,
                    )
                )
            hint = (requirement.requirement_key or "").strip().casefold()
            if hint:
                observations_by_key.setdefault(hint, []).append(
                    _CandidateDiagnostic(
                        category=_CATEGORY_KEY_COLLISION,
                        semantic_field="requirement_key",
                        **base,
                    )
                )

    for observations in observations_by_key.values():
        signatures = {observation.normalized_atomic_signature for observation in observations}
        if len(signatures) > 1:
            diagnostics.extend(observations)

    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic.fact_index,
            diagnostic.requirement_index,
            diagnostic.category,
        )
    )
    return diagnostics


def _canonicalize_conflicting_keys(
    output: RequirementDiscoveryChunkOutput,
    diagnostics: list[_CandidateDiagnostic],
) -> None:
    """Deterministically repair a surviving non-authoritative key collision.

    Every candidate whose ``requirement_key`` participated in a collision is
    rewritten to ``<key>:<8-hex signature token>``. The suffix derives from the
    validated semantic signature, so it is stable across identical reruns and
    independent of list position: equivalent atomic requirements keep the same
    repaired key, incompatible ones get distinct keys. Permanent REQ/REQREV/REQEVID
    IDs stay Python-owned and are unaffected.
    """
    conflict_keys = {
        diagnostic.requirement_key.strip().casefold()
        for diagnostic in diagnostics
        if diagnostic.category == _CATEGORY_KEY_COLLISION
    }
    if not conflict_keys:
        return
    for fact in output.facts:
        for requirement in fact.requirements:
            hint = (requirement.requirement_key or "").strip()
            if not hint or hint.casefold() not in conflict_keys:
                continue
            suffix = stable_token(_candidate_signature(requirement), length=8)
            object.__setattr__(requirement, "requirement_key", f"{hint}:{suffix}")


def _semantic_support_tokens(value: str) -> set[str]:
    """Execute the semantic support tokens operation within its declared architectural boundary.

    Args:
        value (str): Value required by the operation's typed contract.

    Returns:
        set[str]: The typed result produced by the operation.
    """
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
    """Execute the source trace from quote operation within its declared architectural boundary.

    Args:
        fact (LLMDiscoveredFact): Fact required by the operation's typed contract.
        context (_NormalizedChunk): Context required by the operation's typed contract.
        fact_label (str): Fact label required by the operation's typed contract.

    Returns:
        SourceTrace: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
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
    """Execute the quote search candidates operation within its declared architectural boundary.

    Args:
        normalized_quote (str): Normalized quote required by the operation's typed contract.

    Returns:
        list[str]: The typed result produced by the operation.
    """
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
    """Return whether a chunk appears to contain normative requirement language.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        bool: The typed result produced by the operation.
    """
    return bool(_REQUIREMENT_HINT.search(text))


def _set_response_context(
    reasoning_model: ReasoningModel,
    *,
    batch_index: int,
    attempt: int,
    chunk_ids: list[str],
) -> None:
    """Execute the set response context operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
        batch_index (int): Batch index required by the operation's typed contract.
        attempt (int): Bounded attempt used for deterministic processing.
        chunk_ids (list[str]): Chunk ids required by the operation's typed contract.
    """
    setter = getattr(reasoning_model, "set_response_context", None)
    if callable(setter):
        setter(batch_index=batch_index, attempt=attempt, chunk_ids=chunk_ids)


def _clear_response_context(reasoning_model: ReasoningModel) -> None:
    """Execute the clear response context operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
    """
    clearer = getattr(reasoning_model, "clear_response_context", None)
    if callable(clearer):
        clearer()


def _persist_last_response(reasoning_model: ReasoningModel) -> str | None:
    """Persist the last raw response under the model's canonical diagnostic filename.

    The adapter derives the name from the call's operation, request id, monotonic call index,
    and attempt (``llm_response_<operation>_<request_id>_call-<n>_attempt-<n>.txt``), so every
    capture -- success (LOG_LLM_RESPONSES) or validation failure -- shares one scheme and one
    file per physical call. Passing a hand-rolled name here previously produced a second,
    differently-named file on failures only (e.g. ``llm_response_3_1.txt``).

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.

    Returns:
        str | None: The typed result produced by the operation.
    """
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        return _last_response_path(reasoning_model)
    path = persister()
    return str(path) if path is not None else None


def _last_response_path(reasoning_model: ReasoningModel) -> str | None:
    """Execute the last response path operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.

    Returns:
        str | None: The typed result produced by the operation.
    """
    response_path = getattr(reasoning_model, "last_response_path", None)
    return str(response_path) if response_path else None


def _merge_outputs(outputs: Iterable[RequirementDiscoveryOutput]) -> RequirementDiscoveryOutput:
    """Merge outputs.

    Args:
        outputs (Iterable[RequirementDiscoveryOutput]): Outputs required by the operation's typed
                                                        contract.

    Returns:
        RequirementDiscoveryOutput: The typed result produced by the operation.
    """
    chunk_outputs: list[LLMChunkExtraction] = []
    for output in outputs:
        chunk_outputs.extend(output.chunks)
    return RequirementDiscoveryOutput(chunks=chunk_outputs)


def _output_requirement_count(output: RequirementDiscoveryOutput) -> int:
    """Execute the output requirement count operation within its declared architectural boundary.

    Args:
        output (RequirementDiscoveryOutput): Output required by the operation's typed contract.

    Returns:
        int: The typed result produced by the operation.
    """
    return sum(
        len(fact.requirements) for chunk_output in output.chunks for fact in chunk_output.facts
    )


def _verify_traces(
    manifest: DocumentManifest,
    output: RequirementDiscoveryOutput,
) -> None:
    """Verify traces against the enforced runtime contract.

    Args:
        manifest (DocumentManifest): Manifest required by the operation's typed contract.
        output (RequirementDiscoveryOutput): Output required by the operation's typed contract.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
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
    """Verify trace against the enforced runtime contract.

    Args:
        chunk (DocumentChunk): Chunk required by the operation's typed contract.
        trace (SourceTrace): Trace required by the operation's typed contract.
        label (str): Label required by the operation's typed contract.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
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
