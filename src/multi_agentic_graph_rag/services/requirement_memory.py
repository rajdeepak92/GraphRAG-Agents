"""Durable cross-version requirement memory + fail-closed semantic reconciliation.

Seeded before discovery with the prior document version's *active* requirements
(and grown with the current run's discoveries), this is the recall+judge layer
that lets a paraphrased obligation reuse an existing canonical lineage instead of
spawning a duplicate — while never merging on similarity alone.

Decision pipeline (deterministic first, semantic last, fail closed):

1. exact normalized-statement hash            -> ``EXACT`` (reuse id + revision)
2. exact structured signature (diff wording)  -> ``SAME_LINEAGE_REVISION``
3. bounded candidate recall (signature/source/token/embedding) + reranking, then
   a strict **bidirectional** entailment judge and compatible family/signature
   -> ``EXACT``; anything ambiguous fails closed to ``DISTINCT`` with an audit
   reason. Candidates are verified against the canonical representative only, so
   A≈B and B≈C never transitively merge A and C.

Embedding cosine and reranker scores gate *recall*, never the merge decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementIdentity
from multi_agentic_graph_rag.config.settings import RequirementIdentitySettings
from multi_agentic_graph_rag.domain.schemas import StrictModel
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    normalize_requirement_family,
    requirement_lineage_signature,
)

_TOKEN = re.compile(r"[a-z0-9]+")

Decision = Literal["EXACT", "SAME_LINEAGE_REVISION", "DISTINCT", "AMBIGUOUS"]


class Embedder(Protocol):
    """Specify the provider-neutral embedder interface required by this boundary."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    """Specify the provider-neutral reranker interface required by this boundary."""

    def rerank(self, query: str, documents: list[str]) -> list[int]: ...


class EntailmentJudge(Protocol):
    """Strict bidirectional equivalence judge."""

    def equivalent(self, premise: str, hypothesis: str) -> bool: ...


class _BidirectionalEntailmentOutput(StrictModel):
    """Define the validated bidirectional entailment output data contract."""

    premise_entails_hypothesis: bool
    hypothesis_entails_premise: bool


class ModelEntailmentJudge:
    """Provider-neutral strict entailment judge backed by the configured reasoner."""

    def __init__(self, model: ReasoningModel, *, max_attempts: int = 2) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            model (ReasoningModel): Provider-neutral model adapter used by the operation.
            max_attempts (int): Hard ceiling for structured generation attempts.
        """
        self._model = model
        self._max_attempts = max(1, min(max_attempts, 2))
        self._request_count = 0

    def equivalent(self, premise: str, hypothesis: str) -> bool:
        """Return whether both requirements entail each other.

        Args:
            premise (str): Premise required by the operation's typed contract.
            hypothesis (str): Hypothesis required by the operation's typed contract.

        Returns:
            bool: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
        prompt = json.dumps(
            {"premise": premise, "hypothesis": hypothesis},
            ensure_ascii=False,
        )
        self._request_count += 1
        pair_fingerprint = hashlib.sha256(
            f"{_norm(premise)}\n{_norm(hypothesis)}".encode()
        ).hexdigest()[:16]
        result = self._model.generate_structured(
            prompt=prompt,
            schema=_BidirectionalEntailmentOutput,
            system_message=PromptRequirementIdentity.SYS_PROMPT_REQUIREMENT_IDENTITY.value,
            operation="requirement_identity.entailment",
            request_id=f"pair-{self._request_count:06d}-{pair_fingerprint}",
            max_attempts=self._max_attempts,
        )
        return result.premise_entails_hypothesis and result.hypothesis_entails_premise


@dataclass
class MemoryEntry:
    """Coordinate memory entry behavior within the services boundary."""

    requirement_id: str
    revision_id: str
    statement: str
    normalized_statement: str
    requirement_type: str
    source_req_id: str | None = None
    signature: str = ""
    embedding: list[float] | None = None
    semantic_recall_enabled: bool = True

    def __post_init__(self) -> None:
        """Execute the post init operation within its declared architectural boundary."""
        if not self.signature:
            self.signature = requirement_lineage_signature(self.statement, self.requirement_type)
        if not self.normalized_statement:
            self.normalized_statement = self.statement.strip().lower()


@dataclass(frozen=True)
class Candidate:
    """Coordinate candidate behavior within the services boundary."""

    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReconcileResult:
    """Coordinate reconcile result behavior within the services boundary."""

    decision: Decision
    requirement_id: str | None
    revision_id: str | None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    reranker_order: tuple[str, ...] = field(default_factory=tuple)
    judge_result: str | None = None


def _family_hash(requirement_type: str, text: str) -> str:
    # Exact-statement identity is family-scoped: identical wording under a different
    # requirement family (an Acceptance Criterion vs a Business Requirement) is not
    # the same obligation and must not collapse.
    """Execute the family hash operation within its declared architectural boundary.

    Args:
        requirement_type (str): Requirement type required by the operation's typed contract.
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    family = normalize_requirement_family(requirement_type)
    return hashlib.sha256(f"{family}::{_norm(text)}".encode()).hexdigest()


def _norm(text: str) -> str:
    """Execute the norm operation within its declared architectural boundary.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    """Execute the tokens operation within its declared architectural boundary.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        set[str]: The typed result produced by the operation.
    """
    return set(_TOKEN.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    """Execute the jaccard operation within its declared architectural boundary.

    Args:
        left (set[str]): Left required by the operation's typed contract.
        right (set[str]): Right required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(left: list[float], right: list[float]) -> float:
    """Execute the cosine operation within its declared architectural boundary.

    Args:
        left (list[float]): Left required by the operation's typed contract.
        right (list[float]): Right required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class RequirementMemory:
    """Cross-version + intra-run requirement recall with fail-closed reconciliation."""

    def __init__(
        self,
        *,
        settings: RequirementIdentitySettings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        judge: EntailmentJudge | None = None,
        logger: object | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (RequirementIdentitySettings): Validated settings that control this operation.
            embedder (Embedder | None): Provider-neutral model adapter used by the operation.
            reranker (Reranker | None): Provider-neutral model adapter used by the operation.
            judge (EntailmentJudge | None): Judge required by the operation's typed contract.
            logger (object | None): Optional run-scoped logger used for sanitized diagnostics.
        """
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker if settings.use_reranker else None
        self._judge = judge
        self._logger = logger
        self._entries: list[MemoryEntry] = []
        self._by_hash: dict[str, MemoryEntry] = {}
        self._by_signature: dict[str, MemoryEntry] = {}
        self._entailment_cache: dict[tuple[str, str], bool] = {}
        self._entailment_calls_used = 0
        self._entailment_cache_hits = 0
        self._budget_warning_emitted = False

    @property
    def size(self) -> int:
        """Execute the size operation within its declared architectural boundary.

        Returns:
            int: The typed result produced by the operation.
        """
        return len(self._entries)

    @property
    def entailment_calls_used(self) -> int:
        """Return the number of uncached directional entailment calls used by this memory."""
        return self._entailment_calls_used

    @property
    def entailment_cache_hits(self) -> int:
        """Return the number of cached pairwise decisions reused by this memory."""
        return self._entailment_cache_hits

    def seed(self, entries: Iterable[MemoryEntry]) -> None:
        """Seed seed.

        Args:
            entries (Iterable[MemoryEntry]): Ordered entries processed without changing their
                                             identities.
        """
        for entry in entries:
            self.add(entry)

    def add(self, entry: MemoryEntry) -> None:
        """Execute the add operation within its declared architectural boundary.

        Args:
            entry (MemoryEntry): Entry required by the operation's typed contract.

        Side Effects:
            May invoke configured model or workflow providers.
        """
        if entry.embedding is None and self._embedder is not None:
            vectors = self._embedder.embed_documents([entry.statement])
            entry.embedding = vectors[0] if vectors else None
        self._entries.append(entry)
        self._by_hash.setdefault(
            _family_hash(entry.requirement_type, entry.normalized_statement or entry.statement),
            entry,
        )
        self._by_signature.setdefault(entry.signature, entry)

    def exact_statement_match(self, statement: str, requirement_type: str) -> MemoryEntry | None:
        """Execute the exact statement match operation within its declared architectural boundary.

        Args:
            statement (str): Statement required by the operation's typed contract.
            requirement_type (str): Requirement type required by the operation's typed contract.

        Returns:
            MemoryEntry | None: The typed result produced by the operation.
        """
        return self._by_hash.get(_family_hash(requirement_type, statement))

    def exact_signature_match(self, statement: str, requirement_type: str) -> MemoryEntry | None:
        """Execute the exact signature match operation within its declared architectural boundary.

        Args:
            statement (str): Statement required by the operation's typed contract.
            requirement_type (str): Requirement type required by the operation's typed contract.

        Returns:
            MemoryEntry | None: The typed result produced by the operation.
        """
        return self._by_signature.get(requirement_lineage_signature(statement, requirement_type))

    def candidates(
        self,
        *,
        statement: str,
        requirement_type: str,
        source_req_id: str | None = None,
    ) -> list[Candidate]:
        """Bounded, deduped candidate recall combining every non-authoritative signal."""
        signature = requirement_lineage_signature(statement, requirement_type)
        query_tokens = _tokens(statement)
        query_vector: list[float] | None = None
        if self._embedder is not None:
            vectors = self._embedder.embed_documents([statement])
            query_vector = vectors[0] if vectors else None

        scored: dict[str, tuple[float, list[str]]] = {}

        def _bump(entry: MemoryEntry, score: float, reason: str) -> None:
            """Execute the bump operation within its declared architectural boundary.

            Args:
                entry (MemoryEntry): Entry required by the operation's typed contract.
                score (float): Score required by the operation's typed contract.
                reason (str): Reason required by the operation's typed contract.
            """
            key = entry.revision_id or entry.requirement_id
            current = scored.get(key)
            if current is None or score > current[0]:
                reasons = list(current[1]) if current else []
                reasons.append(reason)
                scored[key] = (max(score, current[0] if current else score), reasons)
            else:
                current[1].append(reason)

        for entry in self._entries:
            if not entry.semantic_recall_enabled:
                continue
            if entry.signature == signature:
                _bump(entry, 1.0, "signature")
            if source_req_id and entry.source_req_id and entry.source_req_id == source_req_id:
                _bump(entry, 0.5, "source_req_id")
            overlap = _jaccard(query_tokens, _tokens(entry.statement))
            if overlap >= self._settings.token_overlap_threshold:
                _bump(entry, overlap, "token_overlap")
            if query_vector is not None and entry.embedding is not None:
                cosine = _cosine(query_vector, entry.embedding)
                if cosine >= self._settings.recall_cosine_threshold:
                    _bump(entry, cosine, "embedding_cosine")

        entries_by_key = {e.revision_id or e.requirement_id: e for e in self._entries}
        pool = [
            Candidate(entry=entries_by_key[key], score=score, reasons=tuple(reasons))
            for key, (score, reasons) in scored.items()
        ]
        pool.sort(key=lambda candidate: (-candidate.score, candidate.entry.requirement_id))
        pool = pool[: self._settings.candidate_top_k]
        if self._reranker is not None and len(pool) > 1:
            order = self._reranker.rerank(statement, [c.entry.statement for c in pool])
            if sorted(order) == list(range(len(pool))):
                pool = [pool[index] for index in order]
        return pool

    def reconcile(
        self,
        *,
        statement: str,
        requirement_type: str,
        normalized_statement: str,
        source_req_id: str | None = None,
        identity_index: int | None = None,
        identity_total: int | None = None,
    ) -> ReconcileResult:
        """Decide identity for one candidate against memory; never merges on score."""
        started = perf_counter()
        calls_before = self._entailment_calls_used
        cache_hits_before = self._entailment_cache_hits
        candidate_count = 0

        def _finish(result: ReconcileResult) -> ReconcileResult:
            """Emit the terminal safe metrics for this identity and return its result."""
            self._log_decision(
                result=result,
                identity_index=identity_index,
                identity_total=identity_total,
                candidate_count=candidate_count,
                model_call_count=self._entailment_calls_used - calls_before,
                cache_hits=self._entailment_cache_hits - cache_hits_before,
                elapsed_ms=(perf_counter() - started) * 1000.0,
            )
            return result

        family = normalize_requirement_family(requirement_type)

        exact = self.exact_statement_match(normalized_statement or statement, requirement_type)
        if exact is not None:
            candidate_count = 1
            return _finish(
                ReconcileResult(
                    decision="EXACT",
                    requirement_id=exact.requirement_id,
                    revision_id=exact.revision_id,
                    reasons=("exact_normalized_statement",),
                    candidate_ids=(exact.requirement_id,),
                )
            )

        signature_match = self.exact_signature_match(statement, requirement_type)
        if signature_match is not None:
            candidate_count = 1
            return _finish(
                ReconcileResult(
                    decision="SAME_LINEAGE_REVISION",
                    requirement_id=signature_match.requirement_id,
                    revision_id=None,
                    reasons=("exact_signature",),
                    candidate_ids=(signature_match.requirement_id,),
                )
            )

        pool = self.candidates(
            statement=statement,
            requirement_type=requirement_type,
            source_req_id=source_req_id,
        )
        candidate_scores = {candidate.entry.requirement_id: candidate.score for candidate in pool}
        reranker_order = tuple(candidate.entry.requirement_id for candidate in pool)
        # Only same-family candidates may merge (AC never becomes a BR revision).
        eligible = [
            candidate
            for candidate in pool
            if normalize_requirement_family(candidate.entry.requirement_type) == family
        ]
        candidate_count = len(eligible)
        if not eligible:
            return _finish(
                ReconcileResult(
                    decision="DISTINCT",
                    requirement_id=None,
                    revision_id=None,
                    reasons=("no_recall_candidate",),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                )
            )

        if not self._settings.require_entailment_for_merge:
            return _finish(
                ReconcileResult(
                    decision="DISTINCT",
                    requirement_id=None,
                    revision_id=None,
                    reasons=("semantic_merge_disabled",),
                    candidate_ids=tuple(c.entry.requirement_id for c in eligible),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                )
            )

        if self._judge is None:
            # No judge available: recall alone must not merge. Fail closed.
            return _finish(
                ReconcileResult(
                    decision="DISTINCT",
                    requirement_id=None,
                    revision_id=None,
                    reasons=("recall_only_no_judge_fail_closed",),
                    candidate_ids=tuple(c.entry.requirement_id for c in eligible),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                )
            )

        if self._judge is not None:
            # Complete-linkage against canonical representatives only. More than
            # one mutually-entailing lineage is ambiguous, never a reason to union
            # those lineages transitively.
            matches: list[Candidate] = []
            for candidate in eligible:
                equivalent = self._equivalent(statement, candidate.entry.statement)
                if equivalent is None:
                    return _finish(
                        self._budget_exhausted_result(
                            eligible=eligible,
                            candidate_scores=candidate_scores,
                            reranker_order=reranker_order,
                        )
                    )
                if equivalent:
                    matches.append(candidate)
                    matched_ids = {match.entry.requirement_id for match in matches}
                    if len(matched_ids) > 1:
                        return _finish(
                            ReconcileResult(
                                decision="AMBIGUOUS",
                                requirement_id=None,
                                revision_id=None,
                                reasons=("multiple_mutual_entailment_matches_fail_closed",),
                                candidate_ids=tuple(sorted(matched_ids)),
                                candidate_scores=candidate_scores,
                                reranker_order=reranker_order,
                                judge_result="multiple_mutual_matches",
                            )
                        )
            if matches:
                best = matches[0]
                return _finish(
                    ReconcileResult(
                        decision="EXACT",
                        requirement_id=best.entry.requirement_id,
                        revision_id=best.entry.revision_id,
                        reasons=("bidirectional_entailment", *best.reasons),
                        candidate_ids=(best.entry.requirement_id,),
                        candidate_scores=candidate_scores,
                        reranker_order=reranker_order,
                        judge_result="mutual_entailment",
                    )
                )
            return _finish(
                ReconcileResult(
                    decision="DISTINCT",
                    requirement_id=None,
                    revision_id=None,
                    reasons=("entailment_not_mutual_fail_closed",),
                    candidate_ids=tuple(c.entry.requirement_id for c in eligible),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                    judge_result="not_mutual",
                )
            )

    def _equivalent(self, premise: str, hypothesis: str) -> bool | None:
        """Return a cached equivalence judgment or None when the run budget is exhausted."""
        normalized_premise = _norm(premise)
        normalized_hypothesis = _norm(hypothesis)
        key = (
            (normalized_premise, normalized_hypothesis)
            if normalized_premise <= normalized_hypothesis
            else (normalized_hypothesis, normalized_premise)
        )
        cached = self._entailment_cache.get(key)
        if cached is not None:
            self._entailment_cache_hits += 1
            return cached
        if self._entailment_calls_used >= self._settings.max_entailment_calls:
            self._warn_budget_exhausted()
            return None
        if self._judge is None:
            return None
        self._entailment_calls_used += 1
        result = self._judge.equivalent(premise, hypothesis)
        self._entailment_cache[key] = result
        return result

    def _budget_exhausted_result(
        self,
        *,
        eligible: list[Candidate],
        candidate_scores: dict[str, float],
        reranker_order: tuple[str, ...],
    ) -> ReconcileResult:
        """Return the canonical fail-closed decision for an exhausted call budget."""
        return ReconcileResult(
            decision="DISTINCT",
            requirement_id=None,
            revision_id=None,
            reasons=("entailment_budget_exhausted",),
            candidate_ids=tuple(candidate.entry.requirement_id for candidate in eligible),
            candidate_scores=candidate_scores,
            reranker_order=reranker_order,
            judge_result="budget_exhausted",
        )

    def _warn_budget_exhausted(self) -> None:
        """Emit one sanitized warning when the run-wide entailment budget is exhausted."""
        if self._budget_warning_emitted:
            return
        self._budget_warning_emitted = True
        warning = getattr(self._logger, "warning", None)
        if callable(warning):
            warning(
                "Requirement identity entailment budget exhausted; remaining matches fail closed",
                step="resolve_requirement_identity",
                operation="requirement_identity.entailment",
                entailment_calls_used=self._entailment_calls_used,
                max_entailment_calls=self._settings.max_entailment_calls,
                status="degraded",
            )

    def _log_decision(
        self,
        *,
        result: ReconcileResult,
        identity_index: int | None,
        identity_total: int | None,
        candidate_count: int,
        model_call_count: int,
        cache_hits: int,
        elapsed_ms: float,
    ) -> None:
        """Emit one safe terminal reconciliation record for a requirement identity."""
        if identity_index is None or identity_total is None:
            return
        debug = getattr(self._logger, "debug", None)
        if callable(debug):
            debug(
                "Requirement identity reconciliation completed",
                step="resolve_requirement_identity",
                operation="requirement_identity.reconcile",
                identity_index=identity_index,
                identity_total=identity_total,
                candidate_count=candidate_count,
                cache_hits=cache_hits,
                model_call_count=model_call_count,
                entailment_calls_used=self._entailment_calls_used,
                budget_remaining=max(
                    0,
                    self._settings.max_entailment_calls - self._entailment_calls_used,
                ),
                decision=result.decision,
                reason=result.reasons[0] if result.reasons else "unspecified",
                elapsed_ms=round(elapsed_ms, 3),
                status="completed",
            )
