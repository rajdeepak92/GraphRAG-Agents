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
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[int]: ...


class EntailmentJudge(Protocol):
    """Strict directional entailment: does ``premise`` fully entail ``hypothesis``?"""

    def entails(self, premise: str, hypothesis: str) -> bool: ...


class _EntailmentOutput(StrictModel):
    entails: bool


class ModelEntailmentJudge:
    """Provider-neutral strict entailment judge backed by the configured reasoner."""

    def __init__(self, model: ReasoningModel) -> None:
        self._model = model

    def entails(self, premise: str, hypothesis: str) -> bool:
        prompt = (
            PromptRequirementIdentity.BIDIRECTIONAL_ENTAILMENT.value
            + "PREMISE: "
            + json.dumps(premise)
            + "\nHYPOTHESIS: "
            + json.dumps(hypothesis)
        )
        return self._model.generate_structured(prompt=prompt, schema=_EntailmentOutput).entails


@dataclass
class MemoryEntry:
    requirement_id: str
    revision_id: str
    statement: str
    normalized_statement: str
    requirement_type: str
    source_req_id: str | None = None
    signature: str = ""
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = requirement_lineage_signature(self.statement, self.requirement_type)
        if not self.normalized_statement:
            self.normalized_statement = self.statement.strip().lower()


@dataclass(frozen=True)
class Candidate:
    entry: MemoryEntry
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReconcileResult:
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
    family = normalize_requirement_family(requirement_type)
    return hashlib.sha256(f"{family}::{_norm(text)}".encode()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(left: list[float], right: list[float]) -> float:
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
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker if settings.use_reranker else None
        self._judge = judge
        self._entries: list[MemoryEntry] = []
        self._by_hash: dict[str, MemoryEntry] = {}
        self._by_signature: dict[str, MemoryEntry] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    def seed(self, entries: Iterable[MemoryEntry]) -> None:
        for entry in entries:
            self.add(entry)

    def add(self, entry: MemoryEntry) -> None:
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
        return self._by_hash.get(_family_hash(requirement_type, statement))

    def exact_signature_match(self, statement: str, requirement_type: str) -> MemoryEntry | None:
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
            key = entry.revision_id or entry.requirement_id
            current = scored.get(key)
            if current is None or score > current[0]:
                reasons = list(current[1]) if current else []
                reasons.append(reason)
                scored[key] = (max(score, current[0] if current else score), reasons)
            else:
                current[1].append(reason)

        for entry in self._entries:
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
    ) -> ReconcileResult:
        """Decide identity for one candidate against memory; never merges on score."""
        family = normalize_requirement_family(requirement_type)

        exact = self.exact_statement_match(normalized_statement or statement, requirement_type)
        if exact is not None:
            return ReconcileResult(
                decision="EXACT",
                requirement_id=exact.requirement_id,
                revision_id=exact.revision_id,
                reasons=("exact_normalized_statement",),
                candidate_ids=(exact.requirement_id,),
            )

        signature_match = self.exact_signature_match(statement, requirement_type)
        if signature_match is not None:
            return ReconcileResult(
                decision="SAME_LINEAGE_REVISION",
                requirement_id=signature_match.requirement_id,
                revision_id=None,
                reasons=("exact_signature",),
                candidate_ids=(signature_match.requirement_id,),
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
        if not eligible:
            return ReconcileResult(
                decision="DISTINCT",
                requirement_id=None,
                revision_id=None,
                reasons=("no_recall_candidate",),
                candidate_scores=candidate_scores,
                reranker_order=reranker_order,
            )

        if self._settings.require_entailment_for_merge and self._judge is None:
            # No judge available: recall alone must not merge. Fail closed.
            return ReconcileResult(
                decision="DISTINCT",
                requirement_id=None,
                revision_id=None,
                reasons=("recall_only_no_judge_fail_closed",),
                candidate_ids=tuple(c.entry.requirement_id for c in eligible),
                candidate_scores=candidate_scores,
                reranker_order=reranker_order,
            )

        if self._judge is not None:
            # Complete-linkage against canonical representatives only. More than
            # one mutually-entailing lineage is ambiguous, never a reason to union
            # those lineages transitively.
            matches = [
                candidate
                for candidate in eligible
                if self._judge.entails(statement, candidate.entry.statement)
                and self._judge.entails(candidate.entry.statement, statement)
            ]
            matched_ids = {candidate.entry.requirement_id for candidate in matches}
            if len(matched_ids) > 1:
                return ReconcileResult(
                    decision="AMBIGUOUS",
                    requirement_id=None,
                    revision_id=None,
                    reasons=("multiple_mutual_entailment_matches_fail_closed",),
                    candidate_ids=tuple(sorted(matched_ids)),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                    judge_result="multiple_mutual_matches",
                )
            if matches:
                best = matches[0]
                return ReconcileResult(
                    decision="EXACT",
                    requirement_id=best.entry.requirement_id,
                    revision_id=best.entry.revision_id,
                    reasons=("bidirectional_entailment", *best.reasons),
                    candidate_ids=(best.entry.requirement_id,),
                    candidate_scores=candidate_scores,
                    reranker_order=reranker_order,
                    judge_result="mutual_entailment",
                )
            return ReconcileResult(
                decision="DISTINCT",
                requirement_id=None,
                revision_id=None,
                reasons=("entailment_not_mutual_fail_closed",),
                candidate_ids=tuple(c.entry.requirement_id for c in eligible),
                candidate_scores=candidate_scores,
                reranker_order=reranker_order,
                judge_result="not_mutual",
            )

        return ReconcileResult(
            decision="DISTINCT",
            requirement_id=None,
            revision_id=None,
            reasons=("fail_closed",),
            candidate_ids=tuple(c.entry.requirement_id for c in eligible),
            candidate_scores=candidate_scores,
            reranker_order=reranker_order,
        )
