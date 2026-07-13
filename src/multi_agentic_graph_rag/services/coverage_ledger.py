"""Bounded, in-memory coverage ledger for duplicate-aware requirement discovery.

Production note:

- This ledger is order-dependent because chunk processing is sequential. Entries
  discovered from earlier chunks are injected into the prompts of later chunks.
- It exists only to reduce LLM paraphrase drift during discovery, nudging the model
  to reuse a known ``requirement_key``/``req_text`` when a later chunk restates the
  same obligation. It never skips a requirement and never suppresses source evidence.
- The deterministic requirement builder (``services/requirement_builder.py``) remains
  the source of truth for deduplication, permanent ID assignment, evidence
  accumulation, and revision/supersede semantics.
- Future Phase 2 may add a post-discovery semantic merge that reuses the scenario
  dedup pattern from ``services/scenario_dedup.py`` (canonicalization + embedding
  recall + optional reranking + deterministic LLM entailment). Phase 2 is out of
  scope for this change.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.domain.schemas import (
    RequirementDiscoveryOutput,
    RequirementRevisionSnapshot,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel
from multi_agentic_graph_rag.services.requirement_builder import (
    derive_requirement_key,
    normalize_requirement_statement,
)
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    requirement_lineage_signature,
)


@dataclass(frozen=True)
class LedgerEntry:
    requirement_key: str
    statement: str
    normalized_statement: str
    requirement_family: str = "Functional Requirement"
    semantic_signature: str = ""


class CoverageLedger:
    """Track discovered requirements to reduce paraphrase drift in later chunks.

    Identity is ``(requirement_key, normalized_statement)``, never ``requirement_key``
    alone: the key intentionally masks revision values (numbers, quoted values,
    temperatures), so a same-key/changed-statement pair is a legitimately different
    revision and must be retained as a separate ledger entry.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        injection_top_k: int,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if injection_top_k <= 0:
            raise ValueError("injection_top_k must be positive")
        self._max_entries = max_entries
        self._injection_top_k = min(injection_top_k, max_entries)
        self._embedder = embedder
        self._entries: list[LedgerEntry] = []
        self._identities: set[tuple[str, str]] = set()
        self._vectors: dict[tuple[str, str], list[float]] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    def record(self, output: RequirementDiscoveryOutput) -> int:
        """Record validated per-chunk discovery output; return newly added entries.

        First exact ``(requirement_key, normalized_statement)`` wins; oldest entries are
        FIFO-evicted once ``max_entries`` is exceeded.
        """
        added = 0
        for entry in _entries_from_output(output):
            identity = self._identity(entry)
            if identity in self._identities:
                continue
            self._entries.append(entry)
            self._identities.add(identity)
            added += 1
            self._evict_overflow()
        return added

    def prime_prior_requirements(self, statements: Iterable[str]) -> int:
        """Seed the ledger with prior document-version active requirement statements.

        Loaded *before* discovery so the discovery prompt can show the model the
        obligations already established in earlier versions. The model is then
        nudged to reuse canonical wording (reducing paraphrase drift), but is still
        free to emit a fresh occurrence whenever the current chunk supports it —
        the ledger never suppresses evidence.
        """
        added = 0
        for statement in statements:
            text = statement.strip()
            if not text:
                continue
            entry = LedgerEntry(
                requirement_key=derive_requirement_key(text, None),
                statement=text,
                normalized_statement=normalize_requirement_statement(text),
                semantic_signature=requirement_lineage_signature(text, "Functional Requirement"),
            )
            identity = self._identity(entry)
            if identity in self._identities:
                continue
            self._entries.append(entry)
            self._identities.add(identity)
            added += 1
            self._evict_overflow()
        return added

    def prime_prior_revisions(self, revisions: Iterable[RequirementRevisionSnapshot]) -> int:
        added = 0
        for revision in revisions:
            text = revision.statement.strip()
            if not text:
                continue
            family = revision.requirement_type or "Functional Requirement"
            entry = LedgerEntry(
                requirement_key=derive_requirement_key(text, None),
                statement=text,
                normalized_statement=revision.normalized_statement,
                requirement_family=family,
                semantic_signature=requirement_lineage_signature(text, family),
            )
            identity = self._identity(entry)
            if identity in self._identities:
                continue
            self._entries.append(entry)
            self._identities.add(identity)
            added += 1
            self._evict_overflow()
        return added

    def count_exact_converged(self, output: RequirementDiscoveryOutput) -> int:
        """Count output requirements exactly matching an existing ledger entry.

        Must be called before ``record`` for the same output to measure convergence.
        """
        return sum(
            1 for entry in _entries_from_output(output) if self._identity(entry) in self._identities
        )

    def select_for_chunk(self, chunk_text: str) -> list[LedgerEntry]:
        """Select ledger entries to inject into a chunk prompt."""
        if not self._entries:
            return []
        if len(self._entries) <= self._injection_top_k:
            return list(self._entries)
        embedder = self._embedder
        if embedder is None:
            return list(self._entries[-self._injection_top_k :])
        chunk_vector = embedder.embed_documents([chunk_text])[0]
        self._ensure_vectors(embedder)
        similarities = [
            _cosine(chunk_vector, self._vectors[self._identity(entry)]) for entry in self._entries
        ]
        ranked = sorted(
            range(len(self._entries)),
            key=lambda index: (-similarities[index], index),
        )
        return [self._entries[index] for index in ranked[: self._injection_top_k]]

    def render_prompt_section(self, entries: list[LedgerEntry]) -> str:
        """Render the ledger prompt section, or an empty string when there is nothing."""
        if not entries:
            return ""
        payload = [
            {
                "requirement_key": entry.requirement_key,
                "statement": entry.statement,
                "requirement_family": entry.requirement_family,
                "semantic_signature": entry.semantic_signature,
            }
            for entry in entries
        ]
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"{PromptRequirementDiscovery.LEDGER_SECTION_HEADER.value}{body}\n\n"

    def _ensure_vectors(self, embedder: EmbeddingModel) -> None:
        missing = [entry for entry in self._entries if self._identity(entry) not in self._vectors]
        if not missing:
            return
        vectors = embedder.embed_documents([entry.statement for entry in missing])
        for entry, vector in zip(missing, vectors, strict=True):
            self._vectors[self._identity(entry)] = vector

    def _evict_overflow(self) -> None:
        while len(self._entries) > self._max_entries:
            evicted = self._entries.pop(0)
            identity = self._identity(evicted)
            self._identities.discard(identity)
            self._vectors.pop(identity, None)

    @staticmethod
    def _identity(entry: LedgerEntry) -> tuple[str, str]:
        return (entry.requirement_key, entry.normalized_statement)


def _entries_from_output(output: RequirementDiscoveryOutput) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for chunk_output in output.chunks:
        for fact in chunk_output.facts:
            for requirement in fact.requirements:
                entries.append(
                    LedgerEntry(
                        requirement_key=derive_requirement_key(
                            requirement.statement, requirement.requirement_key
                        ),
                        statement=requirement.statement.strip(),
                        normalized_statement=normalize_requirement_statement(requirement.statement),
                        requirement_family=requirement.requirement_family,
                        semantic_signature=requirement_lineage_signature(
                            requirement.statement, requirement.requirement_family
                        ),
                    )
                )
    return entries


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
