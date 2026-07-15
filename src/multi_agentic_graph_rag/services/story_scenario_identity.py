"""LLM-proposed / Python-decided identity for regenerated stories and scenarios.

This mirrors the requirement-identity pattern and reuses its machinery rather
than creating a parallel mechanism:

* an exact normalized-title match is a deterministic Python decision (fast path,
  no model call) — the same content key ``story_id`` / ``scenario_id`` is derived
  from;
* otherwise a bounded embedding recall (reusing the shared cosine) proposes the
  closest prior candidates under the same parent, and a strict *bidirectional*
  LLM entailment (reusing :class:`ModelEntailmentJudge`) proposes whether the two
  are the same;
* Python always makes the final decision and fails closed to a new lineage when
  the signal is weak or the model is unavailable — identity is never assigned by
  output ordinal.

The resolver is optional: with no models wired in (offline / deterministic runs)
the caller keeps only the exact-title fast path, which remains fully
deterministic and ordinal-free.
"""

from __future__ import annotations

from typing import Protocol

from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel
from multi_agentic_graph_rag.services.similarity import cosine


class EntailmentJudge(Protocol):
    """Strict bidirectional equivalence judge."""

    def equivalent(self, premise: str, hypothesis: str) -> bool: ...


class StoryScenarioIdentityResolver:
    """Propose a semantic lineage match for one regenerated record.

    ``recall_top_k`` bounds how many prior candidates are entailment-checked, and
    ``recall_floor`` is the cosine below which a candidate is not even proposed to
    the judge (keeps the LLM from adjudicating clearly-unrelated pairs).
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingModel,
        judge: EntailmentJudge,
        recall_top_k: int = 5,
        recall_floor: float = 0.60,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            embedder (EmbeddingModel): Provider-neutral embedding model used for candidate recall.
            judge (EntailmentJudge): Strict bidirectional entailment judge the LLM proposes with.
            recall_top_k (int): Bounded number of prior candidates entailment-checked per record.
            recall_floor (float): Cosine below which a candidate is not proposed to the judge.
        """
        self.embedder = embedder
        self.judge = judge
        self.recall_top_k = max(1, recall_top_k)
        self.recall_floor = recall_floor

    def find_semantic_match(
        self,
        new_text: str,
        candidate_pool: list[tuple[str, str]],
    ) -> str | None:
        """Return an existing id whose content is the same obligation, or None.

        ``candidate_pool`` is the still-unconsumed prior records under the same
        parent as ``(existing_id, existing_text)``. A match requires bidirectional
        entailment; anything else fails closed to None (a new lineage).
        """
        if not new_text.strip() or not candidate_pool:
            return None
        documents = [new_text, *[text for _, text in candidate_pool]]
        vectors = self.embedder.embed_documents(documents)
        if len(vectors) != len(documents):
            raise ValueError("embedding provider returned an unexpected vector count")
        query = vectors[0]
        ranked = sorted(
            zip(candidate_pool, vectors[1:], strict=True),
            key=lambda item: cosine(query, item[1]),
            reverse=True,
        )
        for (candidate_id, candidate_text), vector in ranked[: self.recall_top_k]:
            if cosine(query, vector) < self.recall_floor:
                break
            if self.judge.equivalent(new_text, candidate_text):
                return candidate_id
        return None
