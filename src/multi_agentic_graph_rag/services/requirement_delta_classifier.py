"""Strict requirement delta classification built on existing revision scaffolding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from multi_agentic_graph_rag.domain.schemas import (
    RequirementDeltaDecision,
    RequirementDeltaEvent,
    RequirementRevisionSnapshot,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, ReasoningModel


@dataclass(frozen=True)
class RequirementDeltaConfig:
    """Coordinate requirement delta config behavior within the services boundary."""

    contradiction_gate_cosine: float = 0.60


class RequirementDeltaClassifier:
    """Coordinate requirement delta classifier behavior within the services boundary."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        reasoner: ReasoningModel,
        cfg: RequirementDeltaConfig,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            embedder (EmbeddingModel): Provider-neutral model adapter used by the operation.
            reasoner (ReasoningModel): Provider-neutral model adapter used by the operation.
            cfg (RequirementDeltaConfig): Cfg required by the operation's typed contract.
        """
        self.embedder = embedder
        self.reasoner = reasoner
        self.cfg = cfg

    def classify(
        self,
        prior: RequirementRevisionSnapshot | None,
        incoming: VerifiedRequirement,
    ) -> RequirementDeltaDecision:
        """Classify classify.

        Args:
            prior (RequirementRevisionSnapshot | None): Prior required by the operation's typed
                                                        contract.
            incoming (VerifiedRequirement): Incoming required by the operation's typed contract.

        Returns:
            RequirementDeltaDecision: The typed result produced by the operation.

        Side Effects:
            May invoke configured model or workflow providers.
        """
        if prior is None:
            return _decision(incoming, "new", reason="no prior active lineage match")
        if prior.revision_id == incoming.revision_id:
            return _decision(
                incoming,
                "unchanged",
                prior_revision_id=prior.revision_id,
                reason="same active revision id",
            )
        if _looks_like_explicit_removal(prior.statement, incoming.statement):
            return _decision(
                incoming,
                "strictly_outdated",
                prior_revision_id=prior.revision_id,
                reason="explicit contradiction/removal language detected",
            )
        prompt = (
            "Classify a requirement version delta into exactly one label: new, updated, "
            "strictly_outdated, unchanged.\n"
            "strictly_outdated means explicit contradiction or explicit removal only. Simple "
            "absence is not strictly_outdated.\n"
            "Return one JSON object matching this schema:\n"
            '{"requirement_id":"...","revision_id":"...","label":"updated",'
            '"prior_revision_id":"...","reason":"..."}\n\n'
            f"Prior active requirement:\n{json.dumps(prior.model_dump(mode='json'), indent=2)}\n\n"
            f"Incoming requirement:\n{json.dumps(incoming.model_dump(mode='json'), indent=2)}\n"
        )
        result = self.reasoner.generate_structured(prompt=prompt, schema=RequirementDeltaDecision)
        if result.label == "new":
            return result.model_copy(update={"label": "updated"})
        return result

    def map_event(
        self, event: RequirementDeltaEvent
    ) -> Literal[
        "new",
        "updated",
        "strictly_outdated",
        "unchanged",
    ]:
        """Execute the map event operation within its declared architectural boundary.

        Args:
            event (RequirementDeltaEvent): Event required by the operation's typed contract.

        Returns:
            Literal['new', 'updated', 'strictly_outdated', 'unchanged']: The typed result produced
            by the operation.
        """
        if event.event_type == "new":
            return "new"
        if event.event_type == "duplicate":
            return "unchanged"
        if event.event_type in {"changed", "superseded", "updated"}:
            return "updated"
        if event.event_type == "strictly_outdated":
            return "strictly_outdated"
        return "unchanged"


def _decision(
    incoming: VerifiedRequirement,
    label: Literal["new", "updated", "strictly_outdated", "unchanged"],
    *,
    prior_revision_id: str | None = None,
    reason: str,
) -> RequirementDeltaDecision:
    """Execute the decision operation within its declared architectural boundary.

    Args:
        incoming (VerifiedRequirement): Incoming required by the operation's typed contract.
        label (Literal['new', 'updated', 'strictly_outdated', 'unchanged']): Label required by the
                                                                             operation's typed
                                                                             contract.
        prior_revision_id (str | None): Canonical prior revision id used as a safe operational
                                        anchor.
        reason (str): Reason required by the operation's typed contract.

    Returns:
        RequirementDeltaDecision: The typed result produced by the operation.
    """
    return RequirementDeltaDecision(
        requirement_id=incoming.requirement_id,
        revision_id=incoming.revision_id,
        label=label,
        prior_revision_id=prior_revision_id,
        reason=reason,
    )


def _looks_like_explicit_removal(prior: str, incoming: str) -> bool:
    """Execute the looks like explicit removal operation within its declared architectural boundary.

    Args:
        prior (str): Prior required by the operation's typed contract.
        incoming (str): Incoming required by the operation's typed contract.

    Returns:
        bool: The typed result produced by the operation.

    Side Effects:
        May create or atomically replace files in the configured artifact boundary.
    """
    prior_norm = prior.lower()
    incoming_norm = incoming.lower()
    removal_markers = (
        "does not support",
        "do not support",
        "no longer supports",
        "must not",
        "shall not",
        "removed",
        "disabled",
        "not supported",
    )
    if not any(marker in incoming_norm for marker in removal_markers):
        return False
    prior_tokens = {token for token in prior_norm.replace("-", " ").split() if len(token) > 3}
    incoming_tokens = {token for token in incoming_norm.replace("-", " ").split() if len(token) > 3}
    if not prior_tokens:
        return False
    overlap = len(prior_tokens & incoming_tokens) / len(prior_tokens)
    return overlap >= 0.35
