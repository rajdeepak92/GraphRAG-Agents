"""Assemble bounded, reproducible context manifests (plan §10.4).

Continuously supplying context is an iterative tool-assisted process, not one
enormous prompt (§10.1). This builds a checksummed ``ContextManifest`` recording
exactly which code and data caused a patch, enforcing a token budget so the
model receives a bounded working set rather than the whole repository.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_graph_rag.domain.codegen_schemas import (
    ContextManifest,
    ContextManifestItem,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.identifiers import make_context_manifest_id

# Rough token estimate: ~4 characters per token (deterministic, no tokenizer).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class RetrievedItem:
    """A candidate context item with its estimated size, prior to budgeting."""

    item: ContextManifestItem
    estimated_chars: int
    tier: int  # 0 = always-present contract, higher = more optional


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate used for budgeting (no external tokenizer)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def assemble_manifest(
    *,
    codegen_run_id: str,
    scenario_id: str,
    sequence: int,
    framework_snapshot_id: str,
    test_data_snapshot_id: str,
    candidates: list[RetrievedItem],
    token_budget: int,
    filesystem_snapshot_checksum: str | None = None,
    worktree_tree_hash: str | None = None,
) -> ContextManifest:
    """Select items by tier within the token budget and build a checksummed manifest."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    ordered = sorted(enumerate(candidates), key=lambda pair: (pair[1].tier, pair[0]))
    selected: list[ContextManifestItem] = []
    used = 0
    for _, candidate in ordered:
        cost = max(1, candidate.estimated_chars // _CHARS_PER_TOKEN)
        # Tier-0 contract items are mandatory and do not consume the optional budget.
        if candidate.tier == 0:
            selected.append(candidate.item)
            continue
        if used + cost <= token_budget:
            selected.append(candidate.item)
            used += cost
    manifest_id = make_context_manifest_id(
        codegen_run_id=codegen_run_id, scenario_id=scenario_id, sequence=sequence
    )
    draft = ContextManifest.model_construct(
        manifest_id=manifest_id,
        scenario_id=scenario_id,
        framework_snapshot_id=framework_snapshot_id,
        test_data_snapshot_id=test_data_snapshot_id,
        filesystem_snapshot_checksum=filesystem_snapshot_checksum or worktree_tree_hash or "",
        worktree_tree_hash=(worktree_tree_hash if filesystem_snapshot_checksum is None else None),
        items=selected,
        token_budget=token_budget,
        checksum="",
    )
    return ContextManifest.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


__all__ = ["RetrievedItem", "assemble_manifest", "estimate_tokens"]
