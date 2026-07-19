"""Tests for capability decisions, context budgeting, and the sufficiency gate."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.codegen_schemas import ContextManifestItem, canonical_checksum
from multi_agentic_graph_rag.services.capability_mapper import (
    CapabilityCandidate,
    decide_capability,
)
from multi_agentic_graph_rag.services.context_assembler import RetrievedItem, assemble_manifest
from multi_agentic_graph_rag.services.context_sufficiency import (
    SufficiencyInputs,
    assess_sufficiency,
)


def _candidate(fqn: str, match: str) -> CapabilityCandidate:
    return CapabilityCandidate(symbol_id=f"SYM-{fqn}", fqn=fqn, match_kind=match)


# --- Capability decisions ----------------------------------------------------


def test_exact_match_is_reused() -> None:
    result = decide_capability(
        scenario_id="TS-1",
        scenario_action="Start a simulated slave",
        capability_id="simulator.lifecycle.start",
        candidates=[_candidate("fw.start", "exact"), _candidate("fw.other", "compatible")],
        modification_authorized=False,
    )
    assert result.is_ready
    assert result.binding is not None
    assert result.binding.decision == "reuse"
    assert result.binding.selected_symbol == "fw.start"


def test_compatible_only_is_wrapped_with_adapter() -> None:
    result = decide_capability(
        scenario_id="TS-1",
        scenario_action="Start a simulated slave",
        capability_id="cap.start",
        candidates=[_candidate("fw.async_start", "compatible")],
        modification_authorized=False,
        adapter_path="tests/support/sim.py::start",
    )
    assert result.binding is not None
    assert result.binding.decision == "wrap"
    assert result.binding.adapter_to_create == "tests/support/sim.py::start"


def test_multiple_partials_compose() -> None:
    result = decide_capability(
        scenario_id="TS-1",
        scenario_action="Bring device to ready state",
        capability_id="cap.ready",
        candidates=[_candidate("fw.a", "partial"), _candidate("fw.b", "partial")],
        modification_authorized=False,
    )
    assert result.binding is not None
    assert result.binding.decision == "compose"


def test_missing_capability_blocks_when_unauthorized() -> None:
    result = decide_capability(
        scenario_id="TS-1",
        scenario_action="Do something novel",
        capability_id="cap.novel",
        candidates=[],
        modification_authorized=False,
    )
    assert not result.is_ready
    assert result.blocker == "BLOCKED_MISSING_CAPABILITY"


def test_missing_capability_is_implemented_when_authorized() -> None:
    result = decide_capability(
        scenario_id="TS-1",
        scenario_action="Do something novel",
        capability_id="cap.novel",
        candidates=[],
        modification_authorized=True,
        adapter_path="tests/support/helper.py::do",
    )
    assert result.binding is not None
    assert result.binding.decision == "implement"
    assert result.binding.selected_symbol is None


# --- Context budgeting -------------------------------------------------------


def _item(source_id: str) -> ContextManifestItem:
    return ContextManifestItem(
        source_type="code_symbol",
        source_id=source_id,
        revision="FWS-1",
        content_hash="sha256:aa",
        retrieval_reason="candidate",
        authoritative_source="repository",
    )


def test_manifest_respects_budget_but_keeps_tier0() -> None:
    candidates = [
        RetrievedItem(item=_item("contract"), estimated_chars=100_000, tier=0),
        RetrievedItem(item=_item("cheap"), estimated_chars=40, tier=2),
        RetrievedItem(item=_item("expensive"), estimated_chars=100_000, tier=2),
    ]
    manifest = assemble_manifest(
        codegen_run_id="CGR-1",
        scenario_id="TS-1",
        sequence=1,
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        worktree_tree_hash="tree",
        candidates=candidates,
        token_budget=100,
    )
    ids = {item.source_id for item in manifest.items}
    assert "contract" in ids  # tier-0 always kept
    assert "cheap" in ids
    assert "expensive" not in ids  # dropped by budget
    assert manifest.checksum == canonical_checksum(manifest)


# --- Sufficiency gate --------------------------------------------------------


def test_sufficiency_flags_missing_data() -> None:
    result = assess_sufficiency(
        SufficiencyInputs(bundle_resolved=False, all_actions_have_capability=True)
    )
    assert result.status == "SPEC_DATA_MISSING"


def test_sufficiency_flags_stale_worktree() -> None:
    result = assess_sufficiency(
        SufficiencyInputs(
            bundle_resolved=True,
            all_actions_have_capability=True,
            worktree_matches_index=False,
        )
    )
    assert result.status == "SPEC_DATA_MISSING"


def test_sufficiency_flags_capability_gap() -> None:
    result = assess_sufficiency(
        SufficiencyInputs(
            bundle_resolved=True,
            all_actions_have_capability=False,
            unresolved_capability_ids=("cap.x",),
        )
    )
    assert result.status == "CAPABILITY_MISSING"
    assert result.reasons


def test_sufficiency_passes_when_everything_resolved() -> None:
    result = assess_sufficiency(
        SufficiencyInputs(bundle_resolved=True, all_actions_have_capability=True)
    )
    assert result.is_sufficient
