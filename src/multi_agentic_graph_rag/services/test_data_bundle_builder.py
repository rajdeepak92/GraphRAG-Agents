"""Build the immutable ResolvedScenarioTestDataBundle the model consumes (§8.5).

The coding model consumes a generated, checksummed bundle of resolved typed
values — never the raw workbook. Exact values are retrieved by stable ID
(§8.10), only ``APPROVED`` records enter the bundle, and secret material stays a
``secret://`` reference resolved only at controlled runtime.
"""

from __future__ import annotations

from typing import Any

from multi_agentic_graph_rag.domain.codegen_schemas import (
    ActionStepRef,
    OracleRef,
    RecordRef,
    ResolvedScenarioTestDataBundle,
    VectorRef,
)
from multi_agentic_graph_rag.domain.identifiers import make_resolved_bundle_id
from multi_agentic_graph_rag.domain.schemas import canonical_checksum
from multi_agentic_graph_rag.domain.test_data_schemas import (
    NormalizedTestData,
    ScenarioDataBinding,
    TestDataRecord,
)


class BundleBuildError(ValueError):
    """Raised when a binding cannot be resolved into a complete bundle."""


def _resolve(
    record_id: str,
    index: dict[str, TestDataRecord],
    *,
    binding_id: str,
) -> TestDataRecord:
    record = index.get(record_id)
    if record is None:
        raise BundleBuildError(f"binding '{binding_id}' references unknown record '{record_id}'")
    if record.record_status != "APPROVED":
        raise BundleBuildError(
            f"binding '{binding_id}' references non-APPROVED record '{record_id}'"
        )
    return record


def _secret_refs(records: list[TestDataRecord]) -> list[str]:
    refs: list[str] = []
    for record in records:
        value = record.payload.get("secret_ref")
        if isinstance(value, str) and value.startswith("secret://") and value not in refs:
            refs.append(value)
    return refs


def _action_steps(
    binding: ScenarioDataBinding, index: dict[str, TestDataRecord]
) -> list[ActionStepRef]:
    if binding.action_sequence_id is None:
        return []
    steps: list[tuple[int, ActionStepRef]] = []
    for record in index.values():
        if record.record_type != "ActionStep":
            continue
        if record.payload.get("sequence_id") != binding.action_sequence_id:
            continue
        step_no = int(record.payload.get("step", 0))
        action_id = record.payload.get("action_id")
        if not action_id:
            raise BundleBuildError(f"action step '{record.record_id}' has no action_id")
        arguments = record.payload.get("arguments", {})
        steps.append(
            (step_no, ActionStepRef(step=step_no, action_id=str(action_id), arguments=arguments))
        )
    return [ref for _, ref in sorted(steps, key=lambda item: item[0])]


def build_bundle(
    *,
    binding: ScenarioDataBinding,
    normalized: NormalizedTestData,
) -> ResolvedScenarioTestDataBundle:
    """Resolve one approved binding into a checksummed executable bundle."""
    if binding.approval_status != "APPROVED":
        raise BundleBuildError(f"binding '{binding.binding_id}' is not APPROVED")
    index = normalized.record_index()

    profile = _resolve(binding.execution_profile_id, index, binding_id=binding.binding_id)
    fixture = _resolve(binding.fixture_id, index, binding_id=binding.binding_id)
    cleanup = _resolve(binding.cleanup_id, index, binding_id=binding.binding_id)
    oracle_records = [
        _resolve(oracle_id, index, binding_id=binding.binding_id)
        for oracle_id in binding.oracle_ids
    ]
    vector_records = [
        _resolve(vector_id, index, binding_id=binding.binding_id)
        for vector_id in binding.test_vector_ids
    ]
    timing = (
        _resolve(binding.timing_policy_id, index, binding_id=binding.binding_id)
        if binding.timing_policy_id
        else None
    )

    oracles = [
        OracleRef(record_id=record.record_id, predicate=_as_dict(record.payload.get("predicate")))
        for record in oracle_records
    ]
    vectors = [
        VectorRef(record_id=record.record_id, parameters=_as_dict(record.payload.get("parameters")))
        for record in vector_records
    ]
    action_steps = _action_steps(binding, index)

    resolved_records = [profile, fixture, cleanup, *oracle_records, *vector_records]
    if timing is not None:
        resolved_records.append(timing)
    provenance = [
        {
            "record_id": record.record_id,
            "record_type": record.record_type,
            "source_sheet": record.source_sheet,
            "source_row": record.source_row,
            "record_checksum": record.payload_checksum,
        }
        for record in resolved_records
    ]

    parameterization_key = "|".join(sorted(binding.test_vector_ids))
    bundle_id = make_resolved_bundle_id(
        binding_id=binding.binding_id,
        test_data_snapshot_id=normalized.snapshot_id,
        parameterization_key=parameterization_key,
    )
    draft = ResolvedScenarioTestDataBundle.model_construct(
        bundle_id=bundle_id,
        scenario_id=binding.scenario_id,
        binding_id=binding.binding_id,
        snapshot_id=normalized.snapshot_id,
        execution_profile=RecordRef(record_id=profile.record_id),
        resources=[],
        endpoints=[],
        interface_profiles=[],
        fixture=RecordRef(record_id=fixture.record_id),
        action_steps=action_steps,
        vectors=vectors,
        oracles=oracles,
        timing_policy=RecordRef(record_id=timing.record_id) if timing else None,
        cleanup=RecordRef(record_id=cleanup.record_id),
        safety_rules=list(binding.safety_rule_ids),
        secret_refs=_secret_refs(resolved_records),
        source_provenance=provenance,
        checksum="",
    )
    return ResolvedScenarioTestDataBundle.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["BundleBuildError", "build_bundle"]
