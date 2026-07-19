"""Deterministic validation gates for ingested test data (plan §8.8).

Reports errors by stable issue code with exact provenance. Placeholder tokens
(``PLACEHOLDER``/``TBD``/``UNKNOWN``) and blank mandatory executable values are
never converted into model assumptions. Coverage and reference integrity gates
decide whether a scenario can proceed to binding.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from multi_agentic_graph_rag.domain.test_data_schemas import (
    PLACEHOLDER_TOKENS,
    CoverageEntry,
    RecordType,
    ScenarioDataBinding,
    TestDataRecord,
    TestDataValidationReport,
    ValidationIssue,
)


def _scan_placeholders(payload: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value.strip().upper() in PLACEHOLDER_TOKENS or value.strip() == "":
                found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _scan_invalid_secrets(payload: dict[str, Any]) -> list[str]:
    invalid: list[str] = []
    secret_keys = ("secret", "password", "credential", "api_key", "token")

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key).casefold())
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif (
            any(part in key for part in secret_keys)
            and value not in (None, "")
            and (not isinstance(value, str) or not value.startswith("secret://"))
        ):
            invalid.append(key)

    walk(payload)
    return invalid


def validate_test_data(
    *,
    project: str,
    snapshot_id: str,
    schema_version: str,
    records: list[TestDataRecord],
    bindings: list[ScenarioDataBinding],
    scenario_ids: set[str],
    structural_issues: list[ValidationIssue] | None = None,
) -> TestDataValidationReport:
    """Run identity, placeholder, reference, and coverage gates (§8.8)."""
    issues: list[ValidationIssue] = list(structural_issues or [])
    by_id: dict[str, TestDataRecord] = {}
    by_type: dict[str, set[str]] = {}

    # Identity gate: duplicate record IDs.
    for record_id, count in Counter(r.record_id for r in records).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    issue_code="IDENTITY_DUPLICATE_RECORD_ID",
                    severity="ERROR",
                    message=f"record_id '{record_id}' appears {count} times",
                    record_id=record_id,
                )
            )
    for record in records:
        by_id.setdefault(record.record_id, record)
        by_type.setdefault(record.record_type, set()).add(record.record_id)
        # Placeholder / blank executable value gate.
        for value in _scan_placeholders(record.payload):
            token = value.strip() or "<blank>"
            issues.append(
                ValidationIssue(
                    issue_code="SEMANTIC_PLACEHOLDER_VALUE",
                    severity="ERROR",
                    message=f"record '{record.record_id}' contains placeholder value '{token}'",
                    record_id=record.record_id,
                    sheet=record.source_sheet,
                    row=record.source_row,
                )
            )
        for key in _scan_invalid_secrets(record.payload):
            issues.append(
                ValidationIssue(
                    issue_code="SEMANTIC_PLAINTEXT_SECRET",
                    severity="ERROR",
                    message=(
                        f"record '{record.record_id}' field '{key}' must contain a "
                        "secret:// reference"
                    ),
                    record_id=record.record_id,
                    sheet=record.source_sheet,
                    row=record.source_row,
                )
            )

    _validate_binding_references(bindings, by_id, issues)
    coverage = _validate_coverage(bindings, scenario_ids, by_id, issues)

    status: str = "BLOCKED" if any(i.severity == "ERROR" for i in issues) else "READY"
    return TestDataValidationReport(
        project=project,
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        issues=issues,
        coverage=coverage,
        status=status,  # type: ignore[arg-type]
    )


def _check_ref(
    *,
    binding: ScenarioDataBinding,
    record_id: str,
    expected_type: RecordType,
    by_id: dict[str, TestDataRecord],
    issues: list[ValidationIssue],
) -> None:
    record = by_id.get(record_id)
    if record is None:
        issues.append(
            ValidationIssue(
                issue_code="REFERENCE_MISSING_FK",
                severity="ERROR",
                message=f"binding '{binding.binding_id}' references unknown record '{record_id}'",
                record_id=binding.binding_id,
            )
        )
        return
    if record.record_type != expected_type:
        issues.append(
            ValidationIssue(
                issue_code="REFERENCE_WRONG_TYPE",
                severity="ERROR",
                message=(
                    f"binding '{binding.binding_id}' expects {expected_type} for "
                    f"'{record_id}' but found {record.record_type}"
                ),
                record_id=binding.binding_id,
            )
        )
    elif record.record_status != "APPROVED":
        issues.append(
            ValidationIssue(
                issue_code="SEMANTIC_UNAPPROVED_RECORD",
                severity="ERROR",
                message=f"binding '{binding.binding_id}' references non-APPROVED '{record_id}'",
                record_id=binding.binding_id,
            )
        )


def _validate_binding_references(
    bindings: list[ScenarioDataBinding],
    by_id: dict[str, TestDataRecord],
    issues: list[ValidationIssue],
) -> None:
    for binding in bindings:
        _check_ref(
            binding=binding,
            record_id=binding.execution_profile_id,
            expected_type="ExecutionProfile",
            by_id=by_id,
            issues=issues,
        )
        _check_ref(
            binding=binding,
            record_id=binding.fixture_id,
            expected_type="Fixture",
            by_id=by_id,
            issues=issues,
        )
        _check_ref(
            binding=binding,
            record_id=binding.cleanup_id,
            expected_type="Cleanup",
            by_id=by_id,
            issues=issues,
        )
        for oracle_id in binding.oracle_ids:
            _check_ref(
                binding=binding,
                record_id=oracle_id,
                expected_type="Oracle",
                by_id=by_id,
                issues=issues,
            )
        for vector_id in binding.test_vector_ids:
            _check_ref(
                binding=binding,
                record_id=vector_id,
                expected_type="TestVector",
                by_id=by_id,
                issues=issues,
            )
        if binding.action_sequence_id:
            _check_ref(
                binding=binding,
                record_id=binding.action_sequence_id,
                expected_type="Action",
                by_id=by_id,
                issues=issues,
            )
        if binding.timing_policy_id:
            _check_ref(
                binding=binding,
                record_id=binding.timing_policy_id,
                expected_type="TimingPolicy",
                by_id=by_id,
                issues=issues,
            )
        for fault_id in binding.fault_profile_ids:
            _check_ref(
                binding=binding,
                record_id=fault_id,
                expected_type="FaultProfile",
                by_id=by_id,
                issues=issues,
            )
        for safety_id in binding.safety_rule_ids:
            _check_ref(
                binding=binding,
                record_id=safety_id,
                expected_type="SafetyRule",
                by_id=by_id,
                issues=issues,
            )


def _validate_coverage(
    bindings: list[ScenarioDataBinding],
    scenario_ids: set[str],
    by_id: dict[str, TestDataRecord],
    issues: list[ValidationIssue],
) -> list[CoverageEntry]:
    approved_by_scenario: dict[str, list[ScenarioDataBinding]] = {}
    for binding in bindings:
        if binding.approval_status == "APPROVED":
            approved_by_scenario.setdefault(binding.scenario_id, []).append(binding)

    coverage: list[CoverageEntry] = []
    for scenario_id in sorted(scenario_ids):
        applicable = approved_by_scenario.get(scenario_id, [])
        has_oracle = any(b.oracle_ids for b in applicable)
        has_cleanup = any(b.cleanup_id in by_id for b in applicable)
        if not applicable:
            issues.append(
                ValidationIssue(
                    issue_code="COVERAGE_NO_BINDING",
                    severity="ERROR",
                    message=f"scenario '{scenario_id}' has no approved binding",
                    record_id=scenario_id,
                )
            )
        coverage.append(
            CoverageEntry(
                scenario_id=scenario_id,
                has_binding=bool(applicable),
                has_oracle=has_oracle,
                has_cleanup=has_cleanup,
                binding_count=len(applicable),
            )
        )
    return coverage


__all__ = ["validate_test_data"]
