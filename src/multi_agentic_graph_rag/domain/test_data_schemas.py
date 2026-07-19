"""Typed contracts for the Stage-4A test-data ingestion pipeline (plan §8).

The XLSX workbook is treated as a versioned *test-data contract*, not as
retrieval text. A deterministic ingestion layer converts approved records into
exact, typed, referentially valid data; the coding model never reads raw rows.
These contracts model the common row envelope (§8.3), typed records, scenario
bindings (§8.4), validation issues (§8.8), and the published snapshot (§8.9).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from multi_agentic_graph_rag.domain.schemas import StrictModel, canonical_checksum

RecordStatus = Literal["DRAFT", "APPROVED", "DEPRECATED", "BLOCKED"]
ApprovalStatus = Literal["DRAFT", "APPROVED", "DEPRECATED"]
Sensitivity = Literal["public", "internal", "secret"]

# Tokens that must never be silently converted into model assumptions (§8.8).
PLACEHOLDER_TOKENS: frozenset[str] = frozenset({"PLACEHOLDER", "TBD", "UNKNOWN", "N/A", "TODO"})

# Stable issue severities for the validation taxonomy (§8.8).
IssueSeverity = Literal["ERROR", "WARNING"]

# Record types drawn from the core workbook sheets (§8.2). Kept open enough for
# domain-extension records while still being an explicit controlled vocabulary.
RecordType = Literal[
    "ExecutionProfile",
    "Resource",
    "Endpoint",
    "Identity",
    "InterfaceProfile",
    "DataField",
    "BitField",
    "Capability",
    "LifecycleOperation",
    "Fixture",
    "Action",
    "ActionStep",
    "Oracle",
    "TestVector",
    "TestVectorValue",
    "TimingPolicy",
    "FaultProfile",
    "Cleanup",
    "Decision",
    "SafetyRule",
    "Dependency",
]


class RecordEnvelope(StrictModel):
    """Common identity/governance envelope every executable sheet carries (§8.3)."""

    record_id: str
    record_status: RecordStatus
    enabled: bool = True
    applicable_profile_ids: list[str] = Field(default_factory=list)
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source_ref: str | None = None
    decision_ref: str | None = None
    owner: str
    valid_from_revision: str
    deprecated_by: str | None = None


class TestDataRecord(RecordEnvelope):
    """One normalized, typed test-data record with provenance (§8.9).

    IDs — never sheet names, row numbers, or free-text names — form all
    foreign-key relationships. ``source_sheet``/``source_row`` are provenance.
    """

    record_type: RecordType
    natural_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_checksum: str
    source_sheet: str
    source_row: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_payload_checksum(self) -> TestDataRecord:
        expected = canonical_checksum({"payload": self.payload})
        if self.payload_checksum != expected:
            raise ValueError("test-data record payload checksum mismatch")
        return self


class ScenarioDataBinding(StrictModel):
    """The bridge from a declarative Stage-3 scenario to executable data (§8.4).

    A binding references a scenario; it never duplicates or rewrites scenario
    intent (rule 4). Only ``APPROVED`` bindings may enter an executable bundle.
    """

    binding_id: str
    scenario_id: str
    execution_profile_id: str
    fixture_id: str
    action_sequence_id: str | None = None
    test_vector_ids: list[str] = Field(default_factory=list)
    oracle_ids: list[str] = Field(min_length=1)
    cleanup_id: str
    timing_policy_id: str | None = None
    fault_profile_ids: list[str] = Field(default_factory=list)
    safety_rule_ids: list[str] = Field(default_factory=list)
    applicability_condition: str | None = None
    parameterization_mode: Literal["single", "parameterized"] = "single"
    selection_priority: int = 0
    approval_status: ApprovalStatus


class ValidationIssue(StrictModel):
    """One ingestion problem with a stable code and exact workbook provenance (§8.8)."""

    issue_code: str
    severity: IssueSeverity
    message: str
    record_id: str | None = None
    sheet: str | None = None
    row: int | None = None
    cell: str | None = None


class MissingTestDataRequirement(StrictModel):
    """A scenario-specific data category the workbook does not yet supply (§8.4)."""

    scenario_id: str
    required_record_type: RecordType
    reason: str


class CoverageEntry(StrictModel):
    """Per-scenario binding coverage for the readiness gate (§8.7 step 11)."""

    scenario_id: str
    has_binding: bool
    has_oracle: bool
    has_cleanup: bool
    binding_count: int = Field(ge=0)


class TestDataValidationReport(StrictModel):
    """Errors, warnings, coverage, and provenance for the readiness gate (§8.1)."""

    project: str
    snapshot_id: str
    schema_version: str
    issues: list[ValidationIssue] = Field(default_factory=list)
    coverage: list[CoverageEntry] = Field(default_factory=list)
    missing_requirements: list[MissingTestDataRequirement] = Field(default_factory=list)
    status: Literal["READY", "BLOCKED"] = "BLOCKED"

    @property
    def has_errors(self) -> bool:
        """Whether any blocking error was recorded."""
        return any(issue.severity == "ERROR" for issue in self.issues)


class NormalizedTestData(StrictModel):
    """Canonical typed output published as an immutable test-data snapshot (§8.9)."""

    snapshot_id: str
    project: str
    schema_version: str
    workbook_checksum: str
    decision_revision: str
    records: list[TestDataRecord] = Field(default_factory=list)
    bindings: list[ScenarioDataBinding] = Field(default_factory=list)
    checksum: str

    @model_validator(mode="after")
    def validate_identity(self) -> NormalizedTestData:
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("duplicate test-data record_id within snapshot")
        binding_ids = [binding.binding_id for binding in self.bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("duplicate scenario-data binding_id within snapshot")
        if self.checksum != canonical_checksum(self):
            raise ValueError("normalized test-data checksum mismatch")
        return self

    def record_index(self) -> dict[str, TestDataRecord]:
        """Return an ID-keyed index for deterministic exact-value lookup (§8.10)."""
        return {record.record_id: record for record in self.records}


__all__ = [
    "PLACEHOLDER_TOKENS",
    "ApprovalStatus",
    "CoverageEntry",
    "IssueSeverity",
    "MissingTestDataRequirement",
    "NormalizedTestData",
    "RecordEnvelope",
    "RecordStatus",
    "RecordType",
    "ScenarioDataBinding",
    "Sensitivity",
    "TestDataRecord",
    "TestDataValidationReport",
    "ValidationIssue",
]
