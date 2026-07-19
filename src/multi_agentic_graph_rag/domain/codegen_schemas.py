"""Strict contracts for Stage 4 — enterprise test-code generation.

Kept separate from the large shared ``schemas.py`` (see the Stage-4 plan §17).
These are the stable foundation contracts every later Stage-4 phase depends on:
snapshot identities, the resolved test-data bundle the coding model consumes,
the context manifest that records what caused a patch, capability bindings,
blockers, and the persisted test-case master. Runtime stores (Neo4j code graph,
XLSX ingestion, worktree/patch tooling, LangGraph workflow) build on top of
these and are intentionally out of scope for this module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from multi_agentic_graph_rag.domain.schemas import (
    Priority,
    StrictModel,
    canonical_checksum,
    utc_now,
)

# --- Controlled vocabularies -------------------------------------------------

ReadinessStatus = Literal[
    "READY",
    "BLOCKED_MISSING_DATA",
    "BLOCKED_INVALID_TEST_DATA_DOCUMENT",
    "BLOCKED_AMBIGUOUS_BINDING",
    "BLOCKED_AMBIGUOUS_SPEC",
    "BLOCKED_MISSING_CAPABILITY",
    "BLOCKED_FRAMEWORK_STALE",
    "BLOCKED_DEPENDENCY",
    "BLOCKED_AUTHORIZATION",
    "NOT_APPLICABLE",
]

# Precise validation lifecycle (plan §19). A test is never reported "EXECUTABLE"
# unless it has actually run in its declared execution profile.
ValidationStatus = Literal[
    "GENERATED",
    "COLLECTABLE",
    "STATICALLY_VALIDATED",
    "EXECUTABLE",
    "REGRESSION_VALIDATED",
    "BLOCKED",
    "FAILED_VALIDATION",
]

# Deterministic capability decision per scenario action (plan §11).
CapabilityDecision = Literal[
    "reuse",
    "compose",
    "wrap",
    "implement",
    "modify_production",
    "block",
]

SnapshotStatus = Literal["building", "ready", "failed"]

# Provenance for a retrieved context item — the repository/worktree is
# authoritative for source bodies, never the graph or the model.
AuthoritativeSource = Literal["repository", "postgresql", "neo4j"]


# --- Snapshot identities (plan §6) -------------------------------------------


class FrameworkSnapshotRef(StrictModel):
    """Immutable, revision-pinned framework code snapshot identity.

    An identical snapshot can be shared across projects because its identity is
    derived from repository identity, Git tree/dirty hashes, and extractor
    version/config (see ``make_framework_snapshot_id``).
    """

    snapshot_id: str
    repository_id: str
    commit: str
    tree_hash: str
    dirty_hash: str
    extractor_version: str
    extractor_config_hash: str
    status: SnapshotStatus


class TestDataSnapshotRef(StrictModel):
    """Immutable, project-scoped test-data snapshot identity (plan §8.9)."""

    snapshot_id: str
    project: str
    schema_version: str
    workbook_checksum: str
    normalized_checksum: str
    decision_revision: str
    status: SnapshotStatus


# --- Resolved scenario test-data bundle (plan §8.5) --------------------------


class RecordRef(StrictModel):
    """A reference to one approved, typed test-data record by stable ID."""

    record_id: str


class ActionStepRef(StrictModel):
    """One ordered action instance within a scenario binding."""

    step: int = Field(ge=1)
    action_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class VectorRef(StrictModel):
    """A selected test vector with its resolved parameter values."""

    record_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class OracleRef(StrictModel):
    """An observable oracle with its resolved predicate."""

    record_id: str
    predicate: dict[str, Any] = Field(default_factory=dict)


class ResolvedScenarioTestDataBundle(StrictModel):
    """Immutable, checksummed executable bundle the coding model consumes.

    The model never reads the raw workbook. Secret references remain unresolved
    (``secret://…``) and are redacted before reaching the reasoning model.
    """

    bundle_id: str
    scenario_id: str
    binding_id: str
    snapshot_id: str
    execution_profile: RecordRef
    resources: list[RecordRef] = Field(default_factory=list)
    endpoints: list[RecordRef] = Field(default_factory=list)
    interface_profiles: list[RecordRef] = Field(default_factory=list)
    fixture: RecordRef
    action_steps: list[ActionStepRef] = Field(default_factory=list)
    vectors: list[VectorRef] = Field(default_factory=list)
    oracles: list[OracleRef] = Field(min_length=1)
    timing_policy: RecordRef | None = None
    cleanup: RecordRef
    safety_rules: list[str] = Field(default_factory=list)
    secret_refs: list[str] = Field(default_factory=list)
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)
    checksum: str

    @field_validator("secret_refs")
    @classmethod
    def secrets_are_references_only(cls, value: list[str]) -> list[str]:
        """Reject plaintext secrets; only ``secret://`` references are allowed."""
        for ref in value:
            if not ref.startswith("secret://"):
                raise ValueError("secret_refs must be 'secret://' references, never plaintext")
        return value

    @model_validator(mode="after")
    def validate_checksum(self) -> ResolvedScenarioTestDataBundle:
        if self.checksum != canonical_checksum(self):
            raise ValueError("resolved test-data bundle checksum mismatch")
        return self


# --- Context manifest (plan §10.4) -------------------------------------------


class ContextManifestItem(StrictModel):
    """One retrieved, hash-pinned context item supplied to a model invocation."""

    source_type: Literal[
        "code_symbol",
        "file_slice",
        "test_data",
        "scenario",
        "capability",
        "diagnostic",
    ]
    source_id: str
    revision: str
    content_hash: str
    retrieval_reason: str
    authoritative_source: AuthoritativeSource


class ContextManifest(StrictModel):
    """Reproducible record of exactly which code and data caused a patch."""

    manifest_id: str
    scenario_id: str
    framework_snapshot_id: str
    test_data_snapshot_id: str
    worktree_tree_hash: str
    items: list[ContextManifestItem] = Field(default_factory=list)
    token_budget: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)
    checksum: str

    @model_validator(mode="after")
    def validate_checksum(self) -> ContextManifest:
        if self.checksum != canonical_checksum(self):
            raise ValueError("context manifest checksum mismatch")
        return self


# --- Capability-gap analysis (plan §11) --------------------------------------


class CapabilityBinding(StrictModel):
    """Deterministic reuse/compose/wrap/implement decision for a scenario action."""

    binding_id: str
    scenario_id: str
    scenario_action: str
    capability_id: str
    candidate_symbols: list[str] = Field(default_factory=list)
    decision: CapabilityDecision
    selected_symbol: str | None = None
    adapter_to_create: str | None = None
    reason: str

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> CapabilityBinding:
        """Reuse/compose/wrap must select an existing symbol; implement must not."""
        if self.decision in ("reuse", "compose", "wrap") and not self.selected_symbol:
            raise ValueError(f"decision '{self.decision}' requires a selected_symbol")
        return self


# --- Blockers (plan §9) ------------------------------------------------------


class CodegenBlocker(StrictModel):
    """A structured blocker; a blocker is never silently turned into an assumption."""

    blocker_id: str
    scenario_id: str
    blocker_type: ReadinessStatus
    evidence: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("blocker_type")
    @classmethod
    def must_be_a_blocking_status(cls, value: str) -> str:
        if value in ("READY", "NOT_APPLICABLE"):
            raise ValueError("blocker_type must be a BLOCKED_* status")
        return value


# --- Test-case master (plan §16.1) -------------------------------------------


class TestCaseRecord(StrictModel):
    """Persisted Stage-4 test case with permanent identity and revision.

    Regenerating the same logical test reuses ``test_case_id`` and increments
    ``test_case_revision`` (plan §15.5).
    """

    test_case_id: str
    test_case_revision: int = Field(ge=1)
    scenario_id: str
    story_id: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)
    test_name: str
    test_file: str
    test_function: str
    execution_profile_id: str
    scenario_data_binding_id: str
    resolved_test_data_bundle_id: str
    resolved_test_data_bundle_checksum: str
    fixture_ids: list[str] = Field(default_factory=list)
    vector_ids: list[str] = Field(default_factory=list)
    oracle_ids: list[str] = Field(default_factory=list)
    called_capability_ids: list[str] = Field(default_factory=list)
    priority: Priority
    status: Literal["active", "superseded", "blocked"] = "active"
    validation_status: ValidationStatus
    generated_file_hashes: dict[str, str] = Field(default_factory=dict)
    context_manifest_id: str
    generation_model: str
    prompt_revision: str


class TestCasesArtifact(StrictModel):
    """Public Stage 4 artifact — the PostgreSQL fallback generates test-cases.json."""

    artifact_schema_version: Literal["1.0-test-cases"] = "1.0-test-cases"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    test_cases: list[TestCaseRecord]

    @model_validator(mode="after")
    def validate_checksum(self) -> TestCasesArtifact:
        if self.checksum != canonical_checksum(self):
            raise ValueError("test-cases artifact checksum mismatch")
        return self


__all__ = [
    "ActionStepRef",
    "AuthoritativeSource",
    "CapabilityBinding",
    "CapabilityDecision",
    "CodegenBlocker",
    "ContextManifest",
    "ContextManifestItem",
    "FrameworkSnapshotRef",
    "OracleRef",
    "ReadinessStatus",
    "RecordRef",
    "ResolvedScenarioTestDataBundle",
    "SnapshotStatus",
    "TestCaseRecord",
    "TestCasesArtifact",
    "TestDataSnapshotRef",
    "ValidationStatus",
    "VectorRef",
    "canonical_checksum",
    "utc_now",
]
