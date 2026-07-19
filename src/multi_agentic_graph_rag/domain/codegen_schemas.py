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

import re
from datetime import datetime
from pathlib import Path, PurePosixPath
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
    """Immutable filesystem-derived framework snapshot identity.

    Stage 4 deliberately has no Git provenance.  ``filesystem_checksum`` is a
    deterministic hash over normalized relative paths and file hashes, while
    ``canonical_path`` identifies the local framework root that was indexed.
    """

    snapshot_id: str
    repository_id: str
    canonical_path: str
    filesystem_checksum: str
    extractor_version: str
    extractor_config_hash: str
    status: SnapshotStatus
    active: bool = False


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
    resolved_records: dict[str, dict[str, Any]] = Field(default_factory=dict)
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
    filesystem_snapshot_checksum: str = ""
    # Temporary compatibility for the pre-optimized scaffold.  New Stage-4
    # code must populate ``filesystem_snapshot_checksum`` and leave this unset.
    worktree_tree_hash: str | None = None
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
    run_id: str | None = None
    tc_id: int | None = Field(default=None, ge=100001, le=999999)
    logical_key_hash: str | None = None
    error_code: str | None = None
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


# --- Stage 4 rework: six-digit identity, provider pinning, LLM bundle --------
#
# The types below are the P0 foundation of the optimized Stage-4 masterplan
# (permanent integer TC identity, provider isolation, frozen one-shot run,
# structured LLM I/O). They are additive: the legacy ``TestCaseRecord`` above
# stays until the per-scenario apply graph is rewired (masterplan P6), at which
# point the integer-identity ``Stage4TestCaseRecord`` supersedes it.

# Selected reasoning execution mode. There is no provider fallback (plan §9).
ReasoningProviderName = Literal["azure_openai", "huggingface"]

# TC lifecycle for the integer-identity master (plan §16.1).
TcStatus = Literal["RESERVED", "GENERATING", "VALIDATED", "ACCEPTED", "BLOCKED"]

# Whole-run outcome for the frozen one-shot semantics (plan §5).
RunStatus = Literal["FROZEN", "RUNNING", "COMPLETED", "PARTIAL_FAILED", "STOPPED"]

# The three module-specific writable roots plus the test suites (plan §6.1).
GeneratedFileRole = Literal["test", "robot", "wrapper", "helper", "init"]


class Stage4Request(StrictModel):
    """Complete request contract for one frozen Stage-4 generation run."""

    project_name: str
    run_id: str
    framework_path: Path
    test_data_document: Path
    execution_profile_id: str
    reasoning_provider: ReasoningProviderName = "azure_openai"
    variant_selection: dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False

    @field_validator("project_name", "run_id", "execution_profile_id")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Stage-4 request identity fields must not be empty")
        return normalized

    @field_validator("variant_selection")
    @classmethod
    def deterministic_variants(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for scenario_id, variant_id in value.items():
            scenario = scenario_id.strip()
            variant = variant_id.strip() or "default"
            if not scenario:
                raise ValueError("variant_selection scenario IDs must not be empty")
            normalized[scenario] = variant
        return normalized


class LogicalTestCaseKey(StrictModel):
    """Canonical logical identity: project + scenario + profile + variant (plan §8.1).

    ``variant_id`` is deterministic Stage-3 input (parameterization/test-data
    binding), never model prose, and defaults to ``"default"``. The stored hash
    backs a uniqueness constraint independent of the raw fields.
    """

    project_name: str
    scenario_id: str
    execution_profile_id: str
    variant_id: str = "default"
    logical_key_hash: str

    @model_validator(mode="after")
    def validate_hash(self) -> LogicalTestCaseKey:
        from multi_agentic_graph_rag.domain.identifiers import make_logical_key_hash

        expected = make_logical_key_hash(
            project_name=self.project_name,
            scenario_id=self.scenario_id,
            execution_profile_id=self.execution_profile_id,
            variant_id=self.variant_id,
        )
        if self.logical_key_hash != expected:
            raise ValueError("logical_key_hash does not match its canonical fields")
        return self


class ProviderFingerprint(StrictModel):
    """Pinned reasoning-provider identity; every resume must match it (plan §9.3).

    Changing provider, model/deployment, revision, generation parameters, or the
    prompt revision requires a brand-new Stage-4 run. A run never mixes providers.
    """

    provider: ReasoningProviderName
    model: str
    model_revision: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    prompt_revision: str
    fingerprint_hash: str

    @model_validator(mode="after")
    def validate_hash(self) -> ProviderFingerprint:
        from multi_agentic_graph_rag.domain.identifiers import (
            make_provider_fingerprint_hash,
        )

        expected = make_provider_fingerprint_hash(
            provider=self.provider,
            model=self.model,
            model_revision=self.model_revision,
            generation_params_checksum=canonical_checksum({"params": self.generation_params}),
            prompt_revision=self.prompt_revision,
        )
        if self.fingerprint_hash != expected:
            raise ValueError("provider fingerprint_hash does not match its fields")
        return self

    def matches(self, other: ProviderFingerprint) -> bool:
        """True only when the pinned identity is byte-for-byte the same run pin."""
        return self.fingerprint_hash == other.fingerprint_hash


class TcStep(StrictModel):
    """One decomposed, logged execution step of a generated test case (plan §11.2)."""

    step: int = Field(ge=1)
    description: str
    action_ref: str | None = None
    critical: bool = True


class FrozenScenarioEntry(StrictModel):
    """One immutable scenario entry inside the frozen run manifest (plan §5)."""

    scenario_id: str
    scenario_checksum: str
    story_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    variant_id: str = "default"


class FrozenRunManifest(StrictModel):
    """Immutable one-shot input manifest created by ``freeze_inputs`` (plan §5).

    The scenario list, provider pin, and baseline framework checksum never
    change during a run; a mismatch on resume raises ``INPUT_MANIFEST_CHANGED``.
    """

    project_name: str
    run_id: str
    scenarios: list[FrozenScenarioEntry] = Field(default_factory=list)
    execution_profile_id: str
    test_data_snapshot_checksum: str
    baseline_framework_checksum: str
    provider_fingerprint: ProviderFingerprint
    generation_policy_version: str
    frozen_at: datetime = Field(default_factory=utc_now)
    checksum: str

    @model_validator(mode="after")
    def validate_checksum(self) -> FrozenRunManifest:
        if self.checksum != canonical_checksum(self):
            raise ValueError("frozen run manifest checksum mismatch")
        return self


# --- Structured LLM plan + bundle (plan §10.2) -------------------------------


class PlannedStep(StrictModel):
    """One planned execution step the model proposes for the test body."""

    step: int = Field(ge=1)
    description: str
    capability_id: str | None = None
    critical: bool = True


class PlannedFile(StrictModel):
    """One file the model intends to write, with its deterministic role."""

    role: GeneratedFileRole
    relative_path: str
    reuse_existing: bool = False


class TestCaseImplementationPlan(StrictModel):
    """Structured output of the single planning call (plan §10.2)."""

    scenario_id: str
    module: str
    test_title: str
    steps: list[PlannedStep] = Field(min_length=1)
    files: list[PlannedFile] = Field(min_length=1)
    reused_symbols: list[str] = Field(default_factory=list)
    test_variables: dict[str, Any] = Field(default_factory=dict)


class GeneratedFileContent(StrictModel):
    """Complete generated content for one permitted file."""

    role: GeneratedFileRole
    relative_path: str
    content: str
    content_hash: str

    @model_validator(mode="after")
    def validate_hash(self) -> GeneratedFileContent:
        import hashlib

        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_hash != f"sha256:{digest}":
            raise ValueError("generated file content_hash mismatch")
        return self


class GeneratedPatchBundle(StrictModel):
    """Structured multi-file output of the single bundle call (plan §10.2)."""

    scenario_id: str
    tc_id: int = Field(ge=100001, le=999999)
    plan_checksum: str
    files: list[GeneratedFileContent] = Field(min_length=1)


class Stage4TestCaseRecord(StrictModel):
    """Persisted Stage-4 test case with permanent six-digit integer identity.

    Supersedes the legacy revision-keyed ``TestCaseRecord`` (plan §16.1). No
    revision authoring exists in this phase: a changed logical key returns
    ``REVISION_REQUIRED`` rather than overwriting an accepted case (plan §8.4).
    """

    tc_id: int = Field(ge=100001, le=999999)
    project_name: str
    scenario_id: str
    execution_profile_id: str
    variant_id: str = "default"
    logical_key_hash: str
    status: TcStatus = "RESERVED"
    story_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    tc_steps: list[TcStep] = Field(default_factory=list)
    module: str = ""
    test_file: str = ""
    robot_file: str = ""
    helper_files: list[str] = Field(default_factory=list)
    generated_file_hashes: dict[str, str] = Field(default_factory=dict)
    framework_snapshot_id: str | None = None
    test_data_snapshot_id: str = ""
    provider_fingerprint: ProviderFingerprint | None = None
    prompt_revision: str = ""
    input_fingerprint: str = ""
    context_manifest: dict[str, Any] = Field(default_factory=dict)
    validation_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_logical_identity(self) -> Stage4TestCaseRecord:
        from multi_agentic_graph_rag.domain.identifiers import make_logical_key_hash

        expected = make_logical_key_hash(
            project_name=self.project_name,
            scenario_id=self.scenario_id,
            execution_profile_id=self.execution_profile_id,
            variant_id=self.variant_id,
        )
        if self.logical_key_hash != expected:
            raise ValueError("logical_key_hash does not match test-case canonical fields")
        if self.tc_steps:
            actual = [item.step for item in self.tc_steps]
            expected_steps = list(range(1, len(self.tc_steps) + 1))
            if actual != expected_steps:
                raise ValueError("tc_steps must be ordered and contiguous from 1")
        if self.status == "ACCEPTED":
            self._validate_accepted_completeness()
        return self

    def _validate_accepted_completeness(self) -> None:
        """Reject partially populated records at the permanent acceptance boundary."""
        required_strings = {
            "module": self.module,
            "test_file": self.test_file,
            "robot_file": self.robot_file,
            "framework_snapshot_id": self.framework_snapshot_id or "",
            "test_data_snapshot_id": self.test_data_snapshot_id,
            "prompt_revision": self.prompt_revision,
            "input_fingerprint": self.input_fingerprint,
        }
        missing = sorted(name for name, value in required_strings.items() if not value.strip())
        if self.provider_fingerprint is None:
            missing.append("provider_fingerprint")
        if not self.tc_steps:
            missing.append("tc_steps")
        if not self.generated_file_hashes:
            missing.append("generated_file_hashes")
        if missing:
            raise ValueError("ACCEPTED test case is incomplete: " + ", ".join(missing))

        module_pattern = re.compile(r"[a-z][a-z0-9_]*")
        if module_pattern.fullmatch(self.module) is None:
            raise ValueError("module must be a lowercase module identifier")
        expected_prefix = f"Tc{self.tc_id}"
        test_path = PurePosixPath(self.test_file)
        robot_path = PurePosixPath(self.robot_file)
        if (
            len(test_path.parts) != 3
            or test_path.parts[:2] != ("tests", self.module)
            or test_path.suffix != ".py"
            or not test_path.stem.startswith(expected_prefix)
            or test_path.stem == expected_prefix
        ):
            raise ValueError("test_file does not match tests/<module>/Tc<id><Title>.py")
        if (
            len(robot_path.parts) != 3
            or robot_path.parts[:2] != ("tests_robot", self.module)
            or robot_path.suffix != ".robot"
            or robot_path.stem != test_path.stem
        ):
            raise ValueError("robot_file must use the exact generated Python stem")

        required_hashes = {self.test_file, self.robot_file, *self.helper_files}
        absent_hashes = sorted(required_hashes.difference(self.generated_file_hashes))
        if absent_hashes:
            raise ValueError("accepted generated files lack hashes: " + ", ".join(absent_hashes))
        sha256_pattern = re.compile(r"sha256:[0-9a-f]{64}")
        invalid_hashes = sorted(
            path
            for path, digest in self.generated_file_hashes.items()
            if sha256_pattern.fullmatch(digest) is None
        )
        if invalid_hashes:
            raise ValueError("generated_file_hashes must contain canonical SHA-256 values")

        for helper_file in self.helper_files:
            helper_path = PurePosixPath(helper_file)
            allowed_names = {
                "__init__.py",
                f"{self.module}_wrappers.py",
                f"{self.module}_helpers.py",
            }
            if (
                len(helper_path.parts) != 3
                or helper_path.parts[:2] != ("test_lib", self.module)
                or helper_path.name not in allowed_names
            ):
                raise ValueError("helper_files must be module-specific test_lib files")


class TestCasesArtifact(StrictModel):
    """Run-scoped review artifact mirrored from authoritative PostgreSQL state."""

    artifact_schema_version: Literal["1.0-test-cases"] = "1.0-test-cases"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    test_cases: list[Stage4TestCaseRecord] = Field(default_factory=list)
    blockers: list[CodegenBlocker] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_checksum(self) -> TestCasesArtifact:
        invalid = [record.tc_id for record in self.test_cases if record.status != "ACCEPTED"]
        if invalid:
            raise ValueError(f"test-cases artifact may contain only ACCEPTED records: {invalid}")
        wrong_project = [
            record.tc_id for record in self.test_cases if record.project_name != self.project
        ]
        if wrong_project:
            raise ValueError(f"test-cases artifact contains another project: {wrong_project}")
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
    "FrozenRunManifest",
    "FrozenScenarioEntry",
    "GeneratedFileContent",
    "GeneratedFileRole",
    "GeneratedPatchBundle",
    "LogicalTestCaseKey",
    "OracleRef",
    "PlannedFile",
    "PlannedStep",
    "ProviderFingerprint",
    "ReadinessStatus",
    "ReasoningProviderName",
    "RecordRef",
    "ResolvedScenarioTestDataBundle",
    "RunStatus",
    "SnapshotStatus",
    "Stage4Request",
    "Stage4TestCaseRecord",
    "TcStatus",
    "TcStep",
    "TestCaseImplementationPlan",
    "TestCaseRecord",
    "TestCasesArtifact",
    "TestDataSnapshotRef",
    "ValidationStatus",
    "VectorRef",
    "canonical_checksum",
    "utc_now",
]
