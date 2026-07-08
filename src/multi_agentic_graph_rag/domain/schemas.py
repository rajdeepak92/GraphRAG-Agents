"""Strict Pydantic contracts used by ingestion and generated artifacts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    __test__: ClassVar[bool] = False

    model_config = ConfigDict(extra="forbid")


_SOURCE_REQUIREMENT_ID_RE = re.compile(
    r"\b(?:(?:BR|AC|FR|NFR)\s*[-_]\s*[A-Z0-9]+(?:\s*[-_]\s*[A-Z0-9]+)*|"
    r"SYS\s*[-_ ]\s*REQ\s*[-_ ]\s*[A-Z0-9]+(?:\s*[-_]\s*[A-Z0-9]+)*)\b",
    re.I,
)
_PLACEHOLDER_REQUIREMENT_TEXT_RE = re.compile(
    r"^(?:requirement|business requirement|acceptance criteria|functional "
    r"requirement|non-functional requirement|placeholder|tbd|n/?a|none)$",
    re.I,
)
_PLACEHOLDER_USER_STORY_TEXT_RE = re.compile(
    r"^(?:user story|story|title|persona|role|user|actor|business value|value|"
    r"epic|feature|tbd|todo|placeholder|n/?a|na|none|null|unknown)$",
    re.I,
)


def normalize_priority_label(value: object) -> Literal["High", "Medium", "Low"]:
    """Normalize an arbitrary priority string to High/Medium/Low."""
    if value is None:
        return "Medium"
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return "Medium"
    normalized = text.lower()
    if normalized in {"high", "critical", "mandatory", "must", "required"}:
        return "High"
    if normalized in {"low", "optional", "future", "nice-to-have", "nice to have"}:
        return "Low"
    return "Medium"


def normalize_source_req_id(value: object) -> str | None:
    """Normalize an optional source requirement identifier."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return None
    match = _SOURCE_REQUIREMENT_ID_RE.search(text)
    if match is None:
        return None
    parts = [part for part in re.split(r"[\s_-]+", match.group(0).upper()) if part]
    if len(parts) >= 3 and parts[0] == "SYS" and parts[1] == "REQ":
        return "SYS_REQ_" + "_".join(parts[2:])
    return "-".join(parts)


class IngestionRequest(StrictModel):
    project: str
    document: Path
    version: str
    logical_name: str | None = None
    replace_version: bool = False
    reasoning_provider: str | None = None
    embedding_provider: str | None = None

    @field_validator("project", "version")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class SourceTrace(StrictModel):
    chunk_id: str
    quote: str
    start_char: int
    end_char: int
    page: int | None = None
    section: str | None = None

    @field_validator("quote")
    @classmethod
    def quote_is_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("quote must not be empty")
        return value

    @model_validator(mode="after")
    def validate_span(self) -> SourceTrace:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class ParsedBlock(StrictModel):
    block_id: str
    original_text: str
    normalized_text: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(StrictModel):
    chunk_id: str
    ordinal: int
    text: str
    normalized_text: str
    page: int | None = None
    section: str | None = None
    start_char: int
    end_char: int
    source_block_ids: list[str]


class DocumentManifest(StrictModel):
    project: str
    document_id: str
    document_version_id: str
    logical_name: str
    version: str
    source_path: str
    source_checksum: str
    parser_fingerprint: str
    chunker_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunks: list[DocumentChunk]


class LLMRequirementCandidate(StrictModel):
    temp_id: str
    statement: str
    requirement_type: str = "functional"
    priority: str = "medium"
    requirement_key: str | None = None
    source_req_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_trace: SourceTrace


class LLMFactCandidate(StrictModel):
    temp_id: str
    text: str
    source_trace: SourceTrace
    requirements: list[LLMRequirementCandidate] = Field(default_factory=list)


class LLMDiscoveredRequirement(StrictModel):
    req_text: str
    requirement_type: str = "Functional Requirement"
    priority: str = "Medium"
    requirement_key: str | None = None
    source_req_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("source_req_id", mode="before")
    @classmethod
    def normalize_source_identifier(cls, value: object) -> str | None:
        return normalize_source_req_id(value)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> str:
        if value is None:
            return "Medium"

        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return "Medium"

        normalized = text.lower()
        if normalized in {"high", "critical", "mandatory", "must", "required"}:
            return "High"
        if normalized in {"medium", "med", "normal", "default"}:
            return "Medium"
        if normalized in {"low", "optional", "future", "nice-to-have", "nice to have"}:
            return "Low"

        return "Medium"

    @field_validator("requirement_type", mode="before")
    @classmethod
    def normalize_requirement_type(cls, value: object) -> str:
        if value is None:
            return "Functional Requirement"

        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return "Functional Requirement"

        normalized = text.lower()
        if normalized in {"functional", "fr", "functional requirement"}:
            return "Functional Requirement"
        if normalized in {"business", "br", "business requirement"}:
            return "Business Requirement"
        if normalized in {"acceptance", "ac", "acceptance criteria", "acceptance criterion"}:
            return "Acceptance Criteria"
        if normalized in {"non-functional", "non functional", "nfr", "non-functional requirement"}:
            return "Non-Functional Requirement"
        if normalized in {"security", "security requirement"}:
            return "Security Requirement"
        if normalized in {"configuration", "configuration requirement"}:
            return "Configuration Requirement"
        if normalized in {"validation", "validation requirement"}:
            return "Validation Requirement"
        if normalized in {"alerting", "alerting requirement"}:
            return "Alerting Requirement"
        if normalized in {"health", "health requirement"}:
            return "Health Requirement"
        if normalized in {"data quality", "data quality requirement"}:
            return "Data Quality Requirement"
        if normalized in {"application", "application requirement"}:
            return "Application Requirement"
        if normalized in {"offline", "offline requirement"}:
            return "Offline Requirement"

        return text

    @field_validator("req_text")
    @classmethod
    def meaningful_requirement_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("req_text must not be empty")
        if _PLACEHOLDER_REQUIREMENT_TEXT_RE.match(value):
            raise ValueError(f"req_text must not be a placeholder: {value!r}")
        if _SOURCE_REQUIREMENT_ID_RE.search(value):
            raise ValueError(f"req_text must not contain source identifiers: {value!r}")
        return value


class LLMDiscoveredFact(StrictModel):
    fact_text: str
    quote: str
    requirements: list[LLMDiscoveredRequirement] = Field(default_factory=list)

    @field_validator("fact_text", "quote")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class RequirementDiscoveryChunkOutput(StrictModel):
    facts: list[LLMDiscoveredFact] = Field(default_factory=list)


class LLMChunkExtraction(StrictModel):
    chunk_id: str
    facts: list[LLMFactCandidate] = Field(default_factory=list)


class RequirementDiscoveryOutput(StrictModel):
    chunks: list[LLMChunkExtraction] = Field(default_factory=list)


class CanonicalFact(StrictModel):
    canonical_fact_id: str
    normalized_text: str
    representative_text: str


class VerifiedFact(StrictModel):
    fact_id: str
    canonical_fact_id: str = ""
    text: str
    source_trace: SourceTrace


class RequirementEvidence(StrictModel):
    evidence_id: str
    fact_ids: list[str]
    source_trace: SourceTrace


class VerifiedRequirement(StrictModel):
    requirement_id: str
    revision_id: str = ""
    display_id: str = ""
    requirement_key: str = ""
    source_req_id: str | None = None
    id_generation_type: Literal["source", "generated"] = "generated"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    statement: str
    normalized_statement: str = ""
    requirement_type: str
    priority: str
    status: Literal["active", "superseded"] = "active"
    fact_ids: list[str]
    source_trace: SourceTrace
    evidence: list[RequirementEvidence] = Field(default_factory=list)


class RequirementDeltaEvent(StrictModel):
    event_id: str
    event_type: Literal[
        "new",
        "duplicate",
        "changed",
        "superseded",
        "updated",
        "strictly_outdated",
        "unchanged",
    ]
    requirement_id: str
    revision_id: str | None = None
    previous_revision_id: str | None = None
    superseded_by_revision_id: str | None = None
    document_version_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    impacted_artifact_types: list[str] = Field(default_factory=list)


class RequirementRevisionSnapshot(StrictModel):
    requirement_id: str
    revision_id: str
    statement: str
    normalized_statement: str


class RequirementArtifact(StrictModel):
    artifact_schema_version: Literal["1.0", "2.0", "2.1"] = "2.1"
    project: str
    document_id: str
    document_version_id: str
    version: str
    source_path: str = ""
    source_checksum: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    canonical_facts: list[CanonicalFact] = Field(default_factory=list)
    facts: list[VerifiedFact]
    requirements: list[VerifiedRequirement]
    delta_events: list[RequirementDeltaEvent] = Field(default_factory=list)


class CompactRequirementOccurrence(StrictModel):
    chunk_id: str
    fact_id: str
    requirement_text: str
    requirement_type: str
    priority: Literal["High", "Medium", "Low"]
    status: Literal["Active", "Superseded"]
    doc_version: str


class CompactRequirementArtifact(StrictModel):
    artifact_schema_version: Literal["3.0-compact"] = "3.0-compact"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requirements: dict[str, list[CompactRequirementOccurrence]]


class RequirementCatalogEntry(StrictModel):
    display_id: str
    requirement_uid: str
    revision_id: str
    source_req_id: str | None = None
    id_generation_type: Literal["source", "generated"] = "generated"
    confidence: float = Field(ge=0.0, le=1.0)
    chunk_id: str
    fact_id: str
    requirement_text: str
    requirement_type: str
    priority: Literal["High", "Medium", "Low"]
    status: Literal["Active", "Superseded"]
    doc_version: str


class RequirementCatalogTraceability(StrictModel):
    req_id: str
    requirement_uid: str
    revision_id: str
    chunk_id: str
    fact_ids: list[str]


class RequirementsCatalogArtifact(StrictModel):
    artifact_schema_version: Literal["4.0-catalog"] = "4.0-catalog"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requirements: list[RequirementCatalogEntry]
    traceability: list[RequirementCatalogTraceability] = Field(default_factory=list)


class RequirementInput(StrictModel):
    """Normalized requirement fed into user-story generation (stage 3 input)."""

    requirement_id: str
    revision_id: str = ""
    display_id: str = ""
    source_req_id: str | None = None
    id_generation_type: Literal["source", "generated"] = "generated"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requirement_text: str
    requirement_type: str = "Functional Requirement"
    priority: Literal["High", "Medium", "Low"] = "Medium"
    evidence_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> str:
        return normalize_priority_label(value)

    @field_validator("requirement_id", "requirement_text")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class UserStoryStatement(StrictModel):
    as_a: str
    i_want: str
    so_that: str

    @field_validator("as_a", "i_want", "so_that")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class AcceptanceCriterion(StrictModel):
    id: str = ""
    title: str
    given: str
    when: str
    then: str

    @field_validator("title", "given", "when", "then")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class BusinessRule(StrictModel):
    id: str = ""
    rule: str

    @field_validator("rule")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class TestScenario(StrictModel):
    id: str = ""
    scenario: str

    @field_validator("scenario")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class _UserStoryContent(StrictModel):
    title: str
    priority: Literal["High", "Medium", "Low"] = "Medium"
    persona: str
    user_story: UserStoryStatement
    acceptance_criteria: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> str:
        return normalize_priority_label(value)

    @field_validator("title", "persona")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        if _PLACEHOLDER_USER_STORY_TEXT_RE.match(value):
            raise ValueError(f"value must not be a placeholder: {value!r}")
        return value

    @field_validator("acceptance_criteria")
    @classmethod
    def non_empty_acceptance_criteria(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("acceptance_criteria entries must not be empty")
        return cleaned


class UserStoryModel(_UserStoryContent):
    """A single user story exactly as returned by the reasoning model.

    The model returns content only; Python assigns the permanent ``story_id`` and
    renumbers nested acceptance-criteria / business-rule / test-scenario ids.
    """


class UserStoryGenerationOutput(StrictModel):
    user_stories: list[UserStoryModel] = Field(min_length=1)


class UserStoryRecord(_UserStoryContent):
    """A persisted user story with permanent id and provenance."""

    story_id: str
    display_id: str = ""
    requirement_id: str
    requirement_display_id: str = ""
    requirement_revision_id: str = ""
    source_req_id: str | None = None
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    origin_version: str = ""
    status: Literal["active", "superseded", "outdated"] = "active"
    origin: Literal["generation", "feedback"] = "generation"
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class UserStoryProjection(_UserStoryContent):
    display_id: str
    req_id: str
    source_req_id: str | None = None


class UserStoryTraceability(StrictModel):
    us_id: str
    req_id: str
    source_req_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class UserStoryArtifact(StrictModel):
    artifact_schema_version: Literal["2.0-user-stories"] = "2.0-user-stories"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    stories: list[UserStoryProjection]
    traceability: list[UserStoryTraceability] = Field(default_factory=list)


class UserStoryBuildResult(StrictModel):
    artifact: UserStoryArtifact
    records: dict[str, UserStoryRecord]
    coverage: dict[str, list[str]] = Field(default_factory=dict)


class UserStoryRequest(StrictModel):
    requirements_path: Path | None = None
    project: str | None = None
    document_version_id: str | None = None
    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None
    top_k: int | None = None


class UserStoryResult(StrictModel):
    run_id: str
    status: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: Path
    requirement_count: int
    story_ids: list[str]
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


ScenarioTypeLabel = Literal[
    "Positive",
    "Negative",
    "Boundary",
    "Alternative",
    "Exception",
    "Performance",
    "Security",
    "Usability",
]

_SCENARIO_TYPE_SYNONYMS: dict[str, ScenarioTypeLabel] = {
    "positive": "Positive",
    "happy path": "Positive",
    "functional": "Positive",
    "smoke": "Positive",
    "negative": "Negative",
    "error": "Negative",
    "invalid": "Negative",
    "boundary": "Boundary",
    "edge": "Boundary",
    "edge case": "Boundary",
    "limit": "Boundary",
    "alternative": "Alternative",
    "alt": "Alternative",
    "alt flow": "Alternative",
    "exception": "Exception",
    "recovery": "Exception",
    "failover": "Exception",
    "error handling": "Exception",
    "performance": "Performance",
    "load": "Performance",
    "stress": "Performance",
    "latency": "Performance",
    "security": "Security",
    "authn": "Security",
    "authz": "Security",
    "usability": "Usability",
    "ux": "Usability",
    "accessibility": "Usability",
}


def normalize_scenario_type_label(value: object) -> ScenarioTypeLabel:
    """Normalize arbitrary scenario-type labels to the curated stage-4 vocabulary."""
    if value is None:
        return "Positive"
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return "Positive"
    normalized = " ".join(text.lower().replace("-", " ").replace("_", " ").split())
    return _SCENARIO_TYPE_SYNONYMS.get(normalized, "Positive")


class _TestScenarioContent(StrictModel):
    title: str
    description: str
    scenario_type: ScenarioTypeLabel = "Positive"
    preconditions: list[str] = Field(default_factory=list)
    expected_result: str
    priority: Literal["High", "Medium", "Low"] = "Medium"
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> str:
        return normalize_priority_label(value)

    @field_validator("scenario_type", mode="before")
    @classmethod
    def normalize_scenario_type(cls, value: object) -> str:
        return normalize_scenario_type_label(value)

    @field_validator("title", "description", "expected_result")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        if _PLACEHOLDER_USER_STORY_TEXT_RE.match(value):
            raise ValueError(f"value must not be a placeholder: {value!r}")
        return value


class TestScenarioModel(_TestScenarioContent):
    """A single test scenario exactly as returned by the reasoning model.

    The model returns content only; Python assigns the permanent ``scenario_id``.
    """


class TestScenarioGenerationOutput(StrictModel):
    test_scenarios: list[TestScenarioModel] = Field(min_length=1)


class TestScenarioRecord(_TestScenarioContent):
    """A persisted test scenario with permanent id and full backward provenance."""

    scenario_id: str
    display_id: str = ""
    story_id: str
    story_display_id: str = ""
    requirement_id: str
    requirement_display_id: str = ""
    requirement_revision_id: str = ""
    source_req_id: str | None = None
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    origin_version: str = ""
    status: Literal["active", "superseded", "outdated"] = "active"
    origin: Literal["generation", "feedback"] = "generation"
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class TestScenarioProjection(_TestScenarioContent):
    display_id: str
    us_id: str
    req_id: str
    source_req_id: str | None = None


class TestScenarioTraceability(StrictModel):
    ts_id: str
    us_id: str
    req_id: str
    source_req_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class TestScenarioArtifact(StrictModel):
    artifact_schema_version: Literal["2.0-test-scenarios"] = "2.0-test-scenarios"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    scenarios: list[TestScenarioProjection]
    traceability: list[TestScenarioTraceability] = Field(default_factory=list)


class TestScenarioBuildResult(StrictModel):
    artifact: TestScenarioArtifact
    records: dict[str, TestScenarioRecord]
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    requirement_coverage: dict[str, list[str]] = Field(default_factory=dict)


class TestScenarioRequest(StrictModel):
    user_stories_path: Path | None = None
    requirements_path: Path | None = None
    project: str | None = None
    document_version_id: str | None = None
    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None
    top_k: int | None = None
    hfil_enabled: bool | None = None
    emit_md: bool = False
    thread_id: str | None = None


class TestScenarioResult(StrictModel):
    run_id: str
    status: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: Path
    story_count: int
    scenario_ids: list[str]
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    requirement_coverage: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class FeedbackRequest(StrictModel):
    """Human feedback request envelope. Requirements are intentionally immutable."""

    stage: Literal["user_story", "test_scenario"]
    target_id: str
    comment: str


class CanonicalScenario(StrictModel):
    entity: str
    action: str
    condition: str
    expected_behavior: str
    given: str
    when: str
    then: str
    canonical_text: str


class DuplicateCandidate(StrictModel):
    left_id: str
    right_id: str
    cosine: float
    reason: str = "embedding_recall"


class DuplicateGroup(StrictModel):
    scenario_ids: list[str]
    scenarios: list[TestScenarioRecord]
    story_ids: list[str]
    reason: str
    confidence: float
    verification_method: str


class DuplicateJudgeResult(StrictModel):
    a_entails_b: bool
    b_entails_a: bool
    verdict: Literal["DUPLICATE", "DISTINCT"]
    reason: str


class HFILTurn(StrictModel):
    command: str
    message: str = ""
    deleted_scenario_ids: list[str] = Field(default_factory=list)
    added_scenario_ids: list[str] = Field(default_factory=list)


class HFILState(StrictModel):
    hfil_enabled: bool = False
    hfil_done: bool = False
    hfil_phase: str = "start"
    hfil_pending_prompt: str = ""
    hfil_scenarios: list[TestScenarioRecord] = Field(default_factory=list)
    hfil_user_stories: list[UserStoryRecord] = Field(default_factory=list)
    hfil_last_duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    hfil_messages: list[str] = Field(default_factory=list)


class RequirementDeltaDecision(StrictModel):
    requirement_id: str
    revision_id: str
    label: Literal["new", "updated", "strictly_outdated", "unchanged"]
    prior_revision_id: str | None = None
    reason: str = ""


class ArtifactReadResult(StrictModel):
    source: Literal["local_json", "postgres"]
    valid_local: bool
    artifact_path: str | None = None
    payload: dict[str, Any]
    repaired: bool = False
    reason: str = ""


class ReconcileReport(StrictModel):
    project: str
    document_version_id: str | None = None
    repaired_paths: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class IngestionResult(StrictModel):
    run_id: str
    status: str
    project: str
    version: str
    document_id: str
    document_version_id: str
    checksum: str
    manifest_path: Path
    artifact_path: Path
    full_artifact_path: Path | None = None
    chunk_ids: list[str]
    fact_ids: list[str]
    requirement_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
