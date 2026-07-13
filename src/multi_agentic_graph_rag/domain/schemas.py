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
_SEMANTIC_MODAL_RE = re.compile(r"\b(shall|must|should|may|will)\b", re.I)
_ENTITY_DISCRIMINATOR_RE = re.compile(
    r"\b(?:[A-Za-z]+[-_]\d+[A-Za-z0-9_-]*|[A-Z]{2,}[A-Z0-9_-]*|"
    r"(?:register|channel|sensor|api|port)\s*[-_:]?\s*[A-Za-z0-9_-]+)\b"
)
_MUTABLE_PARAMETER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:°\s*[CF]|ms|milliseconds?|seconds?|minutes?|hours?|%|percent|"
    r"degrees?|bytes?|kb|mb|gb|items?|records?|requests?)\b",
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


class RequirementSemanticFields(StrictModel):
    """Validated semantic slots used for identity resolution, never permanent IDs."""

    actor: str = ""
    modality: str = ""
    action: str = ""
    object: str = ""
    condition: str = ""
    polarity: Literal["positive", "negative"] = "positive"
    requirement_family: str = ""
    entity_discriminators: list[str] = Field(default_factory=list)
    mutable_parameters: list[str] = Field(default_factory=list)

    def populate_from_statement(self, statement: str, requirement_type: str) -> None:
        text = " ".join(statement.strip().split())
        modal_match = _SEMANTIC_MODAL_RE.search(text)
        if modal_match is not None:
            actor = text[: modal_match.start()].strip(" ,:;-")
            remainder = text[modal_match.end() :].strip()
            negative = remainder.lower().startswith("not ")
            if negative:
                remainder = remainder[4:].strip()
            action, _, object_text = remainder.partition(" ")
            object.__setattr__(self, "actor", self.actor.strip() or actor or "system")
            object.__setattr__(
                self, "modality", self.modality.strip() or modal_match.group(1).lower()
            )
            object.__setattr__(self, "action", self.action.strip() or action)
            object.__setattr__(self, "object", self.object.strip() or object_text)
            if negative:
                object.__setattr__(self, "polarity", "negative")
        else:
            object.__setattr__(self, "actor", self.actor.strip() or "system")
            object.__setattr__(self, "action", self.action.strip() or text)
        condition_match = re.search(
            r"\b(if|when|unless|while|during|after|before|where)\b.+$", text, re.I
        )
        if condition_match is not None and not self.condition.strip():
            object.__setattr__(self, "condition", condition_match.group(0))
        object.__setattr__(
            self,
            "requirement_family",
            self.requirement_family.strip() or requirement_type.strip() or "Functional Requirement",
        )
        discriminators = list(dict.fromkeys(_ENTITY_DISCRIMINATOR_RE.findall(text)))
        object.__setattr__(
            self,
            "entity_discriminators",
            list(dict.fromkeys([*self.entity_discriminators, *discriminators])),
        )
        mutable = [match.group(0) for match in _MUTABLE_PARAMETER_RE.finditer(text)]
        object.__setattr__(
            self,
            "mutable_parameters",
            list(dict.fromkeys([*self.mutable_parameters, *mutable])),
        )


class LLMRequirementCandidate(RequirementSemanticFields):
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


class LLMDiscoveredRequirement(RequirementSemanticFields):
    req_text: str
    requirement_type: str = "Functional Requirement"
    priority: str = "Medium"
    requirement_key: str | None = None
    source_req_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_atomic_semantics(self) -> LLMDiscoveredRequirement:
        modal_count = len(_SEMANTIC_MODAL_RE.findall(self.req_text))
        if modal_count > 1:
            raise ValueError(
                "req_text contains multiple modal obligations; split into atomic records"
            )
        self.populate_from_statement(self.req_text, self.requirement_type)
        if not self.actor or not self.action or not self.requirement_family:
            raise ValueError("actor, action, and requirement_family must be non-empty")
        return self

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


class VerifiedRequirement(RequirementSemanticFields):
    requirement_id: str
    revision_id: str = ""
    requirement_key: str = ""
    source_req_id: str | None = None
    id_generation_type: Literal["source", "generated"] = "generated"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    statement: str
    normalized_statement: str = ""
    semantic_signature: str = ""
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
    requirement_type: str = ""
    semantic_signature: str = ""
    evidence_ids: dict[str, str] = Field(default_factory=dict)


class RequirementIdentityResolutionRecord(StrictModel):
    incoming_fingerprint: str
    document_version_id: str
    chunk_id: str
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    reranker_order: list[str] = Field(default_factory=list)
    deterministic_rule: str
    judge_result: str | None = None
    decision: Literal["EXACT", "SAME_LINEAGE_REVISION", "DISTINCT", "AMBIGUOUS"]
    reason: str
    requirement_id: str
    revision_id: str


class RequirementIdentityResolutionArtifact(StrictModel):
    artifact_schema_version: Literal["1.0-requirement-identity-resolution"] = (
        "1.0-requirement-identity-resolution"
    )
    project: str
    document_id: str
    document_version_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolutions: list[RequirementIdentityResolutionRecord] = Field(default_factory=list)


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
    identity_resolutions: list[RequirementIdentityResolutionRecord] = Field(default_factory=list)


class CanonicalRequirementEvidence(StrictModel):
    """One preserved source occurrence for a canonical requirement revision."""

    evidence_id: str
    document_version_id: str
    chunk_id: str
    fact_ids: list[str] = Field(default_factory=list)
    quote: str
    start_char: int
    end_char: int
    page: int | None = None
    section: str | None = None
    source_path: str = ""


class CanonicalRequirement(RequirementSemanticFields):
    """Canonical public requirement row; occurrences are nested, never duplicated."""

    requirement_id: str
    revision_id: str
    source_req_id: str | None = None
    id_generation_type: Literal["source", "generated"] = "generated"
    confidence: float = Field(ge=0.0, le=1.0)
    requirement_text: str
    semantic_signature: str
    requirement_type: str
    priority: Literal["High", "Medium", "Low"]
    status: Literal["Active", "Superseded"]
    evidence: list[CanonicalRequirementEvidence] = Field(default_factory=list)


class CanonicalRequirementsArtifact(StrictModel):
    """Public schema 5.0: one row per canonical requirement revision."""

    artifact_schema_version: Literal["5.0-requirements"] = "5.0-requirements"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requirements: list[CanonicalRequirement]


class RequirementInput(StrictModel):
    """Normalized requirement fed into user-story generation (stage 3 input)."""

    requirement_id: str
    revision_id: str = ""
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
    requirement_id: str
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
    # Generation grounding (Area 7): the retrieved context that actually produced
    # this story, kept separate from ``evidence_chunk_ids`` (requirement source).
    generation_context_run_id: str = ""
    retrieved_assertion_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_mode: str = ""


class UserStoryProjection(_UserStoryContent):
    story_id: str
    requirement_id: str
    revision_id: str = ""
    source_req_id: str | None = None


class UserStoryTraceability(StrictModel):
    story_id: str
    requirement_id: str
    revision_id: str = ""
    source_req_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    generation_context_run_id: str = ""
    retrieved_assertion_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_mode: str = ""


class UserStoryArtifact(StrictModel):
    artifact_schema_version: Literal["3.0-user-stories"] = "3.0-user-stories"
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
    story_id: str
    requirement_id: str
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
    # Generation grounding (Area 7): the retrieved context that actually produced
    # this scenario, kept separate from ``evidence_chunk_ids`` (requirement source).
    generation_context_run_id: str = ""
    retrieved_assertion_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_mode: str = ""


class TestScenarioProjection(_TestScenarioContent):
    scenario_id: str
    story_id: str
    requirement_id: str
    revision_id: str = ""
    source_req_id: str | None = None


class TestScenarioTraceability(StrictModel):
    scenario_id: str
    story_id: str
    requirement_id: str
    revision_id: str = ""
    source_req_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    generation_context_run_id: str = ""
    retrieved_assertion_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_mode: str = ""


class TestScenarioArtifact(StrictModel):
    artifact_schema_version: Literal["3.0-test-scenarios"] = "3.0-test-scenarios"
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
    warnings: list[str] = Field(default_factory=list)


MasterStage = Literal["requirements", "user_stories", "test_scenarios"]


class StageMasterArtifact(StrictModel):
    """Cumulative per-project master projection for one pipeline stage.

    One of these exists per project per stage. ``records`` holds the full
    cumulative set (active + all preserved historical revisions), materialized
    deterministically from the normalized PostgreSQL rows. ``checksum`` covers
    only the reproducible content (schema version, stage, project, document id,
    records) — never the run/time/revision metadata — so a re-materialization
    from unchanged rows is byte-identical and drift detection never cries wolf.
    """

    artifact_schema_version: str
    stage: MasterStage
    project: str
    document_id: str
    current_document_version_id: str = ""
    payload_revision: int = 0
    run_id: str = ""
    checksum: str = ""
    record_count: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    records: list[dict[str, Any]] = Field(default_factory=list)


KnowledgeGraphStatus = Literal["pending", "building", "ready", "failed", "rebuilding"]


class KnowledgeGraphStateRecord(StrictModel):
    """Version-scoped readiness state for the semantic knowledge graph.

    ``ready`` is reached only at the end of the full successful build path
    (extraction -> validation -> projection -> active-pointer move). A partial or
    crashed build stays ``building`` / ``failed``, so graph-primary generation is
    blocked fail-closed until an explicit rebuild succeeds.
    """

    document_version_id: str
    project: str
    document_id: str
    doc_version: str
    status: KnowledgeGraphStatus
    run_id: str = ""
    attempt: int = 0
    failure_reason: str | None = None
    chunk_count: int = 0
    assertion_count: int = 0
    evidence_count: int = 0
    extractor_fingerprint: str = ""
    graph_schema_version: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    chunk_ids: list[str]
    fact_ids: list[str]
    requirement_ids: list[str]
    # Requirement discovery and the KG build are separate commit boundaries:
    # ``ingestion_status`` is ``completed`` when both succeed and ``degraded`` when
    # requirements persisted but the KG build failed. ``downstream_blocked`` mirrors
    # the graph-primary gate so operators see immediately that story/scenario
    # generation is blocked until the rebuild command runs.
    ingestion_status: Literal["completed", "degraded"] = "completed"
    kg_status: KnowledgeGraphStatus | None = None
    kg_failure_reason: str | None = None
    downstream_blocked: bool = False
    kg_rebuild_command: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --- Knowledge graph (source-semantics layer built by ``marag build-knowledge-graph``) ---

_ENTITY_PLACEHOLDER_RE = re.compile(
    r"^(?:entity|thing|item|object|subject|actor|system|component|tbd|todo|"
    r"placeholder|n/?a|na|none|null|unknown)$",
    re.I,
)

_ASSERTION_MODALITIES = ("fact", "shall", "must", "should", "may", "must_not")


def _normalize_assertion_modality(value: object) -> str:
    if value is None:
        return "fact"
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if text in {"", "null", "none", "n/a", "na", "unknown"}:
        return "fact"
    if text in {"must_not", "shall_not", "may_not", "should_not", "prohibited", "forbidden"}:
        return "must_not"
    if text in _ASSERTION_MODALITIES:
        return text
    if text in {"mandatory", "required", "requirement"}:
        return "must"
    if text in {"recommended", "recommendation"}:
        return "should"
    if text in {"optional", "permitted", "allowed"}:
        return "may"
    return "fact"


class LLMExtractedEntity(StrictModel):
    """One entity exactly as returned by the extraction model for a single chunk."""

    name: str
    entity_type: str = "concept"

    @field_validator("name")
    @classmethod
    def meaningful_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("entity name must not be empty")
        if _ENTITY_PLACEHOLDER_RE.match(value):
            raise ValueError(f"entity name must not be a placeholder: {value!r}")
        return value

    @field_validator("entity_type", mode="before")
    @classmethod
    def default_entity_type(cls, value: object) -> str:
        if value is None:
            return "concept"
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
            return "concept"
        return text


class LLMExtractedAssertion(StrictModel):
    """One assertion exactly as returned by the extraction model for a single chunk.

    ``subject`` and ``object_name`` reference entities by name from the same chunk
    output. Exactly one of ``object_name`` / ``object_literal`` must be non-empty;
    the repo-wide LLM convention is an empty string for absent optional fields.
    """

    subject: str
    predicate: str
    object_name: str = ""
    object_literal: str = ""
    modality: Literal["fact", "shall", "must", "should", "may", "must_not"] = "fact"
    polarity: Literal["positive", "negative"] = "positive"
    explicitness: Literal["explicit", "inferred"] = "explicit"
    condition: str = ""
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("subject", "predicate", "quote")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("modality", mode="before")
    @classmethod
    def normalize_modality(cls, value: object) -> str:
        return _normalize_assertion_modality(value)

    @field_validator("polarity", mode="before")
    @classmethod
    def normalize_polarity(cls, value: object) -> str:
        text = str(value).strip().lower() if value is not None else ""
        if text in {"negative", "neg", "not", "false"}:
            return "negative"
        return "positive"

    @field_validator("explicitness", mode="before")
    @classmethod
    def normalize_explicitness(cls, value: object) -> str:
        text = str(value).strip().lower() if value is not None else ""
        if text in {"inferred", "implicit", "implied", "derived"}:
            return "inferred"
        return "explicit"

    @model_validator(mode="after")
    def exactly_one_object(self) -> LLMExtractedAssertion:
        has_entity = bool(self.object_name.strip())
        has_literal = bool(self.object_literal.strip())
        if has_entity == has_literal:
            raise ValueError(
                "exactly one of object_name or object_literal must be non-empty "
                f"(subject={self.subject!r}, predicate={self.predicate!r})"
            )
        return self


class KnowledgeExtractionChunkOutput(StrictModel):
    entities: list[LLMExtractedEntity] = Field(default_factory=list)
    assertions: list[LLMExtractedAssertion] = Field(default_factory=list)


class EntityCandidate(StrictModel):
    """A grounded per-chunk entity occurrence awaiting resolution."""

    chunk_id: str
    surface_text: str
    normalized_name: str
    entity_type: str


class AssertionCandidate(StrictModel):
    """A grounded per-chunk assertion occurrence awaiting canonicalization."""

    chunk_id: str
    subject_name: str
    predicate: str
    object_name: str | None = None
    object_literal: str | None = None
    modality: str
    polarity: Literal["positive", "negative"]
    explicitness: Literal["explicit", "inferred"]
    condition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_trace: SourceTrace


class ChunkKnowledgeCandidates(StrictModel):
    chunk_id: str
    entities: list[EntityCandidate] = Field(default_factory=list)
    assertions: list[AssertionCandidate] = Field(default_factory=list)


class KnowledgeExtractionOutput(StrictModel):
    chunks: list[ChunkKnowledgeCandidates] = Field(default_factory=list)


class TextUnit(StrictModel):
    """An atomic source segment (sentence/bullet) with source-level offsets.

    Text units are derived deterministically from the ingested chunks at
    knowledge-graph build time; overlapping chunk windows dedupe to one unit
    per source span, and ``chunk_ids`` lists every chunk containing the unit.
    """

    text_unit_id: str
    document_version_id: str
    ordinal: int
    unit_type: Literal["sentence", "bullet"]
    text: str
    start_char: int
    end_char: int
    page: int | None = None
    section: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class EntityRecord(StrictModel):
    """A resolved project-scoped canonical entity."""

    entity_id: str
    project: str
    canonical_name: str
    normalized_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)


class EntityMentionRecord(StrictModel):
    """One grounded occurrence of an entity inside a chunk."""

    mention_id: str
    entity_id: str
    chunk_id: str
    surface_text: str
    start_char: int | None = None
    end_char: int | None = None


class AssertionRecord(StrictModel):
    """A canonical (deduplicated) assertion scoped to one document version."""

    assertion_id: str
    assertion_key: str
    assertion_lineage_key: str = ""
    project: str
    document_id: str
    document_version_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str | None = None
    object_literal: str | None = None
    modality: str
    polarity: str
    explicitness: str
    condition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    display_text: str
    # Cross-version lifecycle (Area 6). ``status`` is the live/history flag;
    # ``previous_assertion_id`` links a revision to the one it replaced, and
    # ``superseded_by_assertion_id`` is set on the prior node when replaced.
    status: Literal["active", "superseded", "retired"] = "active"
    previous_assertion_id: str | None = None
    superseded_by_assertion_id: str | None = None
    revision_type: Literal["new", "unchanged", "changed"] = "new"


class AssertionEvidenceRecord(StrictModel):
    """One exact source occurrence supporting a canonical assertion."""

    evidence_id: str
    assertion_id: str
    source_trace: SourceTrace
    text_unit_ids: list[str] = Field(default_factory=list)


class KnowledgeGraphArtifact(StrictModel):
    artifact_schema_version: Literal["1.0-knowledge-graph"] = "1.0-knowledge-graph"
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    text_units: list[TextUnit] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    mentions: list[EntityMentionRecord] = Field(default_factory=list)
    assertions: list[AssertionRecord] = Field(default_factory=list)
    evidence: list[AssertionEvidenceRecord] = Field(default_factory=list)


class KnowledgeGraphRequest(StrictModel):
    project: str
    document_version_id: str
    reasoning_provider: str | None = None

    @field_validator("project", "document_version_id")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class KnowledgeGraphResult(StrictModel):
    run_id: str
    status: str
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    artifact_path: Path
    chunk_count: int
    entity_count: int
    assertion_count: int
    evidence_count: int
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GenerationContextRun(StrictModel):
    """One retrieval invocation snapshot (shadow comparison / audit)."""

    context_run_id: str
    stage: str
    anchor_id: str
    project: str = ""
    document_version_id: str
    source: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class GenerationContextItem(StrictModel):
    """One candidate or selected item inside a retrieval snapshot.

    The base fields (``item_id``/``source``/``score``/``selected``) describe the
    legacy chunk-shaped snapshot. The additive assertion fields carry the
    structured knowledge-graph retrieval channel (schema §18): they stay ``None``
    for chunk items so historical rows keep validating.
    """

    context_run_id: str
    rank: int
    item_type: str = "chunk"
    item_id: str
    source: str
    score: float | None = None
    selected: bool = True
    assertion_id: str | None = None
    text_unit_id: str | None = None
    entity_id: str | None = None
    predicate: str | None = None
    hop_count: int | None = None
    normalized_score: float | None = None
    reranker_score: float | None = None
    mandatory: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssertionEvidenceContext(StrictModel):
    """One exact source occurrence backing an assertion in retrieval context."""

    text_unit_id: str | None = None
    chunk_id: str
    quote: str
    page: int | None = None
    section: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class AssertionContextItem(StrictModel):
    """One structured, evidence-backed assertion selected for generation.

    This is the bounded semantic-context unit (schema §14) that replaces raw
    chunk text: subject/predicate/object plus modality, polarity, condition,
    confidence, retrieval provenance, and exact TextUnit evidence.
    """

    assertion_id: str
    subject_entity_id: str
    subject_name: str
    subject_type: str
    predicate: str
    object_entity_id: str | None = None
    object_name: str | None = None
    object_literal: str | None = None
    modality: str
    polarity: str
    explicitness: str
    condition: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    hop_count: int = 0
    source_channel: str
    mandatory: bool = False
    retrieval_score: float = 0.0
    evidence: list[AssertionEvidenceContext] = Field(default_factory=list)


class SemanticContext(StrictModel):
    """The bounded structured assertion context assembled for one anchor.

    Shadow-mode contract: generation prompts are unchanged; this payload is only
    recorded as a retrieval snapshot until a stage's graph-primary flag is set.
    """

    stage: str
    anchor_id: str
    document_version_id: str
    items: list[AssertionContextItem] = Field(default_factory=list)
    mandatory_anchor_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GenerationTrace(StrictModel):
    """The retrieved context that actually produced one generated artifact.

    Separate from requirement-source evidence: this records the graph/chunk
    grounding (``context_run_id`` + selected assertions/chunks + mode) so a story
    or scenario can be traced back to exactly what the LLM was shown.
    """

    generation_context_run_id: str = ""
    retrieved_assertion_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_mode: str = ""


class CoverageRequirementRow(StrictModel):
    """One requirement's coverage status inside a strict coverage report."""

    requirement_id: str
    requirement_status: str
    coverage_status: str
    story_ids: list[str] = Field(default_factory=list)
    scenario_count: int = 0


class CoverageSummary(StrictModel):
    """Deterministic, zero-safe rollup of per-requirement coverage rows."""

    total_requirements: int = 0
    requirements_with_stories: int = 0
    requirements_scenario_covered: int = 0
    no_story_count: int = 0
    story_coverage_pct: float = 0.0
    scenario_coverage_pct: float = 0.0


class CoverageReport(StrictModel):
    """Strict per-requirement coverage report with a summary rollup."""

    project: str
    document_version_id: str | None = None
    requirements: list[CoverageRequirementRow] = Field(default_factory=list)
    summary: CoverageSummary = Field(default_factory=CoverageSummary)
