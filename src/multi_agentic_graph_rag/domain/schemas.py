"""Strict contracts for the project/run-scoped QA workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RequirementType = Literal[
    "Functional Requirement",
    "Business Requirement",
    "Non-Functional Requirement",
    "Security Requirement",
    "Configuration Requirement",
    "Validation Requirement",
    "Alerting Requirement",
    "Health Requirement",
    "Data Quality Requirement",
    "Application Requirement",
    "Offline Requirement",
]
Priority = Literal["High", "Medium", "Low"]
SourceRequirementType = Literal["source", "generated"]
EntityType = Literal[
    "ACTOR",
    "SYSTEM",
    "COMPONENT",
    "DATA",
    "PROCESS",
    "RULE",
    "INTERFACE",
    "EXTERNAL_SYSTEM",
    "EVENT",
    "STATE",
]
RelationshipType = Literal[
    "USES",
    "SUPPORTS",
    "CONTROLS",
    "COLLECTS_FROM",
    "COMMUNICATES_VIA",
    "CONNECTS_TO",
    "REFERS_TO",
]
ScenarioType = Literal[
    "Positive",
    "Negative",
    "Boundary",
    "Alternative",
    "Exception",
    "Performance",
    "Security",
    "Usability",
]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def canonical_checksum(payload: BaseModel | dict[str, Any]) -> str:
    """Return the checksum of canonical JSON with the checksum field omitted."""
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    data.pop("checksum", None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


class StrictModel(BaseModel):
    """Base model that rejects undeclared properties."""

    __test__: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SourceProvenance(StrictModel):
    """Source requirement provenance inherited by downstream artifacts."""

    source_req_id: str | None
    source_req_id_type: SourceRequirementType

    @model_validator(mode="after")
    def validate_pair(self) -> SourceProvenance:
        """Enforce the nullable provenance pair."""
        if self.source_req_id_type == "source":
            if self.source_req_id is None or not self.source_req_id.strip():
                raise ValueError("source provenance requires a non-empty source_req_id")
            object.__setattr__(self, "source_req_id", self.source_req_id.strip())
        elif self.source_req_id is not None:
            raise ValueError("generated provenance requires source_req_id=null")
        return self


class IngestionRequest(StrictModel):
    """Business input for Stage 1.1."""

    project_name: str
    source_file: Path
    embedding_provider: str | None = None

    @field_validator("project_name")
    @classmethod
    def project_is_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project_name must not be empty")
        return value


class ParsedBlock(StrictModel):
    """Layout-aware parser output."""

    block_id: str
    original_text: str
    normalized_text: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    start_char: int
    end_char: int
    block_type: Literal[
        "heading",
        "paragraph",
        "list_item",
        "table",
        "table_row",
        "table_cell",
        "code",
        "other",
    ] = "paragraph"
    source_location: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkLayout(StrictModel):
    """Layout information retained on a manifest chunk."""

    page_start: int | None
    page_end: int | None
    section: str | None
    block_types: list[
        Literal[
            "heading",
            "paragraph",
            "list_item",
            "table",
            "table_row",
            "table_cell",
            "code",
            "other",
        ]
    ]
    source_location: str | None


class ManifestChunk(StrictModel):
    """Stable chunk contract shared by every stage."""

    chunk_id: str
    sequence_index: int = Field(ge=0)
    chunk_text: str
    content_hash: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    layout: ChunkLayout
    source_provenance: dict[str, Any] | None
    neo4j_status: Literal["pending", "persisted"]
    chroma_status: Literal["pending", "persisted"]

    @model_validator(mode="after")
    def validate_content(self) -> ManifestChunk:
        if not self.chunk_text.strip():
            raise ValueError("chunk_text must not be empty")
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        expected = f"sha256:{hashlib.sha256(self.chunk_text.encode('utf-8')).hexdigest()}"
        if self.content_hash != expected:
            raise ValueError("content_hash does not match chunk_text")
        return self


class ChunkManifest(StrictModel):
    """Public Stage 1.1 artifact."""

    artifact_schema_version: Literal["1.0-chunk-manifest"] = "1.0-chunk-manifest"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    chunks: list[ManifestChunk]

    @model_validator(mode="after")
    def validate_manifest(self) -> ChunkManifest:
        indexes = [chunk.sequence_index for chunk in self.chunks]
        if indexes != list(range(len(self.chunks))):
            raise ValueError("chunk sequence_index values must be contiguous from zero")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique")
        if any(
            chunk.neo4j_status != "persisted" or chunk.chroma_status != "persisted"
            for chunk in self.chunks
        ):
            raise ValueError("manifest publication requires both stores to be persisted")
        if self.checksum != canonical_checksum(self):
            raise ValueError("chunk manifest checksum mismatch")
        return self


class IngestionResult(StrictModel):
    """Stage 1.1 result."""

    project: str
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    chunk_ids: list[str]


class RequirementEvidenceQuote(StrictModel):
    """Temporary Stage 1.2 evidence supplied by the reasoning model."""

    quote: str


class LLMRequirementCandidate(SourceProvenance):
    """One temporary requirement candidate in the combined response."""

    requirement_ref: str
    requirement_text: str
    requirement_type: RequirementType
    priority: Priority
    constraints: list[str]
    entity_refs: list[str]
    relationship_refs: list[str]
    evidence_quotes: list[str] = Field(min_length=1)
    confidence: Confidence


class LLMEntityCandidate(StrictModel):
    """One temporary entity candidate in the combined response."""

    entity_ref: str
    name: str
    normalized_name: str
    entity_type: EntityType
    aliases: list[str]
    evidence_quotes: list[str] = Field(min_length=1)
    confidence: Confidence


class LLMRelationshipCandidate(StrictModel):
    """One temporary semantic relationship in the combined response."""

    relationship_ref: str
    source_entity_ref: str
    relationship_type: RelationshipType
    target_entity_ref: str
    evidence_quote: str
    confidence: Confidence


class RequirementDiscoveryChunkResponse(StrictModel):
    """The only reasoning-model response allowed for a Stage 1.2 chunk."""

    chunk_id: str
    requirements: list[LLMRequirementCandidate]
    entities: list[LLMEntityCandidate]
    relationships: list[LLMRelationshipCandidate]

    @model_validator(mode="after")
    def validate_references(self) -> RequirementDiscoveryChunkResponse:
        requirement_refs = [item.requirement_ref for item in self.requirements]
        entity_refs = [item.entity_ref for item in self.entities]
        relationship_refs = [item.relationship_ref for item in self.relationships]
        for label, values in (
            ("requirement_ref", requirement_refs),
            ("entity_ref", entity_refs),
            ("relationship_ref", relationship_refs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} values must be unique")
        if not self.requirements and (self.entities or self.relationships):
            raise ValueError("empty requirements requires empty entities and relationships")
        entity_set = set(entity_refs)
        relationship_set = set(relationship_refs)
        referenced_entities: set[str] = set()
        referenced_relationships: set[str] = set()
        for requirement in self.requirements:
            if not set(requirement.entity_refs) <= entity_set:
                raise ValueError("requirement entity_refs contain unresolved references")
            if not set(requirement.relationship_refs) <= relationship_set:
                raise ValueError("requirement relationship_refs contain unresolved references")
            referenced_entities.update(requirement.entity_refs)
            referenced_relationships.update(requirement.relationship_refs)
        for relationship in self.relationships:
            if relationship.source_entity_ref not in entity_set:
                raise ValueError("relationship source entity is unresolved")
            if relationship.target_entity_ref not in entity_set:
                raise ValueError("relationship target entity is unresolved")
            if relationship.source_entity_ref == relationship.target_entity_ref:
                raise ValueError("self-referencing relationships are invalid")
            referenced_entities.update(
                (relationship.source_entity_ref, relationship.target_entity_ref)
            )
        if set(entity_refs) != referenced_entities:
            raise ValueError("orphan entities are invalid")
        if set(relationship_refs) != referenced_relationships:
            raise ValueError("orphan relationships are invalid")
        return self


class Evidence(StrictModel):
    """Canonical evidence span."""

    evidence_id: str
    chunk_id: str
    quote: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class EntityMention(StrictModel):
    """One exact entity mention."""

    chunk_id: str
    surface_text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class CanonicalEntity(StrictModel):
    """Canonical project-scoped entity."""

    entity_id: str
    name: str
    normalized_name: str
    entity_type: EntityType
    aliases: list[str]
    mentions: list[EntityMention]


class CanonicalRelationship(StrictModel):
    """Validated LLM-generated relationship projected to Neo4j."""

    relationship_id: str
    chunk_id: str
    source_entity_id: str
    relationship_type: RelationshipType
    target_entity_id: str
    confidence: Confidence
    extraction_method: Literal["llm"] = "llm"
    evidence: list[Evidence] = Field(min_length=1)


class CanonicalRequirement(SourceProvenance):
    """Canonical active requirement."""

    requirement_id: str
    requirement_text: str
    requirement_type: RequirementType
    priority: Priority
    status: Literal["active"] = "active"
    confidence: Confidence
    constraints: list[str]
    entity_ids: list[str]
    relationship_ids: list[str]
    evidence: list[Evidence] = Field(min_length=1)


class RequirementsArtifact(StrictModel):
    """Public Stage 1.2 artifact."""

    artifact_schema_version: Literal["1.2-requirements"] = "1.2-requirements"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    requirements: list[CanonicalRequirement]
    entities: list[CanonicalEntity]
    relationships: list[CanonicalRelationship]

    @model_validator(mode="after")
    def validate_artifact(self) -> RequirementsArtifact:
        entity_ids = {item.entity_id for item in self.entities}
        relationship_ids = {item.relationship_id for item in self.relationships}
        referenced_relationships: set[str] = set()
        for relationship in self.relationships:
            if relationship.source_entity_id not in entity_ids:
                raise ValueError("relationship source endpoint is unresolved")
            if relationship.target_entity_id not in entity_ids:
                raise ValueError("relationship target endpoint is unresolved")
        for requirement in self.requirements:
            if not set(requirement.entity_ids) <= entity_ids:
                raise ValueError("requirement entity_ids contain unresolved IDs")
            if not set(requirement.relationship_ids) <= relationship_ids:
                raise ValueError("requirement relationship_ids contain unresolved IDs")
            referenced_relationships.update(requirement.relationship_ids)
        if relationship_ids != referenced_relationships:
            raise ValueError("orphan canonical relationships are invalid")
        if self.checksum != canonical_checksum(self):
            raise ValueError("requirements artifact checksum mismatch")
        return self


class RequirementMapEntry(SourceProvenance):
    """Requirement occurrence retained before permanent requirement assignment."""

    requirement_ref: str
    requirement_text: str
    requirement_type: RequirementType
    priority: Priority
    constraints: list[str]
    evidence: list[Evidence] = Field(min_length=1)
    entity_ids: list[str]
    relationship_ids: list[str]
    confidence: Confidence


class RequirementChunkResult(StrictModel):
    """Terminal Stage 1.2 result for one manifest chunk."""

    chunk_id: str
    sequence_index: int
    status: Literal["completed", "no_requirements", "failed"]
    requirements: list[RequirementMapEntry]
    error: str | None

    @model_validator(mode="after")
    def validate_status(self) -> RequirementChunkResult:
        if self.status == "no_requirements" and self.requirements:
            raise ValueError("no_requirements requires requirements=[]")
        if self.status != "failed" and self.error is not None:
            raise ValueError("only failed entries may contain an error")
        return self


class RequirementEntityRelationshipMap(StrictModel):
    """Checkpointed Stage 1.2 aggregation state."""

    state_schema_version: Literal["1.0-requirement-map"] = "1.0-requirement-map"
    project: str
    run_id: str
    chunk_results: list[RequirementChunkResult]


class UserStoryStatement(StrictModel):
    """Structured user-story statement."""

    as_a: str
    i_want: str
    so_that: str


class LLMAcceptanceCriterion(StrictModel):
    """Acceptance criterion before permanent ID assignment."""

    title: str
    given: str
    when: str
    then: str


class AcceptanceCriterion(LLMAcceptanceCriterion):
    """Canonical acceptance criterion."""

    criterion_id: str


class LLMUserStoryDraft(StrictModel):
    """Stage 2 story body as returned by the model, before provenance stamping.

    Provenance (``source_req_id``/``source_req_id_type``) is intentionally absent:
    it is deterministically inherited from the active requirement by the agent, so
    the model is never asked to reproduce it.
    """

    story_ref: str
    title: str
    priority: Priority
    persona: str
    user_story: UserStoryStatement
    acceptance_criteria: list[LLMAcceptanceCriterion] = Field(min_length=1)
    business_rules: list[str]
    evidence_chunk_ids: list[str] = Field(min_length=1)
    supporting_entity_ids: list[str]
    supporting_relationship_ids: list[str]
    confidence: Confidence


class LLMUserStoryCandidate(LLMUserStoryDraft, SourceProvenance):
    """Temporary Stage 2 story candidate with provenance inherited from the requirement."""


class UserStoryGenerationDraft(StrictModel):
    """Model-facing Stage 2 response: story bodies without provenance.

    The agent stamps the requirement's provenance onto each draft to build the
    canonical :class:`UserStoryGenerationResponse`.
    """

    requirement_id: str
    user_stories: list[LLMUserStoryDraft]

    @model_validator(mode="after")
    def unique_refs(self) -> UserStoryGenerationDraft:
        refs = [story.story_ref for story in self.user_stories]
        if len(refs) != len(set(refs)):
            raise ValueError("story_ref values must be unique")
        return self


class UserStoryGenerationResponse(StrictModel):
    """Structured Stage 2 response for one canonical requirement."""

    requirement_id: str
    user_stories: list[LLMUserStoryCandidate]

    @model_validator(mode="after")
    def unique_refs(self) -> UserStoryGenerationResponse:
        refs = [story.story_ref for story in self.user_stories]
        if len(refs) != len(set(refs)):
            raise ValueError("story_ref values must be unique")
        return self


class Traceability(StrictModel):
    """Shared downstream traceability object."""

    evidence_chunk_ids: list[str] = Field(min_length=1)
    entity_ids: list[str]
    relationship_ids: list[str]


class CanonicalUserStory(SourceProvenance):
    """Canonical active user story."""

    story_id: str
    requirement_ids: list[str] = Field(min_length=1)
    title: str
    priority: Priority
    persona: str
    user_story: UserStoryStatement
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    business_rules: list[str]
    status: Literal["active"] = "active"
    confidence: Confidence
    traceability: Traceability


class UserStoriesArtifact(StrictModel):
    """Public Stage 2 artifact."""

    artifact_schema_version: Literal["1.1-user-stories"] = "1.1-user-stories"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    stories: list[CanonicalUserStory]

    @model_validator(mode="after")
    def validate_checksum(self) -> UserStoriesArtifact:
        if self.checksum != canonical_checksum(self):
            raise ValueError("user-stories artifact checksum mismatch")
        return self


class LLMTestScenarioDraft(StrictModel):
    """Stage 3 scenario body as returned by the model, before provenance stamping.

    Provenance is intentionally absent because it is deterministically inherited
    from the active story by the agent.  Keeping it out of the model-facing schema
    prevents canonical requirement IDs from being confused with source IDs.
    """

    scenario_ref: str
    title: str
    description: str
    scenario_type: ScenarioType
    priority: Priority
    preconditions: list[str]
    action: str
    expected_result: str
    covered_acceptance_criterion_ids: list[str] = Field(min_length=1)
    evidence_chunk_ids: list[str] = Field(min_length=1)
    supporting_entity_ids: list[str]
    supporting_relationship_ids: list[str]
    confidence: Confidence


class LLMTestScenarioCandidate(LLMTestScenarioDraft, SourceProvenance):
    """Temporary Stage 3 scenario candidate with provenance inherited from the story."""


class TestScenarioGenerationDraft(StrictModel):
    """Model-facing Stage 3 response containing scenario bodies without provenance."""

    story_id: str
    requirement_ids: list[str]
    test_scenarios: list[LLMTestScenarioDraft]

    @model_validator(mode="after")
    def unique_refs(self) -> TestScenarioGenerationDraft:
        refs = [scenario.scenario_ref for scenario in self.test_scenarios]
        if len(refs) != len(set(refs)):
            raise ValueError("scenario_ref values must be unique")
        return self


class TestScenarioGenerationResponse(StrictModel):
    """Structured Stage 3 response for one canonical story."""

    story_id: str
    requirement_ids: list[str]
    test_scenarios: list[LLMTestScenarioCandidate]

    @model_validator(mode="after")
    def unique_refs(self) -> TestScenarioGenerationResponse:
        refs = [scenario.scenario_ref for scenario in self.test_scenarios]
        if len(refs) != len(set(refs)):
            raise ValueError("scenario_ref values must be unique")
        return self


class CanonicalTestScenario(SourceProvenance):
    """Canonical active behavioral scenario."""

    scenario_id: str
    story_ids: list[str] = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    title: str
    description: str
    scenario_type: ScenarioType
    priority: Priority
    preconditions: list[str]
    action: str
    expected_result: str
    covered_acceptance_criterion_ids: list[str] = Field(min_length=1)
    status: Literal["active"] = "active"
    confidence: Confidence
    traceability: Traceability


class TestScenariosArtifact(StrictModel):
    """Public Stage 3 artifact."""

    artifact_schema_version: Literal["1.1-test-scenarios"] = "1.1-test-scenarios"
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    checksum: str
    scenarios: list[CanonicalTestScenario]

    @model_validator(mode="after")
    def validate_checksum(self) -> TestScenariosArtifact:
        if self.checksum != canonical_checksum(self):
            raise ValueError("test-scenarios artifact checksum mismatch")
        return self


class RetrievedEvidence(StrictModel):
    """One ranked retrieval item."""

    chunk_id: str
    text: str
    source: Literal["authoritative", "graph", "vector"]
    score: float
    entity_ids: list[str]
    relationship_ids: list[str]


class StoryContext(StrictModel):
    """Compact Stage 2 context package."""

    requirement_id: str
    requirement_text: str
    source_req_id: str | None
    source_req_id_type: SourceRequirementType
    authoritative_evidence_chunk_ids: list[str]
    mapped_entity_ids: list[str]
    mapped_relationship_ids: list[str]
    ranked_evidence: list[RetrievedEvidence]
    retrieval_parameters: dict[str, Any]


class ScenarioContext(StrictModel):
    """Compact Stage 3 context package."""

    story_id: str
    requirement_ids: list[str]
    story_text: str
    acceptance_criteria: list[AcceptanceCriterion]
    source_req_id: str | None
    source_req_id_type: SourceRequirementType
    authoritative_evidence_chunk_ids: list[str]
    supporting_entity_ids: list[str]
    supporting_relationship_ids: list[str]
    ranked_evidence: list[RetrievedEvidence]
    retrieval_parameters: dict[str, Any]


class StageRequest(StrictModel):
    """Project/run selector for downstream stages."""

    project_name: str
    run_id: str
    reasoning_provider: str | None = None
    embedding_provider: str | None = None


class ArtifactResult(StrictModel):
    """Generic completed-stage result."""

    project: str
    run_id: str
    artifact_path: Path
    item_ids: list[str]


class KnowledgeGraphReadiness(StrictModel):
    """Project-scoped readiness gate."""

    project: str
    status: Literal["building", "ready", "failed"]
    build_run_id: str
    failure_reason: str | None
    updated_at: datetime = Field(default_factory=utc_now)


class CoverageSummary(StrictModel):
    """Current project/run coverage summary."""

    project: str
    run_id: str
    requirement_count: int
    story_count: int
    scenario_count: int
    requirements_with_stories: int
    stories_with_scenarios: int


class ProgressItem(StrictModel):
    """Per-anchor terminal coverage status for a diagnostic progress report."""

    anchor_id: str
    status: Literal["generated", "no_story", "no_scenario"]
    candidate_count: int


class ProgressReport(StrictModel):
    """Diagnostic Stage 2/3 progress artifact. Never the recovery authority."""

    stage: Literal["user_story", "test_scenario"]
    project: str
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    items: list[ProgressItem]


__all__ = [
    "AcceptanceCriterion",
    "ArtifactResult",
    "CanonicalEntity",
    "CanonicalRelationship",
    "CanonicalRequirement",
    "CanonicalTestScenario",
    "CanonicalUserStory",
    "ChunkLayout",
    "ChunkManifest",
    "CoverageSummary",
    "EntityMention",
    "Evidence",
    "IngestionRequest",
    "IngestionResult",
    "KnowledgeGraphReadiness",
    "LLMEntityCandidate",
    "LLMRelationshipCandidate",
    "LLMRequirementCandidate",
    "LLMTestScenarioCandidate",
    "LLMTestScenarioDraft",
    "LLMUserStoryCandidate",
    "LLMUserStoryDraft",
    "ManifestChunk",
    "ParsedBlock",
    "ProgressItem",
    "ProgressReport",
    "RequirementChunkResult",
    "RequirementDiscoveryChunkResponse",
    "RequirementEntityRelationshipMap",
    "RequirementMapEntry",
    "RequirementsArtifact",
    "RetrievedEvidence",
    "ScenarioContext",
    "SourceProvenance",
    "StageRequest",
    "StoryContext",
    "StrictModel",
    "TestScenarioGenerationDraft",
    "TestScenarioGenerationResponse",
    "TestScenariosArtifact",
    "Traceability",
    "UserStoriesArtifact",
    "UserStoryGenerationDraft",
    "UserStoryGenerationResponse",
    "canonical_checksum",
    "utc_now",
]
