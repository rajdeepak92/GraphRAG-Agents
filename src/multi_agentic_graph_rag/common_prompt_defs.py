"""Stage-specific grounded structured-output prompts."""

from enum import StrEnum

_COMMON = """
Return exactly one JSON object conforming to the supplied schema.
Return JSON only. Include every required property and no additional properties.
Use [] for a valid empty collection and null only where the schema permits it.
Never generate permanent IDs, status, checksum, timestamps, project metadata, or run metadata.
Use only supplied text and context. Do not use outside knowledge.
Copy evidence quotes verbatim. Prefer an empty result over unsupported output.
Preserve project and run isolation.
""".strip()


class PromptSharedFragments(StrEnum):
    """Small provider-adapter repair fragments."""

    CORRECTED_JSON_ONLY = "Return one corrected JSON object only."
    VALIDATION_ERROR_PREFIX = "Validation error: "


class PromptRequirementDiscovery(StrEnum):
    """Combined Stage 1.2 prompt."""

    SYSTEM = (
        _COMMON
        + """

Produce requirements, entities, and semantic relationships for exactly one supplied chunk.
This is the only extraction call for the chunk.

Requirements:
- Return every supported obligation, constraint, rule, behavior, quality, or acceptance condition.
- Return requirements=[] when none exist; then entities and relationships must also be empty.
- A source requirement requires an explicit source ID visibly associated with its sentence.
- Copy source_req_id and source requirement_text exactly. Never normalize or add prefixes.
- Otherwise set source_req_id=null and source_req_id_type="generated".
- Preserve modality, polarity, thresholds, timing, and conditions.
- Every requirement needs an exact evidence quote and unique requirement_ref.

Entities:
- Return only entities used by a requirement or relationship.
- Use only allowed entity types and exact evidence.
- Do not return filler words, quantities, pronouns, or unrelated surrounding concepts.

Relationships:
- A requirement may have zero, one, or many relationships.
- Use only USES, SUPPORTS, CONTROLS, COLLECTS_FROM, COMMUNICATES_VIA,
  CONNECTS_TO, or REFERS_TO.
- Both endpoints must be returned entity_refs.
- Evidence must support direction and meaning.
- Never create fallback, deterministic, guessed, self, or co-occurrence-only relationships.
- Every relationship must be referenced by at least one requirement.
"""
    )


class PromptUserStoryGeneration(StrEnum):
    """Stage 2 story prompt."""

    SYSTEM = (
        _COMMON
        + """

Generate grounded user-story candidates for exactly one active canonical requirement.
Copy source_req_id and source_req_id_type unchanged.
Each story has one supported persona, one capability, one outcome, and at least one
Given/When/Then acceptance criterion. Preserve all thresholds, timing, conditions, polarity,
and security constraints. Reference only supplied chunk, entity, and relationship IDs.
Do not invent UI, APIs, storage, roles, workflows, or implementation details.
Return user_stories=[] when grounded generation is not possible.
"""
    )


class PromptTestScenarioGeneration(StrEnum):
    """Stage 3 scenario prompt."""

    SYSTEM = (
        _COMMON
        + """

Generate declarative behavioral test scenarios for exactly one active canonical story.
Copy source_req_id and source_req_id_type unchanged.
Generate only applicable Positive, Negative, Boundary, Alternative, Exception, Performance,
Security, or Usability scenarios. Do not force categories.
Each scenario verifies one observable behavior with required preconditions, one action, and one
deterministic expected result. Cover only supplied acceptance-criterion IDs. Preserve explicit
thresholds, timing, validation rules, and failure behavior. Do not invent data, environments,
integrations, messages, limits, UI steps, selectors, code, or automation instructions.
Reference only supplied requirement, criterion, chunk, entity, and relationship IDs.
Return test_scenarios=[] when grounded generation is not possible.
"""
    )


__all__ = [
    "PromptRequirementDiscovery",
    "PromptSharedFragments",
    "PromptTestScenarioGeneration",
    "PromptUserStoryGeneration",
]
