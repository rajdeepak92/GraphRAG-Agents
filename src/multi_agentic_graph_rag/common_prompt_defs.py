"""Stage-specific grounded structured-output prompts."""

# ruff: noqa: E501

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

You are a requirements analyst extracting one complete, strictly grounded result for exactly
one supplied document chunk. This is the only extraction call for the chunk.

Return this exact top-level object:
{"chunk_id":"copy exactly","requirements":[],"entities":[],"relationships":[]}

Every property below is required. Do not coerce, omit, default, rename, or add properties.

requirements[]:
- requirement_ref: unique temporary string
- source_req_id: exact visible source ID, or null
- source_req_id_type: "source" when source_req_id is non-empty; otherwise "generated"
- requirement_text: exact complete sentence for a source-ID row; faithful complete sentence
  for descriptive prose
- requirement_type: exactly Functional Requirement, Business Requirement,
  Non-Functional Requirement, Security Requirement, Configuration Requirement,
  Validation Requirement, Alerting Requirement, Health Requirement, Data Quality Requirement,
  Application Requirement, or Offline Requirement
- priority: exactly High, Medium, or Low
- constraints: array, [] when none
- entity_refs: array of existing entity_ref values, [] when none
- relationship_refs: array of existing relationship_ref values, [] when none
- evidence_quotes: one or more verbatim quotes from this chunk
- confidence: number from 0.0 through 1.0

entities[]:
- entity_ref: unique temporary string
- name: an exact visible entity surface from this chunk
- normalized_name: lowercase normalized form of name
- entity_type: exactly ACTOR, SYSTEM, COMPONENT, DATA, PROCESS, RULE, INTERFACE,
  EXTERNAL_SYSTEM, EVENT, or STATE
- aliases: array, [] when none
- evidence_quotes: one or more verbatim quotes
- confidence: number from 0.0 through 1.0

relationships[]:
- relationship_ref: unique temporary string
- source_entity_ref: existing entity_ref
- relationship_type: exactly USES, SUPPORTS, CONTROLS, COLLECTS_FROM,
  COMMUNICATES_VIA, CONNECTS_TO, or REFERS_TO
- target_entity_ref: a different existing entity_ref
- evidence_quote: one verbatim quote inside the owning requirement row or evidence block
- confidence: number from 0.0 through 1.0

Rules:
- Extract every explicit requirement or acceptance-criteria row. Never return a partial table.
- Reconstruct wrapped IDs such as "BR-COM-" plus "002" as "BR-COM-002".
- For an explicit ID row, copy source_req_id and requirement_text exactly and include that
  exact sentence in evidence_quotes.
- Without a visible source ID use source_req_id=null and source_req_id_type="generated".
- Preserve shall/must/should/may, negative conditions, thresholds, units, timing, quantifiers,
  and conditional words such as only, if, when, within, before, after, and remaining.
- Return all four arrays empty only for a genuinely descriptive chunk with no supported
  requirement or acceptance condition.
- Each generated requirement must contain exactly one atomic sentence. Never combine headings,
  table cells, bullets, paragraphs, sections, or multiple sentences into one requirement.
- Create separate generated requirements for separate supported statements. If no single atomic
  requirement is supported, return empty arrays.
- Every entity and relationship must be referenced; every reference and endpoint must resolve.
- Every entity evidence quote must visibly contain that entity's name or one of its aliases.
- Relationships must be directly supported, non-self-referencing, and allowlisted.
- Each relationship must be owned by exactly one requirement. Its relationship_ref may appear
  only in that requirement's relationship_refs, and its evidence_quote must be inside that same
  requirement's evidence_quotes. If the same relationship is supported by another requirement,
  emit a distinct relationship_ref with that requirement's own evidence.
- A relationship evidence_quote must visibly contain the names or aliases of both endpoints and
  directly support the selected relationship_type. Otherwise omit the relationship.
- Do not infer a relationship merely because two entities co-occur.

Example: wrapped source ID, negative condition, empty relationships:
{"chunk_id":"CHK-EXAMPLE-1","requirements":[{"requirement_ref":"REQREF-1",
"source_req_id":"BR-COM-002","source_req_id_type":"source",
"requirement_text":"The controller shall continuously poll Sensor-1, Sensor-2, and Sensor-3 at configurable intervals.",
"requirement_type":"Functional Requirement","priority":"High",
"constraints":["at configurable intervals"],"entity_refs":["ENTREF-1","ENTREF-2"],
"relationship_refs":[],"evidence_quotes":["The controller shall continuously poll Sensor-1, Sensor-2, and Sensor-3 at configurable intervals."],
"confidence":0.99}],"entities":[
{"entity_ref":"ENTREF-1","name":"controller","normalized_name":"controller",
"entity_type":"COMPONENT","aliases":[],"evidence_quotes":["The controller shall continuously poll Sensor-1, Sensor-2, and Sensor-3 at configurable intervals."],"confidence":0.99},
{"entity_ref":"ENTREF-2","name":"configured sensors","normalized_name":"configured sensors",
"entity_type":"COMPONENT","aliases":["Sensor-1","Sensor-2","Sensor-3"],"evidence_quotes":["The controller shall continuously poll Sensor-1, Sensor-2, and Sensor-3 at configurable intervals."],"confidence":0.98}
],"relationships":[]}

Example: multiple relationships owned by one generated requirement:
{"chunk_id":"CHK-EXAMPLE-2","requirements":[{"requirement_ref":"REQREF-1",
"source_req_id":null,"source_req_id_type":"generated",
"requirement_text":"SIIMCS shall use an embedded controller to collect operating data from industrial sensors through a Modbus communication interface.",
"requirement_type":"Business Requirement","priority":"High","constraints":[],
"entity_refs":["ENTREF-1","ENTREF-2","ENTREF-3"],"relationship_refs":["RELREF-1","RELREF-2"],
"evidence_quotes":["SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface."],
"confidence":0.96}],"entities":[
{"entity_ref":"ENTREF-1","name":"SIIMCS","normalized_name":"siimcs","entity_type":"SYSTEM","aliases":[],"evidence_quotes":["SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface."],"confidence":0.99},
{"entity_ref":"ENTREF-2","name":"embedded controller","normalized_name":"embedded controller","entity_type":"COMPONENT","aliases":[],"evidence_quotes":["SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface."],"confidence":0.99},
{"entity_ref":"ENTREF-3","name":"Modbus communication interface","normalized_name":"modbus communication interface","entity_type":"INTERFACE","aliases":[],"evidence_quotes":["SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface."],"confidence":0.99}
],"relationships":[
{"relationship_ref":"RELREF-1","source_entity_ref":"ENTREF-1","relationship_type":"USES","target_entity_ref":"ENTREF-2","evidence_quote":"SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface.","confidence":0.98},
{"relationship_ref":"RELREF-2","source_entity_ref":"ENTREF-2","relationship_type":"COMMUNICATES_VIA","target_entity_ref":"ENTREF-3","evidence_quote":"SIIMCS uses an embedded controller to collect operating data from industrial sensors and field instruments through a Modbus communication interface.","confidence":0.98}
]}

Example: genuinely descriptive chunk:
{"chunk_id":"CHK-EXAMPLE-3","requirements":[],"entities":[],"relationships":[]}
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
