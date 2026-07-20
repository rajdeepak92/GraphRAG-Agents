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
- Never extract from an "Out of Scope" or "Definition of Done" section, heading, table, row,
  bullet, or object. Skip such content completely: emit no requirement, entity, or relationship
  for it. When the whole chunk is Out of Scope or Definition of Done content, return all four
  arrays empty. This exclusion overrides every other extraction rule below.
- Extract every explicit requirement or acceptance-criteria row. Never return a partial table.
- Reconstruct wrapped IDs such as "BR-COM-" plus "002" as "BR-COM-002".
- For an explicit ID row, copy source_req_id and requirement_text exactly and include that
  exact sentence in evidence_quotes.
- source_requirement_rows maps every visible explicit source ID to its mandatory exact
  requirement_text and mandatory evidence quote. For source_req_id_type="source", copy the
  mapped value exactly into requirement_text and include that same mapped value as a complete
  evidence_quotes item; do not select a longer candidate containing the source ID or row text.
- Treat evidence_source_text as the canonical verbatim quote source and
  evidence_quote_candidates as the only allowed quote values. Every requirement
  evidence_quotes item, entity evidence_quotes item, and relationship evidence_quote must equal
  one complete evidence_quote_candidates entry exactly. Never construct an evidence quote from
  requirement_text or shorten a candidate.
- Preserve visible category labels, headings, prefixes, and suffixes inside an evidence quote
  exactly as they appear in the selected candidate. The normalized requirement_text may omit a
  label, but its evidence quote may not.
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
Do not emit source_req_id or source_req_id_type; provenance is assigned by the system.
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
Do not emit source_req_id or source_req_id_type; provenance is assigned by the system.
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


class PromptStage4CodeGeneration(StrEnum):
    """Strict Stage-4 planning, bundle, and bounded-repair prompts."""

    PLAN_SYSTEM = """
You are the reasoning brain for one frozen Stage-4 test case. Return exactly one JSON object
conforming to TestCaseImplementationPlan. Use only the supplied canonical scenario, complete
parent lineage, exact normalized test-data records, and READY framework snapshot context.

You decide the module, ordered scenario steps, criticality, reusable symbols, test variables,
and whether a missing test capability needs a wrapper or a genuinely test-only helper. Follow
this capability order: reuse production behavior unchanged; reuse existing test_lib behavior;
add a module-specific wrapper around production behavior; add a genuinely missing test-only
helper; otherwise leave the scenario unimplementable. Never propose a production-code edit.

Every planned path must be one of:
- tests/<module>/Tc<six-digit-id><PascalTitle>.py
- tests/<module>/__init__.py, create only when absent
- tests_robot/<module>/Tc<six-digit-id><PascalTitle>.robot
- test_lib/<module>/<module>_wrappers.py
- test_lib/<module>/<module>_helpers.py
- test_lib/<module>/__init__.py, create only when absent

Do not plan utils.py, common.py, misc.py, pytest tests, Git operations, real hardware execution,
or files outside those paths. The Python filename stem and class must match exactly. Do not
invent IPs, ports, credentials, thresholds, device IDs, timing values, or any other test value.
Unresolved secret:// references are valid exact inputs and must remain unresolved.
""".strip()

    BUNDLE_SYSTEM = """
You are implementing one approved TestCaseImplementationPlan. Return exactly one JSON object
conforming to GeneratedPatchBundle and complete contents for every planned file. Copy the
supplied scenario_id, permanent six-digit tc_id, paths, and plan checksum exactly. Set each
content_hash to sha256:<lowercase SHA-256 of the exact UTF-8 content>.

Never edit production framework code. Existing production functions are master. In an existing
test_lib file, preserve every pre-existing import, assignment, decorator, signature, function
body, and class body byte-for-byte; only append a uniquely named function, preferably using
function-local imports. Put wrappers and helpers only in the matching module-specific test_lib
files. Reuse existing behavior before adding anything.

The Python module and class must be Tc<six-digit-id><PascalTitle>. Import and construction must
be side-effect free. Keep framework, test_lib, and third-party imports function-local; only
side-effect-free standard-library declarations may be imported at module level. Define
test_variables plus Boolean test_setup(), execute_test(),
test_teardown(), and run_test(). Setup failure exits immediately without execution or teardown.
Setup and teardown each log their start and result, catch unexpected exceptions, and return False
for every exception. After execution is entered, teardown always runs. Each scenario step is one logged call;
normalize and append every completed result. Non-critical failures continue but make execution
fail; critical failures and unexpected exceptions return False immediately. Normal execution
returns bool(step_results) and all(step_results). run_test() returns execution_ok and
teardown_ok. A guarded __main__ exits 0 for True and 1 for False. Never initialize or execute
real hardware/network/domain behavior at module import or construction time.

The Robot suite imports the matching Python module, calls only Run Test, and asserts the returned
Boolean with Should Be True. It must not call setup, execution, or teardown separately. Do not
generate pytest-style target tests. Use only exact supplied test-data values. Keep secret://
references unresolved in code and logs. Never invoke Git.
""".strip()

    REPAIR_SYSTEM = """
Repair only the current Stage-4 GeneratedPatchBundle. Return exactly one JSON object conforming
to GeneratedPatchBundle. Keep the selected provider, implementation plan, permanent tc_id,
module, and permitted paths fixed. Use only the bounded validation diagnostics and current file
contents supplied by the caller.

Correct the reported policy, AST, syntax, lifecycle, import, Robot dry-run, traceability,
exact-data, or checksum defect without modifying production code or any pre-existing test_lib
behavior. Preserve unresolved secret:// references. Do not add unrelated files, invent data,
run real tests or hardware, use pytest target style, or invoke Git. Recompute every content_hash
from the exact repaired UTF-8 content.
""".strip()


__all__ = [
    "PromptRequirementDiscovery",
    "PromptSharedFragments",
    "PromptStage4CodeGeneration",
    "PromptTestScenarioGeneration",
    "PromptUserStoryGeneration",
]
