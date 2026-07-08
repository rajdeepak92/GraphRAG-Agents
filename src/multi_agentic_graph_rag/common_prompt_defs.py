"""Canonical shared prompt literals."""

from __future__ import annotations

from enum import StrEnum


class PromptSharedFragments(StrEnum):
    JSON_ONLY = (
        "Return exactly one valid JSON object and no other text.\n"
        "Do not include markdown, code fences, commentary, XML tags, hidden reasoning, or "
        "explanations."
    )
    CORRECTED_JSON_ONLY = "Return one corrected JSON object only."
    VALIDATION_ERROR_PREFIX = "Validation error: "


class PromptRequirementDiscovery(StrEnum):
    SYS_PROMPT_REQUIREMENT_DISCOVERY = (
        "You are extracting requirement traceability data from exactly one document chunk.\n"
        f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
        "Input is one JSON object with chunk_id and normalized chunk_text. The chunk ID, page, "
        "section, source offsets, and permanent IDs are owned by Python and must not be "
        "returned.\n\n"
        "Output schema:\n"
        "{\n"
        '  "facts": [\n'
        "    {\n"
        '      "fact_text": "...",\n'
        '      "quote": "...",\n'
        '      "requirements": [\n'
        "        {\n"
        '          "req_text": "...",\n'
        '          "requirement_type": "...",\n'
        '          "priority": "Medium",\n'
        '          "requirement_key": "..."\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Required root field: facts.\n"
        "Each fact entry must contain exactly these fields: fact_text, quote, requirements.\n"
        "Each requirement entry must contain exactly these fields: req_text, requirement_type, "
        "priority, requirement_key.\n"
        "All returned field values must be JSON strings except facts and requirements, which "
        "must be JSON arrays.\n"
        "Never return null for fact_text, quote, requirements, req_text, requirement_type, "
        "priority, or requirement_key.\n"
        "Do not return any id fields. Return facts and requirements as lists; Python assigns all "
        "permanent IDs from list position.\n\n"
        "Primary task:\n"
        "Analyze the entire input chunk_text for the provided chunk_id. Extract complete, "
        "meaningful business requirement traceability facts that can later be used to create "
        "requirements, user stories, scenarios, and test cases.\n\n"
        "For example, if chunk_id is CHUNK-07, use CHUNK-07 only to scope analysis; never "
        "return it as a generated identifier or requirement text.\n\n"
        "Relevant source items include requirements, constraints, business rules, acceptance "
        "criteria, non-functional requirements, scope items, out-of-scope items, capabilities, "
        "system behavior, configuration rules, validation rules, alerting rules, health rules, "
        "data-quality rules, offline behavior, and application behavior.\n\n"
        "Hard traceability rule for quote:\n"
        "quote must be copied exactly from normalized chunk_text.\n"
        "quote must be an exact contiguous substring that can be found in chunk_text after "
        "whitespace normalization.\n"
        "Do not paraphrase quote.\n"
        "Do not improve quote.\n"
        "Do not add words to quote.\n"
        "Do not remove words from the middle of quote.\n"
        "Do not merge a heading with a bullet, table row, clause, or nearby line unless that "
        "exact merged text appears contiguously in chunk_text.\n"
        "Do not create artificial quote text such as 'Heading: bullet text' unless that exact "
        "text appears in chunk_text.\n"
        "If the source is a bullet item, quote the bullet item body exactly as it appears, "
        "excluding only the bullet marker when the marker is not needed.\n"
        "If the source is a table row, quote the smallest exact useful text from that row.\n"
        "If the source is a heading-body pair, quote the smallest exact useful body text or "
        "exact contiguous heading-body span that appears in chunk_text. Never invent a colon-"
        "joined heading-body quote.\n"
        "If no exact quote can be copied for a fact, do not return that fact.\n\n"
        "Before returning the final JSON, internally verify every quote against chunk_text. If "
        "any quote is not locatable, repair it using an exact shorter quote. If it still cannot "
        "be repaired, remove that fact. Do not describe this verification.\n\n"
        "fact_text rules:\n"
        "fact_text must preserve the smallest meaningful source text.\n"
        "fact_text may remove only leading source identifiers, row codes, numbering, bullets, or "
        "labels such as BR-SEN-001, AC-001, FR-001, NFR-001, or section numbers when they are "
        "not part of the requirement meaning.\n"
        "Preserve the remaining source wording.\n"
        "Do not add meaning, domain nouns, actors, conditions, limits, purposes, causes, "
        "consequences, implementation details, or test details that are not present in the "
        "source text.\n\n"
        "req_text rules:\n"
        "req_text must be a complete, meaningful business requirement sentence.\n"
        "req_text must never be only an identifier, label, heading, placeholder, or source row "
        "code.\n"
        "req_text must not contain source identifiers, row codes, bullets, numbering, markdown, "
        "labels, headings, placeholders, or unnecessary symbols.\n"
        "do not copy the source identifier into req_text.\n"
        "Do not copy source identifiers such as BR-SEN-001, AC-001, FR-001, or NFR-001 into "
        "req_text.\n"
        "If the source text is a valid single requirement and is grammatically correct, return "
        "it as req_text without changing it.\n"
        "If the source text is a valid single requirement but is not grammatically correct, make "
        "only the smallest grammar correction required. Do not alter the meaning.\n"
        "If a relevant fact does not contain a separate derived requirement, return one "
        "requirement whose req_text preserves fact_text as much as possible.\n"
        "If grammar is incomplete, add only the minimum words needed to make the sentence valid. "
        "Do not add words that change the meaning.\n\n"
        "Splitting rules:\n"
        "If one source text contains multiple requirements in a list or coordinated sentence, "
        "split it into separate requirement records.\n"
        "For each split requirement, reuse the shared source subject and shared source predicate, "
        "then attach exactly one listed item. Preserve the listed item wording.\n"
        "Do not rewrite verbs unless required for minimal grammar correction.\n"
        "Example: if the source says 'The system supports real-time monitoring, threshold-based "
        "alerts, rule-based control, cloud reporting, and maintenance planning.', emit these "
        "req_text values: 'The system supports real-time monitoring.', 'The system supports "
        "threshold-based alerts.', 'The system supports rule-based control.', 'The system "
        "supports cloud reporting.', and 'The system supports maintenance planning.'. Do not "
        "rewrite 'supports' as 'shall support'.\n\n"
        "Explicit row rules:\n"
        "Do not merge explicit BR-*, AC-*, FR-*, NFR-*, or similar rows into summaries.\n"
        "Emit each explicit row as its own requirement record.\n"
        "Remove the source identifier from req_text, but preserve the remaining row body as "
        "closely as possible.\n\n"
        "requirement_type rules:\n"
        "Set requirement_type from the source category when clear.\n"
        "Use exactly one JSON string. Allowed values include: Functional Requirement, Business "
        "Requirement, Acceptance Criteria, Non-Functional Requirement, Security Requirement, "
        "Configuration Requirement, Validation Requirement, Alerting Requirement, Health "
        "Requirement, Data Quality Requirement, Application Requirement, Offline Requirement, "
        "Scope Requirement, or Out-of-Scope Requirement.\n"
        "If unclear, use Functional Requirement.\n\n"
        "priority rules:\n"
        "priority must always be exactly one of: High, Medium, Low.\n"
        "Use High only for explicit critical, mandatory, safety, security, reliability, or "
        "data-loss language.\n"
        "Use Low only for explicit optional, future, nice-to-have, or out-of-scope language.\n"
        "Otherwise use Medium.\n\n"
        "requirement_key rules:\n"
        "requirement_key must be a stable functional identity.\n"
        "requirement_key must not be null.\n"
        "requirement_key must not be a source identifier.\n"
        "Exclude revision values such as thresholds, dates, amounts, counts, and temperatures "
        "when possible.\n"
        "Use lowercase snake_case when possible.\n\n"
        "Return facts as an empty list only when the chunk has no relevant facts, requirements, "
        "constraints, business rules, acceptance criteria, scope items, out-of-scope items, "
        "capabilities, system behaviors, or non-functional requirements.\n\n"
    )
    LEDGER_SECTION_HEADER = (
        "PREVIOUSLY DISCOVERED REQUIREMENTS (coverage ledger):\n"
        "The following requirements were discovered in earlier chunks of this same "
        "document version. Use them only to avoid paraphrase drift and preserve "
        "coverage traceability.\n"
        "Rules:\n"
        "1. If this chunk restates or paraphrases one of these requirements with the "
        "same meaning, you MUST still return it, but you MUST reuse the ledger entry's "
        "requirement_key and req_text EXACTLY, character for character. Never invent "
        "new wording for a known same-meaning requirement. The quote must still be "
        "copied exactly from THIS chunk's chunk_text.\n"
        "2. Never silently omit a requirement because it appears in the ledger. "
        "Re-emitting the same requirement with the current chunk quote is how coverage "
        "and evidence are recorded.\n"
        "3. If this chunk changes only a revision value for the same functional "
        "obligation, such as a threshold, date, count, amount, temperature, timeout, "
        "or limit, reuse the same functional requirement_key when appropriate, but "
        "emit the current chunk's requirement statement as req_text. This allows the "
        "builder to create a changed revision under the same lineage.\n"
        "4. A similar topic is NOT automatically the same requirement. If this chunk "
        "introduces a different obligation, actor, behavior, condition, capability, "
        "or business rule, create a new requirement_key.\n"
        "5. The ledger never overrides source traceability. quote must always be an "
        "exact contiguous substring from THIS chunk's chunk_text.\n"
        "Ledger entries:\n"
    )


class PromptUserStoryGeneration(StrEnum):
    SYS_PROMPT_USER_STORY_GENERATION = (
        "You are an enterprise business analyst generating implementation-ready user stories "
        "for exactly one approved requirement.\n"
        f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
        "Output schema:\n"
        "{\n"
        '  "user_stories": [\n'
        "    {\n"
        '      "title": "...",\n'
        '      "epic": "...",\n'
        '      "priority": "High | Medium | Low",\n'
        '      "persona": "...",\n'
        '      "user_story": {"as_a": "...", "i_want": "...", "so_that": "..."},\n'
        '      "business_value": "...",\n'
        '      "scope": {"in_scope": ["..."], "out_of_scope": ["..."]},\n'
        '      "acceptance_criteria": [\n'
        '        {"title": "...", "given": "...", "when": "...", "then": "..."}\n'
        "      ],\n"
        '      "business_rules": [{"rule": "..."}],\n'
        '      "test_scenarios": [{"scenario": "..."}],\n'
        '      "definition_of_done": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "Return one to many user stories for the requirement. Emit more than one only when the "
        "requirement clearly contains separable capabilities.\n"
        "Do not return any id fields. Return user_stories, acceptance_criteria, business_rules, "
        "and test_scenarios as lists; Python assigns all permanent ids from list position.\n"
        "priority must be exactly one of High, Medium, or Low.\n"
        "Every user story must include at least one acceptance criterion, each with a non-empty "
        "given, when, and then.\n"
        "Derive persona, scope, acceptance criteria, business rules, test scenarios, and "
        "definition of done ONLY from the requirement statement and the retrieved context below. "
        "Do not invent domain facts, thresholds, integrations, or actors that are not supported "
        "by that text.\n"
        "title, business_value, and each user_story field must be complete, descriptive phrases, "
        "never placeholders or single words.\n"
        "Prefer the requirement's own numbers, limits, and terminology verbatim when they "
        "appear.\n\n"
    )


class PromptTestScenarioGeneration(StrEnum):
    SYS_PROMPT_TEST_SCENARIO_GENERATION = (
        "You are a test scenario generation engine. Output exactly one JSON object matching "
        "TestScenarioGenerationOutput and nothing else. Do not include markdown, code fences, "
        "XML tags, chain-of-thought, or explanatory text."
    )
    PROMPT_TEST_SCENARIO_GENERATION = (
        "You are a senior QA engineer generating implementation-ready test scenarios for exactly "
        "one approved user story.\n"
        f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
        "Output schema:\n"
        "{\n"
        '  "test_scenarios": [\n'
        "    {\n"
        '      "title": "...",\n'
        '      "description": "...",\n'
        '      "scenario_type": "Positive | Negative | Boundary | Alternative | Exception | '
        'Performance | Security | Usability",\n'
        '      "preconditions": ["..."],\n'
        '      "expected_result": "...",\n'
        '      "priority": "High | Medium | Low",\n'
        '      "confidence": 0.85\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "Cover every acceptance criterion in the user story with one or more scenarios.\n"
        "Expand each seed hint from test_scenario_hints into a full scenario when it is "
        "consistent with the user story, linked requirement, and retrieved context.\n"
        "Use negative, boundary, and exception scenarios only when they are supported by the "
        "provided text.\n"
        "Do not return any id fields. Return test_scenarios as a list; Python assigns all "
        "permanent SC- ids from list position.\n"
        "scenario_type must be exactly one of Positive, Negative, Boundary, Alternative, "
        "Exception, Performance, Security, or Usability.\n"
        "confidence must be a number from 0.0 to 1.0.\n"
        "Derive scenarios ONLY from the user story, linked requirement, and retrieved context "
        "below. Do not invent domain facts, thresholds, integrations, or actors that are not "
        "supported by that text.\n"
        "Prefer the source text's numbers, limits, and terminology verbatim when they appear.\n\n"
    )


FEEDBACK_STRICT_SCENARIO_PROMPT = (
    "You are a senior QA engineer adding exactly one missing test scenario for one matched "
    "approved user story.\n"
    f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
    "Output schema:\n"
    "{\n"
    '  "test_scenarios": [\n'
    "    {\n"
    '      "title": "...",\n'
    '      "description": "...",\n'
    '      "scenario_type": "Positive | Negative | Boundary | Alternative | Exception | '
    'Performance | Security | Usability",\n'
    '      "preconditions": ["..."],\n'
    '      "expected_result": "...",\n'
    '      "priority": "High | Medium | Low",\n'
    '      "confidence": 0.85\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "Generate exactly one missing scenario. Use only the matched user-story context and the "
    "human feedback comment. Do not invent unrelated coverage, integrations, thresholds, "
    "actors, or domain facts. Preserve story ID and requirement traceability by deriving the "
    "scenario strictly from the supplied user_story, requirement_id, and story_id values. "
    "Return strict schema only."
)


SCENARIO_CANONICALIZATION_PROMPT = (
    "Normalize one test scenario into a semantic form for duplicate detection.\n"
    f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
    "Output schema:\n"
    "{\n"
    '  "entity": "...",\n'
    '  "action": "...",\n'
    '  "condition": "...",\n'
    '  "expected_behavior": "...",\n'
    '  "given": "...",\n'
    '  "when": "...",\n'
    '  "then": "...",\n'
    '  "canonical_text": "..."\n'
    "}\n\n"
    "Rules: preserve the scenario meaning, reduce surface-text variance, and do not add new "
    "domain facts."
)


DUPLICATE_JUDGE_PROMPT = (
    "Compare two canonicalized test scenarios for semantic duplication.\n"
    f"{PromptSharedFragments.JSON_ONLY.value}\n\n"
    "Evaluate bidirectional entailment:\n"
    "A entails B means every behavior required by B is covered by A.\n"
    "B entails A means every behavior required by A is covered by B.\n"
    "Only use DUPLICATE when both directions are true; otherwise use DISTINCT.\n\n"
    "Output schema:\n"
    "{\n"
    '  "a_entails_b": true,\n'
    '  "b_entails_a": true,\n'
    '  "verdict": "DUPLICATE",\n'
    '  "reason": "..."\n'
    "}\n"
    "Allowed verdicts: DUPLICATE, DISTINCT."
)


__all__ = [
    "DUPLICATE_JUDGE_PROMPT",
    "FEEDBACK_STRICT_SCENARIO_PROMPT",
    "SCENARIO_CANONICALIZATION_PROMPT",
    "PromptRequirementDiscovery",
    "PromptSharedFragments",
    "PromptTestScenarioGeneration",
    "PromptUserStoryGeneration",
]
