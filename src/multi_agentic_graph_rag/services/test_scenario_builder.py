"""Canonicalize Stage 3 scenario candidates within provenance partitions."""

from __future__ import annotations

import re
from collections import defaultdict

from multi_agentic_graph_rag.domain.identifiers import new_scenario_id
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalTestScenario,
    LLMTestScenarioCandidate,
    TestScenariosArtifact,
    Traceability,
    canonical_checksum,
)

_SPACE = re.compile(r"\s+")


def build_test_scenarios_artifact(
    *,
    project: str,
    run_id: str,
    candidates: list[tuple[str, list[str], LLMTestScenarioCandidate]],
    existing: TestScenariosArtifact | None = None,
) -> TestScenariosArtifact:
    """Deduplicate scenarios only inside their inherited provenance partition."""
    existing_by_key: dict[tuple[str | None, str, str], CanonicalTestScenario] = {
        (
            scenario.source_req_id,
            scenario.source_req_id_type,
            _canonical_key(scenario),
        ): scenario
        for scenario in (existing.scenarios if existing is not None else [])
    }
    grouped: dict[
        tuple[str | None, str, str],
        list[tuple[str, list[str], LLMTestScenarioCandidate]],
    ] = defaultdict(list)
    for story_id, requirement_ids, candidate in candidates:
        grouped[
            (
                candidate.source_req_id,
                candidate.source_req_id_type,
                _canonical_key(candidate),
            )
        ].append((story_id, requirement_ids, candidate))
    scenarios: list[CanonicalTestScenario] = []
    for key, rows in grouped.items():
        representative = rows[0][2]
        prior = existing_by_key.get(key)
        scenarios.append(
            CanonicalTestScenario(
                scenario_id=prior.scenario_id if prior is not None else new_scenario_id(),
                story_ids=list(dict.fromkeys(story_id for story_id, _, _ in rows)),
                requirement_ids=list(
                    dict.fromkeys(
                        requirement_id
                        for _, requirement_ids, _ in rows
                        for requirement_id in requirement_ids
                    )
                ),
                source_req_id=representative.source_req_id,
                source_req_id_type=representative.source_req_id_type,
                title=representative.title,
                description=representative.description,
                scenario_type=representative.scenario_type,
                priority=representative.priority,
                preconditions=representative.preconditions,
                action=representative.action,
                expected_result=representative.expected_result,
                covered_acceptance_criterion_ids=list(
                    dict.fromkeys(
                        criterion_id
                        for _, _, candidate in rows
                        for criterion_id in candidate.covered_acceptance_criterion_ids
                    )
                ),
                confidence=max(candidate.confidence for _, _, candidate in rows),
                traceability=Traceability(
                    evidence_chunk_ids=list(
                        dict.fromkeys(
                            chunk_id
                            for _, _, candidate in rows
                            for chunk_id in candidate.evidence_chunk_ids
                        )
                    ),
                    entity_ids=list(
                        dict.fromkeys(
                            entity_id
                            for _, _, candidate in rows
                            for entity_id in candidate.supporting_entity_ids
                        )
                    ),
                    relationship_ids=list(
                        dict.fromkeys(
                            relationship_id
                            for _, _, candidate in rows
                            for relationship_id in candidate.supporting_relationship_ids
                        )
                    ),
                ),
            )
        )
    payload = TestScenariosArtifact.model_construct(
        project=project,
        run_id=run_id,
        checksum="",
        scenarios=scenarios,
    )
    return TestScenariosArtifact.model_validate(
        {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
    )


def _canonical_key(candidate: LLMTestScenarioCandidate | CanonicalTestScenario) -> str:
    values = (
        candidate.title,
        candidate.description,
        candidate.scenario_type,
        candidate.action,
        candidate.expected_result,
    )
    return "|".join(_SPACE.sub(" ", value).strip().casefold() for value in values)


__all__ = ["build_test_scenarios_artifact"]
