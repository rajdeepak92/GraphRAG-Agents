"""Convert validated temporary LLM output into permanent records."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.identifiers import fact_id, requirement_id
from multi_agentic_graph_rag.domain.schemas import (
    RequirementArtifact,
    RequirementDiscoveryOutput,
    VerifiedFact,
    VerifiedRequirement,
)


def build_requirement_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    version: str,
    source_checksum: str,
    discovery: RequirementDiscoveryOutput,
) -> RequirementArtifact:
    facts: list[VerifiedFact] = []
    temp_to_fact: dict[str, str] = {}
    seen_fact_text: set[str] = set()
    for ordinal, fact_candidate in enumerate(discovery.facts, start=1):
        if fact_candidate.text in seen_fact_text:
            continue
        seen_fact_text.add(fact_candidate.text)
        permanent_id = fact_id(project, version, fact_candidate.text, ordinal)
        temp_to_fact[fact_candidate.temp_id] = permanent_id
        facts.append(
            VerifiedFact(
                fact_id=permanent_id,
                text=fact_candidate.text,
                source_trace=fact_candidate.source_trace,
            )
        )

    requirements: list[VerifiedRequirement] = []
    seen_req_text: set[str] = set()
    for ordinal, requirement_candidate in enumerate(discovery.requirements, start=1):
        if requirement_candidate.statement in seen_req_text:
            continue
        seen_req_text.add(requirement_candidate.statement)
        requirements.append(
            VerifiedRequirement(
                requirement_id=requirement_id(
                    project, version, requirement_candidate.statement, ordinal
                ),
                statement=requirement_candidate.statement,
                requirement_type=requirement_candidate.requirement_type,
                priority=requirement_candidate.priority,
                fact_ids=[
                    temp_to_fact[temp_id]
                    for temp_id in requirement_candidate.fact_temp_ids
                    if temp_id in temp_to_fact
                ],
                source_trace=requirement_candidate.source_trace,
            )
        )

    return RequirementArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        version=version,
        source_checksum=source_checksum,
        facts=facts,
        requirements=requirements,
    )
