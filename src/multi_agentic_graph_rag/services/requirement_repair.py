"""Detect and repair polluted requirement lineages in stored artifacts.

The legacy scheme keyed a lineage on the LLM ``requirement_key`` (numbers masked),
so distinct obligations (Sensor-1/2/3, or a Business Requirement and an Acceptance
Criterion with similar wording) could be merged under one ``requirement_id``. This
service re-derives the deterministic identity signature for every stored revision
and splits any lineage that turns out to contain more than one signature — never
deleting evidence, and remapping revisions/evidence onto the corrected lineages.

Dry-run (the default in the CLI) only analyzes and reports; apply rewrites the
in-memory artifact deterministically and is idempotent (a clean artifact is a
no-op).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from multi_agentic_graph_rag.domain.identifiers import (
    is_requirement_uuid7,
    new_requirement_evidence_id,
    new_requirement_id,
    new_requirement_revision_id,
    requirement_delta_event_id,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirement,
    CanonicalRequirementEvidence,
    CanonicalRequirementsArtifact,
    RequirementArtifact,
    VerifiedRequirement,
    normalize_priority_label,
)
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    requirement_lineage_signature,
)


def _normalize_statement(value: str) -> str:
    """Normalize statement deterministically within the active scope.

    Args:
        value (str): Value required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True)
class LineageFinding:
    """Coordinate lineage finding behavior within the services boundary."""

    old_requirement_id: str
    signatures: tuple[str, ...]
    revision_remap: dict[str, str]  # revision_id -> corrected requirement_id


@dataclass
class RepairReport:
    """Coordinate repair report behavior within the services boundary."""

    project: str
    document_id: str
    total_requirements: int = 0
    total_lineages: int = 0
    polluted_lineages: int = 0
    findings: list[LineageFinding] = field(default_factory=list)
    id_remap: dict[str, str] = field(default_factory=dict)  # revision_id -> new requirement_id
    revision_id_remap: dict[str, str] = field(default_factory=dict)
    evidence_id_remap: dict[str, str] = field(default_factory=dict)
    ambiguous_cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert the value to dict without mutating its source.

        Returns:
            dict[str, object]: The typed result produced by the operation.
        """
        return {
            "project": self.project,
            "document_id": self.document_id,
            "total_requirements": self.total_requirements,
            "total_lineages": self.total_lineages,
            "polluted_lineages": self.polluted_lineages,
            "old_to_new_requirement_ids_by_revision": self.id_remap,
            "revision_id_remap": self.revision_id_remap,
            "evidence_id_remap": self.evidence_id_remap,
            "ambiguous_cases": self.ambiguous_cases,
            "impact_counts": {
                "requirement_revisions": len(self.id_remap),
                "revision_ids": len(self.revision_id_remap),
                "evidence_occurrences": len(self.evidence_id_remap),
                "artifacts": 1
                if (self.id_remap or self.revision_id_remap or self.evidence_id_remap)
                else 0,
            },
            "findings": [
                {
                    "old_requirement_id": finding.old_requirement_id,
                    "signatures": list(finding.signatures),
                    "revision_remap": finding.revision_remap,
                }
                for finding in self.findings
            ],
        }


@dataclass
class ProjectRepairReport:
    """Project-wide remap plan shared by PostgreSQL and local JSON repair."""

    project: str
    artifact_count: int
    requirement_id_by_revision: dict[str, str] = field(default_factory=dict)
    revision_id_remap: dict[str, str] = field(default_factory=dict)
    evidence_id_remap: dict[str, str] = field(default_factory=dict)
    ambiguous_cases: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Execute the changed operation within its declared architectural boundary.

        Returns:
            bool: The typed result produced by the operation.
        """
        return bool(
            self.requirement_id_by_revision or self.revision_id_remap or self.evidence_id_remap
        )

    def to_dict(self) -> dict[str, object]:
        """Convert the value to dict without mutating its source.

        Returns:
            dict[str, object]: The typed result produced by the operation.
        """
        return {
            "project": self.project,
            "artifact_count": self.artifact_count,
            "requirement_id_by_revision": self.requirement_id_by_revision,
            "revision_id_remap": self.revision_id_remap,
            "evidence_id_remap": self.evidence_id_remap,
            "ambiguous_cases": self.ambiguous_cases,
            "impact_counts": {
                "requirement_revisions": len(self.requirement_id_by_revision),
                "revision_ids": len(self.revision_id_remap),
                "evidence_occurrences": len(self.evidence_id_remap),
                "artifacts": self.artifact_count if self.changed else 0,
            },
        }


def analyze_canonical_project(
    project: str,
    payloads: list[dict[str, Any]],
) -> ProjectRepairReport:
    """Plan a deterministic project repair across all canonical artifacts.

    Exact normalized duplicates merge. Revisions with the same stable signature
    share a lineage. Any reused revision carrying incompatible semantics, or a
    repaired lineage with multiple non-equivalent active revisions, aborts apply.
    """
    artifacts = [CanonicalRequirementsArtifact.model_validate(payload) for payload in payloads]
    report = ProjectRepairReport(project=project, artifact_count=len(artifacts))
    rows = [row for artifact in artifacts for row in artifact.requirements]
    revision_semantics: dict[str, tuple[str, str, str]] = {}
    exact_lineages: dict[tuple[str, str], str] = {}
    signature_lineages: dict[str, str] = {}
    target_by_revision: dict[str, str] = {}
    target_revision_by_exact: dict[tuple[str, str], str] = {}

    for row in sorted(rows, key=lambda item: (item.requirement_id, item.revision_id)):
        normalized = _normalize_statement(row.requirement_text)
        family = row.requirement_type.strip().casefold()
        signature = row.semantic_signature or requirement_lineage_signature(
            row.requirement_text, row.requirement_type
        )
        semantics = (normalized, family, signature)
        previous = revision_semantics.setdefault(row.revision_id, semantics)
        if previous != semantics:
            report.ambiguous_cases.append(
                f"revision {row.revision_id} has incompatible canonical statements"
            )
            continue

        exact_key = (family, normalized)
        target_requirement_id = exact_lineages.get(exact_key) or signature_lineages.get(signature)
        if target_requirement_id is None:
            target_requirement_id = (
                row.requirement_id
                if is_requirement_uuid7(row.requirement_id)
                else new_requirement_id()
            )
        exact_lineages[exact_key] = target_requirement_id
        signature_lineages.setdefault(signature, target_requirement_id)
        target_by_revision[row.revision_id] = target_requirement_id

        canonical_revision = target_revision_by_exact.get(exact_key)
        if canonical_revision is None:
            canonical_revision = (
                row.revision_id
                if is_requirement_uuid7(row.revision_id)
                else new_requirement_revision_id()
            )
            target_revision_by_exact[exact_key] = canonical_revision
        if canonical_revision != row.revision_id:
            report.revision_id_remap[row.revision_id] = canonical_revision
        if target_requirement_id != row.requirement_id:
            report.requirement_id_by_revision[row.revision_id] = target_requirement_id
        for evidence in row.evidence:
            if not is_requirement_uuid7(evidence.evidence_id):
                report.evidence_id_remap.setdefault(
                    evidence.evidence_id, new_requirement_evidence_id()
                )

    active_by_target: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if row.status != "Active":
            continue
        target = target_by_revision.get(row.revision_id, row.requirement_id)
        normalized = _normalize_statement(row.requirement_text)
        target_revision = report.revision_id_remap.get(row.revision_id, row.revision_id)
        active_by_target[target].add((target_revision, normalized))
    for target, active in active_by_target.items():
        if len({revision for revision, _ in active}) > 1:
            report.ambiguous_cases.append(
                f"lineage {target} would have multiple active non-equivalent revisions"
            )
    return report


def remap_canonical_payload(payload: dict[str, Any], report: ProjectRepairReport) -> dict[str, Any]:
    """Apply a project plan to one schema-5 artifact without losing evidence."""
    if report.ambiguous_cases:
        raise ValueError("identity repair aborted: unresolved ambiguous historical identities")
    artifact = CanonicalRequirementsArtifact.model_validate(payload)
    grouped: dict[tuple[str, str], CanonicalRequirement] = {}
    for row in artifact.requirements:
        requirement_id = report.requirement_id_by_revision.get(row.revision_id, row.requirement_id)
        revision_id = report.revision_id_remap.get(row.revision_id, row.revision_id)
        evidence = [
            item.model_copy(
                update={
                    "evidence_id": report.evidence_id_remap.get(item.evidence_id, item.evidence_id)
                }
            )
            for item in row.evidence
        ]
        key = (requirement_id, revision_id)
        updated = row.model_copy(
            update={
                "requirement_id": requirement_id,
                "revision_id": revision_id,
                "evidence": evidence,
            }
        )
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = updated
        else:
            by_id = {item.evidence_id: item for item in [*existing.evidence, *evidence]}
            grouped[key] = existing.model_copy(
                update={"evidence": [by_id[value] for value in sorted(by_id)]}
            )
    return artifact.model_copy(
        update={
            "requirements": sorted(
                grouped.values(), key=lambda row: (row.requirement_id, row.revision_id)
            )
        }
    ).model_dump(mode="json")


def analyze_requirement_artifact(artifact: RequirementArtifact) -> RepairReport:
    """Detect lineages whose revisions carry more than one identity signature."""
    by_lineage: dict[str, list[VerifiedRequirement]] = defaultdict(list)
    for requirement in artifact.requirements:
        by_lineage[requirement.requirement_id].append(requirement)

    report = RepairReport(
        project=artifact.project,
        document_id=artifact.document_id,
        total_requirements=len(artifact.requirements),
        total_lineages=len(by_lineage),
    )
    for requirement_id, group in by_lineage.items():
        signatures: dict[str, str] = {}
        for requirement in group:
            signature = requirement_lineage_signature(
                requirement.statement, requirement.requirement_type
            )
            previous = signatures.get(requirement.revision_id)
            if previous is not None and previous != signature:
                report.ambiguous_cases.append(
                    f"revision {requirement.revision_id} has conflicting semantic signatures"
                )
            signatures[requirement.revision_id] = signature
        distinct = set(signatures.values())
        for signature in distinct:
            active_revisions = {
                requirement.revision_id
                for requirement in group
                if signatures[requirement.revision_id] == signature
                and requirement.status == "active"
            }
            if len(active_revisions) > 1:
                report.ambiguous_cases.append(
                    f"lineage {requirement_id} has multiple active revisions for one signature"
                )
        polluted = len(distinct) > 1
        if polluted:
            report.polluted_lineages += 1
        new_by_signature = {
            signature: (
                requirement_id
                if not polluted and is_requirement_uuid7(requirement_id)
                else new_requirement_id()
            )
            for signature in distinct
        }
        revision_remap = {
            revision_id: new_by_signature[signature]
            for revision_id, signature in signatures.items()
        }
        if polluted:
            report.findings.append(
                LineageFinding(
                    old_requirement_id=requirement_id,
                    signatures=tuple(sorted(distinct)),
                    revision_remap=revision_remap,
                )
            )
        report.id_remap.update(
            {
                revision_id: new_id
                for revision_id, new_id in revision_remap.items()
                if new_id != requirement_id
            }
        )
        for requirement in group:
            if not is_requirement_uuid7(requirement.revision_id):
                report.revision_id_remap.setdefault(
                    requirement.revision_id, new_requirement_revision_id()
                )
            for evidence in requirement.evidence:
                if not is_requirement_uuid7(evidence.evidence_id):
                    report.evidence_id_remap.setdefault(
                        evidence.evidence_id, new_requirement_evidence_id()
                    )
    split_ids = {finding.old_requirement_id for finding in report.findings}
    for event in artifact.delta_events:
        if (
            event.requirement_id in split_ids
            and not event.revision_id
            and not event.previous_revision_id
            and not event.superseded_by_revision_id
        ):
            report.ambiguous_cases.append(
                f"delta event {event.event_id} cannot be assigned after lineage split"
            )
    return report


def migrate_legacy_catalog_payload(payload: dict[str, object]) -> CanonicalRequirementsArtifact:
    """Explicitly migrate a legacy 4.0 occurrence catalog to canonical schema 5.0.

    Repeated occurrence rows become nested evidence. Polluted lineages split by
    semantic signature. A reused revision carrying conflicting semantics is
    ambiguous and aborts instead of being silently repaired.
    """
    if payload.get("artifact_schema_version") != "4.0-catalog":
        raise ValueError("expected a 4.0-catalog artifact")
    rows = payload.get("requirements")
    if not isinstance(rows, list):
        raise ValueError("legacy requirements must be a list")
    signatures_by_revision: dict[str, str] = {}
    requirement_ids: dict[tuple[str, str], str] = {}
    revision_ids: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("legacy requirement row must be an object")
        old_requirement_id = str(raw.get("requirement_uid", "")).strip()
        old_revision_id = str(raw.get("revision_id", "")).strip()
        statement = str(raw.get("requirement_text", "")).strip()
        requirement_type = str(raw.get("requirement_type", "")).strip()
        if not old_requirement_id or not old_revision_id or not statement or not requirement_type:
            raise ValueError("legacy requirement identity and semantic fields must be non-empty")
        signature = requirement_lineage_signature(statement, requirement_type)
        previous = signatures_by_revision.setdefault(old_revision_id, signature)
        if previous != signature:
            raise ValueError(
                f"ambiguous legacy revision {old_revision_id}: conflicting semantic signatures"
            )
        grouped[(old_requirement_id, old_revision_id)].append(raw)
        requirement_ids.setdefault((old_requirement_id, signature), new_requirement_id())
        revision_ids.setdefault(old_revision_id, new_requirement_revision_id())

    migrated: list[CanonicalRequirement] = []
    for (old_requirement_id, old_revision_id), occurrences in grouped.items():
        head = occurrences[0]
        statement = str(head["requirement_text"])
        requirement_type = str(head["requirement_type"])
        signature = requirement_lineage_signature(statement, requirement_type)
        migrated.append(
            CanonicalRequirement(
                requirement_id=requirement_ids[(old_requirement_id, signature)],
                revision_id=revision_ids[old_revision_id],
                source_req_id=(
                    str(head["source_req_id"]) if head.get("source_req_id") is not None else None
                ),
                id_generation_type=(
                    "source"
                    if str(head.get("id_generation_type", "generated")) == "source"
                    else "generated"
                ),
                confidence=float(str(head.get("confidence", 0.0))),
                requirement_text=statement,
                semantic_signature=signature,
                requirement_type=requirement_type,
                priority=normalize_priority_label(head.get("priority")),
                status=(
                    "Superseded"
                    if str(head.get("status", "Active")).lower() == "superseded"
                    else "Active"
                ),
                evidence=[
                    CanonicalRequirementEvidence(
                        evidence_id=new_requirement_evidence_id(),
                        document_version_id=str(payload.get("document_version_id", "")),
                        chunk_id=str(row.get("chunk_id", "")),
                        fact_ids=[str(row["fact_id"])] if row.get("fact_id") else [],
                        quote=str(row.get("quote", "")),
                        start_char=int(str(row.get("start_char", 0))),
                        end_char=int(str(row.get("end_char", 0))),
                        page=(int(str(row["page"])) if row.get("page") is not None else None),
                        section=(str(row["section"]) if row.get("section") is not None else None),
                        source_path=str(payload.get("source_path", "")),
                    )
                    for row in occurrences
                ],
            )
        )
    return CanonicalRequirementsArtifact(
        project=str(payload.get("project", "")),
        document_id=str(payload.get("document_id", "")),
        document_version_id=str(payload.get("document_version_id", "")),
        doc_version=str(payload.get("doc_version", "")),
        requirements=migrated,
    )


def canonicalize_legacy_repair_payload(
    payload: dict[str, Any],
) -> CanonicalRequirementsArtifact:
    """Migration-only reader for schema 5, legacy 4.0, or internal 2.1 payloads."""
    version = payload.get("artifact_schema_version")
    if version == "5.0-requirements":
        return CanonicalRequirementsArtifact.model_validate(payload)
    if version == "4.0-catalog":
        return migrate_legacy_catalog_payload(payload)
    if version == "2.1":
        from multi_agentic_graph_rag.services.requirement_builder import (
            build_canonical_requirements_artifact,
        )

        return build_canonical_requirements_artifact(RequirementArtifact.model_validate(payload))
    raise ValueError(f"unsupported legacy requirement artifact schema: {version!r}")


def apply_repair(artifact: RequirementArtifact, report: RepairReport) -> RequirementArtifact:
    """Return a corrected artifact: revisions moved onto their signature lineage.

    Idempotent and evidence-preserving. After a split, every corrected lineage is
    guaranteed at least one active revision (a distinct obligation that was wrongly
    superseded under the merged lineage is restored to active).
    """
    if report.ambiguous_cases:
        raise ValueError("identity repair aborted: unresolved ambiguous historical identities")
    if not report.id_remap and not report.revision_id_remap and not report.evidence_id_remap:
        return artifact

    repaired: list[VerifiedRequirement] = []
    for requirement in artifact.requirements:
        new_id = report.id_remap.get(requirement.revision_id, requirement.requirement_id)
        repaired.append(
            requirement.model_copy(
                update={
                    "requirement_id": new_id,
                    "revision_id": report.revision_id_remap.get(
                        requirement.revision_id, requirement.revision_id
                    ),
                    "evidence": [
                        evidence.model_copy(
                            update={
                                "evidence_id": report.evidence_id_remap.get(
                                    evidence.evidence_id, evidence.evidence_id
                                )
                            }
                        )
                        for evidence in requirement.evidence
                    ],
                }
            )
        )

    # Guarantee exactly one active revision per corrected lineage.
    by_lineage: dict[str, list[VerifiedRequirement]] = defaultdict(list)
    for requirement in repaired:
        by_lineage[requirement.requirement_id].append(requirement)
    normalized: list[VerifiedRequirement] = []
    for group in by_lineage.values():
        if not any(requirement.status == "active" for requirement in group):
            group = [group[0].model_copy(update={"status": "active"}), *group[1:]]
        normalized.extend(group)

    events = []
    for event in artifact.delta_events:
        identity_revision = (
            event.revision_id or event.previous_revision_id or event.superseded_by_revision_id or ""
        )
        requirement_id = report.id_remap.get(identity_revision, event.requirement_id)
        revision_id = report.revision_id_remap.get(event.revision_id or "", event.revision_id)
        previous_revision_id = report.revision_id_remap.get(
            event.previous_revision_id or "", event.previous_revision_id
        )
        superseded_by_revision_id = report.revision_id_remap.get(
            event.superseded_by_revision_id or "", event.superseded_by_revision_id
        )
        events.append(
            event.model_copy(
                update={
                    "event_id": requirement_delta_event_id(
                        event_type=event.event_type,
                        requirement_identifier=requirement_id,
                        revision_identifier=revision_id,
                        previous_revision_identifier=(
                            previous_revision_id or superseded_by_revision_id
                        ),
                        document_version_identifier=event.document_version_id,
                    ),
                    "requirement_id": requirement_id,
                    "revision_id": revision_id,
                    "previous_revision_id": previous_revision_id,
                    "superseded_by_revision_id": superseded_by_revision_id,
                    "evidence_ids": [
                        report.evidence_id_remap.get(value, value) for value in event.evidence_ids
                    ],
                }
            )
        )

    return artifact.model_copy(update={"requirements": normalized, "delta_events": events})
