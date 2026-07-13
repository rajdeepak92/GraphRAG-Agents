"""Generated JSON artifact IO and verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirementsArtifact,
    RequirementArtifact,
    RequirementIdentityResolutionArtifact,
    TestScenarioArtifact,
    UserStoryArtifact,
)
from multi_agentic_graph_rag.observability.logging import RunLogger


def write_canonical_requirements_artifact(
    artifact: CanonicalRequirementsArtifact,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = run_dir / "requirements.json"
    if logger is not None:
        logger.debug(
            "Writing canonical requirements for {document_version_id} to {path}",
            step="write_canonical_requirements_artifact",
            document_version_id=artifact.document_version_id,
            path=str(path),
            requirement_count=len(artifact.requirements),
        )
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def write_requirement_identity_resolution_artifact(
    artifact: RequirementArtifact,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = run_dir / "identity_resolution.json"
    payload = RequirementIdentityResolutionArtifact(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        generated_at=artifact.generated_at,
        resolutions=artifact.identity_resolutions,
    )
    if logger is not None:
        logger.debug(
            "Writing requirement identity-resolution audit to {path}",
            step="write_requirement_identity_resolution_artifact",
            path=str(path),
            resolution_count=len(payload.resolutions),
        )
    _atomic_write_json(path, payload.model_dump(mode="json"))
    return path


def write_user_story_artifact(
    artifact: UserStoryArtifact,
    out_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = out_dir / "user_stories.json"
    if logger is not None:
        logger.debug(
            "Writing user-story artifact for {document_version_id} to {path}",
            step="write_user_story_artifact",
            document_version_id=artifact.document_version_id,
            path=str(path),
            story_count=len(artifact.stories),
            requirement_count=len({row.requirement_id for row in artifact.traceability}),
        )
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def write_test_scenario_artifact(
    artifact: TestScenarioArtifact,
    out_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = out_dir / "test_scenarios.json"
    if logger is not None:
        logger.debug(
            "Writing test-scenario artifact for {document_version_id} to {path}",
            step="write_test_scenario_artifact",
            document_version_id=artifact.document_version_id,
            path=str(path),
            scenario_count=len(artifact.scenarios),
            story_count=len({row.story_id for row in artifact.traceability}),
            requirement_count=len({row.requirement_id for row in artifact.traceability}),
        )
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def verify_requirement_artifact(
    path: Path,
) -> RequirementArtifact | CanonicalRequirementsArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("artifact_schema_version") == "5.0-requirements":
        return CanonicalRequirementsArtifact.model_validate(data)
    return RequirementArtifact.model_validate(data)


def verify_user_story_artifact(path: Path) -> UserStoryArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return UserStoryArtifact.model_validate(data)


def verify_test_scenario_artifact(path: Path) -> TestScenarioArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TestScenarioArtifact.model_validate(data)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_name = Path(handle.name)
    temp_name.replace(path)
