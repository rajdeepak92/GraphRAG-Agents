"""Generated JSON artifact IO and verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import (
    CompactRequirementArtifact,
    RequirementArtifact,
    TestScenarioArtifact,
    UserStoryArtifact,
)
from multi_agentic_graph_rag.observability.logging import RunLogger


def write_requirement_artifact(
    artifact: RequirementArtifact,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = run_dir / "requirements_full.json"
    if logger is not None:
        logger.debug(
            "Writing requirement artifact for {document_version_id} to {path}",
            step="write_requirement_artifact",
            document_version_id=artifact.document_version_id,
            path=str(path),
            requirement_count=len(artifact.requirements),
            fact_count=len(artifact.facts),
        )
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def write_compact_requirement_artifact(
    compact_artifact: CompactRequirementArtifact,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = run_dir / "requirements.json"
    occurrence_count = sum(
        len(occurrences) for occurrences in compact_artifact.requirements.values()
    )
    if logger is not None:
        logger.debug(
            "Writing compact requirement artifact for {document_version_id} to {path}",
            step="write_compact_requirement_artifact",
            document_version_id=compact_artifact.document_version_id,
            path=str(path),
            requirement_count=len(compact_artifact.requirements),
            occurrence_count=occurrence_count,
        )
    _atomic_write_json(path, compact_artifact.model_dump(mode="json"))
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
            requirement_count=len(artifact.coverage),
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
            story_count=len(artifact.coverage),
            requirement_count=len(artifact.requirement_coverage),
        )
    _atomic_write_json(path, artifact.model_dump(mode="json"))
    return path


def verify_requirement_artifact(path: Path) -> RequirementArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
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
