"""Generated JSON artifact IO and verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import RequirementArtifact


def write_requirement_artifact(
    artifact: RequirementArtifact,
    generated_root: Path,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    project_dir = generated_root / artifact.project / "requirements"
    project_dir.mkdir(parents=True, exist_ok=True)
    for existing in sorted(project_dir.glob(f"requirements_*_{artifact.version}.json")):
        try:
            current = verify_requirement_artifact(existing)
        except Exception:
            continue
        if current.document_version_id == artifact.document_version_id:
            path = existing
            break
    else:
        path = project_dir / f"requirements_{timestamp}_{artifact.version}.json"
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path


def verify_requirement_artifact(path: Path) -> RequirementArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RequirementArtifact.model_validate(data)
