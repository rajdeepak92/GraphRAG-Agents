"""Generated JSON artifact IO and verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import RequirementArtifact
from multi_agentic_graph_rag.observability.logging import RunLogger


def write_requirement_artifact(
    artifact: RequirementArtifact,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    path = run_dir / "requirements.json"
    if logger is not None:
        logger.debug(
            "Writing requirement artifact for {document_version_id} to {path}",
            step="write_requirement_artifact",
            document_version_id=artifact.document_version_id,
            path=str(path),
            requirement_count=len(artifact.requirements),
            fact_count=len(artifact.facts),
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=run_dir,
        prefix=path.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(artifact.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_name = Path(handle.name)
    temp_name.replace(path)
    return path


def verify_requirement_artifact(path: Path) -> RequirementArtifact:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RequirementArtifact.model_validate(data)
