from __future__ import annotations

import json
from pathlib import Path

from multi_agentic_graph_rag.domain.chunks import ChunkManifest
from multi_agentic_graph_rag.domain.commands import IngestDocumentCommand
from multi_agentic_graph_rag.domain.documents import ParsedDocument
from multi_agentic_graph_rag.domain.facts import FactCandidate
from multi_agentic_graph_rag.domain.requirements import (
    DiscoveryBatch,
    DiscoveryResult,
    RequirementArtifact,
    RequirementCandidate,
)
from multi_agentic_graph_rag.domain.runs import IngestionRun, RunStep

MODELS = [
    IngestDocumentCommand,
    ParsedDocument,
    ChunkManifest,
    FactCandidate,
    RequirementCandidate,
    DiscoveryBatch,
    DiscoveryResult,
    RequirementArtifact,
    IngestionRun,
    RunStep,
]


def main() -> None:
    output_dir = Path("generated_artifacts/schema_drafts")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        schema = model.model_json_schema()
        output_path = output_dir / f"{model.__name__}.schema.json"
        output_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        print(f"PASS {model.__name__} -> {output_path}")


if __name__ == "__main__":
    main()
