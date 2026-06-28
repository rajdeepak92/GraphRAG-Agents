from __future__ import annotations

from multi_agentic_graph_rag.infrastructure.postgres.identifiers import (
    format_fact_id,
    format_requirement_id,
)
from multi_agentic_graph_rag.infrastructure.postgres.models import (
    ChunkRow,
    DocumentRow,
    DocumentVersionRow,
    FactRow,
    IngestionRunRow,
    ProjectRow,
    RequirementRow,
)


def main() -> None:
    checks = {
        "projects internal pk": hasattr(ProjectRow, "project_id"),
        "projects public id": hasattr(ProjectRow, "project_key"),
        "documents internal pk": hasattr(DocumentRow, "document_id"),
        "document_versions internal pk": hasattr(DocumentVersionRow, "document_version_id"),
        "runs internal pk": hasattr(IngestionRunRow, "run_pk"),
        "runs public id": hasattr(IngestionRunRow, "run_id"),
        "chunks internal pk": hasattr(ChunkRow, "chunk_pk"),
        "chunks public id": hasattr(ChunkRow, "chunk_id"),
        "facts internal pk": hasattr(FactRow, "fact_pk"),
        "facts sequence": hasattr(FactRow, "fact_sequence"),
        "facts public id": hasattr(FactRow, "fact_id"),
        "requirements internal pk": hasattr(RequirementRow, "requirement_pk"),
        "requirements sequence": hasattr(RequirementRow, "requirement_sequence"),
        "requirements public id": hasattr(RequirementRow, "requirement_id"),
    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        for name in failed:
            print(f"FAIL {name}")
        raise SystemExit(1)

    assert format_fact_id(1) == "FACT-000001"
    assert format_requirement_id(1) == "REQ-000001"

    print("PASS Phase 4 identifier ownership policy")


if __name__ == "__main__":
    main()
