"""Public ingestion agent contract."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.schemas import IngestionRequest, IngestionResult
from multi_agentic_graph_rag.workflows.ingestion_graph import run_ingestion


class IngestionDocumentAgent:
    def run(self, request: IngestionRequest) -> IngestionResult:
        return run_ingestion(request)
