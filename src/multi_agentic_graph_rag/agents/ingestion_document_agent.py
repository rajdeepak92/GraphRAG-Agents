"""Public ingestion agent contract."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.schemas import IngestionRequest, IngestionResult
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.workflows.ingestion_graph import run_ingestion


class IngestionDocumentAgent:
    def run(
        self,
        request: IngestionRequest,
        *,
        session: RunSession | None = None,
    ) -> IngestionResult:
        return run_ingestion(request, session=session)
