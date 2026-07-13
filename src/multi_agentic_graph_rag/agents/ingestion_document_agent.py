"""Public ingestion agent contract."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.schemas import IngestionRequest, IngestionResult
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.workflows.ingestion_graph import run_ingestion


class IngestionDocumentAgent:
    """Coordinate ingestion document agent behavior within the agents boundary."""

    def run(
        self,
        request: IngestionRequest,
        *,
        session: RunSession | None = None,
    ) -> IngestionResult:
        """Run run.

        Args:
            request (IngestionRequest): Request required by the operation's typed contract.
            session (RunSession | None): Optional command session that owns run artifacts and
                                         diagnostics.

        Returns:
            IngestionResult: The typed result produced by the operation.
        """
        return run_ingestion(request, session=session)
