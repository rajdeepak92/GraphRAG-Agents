"""Application repository ports."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from multi_agentic_graph_rag.domain.chunks import Chunk
from multi_agentic_graph_rag.domain.documents import Document, DocumentVersion, Project
from multi_agentic_graph_rag.domain.enums import RunStatus
from multi_agentic_graph_rag.domain.runs import IngestionRun, RunStep


class ProjectRepository(Protocol):
    async def get_by_key(self, project_key: str) -> Project | None: ...

    async def create(self, *, project_key: str, name: str | None = None) -> Project: ...


class DocumentRepository(Protocol):
    async def get_by_project_and_name(
        self,
        *,
        project_id: UUID,
        logical_document_name: str,
    ) -> Document | None: ...

    async def create(
        self,
        *,
        project_id: UUID,
        logical_document_name: str,
    ) -> Document: ...


class DocumentVersionRepository(Protocol):
    async def get_by_document_and_version(
        self,
        *,
        document_id: UUID,
        supplied_version: str,
    ) -> DocumentVersion | None: ...

    async def create(self, version: DocumentVersion) -> DocumentVersion: ...


class ChunkRepository(Protocol):
    async def list_active_by_document_version(
        self,
        *,
        document_version_id: UUID,
    ) -> tuple[Chunk, ...]: ...


class IngestionRunRepository(Protocol):
    async def create(self, run: IngestionRun) -> IngestionRun: ...

    async def get_by_run_id(self, run_id: str) -> IngestionRun | None: ...

    async def transition(
        self,
        *,
        run_id: str,
        target_status: RunStatus,
    ) -> IngestionRun: ...


class RunStepRepository(Protocol):
    async def create(self, step: RunStep) -> RunStep: ...
