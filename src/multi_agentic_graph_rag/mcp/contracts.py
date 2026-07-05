"""Strict contracts for MARAG MCP tools and orchestration results."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ServiceState = Literal["pass", "fail", "warn", "skipped"]


class StrictContract(BaseModel):
    __test__: ClassVar[bool] = False

    model_config = ConfigDict(extra="forbid")


class ServiceStatus(StrictContract):
    name: str
    status: ServiceState
    detail: str


class HealthReport(StrictContract):
    overall_status: Literal["pass", "fail", "warn"]
    checks: list[ServiceStatus]


class CliRunResult(StrictContract):
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    parsed_json: dict[str, Any] | None = None


class IngestToolInput(StrictContract):
    project: str
    document: str
    version: str
    logical_name: str | None = None
    replace_version: bool = False
    reasoning_provider: str | None = None
    embedding_provider: str | None = None


class UserStoryToolInput(StrictContract):
    project: str | None = None
    requirements: str | None = None
    document_version_id: str | None = None
    top_k: int | None = Field(default=None, gt=0)
    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None

    @model_validator(mode="after")
    def has_source(self) -> UserStoryToolInput:
        if not self.requirements and not self.document_version_id:
            raise ValueError("requirements or document_version_id is required")
        return self


class TestScenarioToolInput(StrictContract):
    project: str | None = None
    user_stories: str | None = None
    requirements: str | None = None
    document_version_id: str | None = None
    top_k: int | None = Field(default=None, gt=0)
    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None

    @model_validator(mode="after")
    def has_source(self) -> TestScenarioToolInput:
        if not self.user_stories and not self.document_version_id:
            raise ValueError("user_stories or document_version_id is required")
        return self


class FullPipelineToolInput(StrictContract):
    project: str
    document: str
    version: str
    logical_name: str | None = None
    replace_version: bool = False
    top_k: int | None = Field(default=None, gt=0)
    reasoning_provider: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None


class ArtifactLookupResult(StrictContract):
    project: str
    requirements: str | None = None
    user_stories: str | None = None
    test_scenarios: str | None = None


class VerifyArtifactToolInput(StrictContract):
    artifact_type: Literal["requirements", "user_stories", "test_scenarios"]
    path: str


class RunStatusToolInput(StrictContract):
    run_id: str
