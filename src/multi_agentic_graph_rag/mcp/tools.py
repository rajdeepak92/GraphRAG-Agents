"""MCP tool registration for Windows-native MARAG orchestration."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from multi_agentic_graph_rag.mcp.artifact_index import find_latest_artifacts
from multi_agentic_graph_rag.mcp.cli_runner import CliExecutionError, run_marag_command
from multi_agentic_graph_rag.mcp.contracts import (
    FullPipelineToolInput,
    HealthReport,
    IngestToolInput,
    TestScenarioToolInput,
    UserStoryToolInput,
    VerifyArtifactToolInput,
)
from multi_agentic_graph_rag.mcp.path_safety import UnsafePathError, resolve_project_path
from multi_agentic_graph_rag.mcp.service_manager import (
    check_marag_stack,
    start_local_stack,
    stop_local_stack,
)

ProjectRootFactory = Callable[[], Path]


def register_tools(mcp: Any, *, project_root: ProjectRootFactory) -> None:
    @mcp.tool(description="Check PostgreSQL, Neo4j, Chroma, and application health.")
    def marag_health_check() -> dict[str, Any]:
        return health_check_tool(project_root()).model_dump(mode="json")

    @mcp.tool(description="Start the configured local stack and return health details.")
    def marag_start_stack() -> dict[str, Any]:
        return start_stack_tool(project_root()).model_dump(mode="json")

    @mcp.tool(description="Stop the configured local stack when shutdown is enabled.")
    def marag_stop_stack() -> dict[str, Any]:
        return stop_stack_tool(project_root()).model_dump(mode="json")

    @mcp.tool(
        description=(
            "Ingest a BRD/SRS document, discover requirements, and write generated artifacts."
        )
    )
    def marag_ingest_document(
        project: str,
        document: str,
        version: str,
        logical_name: str | None = None,
        replace_version: bool = False,
        reasoning_provider: str | None = None,
        embedding_provider: str | None = None,
    ) -> dict[str, Any]:
        request = IngestToolInput(
            project=project,
            document=document,
            version=version,
            logical_name=logical_name,
            replace_version=replace_version,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
        )
        return ingest_document_tool(project_root(), request)

    @mcp.tool(
        description=(
            "Generate user stories from requirements or a document version "
            "using retrieval and reranking."
        )
    )
    def marag_generate_user_stories(
        project: str | None = None,
        requirements: str | None = None,
        document_version_id: str | None = None,
        top_k: int | None = None,
        reasoning_provider: str | None = None,
        embedding_provider: str | None = None,
        reranker_provider: str | None = None,
    ) -> dict[str, Any]:
        request = UserStoryToolInput(
            project=project,
            requirements=requirements,
            document_version_id=document_version_id,
            top_k=top_k,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
        )
        return generate_user_stories_tool(project_root(), request)

    @mcp.tool(
        description=(
            "Generate test scenarios from user stories and requirements "
            "using retrieval and reranking."
        )
    )
    def marag_generate_test_scenarios(
        project: str | None = None,
        user_stories: str | None = None,
        requirements: str | None = None,
        document_version_id: str | None = None,
        top_k: int | None = None,
        reasoning_provider: str | None = None,
        embedding_provider: str | None = None,
        reranker_provider: str | None = None,
    ) -> dict[str, Any]:
        request = TestScenarioToolInput(
            project=project,
            user_stories=user_stories,
            requirements=requirements,
            document_version_id=document_version_id,
            top_k=top_k,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
        )
        return generate_test_scenarios_tool(project_root(), request)

    @mcp.tool(
        description=(
            "Run the full pipeline: stack check, ingest, user stories, "
            "test scenarios, and verification."
        )
    )
    def marag_run_full_pipeline(
        project: str,
        document: str,
        version: str,
        logical_name: str | None = None,
        replace_version: bool = False,
        top_k: int | None = None,
        reasoning_provider: str | None = None,
        embedding_provider: str | None = None,
        reranker_provider: str | None = None,
    ) -> dict[str, Any]:
        request = FullPipelineToolInput(
            project=project,
            document=document,
            version=version,
            logical_name=logical_name,
            replace_version=replace_version,
            top_k=top_k,
            reasoning_provider=reasoning_provider,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
        )
        return full_pipeline_tool(project_root(), request)

    @mcp.tool(
        description=(
            "Find the latest generated requirement, user-story, and test-scenario artifacts."
        )
    )
    def marag_find_latest_artifacts(project: str) -> dict[str, Any]:
        return find_latest_artifacts(project_root(), project).model_dump(mode="json")

    @mcp.tool(
        description="Verify a generated requirements, user-stories, or test-scenarios artifact."
    )
    def marag_verify_artifact(artifact_type: str, path: str) -> dict[str, Any]:
        request = VerifyArtifactToolInput.model_validate(
            {"artifact_type": artifact_type, "path": path}
        )
        return verify_artifact_tool(project_root(), request)

    @mcp.tool(description="Open run status and logs for a specific run id.")
    def marag_open_run_status(run_id: str) -> dict[str, Any]:
        return run_status_tool(project_root(), run_id)


def health_check_tool(project_root: Path) -> HealthReport:
    return check_marag_stack(project_root)


def start_stack_tool(project_root: Path) -> HealthReport:
    return start_local_stack(project_root)


def stop_stack_tool(project_root: Path) -> HealthReport:
    return stop_local_stack(project_root)


def ingest_document_tool(project_root: Path, request: IngestToolInput) -> dict[str, Any]:
    try:
        resolve_project_path(project_root, request.document, allowed_roots=("documents",))
    except UnsafePathError as exc:
        return _failure(str(exc))

    health = _ensure_stack_ready(project_root)
    if health.overall_status == "fail":
        return _failure("MARAG stack is not healthy", health=health)

    args = [
        "ingest",
        "--project",
        request.project,
        "--document",
        request.document,
        "--version",
        request.version,
        "--json-output",
    ]
    if request.logical_name:
        args.extend(["--logical-name", request.logical_name])
    if request.replace_version:
        args.append("--replace-version")
    _append_provider_args(
        args,
        reasoning_provider=request.reasoning_provider,
        embedding_provider=request.embedding_provider,
    )
    return _run_json_command(project_root, args, health=health)


def generate_user_stories_tool(project_root: Path, request: UserStoryToolInput) -> dict[str, Any]:
    if request.requirements:
        try:
            resolve_project_path(project_root, request.requirements, allowed_roots=("generated",))
        except UnsafePathError as exc:
            return _failure(str(exc))

    health = _ensure_stack_ready(project_root)
    if health.overall_status == "fail":
        return _failure("MARAG stack is not healthy", health=health)

    args = ["generate-user-stories", "--json-output"]
    if request.requirements:
        args.extend(["--requirements", request.requirements])
    if request.document_version_id:
        args.extend(["--document-version-id", request.document_version_id])
    if request.project:
        args.extend(["--project", request.project])
    if request.top_k is not None:
        args.extend(["--top-k", str(request.top_k)])
    _append_provider_args(
        args,
        reasoning_provider=request.reasoning_provider,
        embedding_provider=request.embedding_provider,
        reranker_provider=request.reranker_provider,
    )
    return _run_json_command(project_root, args, health=health)


def generate_test_scenarios_tool(
    project_root: Path,
    request: TestScenarioToolInput,
) -> dict[str, Any]:
    for path_value in (request.user_stories, request.requirements):
        if path_value:
            try:
                resolve_project_path(project_root, path_value, allowed_roots=("generated",))
            except UnsafePathError as exc:
                return _failure(str(exc))

    health = _ensure_stack_ready(project_root)
    if health.overall_status == "fail":
        return _failure("MARAG stack is not healthy", health=health)

    args = ["generate-test-scenarios", "--json-output"]
    if request.user_stories:
        args.extend(["--user-stories", request.user_stories])
    if request.requirements:
        args.extend(["--requirements", request.requirements])
    if request.document_version_id:
        args.extend(["--document-version-id", request.document_version_id])
    if request.project:
        args.extend(["--project", request.project])
    if request.top_k is not None:
        args.extend(["--top-k", str(request.top_k)])
    _append_provider_args(
        args,
        reasoning_provider=request.reasoning_provider,
        embedding_provider=request.embedding_provider,
        reranker_provider=request.reranker_provider,
    )
    return _run_json_command(project_root, args, health=health)


def full_pipeline_tool(project_root: Path, request: FullPipelineToolInput) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "project": request.project,
        "version": request.version,
        "health": None,
        "ingest": None,
        "user_stories": None,
        "test_scenarios": None,
        "verification": {},
    }
    health = _ensure_stack_ready(project_root)
    result["health"] = health.model_dump(mode="json")
    if health.overall_status == "fail":
        result["error"] = "MARAG stack is not healthy"
        return result

    ingest = ingest_document_tool(
        project_root,
        IngestToolInput(
            project=request.project,
            document=request.document,
            version=request.version,
            logical_name=request.logical_name,
            replace_version=request.replace_version,
            reasoning_provider=request.reasoning_provider,
            embedding_provider=request.embedding_provider,
        ),
    )
    result["ingest"] = ingest
    if not ingest.get("ok"):
        result["error"] = "ingest failed"
        return result

    ingest_json = _result_dict(ingest)
    requirements_path = str(ingest_json.get("artifact_path", ""))
    full_requirements_path = str(ingest_json.get("full_artifact_path") or requirements_path)

    user_stories = generate_user_stories_tool(
        project_root,
        UserStoryToolInput(
            project=request.project,
            requirements=requirements_path,
            top_k=request.top_k,
            reasoning_provider=request.reasoning_provider,
            embedding_provider=request.embedding_provider,
            reranker_provider=request.reranker_provider,
        ),
    )
    result["user_stories"] = user_stories
    if not user_stories.get("ok"):
        result["error"] = "user-story generation failed"
        return result

    user_story_json = _result_dict(user_stories)
    user_story_path = str(user_story_json.get("artifact_path", ""))

    test_scenarios = generate_test_scenarios_tool(
        project_root,
        TestScenarioToolInput(
            project=request.project,
            user_stories=user_story_path,
            requirements=requirements_path,
            top_k=request.top_k,
            reasoning_provider=request.reasoning_provider,
            embedding_provider=request.embedding_provider,
            reranker_provider=request.reranker_provider,
        ),
    )
    result["test_scenarios"] = test_scenarios
    if not test_scenarios.get("ok"):
        result["error"] = "test-scenario generation failed"
        return result

    test_scenario_json = _result_dict(test_scenarios)
    test_scenario_path = str(test_scenario_json.get("artifact_path", ""))
    result["verification"] = {
        "requirements": verify_artifact_tool(
            project_root,
            VerifyArtifactToolInput(artifact_type="requirements", path=full_requirements_path),
        ),
        "user_stories": verify_artifact_tool(
            project_root,
            VerifyArtifactToolInput(artifact_type="user_stories", path=user_story_path),
        ),
        "test_scenarios": verify_artifact_tool(
            project_root,
            VerifyArtifactToolInput(artifact_type="test_scenarios", path=test_scenario_path),
        ),
    }
    result["ok"] = all(
        isinstance(value, dict) and bool(value.get("ok"))
        for value in result["verification"].values()
    )
    if not result["ok"]:
        result["error"] = "artifact verification failed"
    return result


def verify_artifact_tool(project_root: Path, request: VerifyArtifactToolInput) -> dict[str, Any]:
    try:
        resolve_project_path(project_root, request.path, allowed_roots=("generated", ".generated"))
    except UnsafePathError as exc:
        return _failure(str(exc))
    command_by_type = {
        "requirements": ["artifact", "verify", request.path],
        "user_stories": ["artifact", "verify-user-stories", request.path],
        "test_scenarios": ["artifact", "verify-test-scenarios", request.path],
    }
    return _run_plain_command(project_root, command_by_type[request.artifact_type])


def run_status_tool(project_root: Path, run_id: str) -> dict[str, Any]:
    return _run_plain_command(project_root, ["run", "status", run_id])


def _ensure_stack_ready(project_root: Path) -> HealthReport:
    health = check_marag_stack(project_root)
    if health.overall_status != "fail":
        return health
    started = start_local_stack(project_root)
    if started.overall_status == "fail":
        return started
    return check_marag_stack(project_root)


def _append_provider_args(
    args: list[str],
    *,
    reasoning_provider: str | None = None,
    embedding_provider: str | None = None,
    reranker_provider: str | None = None,
) -> None:
    if reasoning_provider:
        args.extend(["--reasoning-provider", reasoning_provider])
    if embedding_provider:
        args.extend(["--embedding-provider", embedding_provider])
    if reranker_provider:
        args.extend(["--reranker-provider", reranker_provider])


def _run_json_command(
    project_root: Path,
    args: list[str],
    *,
    health: HealthReport,
) -> dict[str, Any]:
    try:
        cli_result = run_marag_command(args, project_root=project_root, expect_json=True)
    except CliExecutionError as exc:
        return _failure(str(exc), health=health)
    return {
        "ok": cli_result.exit_code == 0,
        "health": health.model_dump(mode="json"),
        "cli": cli_result.model_dump(mode="json"),
        "result": cli_result.parsed_json,
    }


def _run_plain_command(project_root: Path, args: list[str]) -> dict[str, Any]:
    try:
        cli_result = run_marag_command(args, project_root=project_root)
    except CliExecutionError as exc:
        return _failure(str(exc))
    return {"ok": cli_result.exit_code == 0, "cli": cli_result.model_dump(mode="json")}


def _result_dict(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("result")
    return value if isinstance(value, dict) else {}


def _failure(message: str, *, health: HealthReport | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": message}
    if health is not None:
        payload["health"] = health.model_dump(mode="json")
    return payload


def validate_tool_contract(model: type[Any], payload: dict[str, Any]) -> Any:
    try:
        return model.model_validate(payload)
    except ValidationError:
        raise
