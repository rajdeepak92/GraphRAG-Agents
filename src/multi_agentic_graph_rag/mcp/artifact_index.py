"""Locate generated MARAG artifacts for MCP resources and tools."""

from __future__ import annotations

from pathlib import Path

from multi_agentic_graph_rag.mcp.contracts import ArtifactLookupResult
from multi_agentic_graph_rag.observability.logging import session_slug


def find_latest_artifacts(project_root: Path, project: str) -> ArtifactLookupResult:
    generated_root = project_root.resolve() / "generated"
    project_names = _project_dir_candidates(project)

    return ArtifactLookupResult(
        project=project,
        requirements=_latest_match(
            generated_root,
            project_names,
            ("req/**/requirements.json", "**/requirements.json"),
        ),
        user_stories=_latest_match(
            generated_root,
            project_names,
            ("user_stories/**/user_stories.json", "req/**/user_stories.json"),
        ),
        test_scenarios=_latest_match(
            generated_root,
            project_names,
            ("test_scenarios/**/test_scenarios.json", "req/**/test_scenarios.json"),
        ),
    )


def _project_dir_candidates(project: str) -> tuple[str, ...]:
    slug = session_slug(project)
    return (project,) if project == slug else (project, slug)


def _latest_match(
    root: Path,
    project_names: tuple[str, ...],
    patterns: tuple[str, ...],
) -> str | None:
    matches: dict[Path, None] = {}
    for project_name in project_names:
        project_root = root / project_name
        for pattern in patterns:
            for match in project_root.glob(pattern):
                if match.is_file():
                    matches[match.resolve()] = None
    if not matches:
        return None
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    return str(latest)
