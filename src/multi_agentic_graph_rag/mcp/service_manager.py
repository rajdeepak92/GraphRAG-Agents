"""Windows-native MARAG stack orchestration for MCP tools."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from common_defs import EnvVar, ModeName

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.mcp.cli_runner import CliExecutionError, run_marag_command
from multi_agentic_graph_rag.mcp.contracts import HealthReport, ServiceStatus
from multi_agentic_graph_rag.mcp.powershell import run_powershell_script
from multi_agentic_graph_rag.mcp.windows_services import (
    get_windows_service_status,
    start_windows_service,
    stop_windows_service,
    test_tcp_port,
    wait_for_tcp_port,
)

LAST_HEALTH_REPORT: HealthReport | None = None
OverallStatus = Literal["pass", "fail", "warn"]


def check_marag_stack(project_root: Path) -> HealthReport:
    root = project_root.resolve()
    env = read_project_env(root)
    settings = _load_project_settings(root)
    checks: list[ServiceStatus] = []

    if settings.postgres.mode == ModeName.POSTGRES.value:
        service_name = env.get(EnvVar.MARAG_POSTGRES_SERVICE_NAME.value, "")
        if service_name:
            service_status = get_windows_service_status(service_name)
            checks.append(_service_status_from_raw(service_name, service_status))
        else:
            checks.append(
                ServiceStatus(
                    name="postgres_service",
                    status="skipped",
                    detail="MARAG_POSTGRES_SERVICE_NAME is not configured",
                )
            )
        pg_host, pg_port = _postgres_host_port(settings, env)
        checks.append(_port_check("postgres_tcp", pg_host, pg_port))
    else:
        checks.append(
            ServiceStatus(
                name="postgres",
                status="skipped",
                detail=f"POSTGRES_MODE={settings.postgres.mode}",
            )
        )

    if settings.neo4j.mode == ModeName.NEO4J.value:
        neo4j_host = _neo4j_host(settings)
        checks.append(
            _port_check(
                "neo4j_http",
                neo4j_host,
                _int_env(env, EnvVar.MARAG_NEO4J_HTTP_PORT.value, 7474),
            )
        )
        checks.append(
            _port_check(
                "neo4j_bolt",
                neo4j_host,
                _int_env(env, EnvVar.MARAG_NEO4J_BOLT_PORT.value, 7687),
            )
        )
    else:
        checks.append(
            ServiceStatus(
                name="neo4j",
                status="skipped",
                detail=f"NEO4J_MODE={settings.neo4j.mode}",
            )
        )

    chroma_dir = settings.paths.chroma_persist_dir
    checks.append(
        ServiceStatus(
            name="chroma_persist_dir",
            status="pass" if chroma_dir.exists() else "fail",
            detail=str(chroma_dir),
        )
    )
    checks.append(_db_check(root))
    return _remember(HealthReport(overall_status=_overall_status(checks), checks=checks))


def start_local_stack(project_root: Path) -> HealthReport:
    root = project_root.resolve()
    env = read_project_env(root)
    settings = _load_project_settings(root)
    checks: list[ServiceStatus] = []

    if settings.postgres.mode == ModeName.POSTGRES.value:
        if _env_bool(env, EnvVar.MARAG_POSTGRES_AUTO_START.value, default=False):
            service_name = env.get(EnvVar.MARAG_POSTGRES_SERVICE_NAME.value, "")
            checks.append(start_windows_service(service_name))
        else:
            checks.append(
                ServiceStatus(
                    name="postgres_service",
                    status="skipped",
                    detail="MARAG_POSTGRES_AUTO_START is not true",
                )
            )
        pg_host, pg_port = _postgres_host_port(settings, env)
        checks.append(wait_for_tcp_port(pg_host, pg_port, timeout_seconds=60))
    else:
        checks.append(
            ServiceStatus(
                name="postgres",
                status="skipped",
                detail=f"POSTGRES_MODE={settings.postgres.mode}",
            )
        )

    if settings.neo4j.mode == ModeName.NEO4J.value:
        checks.extend(_start_neo4j(root, settings, env))
    else:
        checks.append(
            ServiceStatus(
                name="neo4j",
                status="skipped",
                detail=f"NEO4J_MODE={settings.neo4j.mode}",
            )
        )

    settings.paths.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    checks.append(
        ServiceStatus(
            name="chroma_persist_dir",
            status="pass",
            detail=f"ready at {settings.paths.chroma_persist_dir}",
        )
    )
    checks.append(_db_check(root))
    return _remember(HealthReport(overall_status=_overall_status(checks), checks=checks))


def stop_local_stack(project_root: Path) -> HealthReport:
    root = project_root.resolve()
    env = read_project_env(root)
    checks: list[ServiceStatus] = []
    if not _env_bool(env, EnvVar.MARAG_ALLOW_SERVICE_STOP.value, default=False):
        checks.append(
            ServiceStatus(
                name="service_stop",
                status="warn",
                detail="MARAG_ALLOW_SERVICE_STOP=false; no services stopped",
            )
        )
        return _remember(HealthReport(overall_status="warn", checks=checks))

    if _env_bool(env, EnvVar.MARAG_POSTGRES_ALLOW_STOP.value, default=False):
        checks.append(stop_windows_service(env.get(EnvVar.MARAG_POSTGRES_SERVICE_NAME.value, "")))
    else:
        checks.append(
            ServiceStatus(
                name="postgres_service",
                status="skipped",
                detail="MARAG_POSTGRES_ALLOW_STOP is not true",
            )
        )

    checks.append(
        ServiceStatus(
            name="neo4j",
            status="skipped",
            detail="Neo4j stop is skipped unless a future MCP-managed PID is available",
        )
    )
    return _remember(HealthReport(overall_status=_overall_status(checks), checks=checks))


def read_project_env(project_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = project_root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def _start_neo4j(root: Path, settings: AppSettings, env: dict[str, str]) -> list[ServiceStatus]:
    checks: list[ServiceStatus] = []
    host = _neo4j_host(settings)
    bolt_port = _int_env(env, EnvVar.MARAG_NEO4J_BOLT_PORT.value, 7687)
    http_port = _int_env(env, EnvVar.MARAG_NEO4J_HTTP_PORT.value, 7474)

    if test_tcp_port(host, bolt_port, timeout_seconds=1.0):
        checks.append(ServiceStatus(name="neo4j_bolt", status="pass", detail="already listening"))
    elif not _env_bool(env, EnvVar.MARAG_NEO4J_AUTO_START.value, default=False):
        checks.append(
            ServiceStatus(
                name="neo4j",
                status="skipped",
                detail="MARAG_NEO4J_AUTO_START is not true",
            )
        )
    else:
        mode = env.get(EnvVar.MARAG_NEO4J_START_MODE.value, ModeName.DESKTOP_DBMS.value)
        if mode != ModeName.DESKTOP_DBMS.value:
            checks.append(
                ServiceStatus(
                    name="neo4j",
                    status="fail",
                    detail=f"unsupported MARAG_NEO4J_START_MODE={mode}",
                )
            )
        else:
            dbms_home = env.get("NEO4J_DBMS_HOME", "")
            if not dbms_home:
                checks.append(
                    ServiceStatus(
                        name="neo4j",
                        status="fail",
                        detail="NEO4J_DBMS_HOME is required for desktop_dbms start mode",
                    )
                )
            elif not Path(dbms_home).exists():
                checks.append(
                    ServiceStatus(
                        name="neo4j",
                        status="fail",
                        detail=f"NEO4J_DBMS_HOME not found: {dbms_home}",
                    )
                )
            else:
                script_args = ["-Neo4jDbmsHome", dbms_home]
                java_home = env.get("NEO4J_JAVA_HOME", "")
                if java_home:
                    script_args.extend(["-JavaHome", java_home])
                result = run_powershell_script(
                    root / "scripts" / "mcp" / "start-neo4j-dbms.ps1",
                    project_root=root,
                    args=script_args,
                    timeout_seconds=120,
                )
                checks.append(
                    ServiceStatus(
                        name="neo4j_start",
                        status="pass" if result.exit_code == 0 else "fail",
                        detail=result.stdout.strip() or result.stderr.strip(),
                    )
                )

    checks.append(wait_for_tcp_port(host, http_port, timeout_seconds=60))
    checks.append(wait_for_tcp_port(host, bolt_port, timeout_seconds=60))
    return checks


def _service_status_from_raw(service_name: str, raw_status: str) -> ServiceStatus:
    normalized = raw_status.lower()
    if normalized == "running":
        return ServiceStatus(name=service_name, status="pass", detail=raw_status)
    if normalized == "not_found":
        return ServiceStatus(name=service_name, status="fail", detail="service not found")
    if normalized == "skipped_non_windows":
        return ServiceStatus(
            name=service_name,
            status="skipped",
            detail="Windows service check skipped on non-Windows platform",
        )
    if normalized.startswith("error:"):
        return ServiceStatus(name=service_name, status="fail", detail=raw_status)
    return ServiceStatus(name=service_name, status="warn", detail=raw_status or "unknown")


def _port_check(name: str, host: str, port: int) -> ServiceStatus:
    ok = test_tcp_port(host, port)
    return ServiceStatus(
        name=name,
        status="pass" if ok else "fail",
        detail=f"{host}:{port} {'open' if ok else 'not accepting connections'}",
    )


def _db_check(project_root: Path) -> ServiceStatus:
    env = read_project_env(project_root)

    if _env_bool(env, "MARAG_MCP_SKIP_DB_CHECK", default=False):
        return ServiceStatus(
            name="marag_db_check",
            status="skipped",
            detail=(
                "Skipped via MARAG_MCP_SKIP_DB_CHECK=true. "
                "Run `uv run marag db-check` or the dev helper for deep DB validation."
            ),
        )

    try:
        result = run_marag_command(["db-check"], project_root=project_root, timeout_seconds=180)
    except CliExecutionError as exc:
        return ServiceStatus(name="marag_db_check", status="fail", detail=str(exc))

    detail = result.stdout.strip() or result.stderr.strip()
    return ServiceStatus(
        name="marag_db_check",
        status="pass" if result.exit_code == 0 else "fail",
        detail=detail,
    )


def _postgres_host_port(settings: AppSettings, env: dict[str, str]) -> tuple[str, int]:
    parsed = urlparse(settings.postgres.dsn)
    host = env.get(EnvVar.MARAG_POSTGRES_HOST.value) or parsed.hostname or "127.0.0.1"
    port = _int_env(env, EnvVar.MARAG_POSTGRES_PORT.value, parsed.port or 5432)
    return host, port


def _neo4j_host(settings: AppSettings) -> str:
    parsed = urlparse(settings.neo4j.uri)
    return parsed.hostname or "127.0.0.1"


def _int_env(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, "") or default)
    except ValueError:
        return default


def _env_bool(env: dict[str, str], key: str, *, default: bool) -> bool:
    value = env.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _overall_status(checks: list[ServiceStatus]) -> OverallStatus:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _remember(report: HealthReport) -> HealthReport:
    global LAST_HEALTH_REPORT
    LAST_HEALTH_REPORT = report
    return report


def _load_project_settings(project_root: Path) -> AppSettings:
    with _project_root_env(project_root):
        return load_config()


@contextmanager
def _project_root_env(project_root: Path) -> Iterator[None]:
    previous = os.environ.get("PROJECT_ROOT")
    os.environ["PROJECT_ROOT"] = str(project_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PROJECT_ROOT", None)
        else:
            os.environ["PROJECT_ROOT"] = previous
