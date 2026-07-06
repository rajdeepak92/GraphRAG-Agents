"""Windows-native service and TCP health helpers."""

from __future__ import annotations

import platform
import socket
import subprocess
import time

from common_defs import EnvVar, ServiceName

from multi_agentic_graph_rag.mcp.contracts import ServiceStatus


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def test_tcp_port(host: str, port: int, timeout_seconds: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def get_windows_service_status(service_name: str) -> str:
    if not service_name:
        return "not_configured"
    if not is_windows():
        return "skipped_non_windows"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                "$svc = Get-Service -Name $args[0] -ErrorAction SilentlyContinue; "
                "if ($null -eq $svc) { 'not_found' } else { $svc.Status.ToString() }"
            ),
            service_name,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip() or completed.stdout.strip()}"
    return completed.stdout.strip() or "unknown"


def start_windows_service(service_name: str) -> ServiceStatus:
    if not service_name:
        return ServiceStatus(
            name=ServiceName.POSTGRES_SERVICE.value,
            status="skipped",
            detail=f"{EnvVar.MARAG_POSTGRES_SERVICE_NAME.value} is not configured",
        )
    if not is_windows():
        return ServiceStatus(
            name=service_name,
            status="skipped",
            detail="Windows service start skipped on non-Windows platform",
        )

    current_status = get_windows_service_status(service_name)
    if current_status == "not_found":
        return ServiceStatus(name=service_name, status="fail", detail="service not found")
    if current_status.lower() == "running":
        return ServiceStatus(name=service_name, status="pass", detail="service already running")
    if current_status.startswith("error:"):
        return ServiceStatus(name=service_name, status="fail", detail=current_status)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Start-Service -Name $args[0]",
            service_name,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Start-Service failed"
        return ServiceStatus(name=service_name, status="fail", detail=detail)

    final_status = get_windows_service_status(service_name)
    if final_status.lower() == "running":
        return ServiceStatus(name=service_name, status="pass", detail="service started")
    return ServiceStatus(
        name=service_name,
        status="warn",
        detail=f"start requested; current status is {final_status}",
    )


def stop_windows_service(service_name: str) -> ServiceStatus:
    if not service_name:
        return ServiceStatus(
            name=ServiceName.POSTGRES_SERVICE.value,
            status="skipped",
            detail=f"{EnvVar.MARAG_POSTGRES_SERVICE_NAME.value} is not configured",
        )
    if not is_windows():
        return ServiceStatus(
            name=service_name,
            status="skipped",
            detail="Windows service stop skipped on non-Windows platform",
        )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Stop-Service -Name $args[0]",
            service_name,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Stop-Service failed"
        return ServiceStatus(name=service_name, status="fail", detail=detail)
    return ServiceStatus(name=service_name, status="pass", detail="service stop requested")


def wait_for_tcp_port(host: str, port: int, timeout_seconds: int = 60) -> ServiceStatus:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if test_tcp_port(host, port, timeout_seconds=1.0):
            return ServiceStatus(
                name=f"{host}:{port}",
                status="pass",
                detail="TCP port is accepting connections",
            )
        time.sleep(1)
    return ServiceStatus(
        name=f"{host}:{port}",
        status="fail",
        detail=f"TCP port did not open within {timeout_seconds} seconds",
    )
