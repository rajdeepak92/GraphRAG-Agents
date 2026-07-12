"""Run session bootstrap and artifact coordination."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.common_defs import IdentifierPrefix, PathDef, RuntimeCommand
from multi_agentic_graph_rag.observability.logging import RunLogger, session_slug


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        temp_name = Path(handle.name)
    temp_name.replace(path)


def _command_run_id(command: str) -> str:
    timestamp = _utc_stamp().replace("-", "")
    return f"{IdentifierPrefix.RUN.value}{timestamp}-{session_slug(command).upper()}"


@dataclass
class RunSession:
    """Durable command-scoped session state."""

    project_root: Path
    project: str
    version: str
    command: str
    run_id: str
    stamp: str
    run_dir: Path
    log_path: Path
    jsonl_path: Path
    req_path: Path
    logger: RunLogger
    request_payload: dict[str, Any] | None = None
    artifact_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def bootstrap(
        cls,
        *,
        project_root: Path | None = None,
        project: str,
        version: str,
        command: str,
        run_id: str,
    ) -> RunSession:
        root = (project_root or Path(os.environ.get("PROJECT_ROOT", Path.cwd()))).resolve()
        project_name = project if project == "_system" else session_slug(project)
        version_name = session_slug(version)
        stamp = f"{_utc_stamp()}_{project_name}_{version_name}_{run_id}"
        if command == RuntimeCommand.INGEST.value:
            run_dir = (
                root / PathDef.GENERATED_REQUIREMENTS_DIR.value / project_name / "req" / run_id
            )
            log_path = run_dir / "run.log"
            jsonl_path = run_dir / "run.jsonl"
            req_path = run_dir / "requirements.json"
        else:
            run_dir = root / PathDef.LEGACY_GENERATED_DIR.value / project_name / "run"
            log_path = run_dir / f"run_{stamp}.log"
            jsonl_path = run_dir / f"run_{stamp}.jsonl"
            req_path = run_dir / f"req_{stamp}.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        jsonl_path.touch(exist_ok=True)
        logger = RunLogger(
            jsonl_path,
            log_path,
            run_id=run_id,
            project=project,
            version=version,
        ).bind(command=command)
        session = cls(
            project_root=root,
            project=project,
            version=version,
            command=command,
            run_id=run_id,
            stamp=stamp,
            run_dir=run_dir,
            log_path=log_path,
            jsonl_path=jsonl_path,
            req_path=req_path,
            logger=logger,
        )
        session.logger.info(
            "Command {command} started",
            step=command,
            command=command,
            project=project,
            version=version,
            run_dir=str(run_dir),
            stamp=stamp,
            status="started",
        )
        return session

    def set_log_level(self, level: str) -> None:
        self.logger.set_level(level)

    def write_artifact(self, payload: dict[str, Any]) -> Path:
        self.artifact_payload = payload
        self.logger.debug(
            "Writing requirement artifact to {path}",
            step="write_requirement_artifact",
            path=str(self.req_path),
            payload_keys=sorted(payload),
        )
        _atomic_write_json(self.req_path, payload)
        return self.req_path

    def write_session_summary(self, *, status: str, extra: dict[str, Any] | None = None) -> Path:
        if self.req_path.exists():
            return self.req_path
        payload: dict[str, Any] = {
            "artifact_schema_version": "session-metadata",
            "status": status,
            "run_id": self.run_id,
            "project": self.project,
            "version": self.version,
            "command": self.command,
            "stamp": self.stamp,
            "request": self.request_payload,
            "metadata": self.metadata,
        }
        if self.artifact_payload is not None:
            payload["artifact"] = self.artifact_payload
        if extra:
            payload["extra"] = extra
        _atomic_write_json(self.req_path, payload)
        return self.req_path

    def write_failure_envelope(
        self,
        *,
        error: BaseException,
        include_artifact: bool = True,
    ) -> Path:
        payload: dict[str, Any] = {
            "artifact_schema_version": "session-failure",
            "status": "failed",
            "run_id": self.run_id,
            "project": self.project,
            "version": self.version,
            "command": self.command,
            "stamp": self.stamp,
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
            "request": self.request_payload,
            "metadata": self.metadata,
        }
        if include_artifact and self.artifact_payload is not None:
            payload["artifact"] = self.artifact_payload
        _atomic_write_json(self.req_path, payload)
        return self.req_path

    def close(self) -> None:
        self.logger.close()


@contextmanager
def command_session(
    *,
    project: str,
    version: str,
    command: str,
    run_id: str,
    project_root: Path | None = None,
) -> Iterator[RunSession]:
    session = RunSession.bootstrap(
        project_root=project_root,
        project=project,
        version=version,
        command=command,
        run_id=run_id,
    )
    try:
        yield session
    except KeyboardInterrupt as exc:
        session.logger.exception(
            "Command {command} interrupted",
            step=command,
            command=command,
            exc=exc,
            status="interrupted",
        )
        session.write_failure_envelope(error=exc)
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "Exit":
            session.write_session_summary(
                status="exited",
                extra={"exit_code": getattr(exc, "exit_code", None)},
            )
            session.logger.info(
                "Command {command} exited",
                step=command,
                command=command,
                exit_code=getattr(exc, "exit_code", None),
                status="exited",
            )
            raise
        session.logger.exception(
            "Command {command} failed",
            step=command,
            command=command,
            exc=exc,
            status="failed",
        )
        session.write_failure_envelope(error=exc)
        raise
    else:
        session.write_session_summary(status="completed")
        session.logger.info(
            "Command {command} completed",
            step=command,
            command=command,
            stamp=session.stamp,
            status="completed",
        )
    finally:
        session.close()


def command_run_id(command: str) -> str:
    return _command_run_id(command)


def find_run_jsonl(project_root: Path, run_id: str) -> Path | None:
    generated_root = project_root / PathDef.GENERATED_REQUIREMENTS_DIR.value
    if generated_root.exists():
        matches = sorted(generated_root.glob(f"*/req/{run_id}/run.jsonl"))
        if matches:
            return matches[-1]
    legacy_generated_root = project_root / PathDef.LEGACY_GENERATED_DIR.value
    if legacy_generated_root.exists():
        matches = sorted(legacy_generated_root.glob(f"*/run/run_*{run_id}*.jsonl"))
        if matches:
            return matches[-1]
    legacy = project_root / "runtime" / "logs" / f"{run_id}.jsonl"
    if legacy.exists():
        return legacy
    return None
