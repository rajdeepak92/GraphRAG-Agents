"""Small cross-process file lock for local-JSON transaction boundaries."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from pathlib import Path


class LocalFileLock:
    def __init__(self, path: Path, *, timeout_seconds: float = 30.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def __enter__(self) -> LocalFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for local lock {self.path}") from None
                time.sleep(0.02)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
        with suppress(FileNotFoundError):
            self.path.unlink()
