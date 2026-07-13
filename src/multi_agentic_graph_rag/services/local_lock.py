"""Small cross-process file lock for local-JSON transaction boundaries."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from pathlib import Path


class LocalFileLock:
    """Coordinate local file lock behavior within the services boundary."""

    def __init__(self, path: Path, *, timeout_seconds: float = 30.0) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            path (Path): Filesystem location authorized for this operation.
            timeout_seconds (float): Timeout seconds required by the operation's typed contract.
        """
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def __enter__(self) -> LocalFileLock:
        """Execute the enter operation within its declared architectural boundary.

        Returns:
            LocalFileLock: The typed result produced by the operation.

        Raises:
            TimeoutError: If validated inputs or required dependencies cannot satisfy the contract.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
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
        """Execute the exit operation within its declared architectural boundary.

        Args:
            exc_type (object): Exc type required by the operation's typed contract.
            exc (object): Failure being classified or converted without exposing its message.
            traceback (object): Traceback required by the operation's typed contract.
        """
        if self._fd is not None:
            os.close(self._fd)
        with suppress(FileNotFoundError):
            self.path.unlink()
