"""Journaled, no-Git direct writes for one generated Stage-4 test case.

The transaction deliberately knows nothing about source-control state.  Its
durability boundary is an on-disk intent journal: preimages, the target hash,
and any directories that may be created are persisted *before* the framework is
mutated.  Recovery therefore remains safe across process termination at every
write boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from multi_agentic_graph_rag.domain.codegen_schemas import GeneratedFileRole
from multi_agentic_graph_rag.domain.identifiers import TC_ID_MAX, TC_ID_MIN

_MODULE = r"[a-z][a-z0-9_]*"
_TC_STEM = r"Tc(?P<tc_id>\d{6})(?P<title>[A-Z][A-Za-z0-9]*)"
_JOURNAL_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}

_ALLOW_TEST = re.compile(rf"^tests/(?P<module>{_MODULE})/(?P<stem>{_TC_STEM})\.py$")
_ALLOW_TEST_INIT = re.compile(rf"^tests/(?P<module>{_MODULE})/__init__\.py$")
_ALLOW_ROBOT = re.compile(rf"^tests_robot/(?P<module>{_MODULE})/(?P<stem>{_TC_STEM})\.robot$")
_ALLOW_WRAPPER = re.compile(rf"^test_lib/(?P<module>{_MODULE})/(?P=module)_wrappers\.py$")
_ALLOW_HELPER = re.compile(rf"^test_lib/(?P<module>{_MODULE})/(?P=module)_helpers\.py$")
_ALLOW_LIB_INIT = re.compile(rf"^test_lib/(?P<module>{_MODULE})/__init__\.py$")


class FileTransactionError(RuntimeError):
    """A transaction could not safely complete or recover."""


class PathPolicyError(FileTransactionError):
    """A path or identity is outside the Stage-4 direct-write policy."""


class FileDriftError(FileTransactionError):
    """A transaction-owned file changed after this transaction wrote it."""


class JournalStatus(StrEnum):
    OPEN = "OPEN"
    SEALED = "SEALED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class ClassifiedPath:
    """A validated framework-relative path and its deterministic role."""

    relative_path: str
    role: GeneratedFileRole
    module: str
    append_allowed: bool
    create_only: bool
    tc_id: int | None = None
    stem: str | None = None


def _validate_tc_id(tc_id: int) -> int:
    if isinstance(tc_id, bool) or not isinstance(tc_id, int):
        raise PathPolicyError("tc_id must be an integer")
    if not TC_ID_MIN <= tc_id <= TC_ID_MAX:
        raise PathPolicyError(f"tc_id must be between {TC_ID_MIN} and {TC_ID_MAX}: {tc_id}")
    return tc_id


def _validate_journal_component(value: str, label: str) -> str:
    """Reject path syntax and Windows device names in journal components."""
    if not isinstance(value, str) or not _JOURNAL_COMPONENT.fullmatch(value):
        raise PathPolicyError(f"{label} must be a safe 1-128 character path component: {value!r}")
    device_stem = value.rstrip(". ").split(".", 1)[0].upper()
    if value in {".", ".."} or device_stem in _WINDOWS_RESERVED:
        raise PathPolicyError(f"unsafe {label} path component: {value!r}")
    return value


def classify_path(relative_path: str) -> ClassifiedPath:
    """Validate a path against the complete Stage-4 framework write allowlist."""
    if not relative_path or relative_path != relative_path.strip():
        raise PathPolicyError(f"empty or padded path: {relative_path!r}")
    if "\\" in relative_path or ":" in relative_path:
        raise PathPolicyError(f"backslash/drive not allowed: {relative_path!r}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or relative_path.startswith("/"):
        raise PathPolicyError(f"absolute path not allowed: {relative_path!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise PathPolicyError(f"non-canonical path not allowed: {relative_path!r}")

    for pattern, role, append_allowed, create_only in (
        (_ALLOW_TEST, "test", False, True),
        (_ALLOW_TEST_INIT, "init", False, True),
        (_ALLOW_ROBOT, "robot", False, True),
        (_ALLOW_WRAPPER, "wrapper", True, False),
        (_ALLOW_HELPER, "helper", True, False),
        (_ALLOW_LIB_INIT, "init", False, True),
    ):
        match = pattern.fullmatch(relative_path)
        if match is None:
            continue
        matched_tc = int(match.group("tc_id")) if "tc_id" in match.groupdict() else None
        if matched_tc is not None:
            _validate_tc_id(matched_tc)
        return ClassifiedPath(
            relative_path=relative_path,
            role=role,  # type: ignore[arg-type]
            module=match.group("module"),
            append_allowed=append_allowed,
            create_only=create_only,
            tc_id=matched_tc,
            stem=match.groupdict().get("stem"),
        )
    raise PathPolicyError(f"path is outside the Stage-4 write allowlist: {relative_path!r}")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    """Write and fsync a same-directory temporary file before atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@dataclass
class JournalEntry:
    """Recovery intent for one framework-file replacement."""

    relative_path: str
    role: GeneratedFileRole
    existed: bool
    original_sha256: str | None
    expected_sha256: str
    created: bool
    preimage_name: str | None = None
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "existed": self.existed,
            "original_sha256": self.original_sha256,
            "expected_sha256": self.expected_sha256,
            "created": self.created,
            "preimage_name": self.preimage_name,
            "applied": self.applied,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JournalEntry:
        expected = data.get("expected_sha256")
        if not isinstance(expected, str):
            raise FileTransactionError("journal entry is missing expected_sha256")
        return cls(
            relative_path=str(data["relative_path"]),
            role=data["role"],
            existed=bool(data["existed"]),
            original_sha256=data.get("original_sha256"),
            expected_sha256=expected,
            created=bool(data["created"]),
            preimage_name=data.get("preimage_name"),
            applied=bool(data.get("applied", False)),
        )


@dataclass
class FileChangeJournal:
    """Scoped recovery journal for one permanent TC identity."""

    tc_id: int
    project: str
    run_id: str
    journal_dir: Path
    status: JournalStatus = JournalStatus.OPEN
    attempt: int = 1
    entries: list[JournalEntry] = field(default_factory=list)
    created_dirs: list[str] = field(default_factory=list)

    @property
    def _state_path(self) -> Path:
        return self.journal_dir / "journal.json"

    @property
    def exists(self) -> bool:
        """Whether this case already has durable journal state on disk."""
        return self._state_path.is_file()

    @property
    def _preimage_dir(self) -> Path:
        return self.journal_dir / "preimages"

    def load_or_init(self) -> None:
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._preimage_dir.mkdir(parents=True, exist_ok=True)
        if not self._state_path.exists():
            self.persist()
            return
        data = json.loads(self._state_path.read_text(encoding="utf-8"))
        expected_identity = (self.tc_id, self.project, self.run_id)
        actual_identity = (int(data["tc_id"]), str(data["project"]), str(data["run_id"]))
        if actual_identity != expected_identity:
            raise FileTransactionError(
                f"journal identity mismatch: expected {expected_identity}, found {actual_identity}"
            )
        self.status = JournalStatus(data["status"])
        self.attempt = int(data.get("attempt", 1))
        self.entries = [JournalEntry.from_dict(row) for row in data.get("entries", [])]
        self.created_dirs = [str(value) for value in data.get("created_dirs", [])]

    def persist(self) -> None:
        payload = {
            "tc_id": self.tc_id,
            "project": self.project,
            "run_id": self.run_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "entries": [entry.to_dict() for entry in self.entries],
            "created_dirs": self.created_dirs,
        }
        _atomic_replace_bytes(
            self._state_path,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )

    def store_preimage(self, entry: JournalEntry, data: bytes) -> None:
        name = (
            f"a{self.attempt:04d}_{len(self.entries):04d}_{PurePosixPath(entry.relative_path).name}"
        )
        _atomic_replace_bytes(self._preimage_dir / name, data)
        if sha256_bytes((self._preimage_dir / name).read_bytes()) != entry.original_sha256:
            raise FileTransactionError(f"preimage verification failed for {entry.relative_path}")
        entry.preimage_name = name

    def read_preimage(self, entry: JournalEntry) -> bytes:
        name = entry.preimage_name
        if not name or PurePosixPath(name).name != name or name in {".", ".."}:
            raise FileTransactionError(f"unsafe or missing preimage for {entry.relative_path}")
        data = (self._preimage_dir / name).read_bytes()
        if sha256_bytes(data) != entry.original_sha256:
            raise FileTransactionError(f"preimage hash mismatch for {entry.relative_path}")
        return data

    def validation_preimages(self) -> dict[str, bytes]:
        """Return each shared module's original, pre-transaction bytes once."""
        preimages: dict[str, bytes] = {}
        for entry in self.entries:
            if entry.role not in {"wrapper", "helper"} or not entry.existed:
                continue
            if entry.relative_path not in preimages:
                preimages[entry.relative_path] = self.read_preimage(entry)
        return preimages

    def expected_file_hashes(self) -> dict[str, str]:
        """Return the final expected hash for every path in replay order."""
        return {entry.relative_path: entry.expected_sha256 for entry in self.entries}

    def created_files(self) -> frozenset[str]:
        """Return paths that were absent at this transaction's first write."""
        return frozenset(entry.relative_path for entry in self.entries if entry.created)

    def metadata(self) -> dict[str, Any]:
        """Expose non-secret coordination metadata for persistence/validation."""
        return {
            "tc_id": self.tc_id,
            "project": self.project,
            "run_id": self.run_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "journal_dir": str(self.journal_dir),
            "entry_count": len(self.entries),
            "expected_file_hashes": self.expected_file_hashes(),
            "created_files": sorted(self.created_files()),
        }


FinalizationCheck = Callable[[FileChangeJournal], bool]


class DirectFileTransaction:
    """One case's direct writes, recoverable without any source-control operation."""

    def __init__(
        self,
        *,
        framework_root: Path,
        project: str,
        run_id: str,
        tc_id: int,
        journal_root: Path,
    ) -> None:
        root = framework_root.resolve()
        if not root.is_dir():
            raise PathPolicyError(f"framework root is not a directory: {root}")
        self.framework_root = root
        self.tc_id = _validate_tc_id(tc_id)
        safe_project = _validate_journal_component(project, "project")
        safe_run_id = _validate_journal_component(run_id, "run_id")
        resolved_journal_root = journal_root.resolve()
        journal_dir = (
            resolved_journal_root
            / safe_project
            / safe_run_id
            / "stage-4"
            / "journals"
            / str(self.tc_id)
        )
        try:
            journal_dir.resolve(strict=False).relative_to(resolved_journal_root)
        except ValueError as exc:
            raise PathPolicyError("journal path escapes its configured root") from exc
        cursor = resolved_journal_root
        for component in (safe_project, safe_run_id, "stage-4", "journals", str(self.tc_id)):
            cursor /= component
            if cursor.is_symlink():
                raise PathPolicyError(f"journal path contains a symlink: {cursor}")
        self.journal = FileChangeJournal(
            tc_id=self.tc_id,
            project=safe_project,
            run_id=safe_run_id,
            journal_dir=journal_dir,
        )
        self._begun = False

    def __enter__(self) -> DirectFileTransaction:
        self.begin()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if exc_type is not None and self.journal.status is JournalStatus.OPEN:
            self.rollback()
        return False

    def begin(self) -> None:
        self.journal.load_or_init()
        self._begun = True

    def write(self, relative_path: str, content: str | bytes) -> Path:
        """Durably record intent, then atomically replace one permitted file."""
        if not self._begun:
            raise FileTransactionError("transaction must begin before writing")
        if self.journal.status is not JournalStatus.OPEN:
            raise FileTransactionError(
                f"journal for TC {self.tc_id} is {self.journal.status.value}, not writable"
            )
        classified = classify_path(relative_path)
        self._require_matching_tc(classified)
        data = content.encode("utf-8") if isinstance(content, str) else content
        expected_hash = sha256_bytes(data)
        target = self._resolve_target(classified, check_collision=True)

        if self._is_idempotent_replay(relative_path, target, expected_hash):
            return target

        self._require_owned_rewrite_state(relative_path, target)

        existed = target.exists()
        repair_owned_create = self._open_journal_created_path(relative_path, classified)
        if existed and classified.create_only and not repair_owned_create:
            if classified.role == "init":
                initializer_bytes = target.read_bytes()
                original_hash = sha256_bytes(initializer_bytes)
                entry = JournalEntry(
                    relative_path=classified.relative_path,
                    role=classified.role,
                    existed=True,
                    original_sha256=original_hash,
                    expected_sha256=original_hash,
                    created=False,
                    applied=True,
                )
                self.journal.store_preimage(entry, initializer_bytes)
                self.journal.entries.append(entry)
                self.journal.persist()
                return target
            raise PathPolicyError(
                f"refusing to overwrite existing test file (accepted TC): {relative_path}"
            )

        original_bytes = target.read_bytes() if existed else None
        entry = JournalEntry(
            relative_path=classified.relative_path,
            role=classified.role,
            existed=existed,
            original_sha256=sha256_bytes(original_bytes) if original_bytes is not None else None,
            expected_sha256=expected_hash,
            created=not existed,
        )
        if existed:
            self.journal.store_preimage(entry, original_bytes or b"")

        missing_dirs = self._missing_parent_directories(target.parent)
        for directory in missing_dirs:
            relative_dir = directory.relative_to(self.framework_root).as_posix()
            if relative_dir not in self.journal.created_dirs:
                self.journal.created_dirs.append(relative_dir)

        # The complete recovery intent is durable before mkdir or replacement.
        self.journal.entries.append(entry)
        self.journal.persist()
        for directory in reversed(missing_dirs):
            directory.mkdir(exist_ok=True)
            self._assert_inside_framework(directory.resolve(), relative_path)

        self._atomic_write(target, data)
        actual_hash = sha256_bytes(target.read_bytes())
        if actual_hash != expected_hash:
            raise FileTransactionError(
                f"post-write verification failed for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )
        entry.applied = True
        self.journal.persist()
        return target

    def _require_matching_tc(self, classified: ClassifiedPath) -> None:
        if classified.tc_id is not None and classified.tc_id != self.tc_id:
            raise PathPolicyError(
                f"path TC {classified.tc_id} does not match transaction TC {self.tc_id}"
            )

    def _resolve_target(self, classified: ClassifiedPath, *, check_collision: bool) -> Path:
        self._reject_symlink_components(classified.relative_path)
        if check_collision:
            self._reject_component_case_collisions(classified.relative_path)
        target = (self.framework_root / classified.relative_path).resolve()
        self._assert_inside_framework(target, classified.relative_path)
        return target

    def _reject_symlink_components(self, relative_path: str) -> None:
        cursor = self.framework_root
        for part in PurePosixPath(relative_path).parts:
            cursor /= part
            if cursor.is_symlink():
                raise PathPolicyError(
                    f"writable path contains a symlink component: {relative_path}"
                )

    def _assert_inside_framework(self, target: Path, relative_path: str) -> None:
        try:
            target.relative_to(self.framework_root)
        except ValueError as exc:
            raise PathPolicyError(f"resolved path escapes framework root: {relative_path}") from exc

    def _reject_component_case_collisions(self, relative_path: str) -> None:
        cursor = self.framework_root
        for part in PurePosixPath(relative_path).parts:
            if cursor.is_dir():
                for child in cursor.iterdir():
                    if child.name != part and child.name.casefold() == part.casefold():
                        raise PathPolicyError(
                            f"case-normalization collision with '{child.name}' for {relative_path}"
                        )
            cursor /= part

    def _missing_parent_directories(self, parent: Path) -> list[Path]:
        missing: list[Path] = []
        cursor = parent
        while not cursor.exists():
            self._assert_inside_framework(cursor.resolve(), str(cursor))
            missing.append(cursor)
            cursor = cursor.parent
        return missing

    def _is_idempotent_replay(self, relative_path: str, target: Path, expected: str) -> bool:
        for entry in reversed(self.journal.entries):
            if entry.relative_path != relative_path:
                continue
            if entry.expected_sha256 != expected or not target.is_file():
                return False
            return sha256_bytes(target.read_bytes()) == expected
        return False

    def _require_owned_rewrite_state(self, relative_path: str, target: Path) -> None:
        """Reject a rewrite when bytes changed outside this open transaction.

        A durable but not-yet-applied intent may legitimately observe either
        its original state or its expected replacement after a crash. Once an
        entry is marked applied, only the transaction's last expected bytes
        are an owned rewrite base.
        """
        latest = next(
            (
                entry
                for entry in reversed(self.journal.entries)
                if entry.relative_path == relative_path
            ),
            None,
        )
        if latest is None:
            return
        if target.exists() and not target.is_file():
            raise FileDriftError(f"refusing to replace non-file target {relative_path}")
        current_hash = sha256_bytes(target.read_bytes()) if target.is_file() else None
        owned_states: set[str | None] = {latest.expected_sha256}
        if not latest.applied:
            owned_states.add(latest.original_sha256)
        if current_hash not in owned_states:
            raise FileDriftError(
                f"refusing to overwrite drifted file {relative_path}: "
                f"{current_hash} is not the transaction's expected state"
            )

    def _open_journal_created_path(self, relative_path: str, classified: ClassifiedPath) -> bool:
        """Whether this OPEN case owns a create-only target and may repair it."""
        if self.journal.status is not JournalStatus.OPEN or classified.role not in {
            "test",
            "robot",
        }:
            return False
        return any(
            entry.relative_path == relative_path and entry.created for entry in self.journal.entries
        )

    def _atomic_write(self, target: Path, data: bytes) -> None:
        _atomic_replace_bytes(target, data)

    def _verify_expected_state(self) -> None:
        final_entries: dict[str, JournalEntry] = {}
        for entry in self.journal.entries:
            final_entries[entry.relative_path] = entry
        for relative_path, entry in final_entries.items():
            classified = classify_path(relative_path)
            self._require_matching_tc(classified)
            target = self._resolve_target(classified, check_collision=False)
            if not target.is_file():
                raise FileDriftError(f"expected generated file is missing: {relative_path}")
            actual = sha256_bytes(target.read_bytes())
            if actual != entry.expected_sha256:
                raise FileDriftError(
                    f"generated file drift for {relative_path}: {actual} != {entry.expected_sha256}"
                )

    def verify_open(self) -> None:
        """Verify all current outputs while leaving the journal open."""
        if self.journal.status is not JournalStatus.OPEN:
            raise FileTransactionError("only an OPEN journal may be verified before finalization")
        self._verify_expected_state()

    def restart(self) -> None:
        """Start a new attempt after an exact completed rollback for the same TC.

        The permanent TC reservation is reused, while the prior recovery record
        remains archived inside the same TC journal directory. Preimage names
        are attempt-scoped so old rollback evidence is never overwritten.
        """
        if self.journal.status is not JournalStatus.ROLLED_BACK:
            raise FileTransactionError("only a ROLLED_BACK journal may start a new attempt")
        archive = self.journal.journal_dir / f"journal.attempt-{self.journal.attempt:04d}.json"
        if archive.exists():
            raise FileTransactionError(f"journal attempt archive already exists: {archive.name}")
        _atomic_replace_bytes(archive, self.journal._state_path.read_bytes())
        self.journal.attempt += 1
        self.journal.entries = []
        self.journal.created_dirs = []
        self.journal.status = JournalStatus.OPEN
        self.journal.persist()

    def seal(self, *, verify: bool = True) -> None:
        """Seal only after external KG/database finalization has succeeded."""
        if self.journal.status is JournalStatus.ROLLED_BACK:
            raise FileTransactionError("cannot seal a rolled-back journal")
        if self.journal.status is JournalStatus.SEALED:
            return
        if verify:
            self._verify_expected_state()
        self.journal.status = JournalStatus.SEALED
        self.journal.persist()

    def rollback(self) -> None:
        """Undo only bytes that still belong to this transaction."""
        if self.journal.status is JournalStatus.SEALED:
            raise FileTransactionError("cannot roll back a sealed (accepted) journal")
        if self.journal.status is JournalStatus.ROLLED_BACK:
            return
        for entry in reversed(self.journal.entries):
            classified = classify_path(entry.relative_path)
            self._require_matching_tc(classified)
            if entry.role != classified.role:
                raise FileTransactionError(
                    f"journal role mismatch for {entry.relative_path}: "
                    f"{entry.role} != {classified.role}"
                )
            target = self._resolve_target(classified, check_collision=False)
            current_hash = sha256_bytes(target.read_bytes()) if target.is_file() else None
            if entry.created:
                if current_hash is None:
                    # Intent was durable, but creation never happened or is already undone.
                    continue
                if current_hash != entry.expected_sha256:
                    raise FileDriftError(
                        f"refusing to delete drifted file {entry.relative_path}: "
                        f"{current_hash} != {entry.expected_sha256}"
                    )
                target.unlink()
                if target.exists():
                    raise FileTransactionError(f"rollback deletion failed: {entry.relative_path}")
                continue

            if current_hash == entry.original_sha256:
                continue  # replacement never happened, or this entry was already restored
            if current_hash != entry.expected_sha256:
                raise FileDriftError(
                    f"refusing to restore drifted file {entry.relative_path}: "
                    f"{current_hash} is neither transaction nor original content"
                )
            preimage = self.journal.read_preimage(entry)
            self._atomic_write(target, preimage)
            restored = sha256_bytes(target.read_bytes())
            if restored != entry.original_sha256:
                raise FileTransactionError(
                    f"rollback verification failed for {entry.relative_path}: "
                    f"{restored} != {entry.original_sha256}"
                )

        for relative_path in sorted(
            self.journal.created_dirs,
            key=lambda value: len(PurePosixPath(value).parts),
            reverse=True,
        ):
            pure = PurePosixPath(relative_path)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise FileTransactionError(
                    f"unsafe created-directory journal path: {relative_path}"
                )
            directory = (self.framework_root / relative_path).resolve()
            self._assert_inside_framework(directory, relative_path)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        self.journal.status = JournalStatus.ROLLED_BACK
        self.journal.persist()

    def reconcile(
        self,
        *,
        externally_finalized: bool = False,
        finalization_check: FinalizationCheck | None = None,
    ) -> JournalStatus:
        """Resolve an OPEN journal using durable external state when available.

        The orchestration layer can pass either a known finalization result or a
        read-only callback that checks PostgreSQL/KG state.  A finalized case is
        hash-verified and sealed; only a non-finalized case is rolled back.
        """
        self.begin()
        if self.journal.status is not JournalStatus.OPEN:
            return self.journal.status
        finalized = externally_finalized
        if finalization_check is not None:
            finalized = bool(finalization_check(self.journal))
        if finalized:
            self.seal()
        elif self.journal.entries:
            self.rollback()
        return self.journal.status


__all__ = [
    "ClassifiedPath",
    "DirectFileTransaction",
    "FileChangeJournal",
    "FileDriftError",
    "FileTransactionError",
    "FinalizationCheck",
    "JournalEntry",
    "JournalStatus",
    "PathPolicyError",
    "classify_path",
    "sha256_bytes",
]
