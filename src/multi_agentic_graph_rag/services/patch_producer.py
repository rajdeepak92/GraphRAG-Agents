"""The reasoning-model patch seam for Stage-4 code generation (plan §13.1, §23).

The apply loop drives generation through a stable ``PatchProducer`` contract
rather than calling a model directly, so the deterministic controls (worktree
isolation, hash guards, validation) are exercised the same way whether the
producer is a live coding model or a fixture. A real producer formats the
redacted bundle + context manifest into a prompt and parses the model's tool
calls into ``FileOp`` values; it never receives raw secrets (§8.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

FileOpKind = Literal["create", "patch"]


@dataclass(frozen=True)
class FileOp:
    """One controlled write the producer proposes (applied via PatchExecutor)."""

    kind: FileOpKind
    relative_path: str
    content: str
    expected_hash: str | None = None  # required for 'patch' (hash guard, §10.6)


@dataclass(frozen=True)
class PatchRequest:
    """Bounded, redacted context handed to the producer for one scenario."""

    scenario_id: str
    execution_profile_id: str
    resolved_bundle_id: str | None
    context_manifest_id: str | None
    capability_bindings: list[dict[str, Any]] = field(default_factory=list)
    redacted_bundle: dict[str, Any] = field(default_factory=dict)
    repair_diagnostics: tuple[str, ...] = ()


@runtime_checkable
class PatchProducer(Protocol):
    """Stable contract for producing file operations for a scenario (LLM seam)."""

    def produce(self, request: PatchRequest) -> list[FileOp]: ...


class StaticPatchProducer:
    """Deterministic producer that replays preconfigured file ops.

    Used for tests and for scaffolding the apply loop without a live model. A
    live implementation replaces this with one that prompts the reasoning model
    and parses its tool calls into the same ``FileOp`` values.
    """

    def __init__(self, ops: list[FileOp]) -> None:
        self._ops = list(ops)

    def produce(self, request: PatchRequest) -> list[FileOp]:
        return list(self._ops)


__all__ = ["FileOp", "FileOpKind", "PatchProducer", "PatchRequest", "StaticPatchProducer"]
