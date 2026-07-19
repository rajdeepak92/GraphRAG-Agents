"""Provider-neutral Stage-4 planning and generated-file seam.

The workflow owns identity, paths, transactions, validation, persistence, and KG
publication.  A producer owns only the reasoning decisions and structured file
contents.  Production uses :class:`LLMPatchProducer`; unit tests use the static
fixture below.  The legacy ``produce`` method remains temporarily available so
the existing apply graph can be rewired without breaking unrelated tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from multi_agentic_graph_rag.domain.codegen_schemas import (
    GeneratedPatchBundle,
    TestCaseImplementationPlan,
)

FileOpKind = str


@dataclass(frozen=True)
class FileOp:
    """Legacy complete-content operation used by the old apply graph."""

    kind: FileOpKind
    relative_path: str
    content: str
    expected_hash: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"create", "patch"}:
            raise ValueError(f"unsupported file operation kind: {self.kind}")


@dataclass(frozen=True)
class PatchRequest:
    """Immutable, provider-neutral input for one logical test case.

    ``context`` is the complete bounded result returned by
    ``CodegenContextRetriever``.  Secret values must never be present, but
    unresolved ``secret://`` references are intentionally retained so generated
    code can resolve them at controlled test runtime.
    """

    scenario_id: str
    execution_profile_id: str
    resolved_bundle_id: str | None
    context_manifest_id: str | None
    tc_id: int | None = None
    variant_id: str = "default"
    capability_bindings: list[dict[str, Any]] = field(default_factory=list)
    redacted_bundle: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    repair_diagnostics: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, Any]:
        """Return a JSON-ready payload without removing ``secret://`` references."""

        return {
            "scenario_id": self.scenario_id,
            "execution_profile_id": self.execution_profile_id,
            "variant_id": self.variant_id,
            "tc_id": self.tc_id,
            "resolved_bundle_id": self.resolved_bundle_id,
            "context_manifest_id": self.context_manifest_id,
            "capability_bindings": self.capability_bindings,
            "resolved_test_data": self.redacted_bundle,
            "context": self.context,
        }


@runtime_checkable
class PatchProducer(Protocol):
    """The single reasoning seam used by the Stage-4 workflow."""

    def plan(self, request: PatchRequest) -> TestCaseImplementationPlan:
        """Return the structured implementation plan for one case."""

    def generate_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
    ) -> GeneratedPatchBundle:
        """Return complete contents for every file in ``plan``."""

    def repair_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
        current_bundle: GeneratedPatchBundle,
        diagnostics: tuple[str, ...],
    ) -> GeneratedPatchBundle:
        """Repair one current-case bundle using bounded diagnostics."""

    def produce(self, request: PatchRequest) -> list[FileOp]:
        """Legacy compatibility method; new workflows use structured methods."""


class StaticPatchProducer:
    """Deterministic structured producer for unit tests.

    Positional ``ops`` preserves compatibility with the former fixture.  New
    tests normally provide ``implementation_plan`` and ``bundle``.
    """

    def __init__(
        self,
        ops: list[FileOp] | None = None,
        *,
        implementation_plan: TestCaseImplementationPlan | None = None,
        bundle: GeneratedPatchBundle | None = None,
        repairs: list[GeneratedPatchBundle] | None = None,
    ) -> None:
        self._ops = list(ops or [])
        self._plan = implementation_plan
        self._bundle = bundle
        self._repairs = list(repairs or [])

    def plan(self, request: PatchRequest) -> TestCaseImplementationPlan:
        del request
        if self._plan is None:
            raise RuntimeError("StaticPatchProducer has no implementation plan")
        return self._plan

    def generate_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
    ) -> GeneratedPatchBundle:
        del request, plan
        if self._bundle is None:
            raise RuntimeError("StaticPatchProducer has no generated bundle")
        return self._bundle

    def repair_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
        current_bundle: GeneratedPatchBundle,
        diagnostics: tuple[str, ...],
    ) -> GeneratedPatchBundle:
        del request, plan, current_bundle, diagnostics
        if self._repairs:
            return self._repairs.pop(0)
        if self._bundle is None:
            raise RuntimeError("StaticPatchProducer has no repair bundle")
        return self._bundle

    def produce(self, request: PatchRequest) -> list[FileOp]:
        del request
        return list(self._ops)


__all__ = [
    "FileOp",
    "FileOpKind",
    "PatchProducer",
    "PatchRequest",
    "StaticPatchProducer",
]
