"""Canonical code-property-graph contracts for the Stage-4 framework code graph.

A code-property graph is a directed, edge-labelled, attributed multigraph
(plan §6.3): several distinct relationships may validly exist between the same
two symbols. These are the canonical shapes the ``GraphifyAdapter`` produces and
the ``CodeGraphStore`` persists — deliberately independent of Graphify's own
node/edge model so the extractor can be swapped behind the adapter (§3.3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from multi_agentic_graph_rag.domain.schemas import StrictModel

CodeLanguage = Literal["python"]

# Structural symbol kinds. Specialized labels layered on CodeSymbol (plan §6.2).
SymbolKind = Literal[
    "Module",
    "Class",
    "Function",
    "Method",
    "Variable",
    "Parameter",
    "Fixture",
    "Test",
    "Decorator",
    "Constant",
]

# Directed, labelled code relationships (plan §6.3).
CodeRelation = Literal[
    "CONTAINS",
    "DECLARES",
    "IMPORTS",
    "CALLS",
    "REFERENCES",
    "INHERITS",
    "IMPLEMENTS",
    "DECORATED_BY",
    "READS",
    "WRITES",
    "RETURNS",
    "HAS_PARAMETER",
    "DEPENDS_ON",
    "TESTS",
    "PROVIDES_CAPABILITY",
]

# Graphify-style relationship confidence labels (plan §3.1).
ExtractionConfidence = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]

Visibility = Literal["public", "protected", "private"]


class CodeFile(StrictModel):
    """One indexed source file within a framework snapshot (plan §6.2)."""

    snapshot_id: str
    relative_path: str
    language: CodeLanguage
    content_hash: str


class CodeSymbol(StrictModel):
    """A snapshot-scoped symbol with an exact, hash-pinned source span.

    ``symbol_id`` is derived from the snapshot, FQN, kind, and byte span so that
    short names like ``start`` are never merged globally (plan §6.6).
    """

    snapshot_id: str
    symbol_id: str
    fqn: str
    kind: SymbolKind
    signature: str | None = None
    visibility: Visibility = "public"
    relative_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_byte: int = Field(ge=0)
    end_byte: int = Field(ge=0)
    body_hash: str

    @model_validator(mode="after")
    def spans_are_ordered(self) -> CodeSymbol:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_byte < self.start_byte:
            raise ValueError("end_byte must be >= start_byte")
        return self


class CodeEdge(StrictModel):
    """One directed, labelled relationship between two symbols (plan §6.3)."""

    edge_id: str
    snapshot_id: str
    source_symbol_id: str
    target_symbol_id: str
    relation: CodeRelation
    confidence: ExtractionConfidence
    provenance: str
    source_location: str
    extractor: str


class PackageDependency(StrictModel):
    """A declared package dependency of the framework (plan §6.2)."""

    name: str
    version_constraint: str | None = None
    source: str


class CodeExtractionResult(StrictModel):
    """Validated canonical output of one extraction pass over a snapshot.

    Enforces per-snapshot identity uniqueness and referential integrity so a
    snapshot is never published with dangling edges (plan §7.2 steps 11/15).
    """

    snapshot_id: str
    files: list[CodeFile] = Field(default_factory=list)
    symbols: list[CodeSymbol] = Field(default_factory=list)
    edges: list[CodeEdge] = Field(default_factory=list)
    dependencies: list[PackageDependency] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> CodeExtractionResult:
        if any(f.snapshot_id != self.snapshot_id for f in self.files):
            raise ValueError("all files must share the result snapshot_id")
        if any(s.snapshot_id != self.snapshot_id for s in self.symbols):
            raise ValueError("all symbols must share the result snapshot_id")
        paths = {f.relative_path for f in self.files}
        if len(paths) != len(self.files):
            raise ValueError("duplicate CodeFile relative_path within snapshot")
        symbol_ids = {s.symbol_id for s in self.symbols}
        if len(symbol_ids) != len(self.symbols):
            raise ValueError("duplicate CodeSymbol symbol_id within snapshot")
        for symbol in self.symbols:
            if symbol.relative_path not in paths:
                raise ValueError(f"symbol {symbol.symbol_id} references unknown file")
        edge_ids = {e.edge_id for e in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("duplicate CodeEdge edge_id within snapshot")
        for edge in self.edges:
            if edge.snapshot_id != self.snapshot_id:
                raise ValueError("all edges must share the result snapshot_id")
            if edge.source_symbol_id not in symbol_ids:
                raise ValueError(f"edge {edge.edge_id} has a dangling source symbol")
            if edge.target_symbol_id not in symbol_ids:
                raise ValueError(f"edge {edge.edge_id} has a dangling target symbol")
        return self


class FrameworkSnapshot(StrictModel):
    """Immutable framework snapshot identity plus its Git provenance (plan §6.2)."""

    snapshot_id: str
    repository_id: str
    canonical_path: str
    branch: str
    commit: str
    tree_hash: str
    dirty: bool
    dirty_hash: str
    extractor_version: str
    extractor_config_hash: str
    language: CodeLanguage = "python"
    status: Literal["building", "ready", "failed"] = "building"


__all__ = [
    "CodeEdge",
    "CodeExtractionResult",
    "CodeFile",
    "CodeLanguage",
    "CodeRelation",
    "CodeSymbol",
    "ExtractionConfidence",
    "FrameworkSnapshot",
    "PackageDependency",
    "SymbolKind",
    "Visibility",
]
