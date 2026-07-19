"""Adapter that turns a Graphify extraction into the canonical code graph.

Graphify is used strictly as an extraction engine behind a stable contract
(plan §3.3); its own Neo4j exporter is *not* used as the Stage-4 datastore.
Graphify's AST extraction is deterministic but coarse — it does not always
supply exact signatures or byte spans — so this adapter maps its nodes/edges
into the canonical schemas and computes real content/body hashes from the live
worktree, which remains authoritative for source bodies (plan §3.2, §7.2 s.12).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from multi_agentic_graph_rag.domain.code_graph_schemas import (
    CodeEdge,
    CodeExtractionResult,
    CodeFile,
    CodeRelation,
    CodeSymbol,
    ExtractionConfidence,
    PackageDependency,
    SymbolKind,
)
from multi_agentic_graph_rag.domain.identifiers import make_code_edge_id, make_code_symbol_id

_LINE = re.compile(r"L(\d+)")

# Graphify relation label -> canonical code relation. Non-structural relations
# (rationale_for, conceptually_related_to, semantically_similar_to) are dropped;
# depends_on is projected as a PackageDependency instead of a symbol edge.
_RELATION_MAP: dict[str, CodeRelation] = {
    "calls": "CALLS",
    "indirect_call": "CALLS",
    "uses": "REFERENCES",
    "references": "REFERENCES",
    "contains": "CONTAINS",
    "method": "DECLARES",
    "declares": "DECLARES",
    "inherits": "INHERITS",
    "implements": "IMPLEMENTS",
    "imports": "IMPORTS",
    "imports_from": "IMPORTS",
    "decorated_by": "DECORATED_BY",
    "returns": "RETURNS",
}

_VALID_CONFIDENCE: frozenset[str] = frozenset({"EXTRACTED", "INFERRED", "AMBIGUOUS"})


@runtime_checkable
class CodeExtractionAdapter(Protocol):
    """Stable Stage-4 extraction contract (plan §3.3)."""

    def extract_snapshot(
        self,
        repository_root: Path,
        snapshot_id: str,
        changed_files: list[Path] | None,
    ) -> CodeExtractionResult: ...


class GraphifyExtractionError(RuntimeError):
    """Raised when a Graphify extraction cannot be loaded or is malformed."""


def _confidence(value: Any) -> ExtractionConfidence:
    text = str(value or "EXTRACTED").upper()
    return text if text in _VALID_CONFIDENCE else "EXTRACTED"  # type: ignore[return-value]


def _parse_span(source_location: str | None) -> tuple[int, int]:
    matches = _LINE.findall(source_location or "")
    if not matches:
        return 1, 1
    lines = [int(value) for value in matches]
    return min(lines), max(lines)


def _infer_kind(label: str) -> SymbolKind:
    name = label.strip()
    if name.endswith("()"):
        bare = name[:-2]
        return "Method" if "." in bare else "Function"
    if name and name[0].isupper() and name.replace("_", "").isupper():
        return "Constant"
    if name[:1].isupper():
        return "Class"
    return "Variable"


def _is_code(node: Mapping[str, Any]) -> bool:
    return node.get("file_type") == "code"


def _is_file(node: Mapping[str, Any]) -> bool:
    return str(node.get("label", "")).lower().endswith((".py", ".robot"))


def _sha256(text: bytes) -> str:
    return f"sha256:{hashlib.sha256(text).hexdigest()}"


class GraphifyAdapter:
    """Transform a Graphify extraction dict into a validated code graph."""

    def __init__(self, *, extractor_version: str, graphify_out_dir: Path | None = None) -> None:
        self.extractor_version = extractor_version
        self.graphify_out_dir = graphify_out_dir

    def load_extraction(self, graphify_out_dir: Path | None = None) -> Mapping[str, Any]:
        """Load Graphify's deterministic AST extraction output."""
        base = graphify_out_dir or self.graphify_out_dir
        if base is None:
            raise GraphifyExtractionError("no graphify output directory configured")
        for candidate in (".graphify_extract.json", ".graphify_ast.json", "graph.json"):
            path = base / candidate
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, Mapping) and "nodes" in data:
                    return data
        raise GraphifyExtractionError(f"no usable graphify extraction found under {base}")

    def transform(
        self,
        extraction: Mapping[str, Any],
        *,
        snapshot_id: str,
        repository_root: Path,
    ) -> CodeExtractionResult:
        """Map Graphify nodes/edges into canonical, hash-verified code records."""
        nodes: list[Mapping[str, Any]] = list(extraction.get("nodes", []))
        raw_edges = extraction.get("edges") or extraction.get("links", [])
        edges: list[Mapping[str, Any]] = list(raw_edges)
        node_by_id = {str(node["id"]): node for node in nodes if "id" in node}

        file_cache: dict[str, str] = {}
        files = self._build_files(nodes, snapshot_id, repository_root, file_cache)
        known_paths = {file.relative_path for file in files}
        symbols, symbol_id_by_node = self._build_symbols(
            nodes, snapshot_id, repository_root, known_paths, file_cache
        )
        code_edges = self._build_edges(edges, snapshot_id, symbol_id_by_node)
        dependencies = self._build_dependencies(edges, node_by_id)
        return CodeExtractionResult(
            snapshot_id=snapshot_id,
            files=files,
            symbols=symbols,
            edges=code_edges,
            dependencies=dependencies,
        )

    def extract_snapshot(
        self,
        repository_root: Path,
        snapshot_id: str,
        changed_files: list[Path] | None = None,
    ) -> CodeExtractionResult:
        """Load and transform the current Graphify extraction (adapter contract)."""
        extraction = self.load_extraction()
        return self.transform(extraction, snapshot_id=snapshot_id, repository_root=repository_root)

    # --- internals -----------------------------------------------------------

    def _read_file(self, repository_root: Path, relative_path: str, cache: dict[str, str]) -> str:
        if relative_path not in cache:
            root = repository_root.resolve()
            path = (repository_root / relative_path).resolve()
            if path != root and root not in path.parents:
                raise GraphifyExtractionError(f"path escapes repository: {relative_path}")
            if not path.exists():
                raise GraphifyExtractionError(f"indexed file missing: {relative_path}")
            cache[relative_path] = path.read_text(encoding="utf-8", errors="replace")
        return cache[relative_path]

    def _build_files(
        self,
        nodes: Iterable[Mapping[str, Any]],
        snapshot_id: str,
        repository_root: Path,
        cache: dict[str, str],
    ) -> list[CodeFile]:
        files: dict[str, CodeFile] = {}
        for node in nodes:
            if not _is_code(node) or not _is_file(node):
                continue
            relative_path = str(node.get("source_file", "")).replace("\\", "/")
            if not relative_path or relative_path in files:
                continue
            content = self._read_file(repository_root, relative_path, cache)
            files[relative_path] = CodeFile(
                snapshot_id=snapshot_id,
                relative_path=relative_path,
                language="robot" if relative_path.lower().endswith(".robot") else "python",
                content_hash=_sha256(content.encode("utf-8")),
            )
        return list(files.values())

    def _build_symbols(
        self,
        nodes: Iterable[Mapping[str, Any]],
        snapshot_id: str,
        repository_root: Path,
        known_paths: set[str],
        cache: dict[str, str],
    ) -> tuple[list[CodeSymbol], dict[str, str]]:
        symbols: list[CodeSymbol] = []
        symbol_id_by_node: dict[str, str] = {}
        seen: set[str] = set()
        for node in nodes:
            if not _is_code(node) or _is_file(node) or node.get("type") == "package":
                continue
            relative_path = str(node.get("source_file", "")).replace("\\", "/")
            if relative_path not in known_paths:
                continue
            # Robot suites are represented as CodeFile artifacts. Graphify does
            # not expose a stable Python-style symbol model for them.
            if relative_path.lower().endswith(".robot"):
                continue
            label = str(node.get("label", "")).strip()
            if not label:
                continue
            start_line, end_line = _parse_span(node.get("source_location"))
            content_lines = self._read_file(repository_root, relative_path, cache).splitlines()
            body = "\n".join(content_lines[start_line - 1 : end_line]) or label
            fqn = f"{relative_path}::{label}"
            kind = _infer_kind(label)
            symbol_id = make_code_symbol_id(
                snapshot_id=snapshot_id,
                fqn=fqn,
                kind=kind,
                start_byte=start_line,
                end_byte=end_line,
            )
            if symbol_id in seen:
                symbol_id_by_node[str(node["id"])] = symbol_id
                continue
            seen.add(symbol_id)
            symbol_id_by_node[str(node["id"])] = symbol_id
            symbols.append(
                CodeSymbol(
                    snapshot_id=snapshot_id,
                    symbol_id=symbol_id,
                    fqn=fqn,
                    kind=kind,
                    signature=label if label.endswith("()") else None,
                    relative_path=relative_path,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=0,
                    end_byte=0,
                    body_hash=_sha256(body.encode("utf-8")),
                )
            )
        return symbols, symbol_id_by_node

    def _build_edges(
        self,
        edges: Iterable[Mapping[str, Any]],
        snapshot_id: str,
        symbol_id_by_node: dict[str, str],
    ) -> list[CodeEdge]:
        code_edges: list[CodeEdge] = []
        seen: set[str] = set()
        for edge in edges:
            relation = _RELATION_MAP.get(str(edge.get("relation") or edge.get("type", "")))
            if relation is None:
                continue
            source = symbol_id_by_node.get(str(edge.get("source", "")))
            target = symbol_id_by_node.get(str(edge.get("target", "")))
            if source is None or target is None:
                continue
            source_location = str(edge.get("source_location", ""))
            edge_id = make_code_edge_id(
                snapshot_id=snapshot_id,
                source_symbol_id=source,
                relation=relation,
                target_symbol_id=target,
                source_location=source_location,
            )
            if edge_id in seen:
                continue
            seen.add(edge_id)
            code_edges.append(
                CodeEdge(
                    edge_id=edge_id,
                    snapshot_id=snapshot_id,
                    source_symbol_id=source,
                    target_symbol_id=target,
                    relation=relation,
                    confidence=_confidence(edge.get("confidence")),
                    provenance=str(edge.get("_origin", "ast")),
                    source_location=source_location,
                    extractor=self.extractor_version,
                )
            )
        return code_edges

    def _build_dependencies(
        self,
        edges: Iterable[Mapping[str, Any]],
        node_by_id: Mapping[str, Mapping[str, Any]],
    ) -> list[PackageDependency]:
        dependencies: dict[str, PackageDependency] = {}
        for edge in edges:
            if str(edge.get("relation")) != "depends_on":
                continue
            target_node = node_by_id.get(str(edge.get("target", "")))
            if target_node is None:
                continue
            name = str(target_node.get("label", "")).strip()
            if not name or name in dependencies:
                continue
            version = target_node.get("version")
            dependencies[name] = PackageDependency(
                name=name,
                version_constraint=str(version) if version else None,
                source=str(edge.get("source_file", "")),
            )
        return list(dependencies.values())


__all__ = [
    "CodeExtractionAdapter",
    "GraphifyAdapter",
    "GraphifyExtractionError",
]
