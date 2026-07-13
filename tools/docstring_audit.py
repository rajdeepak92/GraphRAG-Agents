"""Audit production docstring coverage and Google-style signature consistency."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_TRIVIAL_DUNDERS = {
    "__enter__",
    "__exit__",
    "__iter__",
    "__len__",
    "__repr__",
    "__str__",
}
_ARG_ENTRY = re.compile(r"^\s{4}([*]*[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]+)\))?:")


@dataclass(frozen=True)
class Finding:
    """Describe one actionable documentation policy violation."""

    path: Path
    line: int
    symbol: str
    reason: str

    def render(self, root: Path) -> str:
        """Render the finding with a repository-relative path and source line."""
        return f"{self.path.relative_to(root)}:{self.line}: {self.symbol}: {self.reason}"


def _decorator_name(node: ast.expr) -> str:
    """Return the final identifier for a decorator expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _is_protocol_class(node: ast.ClassDef) -> bool:
    """Return whether a class explicitly derives from ``Protocol``."""
    return any(
        (isinstance(base, ast.Name) and base.id == "Protocol")
        or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
        for base in node.bases
    )


def _is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef, parent: ast.AST | None) -> bool:
    """Return whether a callable is an overload or protocol-only stub."""
    if any(_decorator_name(item) == "overload" for item in node.decorator_list):
        return True
    if isinstance(parent, ast.ClassDef) and _is_protocol_class(parent):
        return len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Expr))
    return False


def _is_trivial_dunder(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether an approved simple dunder may omit its own docstring."""
    return node.name in _TRIVIAL_DUNDERS and len(node.body) == 1


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Return documented parameters in declaration order, excluding ``self`` and ``cls``."""
    args = [*node.args.posonlyargs, *node.args.args]
    if args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    args.extend(node.args.kwonlyargs)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    return args


def _is_trivial_callable(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a complete callable contract reasonably fits on one line."""
    meaningful = [item for item in node.body if not isinstance(item, ast.Pass)]
    return len(meaningful) <= 1 and not any(
        isinstance(item, (ast.Raise, ast.Yield, ast.YieldFrom, ast.Await))
        for item in ast.walk(node)
    )


def _section_entries(docstring: str, section: str) -> dict[str, str | None]:
    """Parse parameter names and optional types from one Google-style section."""
    lines = docstring.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == f"{section}:") + 1
    except StopIteration:
        return {}
    entries: dict[str, str | None] = {}
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        match = _ARG_ENTRY.match(line)
        if match:
            entries[match.group(1).lstrip("*")] = match.group(2)
    return entries


def _annotation_text(annotation: ast.expr | None) -> str | None:
    """Return a normalized source representation for a type annotation."""
    if (
        isinstance(annotation, ast.Subscript)
        and ast.unparse(annotation.value).endswith("Annotated")
        and isinstance(annotation.slice, ast.Tuple)
        and annotation.slice.elts
    ):
        annotation = annotation.slice.elts[0]
    return ast.unparse(annotation).replace("typing.", "") if annotation is not None else None


def _audit_callable(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent: ast.AST | None,
) -> list[Finding]:
    """Audit coverage and signature consistency for one production callable."""
    if _is_stub(node, parent) or _is_trivial_dunder(node):
        return []
    docstring = ast.get_docstring(node, clean=True)
    if not docstring:
        return [Finding(path, node.lineno, node.name, "missing callable docstring")]
    if _is_trivial_callable(node):
        return []
    findings: list[Finding] = []
    parameters = _parameters(node)
    entries = _section_entries(docstring, "Args")
    expected_names = {parameter.arg for parameter in parameters}
    for parameter in parameters if entries else []:
        if parameter.arg not in entries:
            findings.append(
                Finding(path, node.lineno, node.name, f"Args omits parameter {parameter.arg!r}")
            )
            continue
        expected_type = _annotation_text(parameter.annotation)
        documented_type = entries[parameter.arg]
        if expected_type and documented_type and documented_type != expected_type:
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    node.name,
                    f"parameter {parameter.arg!r} type is {documented_type!r}, "
                    f"expected {expected_type!r}",
                )
            )
    for stale_name in sorted(set(entries) - expected_names):
        findings.append(
            Finding(path, node.lineno, node.name, f"Args contains stale parameter {stale_name!r}")
        )
    return_annotation = _annotation_text(node.returns)
    if (
        entries
        and return_annotation not in {None, "None"}
        and "Returns:" not in docstring
        and "Yields:" not in docstring
    ):
        findings.append(Finding(path, node.lineno, node.name, "missing Returns or Yields section"))
    return findings


def audit(root: Path) -> tuple[dict[str, int], list[Finding]]:
    """Audit every project-owned production module below ``root``.

    Args:
        root (Path): Repository root containing ``src/multi_agentic_graph_rag``.

    Returns:
        tuple[dict[str, int], list[Finding]]: Symbol totals and ordered findings.
    """
    source_root = root / "src" / "multi_agentic_graph_rag"
    totals = {"modules": 0, "classes": 0, "functions": 0}
    findings: list[Finding] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        totals["modules"] += 1
        if ast.get_docstring(tree) is None and path.name != "__init__.py":
            findings.append(Finding(path, 1, "<module>", "missing module docstring"))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                totals["classes"] += 1
                if ast.get_docstring(node) is None:
                    findings.append(
                        Finding(path, node.lineno, node.name, "missing class docstring")
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                totals["functions"] += 1
                findings.extend(_audit_callable(path, node, parents.get(node)))
    return totals, sorted(findings, key=lambda item: (str(item.path), item.line, item.reason))


def main(argv: list[str] | None = None) -> int:
    """Run the repository audit and return a process-compatible status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    totals, findings = audit(root)
    print(
        "Docstring audit: "
        f"modules={totals['modules']} classes={totals['classes']} functions={totals['functions']}"
    )
    for finding in findings:
        print(finding.render(root))
    if findings:
        print(f"FAILED: {len(findings)} documentation violation(s)", file=sys.stderr)
        return 1
    print("PASS: production docstring coverage and signature consistency are complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
