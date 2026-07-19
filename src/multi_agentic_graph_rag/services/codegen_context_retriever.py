"""Complete, bounded Stage-4 context retrieval for one frozen scenario."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from multi_agentic_graph_rag.domain.code_graph_schemas import CodeSymbol, FrameworkSnapshot
from multi_agentic_graph_rag.domain.test_data_schemas import (
    NormalizedTestData,
    ScenarioDataBinding,
    TestDataRecord,
)
from multi_agentic_graph_rag.services.scenario_data_binder import resolve_binding

_NEIGHBOR_RELATIONS = ("CALLS", "DECLARES", "CONTAINS")
_SECRET_PREFIX = "secret://"
_SEARCH_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class ContextRetrievalError(RuntimeError):
    """Required immutable context is missing, ambiguous, or stale."""


class LineageStore(Protocol):
    """Stage 1-3 artifact reads required by Stage 4."""

    def load_test_scenarios(self, project: str, run_id: str) -> Any | None: ...

    def load_user_stories(self, project: str, run_id: str) -> Any | None: ...

    def load_requirements(self, project: str, run_id: str) -> Any | None: ...


class EvidenceStore(Protocol):
    """Authoritative BRD/SRS chunk lookup."""

    def fetch_chunks(self, project: str, chunk_ids: set[str]) -> list[tuple[str, str]]: ...


class CodeGraphReader(Protocol):
    """READY framework-snapshot query surface."""

    def get_symbol(self, snapshot_id: str, symbol_id: str) -> CodeSymbol | None: ...

    def search_symbols(
        self,
        snapshot_id: str,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CodeSymbol]: ...

    def get_neighbors(
        self,
        snapshot_id: str,
        symbol_id: str,
        *,
        relations: list[str] | None = None,
        depth: int = 1,
    ) -> list[CodeSymbol]: ...


@dataclass(frozen=True)
class RetrievedCodeContext:
    """Exact source body for one matched, neighboring, or test_lib symbol."""

    symbol_id: str
    fqn: str
    kind: str
    signature: str | None
    relative_path: str
    start_line: int
    end_line: int
    body_hash: str
    source_body: str
    relations: tuple[str, ...] = ()

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "fqn": self.fqn,
            "kind": self.kind,
            "signature": self.signature,
            "relative_path": self.relative_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "body_hash": self.body_hash,
            "source_body": self.source_body,
            "relations": list(self.relations),
        }


@dataclass(frozen=True)
class ScenarioCodegenContext:
    """Prompt-ready context with mandatory lineage/data retained regardless of budget."""

    project: str
    run_id: str
    scenario_id: str
    execution_profile_id: str
    variant_id: str
    scenario: dict[str, Any]
    stories: tuple[dict[str, Any], ...]
    requirements: tuple[dict[str, Any], ...]
    evidence_chunk_ids: tuple[str, ...]
    evidence_chunks: tuple[dict[str, str], ...]
    resolved_binding: dict[str, Any]
    test_data_snapshot: dict[str, Any]
    test_data_records: tuple[dict[str, Any], ...]
    secret_refs: tuple[str, ...]
    framework_snapshot: dict[str, Any]
    matched_symbols: tuple[RetrievedCodeContext, ...]
    test_lib_symbols: tuple[RetrievedCodeContext, ...]
    neighbor_symbols: tuple[RetrievedCodeContext, ...]
    token_budget: int
    mandatory_tokens: int
    selected_tokens: int
    mandatory_over_budget: bool
    dropped_items: dict[str, int]

    @property
    def framework_snapshot_id(self) -> str:
        return str(self.framework_snapshot["snapshot_id"])

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return the complete JSON-ready retrieval bundle for a model call."""

        return {
            "project": self.project,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "execution_profile_id": self.execution_profile_id,
            "variant_id": self.variant_id,
            "scenario": self.scenario,
            "stories": list(self.stories),
            "requirements": list(self.requirements),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "evidence_chunks": list(self.evidence_chunks),
            "resolved_binding": self.resolved_binding,
            "test_data_snapshot": self.test_data_snapshot,
            "test_data_records": list(self.test_data_records),
            "secret_refs": list(self.secret_refs),
            "framework_snapshot": self.framework_snapshot,
            "matched_symbols": [item.to_prompt_payload() for item in self.matched_symbols],
            "test_lib_symbols": [item.to_prompt_payload() for item in self.test_lib_symbols],
            "neighbor_symbols": [item.to_prompt_payload() for item in self.neighbor_symbols],
            "budget": {
                "token_budget": self.token_budget,
                "mandatory_tokens": self.mandatory_tokens,
                "selected_tokens": self.selected_tokens,
                "mandatory_over_budget": self.mandatory_over_budget,
                "dropped_items": self.dropped_items,
            },
        }


@dataclass(frozen=True)
class _BudgetCandidate:
    tier: int
    group: str
    key: str
    payload: dict[str, Any]
    value: RetrievedCodeContext | dict[str, str]


class CodegenContextRetriever:
    """Retrieve complete lineage/data plus budgeted exact framework context."""

    def __init__(
        self,
        *,
        lineage_store: LineageStore,
        evidence_store: EvidenceStore,
        code_graph: CodeGraphReader,
        framework_root: Path,
    ) -> None:
        self._lineage = lineage_store
        self._evidence = evidence_store
        self._code_graph = code_graph
        self._framework_root = framework_root.resolve()

    def scenario_codegen_context(
        self,
        *,
        project: str,
        run_id: str,
        scenario_id: str,
        execution_profile_id: str,
        normalized_test_data: NormalizedTestData,
        framework_snapshot: FrameworkSnapshot,
        variant_id: str = "default",
        binding_id: str | None = None,
        matched_symbol_ids: tuple[str, ...] = (),
        symbol_queries: tuple[str, ...] = (),
        module_hints: tuple[str, ...] = (),
        token_budget: int = 24_000,
        symbol_search_limit: int = 20,
        max_context_symbols: int = 40,
    ) -> ScenarioCodegenContext:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if framework_snapshot.status.casefold() != "ready":
            raise ContextRetrievalError("framework snapshot is not READY")
        if normalized_test_data.project != project:
            raise ContextRetrievalError("test-data snapshot belongs to a different project")

        scenario, stories, requirements = self._lineage_context(
            project=project,
            run_id=run_id,
            scenario_id=scenario_id,
        )
        binding = _select_binding(
            normalized_test_data,
            scenario_id=scenario_id,
            execution_profile_id=execution_profile_id,
            variant_id=variant_id or "default",
            binding_id=binding_id,
        )
        records = _resolved_records(normalized_test_data, binding)
        secret_refs = tuple(sorted(_collect_secret_refs([_dump(item) for item in records])))

        evidence_ids = _evidence_ids(scenario, stories, requirements)
        evidence_rows = self._evidence.fetch_chunks(project, set(evidence_ids))
        evidence_by_id = {chunk_id: text for chunk_id, text in evidence_rows}
        missing_evidence = set(evidence_ids) - set(evidence_by_id)
        if missing_evidence:
            raise ContextRetrievalError(
                f"supporting evidence chunks are missing: {sorted(missing_evidence)}"
            )

        matched = self._matched_symbols(
            snapshot_id=framework_snapshot.snapshot_id,
            scenario=scenario,
            symbol_ids=matched_symbol_ids,
            queries=symbol_queries,
            search_limit=symbol_search_limit,
            max_symbols=max_context_symbols,
        )
        neighbors = self._neighbor_symbols(
            snapshot_id=framework_snapshot.snapshot_id,
            matched=matched,
            max_symbols=max_context_symbols,
        )
        test_lib = self._test_lib_symbols(module_hints=module_hints)

        framework_ref = {
            "snapshot_id": framework_snapshot.snapshot_id,
            "status": framework_snapshot.status,
            "filesystem_checksum": framework_snapshot.filesystem_checksum,
            "extractor_version": framework_snapshot.extractor_version,
        }
        data_ref = {
            "snapshot_id": normalized_test_data.snapshot_id,
            "checksum": normalized_test_data.checksum,
            "workbook_checksum": normalized_test_data.workbook_checksum,
            "schema_version": normalized_test_data.schema_version,
        }
        mandatory = {
            "scenario": scenario,
            "stories": stories,
            "requirements": requirements,
            "evidence_chunk_ids": evidence_ids,
            "resolved_binding": _dump(binding),
            "test_data_snapshot": data_ref,
            "test_data_records": [_dump(item) for item in records],
            "secret_refs": secret_refs,
            "framework_snapshot": framework_ref,
        }
        mandatory_tokens = _estimate_tokens(mandatory)

        candidates: list[_BudgetCandidate] = []
        candidates.extend(_code_candidates(matched, tier=2, group="matched_symbols"))
        candidates.extend(_code_candidates(test_lib, tier=2, group="test_lib_symbols"))
        candidates.extend(
            _BudgetCandidate(
                tier=3,
                group="evidence_chunks",
                key=chunk_id,
                payload={"chunk_id": chunk_id, "text": evidence_by_id[chunk_id]},
                value={"chunk_id": chunk_id, "text": evidence_by_id[chunk_id]},
            )
            for chunk_id in evidence_ids
        )
        candidates.extend(_code_candidates(neighbors, tier=4, group="neighbor_symbols"))
        selected, selected_tokens, dropped = _apply_budget(
            candidates,
            remaining=max(0, token_budget - mandatory_tokens),
            max_context_symbols=max_context_symbols,
        )

        return ScenarioCodegenContext(
            project=project,
            run_id=run_id,
            scenario_id=scenario_id,
            execution_profile_id=execution_profile_id,
            variant_id=variant_id or "default",
            scenario=scenario,
            stories=tuple(stories),
            requirements=tuple(requirements),
            evidence_chunk_ids=tuple(evidence_ids),
            evidence_chunks=tuple(selected["evidence_chunks"]),
            resolved_binding=_dump(binding),
            test_data_snapshot=data_ref,
            test_data_records=tuple(_dump(item) for item in records),
            secret_refs=secret_refs,
            framework_snapshot=framework_ref,
            matched_symbols=tuple(selected["matched_symbols"]),
            test_lib_symbols=tuple(selected["test_lib_symbols"]),
            neighbor_symbols=tuple(selected["neighbor_symbols"]),
            token_budget=token_budget,
            mandatory_tokens=mandatory_tokens,
            selected_tokens=selected_tokens,
            mandatory_over_budget=mandatory_tokens > token_budget,
            dropped_items=dropped,
        )

    def _lineage_context(
        self,
        *,
        project: str,
        run_id: str,
        scenario_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        scenario_artifact = self._lineage.load_test_scenarios(project, run_id)
        story_artifact = self._lineage.load_user_stories(project, run_id)
        requirement_artifact = self._lineage.load_requirements(project, run_id)
        if scenario_artifact is None or story_artifact is None or requirement_artifact is None:
            raise ContextRetrievalError("complete Stage 1-3 lineage is unavailable")

        scenarios = [_dump(item) for item in _artifact_items(scenario_artifact, "scenarios")]
        scenario = next(
            (item for item in scenarios if item.get("scenario_id") == scenario_id),
            None,
        )
        if scenario is None:
            raise ContextRetrievalError(f"frozen scenario is unavailable: {scenario_id}")

        story_index = {
            str(item.get("story_id")): item
            for item in (_dump(row) for row in _artifact_items(story_artifact, "stories"))
        }
        story_ids = [str(value) for value in scenario.get("story_ids", [])]
        missing_stories = set(story_ids) - set(story_index)
        if missing_stories:
            raise ContextRetrievalError(f"parent stories are missing: {sorted(missing_stories)}")
        stories = [story_index[story_id] for story_id in story_ids]

        requirement_ids = [str(value) for value in scenario.get("requirement_ids", [])]
        for story in stories:
            for requirement_id in story.get("requirement_ids", []):
                value = str(requirement_id)
                if value not in requirement_ids:
                    requirement_ids.append(value)
        requirement_index = {
            str(item.get("requirement_id")): item
            for item in (
                _dump(row) for row in _artifact_items(requirement_artifact, "requirements")
            )
        }
        missing_requirements = set(requirement_ids) - set(requirement_index)
        if missing_requirements:
            raise ContextRetrievalError(
                f"parent requirements are missing: {sorted(missing_requirements)}"
            )
        requirements = [requirement_index[requirement_id] for requirement_id in requirement_ids]
        return scenario, stories, requirements

    def _matched_symbols(
        self,
        *,
        snapshot_id: str,
        scenario: dict[str, Any],
        symbol_ids: tuple[str, ...],
        queries: tuple[str, ...],
        search_limit: int,
        max_symbols: int,
    ) -> list[RetrievedCodeContext]:
        symbols: dict[str, CodeSymbol] = {}
        for symbol_id in symbol_ids:
            symbol = self._code_graph.get_symbol(snapshot_id, symbol_id)
            if symbol is not None:
                symbols.setdefault(symbol.symbol_id, symbol)
        resolved_queries = queries or _scenario_queries(scenario)
        for query in resolved_queries:
            for symbol in self._code_graph.search_symbols(
                snapshot_id,
                query,
                limit=max(1, search_limit),
            ):
                symbols.setdefault(symbol.symbol_id, symbol)
                if len(symbols) >= max_symbols:
                    break
            if len(symbols) >= max_symbols:
                break
        return [self._source_context(symbol) for symbol in list(symbols.values())[:max_symbols]]

    def _neighbor_symbols(
        self,
        *,
        snapshot_id: str,
        matched: list[RetrievedCodeContext],
        max_symbols: int,
    ) -> list[RetrievedCodeContext]:
        matched_ids = {item.symbol_id for item in matched}
        neighbors: dict[str, CodeSymbol] = {}
        relations: dict[str, set[str]] = {}
        for item in matched:
            for relation in _NEIGHBOR_RELATIONS:
                for symbol in self._code_graph.get_neighbors(
                    snapshot_id,
                    item.symbol_id,
                    relations=[relation],
                    depth=1,
                ):
                    if symbol.symbol_id in matched_ids:
                        continue
                    neighbors.setdefault(symbol.symbol_id, symbol)
                    relations.setdefault(symbol.symbol_id, set()).add(relation)
                    if len(neighbors) >= max_symbols:
                        break
                if len(neighbors) >= max_symbols:
                    break
            if len(neighbors) >= max_symbols:
                break
        return [
            self._source_context(
                symbol,
                relations=tuple(sorted(relations.get(symbol.symbol_id, set()))),
            )
            for symbol in neighbors.values()
        ]

    def _source_context(
        self,
        symbol: CodeSymbol,
        *,
        relations: tuple[str, ...] = (),
    ) -> RetrievedCodeContext:
        relative = _safe_relative(symbol.relative_path)
        target = (self._framework_root / relative).resolve()
        if target != self._framework_root and self._framework_root not in target.parents:
            raise ContextRetrievalError(f"framework symbol escapes root: {relative}")
        if not target.is_file():
            raise ContextRetrievalError(f"framework symbol source is missing: {relative}")
        lines = target.read_text(encoding="utf-8").splitlines()
        body = "\n".join(lines[symbol.start_line - 1 : symbol.end_line])
        body_hash = _sha256(body)
        if symbol.body_hash and symbol.body_hash != body_hash:
            raise ContextRetrievalError(
                f"framework snapshot is stale for symbol {symbol.symbol_id}: body hash mismatch"
            )
        return RetrievedCodeContext(
            symbol_id=symbol.symbol_id,
            fqn=symbol.fqn,
            kind=symbol.kind,
            signature=symbol.signature,
            relative_path=relative,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            body_hash=body_hash,
            source_body=body,
            relations=relations,
        )

    def _test_lib_symbols(self, *, module_hints: tuple[str, ...]) -> list[RetrievedCodeContext]:
        root = self._framework_root / "test_lib"
        if not root.is_dir():
            return []
        allowed_modules = {value.casefold() for value in module_hints if value}
        results: list[RetrievedCodeContext] = []
        for module_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            module = module_dir.name
            if allowed_modules and module.casefold() not in allowed_modules:
                continue
            for suffix in ("wrappers", "helpers"):
                path = module_dir / f"{module}_{suffix}.py"
                if not path.is_file():
                    continue
                results.extend(_python_functions(path, self._framework_root))
        return results


def _select_binding(
    normalized: NormalizedTestData,
    *,
    scenario_id: str,
    execution_profile_id: str,
    variant_id: str,
    binding_id: str | None,
) -> ScenarioDataBinding:
    if binding_id is not None:
        matches = [item for item in normalized.bindings if item.binding_id == binding_id]
        if len(matches) != 1:
            raise ContextRetrievalError(f"selected binding is unavailable: {binding_id}")
        binding = matches[0]
        if (
            binding.scenario_id != scenario_id
            or binding.execution_profile_id != execution_profile_id
            or binding.variant_id != variant_id
            or binding.approval_status != "APPROVED"
        ):
            raise ContextRetrievalError("selected binding does not match scenario/profile")
        return binding
    resolution = resolve_binding(
        scenario_id=scenario_id,
        execution_profile_id=execution_profile_id,
        variant_id=variant_id,
        bindings=normalized.bindings,
    )
    if not resolution.is_ready or resolution.binding is None:
        raise ContextRetrievalError(f"scenario test-data binding is not READY: {resolution.status}")
    return resolution.binding


def _resolved_records(
    normalized: NormalizedTestData,
    binding: ScenarioDataBinding,
) -> list[TestDataRecord]:
    index = normalized.record_index()
    required_ids = [
        binding.execution_profile_id,
        binding.fixture_id,
        binding.cleanup_id,
        *binding.test_vector_ids,
        *binding.oracle_ids,
        *binding.fault_profile_ids,
        *binding.safety_rule_ids,
    ]
    if binding.timing_policy_id:
        required_ids.append(binding.timing_policy_id)
    selected: dict[str, TestDataRecord] = {}
    for record_id in required_ids:
        record = index.get(record_id)
        if record is None:
            raise ContextRetrievalError(f"binding references unknown test-data record: {record_id}")
        if record.record_status != "APPROVED":
            raise ContextRetrievalError(f"binding references unapproved record: {record_id}")
        selected[record_id] = record
    if binding.action_sequence_id:
        for record in normalized.records:
            if (
                record.record_type == "ActionStep"
                and record.payload.get("sequence_id") == binding.action_sequence_id
            ):
                if record.record_status != "APPROVED":
                    raise ContextRetrievalError(
                        f"action sequence references unapproved record: {record.record_id}"
                    )
                selected[record.record_id] = record

    changed = True
    while changed:
        changed = False
        referenced = _record_id_references([record.payload for record in selected.values()], index)
        for record_id in sorted(referenced):
            if record_id in selected:
                continue
            record = index[record_id]
            if record.record_status != "APPROVED":
                raise ContextRetrievalError(
                    f"test-data payload references unapproved record: {record_id}"
                )
            selected[record_id] = record
            changed = True
    return list(selected.values())


def _record_id_references(values: list[Any], index: dict[str, TestDataRecord]) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value in index:
                found.add(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list | tuple | set):
            for child in value:
                visit(child)

    for value in values:
        visit(value)
    return found


def _collect_secret_refs(values: list[Any]) -> set[str]:
    refs: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value.startswith(_SECRET_PREFIX):
                refs.add(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list | tuple | set):
            for child in value:
                visit(child)

    for value in values:
        visit(value)
    return refs


def _evidence_ids(
    scenario: dict[str, Any],
    stories: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        text = str(value)
        if text and text not in values:
            values.append(text)

    for item in [scenario, *stories]:
        traceability = item.get("traceability", {})
        if isinstance(traceability, dict):
            for chunk_id in traceability.get("evidence_chunk_ids", []):
                add(chunk_id)
    for requirement in requirements:
        for evidence in requirement.get("evidence", []):
            if isinstance(evidence, dict) and evidence.get("chunk_id"):
                add(evidence["chunk_id"])
    return values


def _artifact_items(artifact: Any, name: str) -> list[Any]:
    value = artifact.get(name, []) if isinstance(artifact, dict) else getattr(artifact, name, [])
    return list(value) if isinstance(value, list | tuple) else []


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError(f"context value is not serializable: {type(value).__name__}")


def _safe_relative(relative_path: str) -> str:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ContextRetrievalError(f"unsafe framework source path: {relative_path}")
    return path.as_posix()


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _scenario_queries(scenario: dict[str, Any]) -> tuple[str, ...]:
    text = " ".join(str(scenario.get(key, "")) for key in ("title", "action", "expected_result"))
    seen: set[str] = set()
    values: list[str] = []
    for match in _SEARCH_WORD.finditer(text):
        value = match.group(0).casefold()
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
        if len(values) >= 12:
            break
    return tuple(values)


def _python_functions(path: Path, root: Path) -> list[RetrievedCodeContext]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()
    relative = path.resolve().relative_to(root).as_posix()
    results: list[RetrievedCodeContext] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        start_line = min(
            [node.lineno, *(item.lineno for item in node.decorator_list)],
            default=node.lineno,
        )
        end_line = node.end_lineno or node.lineno
        body = "\n".join(lines[start_line - 1 : end_line])
        results.append(
            RetrievedCodeContext(
                symbol_id=f"testlib:{relative}:{node.name}:{start_line}",
                fqn=f"{relative}::{node.name}",
                kind="function",
                signature=lines[node.lineno - 1].strip(),
                relative_path=relative,
                start_line=start_line,
                end_line=end_line,
                body_hash=_sha256(body),
                source_body=body,
                relations=("TEST_LIB",),
            )
        )
    return results


def _code_candidates(
    values: list[RetrievedCodeContext],
    *,
    tier: int,
    group: str,
) -> list[_BudgetCandidate]:
    return [
        _BudgetCandidate(
            tier=tier,
            group=group,
            key=value.symbol_id,
            payload=value.to_prompt_payload(),
            value=value,
        )
        for value in values
    ]


def _apply_budget(
    candidates: list[_BudgetCandidate],
    *,
    remaining: int,
    max_context_symbols: int,
) -> tuple[dict[str, list[Any]], int, dict[str, int]]:
    selected: dict[str, list[Any]] = {
        "matched_symbols": [],
        "test_lib_symbols": [],
        "evidence_chunks": [],
        "neighbor_symbols": [],
    }
    dropped = {key: 0 for key in selected}
    used = 0
    selected_symbols = 0
    for candidate in sorted(candidates, key=lambda item: (item.tier, item.group, item.key)):
        cost = _estimate_tokens(candidate.payload)
        is_symbol = candidate.group != "evidence_chunks"
        if used + cost > remaining or (is_symbol and selected_symbols >= max_context_symbols):
            dropped[candidate.group] += 1
            continue
        selected[candidate.group].append(candidate.value)
        used += cost
        if is_symbol:
            selected_symbols += 1
    return selected, used, dropped


def _estimate_tokens(value: Any) -> int:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return max(1, (len(text) + 3) // 4)


__all__ = [
    "CodeGraphReader",
    "CodegenContextRetriever",
    "ContextRetrievalError",
    "EvidenceStore",
    "LineageStore",
    "RetrievedCodeContext",
    "ScenarioCodegenContext",
]
