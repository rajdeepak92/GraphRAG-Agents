"""Advisory framework-symbol candidates for a frozen scenario.

This service never decides reuse, wrapping, helper creation, module selection, or
step decomposition.  It only supplies deterministic search candidates; the
selected Stage-4 reasoning model makes the implementation decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from multi_agentic_graph_rag.domain.code_graph_schemas import CodeSymbol

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "should",
    "that",
    "the",
    "then",
    "this",
    "when",
    "with",
}


class SymbolSearch(Protocol):
    """Minimal code-graph search surface used by the advisory mapper."""

    def search_symbols(
        self,
        snapshot_id: str,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CodeSymbol]: ...


@dataclass(frozen=True)
class AdvisorySymbolCandidate:
    """One search hit; deliberately carries no implementation decision."""

    symbol_id: str
    fqn: str
    kind: str
    signature: str | None
    relative_path: str
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class AdvisoryActionMapping:
    """Search candidates for scenario prose, not a generated execution plan."""

    action_text: str
    query_terms: tuple[str, ...]
    candidates: tuple[AdvisorySymbolCandidate, ...]


class ScenarioActionMapper:
    """Find advisory code candidates without making autonomous policy choices."""

    def __init__(self, code_graph: SymbolSearch) -> None:
        self._code_graph = code_graph

    def map_scenario(
        self,
        *,
        snapshot_id: str,
        scenario: Any,
        limit: int = 20,
    ) -> AdvisoryActionMapping:
        """Search using the scenario title/action/result while preserving its action text."""

        data = _dump(scenario)
        action_text = str(data.get("action", "")).strip()
        search_text = " ".join(
            value
            for value in (
                str(data.get("title", "")).strip(),
                action_text,
                str(data.get("expected_result", "")).strip(),
            )
            if value
        )
        return self.map_action(
            snapshot_id=snapshot_id,
            action_text=action_text or search_text,
            search_text=search_text,
            limit=limit,
        )

    def map_action(
        self,
        *,
        snapshot_id: str,
        action_text: str,
        search_text: str | None = None,
        limit: int = 20,
    ) -> AdvisoryActionMapping:
        """Return ranked substring-search hits without assigning a decision."""

        if limit <= 0:
            return AdvisoryActionMapping(action_text=action_text, query_terms=(), candidates=())
        terms = _query_terms(search_text or action_text)
        hits: dict[str, CodeSymbol] = {}
        matched: dict[str, set[str]] = {}
        per_term_limit = max(1, min(limit, 20))
        for term in terms:
            for symbol in self._code_graph.search_symbols(
                snapshot_id,
                term,
                limit=per_term_limit,
            ):
                hits.setdefault(symbol.symbol_id, symbol)
                matched.setdefault(symbol.symbol_id, set()).add(term)
        ordered = sorted(
            hits.values(),
            key=lambda symbol: (
                -len(matched.get(symbol.symbol_id, set())),
                symbol.fqn.casefold(),
                symbol.symbol_id,
            ),
        )[:limit]
        candidates = tuple(
            AdvisorySymbolCandidate(
                symbol_id=symbol.symbol_id,
                fqn=symbol.fqn,
                kind=symbol.kind,
                signature=symbol.signature,
                relative_path=symbol.relative_path,
                matched_terms=tuple(sorted(matched.get(symbol.symbol_id, set()))),
            )
            for symbol in ordered
        )
        return AdvisoryActionMapping(
            action_text=action_text,
            query_terms=terms,
            candidates=candidates,
        )


def _query_terms(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in _WORD.finditer(text):
        term = match.group(0).casefold()
        if term in _STOP_WORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 12:
            break
    return tuple(terms)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json"))
    return {
        name: getattr(value, name)
        for name in ("title", "action", "expected_result")
        if hasattr(value, name)
    }


__all__ = [
    "AdvisoryActionMapping",
    "AdvisorySymbolCandidate",
    "ScenarioActionMapper",
    "SymbolSearch",
]
