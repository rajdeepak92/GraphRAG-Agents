"""The Stage-4 action mapper supplies candidates but never implementation decisions."""

from __future__ import annotations

from multi_agentic_graph_rag.domain.code_graph_schemas import CodeSymbol
from multi_agentic_graph_rag.services.scenario_action_mapper import ScenarioActionMapper


class _Graph:
    def __init__(self, symbols: list[CodeSymbol]) -> None:
        self.symbols = symbols
        self.queries: list[str] = []

    def search_symbols(
        self,
        snapshot_id: str,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CodeSymbol]:
        del snapshot_id, kinds
        self.queries.append(query)
        return [symbol for symbol in self.symbols if query in symbol.fqn.casefold()][:limit]


def _symbol(symbol_id: str, fqn: str) -> CodeSymbol:
    return CodeSymbol(
        snapshot_id="FWS-1",
        symbol_id=symbol_id,
        fqn=fqn,
        kind="Function",
        signature=fqn.rsplit("::", 1)[-1],
        relative_path="sensor.py",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=0,
        body_hash="sha256:body",
    )


def test_mapper_returns_ranked_advice_without_reuse_or_helper_decision() -> None:
    graph = _Graph(
        [
            _symbol("SYM-READ", "sensor.py::read_temperature_sensor"),
            _symbol("SYM-THRESHOLD", "sensor.py::validate_sensor_threshold"),
        ]
    )
    result = ScenarioActionMapper(graph).map_scenario(
        snapshot_id="FWS-1",
        scenario={
            "title": "Validate sensor threshold",
            "action": "Read temperature sensor",
            "expected_result": "Sensor threshold is enforced",
        },
    )

    assert result.action_text == "Read temperature sensor"
    assert {candidate.symbol_id for candidate in result.candidates} == {
        "SYM-READ",
        "SYM-THRESHOLD",
    }
    assert all(not hasattr(candidate, "decision") for candidate in result.candidates)
    assert "sensor" in graph.queries


def test_zero_limit_returns_no_candidates_or_decision() -> None:
    result = ScenarioActionMapper(_Graph([])).map_action(
        snapshot_id="FWS-1",
        action_text="Do something",
        limit=0,
    )
    assert result.candidates == ()
    assert result.query_terms == ()
