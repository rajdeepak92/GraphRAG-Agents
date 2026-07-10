"""Feature flags for the source-knowledge graph migration.

The active application remains chunk-first until these flags are explicitly enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KnowledgeGraphFlags:
    enabled: bool = False
    shadow_mode: bool = False
    graph_primary_user_stories: bool = False
    graph_primary_test_scenarios: bool = False

    @classmethod
    def from_env(cls) -> "KnowledgeGraphFlags":
        return cls(
            enabled=_env_bool("KNOWLEDGE_GRAPH_ENABLED"),
            shadow_mode=_env_bool("KNOWLEDGE_GRAPH_SHADOW_MODE"),
            graph_primary_user_stories=_env_bool("GRAPH_PRIMARY_USER_STORIES"),
            graph_primary_test_scenarios=_env_bool("GRAPH_PRIMARY_TEST_SCENARIOS"),
        )
