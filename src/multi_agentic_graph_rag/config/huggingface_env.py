"""Hugging Face environment helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

from multi_agentic_graph_rag.common_defs import EnvVar

HF_TOKEN_ALIASES = (
    EnvVar.HF_TOKEN.value,
    EnvVar.HUGGINGFACE_TOKEN.value,
    EnvVar.HUGGING_FACE_HUB_TOKEN.value,
)
HF_OFFLINE_FLAGS = (EnvVar.HF_HUB_OFFLINE.value, EnvVar.TRANSFORMERS_OFFLINE.value)


def first_huggingface_token(env: Mapping[str, str]) -> str:
    """Return the first configured Hugging Face token alias."""
    for key in HF_TOKEN_ALIASES:
        value = env.get(key, "").strip()
        if value:
            return value
    return ""


def export_huggingface_environment(*, token: str, offline: bool) -> None:
    """Expose configured Hugging Face values through standard library env vars."""
    if token:
        for key in HF_TOKEN_ALIASES:
            if not os.environ.get(key):
                os.environ[key] = token
    if offline:
        for key in HF_OFFLINE_FLAGS:
            os.environ[key] = "1"


def env_bool(value: str | None, *, default: bool = False) -> bool:
    """Execute the env bool operation within its declared architectural boundary.

    Args:
        value (str | None): Value required by the operation's typed contract.
        default (bool): Default required by the operation's typed contract.

    Returns:
        bool: The typed result produced by the operation.
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_positive_int(value: str | None, *, default: int) -> int:
    """Execute the env positive int operation within its declared architectural boundary.

    Args:
        value (str | None): Value required by the operation's typed contract.
        default (int): Default required by the operation's typed contract.

    Returns:
        int: The typed result produced by the operation.
    """
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, parsed)


def token_alias_status(env: Mapping[str, str]) -> str:
    """Execute the token alias status operation within its declared architectural boundary.

    Args:
        env (Mapping[str, str]): Env required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    visible = [key for key in HF_TOKEN_ALIASES if bool(env.get(key))]
    if not visible:
        return "not set"
    return "set via " + ", ".join(visible)
