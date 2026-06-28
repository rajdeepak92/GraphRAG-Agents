"""Chroma client factory."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI


def create_persistent_chroma_client(path: Path) -> ClientAPI:
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))
