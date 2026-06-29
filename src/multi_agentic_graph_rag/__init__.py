"""Ingestion-first Multi-Agentic Graph RAG package."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

DISTRIBUTION_NAME = "multi-agentic-graph-rag"

try:
    __version__ = distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+uninstalled"

__all__ = ["DISTRIBUTION_NAME", "__version__"]
