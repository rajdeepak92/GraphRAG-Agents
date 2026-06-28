"""Multi-Agentic Knowledge-Graph RAG platform."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

__all__ = [
    "DISTRIBUTION_NAME",
    "__version__",
]

DISTRIBUTION_NAME = "multi-agentic-graph-rag"

try:
    __version__ = distribution_version(DISTRIBUTION_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+uninstalled"
