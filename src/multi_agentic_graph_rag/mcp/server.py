"""Project-local stdio MCP server for MARAG automation."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from multi_agentic_graph_rag.mcp.prompts import register_prompts
from multi_agentic_graph_rag.mcp.resources import register_resources
from multi_agentic_graph_rag.mcp.tools import register_tools

mcp = FastMCP("marag-mcp")


def project_root() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd())).resolve()


register_tools(mcp, project_root=project_root)
register_resources(mcp, project_root=project_root)
register_prompts(mcp)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
