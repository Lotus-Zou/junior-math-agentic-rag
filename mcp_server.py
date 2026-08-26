# -*- coding: utf-8 -*-
"""Registry-generated MCP server for standardized mathematics Skills."""

try:
    from mcp.server.fastmcp import FastMCP as Server
except ImportError:
    from mcp.server.mcpserver import MCPServer as Server

from agentic_rag.skill_runtime.adapters.mcp import MCPRegistryAdapter
from agentic_rag.skill_runtime.executor import SkillExecutor
from agentic_rag.skill_runtime.registry import get_default_registry

registry = get_default_registry()
executor = SkillExecutor(registry)
mcp = MCPRegistryAdapter(registry, executor).register(Server("junior-math-agentic-rag"))


if __name__ == "__main__":
    mcp.run(transport="stdio")
