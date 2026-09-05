"""
tools/registry.py — Centralized Tool Registry
================================================

WHY THIS EXISTS:
    The registry is the SINGLE PLACE where tools are registered and
    discovered. This decouples the agent from specific tool implementations:

    - Agent says: "I want to call 'get_stock_price'"
    - Registry says: "Here's the tool object for that name"
    - Agent calls tool.execute() — doesn't know or care about internals

    Without a registry, you'd have if/elif chains in the agent:
        if action == "get_stock_price": ...
        elif action == "get_company_info": ...
    This violates Open/Closed Principle — adding a tool means modifying agent code.

    With a registry, adding a tool is:
        registry.register(MyNewTool())
    Zero changes to agent code.

HOW IT CONNECTS:
    - main.py creates the registry and registers tools at startup.
    - prompts/system.py gets tool descriptions from registry.get_tool_descriptions().
    - agent/nodes.py calls registry.get(tool_name).execute(input).

SCALABILITY:
    Phase 2+ can add:
    - auto-discovery (scan tools/ directory for BaseTool subclasses),
    - per-tool rate limiting,
    - tool usage analytics,
    - tool capability matching (which tools can answer which query types).
"""

from __future__ import annotations

from typing import Optional

from tools.base import BaseTool
from config.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    Central registry for all available tools.

    Thread-safe for future async usage. Provides lookup by name,
    listing, and formatted descriptions for prompt injection.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: A BaseTool subclass instance.

        Raises:
            ValueError: If a tool with the same name is already registered
                        (prevents silent overwrites, which cause subtle bugs).
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Duplicate tool names cause ambiguous routing."
            )
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """
        Look up a tool by name.

        Returns None instead of raising so the caller can provide
        a helpful error message to the LLM rather than crashing.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """
        Generate formatted tool descriptions for injection into
        the system prompt.

        Returns:
            Multi-line string describing all available tools and
            their parameter schemas.
        """
        if not self._tools:
            return "  No tools available."

        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.get_schema_description())
        return "\n\n".join(descriptions)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
