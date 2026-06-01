"""
tools/base.py — Abstract Base Tool Interface
==============================================

WHY THIS EXISTS:
    Every tool the agent can use MUST conform to a consistent interface.
    Without this:
    - each tool would have a different call signature,
    - the agent execution loop would need tool-specific if/else branches,
    - adding a new tool would require modifying the agent core.

    The base interface enforces:
    - standard metadata (name, description, parameters),
    - standard execution contract (execute → string result),
    - standard error handling.

    This is the Strategy Pattern: the agent doesn't know WHICH tool
    it's running — it just calls .execute() on whatever the registry
    returns.

HOW IT CONNECTS:
    - All concrete tools (stock_price.py, etc.) inherit from BaseTool.
    - tools/registry.py stores instances of BaseTool subclasses.
    - agent/nodes.py calls tool.execute() polymorphically.

SCALABILITY:
    Phase 2+ tools (SEC filing retriever, earnings transcript analyzer,
    technical indicator calculator) all inherit from this same base.
    Zero changes to the agent core required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """
    Schema for a single tool parameter.

    This is exposed to the LLM in the system prompt so it knows
    what arguments each tool expects.
    """
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, number, etc.)")
    description: str = Field(description="What this parameter is for")
    required: bool = Field(default=True, description="Whether this parameter is required")


class ToolResult(BaseModel):
    """
    Standardized tool output.

    Every tool returns this — not raw strings, not dicts, not exceptions.
    This ensures the agent always gets a consistent structure.
    """
    success: bool = Field(description="Whether the tool executed successfully")
    data: str = Field(default="", description="The tool's output as a string")
    error: str = Field(default="", description="Error message if success=False")


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Subclasses MUST implement:
        - name (property): unique identifier used in tool calls
        - description (property): human-readable description for the LLM
        - parameters (property): list of ToolParameter schemas
        - _execute(input): the actual tool logic

    Subclasses SHOULD NOT override execute() — it handles error
    wrapping and logging. Override _execute() instead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g., 'get_stock_price')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.
        This is injected into the system prompt for the LLM.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """Schema of expected input parameters."""
        ...

    @abstractmethod
    def _execute(self, tool_input: dict[str, Any]) -> str:
        """
        Core tool logic. Subclasses implement this.

        Args:
            tool_input: Dictionary of parameter name → value.

        Returns:
            String result to be passed back to the agent.

        Raises:
            Any exception — will be caught by execute().
        """
        ...

    def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        """
        Public execution entry point with error handling.

        This wraps _execute() to guarantee:
        1. Exceptions never crash the agent.
        2. Every result is a ToolResult.
        3. Errors are captured with context.

        DO NOT override this method in subclasses.
        """
        try:
            result = self._execute(tool_input)
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool '{self.name}' failed: {type(e).__name__}: {str(e)}",
            )

    def get_schema_description(self) -> str:
        """
        Generate a formatted description for injection into the system prompt.

        Returns something like:
            Tool: get_stock_price
            Description: Retrieves current stock price...
            Parameters:
              - ticker (string, required): Stock ticker symbol
        """
        params_str = "\n".join(
            f"      - {p.name} ({p.type}, {'required' if p.required else 'optional'}): {p.description}"
            for p in self.parameters
        )
        return (
            f"  Tool: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Parameters:\n{params_str}"
        )
