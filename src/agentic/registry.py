"""In-process typed tool registry for agent harnesses (no MCP).

Shared by Scoping and Research Director. The LLM tool-calling loop
(``src.common.llm`` clients) expects ``.specs()`` + ``.call(name, **kwargs)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """A single registered tool: name + ACI docs + JSON-Schema params + callable."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (type: object)
    fn: Callable[..., Any]

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds tools available to one agent and dispatches calls deterministically."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool '{name}' already registered")
        self._tools[name] = Tool(
            name=name, description=description, parameters=parameters, fn=fn)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-compatible tool specs for the tool-calling loop."""
        return [t.openai_spec() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool '{name}'")
        return self._tools[name].fn(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def obj_schema(
    props: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """JSON Schema object helper for tool parameter specs."""
    return {"type": "object", "properties": props or {}, "required": required or []}


# Backward-compatible alias used throughout the RD codebase.
_obj = obj_schema
