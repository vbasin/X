"""Shared agentic substrate for Scoping, Research Director, and future agents.

**LLM** lives in :mod:`src.common.llm` (factory + clients + cost). This package
adds everything else agents share:

* :mod:`src.agentic.registry` — in-process tool registry (no MCP)
* :mod:`src.agentic.rag` — pgvector research search (primary) + helpers
* :mod:`src.agentic.critic` — re-exports the parameterized critic harness
* :mod:`src.agentic.embeddings` — shared embedder backends
* :mod:`src.agentic.memory` — RD per-agent memory (pgvector)

Plane-specific code stays in ``src.scoping`` / ``src.research_director``; they
should import shared pieces from here (or via thin shims that re-export).
"""

from __future__ import annotations

from src.agentic.registry import Tool, ToolRegistry, obj_schema

__all__ = [
    "Tool",
    "ToolRegistry",
    "obj_schema",
]
