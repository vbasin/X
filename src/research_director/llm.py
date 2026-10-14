"""Backward-compatible shim — canonical module is :mod:`src.common.llm.openai_client`.

Kept so historical imports (`src.research_director.llm.LLMClient`) resolve during migration.
New code should use ``from src.common.llm import LLMClient, make_llm``.
"""

from __future__ import annotations

from src.common.llm.openai_client import LLMClient, LLMResult

__all__ = ["LLMClient", "LLMResult"]
