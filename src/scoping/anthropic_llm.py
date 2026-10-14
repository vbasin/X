"""Backward-compatible shim — canonical module is :mod:`src.common.llm.anthropic_client`.

Anthropic is not scoping-specific; it is a shared provider. New code::

    from src.common.llm import AnthropicLLM, make_llm
"""

from __future__ import annotations

from src.common.llm.anthropic_client import AnthropicLLM, LLMResult

__all__ = ["AnthropicLLM", "LLMResult"]
