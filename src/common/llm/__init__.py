"""Shared LLM substrate for every agentic process (Scoping, Research Director, search tools).

Single home for provider clients, cost metering, and construction. Planes must not own
their own LLM stacks — import from here (or via :func:`make_llm`).

Public API
----------
* :func:`make_llm` / :func:`strip_code_fence` — provider selection + JSON fence helper
* :class:`LLMClient` — OpenAI-compatible chat/tools/embed client
* :class:`AnthropicLLM` — Anthropic Messages API + server web tools
* :class:`CostMeter` / :class:`CostCapExceeded` — warn → degrade → halt spend bands

Duck-type contract (every client)::

    chat / chat_json / chat_tools
    should_degrade / agent_over_budget / cost_tally / reset_cycle
"""

from __future__ import annotations

from src.common.llm.anthropic_client import AnthropicLLM
from src.common.llm.cost import CostCapExceeded, CostMeter
from src.common.llm.factory import make_llm, strip_code_fence
from src.common.llm.openai_client import LLMClient, LLMResult

__all__ = [
    "AnthropicLLM",
    "CostCapExceeded",
    "CostMeter",
    "LLMClient",
    "LLMResult",
    "make_llm",
    "strip_code_fence",
]
