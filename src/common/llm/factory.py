"""Provider-agnostic LLM factory for every agentic process in the codebase.

One construction point so any agent (Scoping, Research Director, feature refine, …)
can be fitted to any LLM provider without importing a concrete client. Selection is
env-driven; both clients expose the same duck-typed interface.

Provider contract (duck type)::

    chat / chat_json / chat_tools
    should_degrade / agent_over_budget / cost_tally / reset_cycle

Selection (env)::

    VTECH_LLM_PROVIDER=anthropic | openai      explicit choice
    unset → auto: anthropic if ANTHROPIC_API_KEY/Claude_API_KEY present,
                  else openai-compatible if RD_LLM_MODEL present, else raise.

Clients live in :mod:`src.common.llm` — not under scoping or research_director.
"""

from __future__ import annotations

import os
from typing import Any


def make_llm(
    *,
    usd_cap_cycle: float = 5.0,
    per_agent_usd_cap: float = 2.5,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """Build the configured LLM client (see module docstring for the contract).

    Raises on missing configuration/dependency — callers that want graceful degradation
    wrap this in try/except and fall back (the established agent pattern).
    """
    prov = (provider or os.environ.get("VTECH_LLM_PROVIDER", "")).strip().lower()
    if not prov:
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Claude_API_KEY"):
            prov = "anthropic"
        elif os.environ.get("RD_LLM_MODEL"):
            prov = "openai"
        else:
            raise ValueError(
                "no LLM provider configured: set VTECH_LLM_PROVIDER, or provide "
                "ANTHROPIC_API_KEY / Claude_API_KEY (anthropic) or RD_LLM_MODEL (openai)")

    if prov == "anthropic":
        from src.common.llm.anthropic_client import AnthropicLLM
        return AnthropicLLM(
            usd_cap_cycle=usd_cap_cycle, per_agent_usd_cap=per_agent_usd_cap, **kwargs)
    if prov == "openai":
        from src.common.llm.openai_client import LLMClient
        return LLMClient(
            usd_cap_cycle=usd_cap_cycle, per_agent_usd_cap=per_agent_usd_cap, **kwargs)
    raise ValueError(f"unknown VTECH_LLM_PROVIDER '{prov}' (expected 'anthropic' or 'openai')")


def strip_code_fence(text: str) -> str:
    """Remove a leading/trailing ```json fence if the model wrapped its reply.

    Shared by every client's ``chat_json`` repair loop (single implementation).
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()
