"""Cost metering for the cognitive layer (shared by all LLM clients).

Pure and dependency-free so it is fully unit-testable. Tracks per-cycle and per-agent LLM
spend, computes **warn → degrade → halt** bands, and raises :class:`CostCapExceeded` at the
hard cap. Both :class:`~src.common.llm.openai_client.LLMClient` and
:class:`~src.common.llm.anthropic_client.AnthropicLLM` delegate to a :class:`CostMeter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

OK = "OK"
WARN = "WARN"
DEGRADE = "DEGRADE"
HALT = "HALT"


class CostCapExceeded(RuntimeError):
    """Raised when per-cycle LLM spend hits the hard cap; the panel aborts gracefully."""


@dataclass
class CostMeter:
    cycle_cap: float = 5.0
    per_agent_cap: float = 2.5
    warn_frac: float = 0.75
    degrade_frac: float = 0.90
    model_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    _cycle: float = 0.0
    _by_agent: dict[str, float] = field(default_factory=dict)

    def cost_of(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost from per-model rates (defaults to 0.0 when a model has no configured rate)."""
        rate = self.model_rates.get(model, {})
        rin = float(rate.get("in_per_1k", 0.0))
        rout = float(rate.get("out_per_1k", 0.0))
        return (prompt_tokens / 1000.0) * rin + (completion_tokens / 1000.0) * rout

    def charge(self, *, model: str, agent: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Debit one call; returns the cost. Raises CostCapExceeded once the cycle cap is hit."""
        cost = self.cost_of(model, prompt_tokens, completion_tokens)
        self._cycle += cost
        self._by_agent[agent] = self._by_agent.get(agent, 0.0) + cost
        if self.cycle_cap > 0 and self._cycle > self.cycle_cap:
            raise CostCapExceeded(
                f"cycle LLM spend ${self._cycle:.4f} exceeded cap ${self.cycle_cap:.2f}"
            )
        return cost

    def band(self) -> str:
        if self.cycle_cap <= 0:
            return OK
        frac = self._cycle / self.cycle_cap
        if frac >= 1.0:
            return HALT
        if frac >= self.degrade_frac:
            return DEGRADE
        if frac >= self.warn_frac:
            return WARN
        return OK

    def should_degrade(self) -> bool:
        return self.band() in (DEGRADE, HALT)

    def agent_over_budget(self, agent: str) -> bool:
        return self.per_agent_cap > 0 and self._by_agent.get(agent, 0.0) >= self.per_agent_cap

    def tally(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {"cycle_usd": round(self._cycle, 6), "band": self.band()}
        for agent, spend in self._by_agent.items():
            out[f"agent_{agent}_usd"] = round(spend, 6)
        return out

    def reset(self) -> None:
        self._cycle = 0.0
        self._by_agent = {}
