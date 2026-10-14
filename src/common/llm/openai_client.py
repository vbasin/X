"""OpenAI-compatible LLM client (shared cognitive substrate under src.common.llm).

Targets any OpenAI-compatible endpoint (OpenAI platform, GitHub Models, OCI GenAI, local
vLLM) via ``base_url``. Responsibilities: chat completion with strict-JSON parsing (one
repair retry), embeddings for vector memory, token accounting, and writing every call to
the `llm_call` audit table + `gpu_budget_ledger` (category='llm').

Endpoint resolution: explicit ``base_url``/``api_key`` kwargs win (used by Scoping v2 to
pin the critic to the OpenAI platform); otherwise the shared RD_LLM_BASE_URL /
RD_LLM_API_KEY / RD_LLM_MODEL env vars apply. Non-secret knobs come from
config/research_director.yaml.

Model compatibility: reasoning models (gpt-5*/o-series) reject ``temperature`` and accept
``reasoning_effort`` — both are auto-detected from the model name (overridable). Hosted
**web search** is available only on the OpenAI platform's Responses API; when
``web_tools_enabled=True``, ``chat_tools`` runs the Responses tool loop (function tools +
``{"type": "web_search"}``) instead of chat.completions — the Anthropic client's
counterpart of server-side web tools.

The `openai` package is an optional dependency (`pip install -e '.[agents]'`); it is
imported lazily so the rest of the codebase does not require it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

from src.common.db import get_pool
from src.common.llm.cost import CostCapExceeded, CostMeter
from src.common.llm.factory import strip_code_fence as _strip_code_fence

log = logging.getLogger("vtech.common.llm.openai")

# Reasoning-model detection (provider prefixes like "openai/" stripped first):
# these families reject `temperature` and accept `reasoning_effort`.
_REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)")


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def usd_cost(self) -> float:
        # Rough cost estimate. Real cost tracking uses CostMeter.cost_of() via model_rates
        # in config/research_director.yaml — the RD_LLM_USD_PER_1K_* env vars have been removed.
        return 0.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMClient:
    """Thin wrapper over an OpenAI-compatible client with audit + cost control."""

    def __init__(
        self,
        *,
        model: str | None = None,
        embed_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout_s: int = 120,
        usd_cap_cycle: float = 5.0,
        per_agent_usd_cap: float = 2.5,
        warn_frac: float = 0.75,
        degrade_frac: float = 0.90,
        model_rates: dict[str, dict[str, float]] | None = None,
        max_json_repair_retries: int = 1,
        send_temperature: bool | None = None,
        reasoning_effort: str | None = None,
        web_tools_enabled: bool = False,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "openai is required for the agent layer: pip install -e '.[agents]'"
            ) from exc

        key = api_key or os.environ.get("RD_LLM_API_KEY")
        if not key:
            raise ValueError("no API key for the OpenAI-compatible client "
                             "(pass api_key or set RD_LLM_API_KEY)")
        base = base_url or os.environ.get("RD_LLM_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": request_timeout_s}
        if base:
            client_kwargs["base_url"] = base
        self._client = OpenAI(**client_kwargs)
        self._model = model or os.environ["RD_LLM_MODEL"]
        self._embed_model = embed_model or os.environ.get("RD_EMBED_MODEL", "text-embedding-3-large")

        bare = self._model.split("/")[-1]                       # strip "openai/" style prefixes
        is_reasoning = bool(_REASONING_MODEL_RE.match(bare))
        # Reasoning models (gpt-5*/o*) 400 on `temperature`; auto-omit unless overridden.
        self._send_temperature = (not is_reasoning) if send_temperature is None else send_temperature
        self._reasoning_effort = reasoning_effort if is_reasoning else None
        # Hosted web search lives on the Responses API (OpenAI platform only — NOT on
        # chat.completions and NOT on GitHub Models); gates the chat_tools engine choice.
        self._web_enabled = web_tools_enabled
        self._meter = CostMeter(
            cycle_cap=usd_cap_cycle,
            per_agent_cap=per_agent_usd_cap,
            warn_frac=warn_frac,
            degrade_frac=degrade_frac,
            model_rates=model_rates or {},
        )
        self._max_repair = max_json_repair_retries

    def _sampling_kwargs(self, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Per-model sampling params: omit `temperature` on reasoning models, add
        `reasoning_effort` when the family supports it."""
        kw: dict[str, Any] = {"max_completion_tokens": max_tokens}
        if self._send_temperature:
            kw["temperature"] = temperature
        if self._reasoning_effort:
            kw["reasoning_effort"] = self._reasoning_effort
        return kw

    # ── cost guard ──────────────────────────────────────────────────────────
    def _charge(self, result: LLMResult, agent: str, experiment_id: int | None) -> None:
        cost = self._meter.charge(
            model=result.model, agent=agent,
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        )
        self._audit(result, agent, experiment_id, cost)

    def _audit(self, result: LLMResult, agent: str, experiment_id: int | None, cost: float) -> None:
        # Best-effort telemetry: persists to the research_director schema when it exists.
        # Callers outside the Research Director (e.g. the Phase 23 Model Scout) may run without
        # that schema/DB — a failed audit must never break the LLM call, so we log and continue.
        try:
            pool = get_pool()
            with pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO research_director.llm_call
                        (experiment_id, agent, model_version, prompt_hash, response_hash,
                         prompt_tokens, completion_tokens, usd_cost)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        experiment_id, agent, result.model, _sha(result.text[:64]),
                        _sha(result.text), result.prompt_tokens, result.completion_tokens,
                        cost,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO research_director.gpu_budget_ledger
                        (category, experiment_id, usd_debited)
                    VALUES ('llm', %s, %s)
                    """,
                    (experiment_id, cost),
                )
        except Exception as exc:  # noqa: BLE001 — audit is telemetry, not control flow
            log.debug("llm audit skipped (%s)", exc)

    # ── chat ────────────────────────────────────────────────────────────────
    def chat(
        self,
        *,
        system: str,
        user: str,
        agent: str,
        temperature: float = 0.2,
        max_tokens: int = 3072,
        experiment_id: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._sampling_kwargs(temperature, max_tokens),
        )
        usage = resp.usage
        result = LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            model=self._model,
        )
        self._charge(result, agent, experiment_id)
        return result.text

    def chat_json(
        self,
        *,
        system: str,
        user: str,
        agent: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
        max_tokens: int = 3072,
        experiment_id: int | None = None,
    ) -> dict[str, Any]:
        """Chat returning a schema-validated JSON object, with one repair retry."""
        prompt = user
        last_err = ""
        for attempt in range(self._max_repair + 1):
            raw = self.chat(
                system=system, user=prompt, agent=agent,
                temperature=temperature, max_tokens=max_tokens,
                experiment_id=experiment_id,
            )
            try:
                obj = json.loads(_strip_code_fence(raw))
                jsonschema.validate(obj, schema)
                return obj
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                last_err = str(exc)
                prompt = (
                    f"{user}\n\nYour previous reply was invalid: {last_err}\n"
                    "Return ONLY valid JSON conforming to the schema."
                )
        raise ValueError(f"LLM JSON did not validate after repair: {last_err}")

    # ── tool-calling loop (agentic harness engine) ────────────────────────────
    def chat_tools(
        self,
        *,
        system: str,
        user: str,
        agent: str,
        registry: Any,
        max_iters: int = 8,
        temperature: float = 0.2,
        max_tokens: int = 3072,
        experiment_id: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run a bounded tool-calling loop; returns (final_text, tool_transcript).

        The model may call registered tools (``registry.specs()``); each call is dispatched
        deterministically via ``registry.call()`` and its result fed back. Every model turn
        is cost-charged through ``_charge`` (so the per-cycle cap still applies). The loop
        stops on a final message with no tool calls, or after ``max_iters`` turns.

        ``registry`` is duck-typed (needs ``.specs()`` and ``.call(name, **kwargs)``) to
        avoid an import cycle with :mod:`research_director.registry`.

        Engine selection: with ``web_tools_enabled`` the loop runs on the **Responses API**
        (function tools + hosted ``web_search``); otherwise on chat.completions exactly as
        before (GitHub Models and other shims do not support the Responses hosted tools).
        """
        if self._web_enabled:
            return self._chat_tools_responses(
                system=system, user=user, agent=agent, registry=registry,
                max_iters=max_iters, max_tokens=max_tokens, experiment_id=experiment_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tool_specs = registry.specs()
        transcript: list[dict[str, Any]] = []

        for _ in range(max_iters):
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
                **self._sampling_kwargs(temperature, max_tokens),
            )
            usage = resp.usage
            msg = resp.choices[0].message
            result = LLMResult(
                text=msg.content or "",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                model=self._model,
            )
            self._charge(result, agent, experiment_id)

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content or "", transcript

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    output = registry.call(name, **args)
                except Exception as exc:  # noqa: BLE001 - tool errors are fed back, not fatal
                    output = {"error": str(exc)}
                transcript.append({"tool": name, "args": tc.function.arguments, "output": output})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(output, default=str)[:8000],
                    }
                )

        return "", transcript
    # ── Responses-API tool loop (hosted web_search + function tools) ──────────
    def _chat_tools_responses(
        self,
        *,
        system: str,
        user: str,
        agent: str,
        registry: Any,
        max_iters: int,
        max_tokens: int,
        experiment_id: int | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Bounded tool loop on the OpenAI **Responses API**: client function tools are
        dispatched via ``registry.call``; ``{"type": "web_search"}`` runs hosted on OpenAI
        infrastructure (the platform's counterpart of Anthropic's server tools — not
        available on chat.completions). Prior output items (incl. reasoning items) are
        echoed back each turn, as reasoning models require."""
        tools: list[dict[str, Any]] = [
            {"type": "function", "name": f["function"]["name"],
             "description": f["function"].get("description", ""),
             "parameters": f["function"].get("parameters") or {"type": "object", "properties": {}}}
            for f in registry.specs()
        ]
        tools.append({"type": "web_search"})
        input_items: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        transcript: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(max_iters):
            kw: dict[str, Any] = {"model": self._model, "input": input_items,
                                  "tools": tools, "max_output_tokens": max_tokens}
            if self._reasoning_effort:
                kw["reasoning"] = {"effort": self._reasoning_effort}
            resp = self._client.responses.create(**kw)
            usage = getattr(resp, "usage", None)
            final_text = getattr(resp, "output_text", "") or final_text
            self._charge(
                LLMResult(text=final_text,
                          prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                          completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                          model=self._model),
                agent, experiment_id)

            fn_calls = [it for it in resp.output if getattr(it, "type", "") == "function_call"]
            for it in resp.output:                           # surface hosted searches
                if getattr(it, "type", "") == "web_search_call":
                    action = getattr(it, "action", None)
                    transcript.append({"server_tool": "web_search",
                                       "input": getattr(action, "query", None) or str(action or "")[:200]})
            if not fn_calls:
                return final_text, transcript

            input_items.extend(it.model_dump(exclude_unset=True) for it in resp.output)
            for call in fn_calls:
                name = getattr(call, "name", "")
                try:
                    args = json.loads(getattr(call, "arguments", None) or "{}")
                    output = registry.call(name, **args)
                except Exception as exc:  # noqa: BLE001 - tool errors are fed back, not fatal
                    output = {"error": str(exc)}
                transcript.append({"tool": name, "args": getattr(call, "arguments", ""),
                                   "output": output})
                input_items.append({"type": "function_call_output",
                                    "call_id": getattr(call, "call_id", ""),
                                    "output": json.dumps(output, default=str)[:8000]})

        return final_text, transcript
    # ── embeddings ────────────────────────────────────────────────────────────
    def embed(self, text: str, *, agent: str = "memory", experiment_id: int | None = None) -> list[float]:
        resp = self._client.embeddings.create(model=self._embed_model, input=text)
        return list(resp.data[0].embedding)

    # ── cost-band introspection (warn -> degrade -> halt) ──────────────────────
    def should_degrade(self) -> bool:
        """True once cycle spend crosses the degrade band — agents skip new mining/reflection."""
        return self._meter.should_degrade()

    def agent_over_budget(self, agent: str) -> bool:
        return self._meter.agent_over_budget(agent)

    def cost_tally(self) -> dict[str, float | str]:
        return self._meter.tally()

    def reset_cycle(self) -> None:
        """Zero the per-cycle and per-agent spend at the start of each research cycle."""
        self._meter.reset()
