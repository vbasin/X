"""Anthropic-native LLM client (shared cognitive substrate under src.common.llm).

Claude Messages API counterpart to :class:`~src.common.llm.openai_client.LLMClient`.
Mixes client tools (registry-dispatched) with Anthropic server tools (web_search / web_fetch).
Duck-type compatible with LLMClient for chat / chat_json / chat_tools + cost bands.
Cost governance uses :class:`~src.common.llm.cost.CostMeter`; audit is best-effort.

Secrets/config (env):
  API key : ANTHROPIC_API_KEY -> Claude_API_KEY
  model   : ANTHROPIC_MODEL   -> "claude-opus-4-8"
  base URL: ANTHROPIC_BASE_URL (optional)

``anthropic`` is an optional dependency (``pip install -e '.[agents]'``), imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import jsonschema

from src.common.llm.cost import CostMeter
from src.common.llm.factory import strip_code_fence as _strip_code_fence

log = logging.getLogger("vtech.common.llm.anthropic")

# Anthropic server-tool type strings (GA — see the tool reference). Pinned to the basic
# versions so no beta header / code-execution sandbox is required; newer dynamic-filtering
# variants (…_20260209+) are drop-in upgrades if desired.
_WEB_SEARCH_TYPE = "web_search_20250305"
_WEB_FETCH_TYPE = "web_fetch_20250910"

_DEFAULT_MODEL = "claude-opus-4-8"
_TOOL_RESULT_CLIP = 8000        # cap bytes of a client tool_result fed back to the model
# chat_json truncation escalation ceiling. claude-opus-4-8 supports 128K output tokens; the
# JSON path STREAMS (see _create) so there is no non-streaming timeout constraint on the size.
# 64K gives the critical scoping steps ample room to finish a large plan. If a schema-bound
# reply is still cut off at max_tokens (stop_reason == "max_tokens" → truncated JSON → parse
# error), chat_json doubles the budget up to this ceiling before falling back to a repair prompt.
_JSON_MAX_TOKENS_CEILING = 64000


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class AnthropicLLM:
    """Anthropic Messages-API client with an agentic tool loop, cost metering, and audit.

    Drop-in for the scoping agent: implements ``chat_tools`` and ``chat_json`` with the same
    keyword surface as ``research_director.llm.LLMClient``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout_s: int = 600,
        max_tokens: int = 4096,
        usd_cap_cycle: float = 5.0,
        per_agent_usd_cap: float = 2.5,
        warn_frac: float = 0.75,
        degrade_frac: float = 0.90,
        model_rates: dict[str, dict[str, float]] | None = None,
        max_json_repair_retries: int = 1,
        send_temperature: bool | None = None,
        web_tools_enabled: bool | None = None,
        web_max_uses: int | None = None,
        web_fetch_max_content_tokens: int | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "anthropic is required for the Claude agent layer: pip install -e '.[agents]'"
            ) from exc

        key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")            
            or os.environ.get("Claude_API_KEY")
        )
        if not key:
            raise ValueError(
                "no Anthropic API key found (set ANTHROPIC_API_KEY or Claude_API_KEY)"
            )
        base = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": key, "timeout": request_timeout_s}
        if base:
            client_kwargs["base_url"] = base
        self._client = Anthropic(**client_kwargs)

        self._model = (
            model
            or os.environ.get("ANTHROPIC_MODEL")
            or _DEFAULT_MODEL
        )
        self._max_tokens = max_tokens
        self._max_repair = max_json_repair_retries
        self._last_stop_reason: str | None = None   # set by _create; read by chat_json for truncation
        # Newer Claude models (e.g. claude-opus-4-8) DEPRECATE `temperature` and 400 if it is
        # sent. Default to omitting it; opt in via arg or ANTHROPIC_SEND_TEMPERATURE=1 for older
        # models that still accept it.
        if send_temperature is None:
            send_temperature = os.environ.get("ANTHROPIC_SEND_TEMPERATURE", "0").strip().lower() \
                in ("1", "true", "yes")
        self._send_temperature = send_temperature
        self._meter = CostMeter(
            cycle_cap=usd_cap_cycle,
            per_agent_cap=per_agent_usd_cap,
            warn_frac=warn_frac,
            degrade_frac=degrade_frac,
            model_rates=model_rates or {},
        )

        # Web (server) tool configuration — reuse the existing VTECH_WEB_* knobs.
        if web_tools_enabled is None:
            web_tools_enabled = os.environ.get("VTECH_WEB_SEARCH_ENABLED", "1").strip().lower() \
                not in ("", "0", "false", "no")
        self._web_enabled = web_tools_enabled
        self._web_max_uses = int(
            web_max_uses if web_max_uses is not None
            else os.environ.get("VTECH_WEB_SEARCH_MAX_QUERIES", "8")
        )
        self._web_fetch_max_tokens = int(
            web_fetch_max_content_tokens if web_fetch_max_content_tokens is not None
            else os.environ.get("VTECH_WEB_FETCH_MAX_TOKENS", "6000")
        )

    # ── cost + audit ────────────────────────────────────────────────────────
    def _charge(self, result: LLMResult, agent: str, experiment_id: int | None) -> None:
        cost = self._meter.charge(
            model=result.model, agent=agent,
            prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        )
        self._audit(result, agent, experiment_id, cost)

    def _audit(self, result: LLMResult, agent: str, experiment_id: int | None, cost: float) -> None:
        # Best-effort telemetry: the scoping agent commonly runs without the research_director
        # schema/DB, so a failed audit must never break the LLM call — log and continue.
        try:
            from src.common.db import get_pool

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
                        _sha(result.text), result.prompt_tokens, result.completion_tokens, cost,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO research_director.gpu_budget_ledger (category, experiment_id, usd_debited)
                    VALUES ('llm', %s, %s)
                    """,
                    (experiment_id, cost),
                )
        except Exception as exc:  # noqa: BLE001 — audit is telemetry, not control flow
            log.debug("anthropic llm audit skipped (%s)", exc)

    # ── low-level create ──────────────────────────────────────────────────────
    def _create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        agent: str,
        experiment_id: int | None,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": messages,
        }
        if self._send_temperature and temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        # Stream large single-shot generations (the JSON path): Opus 4.8 can emit up to 128K
        # output tokens, but a big NON-streaming reply trips the SDK's ~10-min timeout guard.
        # Streaming + get_final_message() removes that ceiling so the budget is bounded only by
        # what the step needs (scoping is correctness-critical — we do not cap to save tokens).
        if stream:
            with self._client.messages.stream(**kwargs) as s:
                resp = s.get_final_message()
        else:
            resp = self._client.messages.create(**kwargs)
        self._last_stop_reason = getattr(resp, "stop_reason", None)
        usage = getattr(resp, "usage", None)
        result = LLMResult(
            text=_text_of(resp),
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=self._model,
        )
        self._charge(result, agent, experiment_id)
        return resp

    # ── plain chat (text) ──────────────────────────────────────────────────────
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
        resp = self._create(
            system=system, messages=[{"role": "user", "content": user}], agent=agent,
            experiment_id=experiment_id, tools=None, temperature=temperature, max_tokens=max_tokens,
            stream=True,   # text/JSON single-shot path — stream so large max_tokens can't time out
        )
        return _text_of(resp)

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
        """Chat returning a schema-validated JSON object.

        Two failure modes are handled distinctly. If the reply was cut off at the token
        ceiling (``stop_reason == "max_tokens"`` → truncated, unparseable JSON — the live
        synthesis crash), asking the model to "return valid JSON" cannot help: the output is
        simply too large for the budget. Instead we DOUBLE ``max_tokens`` (up to
        ``_JSON_MAX_TOKENS_CEILING``) and re-issue the same request. A genuinely malformed but
        complete reply gets the usual repair prompt. Escalations and repairs are counted
        separately so a truncation never consumes the repair budget."""
        prompt = user
        last_err = ""
        cur_max = max_tokens
        repairs_left = self._max_repair
        escalations_left = 3
        while True:
            raw = self.chat(
                system=system, user=prompt, agent=agent,
                temperature=temperature, max_tokens=cur_max, experiment_id=experiment_id,
            )
            truncated = self._last_stop_reason == "max_tokens"
            try:
                obj = json.loads(_strip_code_fence(raw))
                jsonschema.validate(obj, schema)
                return obj
            except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
                last_err = str(exc)
                if truncated and escalations_left > 0 and cur_max < _JSON_MAX_TOKENS_CEILING:
                    escalations_left -= 1
                    cur_max = min(cur_max * 2, _JSON_MAX_TOKENS_CEILING)
                    prompt = user      # same task — it just needs room to finish
                    log.warning("chat_json reply truncated at max_tokens (agent=%s) — "
                                "retrying with max_tokens=%d", agent, cur_max)
                    continue
                if repairs_left > 0:
                    repairs_left -= 1
                    prompt = (
                        f"{user}\n\nYour previous reply was invalid: {last_err}\n"
                        "Return ONLY valid JSON conforming to the schema."
                    )
                    continue
                if truncated:
                    raise ValueError(
                        "Anthropic JSON still truncated at the max_tokens ceiling "
                        f"({_JSON_MAX_TOKENS_CEILING}) for agent={agent}: {last_err}")
                raise ValueError(f"Anthropic JSON did not validate after repair: {last_err}")

    # ── agentic tool loop (client tools + Anthropic server tools) ──────────────
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
        """Run a bounded plan→act→observe loop; returns ``(final_text, transcript)``.

        Custom (client) tools come from ``registry`` (duck-typed: ``.specs()`` in OpenAI shape
        + ``.call(name, **kwargs)``) and are converted to Anthropic ``input_schema`` tools.
        Anthropic-executed ``web_search`` / ``web_fetch`` are appended automatically when web
        access is enabled. Client ``tool_use`` blocks are dispatched via ``registry.call`` and
        fed back as ``tool_result``; server-tool blocks are executed by Anthropic and surfaced
        in the transcript for observability. Every model turn is cost-charged (the per-cycle
        cap still applies); the loop ends on a non-``tool_use`` stop or after ``max_iters``.
        """
        tools = self._compose_tools(registry)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        transcript: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(max_iters):
            resp = self._create(
                system=system, messages=messages, agent=agent, experiment_id=experiment_id,
                tools=tools, temperature=temperature, max_tokens=max_tokens,
            )
            final_text = _text_of(resp)
            stop = getattr(resp, "stop_reason", None)

            # Surface any Anthropic-executed server-tool calls for observability.
            for block in resp.content:
                if getattr(block, "type", None) == "server_tool_use":
                    transcript.append({
                        "server_tool": getattr(block, "name", "?"),
                        "input": getattr(block, "input", None),
                    })

            if stop == "pause_turn":
                # A long-running server tool paused mid-turn: echo content back and continue.
                messages.append({"role": "assistant", "content": resp.content})
                continue
            if stop != "tool_use":
                return final_text, transcript

            # Dispatch client tool_use blocks and feed results back.
            messages.append({"role": "assistant", "content": resp.content})
            tool_results: list[dict[str, Any]] = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = getattr(block, "name", "")
                args = getattr(block, "input", None) or {}
                is_error = False
                try:
                    output = registry.call(name, **args)
                except Exception as exc:  # noqa: BLE001 — tool errors feed back, not fatal
                    output, is_error = {"error": str(exc)}, True
                transcript.append({"tool": name, "args": args, "output": output})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id", ""),
                    "content": json.dumps(output, default=str)[:_TOOL_RESULT_CLIP],
                    "is_error": is_error,
                })
            if not tool_results:
                # stop_reason was tool_use but nothing client-side to run (server-only) — done.
                return final_text, transcript
            messages.append({"role": "user", "content": tool_results})

        return final_text, transcript

    # ── tool assembly ──────────────────────────────────────────────────────────
    def _compose_tools(self, registry: Any) -> list[dict[str, Any]]:
        tools = [_openai_to_anthropic(spec) for spec in registry.specs()]
        tools.extend(self._server_tools())
        return tools

    def _server_tools(self) -> list[dict[str, Any]]:
        if not self._web_enabled:
            return []
        return [
            {"type": _WEB_SEARCH_TYPE, "name": "web_search", "max_uses": self._web_max_uses},
            {
                "type": _WEB_FETCH_TYPE,
                "name": "web_fetch",
                "max_uses": self._web_max_uses,
                "citations": {"enabled": True},
                "max_content_tokens": self._web_fetch_max_tokens,
            },
        ]

    # ── cost-band introspection (parity with LLMClient) ────────────────────────
    def should_degrade(self) -> bool:
        return self._meter.should_degrade()

    def agent_over_budget(self, agent: str) -> bool:
        return self._meter.agent_over_budget(agent)

    def cost_tally(self) -> dict[str, float | str]:
        return self._meter.tally()

    def reset_cycle(self) -> None:
        self._meter.reset()


# ── module helpers ────────────────────────────────────────────────────────────

def _openai_to_anthropic(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI ``{"type":"function","function":{…}}`` spec to Anthropic shape."""
    fn = spec.get("function", spec)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _text_of(resp: Any) -> str:
    """Concatenate all ``text`` blocks from an Anthropic message response."""
    parts = [
        getattr(b, "text", "")
        for b in getattr(resp, "content", []) or []
        if getattr(b, "type", None) == "text"
    ]
    return "".join(parts).strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
