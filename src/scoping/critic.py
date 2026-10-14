"""Parameterized critic harness — the single adversarial reviewer role (ADR-025 §4).

One ``critic.md`` role (shared adversarial instructions + objection schema + artifact-first→
targeted-verification method), **composed at runtime with the reviewed field's persona as
reference knowledge** plus that field's red-flag checklist (config data). Runs fresh-context
— input is the DELIVERABLE + the author's RETRIEVAL TRACE, never the reasoning scratchpad —
and on a **different model family** than the author when a second provider is configured
(cross-model decorrelation: error-generating and error-checking distributions must not
coincide).

Verification tools available to the critic (targeted, never full re-derivation):
    citation_check(source_ref)              deterministic RAG-index existence check
    search_research(query, tier, k)         omission probes + adversarial counter-evidence
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("vtech.scoping.critic")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _REPO_ROOT / "config"
_CRITIC_PERSONA = _CONFIG / "agents" / "critic.md"
_DUTIES = _CONFIG / "agents" / "scoping_duties.yaml"
from src.common.config_paths import schema_path as _schema_path
_REVIEW_SCHEMA_PATH = _schema_path("critic_objection.schema.json")

MATERIAL = ("blocker", "major")
_SHARED_MARKER = "<!-- SHARED-EXPERTISE-END"


def field_persona(field: str) -> str:
    """Role-neutral domain-expertise head of a field persona (``config/agents/<field>.md``).

    The persona files are dual-use: the Research Director panel loads the whole file (its
    optimization-loop protocol — Context Pack / ExperimentSpec), while scoping and the critic
    need only the field's *identity + analytical lenses*. Everything below the
    ``SHARED-EXPERTISE-END`` marker is RD-only and would mislead a scoping author (wrong output
    contract, nonexistent tools), so it is stripped here. Falls back to the whole file if the
    marker is absent, and to a one-line stub if the file is missing."""
    p = _CONFIG / "agents" / f"{field}.md"
    if not p.exists():
        return f"You are the {field} expert."
    text = p.read_text()
    return text.split(_SHARED_MARKER, 1)[0].rstrip() if _SHARED_MARKER in text else text


def load_duties() -> dict[str, Any]:
    """The scoping duties + red-flag config (config/agents/scoping_duties.yaml)."""
    return yaml.safe_load(_DUTIES.read_text()) or {}


def review_schema() -> dict[str, Any]:
    return json.loads(_REVIEW_SCHEMA_PATH.read_text())


def resolve_critic_provider(author_provider: str, policy: str = "other") -> str | None:
    """Pick the critic's provider per config policy. ``other`` = the family the author did
    NOT use; falls back (with a warning) to the author's when only one is configured."""
    if policy not in ("other", "auto"):
        return policy                                   # explicit provider name
    if policy == "auto":
        return None                                     # llm_factory env resolution
    if author_provider == "anthropic":
        if os.environ.get("RD_LLM_MODEL"):
            return "openai"
    elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Claude_API_KEY"):
        return "anthropic"
    logger.warning("cross-model critic unavailable — only one provider configured; "
                   "falling back to author provider '%s'", author_provider)
    return author_provider


class CriticHarness:
    """critic(field=…) — one harness instance per reviewed field."""

    def __init__(
        self,
        field: str,
        *,
        llm: Any = None,
        provider: str | None = None,
        budget_usd: float = 10.0,
        max_tool_iters: int = 6,
        duties: dict[str, Any] | None = None,
    ) -> None:
        self.field = field
        self.max_tool_iters = max_tool_iters
        self._duties = duties or load_duties()
        self._llm = llm or self._make_llm(provider, budget_usd)
        self._provider = provider or "auto"

    # ── public API ────────────────────────────────────────────────────────────
    def review(
        self,
        deliverable: str,
        retrieval_trace: list[dict[str, Any]] | None,
        *,
        target: str,
        objective_summary: str = "",
        registry: Any = None,
    ) -> dict[str, Any]:
        """Artifact-first → targeted verification → structured CriticReview (schema-valid).

        ``registry`` supplies the verification tools; when provided the critic first runs a
        bounded tool loop (citation grounding / omission probes / counter-evidence), then
        emits the JSON review informed by what it verified.
        """
        system = self._system_prompt()
        user = self._review_prompt(deliverable, retrieval_trace, target, objective_summary)

        verification_notes = ""
        if registry is not None:
            notes, _ = self._llm.chat_tools(
                system=system,
                user=user + "\n\nFIRST: run your targeted verification with the tools "
                            "(citation grounding, omission probes, counter-evidence). "
                            "Then summarize what you verified and what it revealed.",
                agent=f"critic_{self.field}", registry=registry,
                max_iters=self.max_tool_iters, temperature=0.2, max_tokens=3072,
            )
            verification_notes = f"\n\n## YOUR VERIFICATION FINDINGS\n{notes[:6000]}"

        review = self._llm.chat_json(
            system=system,
            user=user + verification_notes
                 + "\n\n## TASK\nOutput the CriticReview JSON now. Remember: "
                   "'no_material_objections' requires ≥3 completed named checks."
                 + "\n\n## OUTPUT JSON SCHEMA (conform EXACTLY)\n```json\n"
                 + json.dumps(review_schema(), indent=1) + "\n```",
            schema=review_schema(), agent=f"critic_{self.field}",
            temperature=0.2, max_tokens=4096,
        )
        review.setdefault("reviewer", {})
        review["reviewer"].setdefault("role", "critic")
        review["reviewer"]["field"] = self.field
        review["reviewer"].setdefault("model", getattr(self._llm, "_model", self._provider))
        return review

    # ── prompt composition ─────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        critic = _CRITIC_PERSONA.read_text() if _CRITIC_PERSONA.exists() else ""
        persona = field_persona(self.field)
        flags = (self._duties.get("red_flags") or {}).get(self.field) or []
        parts = [critic,
                 f"\n\n---\n## Reference domain knowledge — the reviewed field's persona ({self.field})\n"
                 "(This is WHO the author is and what expertise applies. You are NOT this persona; "
                 "use it to catch field-specific errors.)\n\n" + persona]
        if flags:
            parts.append("\n\n---\n## Red-flag checklist for this field (check each explicitly)\n"
                         + "\n".join(f"- {f}" for f in flags))
        return "".join(parts)

    def _review_prompt(self, deliverable: str, trace: list[dict[str, Any]] | None,
                       target: str, objective_summary: str) -> str:
        trace_txt = _compact_trace(trace)
        return (
            (f"## OBJECTIVE CONTEXT\n{objective_summary}\n\n" if objective_summary else "")
            + f"## DELIVERABLE UNDER REVIEW ({target})\n{deliverable}\n\n"
            + f"## AUTHOR'S RETRIEVAL TRACE (what they searched/read — NOT their reasoning)\n{trace_txt}"
        )

    def _make_llm(self, provider: str | None, budget_usd: float) -> Any:
        from src.common.llm import make_llm
        return make_llm(usd_cap_cycle=budget_usd, per_agent_usd_cap=budget_usd,
                        provider=provider or None)


# ── objection utilities (used by the orchestrator) ─────────────────────────────

def material_objections(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Objections that force a revision round (blocker/major)."""
    return [o for o in (review.get("objections") or [])
            if o.get("severity") in MATERIAL]


def filter_cross_objections(review: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Interface-scoped cross-review filter (ADR-025 §4.4): keep only objections that tag
    the interface assumption they attack and are not self-declared out-of-domain.
    Returns (kept, n_dropped)."""
    kept, dropped = [], 0
    for o in review.get("objections") or []:
        if o.get("kind") == "out_of_domain" or not (o.get("interface_assumption") or "").strip():
            dropped += 1
            continue
        kept.append(o)
    if dropped:
        logger.info("cross_review_dropped_out_of_domain n=%d field=%s",
                    dropped, (review.get("reviewer") or {}).get("field"))
    return kept, dropped


def _compact_trace(trace: list[dict[str, Any]] | None, limit: int = 40) -> str:
    """Render a retrieval trace compactly: tool + args + a result preview per step."""
    if not trace:
        return "(no retrieval trace recorded)"
    lines = []
    for step in trace[:limit]:
        name = step.get("tool") or step.get("server_tool") or "?"
        args = json.dumps(step.get("args") or step.get("input") or {}, default=str)[:200]
        out = json.dumps(step.get("output", ""), default=str)[:300]
        lines.append(f"- {name}({args}) → {out}")
    if len(trace) > limit:
        lines.append(f"… {len(trace) - limit} more steps")
    return "\n".join(lines)
