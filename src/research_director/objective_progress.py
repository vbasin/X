"""Objective progress — the deterministic target-pursuit contract (ADR-014 closure).

ADR-014 split the objective KPI into ``requirement`` (champion gate) and ``target`` (the
experimentation *goal* — "the whole experimentation effort can stop"). The requirement side
was wired into the Phase-23 champion gate, but nothing in the Phase-20 research loop ever
consulted the target: the loop had no notion of *how far* the current baseline is from the
goal, and no stand-down when the goal is reached. This module closes that gap:

  * :func:`assess_progress` — score the LATEST_STABLE baseline's diagnostics against the
    objective's ``target`` block (via the single evaluator ``evaluation.kpi_criteria``) and
    produce a compact, prompt-injectable **progress report**: met/unmet per condition and
    the numeric distance for each unmet one.
  * The Director calls it each cycle: **target achieved → the Governor-gated loop stands
    down** (journalled ``TARGET_ACHIEVED`` no-op cycles — never silent, always resumable by
    a changed objective or a regressed baseline); **target not achieved → the loop MUST
    continue**, and the per-condition gaps are injected into every agent's Context Pack so
    the agents aim their next experiment at the *specific* unmet condition instead of
    generic improvement.

Pure metric logic — no GPU, no network, no LLM. The LLM never sets or overrides this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectiveProgress:
    achieved: bool                                  # target block fully satisfied
    has_target: bool                                # objective defines a target at all
    has_metrics: bool                               # baseline diagnostics were available
    details: list[dict[str, Any]] = field(default_factory=list)   # per-condition results

    def render(self, *, max_chars: int = 3000) -> str:
        """Human/LLM-readable progress block for Context-Pack injection."""
        if not self.has_target:
            return "No `target` block defined on the objective — pursue the requirement bar."
        if not self.has_metrics:
            return ("Target defined but the baseline has no scored diagnostics yet "
                    "(cold start) — every target condition is unmet; establish a baseline.")
        lines: list[str] = []
        if self.achieved:
            lines.append("TARGET ACHIEVED — all conditions satisfied by the current baseline.")
        else:
            lines.append("TARGET NOT YET ACHIEVED — the research loop continues until it is.")
        for d in self.details:
            status = "MET  " if d.get("passed") else "UNMET"
            gap = d.get("gap")
            gap_s = f"  (gap: {gap:+.4f})" if isinstance(gap, (int, float)) and not d.get("passed") else ""
            lines.append(f"  [{status}] {d.get('label', d)}{gap_s}")
        unmet = [d for d in self.details if not d.get("passed")]
        if unmet and not self.achieved:
            worst = max(unmet, key=lambda d: abs(d.get("gap") or 0.0))
            lines.append(f"  → largest gap: {worst.get('label')} — aim experiments here first.")
        return "\n".join(lines)[:max_chars]


def assess_progress(
    objective: dict[str, Any],
    baseline_diagnostics: dict[str, Any] | None,
    baseline_score: dict[str, Any] | None = None,
) -> ObjectiveProgress:
    """Score the baseline against the objective's ``target`` block.

    Uses ``evaluation.kpi_criteria`` (the single ADR-014 evaluator) so target semantics
    stay defined in exactly one place. Per-condition ``gap`` = ``min - value`` for unmet
    conditions (positive = shortfall to close).
    """
    kpi = (objective or {}).get("kpi") or {}
    block = kpi.get("target")
    if not block:
        return ObjectiveProgress(achieved=False, has_target=False, has_metrics=False)
    if not baseline_diagnostics and not baseline_score:
        return ObjectiveProgress(achieved=False, has_target=True, has_metrics=False)

    try:
        from evaluation.kpi_criteria import meets_target
    except ImportError:
        from src.evaluation.kpi_criteria import meets_target  # repo-root import style

    achieved, details = meets_target(objective, baseline_diagnostics, baseline_score)
    enriched: list[dict[str, Any]] = []
    for d in details:
        e = dict(d)
        cond = e.get("condition") or {}
        metric = cond.get("metric", e.get("metric", "?"))
        direction = cond.get("direction", e.get("direction", ""))
        tau = cond.get("tau", e.get("tau"))
        floor = cond.get("min", e.get("min"))
        value = e.get("value")
        e["label"] = " ".join(
            str(p) for p in (direction, metric, f"@{tau}" if tau is not None else "",
                             f">= {floor}" if floor is not None else "",
                             f"(now {value:.4f})" if isinstance(value, (int, float)) else "(no value)")
            if str(p)
        )
        if isinstance(value, (int, float)) and isinstance(floor, (int, float)):
            e["gap"] = float(floor) - float(value)
        enriched.append(e)
    return ObjectiveProgress(achieved=bool(achieved), has_target=True,
                             has_metrics=True, details=enriched)
