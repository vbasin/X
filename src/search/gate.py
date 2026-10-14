"""Champion gate — Phase 23 W3.

Selects the champion from a set of scored candidates using the objective's
``kpi`` block, which distinguishes two bars:

  requirement — the bar a candidate must clear to be CONSIDERED. Candidates that
                meet ``kpi.requirement`` are *passers*; the highest-ranked passer
                becomes the champion.
  target      — the experimentation goal. Candidates that also meet ``kpi.target``
                are flagged (``target_ids`` / ``champion_meets_target``). This is a
                signal that the program's objective has been reached — never a gate.

Both bars are deterministic; no LLM in the gate. If no candidate meets the
requirement the gate returns ``inconclusive`` and the champion is None; the
search loop continues and the operator reviews the leaderboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.kpi_criteria import (
    DEFAULT_MIN_SIGNALS,
    meets_requirement,
    meets_target,
    requirement_conditions,
)

logger = logging.getLogger("vtech.search.gate")


@dataclass
class GateResult:
    """Outcome of ``apply_champion_gate``."""

    status:       str             # "promoted" | "inconclusive" | "no_candidates"
    champion_id:  str | None = None
    champion_precision: float | None = None
    champion_dir_f1:    float | None = None
    champion_meets_target: bool = False
    passed_ids:   list[str] = field(default_factory=list)   # meet requirement
    target_ids:   list[str] = field(default_factory=list)   # also meet target
    failed_ids:   list[str] = field(default_factory=list)
    reasons:      dict[str, str] = field(default_factory=dict)


def _diag_of(r: Any) -> dict[str, Any]:
    return getattr(r, "diagnostics", None) or {}


# Economics primaries: rank by the economic scalar first, precision as tie-break.
_ECON_PRIMARY = frozenset({"net_expectancy", "hit_rate", "sharpe"})


def _rank_key(r: Any, objective: dict[str, Any] | None = None) -> tuple[float, float]:
    """Rank passers by the objective's primary metric (best first).

    * ``precision_at_tau`` / ``directional_f1`` (default): precision, then dir_f1.
    * ``net_expectancy`` / ``hit_rate`` / ``sharpe`` (ADR-027): economics, then precision.
    """
    sc = r.score or {}
    primary = ((objective or {}).get("kpi") or {}).get("primary") or "precision_at_tau"
    if primary in _ECON_PRIMARY:
        return (float(sc.get(primary) if sc.get(primary) is not None else -1e18),
                float(sc.get("precision_at_tau") or 0.0))
    return (float(sc.get("precision_at_tau") or 0.0),
            float(sc.get("directional_f1") or 0.0))


def _fail_reason(details: list[dict[str, Any]]) -> str:
    """Human-readable summary of why a candidate missed the requirement."""
    if not details:
        return "no_diagnostics"
    parts = []
    for d in details:
        val = d.get("value")
        vs = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
        tau = f"@{d['tau']}" if d.get("tau") is not None else ""
        why = d.get("why")
        tail = f" [{why}]" if why and why != "no_value" else ""
        parts.append(f"{d.get('direction','')}·{d.get('metric')}{tau}={vs}"
                     f"({'>' if d.get('op')=='gt' else '>='}{d.get('min')}){tail}")
    return "requirement_unmet: " + "; ".join(parts)


def champion_gate_enabled(
    objective: dict[str, Any] | None,
    experiment_type: str | None = None,
) -> bool:
    """True only when this run should pick a champion.

    Custom Test never gates (ADR-071): every fitted base/meta is reported.
    Search / HPO / scoping gate only when the objective YAML opts in with
    ``kpi.champion_gate: true`` **and** defines ``kpi.requirement`` conditions
    (the gate). Omitted or false → no champion gate. A confidence cut is used
    only when the YAML declares one (``kpi.tau`` or a ``precision_at_tau``
    requirement row) — never a hidden 0.8.
    """
    if experiment_type == "custom_test":
        return False
    flag = ((objective or {}).get("kpi") or {}).get("champion_gate")
    if flag is not True:
        return False
    return bool(requirement_conditions(objective))


def resolve_champion_gate(
    experiment_type: str | None,
    results: list[Any],
    objective: dict[str, Any],
    *,
    min_signals: int = DEFAULT_MIN_SIGNALS,
) -> GateResult | None:
    """Apply the YAML champion gate, or return ``None`` when it is not enabled."""
    if not champion_gate_enabled(objective, experiment_type):
        return None
    if not results:
        return GateResult(status="no_candidates")
    return apply_champion_gate(results, objective, min_signals=min_signals)


def apply_champion_gate(
    results: list[Any],    # list[CandidateResult] — typed Any to avoid circular import
    objective: dict[str, Any],
    *,
    min_signals: int = DEFAULT_MIN_SIGNALS,
) -> GateResult:
    """Apply the requirement (gate) + target (flag) semantics.

    Parameters
    ----------
    results:
        Scored ``CandidateResult`` objects (status=="ok"), each carrying a
        ``.score`` dict and a ``.diagnostics`` dict.
    objective:
        Parsed Trade Objective dict (reads ``kpi.requirement`` / ``kpi.target``).
    min_signals:
        Default minimum directional signals for precision_at_tau conditions.

    Returns
    -------
    GateResult
        ``status="promoted"`` if ≥1 candidate meets the requirement;
        ``status="inconclusive"`` otherwise.
    """
    if not results:
        return GateResult(status="no_candidates")

    passed, target_ids, failed, reasons = [], [], [], {}
    ranked = sorted(results, key=lambda r: _rank_key(r, objective), reverse=True)

    for r in ranked:
        cid = r.candidate_id
        diag = _diag_of(r)
        score = r.score or {}

        req_ok, req_detail = meets_requirement(
            objective, diag, score, default_min_signals=min_signals)
        tgt_ok, _ = meets_target(
            objective, diag, score, default_min_signals=min_signals)

        # Annotate the candidate so downstream consumers see the target flag.
        try:
            r.meets_target = bool(tgt_ok)
        except Exception:
            pass

        if req_ok:
            passed.append(cid)
            if tgt_ok:
                target_ids.append(cid)
            logger.info(
                "gate_pass cid=%s primary=%s=%.4f prec=%.3f target=%s",
                cid, (objective.get("kpi") or {}).get("primary", "precision_at_tau"),
                float(score.get((objective.get("kpi") or {}).get("primary", "precision_at_tau"))
                      or score.get("precision_at_tau") or 0.0),
                float(score.get("precision_at_tau") or 0.0), tgt_ok,
            )
        else:
            failed.append(cid)
            reasons[cid] = _fail_reason(req_detail)

    if not passed:
        logger.warning("gate_inconclusive: no candidate met the requirement")
        return GateResult(
            status="inconclusive",
            passed_ids=[], target_ids=[], failed_ids=failed, reasons=reasons,
        )

    # Among passers, re-rank by the objective primary (passed list is already in ranked order).
    champ_id = passed[0]
    if ((objective.get("search") or {}).get("topology") or {}).get("mode") == "output_meta":
        meta_pass = [c for c in passed if str(c).startswith("meta__")]
        if meta_pass:
            champ_id = meta_pass[0]
    champ = next(r for r in ranked if r.candidate_id == champ_id)
    return GateResult(
        status="promoted",
        champion_id=champ_id,
        champion_precision=champ.score.get("precision_at_tau"),
        champion_dir_f1=champ.score.get("directional_f1"),
        champion_meets_target=(champ_id in target_ids),
        passed_ids=passed,
        target_ids=target_ids,
        failed_ids=failed,
        reasons=reasons,
    )
