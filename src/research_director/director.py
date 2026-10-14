"""Research Director — thin cycle driver around the model-improvement orchestrator.

The cognitive work lives in :func:`src.research_director.orchestrator.run_rd_loop` (a Scoping-v2-style
diagnose → cross-review → synthesize → auto-validate loop). The Director's remaining job is the
per-cycle control plane, unchanged from the old design:

  run_cycle():  governor gate → assemble the Context Pack → objective-target stand-down (ADR-014)
                → run the orchestrator → record the plan + validation verdict to the journal.

It is the single writer of the journal row for the cycle; it never touches GPU/LLM directly (the
orchestrator owns those). Budget/cadence guardrails stay in the Governor; cross-cycle "don't
re-propose" dedup flows through the journal's recent signatures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import src.research_director.journal as J
from src.research_director.context_assembler import assemble
from src.research_director.journal import Journal
from src.research_director.mlflow_tracking import MLflowTracker

log = logging.getLogger(__name__)

# ModelImprovementPlan validation verdict → terminal journal state.
_VERDICT_STATE = {
    "VALIDATED_IMPROVED": J.VALIDATED_IMPROVED,
    "VALIDATED_NO_IMPROVEMENT": J.VALIDATED_NO_IMPROVEMENT,
    "HANDOFF_REQUIRED": J.HANDOFF_REQUIRED,
    "INCONCLUSIVE": J.INCONCLUSIVE,
}


@dataclass
class CycleReport:
    ran: bool
    reason: str = ""
    experiment_id: int | None = None
    verdict: str | None = None
    n_changes: int = 0


class ResearchDirector:
    """Coordinates one model-improvement cycle."""

    def __init__(
        self,
        *,
        program_id: str,
        objective_id: str,
        program_objectives: dict[str, Any],
        registry_dir: str | Path,
        model_name: str,
        journal: Journal,
        governor: Any = None,
        tracker: Any = None,
        loop_cfg: dict[str, Any] | None = None,
        assemble_fn: Any = assemble,
        dry_run: bool = False,
    ) -> None:
        self._program_id = program_id
        self._objective_id = objective_id
        self._objectives = program_objectives
        self._registry_dir = Path(registry_dir)
        self._model_name = model_name
        self._journal = journal
        self._governor = governor
        self._tracker = tracker or MLflowTracker(enabled=False)
        self._loop_cfg = loop_cfg
        self._cycle_count = 0
        self._assemble = assemble_fn
        self._dry_run = bool(dry_run)

    def run_cycle(self, *, dry_run: bool | None = None) -> CycleReport:
        is_dry = self._dry_run if dry_run is None else bool(dry_run)
        if self._governor is not None and not is_dry:
            gate = self._governor.may_run()
            if not gate.ok:
                log.info("Director: cycle gated off — %s", gate.reason)
                return CycleReport(False, reason=gate.reason)
        self._cycle_count += 1
        try:
            with self._tracker.iteration(self._program_id, self._cycle_count):
                pack = self._assemble(
                    program_id=self._program_id,
                    program_objectives=self._objectives,
                    registry_dir=self._registry_dir,
                    model_name=self._model_name,
                    objective_id=self._objective_id,
                )

                # Target-pursuit stand-down (ADR-014): if the objective's target is already met,
                # no-op cheaply before any LLM/GPU spend.
                progress = getattr(pack, "objective_progress", None)
                if progress is not None and getattr(progress, "achieved", False) and not is_dry:
                    log.info("Director: OBJECTIVE TARGET ACHIEVED — standing down (no-op cycle)")
                    return CycleReport(True, reason="target_achieved")

                recent_sigs: list[str] = []
                if not is_dry:
                    recent_sigs = self._journal.recent_signatures(self._program_id)
                from src.research_director.orchestrator import run_rd_loop
                state = run_rd_loop(
                    self._program_id, self._objective_id, pack,
                    recent_signatures=recent_sigs, cfg=self._loop_cfg,
                    dry_run=is_dry,
                )

                if is_dry:
                    log.info("Director: dry-run loop finished stages=%s",
                             list((state.get("stages") or {}).keys()))
                    return CycleReport(
                        True, reason="dry_run",
                        n_changes=len((state.get("plan") or {}).get("changes") or []),
                        verdict=(state.get("validation") or {}).get("verdict"),
                    )

                report = self._record(state)
                cost = state.get("cost")
                if cost:
                    self._tracker.log_cost(cost)
                return report
        finally:
            if self._governor is not None and not is_dry:
                self._governor.release_lock()

    # ── journal the plan + verdict ───────────────────────────────────────────────
    def _record(self, state: dict[str, Any]) -> CycleReport:
        from src.research_director.orchestrator import _top_change, rd_signature
        plan = state.get("plan") or {}
        val = state.get("validation") or {}
        top = _top_change(plan)
        if top is None:
            return CycleReport(True, reason="no_plan")

        eid = self._journal.create_experiment(
            program_id=self._program_id,
            title=(top.get("title") or "model improvement")[:120],
            scope=top.get("scope"),
            spec_type="MODEL_IMPROVEMENT",
            hypothesis=top.get("mechanism"),
            config_diff=None,
            metric=(top.get("expected_effect") or {}).get("metric"),
            delta_min=float((self._loop_cfg or {}).get("delta_min", 0.0)),
            folds=None, seed=None, est_epochs=None,
            est_cost_usd=(top.get("cost") or {}).get("est_usd"),
            priority=None,
            spec_hash=rd_signature(top),
            state=J.PLAN_SYNTHESIZED,
        )
        verdict = val.get("verdict", "INCONCLUSIVE")
        to_state = _VERDICT_STATE.get(verdict, J.INCONCLUSIVE)
        self._journal.transition(eid, to_state, "rd_orchestrator", reason=val.get("reason"))
        log.info("Director: cycle recorded experiment_id=%d state=%s verdict=%s",
                 eid, to_state, verdict)
        return CycleReport(True, experiment_id=eid, verdict=verdict,
                           n_changes=len(plan.get("changes") or []))
