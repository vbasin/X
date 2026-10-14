"""Autonomous iterate-until-target loop — Phase 23.

Holds ONE model family fixed (the user's pinned choice) and searches its **hyperparameters**
and (optionally) its **feature selection** until the objective's ``kpi.target`` is met, or a
trial / time / cost budget is exhausted — whichever comes first.

It is a thin, deterministic controller layered on the pieces that already exist:

* each iteration builds a **single-candidate manual spec** and runs it through
  :func:`src.search.controller.run_search` (so it inherits fold building, feature selection,
  GPU dispatch, scoring, and the champion gate when ``kpi.champion_gate`` is true);
* the search space is Optuna-driven, reusing :func:`src.search.tuning.suggest_params`;
* "has my threshold been met?" is the gate's own ``kpi.target`` verdict
  (``gate_result.target_ids``) — the exact same contract the automated search uses;
* stopping is target-met **OR** ``budget`` (max_trials / time_budget_min / cost_budget_usd).

The winner is written back out as a ready-to-run manual-candidate spec, so a successful loop
hands you a reproducible model definition.

Usage (CLI)::

    python -m src.search.iterate --spec config/iterate_spec.example.yaml
    python -m src.search.iterate --spec config/iterate_spec.example.yaml --dry-run

Usage (Python)::

    from src.search.iterate import iterate_until_target
    report = iterate_until_target("config/iterate_spec.example.yaml")

GPU note: with ``GPU_INSTANCE_OCID`` set, every iteration dispatches to (and starts) the
billable A10 — bound the run with ``budget.cost_budget_usd`` / ``time_budget_min``, or use
``--dry-run`` / run CPU families with ``GPU_INSTANCE_OCID`` unset to iterate locally.
"""

from __future__ import annotations

from src.common.data_paths import data_root, search_dir

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.models.catalog import get_entry
from src.objectives.loader import load_objective

logger = logging.getLogger("vtech.search.iterate")

_REPO_ROOT = Path(__file__).resolve().parents[2]
from src.common.config_paths import schema_path as _schema_path
_SCHEMA_PATH = _schema_path("iterate_spec.schema.json")
_DATA_ROOT = Path(__import__("os").environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))

_DEFAULT_MAX_TRIALS = 30
_FEATURE_CHOICE_KEY = "__feature_choice__"


@dataclass
class IterateSpec:
    objective_id: str | None
    candidate: dict[str, Any]
    search_space: dict[str, Any]
    feature_choices: list[dict[str, Any]] | None
    budget: dict[str, Any] = field(default_factory=dict)
    source: str = "<dict>"


# ── spec loading ──────────────────────────────────────────────────────────────

def load_iterate_spec(
    spec_or_path: str | Path | dict[str, Any],
    *,
    objective_id: str | None = None,
) -> IterateSpec:
    """Load, validate and normalise an iterate spec (see ``config/iterate_spec.schema.json``)."""
    if isinstance(spec_or_path, dict):
        raw = spec_or_path
        source = "<dict>"
    else:
        from src.common.config_paths import resolve_user_path
        p = resolve_user_path(spec_or_path, kind="specs")
        if not p.exists():
            raise FileNotFoundError(f"Iterate spec not found: {spec_or_path}")
        raw = yaml.safe_load(p.read_text()) or {}
        source = str(p)

    _validate(raw, source)

    cand = raw["candidate"]
    family = cand["family"]
    try:
        base = get_entry(family)
    except KeyError as exc:
        raise ValueError(f"iterate candidate references unknown family '{family}' ({source})") from exc

    spec_obj_id = raw.get("objective_id")
    if objective_id and spec_obj_id and spec_obj_id != objective_id:
        raise ValueError(
            f"iterate spec objective_id '{spec_obj_id}' != run objective '{objective_id}' ({source})")

    search_space = raw.get("search_space") or dict(base.search_space or {})
    if not search_space:
        raise ValueError(
            f"no search_space for family '{family}' and none provided in the spec ({source}) — "
            "cannot iterate hyperparameters without a search space")

    return IterateSpec(
        objective_id=spec_obj_id or objective_id,
        candidate=cand,
        search_space=search_space,
        feature_choices=raw.get("feature_choices"),
        budget=raw.get("budget") or {},
        source=source,
    )


# ── main loop ─────────────────────────────────────────────────────────────────

def iterate_until_target(
    spec_or_path: str | Path | dict[str, Any],
    *,
    objective_id: str | None = None,
    dry_run: bool = False,
    parquet_dir: str | Path | None = None,
    seed: int = 42,
    budget_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Iterate the pinned candidate until the objective target is met or the budget runs out.

    Returns a report dict: ``{objective_id, family, target_met, best, n_trials, elapsed_s,
    est_cost_usd, history, winner_spec_path, report_path}``.
    """
    import optuna

    from src.search.controller import run_search
    from src.search.tuning import suggest_params

    cfg = load_iterate_spec(spec_or_path, objective_id=objective_id)
    obj_id = cfg.objective_id
    if not obj_id:
        raise ValueError("objective_id must be given (in the spec or as an argument)")
    load_objective(obj_id)                       # fail fast on a bad objective

    cand = cfg.candidate
    family = cand["family"]
    cid = cand.get("id") or family
    base_params = dict(cand.get("params") or {})
    base_features = cand.get("features")
    max_rows = cand.get("max_rows")

    budget = {**cfg.budget, **(budget_overrides or {})}
    max_trials = int(budget.get("max_trials") or _DEFAULT_MAX_TRIALS)
    time_budget_s = (float(budget["time_budget_min"]) * 60.0
                     if budget.get("time_budget_min") else None)
    cost_budget_usd = budget.get("cost_budget_usd")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from src.common.run_layout import new_run_dir
    out_dir = new_run_dir(obj_id, f"iterate_{run_ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_trial_cost = _estimate_trial_cost(family)
    state: dict[str, Any] = {
        "target_met": False, "cost": 0.0, "history": [],
        "best": {"value": float("-inf"), "params": None, "features": base_features,
                 "meets_target": False, "score": None},
    }
    t0 = time.time()

    print(f"\n🔁 ITERATE-UNTIL-TARGET — family={family!r} (fixed) id={cid!r} objective={obj_id!r}")
    print(f"     vary: hyperparameters ({len(cfg.search_space)} knobs)"
          + (f" + {len(cfg.feature_choices)} feature choices" if cfg.feature_choices else "")
          + f" | budget: max_trials={max_trials}"
          + (f" time={budget.get('time_budget_min')}min" if time_budget_s else "")
          + (f" cost=${cost_budget_usd}" if cost_budget_usd else ""))

    def _objective(trial: "optuna.Trial") -> float:
        params = {**base_params, **suggest_params(trial, cfg.search_space)}
        if cfg.feature_choices:
            fi = trial.suggest_categorical(_FEATURE_CHOICE_KEY, list(range(len(cfg.feature_choices))))
            features = cfg.feature_choices[fi]
        else:
            features = base_features

        candidate: dict[str, Any] = {"id": cid, "family": family, "params": params}
        if features:
            candidate["features"] = features
        if max_rows is not None:
            candidate["max_rows"] = max_rows
        manual_spec = {"objective_id": obj_id, "candidates": [candidate]}

        summary = run_search(obj_id, manual_spec=manual_spec, ensembles=False,
                             dry_run=dry_run, parquet_dir=parquet_dir)

        cres = next((c for c in summary.get("candidates", []) if c["candidate_id"] == cid), {})
        score = cres.get("score") or {}
        prec = score.get("precision_at_tau")
        value = float(prec) if isinstance(prec, (int, float)) else -1.0
        met = cid in ((summary.get("gate_result") or {}).get("target_ids") or [])

        state["cost"] += per_trial_cost
        state["history"].append({
            "trial": trial.number, "precision_at_tau": prec, "meets_target": met,
            "params": params, "features": features,
        })
        # Best = prefer a target-meeting config; among equal target status, higher precision.
        cur = state["best"]
        better = ((met and not cur["meets_target"])
                  or (met == cur["meets_target"] and value > cur["value"]))
        if better:
            state["best"] = {"value": value, "params": params, "features": features,
                             "meets_target": met, "score": score}
        trial.set_user_attr("meets_target", met)

        icon = "🎯" if met else "  "
        print(f"  {icon} trial {trial.number:>3}  precision@τ={prec!s:<7}  "
              f"target={'MET' if met else 'no':<3}  cost≈${state['cost']:.2f}")

        if met:
            state["target_met"] = True
            trial.study.stop()
        elif cost_budget_usd and state["cost"] >= float(cost_budget_usd):
            logger.info("cost budget $%.2f reached — stopping", float(cost_budget_usd))
            trial.study.stop()
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=max_trials, timeout=time_budget_s)

    elapsed = time.time() - t0
    best = state["best"]
    winner_path = _write_winner_spec(out_dir, obj_id, cid, family, best, max_rows)
    report = {
        "objective_id": obj_id, "family": family, "candidate_id": cid,
        "target_met": state["target_met"],
        "best": {k: best[k] for k in ("value", "params", "features", "meets_target", "score")},
        "n_trials": len(state["history"]),
        "elapsed_s": round(elapsed, 1),
        "est_cost_usd": round(state["cost"], 4),
        "stopped_reason": _stop_reason(state, len(state["history"]), max_trials,
                                       time_budget_s, elapsed, cost_budget_usd),
        "history": state["history"],
        "winner_spec_path": str(winner_path),
    }
    report_path = out_dir / "iterate_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(report_path)

    _print_summary(report)
    return report


# ── helpers ───────────────────────────────────────────────────────────────────

def _estimate_trial_cost(family: str) -> float:
    """Rough per-iteration USD estimate (reuses the search cost model)."""
    try:
        from src.search.cost_model import estimate_cost
        return float(estimate_cost([family])["est_cost_usd"])
    except Exception:
        return 0.0


def _stop_reason(state: dict[str, Any], n_done: int, max_trials: int,
                 time_budget_s: float | None, elapsed: float,
                 cost_budget_usd: Any) -> str:
    if state["target_met"]:
        return "target_met"
    if cost_budget_usd and state["cost"] >= float(cost_budget_usd):
        return "cost_budget_exhausted"
    if time_budget_s and elapsed >= time_budget_s:
        return "time_budget_exhausted"
    if n_done >= max_trials:
        return "max_trials_reached"
    return "stopped"


def _write_winner_spec(out_dir: Path, objective_id: str, cid: str, family: str,
                       best: dict[str, Any], max_rows: Any) -> Path:
    """Emit the best configuration as a ready-to-run manual-candidate spec."""
    candidate: dict[str, Any] = {"id": cid, "family": family, "params": best["params"] or {}}
    if best.get("features"):
        candidate["features"] = best["features"]
    if max_rows is not None:
        candidate["max_rows"] = max_rows
    spec = {"schema_version": "v1", "objective_id": objective_id,
            "ensembles": False, "candidates": [candidate]}
    path = out_dir / "winner_manual_spec.yaml"
    path.write_text("# Best configuration found by iterate_until_target — re-run with:\n"
                    f"#   python -m src.search.controller --objective {objective_id} "
                    f"--manual-spec {path}\n"
                    + yaml.safe_dump(spec, sort_keys=False))
    return path


def _print_summary(r: dict[str, Any]) -> None:
    print(f"\n{'─'*64}")
    print(f"  Iterate complete — {r['family']} / {r['candidate_id']}")
    print(f"  Trials     : {r['n_trials']}   elapsed {r['elapsed_s']}s   est ${r['est_cost_usd']}")
    print(f"  Stopped    : {r['stopped_reason']}")
    tm = "✅ TARGET MET" if r["target_met"] else "— target not met"
    print(f"  Result     : {tm}   best precision@τ={r['best']['value']}")
    print(f"  Winner spec: {r['winner_spec_path']}")
    print(f"{'─'*64}\n")


def _validate(raw: dict[str, Any], source: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        logger.warning("jsonschema not installed — iterate-spec schema validation skipped")
        return
    schema = json.loads(_SCHEMA_PATH.read_text())
    errs = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: list(e.path))
    if errs:
        msgs = "; ".join(f"{list(e.path)}: {e.message}" for e in errs)
        raise ValueError(f"Iterate spec {source} failed schema validation: {msgs}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Phase 23 — autonomously iterate a pinned model+features until the "
                    "objective target is met or a budget is exhausted")
    p.add_argument("--spec", required=True, help="Path to an iterate spec (config/iterate_spec.schema.json)")
    p.add_argument("--objective", default="", help="Objective id (overrides/sets the spec's objective_id)")
    p.add_argument("--parquet-dir", default="", help="Override parquet directory")
    p.add_argument("--dry-run", action="store_true", help="Validate + exercise the loop without training")
    p.add_argument("--max-trials", type=int, default=None, help="Override budget.max_trials")
    p.add_argument("--time-budget-min", type=float, default=None, help="Override budget.time_budget_min")
    p.add_argument("--cost-budget-usd", type=float, default=None, help="Override budget.cost_budget_usd")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    overrides: dict[str, Any] = {}
    if args.max_trials is not None:
        overrides["max_trials"] = args.max_trials
    if args.time_budget_min is not None:
        overrides["time_budget_min"] = args.time_budget_min
    if args.cost_budget_usd is not None:
        overrides["cost_budget_usd"] = args.cost_budget_usd

    iterate_until_target(
        args.spec,
        objective_id=args.objective or None,
        dry_run=args.dry_run,
        parquet_dir=args.parquet_dir or None,
        budget_overrides=overrides or None,
    )


if __name__ == "__main__":
    _cli()
