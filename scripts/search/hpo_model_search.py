#!/usr/bin/env python3
"""HPO for Phase 23 model-search plugins — reduce NO_SIGNAL collapse (Optuna).

Tunes one model family's hyperparameters plus the three NO_SIGNAL-reduction
levers against a trade objective, maximizing per-direction precision@τ on held-out
OOF predictions from the walk-forward fold(s):

  A  dir_threshold      — directional firing threshold (fire up/down even when
                          flat has higher prob) — decouples "trade" from "beat flat"
  B  class_balance      — per-sample class weighting (None | balanced | sqrt)
     max_delta_step     — xgboost imbalance stabiliser
  C  max_depth, min_child_weight, gamma, learning_rate, n_estimators,
     subsample, colsample_bytree — tree-capacity variations

Precision-first (recall / F1 ignored — prefer *not trading* over trading at low
precision). Tiered, dual-direction-preferring objective at confidence >= τ:

  1. best if a single model has BOTH up & down precision@τ >= ``--precision-bar``
     (default 0.85) with >= ``--min-signals`` (default 25) each — maximise both + volume;
  2. else a single direction over the bar (specialist);
  3. else climb toward the bar / floor.

After the search it harvests the best DUAL model and the best up / down SPECIALISTS
from all trials and writes a ``recommendation`` block: use the dual model if one
exists, otherwise two direction-labelled specialist models.

Usage
-----
  python scripts/search/hpo_model_search.py --objective nq-50pt --family xgboost \
      --n-trials 40 --timeout-min 180 --workers 10

Scores each trial on the **full walk-forward** (all folds → full-range, all-regime
OOF) with full training data by default — this is an evaluation stage, so fidelity
is the default. ``--folds N`` / ``--subsample K`` opt into a faster, lower-fidelity
search; always re-validate the winner at full fidelity via the search controller.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/vtech")

from src.common.data_paths import data_root

logger = logging.getLogger("vtech.hpo.model_search")

_DATA_ROOT = data_root()


def _subsample_fold(fold: dict, n: int, seed: int = 42) -> dict:
    """Random-subsample the TRAIN portion of a fold to *n* rows (keeps OOF full)."""
    X_tr, y_tr = fold["X_train"], fold["y_train"]
    if n <= 0 or len(X_tr) <= n:
        return fold
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=n, replace=False)
    out = dict(fold)
    out["X_train"] = X_tr[idx]
    out["y_train"] = y_tr[idx]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 23 model-search HPO (NO_SIGNAL reduction)")
    p.add_argument("--objective",   required=True, help="Objective id, e.g. nq-50pt")
    p.add_argument("--family",      default="xgboost", help="Model family (default xgboost)")
    p.add_argument("--n-trials",    type=int, default=40)
    p.add_argument("--timeout-min", type=float, default=180.0)
    p.add_argument("--subsample",   type=int, default=0,
                   help="Train-row subsample per trial (0 = full data, default). Set >0 only to "
                        "trade eval fidelity for search speed.")
    p.add_argument("--folds",       type=int, default=0,
                   help="Walk-forward folds to score on (0 = ALL folds = full-range, all-regime "
                        "OOF, default). Set N>0 to score on only the last N folds for speed.")
    p.add_argument("--workers",     type=int, default=10, help="xgboost n_jobs per trial")
    p.add_argument("--min-signals", type=int, default=25,
                   help="Min directional signals per direction (at confidence>=tau) for a "
                        "precision score to count — precision-first objective (default 25).")
    p.add_argument("--precision-bar", type=float, default=0.85,
                   help="Per-direction precision@tau target bar; a model clearing it in BOTH "
                        "directions is a 'dual' champion (default 0.85).")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--pass-n",      type=int, default=1, dest="pass_n",
                   help="which pass of the objective this study searches "
                        "(default 1). One HPO study is one canonical run.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    import optuna
    from src.objectives.loader import load_objective
    from src.search.controller import _build_folds, _load_plugin
    from src.models.base import ModelSpec
    from src.evaluation._kpi import canonical_trade_kpi
    from src.features.stores import default_parquet_dir
    from src.search.tuning import dual_preferring_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # One HPO study IS one canonical run (docs/STORAGE_CONTRACT.md 21). No
    # separate hpo/ hierarchy: the study's objective and the folds it scored
    # live at run level, and each trial is a model folder under its pass.
    from src.common.run_layout import new_run_dir, new_run_ts
    run_ts  = new_run_ts()
    out_dir = new_run_dir(args.objective, run_ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    objective = load_objective(args.objective)
    pdir = default_parquet_dir(objective)

    print(f"⏳ Building walk-forward folds for {args.objective} …")
    t0 = time.time()
    all_folds = _build_folds(objective, pdir)
    if not all_folds:
        raise SystemExit("No folds built — check parquet data / objective.")
    folds = all_folds if args.folds <= 0 else all_folds[-args.folds:]
    folds = [_subsample_fold(f, args.subsample, seed=args.seed) for f in folds]
    print(f"✅ Folds ready in {time.time()-t0:.0f}s "
          f"(scoring on {len(folds)} fold(s), ~{len(folds[0]['X_train']):,} train rows each)")

    kpi_tau = _primary_tau(objective)
    floor   = args.min_signals
    bar     = args.precision_bar

    def objective_fn(trial: "optuna.Trial") -> float:
        params = {
            # ── C: tree capacity ──
            "n_estimators":     trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.3, 6.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # ── B: imbalance handling ──
            "max_delta_step":   trial.suggest_int("max_delta_step", 0, 10),
            "class_balance":    trial.suggest_categorical("class_balance", [None, "balanced", "sqrt"]),
            # ── A: directional firing threshold ──
            "dir_threshold":    trial.suggest_float("dir_threshold", 0.30, 0.60),
            "n_jobs":           args.workers,
        }
        plugin = _load_plugin(args.family)
        spec   = ModelSpec(family=args.family, params=params)
        model  = plugin.build(spec, objective)
        fitted = plugin.fit(model, folds, objective=objective)
        oof    = fitted.oof_preds
        if oof is None or "direction_true" not in oof.columns:
            return 0.0
        gt  = pd.DataFrame({"direction_true": oof["direction_true"]})
        kpi = canonical_trade_kpi(oof, gt, objective, tau=kpi_tau)

        score, mode = dual_preferring_score(kpi, bar=bar, floor=floor)
        for k in ("precision_at_tau", "precision_up", "precision_down",
                  "n_up", "n_down", "f1_up", "f1_down", "directional_f1"):
            trial.set_user_attr(k, kpi.get(k))
        trial.set_user_attr("mode", mode)
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.family}_{args.objective}_{run_ts}",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(f"⏳ Optimising {args.family} — {args.n_trials} trials  "
          f"(dual-preferring precision@{kpi_tau}, bar={bar}, n≥{floor}) …\n")

    def _cb(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        ua = trial.user_attrs
        print(f"  trial {trial.number:>3}  score={trial.value:.4f}  "
              f"mode={ua.get('mode')}  "
              f"up={_fmt(ua.get('precision_up'))}/{ua.get('n_up')}  "
              f"down={_fmt(ua.get('precision_down'))}/{ua.get('n_down')}  "
              f"(dir_f1={_fmt(ua.get('directional_f1'))})  | best={study.best_value:.4f}")

    study.optimize(objective_fn, n_trials=args.n_trials,
                   timeout=args.timeout_min * 60, callbacks=[_cb])

    # ── Harvest dual model + per-direction specialists from all trials ──
    recommendation = _harvest(study.trials, bar, floor, kpi_tau)

    best = {
        "objective_id":  args.objective,
        "family":        args.family,
        "run_ts":        run_ts,
        "n_trials":      len(study.trials),
        "objective_metric": f"dual-preferring precision@{kpi_tau} (bar={bar}, n>={floor})",
        "best_value":    study.best_value,
        "best_params":   study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "recommendation": recommendation,
        "search_config": {"subsample": args.subsample, "folds": len(folds),
                          "workers": args.workers, "kpi_tau": kpi_tau,
                          "min_signals": floor, "precision_bar": bar},
    }
    selected_model_id = _write_canonical_study(
        out_dir, objective_id=args.objective, run_ts=run_ts, objective=objective,
        study=study, folds=folds, pass_n=args.pass_n, seed=args.seed,
        family=args.family, kpi_tau=kpi_tau, best=best,
    )
    best["selected_model_id"] = selected_model_id
    # Kept beside the canonical files: best.json carries the harvested
    # per-direction recommendation, which has no canonical artifact class, and
    # trials.csv is Optuna's own frame. Neither is the run's SSOT.
    (out_dir / "best.json").write_text(json.dumps(best, indent=2, default=str))
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)

    print(f"\n{'─'*64}")
    print(f"  HPO complete — {len(study.trials)} trials")
    print(f"  objective : precision@{kpi_tau}, bar={bar}, n≥{floor} (dual-preferring)")
    _print_recommendation(recommendation)
    print(f"  → {out_dir}/manifest.json  (selected model_{best['selected_model_id']})")
    print(f"{'─'*64}")



def _series_len(value) -> int:
    """Length of an array-like. ``None`` is 0; never boolean-tests the array."""
    if value is None:
        return 0
    return int(len(value))


def _scored_rows(folds: list) -> int:
    """OOF rows the trial was actually scored on, across every fold."""
    total = 0
    for fold in folds or []:
        y = fold.get("y_oof")
        if y is None:
            y = fold.get("y_test")
        total += _series_len(y)
    return total


def _fold_window(fold: dict, index: int) -> dict:
    """One scored fold, from its own metadata rather than a fixed schedule."""
    meta = fold.get("fold_meta")
    src = meta if isinstance(meta, dict) else fold
    train = src.get("train") if isinstance(src.get("train"), dict) else None
    oof = src.get("oof") if isinstance(src.get("oof"), dict) else None
    if train is None or oof is None:
        train = _span_ts(fold.get("ts_train"))
        oof = _span_ts(fold.get("ts_oof"))
    split = str(src.get("split") or fold.get("split") or "walk_forward")
    return {
        "fold": int(src.get("fold", fold.get("fold", index))),
        "split": split,
        "train": {"start": _bound(train, "start"), "end": _bound(train, "end")},
        "oof": {"start": _bound(oof, "start"), "end": _bound(oof, "end")},
    }


def _bound(span: dict | None, key: str) -> str:
    if not span:
        return ""
    value = span.get(key)
    return "" if value is None else str(value)


def _span_ts(ts) -> dict:
    """Calendar bounds of a timestamp array. Empty when the fold has none."""
    if ts is None or _series_len(ts) == 0:
        return {"start": "", "end": ""}
    stamps = pd.to_datetime(np.asarray(ts), utc=True)
    return {
        "start": pd.Timestamp(stamps.min()).isoformat(),
        "end": pd.Timestamp(stamps.max()).isoformat(),
    }


def _validation_from_folds(folds: list) -> dict:
    """``manifest.validation`` for the folds this study scored.

    ``--folds`` may keep several walk-forward windows. The recorded split and
    ``n_folds`` are those windows, not a hardcoded ``last_range`` of one fold.
    """
    windows = [_fold_window(fold, i) for i, fold in enumerate(folds or [])]
    if not windows:
        windows = [{
            "fold": 0, "split": "walk_forward",
            "train": {"start": "", "end": ""},
            "oof": {"start": "", "end": ""},
        }]
    splits = list(dict.fromkeys(w["split"] for w in windows))
    return {
        "split": splits[0] if len(splits) == 1 else "+".join(splits),
        "n_folds": len(windows),
        "fold_windows": windows,
    }


def _write_canonical_study(
    out_dir: Path,
    *,
    objective_id: str,
    run_ts: str,
    objective: dict,
    study,
    folds: list,
    pass_n: int,
    seed: int,
    family: str,
    kpi_tau: float,
    best: dict,
) -> str:
    """Write the study as a canonical run and return the selected model id.

    One HPO study = one run under ``research/runs/<objective_id>/<run_ts>/``:

      manifest.json            shared objective, the folds actually scored
      hpo.json                 sampler, seed, search space, metric, trial count,
                               selected_model_id
      trials_summary.parquet   one row per trial (the lightweight index)
      pass_<n>/model_<id>/     one folder per trial configuration:
                               trial.json + model.json + metrics.json

    ``validation`` records every scored window and its split. It is not
    rewritten as ``last_range`` / ``n_folds=1``.

    Deliberately NOT written per trial: an OOF file, SHAP, a report, a trade
    book, or a fitted weight file. The selected configuration is re-validated
    by an ordinary clean run rather than promoted on its search score.
    """
    import pandas as pd

    from src.experiment.model_identity import (
        allocate_model_id, validate_model_id, write_json,
    )
    from src.common.run_layout import (
        hpo_json_path, merge_run_json, model_dir, trials_summary_path,
    )
    from src.evaluation.economic import assumptions_record

    validation = _validation_from_folds(folds)
    scored_rows = _scored_rows(folds)

    rows: list[dict] = []
    models: list[dict] = []
    selected = ""
    for trial in study.trials:
        params = dict(trial.params or {})
        model_id = validate_model_id(
            allocate_model_id(objective_id, run_ts, pass_n, family, params,
                              feature_spec=None, target=None),
            family)
        status = str(getattr(trial.state, "name", trial.state))
        score = None if trial.value is None else float(trial.value)
        d = model_dir(objective_id, run_ts, pass_n, model_id)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "trial.json", {
            "trial_number": int(trial.number),
            "status": status,
            "score": score,
        })
        # model.json is the resolved MODEL configuration for this trial. The
        # run-level config lives on the manifest and is not repeated here.
        write_json(d / "model.json", {
            "model_id": model_id,
            "model_type": family,
            "hyperparameters": params,
            "target": None,
            "hpo_run_ts": run_ts,
            "trial_number": int(trial.number),
        })
        if score is not None:
            attrs = {k: v for k, v in (trial.user_attrs or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
            write_json(d / "metrics.json", {
                "n_samples": scored_rows,
                "metrics": {"objective_value": score, **attrs},
                "notes": f"HPO trial {trial.number}; search score, not a "
                         f"clean-evaluation result",
            })
        if trial.number == getattr(study.best_trial, "number", -1):
            selected = model_id
        rows.append({"trial_number": int(trial.number), "model_id": model_id,
                     "status": status, "score": score,
                     "params": json.dumps(params, sort_keys=True, default=str)})
        models.append({
            "model_id": model_id, "model_type": family,
            "model_json": f"pass_{pass_n}/model_{model_id}/model.json",
            "trial": f"pass_{pass_n}/model_{model_id}/trial.json",
        })

    pd.DataFrame(rows).to_parquet(
        trials_summary_path(objective_id, run_ts), index=False, compression="zstd")
    write_json(hpo_json_path(objective_id, run_ts), {
        "sampler": type(study.sampler).__name__,
        "seed": int(seed),
        "search_space": {k: str(v) for k, v in
                         (study.best_trial.distributions or {}).items()},
        "metric": best.get("objective_metric"),
        "n_trials": len(study.trials),
        "pass_n": int(pass_n),
        "selected_model_id": selected,
        "direction": str(study.direction.name),
        "note": "the selected configuration must be re-validated by an ordinary "
                "clean run; an HPO score is a search signal, not final evidence",
    })
    # Input lineage: the same runtime-derived refs the controller records, so an
    # HPO run is not less identifiable than an ordinary one.
    from src.search.run_manifest import runtime_datasets
    upstream = [{
        "kind": "labels",
        "logical_id": ref.logical_id,
        "label_contract_id": ref.contract_id,
        "path": ref.location,
    } for ref in runtime_datasets(objective)]

    snapshot = dict(objective)
    snapshot["study"] = {"kind": "hpo", "family": family, "sampler":
                         type(study.sampler).__name__, "n_trials": len(study.trials)}
    merge_run_json(
        objective_id, run_ts,
        {
            "pass": int(pass_n),
            "role": "hyperparameter_search",
            "purpose": (
                "Search one model family's hyperparameters. Every trial is "
                "scored on the same walk-forward folds, recorded on "
                "validation.fold_windows. Trials are comparable to each other; "
                "an HPO score is not a clean evaluation result."
            ),
            "models": models,
            "upstream": upstream,
        },
        config_snapshot=snapshot,
        # Run-level config IS resolved: the objective is the loaded, post-default
        # document and the study block records the search itself. Per-trial
        # hyperparameters live in each model.json.
        config_snapshot_resolved=True,
        validation=validation,
        seeds={"sampler": int(seed), "fold_subsample": int(seed)},
        execution_assumptions=assumptions_record(objective),
    )
    return selected


def _primary_tau(objective: dict) -> float:
    from src.evaluation.kpi_criteria import primary_tau
    return primary_tau(objective)


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _harvest(trials, bar: float, floor: int, tau: float) -> dict:
    """From all completed trials, pick the best DUAL model (both directions over
    ``bar`` with >= ``floor`` signals) and the best up/down SPECIALISTS (highest
    precision in that direction with >= ``floor`` signals). Returns a recommendation
    block: prefer the dual model; else two direction-labelled specialists."""
    completed = [t for t in trials if t.value is not None]

    def _spec(d: str):
        elig = [t for t in completed
                if int(t.user_attrs.get(f"n_{d}") or 0) >= floor
                and not _isnan(t.user_attrs.get(f"precision_{d}"))]
        if not elig:
            return None
        t = max(elig, key=lambda t: float(t.user_attrs[f"precision_{d}"]))
        ua = t.user_attrs
        p, n = float(ua[f"precision_{d}"]), int(ua[f"n_{d}"])
        return {"intended_direction": d, "trial": t.number, "precision": p,
                "n_signals": n, "meets_bar": p >= bar, "params": t.params}

    up_spec, dn_spec = _spec("up"), _spec("down")

    def _both(t) -> bool:
        ua = t.user_attrs
        up_n, dn_n = int(ua.get("n_up") or 0), int(ua.get("n_down") or 0)
        if up_n < floor or dn_n < floor:
            return False
        up_p, dn_p = ua.get("precision_up"), ua.get("precision_down")
        return (not _isnan(up_p) and not _isnan(dn_p)
                and float(up_p) >= bar and float(dn_p) >= bar)

    duals = [t for t in completed if _both(t)]
    dual = None
    if duals:
        t = max(duals, key=lambda t: (float(t.user_attrs["precision_up"])
                                      + float(t.user_attrs["precision_down"])) / 2)
        ua = t.user_attrs
        dual = {"trial": t.number,
                "up_precision": float(ua["precision_up"]), "n_up": int(ua["n_up"]),
                "down_precision": float(ua["precision_down"]), "n_down": int(ua["n_down"]),
                "params": t.params}

    if dual:
        mode = "dual"
    elif (up_spec and up_spec["meets_bar"]) and (dn_spec and dn_spec["meets_bar"]):
        mode = "two_specialists"
    elif (up_spec and up_spec["meets_bar"]) or (dn_spec and dn_spec["meets_bar"]):
        mode = "one_specialist"
    else:
        mode = "none_meet_bar"

    return {"mode": mode, "precision_bar": bar, "min_signals": floor, "tau": tau,
            "dual_model": dual, "up_specialist": up_spec, "down_specialist": dn_spec}


def _print_recommendation(rec: dict) -> None:
    mode = rec["mode"]
    if mode == "dual":
        d = rec["dual_model"]
        print(f"  ✅ DUAL model (trial #{d['trial']}) — both directions over bar:")
        print(f"       up  : precision={_fmt(d['up_precision'])}  n={d['n_up']}")
        print(f"       down: precision={_fmt(d['down_precision'])}  n={d['n_down']}")
    elif mode == "two_specialists":
        u, v = rec["up_specialist"], rec["down_specialist"]
        print("  ⚠️  No single dual model over the bar — use TWO direction-specialist models:")
        print(f"       UP   model (trial #{u['trial']}): precision={_fmt(u['precision'])}  n={u['n_signals']}")
        print(f"       DOWN model (trial #{v['trial']}): precision={_fmt(v['precision'])}  n={v['n_signals']}")
    elif mode == "one_specialist":
        u, v = rec["up_specialist"], rec["down_specialist"]
        ok  = u if (u and u["meets_bar"]) else v
        miss = v if ok is u else u
        print(f"  ⚠️  Only the {ok['intended_direction'].upper()} direction cleared the bar "
              f"(trial #{ok['trial']}, precision={_fmt(ok['precision'])}/{ok['n_signals']}).")
        if miss:
            print(f"       best {miss['intended_direction'].upper()}: "
                  f"{_fmt(miss['precision'])}/{miss['n_signals']} (below bar).")
    else:
        u, v = rec.get("up_specialist") or {}, rec.get("down_specialist") or {}
        print("  ❌ No direction reached the precision bar with n≥floor. Best up/down: "
              f"{_fmt(u.get('precision'))}/{u.get('n_signals')} / "
              f"{_fmt(v.get('precision'))}/{v.get('n_signals')}")


def _fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "—"


if __name__ == "__main__":
    main()
