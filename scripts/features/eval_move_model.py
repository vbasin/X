#!/usr/bin/env python3
"""Stage 1 evaluation: does a move *classifier* rank better than a P&L regressor?

The incumbent model is an ``XGBRegressor`` fitted on ``pnl_r`` — realised P&L — and
ranked by predicted expectancy. That conflates two questions. A speculative wide
target beats a high-conviction tight one whenever the extra distance outweighs the
lower probability, so "how likely is the move" and "how much is the move worth"
are optimised as one number.

This script scores the alternative: a classifier on ``y_move``, trained only on
state features, which answers the first question alone. Both are evaluated on the
same walk-forward purged folds and compared on ranking quality, not on P&L, because
P&L is exactly the quantity under dispute.

**Every statistic is uniqueness-weighted.** At a 5-second cadence overlapping labels
outnumber independent ones by two orders of magnitude, and the raw row count is not
a sample size. ``n_eff`` is reported beside ``n`` throughout; where they diverge by
10x, so does the honest t-statistic.

Usage::

    python scripts/features/eval_move_model.py --dir data/sources/labels/move_matrix
    python scripts/features/eval_move_model.py --dir ... --no-order-flow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.uniqueness import (  # noqa: E402
    effective_n, label_uniqueness, weighted_tstat,
)

logger = logging.getLogger("vtech.scripts.eval_move_model")

TAU_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
# Outcome columns and anything carrying an absolute price. A tree that splits on a
# raw level memorises the price regime rather than the setup.
LEAK = {"y_move", "reached", "t_move", "mfe_pts", "mae_pts", "price",
        "ts", "session", "side", "weight"}


def _load(d: Path) -> pd.DataFrame:
    if d.is_file():
        files = [d]
    else:
        files = sorted(d.glob("move_*.parquet")) or sorted(d.glob("training_matrix*.parquet"))
    if not files:
        raise SystemExit(f"no training matrix under {d}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["session"] = pd.to_datetime(df["session"])
    return df.sort_values("ts").reset_index(drop=True)


def _fold_masks(df, fold: dict):
    """(train, test) frames for one shared-validator fold row.

    The schedule comes from ``src.validation.folds_from_objective``; this only
    applies it. A session on or after the test window's first session is removed
    from training: a label can run to the session flatten, so a same-day split
    would leak the test session's own forward path into the fit.
    """
    import pandas as pd

    tr_end = pd.Timestamp(fold["train"]["end"])
    oof_start = pd.Timestamp(fold["oof"]["start"])
    oof_end = pd.Timestamp(fold["oof"]["end"])
    sess = df["session"]
    if getattr(sess.dtype, "tz", None) is not None:
        tr_end, oof_start, oof_end = (t.tz_localize(sess.dt.tz) for t in
                                     (tr_end, oof_start, oof_end))
    train = df[sess <= tr_end]
    test = df[(sess >= oof_start) & (sess <= oof_end)]
    if not test.empty:
        train = train[train["session"] < test["session"].min()]
    return train, test



def _ranking_stats(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    """Weighted AUC and base rate.

    AUC is the weighted Mann-Whitney statistic computed per distinct score, so tied
    predictions contribute half a win rather than being ordered arbitrarily.
    """
    ok = np.isfinite(pred) & np.isfinite(y)
    y, pred, w = y[ok], pred[ok], w[ok]
    pos_w = np.where(y == 1, w, 0.0)
    neg_w = np.where(y == 0, w, 0.0)
    W_pos, W_neg = pos_w.sum(), neg_w.sum()
    base = float(np.average(y, weights=w)) if w.sum() > 0 else np.nan
    if W_pos <= 0 or W_neg <= 0:
        return {"auc": np.nan, "base_rate": base}
    _, inv = np.unique(pred, return_inverse=True)
    neg_by = np.bincount(inv, weights=neg_w)
    pos_by = np.bincount(inv, weights=pos_w)
    neg_below = np.concatenate(([0.0], np.cumsum(neg_by)[:-1]))
    wins = float(np.sum(pos_by * (neg_below + 0.5 * neg_by)))
    return {"auc": wins / (W_pos * W_neg), "base_rate": base}


def _gate_stats(g: pd.DataFrame, label: str, n_sessions: int,
                base_rate: float = 0.0) -> dict:
    """Outcome of a gated selection, with overlap-discounted significance.

    ``t_stat_w`` tests precision against *base_rate* using the effective sample
    size, so a gate that fires on 40,000 overlapping rows covering fifty genuine
    moves cannot borrow significance from the sampling grid.
    """
    if g.empty:
        return {"policy": label, "n": 0}
    w = g["weight"].to_numpy()
    hit = g["y_move"].to_numpy(dtype="float64")
    mfe = g["mfe_pts"].to_numpy() / g["r_value"].to_numpy()
    return {
        "policy": label,
        "n": len(g),
        "n_eff": round(effective_n(w), 1),
        "per_sess": round(len(g) / max(1, n_sessions), 2),
        "precision": round(float(np.average(hit, weights=w)), 4),
        "mean_mfe_r": round(float(np.average(mfe, weights=w)), 4),
        "t_vs_base": round(weighted_tstat(hit - base_rate, w), 3),
        "median_t_move_s": float(g.loc[g["t_move"] >= 0, "t_move"].median())
        if (g["t_move"] >= 0).any() else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="explicit population file or directory. Omit with "
                         "--objective and --run-ts to join the model's inputs.")
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--min-train-months", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--no-order-flow", action="store_true",
                    help="drop TBBO-derived columns; shows what price alone could do")
    ap.add_argument("--no-weights", action="store_true",
                    help="disable uniqueness weighting; shows the inflation it removes")
    ap.add_argument("--out-scores", default=None,
                    help="optional extra parquet of the compact OOF")
    ap.add_argument("--no-shap", action="store_true",
                    help="skip the SHAP report written after the OOF file")
    ap.add_argument("--objective", default=None,
                    help="objective_id; writes oof_predictions.parquet into the run")
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--pass-n", type=int, default=1, dest="pass_n")
    ap.add_argument("--model-id", default=None,
                    help="unique id for this trained model. Omit to allocate one "
                         "from the model type, hyperparameters, feature view, and "
                         "target. Not the model type.")
    ap.add_argument("--model-type", default="xgboost")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.dir and not (args.objective and args.run_ts):
        logger.error("pass --dir, or --objective and --run-ts, so the run can join its inputs")
        return 1
    from xgboost import XGBClassifier

    run_ts = None
    model_id = None
    if args.objective:
        from src.experiment.analysis import load_run_frame
        from src.experiment.model_identity import (
            allocate_model_id, validate_model_id,
        )
        from src.common.run_layout import new_run_ts
        from src.training.cascade_run_spec import move_feature_spec, xgb_hyperparameters
        run_ts = args.run_ts or new_run_ts()
        hyper = xgb_hyperparameters(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, seed=args.seed)
        feature_spec = move_feature_spec(drop_order_flow=args.no_order_flow)
        model_id = validate_model_id(
            args.model_id or allocate_model_id(
                args.objective, run_ts, args.pass_n, args.model_type,
                hyper, feature_spec=feature_spec, target="y_move"),
            args.model_type)
        df = load_run_frame(
            args.objective, run_ts, args.pass_n, model_id,
            explicit=Path(args.dir) if args.dir else None)
    else:
        df = _load(Path(args.dir))

    df["weight"] = 1.0 if args.no_weights else label_uniqueness(
        df, span_col="t_move", horizon=args.horizon)
    logger.info("rows %s | sessions %d | n_eff %s | base rate %.4f",
                f"{len(df):,}", df["session"].nunique(),
                f"{effective_n(df['weight'].to_numpy()):,.0f}", df["y_move"].mean())

    feats = sorted(c for c in df.columns
                   if c not in LEAK and df[c].dtype.kind in "fiu"
                   and c not in ("r_value",))
    if args.no_order_flow:
        drop = [c for c in feats if c.startswith("fs_")
                or any(k in c for k in ("cvd", "ofi", "delta"))]
        feats = [c for c in feats if c not in drop]
    logger.info("features %d", len(feats))

    # ONE validator for both tracks (src/validation/). The objective's
    # eval block chooses the schedule; the CLI flags are the runtime's
    # override of it, and the RESOLVED values are what the manifest records.
    from src.validation import folds_from_objective
    objective_doc: dict = {}
    if args.objective:
        from src.objectives.loader import load_objective
        objective_doc = dict(load_objective(args.objective))
    folds = folds_from_objective(
        objective_doc, sessions=df["session"].unique(),
        overrides={"test_months": args.test_months,
                   "min_train_months": args.min_train_months},
    )
    if not folds:
        logger.error("not enough months for a fold")
        return 1
    logger.info("folds: %d", len(folds))

    oof = []
    for fi, fold in enumerate(folds):
        tr, te = _fold_masks(df, fold)
        if tr.empty or te.empty:
            continue
        pos = float(tr["y_move"].mean())
        m = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                          learning_rate=args.learning_rate, subsample=0.8,
                          colsample_bytree=0.8, random_state=args.seed,
                          n_jobs=0, eval_metric="logloss",
                          scale_pos_weight=(1 - pos) / max(pos, 1e-9))
        m.fit(tr[feats].to_numpy("float32"), tr["y_move"].to_numpy("int8"),
              sample_weight=tr["weight"].to_numpy("float32"))
        p = m.predict_proba(te[feats].to_numpy("float32"))[:, 1]
        t = te.copy()
        t["p_move"] = p
        t["fold"] = fi
        # Thresholds must come from train, never from the test fold being scored.
        p_tr = m.predict_proba(tr[feats].to_numpy("float32"))[:, 1]
        for q in TAU_QUANTILES:
            t[f"tau_{q}"] = float(np.quantile(p_tr, q))
        oof.append(t)
        logger.info("fold %d | train %s..%s (%s rows) | test %s..%s (%s rows) | pos %.3f",
                    fi, fold["train"]["start"], fold["train"]["end"], f"{len(tr):,}",
                    fold["oof"]["start"], fold["oof"]["end"], f"{len(te):,}", pos)

    if not oof:
        logger.error("no folds produced predictions")
        return 1
    o = pd.concat(oof, ignore_index=True)
    n_sess = o["session"].nunique()
    w = o["weight"].to_numpy()

    print("\n=== Ranking quality (uniqueness-weighted) ===")
    rk = _ranking_stats(o["y_move"].to_numpy(), o["p_move"].to_numpy(), w)
    print(f"  rows            {len(o):,}")
    print(f"  n_eff           {effective_n(w):,.0f}   ({effective_n(w)/len(o):.4f} of raw)")
    print(f"  base rate       {rk['base_rate']:.4f}")
    print(f"  weighted AUC    {rk['auc']:.4f}")

    print("\n=== Gate frontier (tau from TRAIN quantiles) ===")
    base = rk["base_rate"]
    rows = [_gate_stats(o, "all rows", n_sess, base)]
    for q in TAU_QUANTILES:
        g = o[o["p_move"] >= o[f"tau_{q}"]]
        rows.append(_gate_stats(g, f"train q={q}", n_sess, base))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Lift over base rate ===")
    for q in TAU_QUANTILES:
        g = o[o["p_move"] >= o[f"tau_{q}"]]
        if g.empty:
            continue
        prec = float(np.average(g["y_move"], weights=g["weight"]))
        print(f"  q={q:<5} precision {prec:.4f}  lift {prec / max(base, 1e-9):.3f}x"
              f"  n={len(g):,}  n_eff={effective_n(g['weight'].to_numpy()):,.0f}")

    print("\n=== Per-fold at q=0.95 ===")
    rows = []
    for fi, g in o.groupby("fold"):
        sel = g[g["p_move"] >= g["tau_0.95"]]
        r = _gate_stats(sel, f"fold {fi}", g["session"].nunique(), base)
        r["months"] = f"{g['session'].min():%Y-%m}..{g['session'].max():%Y-%m}"
        rows.append(r)
    fold_rows = rows
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Per-side at q=0.95 ===")
    rows = []
    for side, g in o.groupby("side"):
        sel = g[g["p_move"] >= g["tau_0.95"]]
        rows.append(_gate_stats(sel, "long" if side == 1 else "short",
                                g["session"].nunique(), base))
    print(pd.DataFrame(rows).to_string(index=False))

    if args.objective:
        from src.experiment.analysis import write_shap_report
        from src.experiment.artifacts import compact_oof_columns
        from src.experiment.model_identity import write_json, write_model_json
        from src.common.run_layout import merge_run_json, read_run, resolve_model
        from src.training.cascade_run_spec import CASCADE_PASS
        paths = resolve_model(args.objective, run_ts, args.pass_n, model_id)
        paths["oof_predictions"].parent.mkdir(parents=True, exist_ok=True)
        kept = compact_oof_columns(list(o.columns), feats)
        o.loc[:, kept].to_parquet(paths["oof_predictions"], index=False, compression="zstd")
        label_id = None
        inputs = None
        try:
            entry = next(p for p in read_run(args.objective, run_ts).get("passes") or []
                         if int(p.get("pass", -1)) == args.pass_n)
            label_id = entry.get("label_contract_id")
            inputs = entry.get("inputs")
        except (FileNotFoundError, StopIteration):
            label_id = None
        if paths["model_json"].is_file():
            prior = json.loads(paths["model_json"].read_text())
            inputs = inputs or prior.get("inputs")
            label_id = label_id or prior.get("label_contract_id")
        write_model_json(
            paths["model_dir"],
            model_id=model_id, model_type=args.model_type,
            hyperparameters=hyper,
            label_contract_id=label_id,
            oof_columns=kept,
            feature_spec=feature_spec,
            feature_columns=feats,
            inputs=inputs,
            target="y_move",
        )
        if not args.no_shap:
            write_shap_report(
                o, feats, "y_move", paths["report_dir"],
                max_depth=args.max_depth, n_estimators=args.n_estimators,
                learning_rate=args.learning_rate, seed=args.seed,
                title=f"pass {args.pass_n} {model_id}",
            )
        gated = o[o["p_move"] >= o["tau_0.95"]]
        write_json(paths["metrics"], {
            "n_samples": int(len(o)),
            "metrics": {
                "base_rate": rk["base_rate"],
                "weighted_auc": rk["auc"],
                "n_eff": float(effective_n(w)),
            },
        })
        write_json(paths["fold_metrics"], {"folds": fold_rows})
        write_json(paths["selected_population"], {
            "n_raw": int(len(o)),
            "n_selected": int(len(gated)),
            "selection_rule": "p_move >= tau_0.95",
        })
        described = CASCADE_PASS.get(args.pass_n, {})
        paths["summary"].write_text(
            f"# {described.get('role', 'pass_' + str(args.pass_n))}\n\n"
            f"{described.get('purpose', '')}\n\n"
            f"- rows: {len(o):,}\n"
            f"- weighted AUC: {rk['auc']:.4f}\n"
            f"- base rate: {rk['base_rate']:.4f}\n"
            f"- selected at tau_0.95: {len(gated):,} of {len(o):,}\n",
            encoding="utf-8",
        )
        rel = f"pass_{args.pass_n}/model_{model_id}/oof_predictions.parquet"
        fold_windows = [
            {"fold": i, "train_start": f["train"]["start"],
             "train_end": f["train"]["end"], "test_start": f["oof"]["start"],
             "test_end": f["oof"]["end"], "split": f["split"]}
            for i, f in enumerate(folds)
        ]
        pass_entry = {
            "pass": args.pass_n,
            "role": described.get("role", f"pass_{args.pass_n}"),
            "purpose": described.get("purpose", ""),
            "fold_windows": fold_windows,
            "models": [{
                "model_id": model_id,
                "model_type": args.model_type,
                "model_json": f"pass_{args.pass_n}/model_{model_id}/model.json",
                "oof_predictions": rel,
                "metrics": f"pass_{args.pass_n}/model_{model_id}/metrics.json",
                "fold_metrics": f"pass_{args.pass_n}/model_{model_id}/fold_metrics.json",
            }],
        }
        if inputs is not None:
            pass_entry["inputs"] = inputs
        merge_run_json(args.objective, run_ts, pass_entry, validation={
            "split": str(folds[0]["split"]),
            "n_folds": len(folds),
            "test_months": args.test_months,
            "min_train_months": args.min_train_months,
            "purge_sessions": 1,
            "fold_windows": fold_windows,
        })
        logger.info("wrote oof predictions -> %s", paths["oof_predictions"])
    if args.out_scores:
        cols = ["ts", "session", "side", "y_move", "reached", "t_move",
                "mfe_pts", "mae_pts", "price", "r_value", "weight", "p_move",
                "fold"] + [c for c in o.columns if c.startswith("tau_")]
        o[cols].to_parquet(args.out_scores, index=False, compression="zstd")
        logger.info("wrote scores -> %s", args.out_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
