#!/usr/bin/env python3
"""Evaluate Stage 3: does the target selector rank levels by reachability?

Stage 1 answered "is a move coming". This asks the second question separately:
**given a confirmed move, which bound does price actually get to before the stop?**

The label is ``y_reach``, not P&L, so the model is free to prefer a near target it
is confident about over a distant one it is not. Ranking on predicted ``pnl_r`` —
what ADR-123 did — cannot express that preference, because distance is worth points
and the two considerations are summed before the model ever sees them.

Three things are reported:

1. **Ranking quality** — uniqueness-weighted AUC against ``y_reach``.
2. **The τ_reach frontier** — precision and expectancy as the gate tightens.
3. **The joint (τ_move, τ_reach) surface** — the cascade's actual operating
   choice. Stage 1's gate is already applied upstream at ``--tau-q``, but
   ``p_move`` rides along in the matrix, so both thresholds can be varied together
   here without rebuilding.

Expectancy is reported in R and is **deliberately pessimistic**: a row that did not
reach its target is charged the full stop distance, even though some of those rows
were censored at the horizon rather than stopped out. An optimistic reading of
censored rows is how a backtest flatters itself.

Usage::

    python scripts/features/eval_target_model.py --dir data/sources/labels/target_matrix
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

logger = logging.getLogger("vtech.scripts.eval_target_model")

TAU_QUANTILES = (0.50, 0.75, 0.90, 0.95)
MOVE_QUANTILES = (0.0, 0.25, 0.50, 0.75)
# Outcomes, absolute prices, and p_move. The last is the cascade's *other* stage:
# letting Stage 3 train on the score would collapse two independent gates into
# one. The es_/fs_/st_ context columns are features, not scores, and stay in.
# t_fill / decision_price describe the resting order's outcome (ADR-126) and are
# only known after the fact.
LEAK = {"y_reach", "t_reach", "t_stop", "t_fill", "price", "decision_price",
        "stop_px", "tgt_level", "ts", "session", "side", "weight", "p_move",
        "r_value"}


def _load(d: Path) -> pd.DataFrame:
    if d.is_file():
        files = [d]
    else:
        files = sorted(d.glob("target_*.parquet")) or sorted(d.glob("training_matrix*.parquet"))
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



def _auc(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC; ties split evenly rather than ordered at random."""
    ok = np.isfinite(pred) & np.isfinite(y)
    y, pred, w = y[ok], pred[ok], w[ok]
    pos_w = np.where(y == 1, w, 0.0)
    neg_w = np.where(y == 0, w, 0.0)
    W_pos, W_neg = pos_w.sum(), neg_w.sum()
    if W_pos <= 0 or W_neg <= 0:
        return float("nan")
    _, inv = np.unique(pred, return_inverse=True)
    neg_by = np.bincount(inv, weights=neg_w)
    pos_by = np.bincount(inv, weights=pos_w)
    neg_below = np.concatenate(([0.0], np.cumsum(neg_by)[:-1]))
    return float(np.sum(pos_by * (neg_below + 0.5 * neg_by)) / (W_pos * W_neg))


def _expectancy_r(g: pd.DataFrame) -> np.ndarray:
    """Per-row R outcome: target distance on a reach, full stop distance otherwise."""
    return np.where(g["y_reach"].to_numpy() == 1,
                    g["tgt_dist_r"].to_numpy(),
                    -g["stp_dist_r"].to_numpy())


def execution_prices(book: pd.DataFrame) -> tuple[str, str]:
    """Columns for persisted trade PnL. Writes ``_exit_px`` onto *book*.

    Entry is the fill (``price``, otherwise ``fill_px``). Exit is ``tgt_level``
    when ``y_reach == 1`` and ``stop_px`` otherwise, including a row censored at
    the horizon. ``decision_price`` is the decision bar, not the fill.
    """
    if "price" in book.columns:
        entry = "price"
    elif "fill_px" in book.columns:
        entry = "fill_px"
    else:
        raise KeyError("trade book has no fill price ('price' or 'fill_px')")
    reached = book["y_reach"].to_numpy() == 1
    book["_exit_px"] = np.where(
        reached,
        book["tgt_level"].to_numpy(dtype="float64"),
        book["stop_px"].to_numpy(dtype="float64"),
    )
    return entry, "_exit_px"


def _stats(g: pd.DataFrame, label: str, n_sessions: int) -> dict:
    if g.empty:
        return {"policy": label, "n": 0}
    w = g["weight"].to_numpy()
    hit = g["y_reach"].to_numpy(dtype="float64")
    exp_r = _expectancy_r(g)
    return {
        "policy": label,
        "n": len(g),
        "n_eff": round(effective_n(w), 1),
        "per_sess": round(len(g) / max(1, n_sessions), 2),
        "reach_rate": round(float(np.average(hit, weights=w)), 4),
        "mean_r": round(float(np.average(exp_r, weights=w)), 4),
        "median_tgt_r": round(float(g["tgt_dist_r"].median()), 3),
        "t_stat": round(weighted_tstat(exp_r, w), 3),
    }


def _best_per_bar(g: pd.DataFrame) -> pd.DataFrame:
    """One candidate per decision bar — the highest-``p_reach`` target.

    A single-contract account chooses one target at a bar, not every target that
    cleared the gate. Without this the "book" can hold several positions opened at
    the same instant, which is not a book at all.
    """
    if g.empty:
        return g
    return g.loc[g.groupby("ts")["p_reach"].idxmax()]


def _one_position(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep only trades a single-contract account could actually have held.

    Two details decide whether this is a real constraint. The sort must be
    **stable**, or tied timestamps order arbitrarily and the greedy walk is
    non-deterministic. And a trade must block for **at least one second**: a target
    reached inside the decision second has ``t_reach == 0`` and would otherwise
    free the book instantly, admitting an unbounded number of simultaneous trades.
    """
    if g.empty:
        return g
    g = g.sort_values("ts", kind="mergesort")
    t_reach = g["t_reach"].to_numpy()
    # Capital commits when the order is placed, not when it fills, so a resting
    # entry blocks the book for the wait as well as the trade.
    t_fill = (g["t_fill"].to_numpy() if "t_fill" in g.columns
              else np.zeros(len(g), dtype="int64"))
    busy = np.maximum(t_fill + np.where(t_reach >= 0, t_reach, horizon), 1)
    ts = g["ts"].values.astype("datetime64[ns]")
    free_at = np.datetime64("1970-01-01", "ns")
    keep = np.zeros(len(g), dtype=bool)
    for i in range(len(g)):
        if ts[i] >= free_at:
            keep[i] = True
            free_at = ts[i] + np.timedelta64(int(busy[i]), "s")
    return g[keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None,
                    help="explicit population file or directory. Omit with "
                         "--objective and --run-ts to join the model's inputs.")
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--min-train-months", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--out-scores", default=None,
                    help="optional extra parquet of the compact OOF")
    ap.add_argument("--no-shap", action="store_true",
                    help="skip the SHAP report written after the OOF file")
    ap.add_argument("--objective", default=None)
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--pass-n", type=int, default=2, dest="pass_n")
    ap.add_argument("--model-id", default=None,
                    help="unique id for this trained model. Omit to allocate one "
                         "from the model type, hyperparameters, feature view, and "
                         "target. Not the model type.")
    ap.add_argument("--model-type", default="xgboost")
    ap.add_argument("--context", choices=("curated", "all", "none"), default="curated",
                    help="feature view recorded in the model id. Must match the "
                         "matrix build. Does not drop columns at fit time.")
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
        from src.training.cascade_run_spec import target_feature_spec, xgb_hyperparameters
        run_ts = args.run_ts or new_run_ts()
        hyper = xgb_hyperparameters(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, seed=args.seed)
        feature_spec = target_feature_spec(context=args.context)
        model_id = validate_model_id(
            args.model_id or allocate_model_id(
                args.objective, run_ts, args.pass_n, args.model_type,
                hyper, feature_spec=feature_spec, target="y_reach"),
            args.model_type)
        df = load_run_frame(
            args.objective, run_ts, args.pass_n, model_id,
            explicit=Path(args.dir) if args.dir else None)
    else:
        df = _load(Path(args.dir))
    df["weight"] = 1.0 if args.no_weights else label_uniqueness(
        df, span_col="t_reach", horizon=args.horizon)
    logger.info("rows %s | sessions %d | n_eff %s | reach rate %.4f",
                f"{len(df):,}", df["session"].nunique(),
                f"{effective_n(df['weight'].to_numpy()):,.0f}", df["y_reach"].mean())

    feats = sorted(c for c in df.columns
                   if c not in LEAK and df[c].dtype.kind in "fiu")
    logger.info("features %d: %s", len(feats), ", ".join(feats))

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

    oof = []
    for fi, fold in enumerate(folds):
        tr, te = _fold_masks(df, fold)
        if tr.empty or te.empty:
            continue
        pos = float(tr["y_reach"].mean())
        m = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                          learning_rate=args.learning_rate, subsample=0.8,
                          colsample_bytree=0.8, random_state=args.seed,
                          n_jobs=0, eval_metric="logloss",
                          scale_pos_weight=(1 - pos) / max(pos, 1e-9))
        m.fit(tr[feats].to_numpy("float32"), tr["y_reach"].to_numpy("int8"),
              sample_weight=tr["weight"].to_numpy("float32"))
        t = te.copy()
        t["p_reach"] = m.predict_proba(te[feats].to_numpy("float32"))[:, 1]
        t["fold"] = fi
        p_tr = m.predict_proba(tr[feats].to_numpy("float32"))[:, 1]
        for q in TAU_QUANTILES:
            t[f"tau_{q}"] = float(np.quantile(p_tr, q))
        for q in MOVE_QUANTILES:
            t[f"mv_{q}"] = float(np.quantile(tr["p_move"], q)) if q > 0 else -np.inf
        oof.append(t)
        logger.info("fold %d | train %s..%s (%s rows) | test %s..%s (%s rows) | pos %.3f",
                    fi, fold["train"]["start"], fold["train"]["end"], f"{len(tr):,}",
                    fold["oof"]["start"], fold["oof"]["end"], f"{len(te):,}", pos)

    if not oof:
        logger.error("no folds produced predictions")
        return 1
    o = pd.concat(oof, ignore_index=True)
    ns = o["session"].nunique()

    print("\n=== Ranking quality (uniqueness-weighted) ===")
    print(f"  rows            {len(o):,}")
    print(f"  n_eff           {effective_n(o['weight'].to_numpy()):,.0f}"
          f"   ({effective_n(o['weight'].to_numpy())/len(o):.4f} of raw)")
    print(f"  reach rate      {np.average(o['y_reach'], weights=o['weight']):.4f}")
    print(f"  weighted AUC    {_auc(o['y_reach'].to_numpy(), o['p_reach'].to_numpy(), o['weight'].to_numpy()):.4f}")

    print("\n=== tau_reach frontier (tau from TRAIN quantiles) ===")
    rows = [_stats(o, "all rows", ns)]
    for q in TAU_QUANTILES:
        rows.append(_stats(o[o["p_reach"] >= o[f"tau_{q}"]], f"reach q={q}", ns))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Nearest-bound baseline (rank 0, no model) ===")
    print(pd.DataFrame([_stats(o[o["tgt_rank"] == 0], "tgt_rank=0", ns)]).to_string(index=False))

    print("\n=== Joint (tau_move, tau_reach) surface — mean R per trade ===")
    grid = []
    for mq in MOVE_QUANTILES:
        for q in TAU_QUANTILES:
            g = o[(o["p_move"] >= o[f"mv_{mq}"]) & (o["p_reach"] >= o[f"tau_{q}"])]
            if g.empty:
                continue
            s = _stats(g, f"move>={mq} reach>={q}", ns)
            grid.append(s)
    gdf = pd.DataFrame(grid)
    print(gdf.to_string(index=False))

    print("\n=== Best cell, after argmax-per-bar + the 1-position book ===")
    book = None
    book_rule = None
    cost_rows: list[dict] = []
    gross_points = None
    if not gdf.empty:
        best = gdf.sort_values("t_stat", ascending=False).iloc[0]
        mq, q = best["policy"].split()[0].split(">=")[1], best["policy"].split()[1].split(">=")[1]
        g = o[(o["p_move"] >= o[f"mv_{float(mq)}"]) & (o["p_reach"] >= o[f"tau_{float(q)}"])]
        bar = _best_per_bar(g)
        bk = _one_position(bar, args.horizon).copy()
        # The weights carried in from the full matrix measure overlap against every
        # candidate that *could* have been taken. The book has already removed the
        # overlap by construction, so reusing them would discount it twice; recompute
        # against the trades actually held.
        bk["weight"] = label_uniqueness(bk, span_col="t_reach", horizon=args.horizon)
        print(f"  cell           {best['policy']}")
        print(pd.DataFrame([_stats(bar, "argmax/bar", ns),
                            _stats(bk, "+ book (reweighted)", ns)]).to_string(index=False))
        print(f"  trades/session {len(bk)/max(1, ns):.2f}")
        print(f"  long/short     {int((bk['side'] == 1).sum())}/{int((bk['side'] == -1).sum())}")
        print(f"  distinct ts    {bk['ts'].nunique()} of {len(bk)} rows (must be equal)")

        exp_r = _expectancy_r(bk)
        rv = bk["r_value"].to_numpy()
        gross = exp_r * rv
        book = bk
        book_rule = str(best["policy"]) + "; argmax per bar; one open position"
        gross_points = float(np.nansum(gross))
        print(f"\n  === Cost sensitivity (1 MNQ, $2/pt) ===")
        print(f"  median target  {np.median(bk['tgt_dist_r'] * rv):.2f} pts | "
              f"median stop {np.median(bk['stp_dist_r'] * rv):.2f} pts | "
              f"R:R {np.median(bk['tgt_dist_r']) / max(np.median(bk['stp_dist_r']), 1e-9):.2f}:1")
        for slip, comm in ((0, 0.0), (1, 1.0), (2, 1.0)):
            cost = slip * 0.25 * 2 + comm / 2.0
            net = gross - cost
            t = net.mean() / max(net.std(), 1e-9) * np.sqrt(len(net))
            cost_rows.append({
                "slip_ticks_per_side": slip,
                "commission_rt_dollars": comm,
                "cost_points": float(cost),
                "net_points": float(net.sum()),
                "net_dollars_at_2usd_per_point": float(net.sum() * 2),
                "net_points_per_trade": float(net.mean()),
                "t": float(t),
                "assumptions": "1 MNQ, $2 per point",
            })
            print(f"  slip {slip} tick/side + ${comm:.2f} RT -> {cost:.2f} pts/trade | "
                  f"net {net.sum():+,.0f} pts (${net.sum() * 2:+,.0f}) | "
                  f"{net.mean():+.3f} pts/trade | t={t:.2f}")

    print("\n=== Per-fold at reach q=0.90 ===")
    rows = []
    for fi, g in o[o["p_reach"] >= o["tau_0.9"]].groupby("fold"):
        s = _stats(g, f"fold {fi}", g["session"].nunique())
        s["months"] = f"{g['session'].min():%Y-%m}..{g['session'].max():%Y-%m}"
        rows.append(s)
    fold_rows = rows
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
            feature_columns=list(feats),
            inputs=inputs,
            target="y_reach",
        )
        if not args.no_shap:
            write_shap_report(
                o, feats, "y_reach", paths["report_dir"],
                max_depth=args.max_depth, n_estimators=args.n_estimators,
                learning_rate=args.learning_rate, seed=args.seed,
                title=f"pass {args.pass_n} {model_id}",
            )
        reach_rate = float(np.average(o["y_reach"], weights=o["weight"]))
        auc = float(_auc(o["y_reach"].to_numpy(), o["p_reach"].to_numpy(), o["weight"].to_numpy()))
        write_json(paths["metrics"], {
            "n_samples": int(len(o)),
            "metrics": {
                "reach_rate": reach_rate,
                "weighted_auc": auc,
                "n_eff": float(effective_n(o["weight"].to_numpy())),
            },
        })
        write_json(paths["fold_metrics"], {"folds": fold_rows})
        if book is not None:
            write_json(paths["selected_population"], {
                "n_raw": int(len(o)),
                "n_selected": int(len(book)),
                "selection_rule": book_rule,
            })
            # ONE economic evaluator for the whole research workflow. This
            # script decides WHICH trades happen and at what prices (the book
            # above); src/evaluation/economic.py decides what they are worth --
            # execution pricing, order semantics, multiplier, commission, PnL,
            # the ledger and fixed-fill cost sensitivity. The previous inline
            # arithmetic here (cost = slip*0.25*2 + comm/2, net at $2/pt) was a
            # second, independent economic truth.
            from src.evaluation.economic import (
                EconomicAssumptions, assumptions_record, trades_from_book,
                write_research_economics,
            )
            bk = book.copy()
            bk["_reached"] = bk["y_reach"].to_numpy() == 1
            entry_col, exit_col = execution_prices(bk)
            record = assumptions_record(objective_doc)
            multiplier = float(record.get("contract_multiplier") or 2.0)
            commission = float(record.get("commission_per_trade") or 0.0)
            base = EconomicAssumptions(
                contract_multiplier=multiplier,
                commission_per_trade=commission,
                economic_contract_unit=record.get("economic_contract_unit"),
                underlying=record.get("underlying"),
                fill_model="path_first_touch on the 1s tape; resolved fills",
                label=str(record.get("label") or "objective"),
            )
            # Fixed-fill scenarios only: commission varies, the fills do not.
            scenarios = [base] + [
                EconomicAssumptions(
                    contract_multiplier=multiplier, commission_per_trade=c,
                    economic_contract_unit=record.get("economic_contract_unit"),
                    underlying=record.get("underlying"),
                    fill_model=base.fill_model, label=f"commission_{c:g}")
                for c in (0.0, 1.0, 2.0) if c != commission
            ]
            econ = write_research_economics(
                paths["model_dir"],
                trades=trades_from_book(
                    bk, entry_price_col=entry_col, exit_price_col=exit_col,
                    side_col="side", reached_col="_reached",
                    stop_price_col="stop_px", target_price_col="tgt_level",
                ),
                assumptions=base, scenarios=scenarios,
            )
            logger.info("economics: %d trades, net $%.2f, commission $%.2f (%s)",
                        econ["n_trades"], econ["net_dollars"],
                        econ["total_commission"], base.label)
        described = CASCADE_PASS.get(args.pass_n, {})
        paths["summary"].write_text(
            f"# {described.get('role', 'pass_' + str(args.pass_n))}\n\n"
            f"{described.get('purpose', '')}\n\n"
            f"- rows: {len(o):,}\n"
            f"- weighted AUC: {auc:.4f}\n"
            f"- reach rate: {reach_rate:.4f}\n"
            f"- trades: {0 if book is None else len(book):,}\n",
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
        o.to_parquet(args.out_scores, index=False, compression="zstd")
        logger.info("wrote scores -> %s", args.out_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
