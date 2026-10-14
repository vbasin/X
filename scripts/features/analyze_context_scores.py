#!/usr/bin/env python3
"""Independence guard + SHAP top-N for the winning context variant.

Writes, for an objective run (xgboost only):
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_importance.csv
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_bar.png
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_beeswarm.png

Without --objective, the legacy files under data/evaluation/context_analysis/ are written:
  data/evaluation/context_analysis/corr_p_move_p_reach.txt
  data/evaluation/context_analysis/shap_top20_context.csv
  data/evaluation/context_analysis/shap_top20_context_rest.csv
  data/evaluation/context_analysis/shap_top20_v3.csv
  data/evaluation/context_analysis/shap_top20_v4.csv
    data/evaluation/context_analysis/shap_bar_context.png
    data/evaluation/context_analysis/shap_beeswarm_context.png
    data/evaluation/context_analysis/shap_bar_v3.png
    data/evaluation/context_analysis/shap_beeswarm_v3.png
    data/evaluation/context_analysis/shap_bar_v4.png
    data/evaluation/context_analysis/shap_beeswarm_v4.png
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/vtech")
from src.common.data_paths import data_root

LOG = logging.getLogger("context_analysis")

LABELS = Path(data_root()) / "sources" / "labels"
OUT = Path(data_root()) / "evaluation" / "context_analysis"
OUT.mkdir(parents=True, exist_ok=True)

LEAK = {"y_reach", "t_reach", "t_stop", "t_fill", "price", "decision_price",
        "stop_px", "tgt_level", "ts", "session", "side", "weight", "p_move",
        "r_value", "fold", "p_reach", "tau_0.5", "tau_0.75", "tau_0.9", "tau_0.95"}


def corr_check(scores_path: Path) -> dict:
    df = pd.read_parquet(scores_path, columns=["p_move", "p_reach", "y_reach", "session"])
    p, r, y = df["p_move"].to_numpy(), df["p_reach"].to_numpy(), df["y_reach"].to_numpy()
    ok = np.isfinite(p) & np.isfinite(r)
    p, r, y = p[ok], r[ok], y[ok]
    pear = float(np.corrcoef(p, r)[0, 1])
    spear = float(pd.Series(p).corr(pd.Series(r), method="spearman"))
    # Conditional correlations at the operating slice.
    hi = (r >= np.quantile(r, 0.75))
    pear_hi = float(np.corrcoef(p[hi], r[hi])[0, 1]) if hi.sum() > 2 else float("nan")
    return {"file": scores_path.name, "rows": int(len(p)),
            "pearson": round(pear, 4), "spearman": round(spear, 4),
            "pearson_top25pct_reach": round(pear_hi, 4),
            "reach_rate": round(float(y.mean()), 4)}


def _matrix_files(matrix_dir: Path) -> list[Path]:
    if matrix_dir.is_file():
        return [matrix_dir]
    one = matrix_dir / "training_matrix.parquet"
    if one.is_file():
        return [one]
    files = sorted(matrix_dir.glob("target_*.parquet"))
    if not files:
        raise SystemExit(f"no training matrix under {matrix_dir}")
    return files


def shap_top(scores_path: Path, matrix_dir: Path, variant: str,
             top_n: int = 20, out_dir: Path | None = None) -> pd.DataFrame:
    import xgboost as xgb
    import matplotlib.pyplot as plt

    del scores_path  # the matrix already carries the label column this fit uses
    files = _matrix_files(matrix_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["session"] = pd.to_datetime(df["session"])
    df = df.sort_values("ts").reset_index(drop=True)

    feats = sorted(c for c in df.columns
                   if c not in LEAK and df[c].dtype.kind in "fiu")
    LOG.info("SHAP: %d features on %d rows", len(feats), len(df))

    tr = df[df["session"] < df["session"].max() - pd.Timedelta(days=60)]
    te = df[df["session"] >= df["session"].max() - pd.Timedelta(days=60)]
    pos = float(tr["y_reach"].mean())

    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=0, eval_metric="logloss",
                          scale_pos_weight=(1 - pos) / max(pos, 1e-9))
    m.fit(tr[feats].to_numpy("float32"), tr["y_reach"].to_numpy("int8"))

    booster = m.get_booster()
    sample = te[feats].sample(min(20_000, len(te)), random_state=42)
    dm = xgb.DMatrix(sample.to_numpy("float32"),
                     feature_names=feats)
    values = booster.predict(dm, pred_contribs=True)[:, :-1]
    mean_abs = np.abs(values).mean(axis=0)
    ranked = (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))
    top = ranked.head(top_n).copy()
    top_idx = [feats.index(feature) for feature in top["feature"]]
    top_values = values[:, top_idx]
    top_features = top["feature"].tolist()

    order = np.arange(len(top_features))[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(order, top["mean_abs_shap"].to_numpy(), color="#2f6f8f")
    ax.set_yticks(order, top_features)
    ax.set_xlabel("mean |SHAP| on XGBoost raw prediction scale")
    ax.set_title(f"Stage 3 SHAP importance — {variant}")
    fig.tight_layout()
    bar_name = "shap_bar.png" if out_dir is not None else f"shap_bar_{variant}.png"
    fig.savefig((out_dir or OUT) / bar_name, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(42)
    for row, feature_idx in enumerate(top_idx):
        y = np.full(len(values), row, dtype="float64")
        y += rng.uniform(-0.22, 0.22, len(values))
        colour = sample.iloc[:, feature_idx].to_numpy(dtype="float64")
        scale = np.nanpercentile(np.abs(colour), 99)
        colour = np.clip(colour / max(scale, 1e-9), -1.0, 1.0)
        ax.scatter(values[:, feature_idx], y, c=colour, cmap="coolwarm",
                   vmin=-1.0, vmax=1.0, s=3, alpha=0.18, linewidths=0,
                   rasterized=True)
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_yticks(np.arange(len(top_features)), top_features)
    ax.set_xlabel("SHAP value on XGBoost raw prediction scale")
    ax.set_title(f"Stage 3 SHAP beeswarm — {variant}\ncolour: normalized feature value")
    fig.tight_layout()
    bee_name = "shap_beeswarm.png" if out_dir is not None else f"shap_beeswarm_{variant}.png"
    fig.savefig((out_dir or OUT) / bee_name, dpi=160)
    plt.close(fig)
    return top


def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default=None)
    ap.add_argument("--run-ts", default=None)
    ap.add_argument("--pass-n", type=int, default=2, dest="pass_n")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--matrix", default=None)
    args = ap.parse_args()

    if args.objective or args.run_ts or args.model_id:
        if not (args.objective and args.run_ts and args.model_id):
            LOG.error("SHAP for a run needs --objective, --run-ts and --model-id")
            return 1
        import json
        from src.experiment.analysis import build_oof_analysis, write_shap_report
        from src.common.run_layout import (
            model_dir, report_dir, resolve_population_cache, training_matrix_path,
        )
        out = report_dir(args.objective, args.run_ts, args.pass_n, args.model_id)
        out.mkdir(parents=True, exist_ok=True)
        LOG.info("SHAP -> %s", out)
        meta_path = model_dir(args.objective, args.run_ts, args.pass_n, args.model_id) / "model.json"
        top = None
        if args.matrix is None and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            columns = list(meta.get("feature_columns") or [])
            if columns:
                frame = build_oof_analysis(
                    args.objective, args.run_ts, args.pass_n, args.model_id)
                hyper = meta.get("hyperparameters") or {}
                top = write_shap_report(
                    frame, columns, str(meta.get("target") or "y_reach"), out,
                    max_depth=int(hyper.get("max_depth", 5)),
                    n_estimators=int(hyper.get("n_estimators", 300)),
                    learning_rate=float(hyper.get("learning_rate", 0.05)),
                    seed=int(hyper.get("seed", 42)),
                    title=args.model_id,
                )
        if top is None:
            if args.matrix:
                mdir = Path(args.matrix)
            else:
                legacy = training_matrix_path(
                    args.objective, args.run_ts, args.pass_n, args.model_id)
                found = legacy if legacy.is_file() else resolve_population_cache(
                    args.objective, args.run_ts, args.pass_n)
                if found is None:
                    LOG.error("no feature columns and no population cache for SHAP")
                    return 1
                mdir = found
            top = shap_top(mdir, mdir, args.model_id, out_dir=out)
        LOG.info("\n%s", top.to_string(index=False))
        top.to_csv(out / "shap_importance.csv", index=False)
        LOG.info("wrote SHAP -> %s", out)
        return 0

    corr_rows = [corr_check(LABELS / "target_scores_context.parquet"),
                 corr_check(LABELS / "target_scores_context_rest.parquet"),
                 corr_check(LABELS / "target_scores.parquet"),
                 corr_check(LABELS / "target_scores_v3.parquet"),
                 corr_check(LABELS / "target_scores_v4.parquet")]
    corr_df = pd.DataFrame(corr_rows)
    LOG.info("\n=== corr(p_move, p_reach) — independence guard ===\n%s",
             corr_df.to_string(index=False))
    (OUT / "corr_p_move_p_reach.txt").write_text(corr_df.to_string(index=False))

    for variant, mdir in (("context", LABELS / "target_matrix_context"),
                          ("context_rest", LABELS / "target_matrix_context_rest"),
                          ("v3", LABELS / "target_matrix_v3"),
                          ("v4", LABELS / "target_matrix_v4")):
        LOG.info("\n=== SHAP top-20 — %s ===", variant)
        top = shap_top(LABELS / f"target_scores_{variant}.parquet", mdir, variant)
        LOG.info("\n%s", top.to_string(index=False))
        top.to_csv(OUT / f"shap_top20_{variant}.csv", index=False)

    LOG.info("wrote outputs -> %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
