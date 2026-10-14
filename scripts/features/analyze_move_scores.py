#!/usr/bin/env python3
"""SHAP top-N for the latest Stage 1 move-detector run.

Same method as ``analyze_context_scores.py``'s Stage 3 SHAP diagnostic: fit a
representative XGBoost classifier on the move matrix, hold out the most
recent 60 days, and take mean |SHAP| on a sample of that holdout. Stage 1's
feature selection is uncurated (see docs/reference/STAGE1_FEATURES.md), so
every non-leak numeric column in the matrix is in scope, not a curated
``CONTEXT`` subset.

Writes, for an objective run (xgboost only):
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_importance.csv
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_bar.png
  research/runs/<objective_id>/<run_ts>/pass_<n>/model_<model_id>/report/shap_beeswarm.png

Without --objective, the legacy files under data/evaluation/move_analysis/ are written.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/vtech")
from src.common.data_paths import data_root

LOG = logging.getLogger("move_analysis")

LABELS = Path(data_root()) / "sources" / "labels"
OUT = Path(data_root()) / "evaluation" / "move_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Same LEAK set as eval_move_model.py: outcome columns, identity columns, and
# r_value (a tree that splits on realized R rather than pre-decision state).
LEAK = {"y_move", "reached", "t_move", "mfe_pts", "mae_pts", "price",
        "ts", "session", "side", "weight", "r_value"}


def _matrix_files(matrix_dir: Path) -> list[Path]:
    if matrix_dir.is_file():
        return [matrix_dir]
    one = matrix_dir / "training_matrix.parquet"
    if one.is_file():
        return [one]
    files = sorted(matrix_dir.glob("move_*.parquet"))
    if not files:
        raise SystemExit(f"no training matrix under {matrix_dir}")
    return files


def shap_top(matrix_dir: Path, variant: str, top_n: int = 20,
             out_dir: Path | None = None) -> pd.DataFrame:
    import xgboost as xgb
    import matplotlib.pyplot as plt

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
    pos = float(tr["y_move"].mean())

    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=0, eval_metric="logloss",
                          scale_pos_weight=(1 - pos) / max(pos, 1e-9))
    m.fit(tr[feats].to_numpy("float32"), tr["y_move"].to_numpy("int8"))

    booster = m.get_booster()
    sample = te[feats].sample(min(20_000, len(te)), random_state=42)
    dm = xgb.DMatrix(sample.to_numpy("float32"), feature_names=feats)
    values = booster.predict(dm, pred_contribs=True)[:, :-1]
    mean_abs = np.abs(values).mean(axis=0)
    ranked = (pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))
    top = ranked.head(top_n).copy()
    top_idx = [feats.index(feature) for feature in top["feature"]]
    top_features = top["feature"].tolist()

    order = np.arange(len(top_features))[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(order, top["mean_abs_shap"].to_numpy(), color="#2f6f8f")
    ax.set_yticks(order, top_features)
    ax.set_xlabel("mean |SHAP| on XGBoost raw prediction scale")
    ax.set_title(f"Stage 1 SHAP importance — {variant}")
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
    ax.set_title(f"Stage 1 SHAP beeswarm — {variant}\ncolour: normalized feature value")
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
    ap.add_argument("--pass-n", type=int, default=1, dest="pass_n")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--matrix", default=None,
                    help="training matrix file or directory. Resolved from the "
                         "run when --objective, --run-ts and --model-id are set.")
    args = ap.parse_args()

    if args.objective or args.run_ts or args.model_id:
        if not (args.objective and args.run_ts and args.model_id):
            LOG.error("SHAP for a run needs --objective, --run-ts and --model-id")
            return 1
        from src.experiment.analysis import build_oof_analysis, write_shap_report
        from src.common.run_layout import model_dir, report_dir
        import json
        out = report_dir(args.objective, args.run_ts, args.pass_n, args.model_id)
        out.mkdir(parents=True, exist_ok=True)
        LOG.info("SHAP -> %s", out)
        meta_path = model_dir(args.objective, args.run_ts, args.pass_n, args.model_id) / "model.json"
        top = None
        if args.matrix is None and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            columns = list(meta.get("feature_columns") or [])
            target = str(meta.get("target") or "y_move")
            if columns:
                frame = build_oof_analysis(
                    args.objective, args.run_ts, args.pass_n, args.model_id)
                hyper = meta.get("hyperparameters") or {}
                top = write_shap_report(
                    frame, columns, target, out,
                    max_depth=int(hyper.get("max_depth", 6)),
                    n_estimators=int(hyper.get("n_estimators", 300)),
                    learning_rate=float(hyper.get("learning_rate", 0.05)),
                    seed=int(hyper.get("seed", 42)),
                    title=args.model_id,
                )
        if top is None:
            from src.common.run_layout import resolve_population_cache
            if args.matrix:
                mdir = Path(args.matrix)
            else:
                found = resolve_population_cache(args.objective, args.run_ts, args.pass_n)
                if found is None:
                    LOG.error("no feature columns and no population cache for SHAP")
                    return 1
                mdir = found
            top = shap_top(mdir, args.model_id, out_dir=out)
        LOG.info("\n%s", top.to_string(index=False))
        top.to_csv(out / "shap_importance.csv", index=False)
        LOG.info("wrote SHAP -> %s", out)
        return 0

    variant, mdir = "move_v4", LABELS / "move_matrix_v4"
    LOG.info("\n=== SHAP top-20 — %s ===", variant)
    top = shap_top(mdir, variant)
    LOG.info("\n%s", top.to_string(index=False))
    top.to_csv(OUT / f"shap_top20_{variant}.csv", index=False)

    LOG.info("wrote outputs -> %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
