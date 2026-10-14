#!/usr/bin/env python3
"""Explain what the first-passage model actually did, and whether levels mattered.

Two questions, answered on the same fitted fold:

1. **Do prior-session volume-profile levels influence selection?** Gain
   importance can understate a feature used in few but decisive splits, so this
   compares the descriptor distribution of *chosen* candidates against the
   candidates rejected at the same bar. If levels were inert, the two
   distributions would coincide.
2. **What was the rationale for one trade?** Exact tree SHAP contributions
   (``pred_contribs``) for the selected row, plus the full candidate slate at
   that bar so the rejected alternatives are visible.

    python scripts/features/explain_trade.py --pick best
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data_paths import data_root
from src.training.first_passage import CENSORED, LONG, STOP, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("explain_trade")

MNQ_POINT_VALUE = 2.0
LEAK = {"event", "t_event", "pnl_pts", "pnl_r", "mfe_pts", "mae_pts",
        "target_level", "stop_level", "entry_price", "ts", "session", "side",
        "tgt_level", "stp_level", "tgt_bin", "stp_bin",
        "t_fill", "fill_ts", "decision_price"}
EVENT_NAME = {CENSORED: "flat at 15:00", TARGET: "TARGET hit", STOP: "STOP hit"}


def _load(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("matrix_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["session"] = pd.to_datetime(df["session"])
    return df.sort_values("ts").reset_index(drop=True)


def _sequence_one_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts")
    ts = df["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    # int32 seconds must widen before scaling to nanoseconds (see eval script).
    busy = df["t_event"].to_numpy().astype("int64")
    if "t_fill" in df.columns:
        busy = busy + df["t_fill"].to_numpy().astype("int64")
    hold = busy * 1_000_000_000
    keep = np.zeros(len(df), dtype=bool)
    free_at = np.iinfo("int64").min
    for i in range(len(df)):
        if ts[i] >= free_at:
            keep[i] = True
            free_at = ts[i] + hold[i]
    return df[keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--tau-q", type=float, default=0.95)
    ap.add_argument("--pick", type=str, default="best",
                    choices=["best", "worst", "median", "first"])
    ap.add_argument("--max-risk-pct", type=float, default=0.02)
    ap.add_argument("--account", type=float, default=5000.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import xgboost as xgb
    from xgboost import XGBRegressor

    d = Path(args.dir) if args.dir else Path(data_root()) / "sources" / "labels" / "firstpass_matrix"
    df = _load(d)
    df = df[df["stp_dist_pts"] <= (args.account * args.max_risk_pct) / MNQ_POINT_VALUE]

    feats = sorted(c for c in df.columns
                   if c not in LEAK and df[c].dtype.kind in "fiu" and c != "r_value")

    months = df["session"].dt.to_period("M").sort_values().unique()
    te_m = months[-args.test_months:]
    tr = df[~df["session"].dt.to_period("M").isin(te_m)]
    te = df[df["session"].dt.to_period("M").isin(te_m)]
    tr = tr[tr["session"] < tr["session"].max()]          # purge one session
    logger.info("train %s rows | test %s rows (%s..%s)", f"{len(tr):,}", f"{len(te):,}",
                te_m[0], te_m[-1])

    m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=args.seed,
                     n_jobs=4, tree_method="hist")
    m.fit(tr[feats].to_numpy("float32"), tr["pnl_r"].to_numpy("float32"))
    te = te.copy()
    te["pred"] = m.predict(te[feats].to_numpy("float32"))
    tau = float(np.quantile(m.predict(tr[feats].to_numpy("float32")), args.tau_q))

    best = te.loc[te.groupby("ts")["pred"].idxmax()]
    seq = _sequence_one_position(best[best["pred"] >= tau])
    if seq.empty:
        raise SystemExit("no trades at this threshold")
    logger.info("trades: %d | tau %.4f", len(seq), tau)

    # ── 1. did levels influence selection? ───────────────────────────────────
    chosen = set(zip(best["ts"], best["tgt_dist_r"], best["stp_dist_r"]))
    te["_chosen"] = [t in chosen for t in zip(te["ts"], te["tgt_dist_r"], te["stp_dist_r"])]
    level_cols = [c for c in te.columns
                  if c.startswith(("tgt_", "stp_")) and te[c].dtype.kind == "f"]
    rows = []
    for c in level_cols:
        a = te.loc[te["_chosen"], c].dropna()
        b = te.loc[~te["_chosen"], c].dropna()
        if len(a) < 30 or len(b) < 30 or b.std() == 0:
            continue
        rows.append({"descriptor": c, "chosen_mean": a.mean(), "rejected_mean": b.mean(),
                     "std_diff": (a.mean() - b.mean()) / b.std()})
    sel = pd.DataFrame(rows).sort_values("std_diff", key=abs, ascending=False)
    print("\n=== Do level descriptors influence selection? ===")
    print("(standardised difference between candidates the optimiser CHOSE and "
          "those it rejected at the same bars; 0 = inert)")
    print(sel.head(14).to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    # ── 2. one trade, in full ────────────────────────────────────────────────
    if args.pick == "best":
        row = seq.loc[seq["pnl_pts"].idxmax()]
    elif args.pick == "worst":
        row = seq.loc[seq["pnl_pts"].idxmin()]
    elif args.pick == "median":
        row = seq.iloc[[np.argsort(seq["pnl_pts"].to_numpy())[len(seq) // 2]]].iloc[0]
    else:
        row = seq.iloc[0]

    ct = row["ts"].tz_convert("America/Chicago")
    side = "LONG" if row["side"] == LONG else "SHORT"
    print(f"\n{'=' * 72}\n=== TRADE: {ct:%Y-%m-%d %H:%M} CT  {side}  ===\n{'=' * 72}")
    print(f"  entry price     {row['entry_price']:>12,.2f}")
    print(f"  target level    {row['target_level']:>12,.2f}   "
          f"({row['tgt_dist_pts']:.1f} pts = {row['tgt_dist_r']:.2f} R away)")
    print(f"  stop level      {row['stop_level']:>12,.2f}   "
          f"({row['stp_dist_pts']:.1f} pts = {row['stp_dist_r']:.2f} R away)")
    rr = row["tgt_dist_pts"] / max(row["stp_dist_pts"], 1e-9)
    print(f"  reward:risk     {rr:>12.2f}")
    print(f"  model E[pnl_r]  {row['pred']:>12.4f}   (fires above tau {tau:.4f})")
    size = (args.account * args.max_risk_pct) / (row["stp_dist_pts"] * MNQ_POINT_VALUE)
    print(f"  size @2% risk   {size:>12.2f} MNQ contracts")
    print(f"\n  OUTCOME         {EVENT_NAME[int(row['event'])]:>12s} after "
          f"{row['t_event'] / 60:.1f} min")
    print(f"  realised        {row['pnl_pts']:>12.2f} pts = "
          f"${row['pnl_pts'] * MNQ_POINT_VALUE * size:,.2f} at that size")
    print(f"  MFE / MAE       {row['mfe_pts']:.1f} / {row['mae_pts']:.1f} pts")

    print("\n  --- the level it targeted ---")
    for c in ("tgt_vol_at_level", "tgt_delta_at_level", "tgt_node_strength",
              "tgt_n_sessions_confluence_5", "tgt_n_sessions_confluence_10",
              "tgt_n_sessions_confluence_30", "tgt_path_volume_between",
              "tgt_room_beyond", "tgt_position_vs_value"):
        if c in row:
            print(f"    {c:34s} {row[c]:>10.4f}")

    print("\n  --- auction state at entry ---")
    for c in ("es_cvd_session", "es_ofi_cum", "es_delta_5m", "es_cvd_price_divergence",
              "es_vs_dev_poc", "es_vs_dev_vah", "es_vs_dev_val", "es_in_value_area",
              "es_vs_prior_poc", "es_open_vs_prior_va", "es_session_range_r",
              "es_min_since_open"):
        if c in row:
            print(f"    {c:34s} {row[c]:>10.4f}")

    # Exact tree SHAP for this row.
    dm = xgb.DMatrix(row[feats].to_numpy("float32").reshape(1, -1), feature_names=feats)
    contrib = m.get_booster().predict(dm, pred_contribs=True)[0]
    s = pd.Series(contrib[:-1], index=feats).sort_values(key=abs, ascending=False)
    print(f"\n  --- why (exact tree SHAP; base {contrib[-1]:+.4f}) ---")
    for name, v in s.head(12).items():
        bar = ("+" if v > 0 else "-") * min(28, int(abs(v) / max(abs(s).max(), 1e-9) * 28))
        print(f"    {name:34s} {v:+8.4f}  {bar}")

    print("\n  --- candidate slate at this bar (what it rejected) ---")
    slate = te[te["ts"] == row["ts"]].sort_values("pred", ascending=False)
    show = slate[["side", "tgt_dist_r", "stp_dist_r", "tgt_node_strength",
                  "tgt_n_sessions_confluence_30", "pred", "pnl_pts"]].head(10).copy()
    show["side"] = np.where(show["side"] == LONG, "long", "short")
    show["<-- taken"] = ["  <--" if i == 0 else "" for i in range(len(show))]
    print(show.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\n  {len(slate)} candidates scored at this bar; "
          f"argmax spread {slate['pred'].max() - slate['pred'].min():.4f} R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
