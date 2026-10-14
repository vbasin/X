"""Confidence-gate / calibration sweep on ONE model's OOF, restricted to high-win regimes.

Answers: as the confidence gate tightens, how high does per-direction precision climb, and how
many signals survive? Reports (a) a fine tau sweep with precision + surviving signal count +
flat->directional FPs, (b) the tau where each direction first crosses the 0.85 / 0.90 floors,
and (c) a reliability table (predicted-confidence bucket vs empirical precision) to show whether
the raw xgboost softmax is over/under-confident (i.e. whether calibration would move the knob).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).astype("int64").to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--keep", nargs="+", type=int, required=True)
    args = ap.parse_args()

    oof = pd.read_parquet(args.oof)
    reg = pd.read_parquet(args.regimes)
    rmap = dict(zip(_ns(reg["ts"]), reg["regime"].to_numpy()))
    oof["_rk"] = [rmap.get(k, -1) for k in _ns(oof["ts"])]
    df = oof[oof["_rk"].isin(set(args.keep))].reset_index(drop=True)
    print(f"regime-filtered OOF rows: {len(df):,}  (regimes {sorted(args.keep)})\n")

    taus = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    print("  tau |    UP: n   prec  |  DOWN: n   prec  |  flat->dir FP")
    print("  " + "-" * 58)
    cross = {"up": {}, "down": {}}
    for t in taus:
        g = df[df["confidence"] >= t]
        line = f"  {t:.2f} |"
        for d in ("up", "down"):
            sig = g[g["direction"] == d]
            n = len(sig)
            prec = float((sig["direction"] == sig["direction_true"]).mean()) if n else float("nan")
            line += f"  {n:5d} {prec:5.3f} |" if d == "up" else f"  {n:5d} {prec:5.3f} |"
            for floor in (0.85, 0.90):
                if n >= 20 and prec >= floor and floor not in cross[d]:
                    cross[d][floor] = (t, n, prec)
        flat_fp = int(((g["direction_true"] == "flat") & (g["direction"].isin(["up", "down"]))).sum())
        line += f"   {flat_fp}"
        print(line)

    print("\n== precision-floor crossings (>=20 signals) ==")
    for d in ("up", "down"):
        for floor in (0.85, 0.90):
            c = cross[d].get(floor)
            print(f"  {d:<4} >= {floor}: " + (f"tau={c[0]:.2f}  n={c[1]}  prec={c[2]:.3f}"
                  if c else "NEVER reached with >=20 signals"))

    # reliability: is raw softmax confidence calibrated for directional preds?
    print("\n== reliability (directional preds): confidence bucket vs empirical precision ==")
    dd = df[df["direction"].isin(["up", "down"])].copy()
    dd["correct"] = (dd["direction"] == dd["direction_true"]).astype(int)
    bins = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    dd["bkt"] = pd.cut(dd["confidence"], bins, right=False)
    rel = dd.groupby("bkt", observed=True).agg(n=("correct", "size"), emp_prec=("correct", "mean"),
                                               mean_conf=("confidence", "mean"))
    for b, r in rel.iterrows():
        gap = r["emp_prec"] - r["mean_conf"]
        tag = "over-confident" if gap < -0.03 else "under-confident" if gap > 0.03 else "~calibrated"
        print(f"  conf {str(b):<14} n={int(r['n']):6d}  mean_conf={r['mean_conf']:.3f}  "
              f"emp_prec={r['emp_prec']:.3f}  ({tag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
