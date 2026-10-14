#!/usr/bin/env python3
"""Join Layer 1 entry-state features onto the Layer 2 first-passage labels.

Produces the matrix a model actually trains on: one row per
``(decision bar, side, target, stop)`` carrying both the candidate descriptors
(``tgt_*`` / ``stp_*``) and the auction context at that bar (``es_*``).

**Bar subsampling is the point of `--bar-stride`.** Entry timing and barrier
response are sample-limited in different ways: adjacent decision bars describe
nearly the same auction (so they add little independent information about *when*
to enter), while the (target, stop) pairs at one bar describe genuinely different
parts of the path (so they do inform the barrier-response surface). Striding the
bars and keeping every pair spends the row budget where it buys information.

    python scripts/features/build_training_matrix.py --bar-stride 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.session_path import prepare_path, session_grid as _session_grid
from src.common.cme_holidays import is_trading_day
from src.common.data_paths import data_root, native_tf_dir, parquet_path1s_dir
from src.training.auction_levels import ProfileConfig, allocate_tf_delta, session_profile
from src.features.cash_session_profile import cash_session_slice
from src.features.flow_state import DIRECTIONAL as FLOW_DIRECTIONAL
from src.features.flow_state import FlowConfig, build_flow_state
from src.training.entry_state import DIRECTIONAL, build_entry_state
from src.training.first_passage import SHORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_training_matrix")

HIVE_COLS = ["bar_end_ts", "of_cvd", "of_delta", "of_ofi_cum", "cs_close"]


def _load_hive(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(native_tf_dir("15s")).rglob("*.parquet")):
        df = pd.read_parquet(p, columns=HIVE_COLS)
        df["bar_end_ts"] = pd.to_datetime(df["bar_end_ts"], utc=True)
        df = df[(df["bar_end_ts"] >= lo) & (df["bar_end_ts"] <= hi)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).set_index("bar_end_ts").sort_index()
    return out[~out.index.duplicated(keep="last")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--bar-stride", type=int, default=15,
                    help="keep every Nth decision bar (1 = all)")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--no-flow", action="store_true",
                    help="skip the fs_ order-flow block")
    args = ap.parse_args()

    root = Path(data_root())
    lab_dir = Path(args.labels) if args.labels else root / "sources" / "labels" / "firstpass"
    out_dir = Path(args.out) if args.out else root / "sources" / "labels" / "firstpass_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_files = sorted(lab_dir.glob("firstpass_*.parquet"))
    if args.start:
        lo = datetime.strptime(args.start, "%Y-%m-%d").date()
        lab_files = [f for f in lab_files
                     if datetime.strptime(f.stem[-10:], "%Y-%m-%d").date() >= lo]
    if args.end:
        hi = datetime.strptime(args.end, "%Y-%m-%d").date()
        lab_files = [f for f in lab_files
                     if datetime.strptime(f.stem[-10:], "%Y-%m-%d").date() <= hi]
    if not lab_files:
        logger.error("no label files under %s", lab_dir)
        return 1

    path_dir = Path(parquet_path1s_dir())
    all_paths = sorted(path_dir.glob("path_*.parquet"))
    path_by_date = {datetime.strptime(p.stem[-10:], "%Y-%m-%d").date(): p
                    for p in all_paths if is_trading_day(
                        datetime.strptime(p.stem[-10:], "%Y-%m-%d").date())}
    ordered = sorted(path_by_date)
    prev_of = {d: path_by_date[ordered[i - 1]] for i, d in enumerate(ordered) if i > 0}

    pcfg = ProfileConfig()
    span_lo = pd.read_parquet(path_by_date[datetime.strptime(
        lab_files[0].stem[-10:], "%Y-%m-%d").date()], columns=["close"]).index.min()
    span_hi = pd.read_parquet(path_by_date[datetime.strptime(
        lab_files[-1].stem[-10:], "%Y-%m-%d").date()], columns=["close"]).index.max()
    hive_all = _load_hive(span_lo, span_hi)
    logger.info("hive rows %s covering %s -> %s", f"{len(hive_all):,}",
                hive_all.index.min() if not hive_all.empty else "-",
                hive_all.index.max() if not hive_all.empty else "-")

    prints_all = pd.DataFrame()
    if not args.no_flow:
        pf = root / "sources" / "features" / "large_prints.parquet"
        if pf.exists():
            prints_all = pd.read_parquet(pf)
            prints_all["ts_event"] = pd.to_datetime(prints_all["ts_event"], utc=True)
            prints_all = prints_all.set_index("ts_event").sort_index()
            logger.info("large prints %s", f"{len(prints_all):,}")
        else:
            logger.warning("no large_prints.parquet; taxonomy will be zero")

    n_rows = n_sess = 0
    for k, lf in enumerate(lab_files):
        d = datetime.strptime(lf.stem[-10:], "%Y-%m-%d").date()
        pf = path_by_date.get(d)
        if pf is None:
            continue
        lab = pd.read_parquet(lf)
        lab["ts"] = pd.to_datetime(lab["ts"], utc=True)

        bars = np.sort(lab["ts"].unique())
        if args.bar_stride > 1:
            bars = bars[:: args.bar_stride]
            lab = lab[lab["ts"].isin(bars)]
        if lab.empty:
            continue

        day = pd.read_parquet(pf)
        grid = _session_grid(prepare_path(day))
        entries = grid.index.get_indexer(pd.DatetimeIndex(bars))
        keep = entries >= 0
        if not keep.any():
            continue
        entries, bars = entries[keep], pd.DatetimeIndex(bars)[keep]

        hive = hive_all.loc[(hive_all.index >= day.index.min())
                            & (hive_all.index <= day.index.max())] \
            if not hive_all.empty else pd.DataFrame()
        dlt = allocate_tf_delta(day, hive["of_delta"]) if not hive.empty else None
        prior_p = prior_c = None
        if d in prev_of:
            prev_day = pd.read_parquet(prev_of[d])
            prior_p = session_profile(prev_day, pcfg)
            prior_c = float(prev_day["close"].iloc[-1])

        es = build_entry_state(day, grid, entries,
                               r_value=float(lab["r_value"].iloc[0]),
                               prior_profile=prior_p, prior_close=prior_c,
                               hive=hive if not hive.empty else None,
                               delta=dlt, pcfg=pcfg)
        merged = lab.merge(es, on="ts", how="inner")
        if merged.empty:
            continue

        if not args.no_flow:
            cash = cash_session_slice(day)
            if not cash.empty:
                fb = cash.copy()
                fb["delta"] = (pd.Series(dlt, index=day.index).reindex(cash.index)
                               .fillna(0.0) if dlt is not None else 0.0)
                fb["ofi"] = fb["delta"]
                tp = prints_all.loc[(prints_all.index >= cash.index.min())
                                    & (prints_all.index <= cash.index.max())] \
                    if not prints_all.empty else None
                pre = day.loc[day.index < cash.index.min()] if len(day) else day.iloc[0:0]
                warm_1m = None
                if pre is not None and not pre.empty and "close" in pre.columns:
                    warm_1m = (pre["close"].resample("1min", label="right", closed="left")
                               .last().dropna())
                fs = build_flow_state(fb, tp, FlowConfig(),
                                      r_value=float(lab["r_value"].iloc[0]),
                                      warmup_close_1m=warm_1m)
                merged = merged.merge(fs.reset_index().rename(columns={"index": "ts"}),
                                      on="ts", how="left")

        # Entry state and flow are built in the absolute frame; mirror the signed
        # columns for shorts so a short is a long in a rotated frame.
        short = merged["side"].to_numpy() == SHORT
        if short.any():
            cols = [c for c in (*DIRECTIONAL, *FLOW_DIRECTIONAL) if c in merged.columns]
            merged.loc[short, cols] = -merged.loc[short, cols]
        merged["session"] = pd.Timestamp(d)
        merged.to_parquet(out_dir / f"matrix_{d.isoformat()}.parquet",
                          index=False, compression="zstd")
        n_rows += len(merged)
        n_sess += 1
        if n_sess % 20 == 0:
            pct = int(20 * (k + 1) / len(lab_files))
            logger.info("[%s%s] %d/%d | %s | %d rows", "#" * pct, "." * (20 - pct),
                        k + 1, len(lab_files), d, n_rows)

    logger.info("done: %d sessions, %d rows -> %s", n_sess, n_rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
