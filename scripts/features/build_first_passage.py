#!/usr/bin/env python3
"""Build the first-passage candidate-exit label set (Layers 0-2).

One pass over ``parquet_path1s`` in chronological order. Each session is labelled
using a composite volume/delta profile built from the **prior**
``--lookback-sessions`` sessions only, so a candidate level is knowable at the
open. Output is long-form: one row per (decision bar, side, target, stop).

    python scripts/features/build_first_passage.py --start 2025-05-01 --end 2026-06-30

``--surface`` additionally prints the unconditional first-passage surface, which
is the pre-model baseline: which exit geometries are near break-even before any
conditioning.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.session_path import prepare_path, session_grid as _session_grid
from src.common.cme_holidays import is_trading_day
from src.common.data_paths import data_root, native_tf_dir, parquet_path1s_dir
from src.features.cash_session_profile import (
    LevelConfig, bound_candidates, bound_interaction_matrices, cash_session_slice,
    entry_levels, interactions_at, levels_for_price, session_levels,
)
from src.training.auction_levels import (
    ProfileConfig, allocate_tf_delta, composite_profile, find_nodes, select_candidates,
    session_profile, union_nodes,
)
from src.training.first_passage import (
    FirstPassageConfig, entry_indices, label_session, unconditional_surface,
)
from src.training.path_labels import PathLabelConfig, session_r

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_first_passage")


def _session_date(path: Path) -> date:
    return datetime.strptime(path.stem.replace("path_", ""), "%Y-%m-%d").date()


def _load_tf_delta(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    """True aggressor delta from the 15s hive, indexed by bar-end timestamp.

    The CLV estimate available from OHLCV alone is uncorrelated with true delta
    at price (ADR-123), so sessions without hive coverage get a zero delta
    channel rather than a noisy one.
    """
    frames = []
    for p in sorted(Path(native_tf_dir("15s")).rglob("*.parquet")):
        df = pd.read_parquet(p, columns=["bar_end_ts", "of_delta"])
        df["bar_end_ts"] = pd.to_datetime(df["bar_end_ts"], utc=True)
        df = df[(df["bar_end_ts"] >= lo) & (df["bar_end_ts"] <= hi)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.Series(dtype="float64")
    out = pd.concat(frames, ignore_index=True).set_index("bar_end_ts")["of_delta"]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _candidates_for_session(
    day: pd.DataFrame,
    grid: pd.DataFrame,
    entries: np.ndarray,
    composite,
    prior_nodes: list[np.ndarray],
    r_value: float,
    pcfg: ProfileConfig,
    developing_stride: int,
    delta: np.ndarray | None = None,
) -> dict[int, pd.DataFrame]:
    """Layer 0/1 per decision bar.

    Candidates come from the prior-session composite **and** today's developing
    profile. History alone goes blind when price trends out of its range, which
    is exactly when a target matters most.
    """
    out: dict[int, pd.DataFrame] = {}
    src, nodes = composite, union_nodes(composite, cfg=pcfg)
    closes = grid["close"].to_numpy()
    for n, entry_i in enumerate(entries):
        if n % developing_stride == 0:
            pos = int(day.index.searchsorted(grid.index[entry_i]))
            dev = (session_profile(day.iloc[:pos], pcfg,
                                   delta=None if delta is None else delta[:pos])
                   if pos > 1 else None)
            if dev is not None:
                src = composite_profile([dev, composite], pcfg)
                nodes = union_nodes(composite, dev, cfg=pcfg)
        cand = select_candidates(src, prior_nodes, float(closes[entry_i]),
                                 r_value, pcfg, node_bins=nodes)
        if not cand.empty:
            out[int(entry_i)] = cand
    return out


def _candidates_from_bounds(
    grid: pd.DataFrame,
    entries: np.ndarray,
    level_hist: list,
    r_value: float,
    cfg: LevelConfig,
    max_per_side: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, tuple[float, float]]]:
    """Layer 0 from closed cash-session levels grouped into bounds.

    Bounds are rebuilt only when price leaves the set it was computed against,
    since the grouped structure is fixed for a session unless the lookback has
    to extend. *r_value* reaches only :func:`bound_candidates`, which decides how
    far away a level is worth aiming at; it plays no part in what counts as a
    level. Returns the candidate frames and the ``(long, short)`` resting entry
    prices per decision bar.
    """
    closes = grid["close"].to_numpy()
    bounds, _ = levels_for_price(level_hist, float(closes[entries[0]]), cfg)
    if not bounds:
        return {}, {}

    # Rebuilds are rare (price must leave the whole clustered range) but the
    # interaction matrices cost O(rows x bounds), so key them on the bound set.
    mat_cache: dict[tuple, tuple] = {}

    def _mats(bs):
        key = (len(bs), round(bs[0].lo, 3), round(bs[-1].hi, 3))
        if key not in mat_cache:
            mat_cache[key] = bound_interaction_matrices(closes, bs)
        return mat_cache[key]

    mats = _mats(bounds)
    out: dict[int, pd.DataFrame] = {}
    fills: dict[int, tuple[float, float]] = {}
    for entry_i in entries:
        px = float(closes[entry_i])
        if px > max(b.hi for b in bounds) or px < min(b.lo for b in bounds):
            bounds, _ = levels_for_price(level_hist, px, cfg)
            if not bounds:
                continue
            mats = _mats(bounds)
        cand = bound_candidates(bounds, px, r_value, max_per_side=max_per_side,
                               interactions=interactions_at(mats, bounds, int(entry_i)))
        if not cand.empty:
            out[int(entry_i)] = cand
            fills[int(entry_i)] = entry_levels(bounds, px, cfg)
    return out, fills


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=str, default=None, help="first session (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="last session (YYYY-MM-DD)")
    ap.add_argument("--lookback-sessions", type=int,
                    default=LevelConfig.lookback_sessions,
                    help="closed sessions whose volume-profile levels build the bounds")
    ap.add_argument("--r-lookback-sessions", type=int,
                    default=PathLabelConfig.vol_lookback_sessions,
                    help="sessions the R volatility yardstick is medianed over; "
                         "unrelated to the level lookback")
    ap.add_argument("--max-candidates", type=int, default=10, help="per side")
    ap.add_argument("--bin-size", type=float, default=1.0, help="profile bin, index points")
    ap.add_argument("--decision-grid", type=int, default=60, help="seconds between entries")
    ap.add_argument("--entry-start", type=str, default="08:30")
    ap.add_argument("--entry-end", type=str, default="14:45")
    ap.add_argument("--flatten", type=str, default="15:00")
    ap.add_argument("--developing-stride", type=int, default=15,
                    help="recompute the developing profile every N decision bars")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default <data>/sources/labels/firstpass)")
    ap.add_argument("--surface", action="store_true", help="print the unconditional surface")
    ap.add_argument("--candidates", choices=["profile", "bounds"], default="profile",
                    help="profile = rolling HVN/LVN nodes; bounds = clustered "
                         "cash-session levels")
    ap.add_argument("--sigma-pts", type=float, default=LevelConfig.sigma_pts,
                    help="KDE bandwidth in index points; sets level resolution")
    ap.add_argument("--dry-run", action="store_true", help="do not write parquet")
    args = ap.parse_args()

    pcfg = ProfileConfig(bin_size=args.bin_size,
                         lookback_sessions=args.lookback_sessions,
                         max_candidates=args.max_candidates)
    fcfg = FirstPassageConfig(entry_start=args.entry_start, entry_end=args.entry_end,
                              flatten=args.flatten,
                              decision_grid_seconds=args.decision_grid)
    rcfg = PathLabelConfig(entry_start=args.entry_start, entry_end=args.flatten,
                           vol_lookback_sessions=args.r_lookback_sessions)
    lcfg = LevelConfig(lookback_sessions=args.lookback_sessions,
                       sigma_pts=args.sigma_pts)

    files = sorted(Path(parquet_path1s_dir()).glob("path_*.parquet"))
    files = [f for f in files if is_trading_day(_session_date(f))]
    lo = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    hi = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    # Warm the lookback and the R history before the requested window opens. Holidays
    # are already dropped, so the slice yields a full lookback of real sessions.
    warm = args.lookback_sessions + rcfg.vol_lookback_sessions
    sel = [f for f in files if (lo is None or _session_date(f) >= lo)
           and (hi is None or _session_date(f) <= hi)]
    if lo is not None:
        sel = [f for f in files if _session_date(f) < lo][-warm:] + sel
    if not sel:
        logger.error("no sessions selected")
        return 1

    out_dir = Path(args.out) if args.out else Path(data_root()) / "sources" / "labels" / "firstpass"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    prof_hist: deque = deque(maxlen=args.lookback_sessions)
    node_hist: deque = deque(maxlen=args.lookback_sessions)
    level_hist: deque = deque(maxlen=lcfg.max_lookback_sessions)
    r_hist: deque = deque(maxlen=rcfg.vol_lookback_sessions)
    frames: list[pd.DataFrame] = []
    n_written = n_rows = n_no_delta = 0

    span_lo = pd.read_parquet(sel[0], columns=["close"]).index.min()
    span_hi = pd.read_parquet(sel[-1], columns=["close"]).index.max()
    tf_delta = _load_tf_delta(span_lo, span_hi)
    logger.info("true 15s delta coverage: %s rows %s -> %s", f"{len(tf_delta):,}",
                tf_delta.index.min() if len(tf_delta) else "-",
                tf_delta.index.max() if len(tf_delta) else "-")

    for k, f in enumerate(sel):
        d = _session_date(f)
        day = pd.read_parquet(f)
        dlt = allocate_tf_delta(day, tf_delta) if len(tf_delta) else None
        if dlt is None or not np.any(dlt):
            n_no_delta += 1
            dlt = None
        prof = session_profile(day, pcfg, delta=dlt)
        hvn, lvn = find_nodes(prof, pcfg)
        nodes = np.unique(np.concatenate([hvn, lvn])) if len(hvn) or len(lvn) else np.array([], np.int64)

        grid = _session_grid(prepare_path(day))
        r_today = session_r(grid, rcfg)
        r_prior = float(np.median(r_hist)) if len(r_hist) == r_hist.maxlen else float("nan")

        ready = len(prof_hist) == args.lookback_sessions and np.isfinite(r_prior)
        if args.candidates == "bounds":
            ready = len(level_hist) >= args.lookback_sessions and np.isfinite(r_prior)
        in_window = (lo is None or d >= lo)
        if ready and in_window:
            entries = entry_indices(grid.index, fcfg)
            fills = None
            if args.candidates == "bounds":
                cands, fills = _candidates_from_bounds(grid, entries, list(level_hist),
                                                       r_prior, lcfg, args.max_candidates)
            else:
                composite = composite_profile(list(prof_hist), pcfg)
                cands = _candidates_for_session(day, grid, entries, composite,
                                                list(node_hist), r_prior, pcfg,
                                                max(1, args.developing_stride), dlt)
            lab = label_session(grid, cands, fcfg, r_value=r_prior,
                                entry_levels_by_entry=fills)
            if not lab.empty:
                n_rows += len(lab)
                n_written += 1
                if args.surface:
                    frames.append(lab)
                if not args.dry_run:
                    lab.to_parquet(out_dir / f"firstpass_{d.isoformat()}.parquet",
                                   index=False, compression="zstd")
                if n_written % 10 == 0:
                    pct = int(20 * (k + 1) / len(sel))
                    logger.info("[%s%s] %d/%d sessions | %s | %d rows so far",
                                "#" * pct, "." * (20 - pct), k + 1, len(sel), d, n_rows)

        prof_hist.appendleft(prof)
        node_hist.appendleft(nodes)
        sl = session_levels(day, pd.Timestamp(d), lcfg)
        if sl is not None:
            level_hist.appendleft(sl)
        if np.isfinite(r_today):
            r_hist.append(r_today)

    logger.info("done: %d sessions labelled, %d rows (%d sessions had no TBBO delta)",
                n_written, n_rows, n_no_delta)

    if args.surface and frames:
        allrows = pd.concat(frames, ignore_index=True)
        print("\n=== Unconditional first-passage surface ===")
        print(unconditional_surface(allrows).to_string(index=False))
        print("\n=== Holding time (minutes) ===")
        hold = allrows["t_event"] / 60.0
        print(hold.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).to_string())
        print("\n=== Outcome mix ===")
        print(allrows["event"].value_counts(normalize=True)
              .rename({0: "censored", 1: "target", 2: "stop"}).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
