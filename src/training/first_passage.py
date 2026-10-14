"""Layer 2 labels — first-passage outcomes over a candidate exit grid.

For every decision bar this emits one row per ``(side, target level, stop level)``
triple, resolved honestly on the 1s high/low path with a session-close time
barrier. That surface is what lets a single model price *any* exit geometry at
inference time instead of being trained against one fixed barrier pair.

**Why not reuse** :func:`src.training.path_labels._first_touch`: it walks the
path once per geometry, costing ``O(horizon x entries)`` per barrier pair. That
is right for one geometry, but a 10x10 candidate grid over a 6.5-hour session
would walk ~23,400 seconds a hundred times per entry. Here the path is walked
**once per entry** to build cumulative extrema, which are monotone, so every
candidate level's first-touch time is a ``searchsorted`` — ``O(T + K log T)``
instead of ``O(T x K)``. Results are identical; only the cost differs.

Censoring at the session close is a first-class outcome, not a defect: the book
flattens at 15:00, so "neither barrier touched" is a real exit at the closing
price and the competing-risks head consumes it directly.

Entry is a **resting order**, not a fill at the decision close. The decision is
still taken at the decision bar — every feature is measured there and nothing
looks ahead — but the position only opens once price reaches the level supplied
by :func:`src.features.cash_session_profile.entry_levels`, and the order is
cancelled if that has not happened within ``max_fill_wait_seconds``. An unfilled
order emits no row: it is not a loss, it is a trade that never happened, and
scoring it as a flat outcome would dilute the conditional expectation the model
is fitted on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.training.first_passage")

# Event codes — identical to src.training.path_labels so consumers are unchanged.
CENSORED, TARGET, STOP = 0, 1, 2

LONG, SHORT = 1, -1


@dataclass(frozen=True)
class FirstPassageConfig:
    """Decision cadence, trading window and tie handling."""

    entry_start: str = "08:30"
    entry_end: str = "14:45"     # last entry; leaves runway before the flatten
    flatten: str = "15:00"
    decision_grid_seconds: int = 60
    max_fill_wait_seconds: int = 900   # a resting entry is cancelled after this
    min_dist_r: float = 0.10           # floor on target/stop distance from the fill
    tie_policy: str = "stop_first"
    timezone: str = "America/Chicago"

    def __post_init__(self) -> None:
        if self.tie_policy not in ("stop_first", "target_first"):
            raise ValueError("tie_policy must be 'stop_first' or 'target_first'")
        if self.decision_grid_seconds < 1:
            raise ValueError("decision_grid_seconds must be >= 1")
        if self.max_fill_wait_seconds < 1:
            raise ValueError("max_fill_wait_seconds must be >= 1")
        if self.min_dist_r < 0:
            raise ValueError("min_dist_r must be >= 0")


def _hhmm(s: str) -> dtime:
    h, m = (int(x) for x in s.split(":"))
    return dtime(h, m)


def touch_offsets(
    high: np.ndarray,
    low: np.ndarray,
    entry_i: int,
    end_i: int,
    levels_up: np.ndarray,
    levels_dn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First-touch offsets (seconds) for every level, plus the running extrema.

    ``high``/``low`` are a contiguous 1s grid. Offsets are 1-based seconds after
    ``entry_i``; a level never reached by ``end_i`` gets ``n_ahead + 1``, which
    sorts after every real touch and marks censoring.
    """
    fwd_hi = high[entry_i + 1: end_i + 1]
    fwd_lo = low[entry_i + 1: end_i + 1]
    n = len(fwd_hi)
    if n == 0:
        miss = np.full(len(levels_up), 1, dtype=np.int64)
        return miss, np.full(len(levels_dn), 1, dtype=np.int64), fwd_hi, fwd_lo

    cummax = np.maximum.accumulate(fwd_hi)      # non-decreasing
    cummin = np.minimum.accumulate(fwd_lo)      # non-increasing

    up = np.searchsorted(cummax, levels_up, side="left") + 1
    # -cummin is non-decreasing, so the same search finds `cummin <= level`.
    dn = np.searchsorted(-cummin, -levels_dn, side="left") + 1
    return up, dn, cummax, cummin


def _resolve_pairs(
    t_tgt: np.ndarray,
    t_stp: np.ndarray,
    tie_policy: str,
    n_ahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Event code and exit offset for each (target, stop) pair."""
    miss = n_ahead + 1
    if tie_policy == "stop_first":
        stop_wins = t_stp <= t_tgt
    else:
        stop_wins = t_stp < t_tgt
    event = np.where(stop_wins, STOP, TARGET).astype(np.int8)
    offset = np.minimum(t_tgt, t_stp)
    censored = offset >= miss
    event[censored] = CENSORED
    offset = np.where(censored, n_ahead, offset).astype(np.int32)
    return event, offset


def label_entry(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_i: int,
    end_i: int,
    targets: np.ndarray,
    stops: np.ndarray,
    side: int,
    cfg: FirstPassageConfig,
    *,
    entry_px: float | None = None,
) -> dict[str, np.ndarray]:
    """Outcomes for the full ``len(targets) x len(stops)`` grid at one entry.

    ``targets`` are favourable-direction levels and ``stops`` adverse-direction
    levels **for this side** — the caller swaps them between long and short.
    *entry_px* overrides the bar close, which is what a resting order that filled
    at a known price needs.
    """
    entry_px = float(close[entry_i]) if entry_px is None else float(entry_px)
    n_ahead = max(0, end_i - entry_i)
    if side == LONG:
        t_up, t_dn, cummax, cummin = touch_offsets(high, low, entry_i, end_i, targets, stops)
        t_tgt_1d, t_stp_1d = t_up, t_dn
    else:
        t_up, t_dn, cummax, cummin = touch_offsets(high, low, entry_i, end_i, stops, targets)
        t_tgt_1d, t_stp_1d = t_dn, t_up

    n_t, n_s = len(targets), len(stops)
    t_tgt = np.repeat(t_tgt_1d, n_s)
    t_stp = np.tile(t_stp_1d, n_t)
    event, offset = _resolve_pairs(t_tgt, t_stp, cfg.tie_policy, n_ahead)

    tgt_px = np.repeat(np.asarray(targets, dtype="float64"), n_s)
    stp_px = np.tile(np.asarray(stops, dtype="float64"), n_t)

    exit_px = np.where(event == TARGET, tgt_px,
                       np.where(event == STOP, stp_px, float(close[end_i])))
    pnl = side * (exit_px - entry_px)

    if n_ahead > 0:
        j = np.clip(offset - 1, 0, n_ahead - 1)
        mfe = side * ((cummax[j] - entry_px) if side == LONG else (entry_px - cummin[j]))
        mae = side * ((entry_px - cummin[j]) if side == LONG else (cummax[j] - entry_px))
    else:
        mfe = np.zeros(n_t * n_s, dtype="float64")
        mae = np.zeros(n_t * n_s, dtype="float64")

    return {"event": event, "t_event": offset, "pnl_pts": pnl,
            "mfe_pts": np.abs(mfe), "mae_pts": np.abs(mae),
            "target_level": tgt_px, "stop_level": stp_px,
            "target_idx": np.repeat(np.arange(n_t), n_s),
            "stop_idx": np.tile(np.arange(n_s), n_t)}


def entry_indices(index: pd.DatetimeIndex, cfg: FirstPassageConfig) -> np.ndarray:
    """Positions of decision bars inside the entry window, on the decision grid."""
    ct = index.tz_convert(cfg.timezone)
    in_win = (ct.time >= _hhmm(cfg.entry_start)) & (ct.time < _hhmm(cfg.entry_end))
    on_grid = (index.asi8 // 1_000_000_000) % cfg.decision_grid_seconds == 0
    return np.flatnonzero(in_win & on_grid)


def flatten_index(index: pd.DatetimeIndex, cfg: FirstPassageConfig) -> int:
    """Position of the flatten bar; the last grid row when 15:00 is absent."""
    ct = index.tz_convert(cfg.timezone)
    at_or_after = np.flatnonzero(ct.time >= _hhmm(cfg.flatten))
    return int(at_or_after[0]) if len(at_or_after) else len(index) - 1


def first_fill(high: np.ndarray, low: np.ndarray, entry_i: int, end_i: int,
               level: float, above: bool, max_wait: int) -> int:
    """Offset in seconds at which a resting order at *level* fills, else ``-1``.

    A buy order resting above the decision price fills on the first bar whose
    high reaches it; a sell order resting below fills on the first low. The scan
    stops at ``max_wait`` because the decision is a statement about the auction
    now, and an order that has not filled in that time is being offered a
    different market.
    """
    stop_i = min(end_i, entry_i + max_wait)
    if stop_i <= entry_i:
        return -1
    seg = high[entry_i + 1: stop_i + 1] if above else low[entry_i + 1: stop_i + 1]
    hit = np.flatnonzero(seg >= level) if above else np.flatnonzero(seg <= level)
    return int(hit[0]) + 1 if hit.size else -1


def _trade_frame(cand: pd.DataFrame, side: int) -> pd.DataFrame:
    """Mirror signed descriptors so a short is a long in a rotated frame."""
    out = cand.copy()
    if side == SHORT:
        for col in ("position_vs_value", "delta_at_level"):
            if col in out.columns:
                out[col] = -out[col]
    return out.drop(columns=[c for c in ("above", "bin") if c in out.columns])


def label_session(
    grid: pd.DataFrame,
    candidates_by_entry: dict[int, pd.DataFrame],
    cfg: FirstPassageConfig,
    *,
    r_value: float,
    entry_levels_by_entry: dict[int, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Long-form first-passage rows for one session.

    *candidates_by_entry* maps an entry position to the Layer-1 candidate frame
    for that bar. *entry_levels_by_entry* maps it to the ``(long, short)``
    resting-order prices; omit it and the trade fills at the decision close.
    Candidates are split into targets and stops against the **fill** price, and
    their distance descriptors are restated against it, so the geometry the model
    sees is the geometry the trade actually had.
    """
    high = grid["high"].to_numpy(dtype="float64")
    low = grid["low"].to_numpy(dtype="float64")
    close = grid["close"].to_numpy(dtype="float64")
    end_i = flatten_index(grid.index, cfg)

    rows: list[pd.DataFrame] = []
    for entry_i, cand in candidates_by_entry.items():
        if cand.empty or entry_i >= end_i:
            continue
        want = (entry_levels_by_entry or {}).get(entry_i)
        for side in (LONG, SHORT):
            if want is None:
                fill_i, fill_px, t_fill = entry_i, float(close[entry_i]), 0
            else:
                level = float(want[0] if side == LONG else want[1])
                off = first_fill(high, low, entry_i, end_i, level, side == LONG,
                                 cfg.max_fill_wait_seconds)
                if off < 0:
                    continue
                fill_i, fill_px, t_fill = entry_i + off, level, off
            if fill_i >= end_i:
                continue

            lvl = cand["level"].to_numpy(dtype="float64")
            # Candidate levels were laid out around the decision price; the fill
            # sits elsewhere, so anything now too close to it is not a geometry
            # that could be traded.
            far = np.abs(lvl - fill_px) >= cfg.min_dist_r * max(r_value, 1e-9)
            above, below = cand[(lvl > fill_px) & far], cand[(lvl < fill_px) & far]
            tgt_c, stp_c = (above, below) if side == LONG else (below, above)
            if tgt_c.empty or stp_c.empty:
                continue

            res = label_entry(high, low, close, fill_i, end_i,
                              tgt_c["level"].to_numpy(), stp_c["level"].to_numpy(),
                              side, cfg, entry_px=fill_px)
            df = pd.DataFrame(res)
            df["ts"] = grid.index[entry_i]
            df["fill_ts"] = grid.index[fill_i]
            df["t_fill"] = np.int32(t_fill)
            df["side"] = side
            df["decision_price"] = close[entry_i]
            df["entry_price"] = fill_px
            df["r_value"] = r_value
            tgt_f = _trade_frame(tgt_c.reset_index(drop=True), side).add_prefix("tgt_")
            stp_f = _trade_frame(stp_c.reset_index(drop=True), side).add_prefix("stp_")
            df = pd.concat([
                df.reset_index(drop=True),
                tgt_f.iloc[df["target_idx"].to_numpy()].reset_index(drop=True),
                stp_f.iloc[df["stop_idx"].to_numpy()].reset_index(drop=True),
            ], axis=1)
            for tag in ("tgt", "stp"):
                d = (df[f"{tag}_level"] - fill_px).abs()
                df[f"{tag}_dist_pts"] = d
                df[f"{tag}_dist_r"] = d / r_value if r_value > 0 else np.nan
            rows.append(df.drop(columns=["target_idx", "stop_idx"]))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["pnl_r"] = out["pnl_pts"] / r_value if r_value > 0 else np.nan
    return out


def unconditional_surface(df: pd.DataFrame, *, n_bins: int = 6) -> pd.DataFrame:
    """Base-rate first-passage surface by (target distance, stop distance) in R.

    The pre-model baseline: which exit geometries are near break-even before any
    conditioning. A model has to beat these cells, not beat zero.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["tgt_bin"] = pd.qcut(d["tgt_dist_r"], n_bins, duplicates="drop")
    d["stp_bin"] = pd.qcut(d["stp_dist_r"], n_bins, duplicates="drop")
    g = d.groupby(["tgt_bin", "stp_bin"], observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "p_target": g["event"].apply(lambda s: float((s == TARGET).mean())),
        "p_stop": g["event"].apply(lambda s: float((s == STOP).mean())),
        "p_censored": g["event"].apply(lambda s: float((s == CENSORED).mean())),
        "mean_pnl_pts": g["pnl_pts"].mean(),
        "median_hold_min": g["t_event"].median() / 60.0,
    }).reset_index()
