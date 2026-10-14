"""Prior-session volume-profile structure features (``pv_`` prefix, ADR-122).

Distinct from :mod:`src.features.volume_profile`, which computes a *rolling
intra-session* profile (``vp_``) plus five single-day-back distances
(``vp_yday_*``, percentage-normalised). This module adds the structural market
context that engine has never carried:

* ATR-normalised signed distances (scale-free across the 2011→2026 price range)
* ordinal zone position relative to the prior value area
* causal interaction history — POC touches, VAL/VAH rejections, VA acceptance
* a multi-day **naked (untested) POC** stack with age in sessions
* open type, gap, and prior/developing value-area overlap
* profile shape (D / P / b / double-distribution) and VA width

Every column at time ``t`` uses only information available at or before ``t``:
prior-session levels are constant within the session, the naked-POC stack is
resolved from sessions strictly before today, and the developing value area is
accumulated incrementally at 1-minute cadence then forward-filled to the grid.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.common.bar_windows import get_timestep

CT = ZoneInfo("America/Chicago")
TICK = 0.25

# pv_shape codes
SHAPE_D = 0.0        # normal / balanced — single fat centre
SHAPE_P = 1.0        # P-shape — volume stacked high, thin tail below
SHAPE_b = 2.0        # b-shape — volume stacked low, thin tail above
SHAPE_DOUBLE = 3.0   # double distribution — two separated modes

_COLUMNS = (
    "pv_dist_poc", "pv_dist_vah", "pv_dist_val",
    "pv_dist_phigh", "pv_dist_plow", "pv_dist_pclose",
    "pv_zone",
    "pv_bars_since_poc_touch", "pv_poc_test_count_today",
    "pv_val_rejection", "pv_vah_rejection", "pv_va_acceptance",
    "pv_naked_poc_above_dist", "pv_naked_poc_below_dist",
    "pv_naked_poc_above_age", "pv_naked_poc_below_age",
    "pv_open_vs_prior_va", "pv_gap_atr", "pv_va_overlap_pct",
    "pv_shape", "pv_va_width_atr",
)


# ── profile primitives ───────────────────────────────────────────────────

def _histogram(bars: pd.DataFrame, *, bin_pts: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Volume-at-price histogram, each bar's volume spread across its range."""
    if bars is None or bars.empty:
        return None
    lo = float(bars["low"].min())
    hi = float(bars["high"].max())
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    n_bins = max(2, min(600, int(round((hi - lo) / bin_pts))))
    edges = np.linspace(lo, hi, n_bins + 1)
    hist = np.zeros(n_bins, dtype="float64")
    b_lo = bars["low"].to_numpy("float64")
    b_hi = bars["high"].to_numpy("float64")
    b_v = bars["volume"].to_numpy("float64")
    for lv, hv, v in zip(b_lo, b_hi, b_v):
        if not (math.isfinite(lv) and math.isfinite(hv) and math.isfinite(v)) or v <= 0:
            continue
        i0 = int(np.searchsorted(edges, lv, side="right")) - 1
        i1 = int(np.searchsorted(edges, hv, side="left"))
        i0 = max(0, min(n_bins - 1, i0))
        i1 = max(i0 + 1, min(n_bins, i1))
        hist[i0:i1] += v / float(i1 - i0)
    if hist.sum() <= 0:
        return None
    return hist, edges


def _value_area(hist: np.ndarray, edges: np.ndarray, *, va_pct: float) -> tuple[float, float, float]:
    """POC / VAH / VAL by expanding outward from the POC bin."""
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(hist)
    total = float(hist.sum())
    # Tie-break to the central bin: a flat profile has no single busiest price,
    # and argmax would otherwise pin the POC to the low edge of the range.
    peak = float(hist.max())
    tied = np.flatnonzero(hist >= peak - 1e-9)
    poc_i = int(tied[len(tied) // 2])
    target = total * va_pct
    lo_i = hi_i = poc_i
    acc = float(hist[poc_i])
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1.0
        up = hist[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            acc += float(hist[hi_i])
        else:
            lo_i -= 1
            acc += float(hist[lo_i])
    return float(centers[poc_i]), float(centers[hi_i]), float(centers[lo_i])


def _shape_code(hist: np.ndarray, edges: np.ndarray) -> float:
    """Classify the profile as D / P / b / double-distribution.

    Double distribution wins when two modes are separated by a valley deep
    enough (<35% of the lower peak) and far enough apart (>25% of the range).
    Otherwise the volume-weighted skew about the midpoint decides P vs b vs D.
    """
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    if total <= 0 or len(hist) < 5:
        return SHAPE_D

    # Smooth so single-bin spikes do not read as modes.
    k = max(3, len(hist) // 12)
    if k % 2 == 0:
        k += 1
    kern = np.ones(k) / k
    sm = np.convolve(hist, kern, mode="same")

    peak_i = int(np.argmax(sm))
    peak_v = float(sm[peak_i])
    if peak_v > 0:
        span = centers[-1] - centers[0]
        min_sep = 0.25 * len(sm)
        for j in range(len(sm)):
            if abs(j - peak_i) < min_sep or sm[j] < 0.55 * peak_v:
                continue
            a, b = (peak_i, j) if peak_i < j else (j, peak_i)
            valley = float(sm[a:b + 1].min())
            if valley < 0.35 * min(peak_v, float(sm[j])) and span > 0:
                return SHAPE_DOUBLE

    mean = float((centers * hist).sum() / total)
    var = float((hist * (centers - mean) ** 2).sum() / total)
    if var <= 0:
        return SHAPE_D
    skew = float((hist * (centers - mean) ** 3).sum() / total) / (var ** 1.5)
    if skew < -0.45:
        return SHAPE_P   # mass high, thin tail reaching down
    if skew > 0.45:
        return SHAPE_b   # mass low, thin tail reaching up
    return SHAPE_D


def session_levels(
    bars: pd.DataFrame, *, va_pct: float = 0.70, bin_pts: float = 1.0,
) -> dict[str, float]:
    """POC / VAH / VAL / high / low / close / open / shape for one session."""
    nan = float("nan")
    empty = {
        "poc": nan, "vah": nan, "val": nan, "high": nan, "low": nan,
        "close": nan, "open": nan, "shape": nan,
    }
    if bars is None or bars.empty:
        return empty
    hx = _histogram(bars, bin_pts=bin_pts)
    if hx is None:
        return empty
    hist, edges = hx
    poc, vah, val = _value_area(hist, edges, va_pct=va_pct)
    return {
        "poc": poc, "vah": vah, "val": val,
        "high": float(bars["high"].max()),
        "low": float(bars["low"].min()),
        "close": float(bars["close"].iloc[-1]),
        "open": float(bars["open"].iloc[0]),
        "shape": _shape_code(hist, edges),
    }


# ── session slicing ──────────────────────────────────────────────────────

def _ct_session_frame(bars_1m: pd.DataFrame, *, start_min: int, end_min: int) -> pd.DataFrame:
    """Attach CT session date and restrict to the [start, end) session window."""
    df = bars_1m
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts_event" in df.columns:
            df = df.set_index(pd.to_datetime(df["ts_event"], utc=True))
        elif "bucket" in df.columns:
            df = df.set_index(pd.to_datetime(df["bucket"], utc=True))
        else:
            raise ValueError("bars_1m needs a DatetimeIndex or ts_event/bucket column")
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    ct = idx.tz_convert(CT)
    mins = ct.hour * 60 + ct.minute
    out = df.copy()
    out.index = idx
    out["_date"] = np.asarray(ct.date, dtype=object)
    out["_ct_min"] = mins.astype("int32")
    return out.loc[(out["_ct_min"] >= start_min) & (out["_ct_min"] < end_min)].sort_index()


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def _naked_pocs(
    ordered: list[tuple[Any, dict[str, float]]], *, upto_i: int,
) -> list[tuple[float, int]]:
    """POCs from sessions < ``upto_i`` never revisited by any later session.

    Returns ``(poc_price, age_in_sessions)`` where age counts sessions elapsed
    since the POC was printed.
    """
    out: list[tuple[float, int]] = []
    for i in range(upto_i):
        poc = ordered[i][1].get("poc", float("nan"))
        if not math.isfinite(poc):
            continue
        touched = False
        for j in range(i + 1, upto_i):
            lv = ordered[j][1].get("low", float("nan"))
            hv = ordered[j][1].get("high", float("nan"))
            if math.isfinite(lv) and math.isfinite(hv) and lv <= poc <= hv:
                touched = True
                break
        if not touched:
            out.append((poc, upto_i - i))
    return out


# ── main entry point ─────────────────────────────────────────────────────

def compute(
    bars_1m: pd.DataFrame,
    underlying_prices: pd.Series,
    *,
    atr_daily: float | None = None,
    timestep: str | None = None,
    va_pct: float = 0.70,
    bin_pts: float = 1.0,
    level_tol_ticks: float = 4.0,
    session_start: str = "08:30",
    session_end: str = "15:00",
) -> pd.DataFrame:
    """Prior-session volume-profile structure on the ``underlying_prices`` grid.

    ``bars_1m`` must cover the prior N sessions **and** the current session
    (1-minute OHLCV). The current session is the CT date of the last grid
    timestamp; every prior-session quantity is resolved from sessions strictly
    before it. ``atr_daily`` defaults to the mean prior-session high−low range.
    """
    timestep = timestep or get_timestep()
    und = underlying_prices.resample(timestep).last().ffill()
    out = pd.DataFrame(index=und.index, columns=list(_COLUMNS), dtype="float64")
    out[:] = np.nan
    if und.empty:
        return out.astype(np.float32)

    start_min, end_min = _hhmm(session_start), _hhmm(session_end)
    try:
        sess = _ct_session_frame(bars_1m, start_min=start_min, end_min=end_min)
    except (ValueError, KeyError):
        return out.astype(np.float32)
    if sess.empty:
        return out.astype(np.float32)

    ordered: list[tuple[Any, dict[str, float]]] = []
    for d, rows in sess.groupby("_date", sort=True):
        ordered.append((d, session_levels(rows, va_pct=va_pct, bin_pts=bin_pts)))
    if len(ordered) < 2:
        return out.astype(np.float32)

    grid_ct = und.index.tz_convert(CT)
    today: date = grid_ct[-1].date()
    cur_i = next((i for i, (d, _) in enumerate(ordered) if d == today), len(ordered) - 1)
    if cur_i == 0:
        return out.astype(np.float32)
    prior = ordered[cur_i - 1][1]

    ranges = [
        lv["high"] - lv["low"] for _, lv in ordered[:cur_i]
        if math.isfinite(lv.get("high", np.nan)) and math.isfinite(lv.get("low", np.nan))
    ]
    atr = float(atr_daily) if atr_daily else (float(np.mean(ranges)) if ranges else float("nan"))
    if not math.isfinite(atr) or atr <= 0:
        return out.astype(np.float32)

    px = und.to_numpy("float64")
    tol = level_tol_ticks * TICK

    # 1 — signed ATR-normalised distances (prior-session levels are constant today)
    for col, key in (
        ("pv_dist_poc", "poc"), ("pv_dist_vah", "vah"), ("pv_dist_val", "val"),
        ("pv_dist_phigh", "high"), ("pv_dist_plow", "low"), ("pv_dist_pclose", "close"),
    ):
        lvl = prior.get(key, float("nan"))
        if math.isfinite(lvl):
            out[col] = (px - lvl) / atr

    poc, vah, val = prior.get("poc"), prior.get("vah"), prior.get("val")

    # 2 — ordinal zone vs the prior value area
    if all(math.isfinite(v) for v in (poc, vah, val)):
        zone = np.full(len(px), np.nan)
        zone[px < val] = 0.0
        zone[(px >= val) & (px < poc - tol)] = 1.0
        zone[np.abs(px - poc) <= tol] = 2.0
        zone[(px > poc + tol) & (px <= vah)] = 3.0
        zone[px > vah] = 4.0
        out["pv_zone"] = zone
        out["pv_va_width_atr"] = (vah - val) / atr

    if math.isfinite(prior.get("shape", float("nan"))):
        out["pv_shape"] = prior["shape"]

    # 3 — causal interaction history, accumulated on today's 1m bars
    cur = sess.loc[sess["_date"] == today]
    if not cur.empty:
        c_hi = cur["high"].to_numpy("float64")
        c_lo = cur["low"].to_numpy("float64")
        c_cl = cur["close"].to_numpy("float64")
        n = len(cur)
        since = np.full(n, np.nan)
        tests = np.zeros(n)
        val_rej = np.zeros(n)
        vah_rej = np.zeros(n)
        accept = np.zeros(n)
        bars_since = -1
        n_tests = 0
        prev_touch = False
        run = 0
        for i in range(n):
            if math.isfinite(poc) and c_lo[i] <= poc <= c_hi[i]:
                bars_since = 0
                if not prev_touch:
                    n_tests += 1
                prev_touch = True
            else:
                prev_touch = False
                if bars_since >= 0:
                    bars_since += 1
            since[i] = bars_since if bars_since >= 0 else np.nan
            tests[i] = n_tests
            if math.isfinite(val) and c_lo[i] < val and c_cl[i] > val:
                val_rej[i] = 1
            if math.isfinite(vah) and c_hi[i] > vah and c_cl[i] < vah:
                vah_rej[i] = 1
            run = run + 1 if (math.isfinite(val) and val <= c_cl[i] <= vah) else 0
            accept[i] = run
        hist_1m = pd.DataFrame(
            {
                "pv_bars_since_poc_touch": since,
                "pv_poc_test_count_today": tests,
                "pv_val_rejection": np.cumsum(val_rej),
                "pv_vah_rejection": np.cumsum(vah_rej),
                "pv_va_acceptance": accept,
            },
            index=cur.index,
        )
        aligned = hist_1m.reindex(out.index, method="ffill")
        for c in hist_1m.columns:
            out[c] = aligned[c]

        # 5 — open type and gap (constant within the session)
        s_open = float(cur["open"].iloc[0])
        if all(math.isfinite(v) for v in (vah, val)):
            out["pv_open_vs_prior_va"] = 2.0 if s_open > vah else (0.0 if s_open < val else 1.0)
        p_close = prior.get("close", float("nan"))
        if math.isfinite(p_close):
            out["pv_gap_atr"] = (s_open - p_close) / atr

        # 6 — developing VA overlap with the prior VA, accumulated causally
        if all(math.isfinite(v) for v in (vah, val)):
            ov = np.full(len(cur), np.nan)
            prior_w = vah - val
            for i in range(n):
                if i < 4:
                    continue
                hx = _histogram(cur.iloc[: i + 1], bin_pts=bin_pts)
                if hx is None:
                    continue
                _, d_vah, d_val = _value_area(hx[0], hx[1], va_pct=va_pct)
                inter = max(0.0, min(vah, d_vah) - max(val, d_val))
                denom = max(prior_w, d_vah - d_val)
                ov[i] = inter / denom if denom > 0 else np.nan
            out["pv_va_overlap_pct"] = pd.Series(ov, index=cur.index).reindex(
                out.index, method="ffill"
            )

    # 4 — naked (untested) POC stack, resolved from sessions before today
    naked = _naked_pocs(ordered, upto_i=cur_i)
    if naked:
        prices = np.array([p for p, _ in naked], dtype="float64")
        ages = np.array([a for _, a in naked], dtype="float64")
        a_d = np.full(len(px), np.nan)
        a_a = np.full(len(px), np.nan)
        b_d = np.full(len(px), np.nan)
        b_a = np.full(len(px), np.nan)
        for i, p in enumerate(px):
            if not math.isfinite(p):
                continue
            above = prices > p
            below = prices < p
            if above.any():
                j = int(np.argmin(prices[above] - p))
                a_d[i] = (prices[above][j] - p) / atr
                a_a[i] = ages[above][j]
            if below.any():
                j = int(np.argmin(p - prices[below]))
                b_d[i] = (p - prices[below][j]) / atr
                b_a[i] = ages[below][j]
        out["pv_naked_poc_above_dist"] = a_d
        out["pv_naked_poc_above_age"] = a_a
        out["pv_naked_poc_below_dist"] = b_d
        out["pv_naked_poc_below_age"] = b_a

    return out.astype(np.float32)
