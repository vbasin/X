"""Causal auction-structure columns from existing 5s + 1m hive (no new vendor data).

IB = first RTH hour 08:30–09:30 CT (running until freeze, then held).
Overnight = Globex 17:00 CT previous to 08:30 CT, from native 1m OHLC.
Flow-at-level = session cum of_ofi / of_delta while price is near that level.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.features.auction_structure")

AUCTION_PREFIX = "au_"
_RTH_OPEN_MIN = 8 * 60 + 30
_IB_END_MIN = 9 * 60 + 30
_NEAR_PTS = 4.0


def attach_auction_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``au_*`` columns onto a UTC DatetimeIndex 5s frame. Idempotent."""
    if df is None or df.empty:
        return df
    existing = [c for c in df.columns if str(c).startswith(AUCTION_PREFIX)]
    if existing:
        return df
    close = _col(df, "cs_close", "_close")
    high = _col(df, "cs_high", "cs_close", "_close")
    low = _col(df, "cs_low", "cs_close", "_close")
    if close is None:
        return df
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    ct = idx.tz_convert("America/Chicago")
    minutes = (ct.hour * 60 + ct.minute).to_numpy()
    day = np.asarray(ct.strftime("%Y-%m-%d"))
    px = close.to_numpy(dtype="float64")
    hi = high.to_numpy(dtype="float64") if high is not None else px
    lo = low.to_numpy(dtype="float64") if low is not None else px
    ofi = _np(df, "of_ofi")
    delta = _np(df, "of_delta")
    vf_up = _np(df, "vf_1m_upper_delta")
    vf_lo = _np(df, "vf_1m_lower_delta")
    vwap_d = _np(df, "vw_vwap_dist")

    ib_h, ib_l = _ib_levels(hi, lo, minutes, day)
    on_h, on_l = _overnight_levels(idx, day)

    out["au_ib_high_dist"] = px - ib_h
    out["au_ib_low_dist"] = px - ib_l
    ib_rng = np.maximum(ib_h - ib_l, 1e-6)
    out["au_ib_pos"] = np.clip((px - ib_l) / ib_rng, 0.0, 1.0)
    broke_h, fail_h = _break_fail(px, ib_h, day, minutes >= _IB_END_MIN)
    broke_l, fail_l = _break_fail(-px, -ib_l, day, minutes >= _IB_END_MIN)
    out["au_ib_broke_high"] = broke_h
    out["au_ib_broke_low"] = broke_l
    out["au_ib_fail_high"] = fail_h
    out["au_ib_fail_low"] = fail_l
    out["au_ofi_at_ib_high"] = _cum_near(ofi, px, ib_h, day)
    out["au_ofi_at_ib_low"] = _cum_near(ofi, px, ib_l, day)
    out["au_delta_at_ib_high"] = _cum_near(delta, px, ib_h, day)
    out["au_delta_at_ib_low"] = _cum_near(delta, px, ib_l, day)
    near_ib_h = np.abs(px - ib_h) <= _NEAR_PTS
    near_ib_l = np.abs(px - ib_l) <= _NEAR_PTS
    out["au_vf_upper_at_ib_high"] = np.where(near_ib_h, vf_up, 0.0)
    out["au_vf_lower_at_ib_low"] = np.where(near_ib_l, vf_lo, 0.0)

    out["au_on_high_dist"] = px - on_h
    out["au_on_low_dist"] = px - on_l
    on_broke_h, on_fail_h = _break_fail(px, on_h, day, minutes >= _RTH_OPEN_MIN)
    on_broke_l, on_fail_l = _break_fail(-px, -on_l, day, minutes >= _RTH_OPEN_MIN)
    out["au_on_broke_high"] = on_broke_h
    out["au_on_broke_low"] = on_broke_l
    out["au_on_fail_high"] = on_fail_h
    out["au_on_fail_low"] = on_fail_l
    out["au_ofi_at_on_high"] = _cum_near(ofi, px, on_h, day)
    out["au_delta_at_on_high"] = _cum_near(delta, px, on_h, day)

    rec_up, rec_dn = _vwap_reclaim(vwap_d, day)
    out["au_vwap_reclaim_up"] = rec_up
    out["au_vwap_reclaim_dn"] = rec_dn

    extra = [c for c in out.columns if str(c).startswith(AUCTION_PREFIX)]
    out[extra] = out[extra].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return None


def _np(df: pd.DataFrame, name: str) -> np.ndarray:
    n = len(df)
    if name not in df.columns:
        return np.zeros(n, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype="float64")


def _ib_levels(
    high: np.ndarray, low: np.ndarray, minutes: np.ndarray, day: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = high.size
    ib_h = np.full(n, np.nan)
    ib_l = np.full(n, np.nan)
    in_ib = (minutes >= _RTH_OPEN_MIN) & (minutes < _IB_END_MIN)
    for d in np.unique(day):
        m = day == d
        ix = np.flatnonzero(m)
        if ix.size == 0:
            continue
        loc_ib = in_ib[ix]
        h = np.where(np.isfinite(high[ix]), high[ix], -np.inf)
        l = np.where(np.isfinite(low[ix]), low[ix], np.inf)
        run_h = np.full(ix.size, np.nan)
        run_l = np.full(ix.size, np.nan)
        ch, cl = -np.inf, np.inf
        frozen_h, frozen_l = np.nan, np.nan
        for j in range(ix.size):
            if loc_ib[j]:
                if h[j] > ch:
                    ch = h[j]
                if l[j] < cl:
                    cl = l[j]
                run_h[j] = ch if np.isfinite(ch) else np.nan
                run_l[j] = cl if np.isfinite(cl) else np.nan
                frozen_h, frozen_l = run_h[j], run_l[j]
            else:
                run_h[j] = frozen_h
                run_l[j] = frozen_l
        ib_h[ix] = run_h
        ib_l[ix] = run_l
    return ib_h, ib_l


def _break_fail(
    signed_px: np.ndarray,
    signed_level: np.ndarray,
    day: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Broke = signed_px > signed_level. Fail = had broken this session, now back."""
    n = signed_px.size
    broke = np.zeros(n, dtype="float64")
    fail = np.zeros(n, dtype="float64")
    for d in np.unique(day):
        m = (day == d) & active & np.isfinite(signed_px) & np.isfinite(signed_level)
        ix = np.flatnonzero(m)
        if ix.size == 0:
            continue
        above = signed_px[ix] > signed_level[ix]
        ever = np.maximum.accumulate(above.astype(np.int8))
        broke[ix] = above.astype("float64")
        fail[ix] = ((ever == 1) & (~above)).astype("float64")
    return broke, fail


def _cum_near(
    flow: np.ndarray, px: np.ndarray, level: np.ndarray, day: np.ndarray,
) -> np.ndarray:
    near = np.isfinite(px) & np.isfinite(level) & (np.abs(px - level) <= _NEAR_PTS)
    contrib = np.where(near, flow, 0.0)
    out = np.zeros(flow.size, dtype="float64")
    for d in np.unique(day):
        m = day == d
        out[m] = np.cumsum(contrib[m])
    return out


def _vwap_reclaim(dist: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.zeros(dist.size, dtype="float64")
    dn = np.zeros(dist.size, dtype="float64")
    sign = np.sign(dist)
    prev = np.roll(sign, 1)
    prev[0] = 0
    for d in np.unique(day):
        m = day == d
        ix = np.flatnonzero(m)
        if ix.size < 2:
            continue
        s = sign[ix]
        p = np.roll(s, 1)
        p[0] = 0
        up[ix] = ((p < 0) & (s > 0)).astype("float64")
        dn[ix] = ((p > 0) & (s < 0)).astype("float64")
    return up, dn


def _overnight_levels(
    idx: pd.DatetimeIndex, day: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(idx)
    on_h = np.full(n, np.nan)
    on_l = np.full(n, np.nan)
    try:
        from src.features.stores import load_feature_frame, store_available
        if not store_available("1m"):
            return on_h, on_l
        t0 = (pd.Timestamp(idx.min()) - timedelta(days=2)).date()
        t1 = pd.Timestamp(idx.max()).date()
        m1 = load_feature_frame("1m", t0, t1)
        if m1 is None or m1.empty or "cs_high" not in m1.columns:
            return on_h, on_l
        m1 = m1.copy()
        m1.index = pd.to_datetime(m1.index, utc=True)
        ct = m1.index.tz_convert("America/Chicago")
        minutes = ct.hour * 60 + ct.minute
        # Overnight for Chicago calendar date D: [D-1 17:00, D 08:30).
        chi_date = np.asarray(ct.strftime("%Y-%m-%d"))
        on_mask = minutes < _RTH_OPEN_MIN
        # bars after 17:00 belong to next calendar day's overnight
        after_17 = minutes >= 17 * 60
        on_date = chi_date.copy()
        # shift 17:00+ to next calendar day
        nxt = pd.to_datetime(chi_date) + pd.Timedelta(days=1)
        on_date = np.where(after_17, np.asarray(nxt.strftime("%Y-%m-%d")), on_date)
        use = on_mask | after_17
        hi = pd.to_numeric(m1["cs_high"], errors="coerce").to_numpy()
        lo = pd.to_numeric(m1["cs_low"], errors="coerce").to_numpy()
        levels: dict[str, tuple[float, float]] = {}
        for d in np.unique(on_date[use]):
            sl = use & (on_date == d)
            if not sl.any():
                continue
            h = np.nanmax(hi[sl]) if np.isfinite(hi[sl]).any() else np.nan
            l = np.nanmin(lo[sl]) if np.isfinite(lo[sl]).any() else np.nan
            levels[str(d)] = (float(h), float(l))
        for i, d in enumerate(day):
            pair = levels.get(str(d))
            if pair:
                on_h[i], on_l[i] = pair
    except Exception as exc:
        logger.warning("auction overnight from 1m failed: %s", exc)
    return on_h, on_l
