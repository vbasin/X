"""Causal Globex session volume profile (POC / 70% VA) + break/reclaim.

Fold-cache attach on the **5m** store only (ADR-099). Not hive parquet.
Session D starts **17:00 CT on D−1** (same Globex open as Live Chart
``session`` origin), not RTH 08:30. Volume = ``cs_volume`` else ``|of_delta|``.
"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHI = ZoneInfo("America/Chicago")
TICK_SIZE = 0.25
VA_PCT = 0.70
SVP_PREFIX = "svp_"

SVP_COLS: tuple[str, ...] = (
    "svp_poc_dist",
    "svp_va_high_dist",
    "svp_va_low_dist",
    "svp_in_value_area",
    "svp_above_poc",
    "svp_broke_vah",
    "svp_broke_val",
    "svp_reclaim_vah",
    "svp_reclaim_val",
    "svp_poc_cross_up",
    "svp_poc_cross_dn",
)


SVP_STORE = "5m"
GLOBEX_OPEN_HOUR = 17


def wants_session_vp(objective: dict[str, Any] | None, store: str | None = None) -> bool:
    topo = ((objective or {}).get("search") or {}).get("topology") or {}
    if not bool(topo.get("session_vp")):
        return False
    if store is None:
        return True
    return str(store).strip() == SVP_STORE


def session_origin_hhmm(objective: dict[str, Any] | None = None) -> str:
    return f"{GLOBEX_OPEN_HOUR:02d}:00"


def value_area_contiguous(
    ticks: np.ndarray,
    vols: np.ndarray,
    va_pct: float = VA_PCT,
) -> tuple[float, float, float]:
    """POC, VAH, VAL — expand from POC along sorted price until *va_pct* of volume."""
    px = np.asarray(ticks, dtype="float64")
    vol = np.asarray(vols, dtype="float64")
    ok = np.isfinite(px) & np.isfinite(vol) & (vol > 0)
    if not bool(ok.any()):
        return (float("nan"), float("nan"), float("nan"))
    px, vol = px[ok], vol[ok]
    order = np.argsort(px, kind="mergesort")
    px, vol = px[order], vol[order]
    i_poc = int(np.argmax(vol))
    total = float(vol.sum())
    target = total * float(va_pct)
    lo = hi = i_poc
    acc = float(vol[i_poc])
    n = vol.size
    while acc + 1e-12 < target and (lo > 0 or hi < n - 1):
        up = float(vol[hi + 1]) if hi + 1 < n else -1.0
        dn = float(vol[lo - 1]) if lo > 0 else -1.0
        if up > dn:
            hi += 1
            acc += float(vol[hi])
        elif dn > up:
            lo -= 1
            acc += float(vol[lo])
        else:
            if up < 0 and dn < 0:
                break
            if up >= 0:
                hi += 1
                acc += float(vol[hi])
            if dn >= 0:
                lo -= 1
                acc += float(vol[lo])
    return float(px[i_poc]), float(px[hi]), float(px[lo])


def globex_session_keys(idx: pd.DatetimeIndex) -> np.ndarray:
    """Session D = [D−1 17:00 CT, D 17:00 CT). Returns YYYY-MM-DD of D."""
    ct = pd.DatetimeIndex(pd.to_datetime(idx, utc=True)).tz_convert(CHI)
    day = pd.to_datetime(ct.strftime("%Y-%m-%d"))
    nxt = day + pd.Timedelta(days=1)
    use_next = ct.hour.to_numpy() >= GLOBEX_OPEN_HOUR
    return np.where(use_next, nxt.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))


def attach_session_vp(
    df: pd.DataFrame,
    *,
    origin_hhmm: str = "17:00",
    tick_size: float = TICK_SIZE,
    va_pct: float = VA_PCT,
) -> pd.DataFrame:
    """Add ``svp_*`` columns. Idempotent. Causal: bar t uses volume through t."""
    if df is None or df.empty:
        return df
    existing = [c for c in df.columns if str(c).startswith(SVP_PREFIX)]
    if existing:
        return df
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    close = _col(df, "cs_close", "_close")
    if close is None:
        return df
    vol = _col(df, "cs_volume")
    if vol is None:
        delta = _col(df, "of_delta")
        vol = delta.abs() if delta is not None else pd.Series(1.0, index=idx)
    high, low = _high_low(df, close)
    typical = (high + low + close) / 3.0
    ticks = (typical.to_numpy(dtype="float64") / tick_size).round() * tick_size
    sizes = vol.to_numpy(dtype="float64")
    px = close.to_numpy(dtype="float64")
    day = globex_session_keys(idx)

    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    in_va = np.zeros(n, dtype="float64")
    above_poc = np.zeros(n, dtype="float64")
    broke_h = np.zeros(n, dtype="float64")
    broke_l = np.zeros(n, dtype="float64")
    rec_h = np.zeros(n, dtype="float64")
    rec_l = np.zeros(n, dtype="float64")
    x_up = np.zeros(n, dtype="float64")
    x_dn = np.zeros(n, dtype="float64")

    from collections import defaultdict

    for d in np.unique(day):
        ix = np.flatnonzero(day == d)
        vbp: dict[float, float] = defaultdict(float)
        ever_h = False
        ever_l = False
        prev_above: float | None = None
        for j in ix:
            if np.isfinite(ticks[j]) and np.isfinite(sizes[j]) and sizes[j] > 0:
                vbp[float(ticks[j])] += float(sizes[j])
            if not vbp:
                continue
            keys = np.fromiter(vbp.keys(), dtype="float64")
            vals = np.fromiter(vbp.values(), dtype="float64")
            p, hi, lo = value_area_contiguous(keys, vals, va_pct=va_pct)
            poc[j], vah[j], val[j] = p, hi, lo
            c = px[j]
            if not np.isfinite(c):
                continue
            inside = np.isfinite(hi) and np.isfinite(lo) and lo <= c <= hi
            in_va[j] = 1.0 if inside else 0.0
            above = c > p if np.isfinite(p) else False
            above_poc[j] = 1.0 if above else 0.0
            if prev_above is not None and np.isfinite(p):
                if (not prev_above) and above:
                    x_up[j] = 1.0
                if prev_above and (not above) and c < p:
                    x_dn[j] = 1.0
            prev_above = above if np.isfinite(p) else prev_above
            if np.isfinite(hi) and c > hi:
                broke_h[j] = 1.0
                ever_h = True
            elif ever_h and inside:
                rec_h[j] = 1.0
            if np.isfinite(lo) and c < lo:
                broke_l[j] = 1.0
                ever_l = True
            elif ever_l and inside:
                rec_l[j] = 1.0

    den = np.where(np.isfinite(px) & (np.abs(px) > 1e-6), px, np.nan)
    out = df.copy()
    out["svp_poc_dist"] = (px - poc) / den
    out["svp_va_high_dist"] = (px - vah) / den
    out["svp_va_low_dist"] = (px - val) / den
    out["svp_in_value_area"] = in_va
    out["svp_above_poc"] = above_poc
    out["svp_broke_vah"] = broke_h
    out["svp_broke_val"] = broke_l
    out["svp_reclaim_vah"] = rec_h
    out["svp_reclaim_val"] = rec_l
    out["svp_poc_cross_up"] = x_up
    out["svp_poc_cross_dn"] = x_dn
    extra = list(SVP_COLS)
    out[extra] = out[extra].replace([np.inf, -np.inf], np.nan)
    return out


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return None


def _high_low(df: pd.DataFrame, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    high = _col(df, "cs_high")
    low = _col(df, "cs_low")
    if high is not None and low is not None:
        return high.reindex(close.index), low.reindex(close.index)
    rng = _col(df, "cs_range")
    clv = _col(df, "cs_clv")
    if rng is None:
        return close.copy(), close.copy()
    from src.features.target_geometry import high_low_from_cs
    h, l = high_low_from_cs(
        close.to_numpy(),
        rng.reindex(close.index).to_numpy(),
        (clv.reindex(close.index).to_numpy() if clv is not None
         else np.full(len(close), np.nan)),
    )
    return pd.Series(h, index=close.index), pd.Series(l, index=close.index)
