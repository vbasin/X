"""CVD measured *relative to price structure* (ADR-122).

:mod:`src.features.order_flow` already carries CVD magnitude and its
derivatives (``of_delta``, ``of_cvd_chg_*``, ``of_delta_zscore_*``) and
:mod:`src.features.large_trade_cvd` the ≥N-lot variant. What neither carries is
delta *in relation to where price is* — which is the quantity a discretionary
reader actually extracts from a CVD pane:

* ``of_cvd_at_level``      — delta accumulated while price sat on a prior-session
                             level (absorption), rather than delta in general
* ``lt_cvd_at_level``      — the same, restricted to large trades ("who")
* ``of_cvd_divergence_slope`` — continuous, signed regression slope of CVD against
                             price, replacing a sign-only divergence flag
* ``of_cvd_session_pctile``   — today's CVD against the same time-of-day across
                             prior sessions, so the value is comparable across
                             volatility regimes
* ``of_cvd_vwap_reversion``   — delta accumulated since the last VWAP cross,
                             signed by the direction of that cross

Columns keep the ``of_``/``lt_`` prefixes of the CVD family they extend rather
than taking a module-specific prefix; the feature engine concatenates on column
name, so the prefix denotes the semantic family.

All columns are causal: cumulative sums reset at the session open and rolling
statistics look backward only.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from src.common.bar_windows import bars_m, get_timestep

CT = ZoneInfo("America/Chicago")
TICK = 0.25

_LEVEL_KEYS = ("poc", "vah", "val", "high", "low", "close")

COLUMNS = (
    "of_cvd_at_level",
    "lt_cvd_at_level",
    "of_cvd_divergence_slope",
    "of_cvd_session_pctile",
    "of_cvd_vwap_reversion",
)


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def _signed(df: pd.DataFrame) -> np.ndarray:
    side = df["side"].to_numpy()
    size = df["size"].to_numpy("float64")
    return np.where(side == "B", size, np.where(side == "A", -size, 0.0))


def _rolling_slope(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Backward-looking OLS slope of ``y`` on ``x`` — cov(x, y) / var(x)."""
    cov = x.rolling(window).cov(y)
    var = x.rolling(window).var()
    return cov / var.replace(0, np.nan)


def compute(
    tbbo: pd.DataFrame,
    underlying_prices: pd.Series,
    *,
    levels: dict[str, float] | None = None,
    timestep: str | None = None,
    session_start: str = "08:30",
    level_tol_ticks: float = 4.0,
    large_trade_size: int = 10,
    slope_minutes: int = 5,
    cvd_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """CVD-versus-structure features on the ``underlying_prices`` grid.

    ``levels`` holds the prior-session levels (``poc`` / ``vah`` / ``val`` /
    ``high`` / ``low`` / ``close``) as produced by
    :func:`src.features.prior_volume_profile.session_levels`. When it is absent
    the two ``*_at_level`` columns stay NaN and the rest are still computed.

    ``cvd_history`` is an optional frame of prior sessions' CVD indexed by CT
    minute-of-day with one column per session date; when supplied,
    ``of_cvd_session_pctile`` reports today's CVD as its quantile within that
    row. Without it the column stays NaN.
    """
    timestep = timestep or get_timestep()
    und = underlying_prices.resample(timestep).last().ffill()
    out = pd.DataFrame(index=und.index, columns=list(COLUMNS), dtype="float64")
    out[:] = np.nan
    if und.empty or tbbo is None or tbbo.empty:
        return out.astype(np.float32)

    df = tbbo.copy()
    if "ts_event" not in df.columns:
        return out.astype(np.float32)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts_event")
    df["_sd"] = _signed(df)

    start_min = _hhmm(session_start)
    ct = df["ts_event"].dt.tz_convert(CT)
    df["_ct_min"] = ct.dt.hour * 60 + ct.dt.minute
    df = df.loc[df["_ct_min"] >= start_min]
    if df.empty:
        return out.astype(np.float32)

    px = df["price"].to_numpy("float64")

    # ── delta accumulated on a prior-session level (absorption) ──────────
    if levels:
        tol = level_tol_ticks * TICK
        lv = [levels[k] for k in _LEVEL_KEYS
              if k in levels and math.isfinite(levels.get(k, float("nan")))]
        if lv:
            near = np.zeros(len(df), dtype=bool)
            for level in lv:
                near |= np.abs(px - level) <= tol
            at_lvl = np.where(near, df["_sd"].to_numpy("float64"), 0.0)
            big = df["size"].to_numpy("float64") >= large_trade_size
            lt_at_lvl = np.where(near & big, df["_sd"].to_numpy("float64"), 0.0)
            s = pd.Series(at_lvl, index=df["ts_event"]).resample(timestep).sum().cumsum()
            lt = pd.Series(lt_at_lvl, index=df["ts_event"]).resample(timestep).sum().cumsum()
            out["of_cvd_at_level"] = s.reindex(out.index, method="ffill")
            out["lt_cvd_at_level"] = lt.reindex(out.index, method="ffill")

    # ── session CVD on the grid ──────────────────────────────────────────
    delta_bar = pd.Series(df["_sd"].to_numpy("float64"), index=df["ts_event"]).resample(timestep).sum()
    cvd = delta_bar.cumsum().reindex(out.index, method="ffill")

    # ── continuous, signed CVD-vs-price regression slope ────────────────
    w = max(2, bars_m(slope_minutes))
    price = und.reindex(out.index)
    out["of_cvd_divergence_slope"] = _rolling_slope(cvd, price, w)

    # ── cross-session normalisation of the CVD level ─────────────────────
    if cvd_history is not None and not cvd_history.empty:
        grid_ct = out.index.tz_convert(CT)
        minute = grid_ct.hour * 60 + grid_ct.minute
        pct = np.full(len(out), np.nan)
        cvd_v = cvd.to_numpy("float64")
        for i, m in enumerate(minute):
            if m not in cvd_history.index or not math.isfinite(cvd_v[i]):
                continue
            row = cvd_history.loc[m].to_numpy("float64")
            row = row[np.isfinite(row)]
            if row.size:
                pct[i] = float((row <= cvd_v[i]).mean())
        out["of_cvd_session_pctile"] = pct

    # ── delta since the last VWAP cross, signed by that cross ────────────
    pv = (df["price"].to_numpy("float64") * df["size"].to_numpy("float64"))
    num = pd.Series(pv, index=df["ts_event"]).resample(timestep).sum().cumsum()
    den = pd.Series(df["size"].to_numpy("float64"), index=df["ts_event"]).resample(timestep).sum().cumsum()
    vwap = (num / den.replace(0, np.nan)).reindex(out.index, method="ffill")

    p = price.to_numpy("float64")
    v = vwap.to_numpy("float64")
    c = cvd.to_numpy("float64")
    sign = np.sign(p - v)
    rev = np.full(len(out), np.nan)
    anchor = np.nan
    cross_dir = 0.0
    last_sign = 0.0
    for i in range(len(out)):
        if not (math.isfinite(p[i]) and math.isfinite(v[i]) and math.isfinite(c[i])):
            continue
        s_i = sign[i]
        if s_i != 0 and last_sign != 0 and s_i != last_sign:
            anchor = c[i]
            cross_dir = s_i
        if s_i != 0:
            last_sign = s_i
        if math.isfinite(anchor):
            rev[i] = (c[i] - anchor) * cross_dir
    out["of_cvd_vwap_reversion"] = rev

    return out.astype(np.float32)
