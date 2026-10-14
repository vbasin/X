"""Structural transitions against the prior session's profile (``st_`` prefix).

``es_vs_prior_vah`` is a *distance*: it reads the same whether price is breaking
down through the value-area high or rallying back up into it. Every scenario a
trader actually describes — entering value from outside, breaking out of it,
retracing to the broken edge and continuing, failing back through it, crossing
VWAP with force — is a **path** state, and a signed distance cannot express one.

This module adds the missing primitives, all measured on the 1-second grid
derived from the print tape:

* **event age**, in seconds since the transition, ``-1`` when it has not happened
  today. ``st_break_age_s = 8`` and ``= 900`` are different trades; a boolean
  makes them identical.
* **level-tagged flow** — the ``fs_`` acceptance/rejection taxonomy sampled *at
  the moment price touched a structural level*, so "rejection at VAL" is
  distinguishable from a rejection at nothing in particular.
* **state combinators** — an ordinal zone with dwell time, and a break state
  machine that separates a failed break (price back through the edge) from a
  retrace-and-continue (price back *to* the edge, then further out).

Signed columns are emitted in the absolute frame and listed in
:data:`DIRECTIONAL`; a short must negate them so both sides share one model.
``st_zone`` is deliberately symmetric about the POC for exactly this reason.

Ages use wall-clock seconds rather than row counts, because the 1s frame from
:func:`src.features.tape.tape_to_path` omits seconds with no print.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.cash_session_profile import SessionLevels

logger = logging.getLogger("vtech.features.structural_transitions")

PREFIX = "st_"

# Signed in the absolute frame; negated for a short so both sides share a model.
DIRECTIONAL_SUFFIXES: tuple[str, ...] = (
    "zone", "dist_prior_hi_r", "dist_prior_lo_r", "va_entry_depth_r",
    "break_level", "break_extension_r",
    "vwap_cross_side", "vwap_cross_strength",
    "level_touch_kind", "level_touch_accept", "level_touch_reject",
)
DIRECTIONAL: tuple[str, ...] = tuple(PREFIX + s for s in DIRECTIONAL_SUFFIXES)

COLUMNS: tuple[str, ...] = (
    "st_zone", "st_zone_dwell_s",
    "st_dist_prior_hi_r", "st_dist_prior_lo_r",
    "st_va_entry_age_s", "st_va_entry_depth_r", "st_va_time_inside_s",
    "st_break_age_s", "st_break_level", "st_break_extension_r",
    "st_break_max_extension_r", "st_break_pullback_r",
    "st_reclaim_flag", "st_reclaim_age_s", "st_reject_flag",
    "st_bounce_confirm", "st_bounce_accept",
    "st_vwap_cross_age_s", "st_vwap_cross_side", "st_vwap_cross_strength",
    "st_vwap_cross_count", "st_vwap_reclaim_flag",
    "st_level_touch_age_s", "st_level_touch_kind",
    "st_level_touch_accept", "st_level_touch_reject",
)

# Level identity, signed so a short's mirror maps hi<->lo and vah<->val.
LEVEL_CODES: tuple[float, ...] = (-3.0, -2.0, 0.0, 2.0, 3.0)   # lo val poc vah hi


@dataclass(frozen=True)
class TransitionConfig:
    """Tolerances in points; windows in seconds."""

    touch_tol_pts: float = 2.0        # how close counts as touching a level
    bounce_tol_pts: float = 2.0       # retrace back to the broken edge
    reject_window_s: int = 300        # a break undone this fast is a rejection
    vwap_velocity_window: int = 60    # lookback for cross strength
    vwap_reclaim_window: int = 300    # two crosses inside this is a failed cross

    def __post_init__(self) -> None:
        for n in ("touch_tol_pts", "bounce_tol_pts"):
            if getattr(self, n) <= 0:
                raise ValueError(f"{n} must be positive")
        for n in ("vwap_velocity_window", "vwap_reclaim_window", "reject_window_s"):
            if getattr(self, n) < 1:
                raise ValueError(f"{n} must be >= 1")


def _empty(index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index, columns=list(COLUMNS), dtype="float64")
    out[:] = np.nan
    return out.astype("float32")


def session_vwap(bars: pd.DataFrame, trades: pd.DataFrame | None = None) -> np.ndarray:
    """Cumulative volume-weighted average price, as of each row's own timestamp.

    Built from individual prints when *trades* is supplied: the true
    trade-weighted average, and causal because only prints at or before a
    row's own timestamp contribute. Falls back to the bar-close proxy when
    *trades* is omitted, which is not causally clean -- a bar's own close can
    carry a print up to ~1s after the bar's own timestamp, since
    ``tape_to_path`` floors bars to their start.
    """
    px = bars["close"].to_numpy(dtype="float64")
    if trades is not None and not trades.empty:
        t_ts = trades.index.to_numpy()
        t_px = trades["price"].to_numpy(dtype="float64")
        t_sz = trades["size"].to_numpy(dtype="float64")
        as_of = np.searchsorted(t_ts, bars.index.to_numpy(), side="right") - 1
        seen = as_of >= 0
        safe = np.maximum(as_of, 0)
        cum_pv = np.cumsum(t_px * t_sz)
        cum_v = np.maximum(np.cumsum(t_sz), 1e-9)
        return np.where(seen, cum_pv[safe] / cum_v[safe], px)
    vol = bars["volume"].to_numpy(dtype="float64")
    cum_v = np.cumsum(vol)
    cum_pv = np.cumsum(px * vol)
    return np.divide(cum_pv, cum_v, out=px.copy(), where=cum_v > 0)


def build_structural_transitions(
    bars: pd.DataFrame,
    prior: SessionLevels | None,
    *,
    r_value: float,
    trades: pd.DataFrame | None = None,
    accept: pd.Series | None = None,
    reject: pd.Series | None = None,
    cfg: TransitionConfig | None = None,
) -> pd.DataFrame:
    """``st_``-prefixed transition state on the *bars* index.

    *prior* is the previous **closed** cash session's five levels; *accept* and
    *reject* are the signed ``fs_accept_continue`` / ``fs_reject_reversal`` series on
    the same index, used only to tag structural touches. *trades* is the
    session's raw prints: the causal source for the current price and VWAP,
    neither of which reads the bar's own close when available -- a bar's own
    close can carry a print up to ~1s after the bar's own timestamp
    (``tape_to_path`` floors bars to their start). Omit it and both fall back
    to the bar's own close, which is not causally clean.
    """
    cfg = cfg or TransitionConfig()
    if bars.empty or prior is None:
        return _empty(bars.index)
    if not np.isfinite(r_value) or r_value <= 0:
        return _empty(bars.index)

    plo, val, poc, vah, phi = (float(prior.lo), float(prior.val), float(prior.poc),
                               float(prior.vah), float(prior.hi))
    if not all(np.isfinite(v) for v in (plo, val, poc, vah, phi)):
        return _empty(bars.index)

    bar_close = bars["close"].to_numpy(dtype="float64")
    if trades is not None and not trades.empty:
        t_ts = trades.index.to_numpy()
        t_px = trades["price"].to_numpy(dtype="float64")
        as_of = np.searchsorted(t_ts, bars.index.to_numpy(), side="right") - 1
        seen = as_of >= 0
        price = np.where(seen, t_px[np.maximum(as_of, 0)], bar_close)
    else:
        price = bar_close
    n = price.size
    ts_s = bars.index.values.astype("int64") // 1_000_000_000
    tol = cfg.touch_tol_pts

    # ── vectorised context ───────────────────────────────────────────────────
    zone = np.select(
        [price < plo, price < val, price < poc - tol, np.abs(price - poc) <= tol,
         price <= vah, price <= phi],
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], default=3.0)

    vwap = session_vwap(bars, trades)
    spread = price - vwap
    back = np.searchsorted(ts_s, ts_s - cfg.vwap_velocity_window, side="left")
    vwap_vel = (spread - spread[back]) / r_value

    lvl_px = np.array([plo, val, poc, vah, phi], dtype="float64")
    gap = np.abs(price[:, None] - lvl_px[None, :])
    near = np.argmin(gap, axis=1)
    touching = gap[np.arange(n), near] <= tol

    acc = (accept.reindex(bars.index).to_numpy(dtype="float64")
           if accept is not None else np.full(n, np.nan))
    rej = (reject.reindex(bars.index).to_numpy(dtype="float64")
           if reject is not None else np.full(n, np.nan))
    # Flow confirmation is only meaningful when the taxonomy was supplied.
    bounce_idle = 0.0 if accept is not None else np.nan

    inside_va = (price >= val) & (price <= vah)

    # ── sequential state ─────────────────────────────────────────────────────
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    zone_start = 0
    time_inside = 0.0
    va_entry_i, va_entry_edge = -1, np.nan
    break_i, break_dir, break_lvl = -1, 0.0, np.nan
    break_max_ext, ext_at_touch, touched_back = 0.0, np.nan, False
    reclaim, reclaim_i, bounce = 0.0, -1, 0.0
    reject_flag, bounce_acc = 0.0, bounce_idle
    vwap_side, vwap_cross_i, prev_cross_i = 0.0, -1, -1
    cross_side, cross_strength, cross_count, vwap_reclaim = np.nan, np.nan, 0.0, 0.0
    touch_i, touch_kind, touch_acc, touch_rej = -1, np.nan, np.nan, np.nan

    for i in range(n):
        p = price[i]
        dt = float(ts_s[i] - ts_s[i - 1]) if i else 0.0

        if i and zone[i] != zone[i - 1]:
            zone_start = i
        if inside_va[i]:
            time_inside += dt
            if i == 0 or not inside_va[i - 1]:
                va_entry_i = i
                va_entry_edge = val if (i and price[i - 1] < val) else vah

        # A break is a cross *through* a value-area edge, in either direction.
        if i:
            if price[i - 1] <= vah and p > vah:
                break_i, break_dir, break_lvl = i, 1.0, vah
                break_max_ext, touched_back, reclaim, bounce = 0.0, False, 0.0, 0.0
                reclaim_i, reject_flag, bounce_acc = -1, 0.0, bounce_idle
            elif price[i - 1] >= val and p < val:
                break_i, break_dir, break_lvl = i, -1.0, val
                break_max_ext, touched_back, reclaim, bounce = 0.0, False, 0.0, 0.0
                reclaim_i, reject_flag, bounce_acc = -1, 0.0, bounce_idle

        if break_dir != 0.0:
            ext = (p - break_lvl) * break_dir
            if ext < 0 and reclaim == 0.0:
                reclaim, reclaim_i = 1.0, i
                # A break undone quickly is a rejection; one undone hours later
                # is a different event, which is why the window exists.
                reject_flag = float(
                    (ts_s[i] - ts_s[break_i]) <= cfg.reject_window_s)
            if 0.0 <= ext <= cfg.bounce_tol_pts and not touched_back:
                touched_back, ext_at_touch = True, break_max_ext
            if (touched_back and reclaim == 0.0 and ext_at_touch > 0
                    and ext > ext_at_touch):
                bounce = 1.0
                if accept is not None and np.isfinite(acc[i]) and acc[i] * break_dir > 0:
                    bounce_acc = 1.0
            break_max_ext = max(break_max_ext, ext)
            out["st_break_age_s"][i] = ts_s[i] - ts_s[break_i]
            out["st_break_level"][i] = break_dir
            out["st_break_extension_r"][i] = (p - break_lvl) / r_value
            out["st_break_max_extension_r"][i] = break_max_ext / r_value
            out["st_break_pullback_r"][i] = max(break_max_ext - ext, 0.0) / r_value
            out["st_reclaim_flag"][i] = reclaim
            out["st_reject_flag"][i] = reject_flag
            out["st_bounce_confirm"][i] = bounce
            out["st_bounce_accept"][i] = bounce_acc
            out["st_reclaim_age_s"][i] = (ts_s[i] - ts_s[reclaim_i]
                                          if reclaim_i >= 0 else -1.0)
        else:
            out["st_break_age_s"][i] = -1.0
            out["st_break_level"][i] = 0.0
            out["st_reclaim_flag"][i] = 0.0
            out["st_reject_flag"][i] = 0.0
            out["st_reclaim_age_s"][i] = -1.0
            out["st_bounce_confirm"][i] = 0.0
            out["st_bounce_accept"][i] = bounce_idle

        side_now = 0.0 if spread[i] == 0 else float(np.sign(spread[i]))
        if side_now != 0.0 and vwap_side != 0.0 and side_now != vwap_side:
            prev_cross_i, vwap_cross_i = vwap_cross_i, i
            cross_side, cross_strength = side_now, vwap_vel[i]
            cross_count += 1.0
            vwap_reclaim = float(
                prev_cross_i >= 0
                and (ts_s[i] - ts_s[prev_cross_i]) <= cfg.vwap_reclaim_window)
        if side_now != 0.0:
            vwap_side = side_now

        if touching[i]:
            touch_i = i
            touch_kind = LEVEL_CODES[int(near[i])]
            touch_acc, touch_rej = acc[i], rej[i]

        out["st_zone"][i] = zone[i]
        out["st_zone_dwell_s"][i] = ts_s[i] - ts_s[zone_start]
        out["st_va_time_inside_s"][i] = time_inside
        out["st_va_entry_age_s"][i] = (ts_s[i] - ts_s[va_entry_i]
                                       if va_entry_i >= 0 else -1.0)
        out["st_va_entry_depth_r"][i] = ((p - va_entry_edge) / r_value
                                         if va_entry_i >= 0 else np.nan)
        out["st_vwap_cross_age_s"][i] = (ts_s[i] - ts_s[vwap_cross_i]
                                         if vwap_cross_i >= 0 else -1.0)
        out["st_vwap_cross_side"][i] = cross_side
        out["st_vwap_cross_strength"][i] = cross_strength
        out["st_vwap_cross_count"][i] = cross_count
        out["st_vwap_reclaim_flag"][i] = vwap_reclaim
        out["st_level_touch_age_s"][i] = (ts_s[i] - ts_s[touch_i]
                                          if touch_i >= 0 else -1.0)
        out["st_level_touch_kind"][i] = touch_kind
        out["st_level_touch_accept"][i] = touch_acc
        out["st_level_touch_reject"][i] = touch_rej

    out["st_dist_prior_hi_r"] = (price - phi) / r_value
    out["st_dist_prior_lo_r"] = (price - plo) / r_value

    frame = pd.DataFrame(out, index=bars.index, columns=list(COLUMNS))
    return frame.replace([np.inf, -np.inf], np.nan).astype("float32")
