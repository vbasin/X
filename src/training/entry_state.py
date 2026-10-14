"""Layer 1 entry-state features — what the auction is doing at the decision bar.

The candidate descriptors in :mod:`src.training.auction_levels` describe a
*level*; these describe the *market* the level sits in. Measured separation on
the candidate descriptors alone was weak (best spread 0.057 R, all Spearman
|rho| < 0.09, ADR-123), which is the expected result: whether price reaches a
node 1.2R away is mostly a question of what the auction is doing, not of the
node's own geometry.

Everything here is **causal** — computed from rows at or before the decision bar.
Signed quantities are emitted in the absolute frame and mirrored for shorts by
:func:`src.training.first_passage._trade_frame`, so a long and a short remain the
same problem rotated.

Order-flow columns come from the native 15s hive (`of_cvd`, `of_delta`,
`of_ofi_cum`, ...) and therefore exist only from 2025-04. Sessions without hive
coverage emit NaN for those columns rather than an estimate — a CLV proxy for
aggression is uncorrelated with the truth at price (ADR-123).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.cash_session_profile import (
    Bound, LevelConfig, SessionLevels, bound_interaction_matrices,
    interactions_at, levels_for_price, next_bounds,
)
from src.features.tape import TAPE_PROFILE_TICK_SIZE, tick_value_area
from src.training.auction_levels import Profile, ProfileConfig, value_area

logger = logging.getLogger("vtech.training.entry_state")

PREFIX = "es_"

# Signed in the absolute frame; negated for a short so both sides share a model.
DIRECTIONAL: tuple[str, ...] = (
    "es_vs_dev_poc", "es_vs_dev_vah", "es_vs_dev_val", "es_profile_skew",
    "es_vs_prior_poc", "es_vs_prior_vah", "es_vs_prior_val", "es_open_vs_prior_va",
    "es_vs_prior_hi", "es_vs_prior_lo",
    "es_gap_r", "es_cvd_session", "es_cvd_slope_5m", "es_cvd_slope_15m",
    "es_delta_1m", "es_delta_5m", "es_ofi_cum", "es_cvd_price_divergence",
    "es_ret_5m", "es_ret_15m", "es_ret_60m", "es_vs_session_open",
    "es_vs_session_high", "es_vs_session_low",
    "es_vs_bound_above_r", "es_vs_bound_below_r", "es_pos_in_channel",
)


@dataclass(frozen=True)
class EntryStateConfig:
    """Lookback windows, in seconds on the 1s grid."""

    ret_windows: tuple[int, ...] = (300, 900, 3600)
    cvd_slopes: tuple[int, ...] = (300, 900)
    delta_windows: tuple[int, ...] = (60, 300)
    divergence_window: int = 900


def _safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(np.asarray(a, dtype="float64")),
                     where=np.asarray(b, dtype="float64") != 0)


def _running_value_area(tape_ticks: np.ndarray, tape_size: np.ndarray,
                        cutoff_pos: np.ndarray, pcfg: ProfileConfig) -> pd.DataFrame:
    """POC / VAH / VAL of the *developing* profile as of each decision instant.

    *tape_ticks*/*tape_size* are every print's price (rounded to the exchange
    tick) and size for the whole session, in print order; *cutoff_pos[i]* is
    the (inclusive) last print position known as of decision *i*. Volume at
    price is exact per print, the same tick-exact convention
    :func:`src.features.tape.tape_levels` uses for the prior (closed)
    session's profile — a bar's volume is no longer smeared uniformly across
    its high/low range.

    One forward pass over the tape accumulates a running per-tick-level
    volume vector, snapshotting it at each cutoff in increasing order —
    linear in the tape length plus ``entries x distinct ticks touched``,
    rather than re-sorting the whole (up to session-length) print prefix from
    scratch at every entry.
    """
    n = cutoff_pos.size
    rows = np.full((n, 4), np.nan)
    if tape_ticks.size == 0:
        return pd.DataFrame(rows, columns=["poc", "val", "vah", "vol"])

    levels, level_idx = np.unique(tape_ticks, return_inverse=True)
    order = np.argsort(cutoff_pos, kind="stable")
    acc = np.zeros(levels.size, dtype="float64")
    filled = 0
    for out_i in order:
        cut = int(cutoff_pos[out_i])
        if cut < 1:
            continue
        target = cut + 1  # exclusive end; prints [0, cut] inclusive are known
        if target > filled:
            np.add.at(acc, level_idx[filled:target], tape_size[filled:target])
            filled = target
        total = float(acc.sum())
        if total <= 0:
            continue
        poc, vah, val = tick_value_area(levels, acc, pcfg.value_area_pct)
        rows[out_i] = (poc, val, vah, total)
    return pd.DataFrame(rows, columns=["poc", "val", "vah", "vol"])


def _order_flow(grid_ts: pd.DatetimeIndex, entries: np.ndarray,
                hive: pd.DataFrame | None, cfg: EntryStateConfig,
                r_value: float) -> dict[str, np.ndarray]:
    """Session CVD, its slopes, recent delta and cumulative OFI at each entry.

    All values are as-of the decision bar (`merge_asof` backward), so a bar that
    has not closed yet cannot contribute.
    """
    n = len(entries)
    nan = np.full(n, np.nan)
    out: dict[str, np.ndarray] = {
        "es_cvd_session": nan.copy(), "es_ofi_cum": nan.copy(),
        "es_cvd_price_divergence": nan.copy(),
    }
    for w in cfg.cvd_slopes:
        out[f"es_cvd_slope_{w // 60}m"] = nan.copy()
    for w in cfg.delta_windows:
        out[f"es_delta_{w // 60}m"] = nan.copy()
    if hive is None or hive.empty:
        return out

    want = pd.DataFrame({"ts": grid_ts[entries]})
    h = hive.sort_index()
    asof = pd.merge_asof(want, h.reset_index().rename(columns={h.index.name or "index": "ts"}),
                         on="ts", direction="backward")
    vol_scale = max(1.0, float(np.nanmax(np.abs(asof.get("of_cvd", pd.Series([1.0])))) or 1.0)) 

    if "of_cvd" in asof:
        cvd = asof["of_cvd"].to_numpy(dtype="float64")
        out["es_cvd_session"] = cvd / vol_scale
        for w in cfg.cvd_slopes:
            prior = pd.merge_asof(
                pd.DataFrame({"ts": grid_ts[entries] - pd.Timedelta(seconds=w)}),
                h.reset_index().rename(columns={h.index.name or "index": "ts"}),
                on="ts", direction="backward")["of_cvd"].to_numpy(dtype="float64")
            out[f"es_cvd_slope_{w // 60}m"] = (cvd - prior) / vol_scale
        # Divergence: CVD rising while price falls (or the reverse) over the window.
        w = cfg.divergence_window
        prior_cvd = pd.merge_asof(
            pd.DataFrame({"ts": grid_ts[entries] - pd.Timedelta(seconds=w)}),
            h.reset_index().rename(columns={h.index.name or "index": "ts"}),
            on="ts", direction="backward")
        if "cs_close" in prior_cvd and "cs_close" in asof:
            d_cvd = (cvd - prior_cvd["of_cvd"].to_numpy(dtype="float64")) / vol_scale
            d_px = ((asof["cs_close"].to_numpy(dtype="float64")
                     - prior_cvd["cs_close"].to_numpy(dtype="float64"))
                    / max(r_value, 1e-9))
            out["es_cvd_price_divergence"] = d_cvd - d_px
    if "of_ofi_cum" in asof:
        o = asof["of_ofi_cum"].to_numpy(dtype="float64")
        scale = max(1.0, float(np.nanmax(np.abs(o)) or 1.0))
        out["es_ofi_cum"] = o / scale
    if "of_delta" in asof:
        bars = h["of_delta"]
        for w in cfg.delta_windows:
            roll = bars.rolling(f"{w}s").sum()
            v = pd.merge_asof(want, roll.rename("d").reset_index().rename(
                columns={roll.index.name or "index": "ts"}), on="ts",
                direction="backward")["d"].to_numpy(dtype="float64")
            out[f"es_delta_{w // 60}m"] = v / vol_scale
    return out


def _causal_1m(bucket_index: pd.DatetimeIndex, stat: pd.Series,
               ts: pd.DatetimeIndex) -> np.ndarray:
    """Hold *stat* (indexed by left-labelled 1-minute bucket) from the instant
    that bucket actually completes -- one minute after its own label -- until
    the next bucket completes, so a decision never reads a still-forming
    candle's close/range.
    """
    completes_at = bucket_index.asi8 + 60_000_000_000  # +1 minute, in ns
    pos = np.searchsorted(completes_at, ts.asi8, side="right") - 1
    vals = stat.to_numpy(dtype="float64")
    return np.where(pos >= 0, vals[np.maximum(pos, 0)], np.nan)


def _bound_state(px: np.ndarray, pos: np.ndarray, tape_px: np.ndarray,
                 level_hist: list[SessionLevels], lcfg: LevelConfig,
                 ) -> dict[str, np.ndarray]:
    """Multi-session clustered structural bounds, read the same way the single
    prior session's levels are above -- the same :mod:`cash_session_profile`
    ``Bound`` structure Stage 3 targets against, at the decision bar.

    *px* is each decision's causal current price (the caller's own ``_asof``
    resolution); *pos* is that decision's tape row index, reused so the
    interaction counts snapshot the same "at or before" cut *px* itself does.
    *tape_px* is the full session's print-order price array that
    :func:`bound_interaction_matrices` accumulates over once per bound set --
    rebuilding the bound set (and its interaction matrices) only when price
    has left the current set's whole span, exactly as the Stage 3 builders do.

    A price that has drifted far enough (months of trend) can escape even
    :func:`levels_for_price`'s own history-extension: it then keeps handing
    back a bound set, built from all ``max_lookback_sessions`` there are,
    that still does not bracket *px*. That result cannot change for any
    price further out the same side -- more history does not exist -- so
    when the lookback has already maxed out and *px* is still beyond the
    returned span, the escaped side's span is pinned to +/-inf. Otherwise
    every following row would re-trigger a rebuild, and with it the
    O(tape x bounds) interaction pass, for the rest of the session.
    """
    n = px.size
    n_bounds = np.full(n, np.nan)
    inside_bound = np.full(n, np.nan)
    inside_strength = np.full(n, np.nan)
    inside_tests = np.full(n, np.nan)
    inside_breaks = np.full(n, np.nan)
    inside_rejects = np.full(n, np.nan)
    dist_above_pts = np.full(n, np.nan)
    dist_below_pts = np.full(n, np.nan)
    channel_width_pts = np.full(n, np.nan)
    pos_in_channel = np.full(n, np.nan)

    if level_hist:
        bounds: list[Bound] = []
        lo_span = hi_span = np.nan
        mats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        max_n = min(lcfg.max_lookback_sessions, len(level_hist))
        for i in range(n):
            p = px[i]
            if not np.isfinite(p):
                continue
            if not bounds or p > hi_span or p < lo_span:
                bounds, n_used = levels_for_price(level_hist, float(p), lcfg)
                if bounds:
                    lo_span = min(b.lo for b in bounds)
                    hi_span = max(b.hi for b in bounds)
                    mats = bound_interaction_matrices(tape_px, bounds)
                    if n_used >= max_n:
                        if p > hi_span:
                            hi_span = np.inf
                        elif p < lo_span:
                            lo_span = -np.inf
                else:
                    lo_span = hi_span = np.nan
                    mats = None
            n_bounds[i] = len(bounds)
            if not bounds:
                continue
            inside, above, below = next_bounds(bounds, float(p))
            inside_bound[i] = 1.0 if inside is not None else 0.0
            inside_strength[i] = inside.strength if inside is not None else 0.0
            if mats is not None:
                snap = interactions_at(mats, bounds, int(pos[i]))
                t, br, rj = snap.get(id(inside) if inside is not None else -1,
                                     (0, 0, 0))
                inside_tests[i], inside_breaks[i], inside_rejects[i] = t, br, rj
            if above is not None:
                dist_above_pts[i] = p - above.lo
            if below is not None:
                dist_below_pts[i] = p - below.hi
            if above is not None and below is not None:
                span = above.lo - below.hi
                channel_width_pts[i] = span
                pos_in_channel[i] = ((p - below.hi) / span - 0.5) if span > 0 else 0.0

    return {
        "n_bounds": n_bounds, "inside_bound": inside_bound,
        "inside_strength": inside_strength, "inside_tests": inside_tests,
        "inside_breaks": inside_breaks, "inside_rejects": inside_rejects,
        "dist_above_pts": dist_above_pts, "dist_below_pts": dist_below_pts,
        "channel_width_pts": channel_width_pts, "pos_in_channel": pos_in_channel,
    }


def build_entry_state(
    tape: pd.DataFrame,
    grid: pd.DataFrame,
    entries: np.ndarray,
    r_value: float,
    *,
    prior_profile: Profile | None = None,
    prior_close: float | None = None,
    prior_poc: float | None = None,
    prior_vah: float | None = None,
    prior_val: float | None = None,
    prior_hi: float | None = None,
    prior_lo: float | None = None,
    hive: pd.DataFrame | None = None,
    candles_1m: pd.DataFrame | None = None,
    level_hist: list[SessionLevels] | None = None,
    lcfg: LevelConfig | None = None,
    pcfg: ProfileConfig | None = None,
    cfg: EntryStateConfig | None = None,
) -> pd.DataFrame:
    """One ``es_``-prefixed row per decision bar.

    *grid*'s index is the contiguous 1s session timeline that *entries*
    positions into (only its index is used, to resolve a decision's
    timestamp -- its OHLC columns, if any, are ignored). *tape* is the
    session's raw prints: it grounds the current price, the developing
    value-area profile and the running session high/low, none of which read
    the 1s bars frame -- a bar's own OHLC can carry a print up to ~1s after
    the bar's own timestamp (``tape_to_path`` floors to the bar's start).
    *candles_1m* is left-labelled 1-minute OHLC (e.g. from
    ``databento.tbbo_1min``), the source for ``es_atr_r``/``es_realized_vol``;
    omit it and both come back NaN. *prior_profile* is the previous session's
    volume profile and *hive* the 15s native-TF slice covering this session
    (order flow; ``None`` outside TBBO coverage). *level_hist* is the
    multi-session history :func:`src.features.cash_session_profile.
    levels_for_price` clusters into structural bounds -- the same historic
    volume-profile structure Stage 3 targets against, read here the same way
    *prior_poc*/*prior_hi*/etc. read the single prior session; omit it and the
    ``es_n_bounds``/``es_vs_bound_*``/``es_inside_bound_*`` columns come back
    NaN. Distances are normalised by *r_value* so the columns are comparable
    across eras.
    """
    pcfg = pcfg or ProfileConfig()
    cfg = cfg or EntryStateConfig()
    lcfg = lcfg or LevelConfig()
    if r_value is None or not np.isfinite(r_value) or r_value <= 0:
        raise ValueError("r_value must be positive and finite")

    decision_ts = grid.index[entries]
    out = pd.DataFrame({"ts": decision_ts})

    tape_ts = tape.index.to_numpy()
    tape_px = tape["price"].to_numpy(dtype="float64")
    tape_sz = tape["size"].to_numpy(dtype="float64")
    tape_ticks = np.round(tape_px / TAPE_PROFILE_TICK_SIZE) * TAPE_PROFILE_TICK_SIZE
    run_hi_all = np.maximum.accumulate(tape_px)
    run_lo_all = np.minimum.accumulate(tape_px)
    first_price = float(tape_px[0]) if tape_px.size else np.nan

    def _asof(ts: np.ndarray) -> np.ndarray:
        """Position of the last print at or before each timestamp, -1 if none."""
        return np.searchsorted(tape_ts, ts, side="right") - 1

    pos = _asof(decision_ts.to_numpy())
    seen = pos >= 0
    safe = np.maximum(pos, 0)
    px = np.where(seen, tape_px[safe], np.nan)
    run_hi = np.where(seen, run_hi_all[safe], np.nan)
    run_lo = np.where(seen, run_lo_all[safe], np.nan)

    # ── developing profile ───────────────────────────────────────────────────
    va = _running_value_area(tape_ticks, tape_sz, pos, pcfg)
    out["es_vs_dev_poc"] = (px - va["poc"].to_numpy()) / r_value
    out["es_vs_dev_vah"] = (px - va["vah"].to_numpy()) / r_value
    out["es_vs_dev_val"] = (px - va["val"].to_numpy()) / r_value
    width = np.maximum(va["vah"].to_numpy() - va["val"].to_numpy(), TAPE_PROFILE_TICK_SIZE)
    out["es_va_width_r"] = width / r_value
    out["es_in_value_area"] = ((px <= va["vah"].to_numpy())
                               & (px >= va["val"].to_numpy())).astype("float64")

    mid = (run_hi + run_lo) / 2.0
    rng = np.maximum(run_hi - run_lo, TAPE_PROFILE_TICK_SIZE)
    out["es_profile_skew"] = (va["poc"].to_numpy() - mid) / rng
    out["es_session_range_r"] = rng / r_value
    out["es_vs_session_open"] = (px - first_price) / r_value
    out["es_vs_session_high"] = (px - run_hi) / r_value
    out["es_vs_session_low"] = (px - run_lo) / r_value

    # ── prior session ────────────────────────────────────────────────────────
    # Explicit levels win over a Profile: the cascade passes ADR-129 tick-parity
    # levels from the session-profile SSOT, which a binned Profile cannot match.
    if prior_poc is None and prior_profile is not None:
        prior_poc, prior_val, prior_vah = value_area(prior_profile, pcfg)
    if None not in (prior_poc, prior_vah, prior_val):
        p_width = max(prior_vah - prior_val, TAPE_PROFILE_TICK_SIZE)
        out["es_vs_prior_poc"] = (px - prior_poc) / r_value
        out["es_vs_prior_vah"] = (px - prior_vah) / r_value
        out["es_vs_prior_val"] = (px - prior_val) / r_value
        out["es_open_vs_prior_va"] = (first_price - prior_poc) / p_width
    else:
        for c in ("es_vs_prior_poc", "es_vs_prior_vah", "es_vs_prior_val",
                  "es_open_vs_prior_va"):
            out[c] = np.nan
    # The prior range boundary, not just its value area: a trend leaving the
    # previous session's high behind is a different state from one leaving VAH.
    out["es_vs_prior_hi"] = ((px - prior_hi) / r_value
                             if prior_hi is not None else np.nan)
    out["es_vs_prior_lo"] = ((px - prior_lo) / r_value
                             if prior_lo is not None else np.nan)
    out["es_gap_r"] = ((first_price - prior_close) / r_value
                       if prior_close is not None else np.nan)

    # ── multi-session clustered bounds ──────────────────────────────────────
    bs = _bound_state(px, pos, tape_px, level_hist or [], lcfg)
    out["es_n_bounds"] = bs["n_bounds"]
    out["es_inside_bound"] = bs["inside_bound"]
    out["es_inside_bound_strength"] = bs["inside_strength"]
    out["es_inside_bound_tests"] = bs["inside_tests"]
    out["es_inside_bound_breaks"] = bs["inside_breaks"]
    out["es_inside_bound_rejects"] = bs["inside_rejects"]
    # A zone broken more often than it has held is weaker overhead, same
    # formula as cash_session_profile.bound_candidates' hold_ratio.
    out["es_inside_bound_hold_ratio"] = bs["inside_rejects"] / np.maximum(
        bs["inside_tests"], 1)
    out["es_channel_width_r"] = bs["channel_width_pts"] / r_value
    out["es_vs_bound_above_r"] = bs["dist_above_pts"] / r_value
    out["es_vs_bound_below_r"] = bs["dist_below_pts"] / r_value
    out["es_pos_in_channel"] = bs["pos_in_channel"]

    # ── returns ──────────────────────────────────────────────────────────────
    for w in cfg.ret_windows:
        back_pos = _asof((decision_ts - pd.Timedelta(seconds=w)).to_numpy())
        back_seen = back_pos >= 0
        back_px = np.where(back_seen, tape_px[np.maximum(back_pos, 0)], np.nan)
        out[f"es_ret_{w // 60}m"] = (px - back_px) / r_value

    # ── volatility, from the latest 5 *completed* 1-minute candles ───────────
    if candles_1m is not None and not candles_1m.empty:
        rng_1m = (candles_1m["high"].to_numpy(dtype="float64")
                 - candles_1m["low"].to_numpy(dtype="float64"))
        avg_rng5 = pd.Series(rng_1m, index=candles_1m.index).rolling(5, min_periods=1).mean()
        out["es_atr_r"] = _causal_1m(candles_1m.index, avg_rng5, decision_ts) / r_value
        logret_1m = np.log(candles_1m["close"].astype("float64").clip(lower=1e-9)).diff()
        rv5 = logret_1m.rolling(5, min_periods=2).std()
        out["es_realized_vol"] = _causal_1m(candles_1m.index, rv5, decision_ts) * np.sqrt(5.0)
    else:
        out["es_atr_r"] = np.nan
        out["es_realized_vol"] = np.nan

    # ── order flow (TBBO-gated) ──────────────────────────────────────────────
    for k, v in _order_flow(grid.index, entries, hive, cfg, r_value).items():
        out[k] = v

    # ── clock ────────────────────────────────────────────────────────────────
    ct = decision_ts.tz_convert("America/Chicago")
    mins = ct.hour * 60 + ct.minute
    out["es_min_since_open"] = (mins - (8 * 60 + 30)).astype("float64")
    out["es_min_to_close"] = ((15 * 60) - mins).astype("float64")

    return out.replace([np.inf, -np.inf], np.nan)
