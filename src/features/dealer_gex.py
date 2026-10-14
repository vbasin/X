"""Dealer GEX (Gamma Exposure) estimation from NQ options.

Gap #25: Estimates net dealer gamma from 0DTE + monthly options.
Requires: NQ.OPT BBO-1s, definitions, underlying prices.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common.bar_windows import bars_m, get_timestep

logger = logging.getLogger("vtech.features.dealer_gex")


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for puts and calls)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def compute(
    bbo_opts: pd.DataFrame,
    definitions: pd.DataFrame,
    underlying_prices: pd.Series,
    timestep: str | None = None,
    risk_free: float = 0.05,
    contract_multiplier: float = 20.0,
) -> pd.DataFrame:
    """Compute dealer GEX features from options data.

    Focuses on 0DTE + monthly expirations for cost efficiency.

    Parameters
    ----------
    bbo_opts : BBO-1s DataFrame for NQ options.
    definitions : Definitions DataFrame with OI if available.
    underlying_prices : Series with NQ futures mid prices.
    timestep : Resample interval.
    contract_multiplier : NQ options multiplier ($20/point).
    """
    timestep = timestep or get_timestep()
    w2m = bars_m(2)

    und = underlying_prices.resample(timestep).last().ffill()

    features = pd.DataFrame(index=und.index)
    features["gx_dealer_gex"] = np.nan
    features["gx_gex_per_strike_max"] = np.nan
    features["gx_gex_flip_dist"] = np.nan

    if bbo_opts.empty or definitions.empty:
        features[f"gx_gex_chg_{w2m}"] = np.nan
        return features.astype(np.float32)

    # Pre-merge options with definitions (once, not per-timestep)
    defs_cols = ["symbol", "strike_price", "expiration", "instrument_class"]
    merged_defs = definitions[defs_cols].drop_duplicates("symbol")
    bbo_merged = bbo_opts.merge(merged_defs, on="symbol", how="inner")
    if bbo_merged.empty:
        features[f"gx_gex_chg_{w2m}"] = np.nan
        return features.astype(np.float32)

    # Internal 30s cadence → ffill to 5s (GEX changes slowly)
    internal_step = "30s"
    internal_index = pd.date_range(und.index.min(), und.index.max(), freq=internal_step, tz=und.index.tz)
    und_internal = underlying_prices.resample(internal_step).last().ffill().reindex(internal_index).ffill()

    bbo_sorted = bbo_merged.sort_values("ts_recv")
    # Convert to tz-naive UTC for numpy searchsorted compatibility
    ts_values = bbo_sorted["ts_recv"].dt.tz_convert("UTC").dt.tz_localize(None).values
    internal_td = pd.Timedelta(internal_step)

    internal_features = pd.DataFrame(index=internal_index)
    internal_features["gx_dealer_gex"] = np.nan
    internal_features["gx_gex_per_strike_max"] = np.nan
    internal_features["gx_gex_flip_dist"] = np.nan

    for ts in internal_index:
        window_start = ts - internal_td
        ws_np = np.datetime64(window_start.tz_convert("UTC").tz_localize(None))
        ts_np = np.datetime64(ts.tz_convert("UTC").tz_localize(None))
        lo = ts_values.searchsorted(ws_np, side="right")
        hi = ts_values.searchsorted(ts_np, side="right")
        if lo >= hi:
            continue
        snap = bbo_sorted.iloc[lo:hi]

        snap_last = snap.groupby("symbol", sort=False, as_index=False).last()
        price = und_internal.get(ts, np.nan)
        if np.isnan(price) or snap_last.empty:
            continue

        price = float(price)
        dte = (snap_last["expiration"] - snap_last["ts_recv"]).dt.total_seconds().values / (365.25 * 86400)
        K = snap_last["strike_price"].values.astype(np.float64)
        # Moneyness prefilter (|log(K/S)| < 0.15 — near-money only)
        moneyness = np.log(np.where(K > 0, K, 1.0) / price)
        mask = (dte > 0) & (np.abs(moneyness) < 0.15) & (K > 0)
        if not mask.any():
            continue
        snap_f = snap_last.loc[mask].copy()
        dte_f = dte[mask]
        K_f = K[mask]

        # Include all expirations in 0 < DTE <= 90 days.
        # This window captures:
        #   - 0DTE options (dte_f ≈ 0.001–0.003 y, highest gamma per OI)
        #   - Front quarterly contract (Dec/Mar/Jun/Sep, typically 45–90 DTE) — the
        #     dominant institutional hedging chain with 100× more BBO rows than
        #     near-term weeklies.  The old "nearest_monthly = min(expiries >= 20 DTE)"
        #     logic picked thin weekly expirations (5–12 instruments) instead of the
        #     quarterly (42–77 instruments), systematically underweighting dealer gamma.
        #   - Near-term weekly chains (negligible OI; their gamma contribution is
        #     naturally small and does not distort the total).
        # 90-day ceiling excludes back-dated quarterlie (168+ DTE from early dates)
        # while keeping the front quarterly in view even for late-Dec sessions
        # (Mar quarterly is ~79–88 DTE from Dec 22–31).
        max_dte = 90.0 / 365.25
        sel_mask = (dte_f > 0) & (dte_f <= max_dte)
        if not sel_mask.any():
            continue

        selected = snap_f.loc[sel_mask]
        dte_s = dte_f[sel_mask]
        K_s = K_f[sel_mask]
        iv_const = 0.30  # Simple IV proxy

        # Vectorized Black-Scholes gamma
        sqrt_T = np.sqrt(np.where(dte_s > 0, dte_s, 1.0))
        d1 = (np.log(price / np.where(K_s > 0, K_s, 1.0)) + (risk_free + 0.5 * iv_const**2) * dte_s) / (iv_const * sqrt_T)
        gamma = norm.pdf(d1) / (price * iv_const * sqrt_T)

        # OI proxy (bid_sz + ask_sz)
        oi = (selected["bid_sz"] + selected["ask_sz"]).values

        gex = gamma * oi * contract_multiplier * price * 0.01
        is_call = (selected["instrument_class"].values == "C")
        dealer_gex = np.where(is_call, -gex, gex)

        total_gex = float(np.nansum(dealer_gex))
        strike_gex = pd.Series(dealer_gex, index=K_s).groupby(level=0).sum().sort_index()
        max_strike_gex = float(strike_gex.abs().max()) if len(strike_gex) else np.nan

        if len(strike_gex) > 1:
            signs = np.sign(strike_gex.values)
            flips = np.where(np.diff(signs) != 0)[0]
            if len(flips) > 0:
                flip_strike = strike_gex.index[flips[0]]
                gex_flip_dist = (price - flip_strike) / price
            else:
                gex_flip_dist = np.nan
        else:
            gex_flip_dist = np.nan

        internal_features.loc[ts, "gx_dealer_gex"] = total_gex
        internal_features.loc[ts, "gx_gex_per_strike_max"] = max_strike_gex
        internal_features.loc[ts, "gx_gex_flip_dist"] = gex_flip_dist

    # Upsample 30s internal → 5s output
    features = internal_features.reindex(und.index, method="ffill")

    # GEX rate of change
    features[f"gx_gex_chg_{w2m}"] = features["gx_dealer_gex"].diff(w2m)

    return features.astype(np.float32)
