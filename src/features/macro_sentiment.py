"""Macro-sentiment features derived from VIXY (VIX short-term futures ETF).

Captures volatility regime, fear spikes, and mean-reversion signals that
influence NQ directional moves.  Produces features prefixed ``mx_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    nq_bars: pd.DataFrame,
    vixy_bars: pd.DataFrame,
    timestep: str = "10s",
) -> pd.DataFrame:
    """Compute macro-sentiment features from VIXY relative to NQ.

    Parameters
    ----------
    nq_bars : Resampled NQ price bars with 'close' column.
    vixy_bars : Resampled VIXY price bars with 'close' column.
    timestep : Resolution (for reference; inputs already resampled).

    Returns features prefixed ``mx_``.
    """
    common = nq_bars.index.intersection(vixy_bars.index)
    if len(common) < 30:
        return pd.DataFrame(index=nq_bars.index)

    nq = nq_bars.reindex(common)["close"]
    vixy = vixy_bars.reindex(common)["close"]

    features = pd.DataFrame(index=common)

    nq_ret = nq.pct_change()
    vixy_ret = vixy.pct_change()

    # ── VIXY level & momentum ──
    features["mx_vixy_ret"] = vixy_ret
    features["mx_vixy_mom_30"] = vixy.pct_change(30)
    features["mx_vixy_mom_60"] = vixy.pct_change(60)

    # ── VIXY z-score (mean-reversion signal) ──
    for w in (30, 60, 180):
        mu = vixy.rolling(w).mean()
        sigma = vixy.rolling(w).std().replace(0, np.nan)
        features[f"mx_vixy_zscore_{w}"] = (vixy - mu) / sigma

    # ── NQ/VIXY inverse correlation ──
    for w in (30, 60):
        features[f"mx_nq_vixy_corr_{w}"] = nq_ret.rolling(w).corr(vixy_ret)

    # ── Volatility regime: VIXY rate of change acceleration ──
    vixy_roc = vixy.pct_change(10)
    features["mx_vixy_accel"] = vixy_roc.diff()

    # ── Fear spike: VIXY jump vs trailing average ──
    vixy_ma60 = vixy.rolling(60).mean()
    features["mx_vixy_spike"] = (vixy / vixy_ma60.replace(0, np.nan)) - 1.0

    return features.astype(np.float32)
