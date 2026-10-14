"""Hawkes event-clustering features: exponentially-weighted large-move counts.

Gap #26: Simplified proxy for self-exciting event process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    price_series: pd.Series,
    timestep: str | None = None,
    large_move_pct: float = 0.001,
) -> pd.DataFrame:
    """Compute Hawkes-style event clustering features.

    Parameters
    ----------
    price_series : Resampled close/mid price series.
    timestep : Resample interval.
    large_move_pct : Threshold for "large move" (default 0.1% = ~22 NQ points).
    """
    timestep = timestep or get_timestep()
    w5m  = bars_m(5)
    w20m = bars_m(20)

    log_ret = np.log(price_series / price_series.shift(1))
    is_large_move = (log_ret.abs() > large_move_pct).astype(np.float32)

    features = pd.DataFrame(index=price_series.index)

    # EWM count of large moves at two time scales
    features[f"hp_large_move_ewm_{w5m}"] = is_large_move.ewm(halflife=w5m).mean()
    features[f"hp_large_move_ewm_{w20m}"] = is_large_move.ewm(halflife=w20m).mean()

    # Cluster ratio: recent intensity / background intensity.
    # On quiet days no 5-s bar exceeds the threshold, so both EWMs are zero
    # and the ratio is 0/0 → NaN. Treat that case as "no clustering" (0.0)
    # rather than letting NaN propagate downstream.
    features["hp_cluster_ratio"] = (
        features[f"hp_large_move_ewm_{w5m}"]
        / features[f"hp_large_move_ewm_{w20m}"].replace(0, np.nan)
    ).fillna(0.0)

    return features.astype(np.float32)
