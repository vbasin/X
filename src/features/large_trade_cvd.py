"""Large-trade CVD features: institutional flow isolation.

Gap #12: CVD filtered to trades >= threshold contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep
from src.common.market_session import market_session_date


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
    size_threshold: int = 10,
) -> pd.DataFrame:
    """Compute large-trade CVD features from TBBO data.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size, side.
    timestep : Resample interval.
    size_threshold : Minimum contract count to qualify as "large".
    """
    timestep = timestep or get_timestep()
    w2m  = bars_m(2)
    w10m = bars_m(10)

    df = tbbo.copy()
    df["signed_size"] = np.where(df["side"] == "B", df["size"], np.where(df["side"] == "A", -df["size"], 0))

    # Filter to large trades only
    large = df[df["size"] >= size_threshold].copy()

    # Also compute total CVD for divergence
    all_bars = df.set_index("ts_event").resample(timestep).agg(
        total_delta=("signed_size", "sum"),
    )
    _idx_all = all_bars.index if all_bars.index.tz is not None else all_bars.index.tz_localize("UTC")
    all_bars["session"] = _idx_all.map(market_session_date)
    all_bars["total_cvd"] = all_bars.groupby("session")["total_delta"].cumsum()

    if large.empty:
        features = pd.DataFrame(index=all_bars.index)
        features["lt_cvd"] = 0.0
        features[f"lt_cvd_chg_{w2m}"] = 0.0
        features[f"lt_cvd_chg_{w10m}"] = 0.0
        features["lt_cvd_divergence"] = 0.0
        return features.astype(np.float32)

    # Resample large trades
    large_bars = large.set_index("ts_event").resample(timestep).agg(
        lt_delta=("signed_size", "sum"),
    )
    _idx_large = large_bars.index if large_bars.index.tz is not None else large_bars.index.tz_localize("UTC")
    large_bars["session"] = _idx_large.map(market_session_date)
    large_bars["lt_cvd"] = large_bars.groupby("session")["lt_delta"].cumsum()

    # Align to all-bars index
    large_bars = large_bars.reindex(all_bars.index).fillna(method="ffill").fillna(0)

    features = pd.DataFrame(index=all_bars.index)
    features["lt_cvd"] = large_bars["lt_cvd"]
    features[f"lt_cvd_chg_{w2m}"] = large_bars["lt_cvd"].diff(w2m)
    features[f"lt_cvd_chg_{w10m}"] = large_bars["lt_cvd"].diff(w10m)

    # Divergence: sign disagreement between large-trade and total CVD
    lt_sign = np.sign(large_bars["lt_cvd"].diff(w2m))
    total_sign = np.sign(all_bars["total_cvd"].diff(w2m))
    features["lt_cvd_divergence"] = (lt_sign != total_sign).astype(np.float32) * lt_sign

    return features.astype(np.float32)
