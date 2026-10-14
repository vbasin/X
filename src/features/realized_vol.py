"""Realized volatility and vol-of-vol features.

Gap #6: Rolling std of log returns + std of rolling vol (compression signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    price_series: pd.Series,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute realized volatility features from a price series.

    Parameters
    ----------
    price_series : Series indexed by timestamp with close/mid prices.
    timestep : Resample interval (for reference; input already resampled).
    """
    timestep = timestep or get_timestep()
    w5m  = bars_m(5)
    w10m = bars_m(10)
    w30m = bars_m(30)

    log_ret = np.log(price_series / price_series.shift(1))

    features = pd.DataFrame(index=price_series.index)

    # Realized volatility at multiple windows
    features[f"rv_realized_vol_{w5m}"] = log_ret.rolling(w5m).std()
    features[f"rv_realized_vol_{w10m}"] = log_ret.rolling(w10m).std()
    features[f"rv_realized_vol_{w30m}"] = log_ret.rolling(w30m).std()

    # Vol-of-vol: rolling std of the short-term realized vol
    rv_short = features[f"rv_realized_vol_{w5m}"]
    features[f"rv_vol_of_vol_{w10m}"] = rv_short.rolling(w10m).std()
    features[f"rv_vol_of_vol_{w30m}"] = rv_short.rolling(w30m).std()

    # Vol ratio: short / long — compression when < 1, expansion when > 1
    features["rv_vol_ratio"] = rv_short / features[f"rv_realized_vol_{w30m}"].replace(0, np.nan)

    return features.astype(np.float32)
