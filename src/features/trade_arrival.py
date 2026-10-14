"""Trade arrival rate and acceleration features.

Gap #18: Trades/second, d(rate)/dt, z-score of arrival rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute trade arrival features from TBBO data.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, size.
    timestep : Resample interval.
    """
    timestep = timestep or get_timestep()
    w5m = bars_m(5)

    df = tbbo.sort_values("ts_event").set_index("ts_event")

    # Count trades per bar, normalized to per-second rate
    bar_data = df.resample(timestep).agg(
        trade_count=("size", "count"),
    )

    import re
    ts_match = re.match(r"(\d+)s", timestep)
    bar_seconds = int(ts_match.group(1)) if ts_match else 5
    bar_data["rate"] = bar_data["trade_count"] / bar_seconds

    features = pd.DataFrame(index=bar_data.index)
    features["ta_arrival_rate"] = bar_data["rate"]

    # Acceleration: d(rate)/dt
    features["ta_arrival_accel"] = bar_data["rate"].diff()

    # Z-score of arrival rate
    features[f"ta_arrival_zscore_{w5m}"] = (
        (bar_data["rate"] - bar_data["rate"].rolling(w5m).mean())
        / bar_data["rate"].rolling(w5m).std().replace(0, np.nan)
    )

    return features.astype(np.float32)
