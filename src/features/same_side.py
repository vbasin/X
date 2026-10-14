"""Same-side run features: consecutive buy/sell trade detection.

Gap #11: buy_run, sell_run, max_run.

The 5s parquet keeps last-run-at-bar-close (`ss_buy_run` / `ss_sell_run`)
and a trailing-window max. Native-TF 1-bar max runs (`ss_buy_run_max` /
`ss_sell_run_max`) reset at the bar boundary and live in
``src.features.bar_features`` — they are not on the 5s cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute consecutive same-side trade run features.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, side.
    timestep : Resample interval.
    """
    timestep = timestep or get_timestep()
    w2m = bars_m(2)

    df = tbbo.sort_values("ts_event").copy()

    # Compute run length for each trade
    is_buy = (df["side"] == "B").astype(int)
    is_sell = (df["side"] == "A").astype(int)

    # Run-length encoding: consecutive same-side
    df["side_change"] = (df["side"] != df["side"].shift(1)).astype(int)
    df["run_group"] = df["side_change"].cumsum()
    df["run_len"] = df.groupby("run_group").cumcount() + 1

    # Buy run and sell run
    df["buy_run"] = np.where(is_buy, df["run_len"], 0)
    df["sell_run"] = np.where(is_sell, df["run_len"], 0)

    # Resample: take the last run length in each bar
    bars = df.set_index("ts_event").resample(timestep).agg(
        ss_buy_run=("buy_run", "last"),
        ss_sell_run=("sell_run", "last"),
        ss_max_run=("run_len", "max"),
    )
    bars = bars.fillna(0)

    features = pd.DataFrame(index=bars.index)
    features["ss_buy_run"] = bars["ss_buy_run"]
    features["ss_sell_run"] = bars["ss_sell_run"]

    # Max run in trailing window
    features[f"ss_max_run_{w2m}"] = bars["ss_max_run"].rolling(w2m).max()

    return features.astype(np.float32)
