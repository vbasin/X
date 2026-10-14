"""Trade location features: where trades execute relative to the spread.

``tl_trade_location`` and ``tl_aggressive_pct`` are size-weighted per 5s bar
(ADR-037). Rolling MAs, z-scores, and short−long diffs are equal-weight stats
of those bar series via ``bar_windows`` (30s / 5m / 30m only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep

# Wall-clock names; bar counts from bar_windows so they track timestep.
# At 5s: 30s=6, 5m=60, 30m=360.
def _stat_windows() -> tuple[tuple[str, int], ...]:
    return (
        ("30s", bars(30)),
        ("5m", bars_m(5)),
        ("30m", bars_m(30)),
    )


_STAT_PAIRS: tuple[tuple[str, str], ...] = (
    ("30s", "5m"),
    ("5m", "30m"),
)


def _attach_bar_stats(
    out: pd.DataFrame,
    series: pd.Series,
    *,
    ma_prefix: str,
    diff_prefix: str,
    zscore_prefix: str,
) -> None:
    """Equal-weight MAs plus pair short−long and pair z-score (30s vs 5m, 5m vs 30m)."""
    ma: dict[str, pd.Series] = {}
    std: dict[str, pd.Series] = {}
    for suffix, win in _stat_windows():
        ma[suffix] = series.rolling(win, min_periods=1).mean()
        std[suffix] = series.rolling(win).std().replace(0, np.nan)
        out[f"{ma_prefix}_{suffix}"] = ma[suffix]
    for short, long in _STAT_PAIRS:
        pair = ma[short] - ma[long]
        out[f"{diff_prefix}_{short}_{long}"] = pair
        out[f"{zscore_prefix}_{short}_{long}"] = pair / std[long]


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute trade location features from TBBO data.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size, side, bid_px, ask_px.
    timestep : Resample interval.

    Bar values are ``sum(size * x) / sum(size)`` for location and sweep.
    Rolling columns are equal-weight stats of those bar series. A 30s MA is
    the last ``bars(30)`` bars (6 at 5s).
    """
    timestep = timestep or get_timestep()

    df = tbbo.copy()
    if "size" not in df.columns:
        df["size"] = 1.0
    df["size"] = df["size"].astype("float64")
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["half_spread"] = (df["ask_px"] - df["bid_px"]) / 2

    # Trade location: -1 = at bid, +1 = at ask, 0 = at mid
    df["trade_location"] = (df["price"] - df["mid"]) / df["half_spread"].replace(0, np.nan)
    df["loc_size"] = df["trade_location"] * df["size"]

    # Aggressive: trade STRICTLY past best (true sweep). At the tick level
    # almost every trade prints exactly at bid/ask (|trade_location| ~ 1.0),
    # so the previous threshold of >= 0.9 saturated to 100 %. We want the
    # fraction of size that crossed the BBO (|trade_location| > 1).
    df["aggressive"] = (
        (df["price"] > df["ask_px"]) | (df["price"] < df["bid_px"])
    ).astype(np.float64)
    df["aggr_size"] = df["aggressive"] * df["size"]

    bars_df = df.set_index("ts_event").resample(timestep).agg(
        _loc_size=("loc_size", "sum"),
        _aggr_size=("aggr_size", "sum"),
        _size=("size", "sum"),
    )
    den = bars_df["_size"].replace(0, np.nan)
    bars_df["tl_trade_location"] = bars_df["_loc_size"] / den
    bars_df["tl_aggressive_pct"] = bars_df["_aggr_size"] / den
    bars_df = bars_df.drop(columns=["_loc_size", "_aggr_size", "_size"])

    _attach_bar_stats(
        bars_df, bars_df["tl_trade_location"],
        ma_prefix="tl_loc_ma", diff_prefix="tl_loc_diff", zscore_prefix="tl_loc_zscore",
    )
    _attach_bar_stats(
        bars_df, bars_df["tl_aggressive_pct"],
        ma_prefix="tl_aggr", diff_prefix="tl_aggr_diff", zscore_prefix="tl_aggr_zscore",
    )

    return bars_df.astype(np.float32)
