"""Sub-bar dynamics: intra-bar statistics from raw BBO-1s and TBBO data.

Gap #14: Captures within-5s-bar behavior that is lost in aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import get_timestep


def compute(
    bbo: pd.DataFrame,
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute sub-bar dynamics features.

    Parameters
    ----------
    bbo : BBO-1s DataFrame with columns: ts_recv, bid_px, ask_px, bid_sz, ask_sz.
    tbbo : TBBO DataFrame with columns: ts_event, price, size, side.
    timestep : Resample interval (default: from bar_windows).
    """
    timestep = timestep or get_timestep()

    features = pd.DataFrame()

    # ── From TBBO: tick-level within-bar stats ──
    if not tbbo.empty:
        df_t = tbbo.set_index("ts_event").sort_index()

        tick_bars = df_t.resample(timestep).agg(
            sd_tick_count=("size", "count"),
            price_high=("price", "max"),
            price_low=("price", "min"),
            volume_total=("size", "sum"),
        )

        features["sd_tick_count"] = tick_bars["sd_tick_count"]

        # Price range within bar / tick count → micro impact per tick
        bar_range = tick_bars["price_high"] - tick_bars["price_low"]
        features["sd_price_impact_per_tick"] = bar_range / tick_bars["sd_tick_count"].replace(0, np.nan)

        # Volume front-loading: fraction of volume in first half of bar
        # Approximate by splitting each bar in half
        import re
        ts_match = re.match(r"(\d+)s", timestep)
        half_secs = int(ts_match.group(1)) // 2 if ts_match else 2
        half_ts = f"{half_secs}s"

        first_half = df_t.resample(timestep).apply(
            lambda x: x.iloc[: len(x) // 2]["size"].sum() if len(x) > 1 else x["size"].sum()
        )
        total_vol = df_t["size"].resample(timestep).sum().replace(0, np.nan)
        features["sd_volume_front_load"] = first_half / total_vol

    # ── From BBO-1s: quote-level within-bar stats ──
    if not bbo.empty:
        df_b = bbo.set_index("ts_recv").sort_index()

        df_b["spread"] = df_b["ask_px"] - df_b["bid_px"]
        df_b["mid"] = (df_b["ask_px"] + df_b["bid_px"]) / 2
        df_b["imbalance"] = (df_b["bid_sz"] - df_b["ask_sz"]) / (df_b["bid_sz"] + df_b["ask_sz"]).replace(0, np.nan)

        bbo_bars = df_b.resample(timestep).agg(
            bid_sz_first=("bid_sz", "first"),
            bid_sz_last=("bid_sz", "last"),
            ask_sz_first=("ask_sz", "first"),
            ask_sz_last=("ask_sz", "last"),
            spread_max=("spread", "max"),
            spread_min=("spread", "min"),
            mid_std=("mid", "std"),
            imbalance_first=("imbalance", "first"),
            imbalance_last=("imbalance", "last"),
        )

        features["sd_bid_sz_delta"] = bbo_bars["bid_sz_last"] - bbo_bars["bid_sz_first"]
        features["sd_ask_sz_delta"] = bbo_bars["ask_sz_last"] - bbo_bars["ask_sz_first"]
        features["sd_spread_range"] = bbo_bars["spread_max"] - bbo_bars["spread_min"]
        features["sd_mid_micro_vol"] = bbo_bars["mid_std"]
        features["sd_imbalance_trajectory"] = bbo_bars["imbalance_last"] - bbo_bars["imbalance_first"]

    return features.astype(np.float32)
