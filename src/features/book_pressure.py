"""Book pressure features from BBO-1s data.

Includes: bid/ask size imbalance, spread dynamics, depth change velocity,
liquidity withdrawal signals, spread volatility, imbalance velocity,
and fleeting liquidity detection.

v3: Uses bar_windows() for timestep-independent window sizing.
    Added spread volatility, imbalance/depth velocity, fleeting liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep


def compute(
    bbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute book pressure features from 1-second BBO snapshots.

    Parameters
    ----------
    bbo : DataFrame with columns: ts_recv, bid_px, ask_px, bid_sz, ask_sz
    timestep : Resample interval (default: from bar_windows).
    """
    timestep = timestep or get_timestep()
    df = bbo.set_index("ts_recv").sort_index()

    # Per-second metrics
    df["spread"] = df["ask_px"] - df["bid_px"]
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    df["imbalance"] = (df["bid_sz"] - df["ask_sz"]) / (df["bid_sz"] + df["ask_sz"]).replace(0, np.nan)
    df["depth_total"] = df["bid_sz"] + df["ask_sz"]

    # Resample to timestep
    bars_df = df.resample(timestep).agg(
        spread_mean=("spread", "mean"),
        spread_max=("spread", "max"),
        spread_std=("spread", "std"),
        imbalance_mean=("imbalance", "mean"),
        imbalance_last=("imbalance", "last"),
        imbalance_first=("imbalance", "first"),
        depth_total_mean=("depth_total", "mean"),
        depth_total_min=("depth_total", "min"),
        bid_sz_mean=("bid_sz", "mean"),
        ask_sz_mean=("ask_sz", "mean"),
        bid_sz_std=("bid_sz", "std"),
        ask_sz_std=("ask_sz", "std"),
        mid_last=("mid", "last"),
    )
    # Forward-fill empty quote bars: a 1s window with no BBO update carries
    # the last known spread/depth.  Without ffill, a single missing bar causes
    # rolling(w).std() (default min_periods=w) to propagate NaN for the full
    # window length across bp_spread_zscore_60/120 and related columns.
    bars_df = bars_df.ffill()

    features = pd.DataFrame(index=bars_df.index)

    # ── Spread features ──
    features["bp_spread_mean"] = bars_df["spread_mean"]
    features["bp_spread_max"] = bars_df["spread_max"]

    w1 = bars(60)    # 1 min
    w5 = bars_m(5)   # 5 min
    w10 = bars_m(10) # 10 min
    for w in (w1, w5, w10):
        features[f"bp_spread_zscore_{w}"] = (
            (bars_df["spread_mean"] - bars_df["spread_mean"].rolling(w).mean())
            / bars_df["spread_mean"].rolling(w).std().replace(0, np.nan)
        )

    # ── Spread volatility (Gap #21) ──
    for w in (w1, w5):
        features[f"bp_spread_vol_{w}"] = bars_df["spread_mean"].rolling(w).std()

    # ── Imbalance features ──
    features["bp_imbalance"] = bars_df["imbalance_last"]
    features[f"bp_imbalance_ma{w1}"] = bars_df["imbalance_mean"].rolling(w1).mean()

    # ── Imbalance velocity (Gap #22) — d(imbalance)/dt ──
    for w in (w1, w5):
        features[f"bp_imbalance_velocity_{w}"] = bars_df["imbalance_mean"].diff(w)

    # ── Depth features — liquidity withdrawal detection ──
    features["bp_depth_total"] = bars_df["depth_total_mean"]
    features["bp_depth_min"] = bars_df["depth_total_min"]
    for w in (w1, w5):
        depth_ma = bars_df["depth_total_mean"].rolling(w).mean()
        features[f"bp_depth_chg_{w}"] = bars_df["depth_total_mean"] / depth_ma.replace(0, np.nan) - 1
        features[f"bp_depth_zscore_{w}"] = (
            (bars_df["depth_total_mean"] - depth_ma)
            / bars_df["depth_total_mean"].rolling(w).std().replace(0, np.nan)
        )

    # ── Depth velocity (Gap #22) — d(depth)/dt ──
    features[f"bp_depth_velocity_{w1}"] = bars_df["depth_total_mean"].diff(w1)

    # ── Bid/ask size ratio ──
    features["bp_bid_ask_ratio"] = bars_df["bid_sz_mean"] / bars_df["ask_sz_mean"].replace(0, np.nan)

    # ── Fleeting liquidity detection (Gap #9) ──
    # Coefficient of variation of bid/ask sz — high CV = flickering quotes
    for w in (w1, w5):
        bid_mean = bars_df["bid_sz_mean"].rolling(w).mean()
        bid_std = bars_df["bid_sz_std"].rolling(w).mean()  # mean of per-bar stds
        ask_mean = bars_df["ask_sz_mean"].rolling(w).mean()
        ask_std = bars_df["ask_sz_std"].rolling(w).mean()
        features[f"bp_fleeting_bid_cv_{w}"] = bid_std / bid_mean.replace(0, np.nan)
        features[f"bp_fleeting_ask_cv_{w}"] = ask_std / ask_mean.replace(0, np.nan)

    # Composite fleeting score: high CV + declining mean
    bid_cv = features.get(f"bp_fleeting_bid_cv_{w1}", pd.Series(0, index=features.index))
    bid_chg = bars_df["bid_sz_mean"].diff(w1) / bars_df["bid_sz_mean"].shift(w1).replace(0, np.nan)
    features["bp_fleeting_score"] = bid_cv - bid_chg  # high CV and declining size → high score

    return features.astype(np.float32)
