"""Microstructure features: VPIN, Kyle's Lambda, and Amihud illiquidity.

v3: Uses bar_windows(). Added Amihud illiquidity ratio (Gap #10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
    vpin_bucket_size: int = 50,
    vpin_window: int = 50,
) -> pd.DataFrame:
    """Compute microstructure features from tick-level TBBO data.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size, side.
    timestep : Output resample interval.
    vpin_bucket_size : Volume per VPIN bucket.
    vpin_window : Number of buckets for VPIN rolling average.
    """
    timestep = timestep or get_timestep()
    w5m  = bars_m(5)
    w10m = bars_m(10)

    df = tbbo.sort_values("ts_event").copy()

    # ── VPIN (volume-bucketed) ──
    df["signed_volume"] = np.where(df["side"] == "B", df["size"], -df["size"])
    df["cumvol"] = df["size"].cumsum()
    df["bucket"] = (df["cumvol"] // vpin_bucket_size).astype(int)

    buckets = df.groupby("bucket").agg(
        buy_vol=("signed_volume", lambda s: s.clip(lower=0).sum()),
        sell_vol=("signed_volume", lambda s: (-s.clip(upper=0)).sum()),
        total_vol=("size", "sum"),
        ts_start=("ts_event", "first"),
        ts_end=("ts_event", "last"),
    )
    buckets["order_imbalance"] = (buckets["buy_vol"] - buckets["sell_vol"]).abs()
    buckets["vpin"] = buckets["order_imbalance"].rolling(vpin_window).mean() / vpin_bucket_size

    # Map VPIN back to time-based index
    vpin_ts = buckets.set_index("ts_end")["vpin"].sort_index()
    vpin_ts = vpin_ts[~vpin_ts.index.duplicated(keep="last")]

    # Resample to timestep
    bar_data = df.set_index("ts_event").resample(timestep).agg(
        total_volume=("size", "sum"),
        price_last=("price", "last"),
        price_first=("price", "first"),
        signed_volume_sum=("signed_volume", "sum"),
        dollar_volume=("size", "sum"),  # approximate; ideally price*size
    )
    # Better dollar volume estimate
    bar_data["dollar_volume"] = (df.set_index("ts_event")["price"] * df.set_index("ts_event")["size"]).resample(timestep).sum()

    # Forward-fill empty bars: no-tick 5s windows carry the last known price.
    # Prevents a single NaN return from propagating through rolling cov/var
    # (Amihud, Kyle's Lambda) for the entire window length.
    bar_data["price_last"] = bar_data["price_last"].ffill()
    bar_data["price_first"] = bar_data["price_first"].ffill()

    features = pd.DataFrame(index=bar_data.index)
    features["ms_vpin"] = vpin_ts.reindex(bar_data.index, method="ffill")

    # ── Kyle's Lambda (price impact) ──
    bar_data["return"] = bar_data["price_last"] / bar_data["price_first"].replace(0, np.nan) - 1
    bar_data["signed_vol_sqrt"] = np.sign(bar_data["signed_volume_sum"]) * np.sqrt(bar_data["signed_volume_sum"].abs())

    cov = bar_data["return"].rolling(w5m).cov(bar_data["signed_vol_sqrt"])
    var = bar_data["signed_vol_sqrt"].rolling(w5m).var().replace(0, np.nan)
    features[f"ms_kyle_lambda_{w5m}"] = cov / var

    # VPIN z-score
    for w in (w5m, w10m):
        features[f"ms_vpin_zscore_{w}"] = (
            (features["ms_vpin"] - features["ms_vpin"].rolling(w).mean())
            / features["ms_vpin"].rolling(w).std().replace(0, np.nan)
        )

    # ── Amihud Illiquidity (Gap #10) ──
    # |return| / dollar_volume — spikes when price moves on thin volume
    abs_ret = bar_data["return"].abs()
    dvol = bar_data["dollar_volume"].replace(0, np.nan)
    amihud_raw = abs_ret / dvol

    for w in (w5m, w10m):
        features[f"ms_amihud_{w}"] = amihud_raw.rolling(w).mean()

    features[f"ms_amihud_zscore_{w10m}"] = (
        (features[f"ms_amihud_{w5m}"] - features[f"ms_amihud_{w5m}"].rolling(w10m).mean())
        / features[f"ms_amihud_{w5m}"].rolling(w10m).std().replace(0, np.nan)
    )

    return features.astype(np.float32)
