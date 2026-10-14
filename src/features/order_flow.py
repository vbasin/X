"""Order flow features computed from TBBO tick data.

Includes: CVD (cumulative volume delta), trade delta, absorption detection,
aggressive flow ratios, volume-bucketed flow metrics, CVD-price divergence,
and Order Flow Imbalance (OFI).

v3: Uses bar_windows() for timestep-independent windows.
    Added of_cvd_price_divergence. Removed of_delta_sum (redundant with zscore).
v3.1: Added OFI (of_ofi, of_ofi_zscore_60).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep

# Columns added by the OFI patch (used by patch_parquets.py).
OFI_COLS: list[str] = ["of_ofi", "of_ofi_zscore_60"]


def compute(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute order flow features from TBBO data resampled to *timestep*.

    Parameters
    ----------
    tbbo : DataFrame with columns: ts_event, price, size, side, bid_px, ask_px
    timestep : Resample interval (default: from bar_windows).
    """
    timestep = timestep or get_timestep()

    # Window definitions in real time
    w30s = bars(30)    # 30 seconds
    w1m  = bars_m(1)   # 1 minute
    w2m  = bars_m(2)   # 2 minutes
    w5m  = bars_m(5)   # 5 minutes
    cvd_windows = [w30s, w1m, w2m, w5m]

    df = tbbo.copy()
    df["signed_size"] = np.where(df["side"] == "B", df["size"], np.where(df["side"] == "A", -df["size"], 0))

    # Resample to timestep bars
    bar_data = df.set_index("ts_event").resample(timestep).agg(
        buy_volume=("signed_size", lambda s: s.clip(lower=0).sum()),
        sell_volume=("signed_size", lambda s: (-s.clip(upper=0)).sum()),
        delta=("signed_size", "sum"),
        trade_count=("size", "count"),
        total_volume=("size", "sum"),
        price_last=("price", "last"),
        price_first=("price", "first"),
    )

    # CVD: cumulative volume delta
    bar_data["cvd"] = bar_data["delta"].cumsum()

    features = pd.DataFrame(index=bar_data.index)
    features["of_delta"] = bar_data["delta"]
    features["of_delta_pct"] = bar_data["delta"] / bar_data["total_volume"].replace(0, np.nan)
    features["of_buy_ratio"] = bar_data["buy_volume"] / bar_data["total_volume"].replace(0, np.nan)
    features["of_trade_count"] = bar_data["trade_count"]

    # Rolling CVD windows
    for w in cvd_windows:
        features[f"of_cvd_chg_{w}"] = bar_data["cvd"].diff(w)
        features[f"of_delta_zscore_{w}"] = (
            (bar_data["delta"] - bar_data["delta"].rolling(w).mean())
            / bar_data["delta"].rolling(w).std().replace(0, np.nan)
        )

    # Volume context
    features[f"of_volume_ma_{w5m}"] = bar_data["total_volume"].rolling(w5m).mean()
    features["of_volume_ratio"] = bar_data["total_volume"] / features[f"of_volume_ma_{w5m}"].replace(0, np.nan)

    # ── CVD-Price Divergence ──
    # When CVD rises (buyers aggressive) but price doesn't follow → absorption
    price_ret = bar_data["price_last"].pct_change(w2m)
    cvd_chg = bar_data["cvd"].diff(w2m)
    # Normalize both to z-scores for comparability
    price_z = (price_ret - price_ret.rolling(w5m).mean()) / price_ret.rolling(w5m).std().replace(0, np.nan)
    cvd_z = (cvd_chg - cvd_chg.rolling(w5m).mean()) / cvd_chg.rolling(w5m).std().replace(0, np.nan)
    features["of_cvd_price_divergence"] = cvd_z - price_z

    # ── Order Flow Imbalance (OFI) ──
    ofi = _compute_ofi(tbbo, timestep)
    if ofi is not None and not ofi.empty:
        ofi = ofi.reindex(features.index)
        for c in ofi.columns:
            features[c] = ofi[c]

    return features.astype(np.float32)


def _compute_ofi(
    tbbo: pd.DataFrame,
    timestep: str,
) -> pd.DataFrame | None:
    """Tick-level OFI aggregated to *timestep* bars.

    e_n = I(bid_px >= prev_bid_px) * bid_sz
        - I(bid_px <= prev_bid_px) * prev_bid_sz
        - I(ask_px <= prev_ask_px) * ask_sz
        + I(ask_px >= prev_ask_px) * prev_ask_sz

    OFI_bar = sum(e_n) over ticks in the bar.
    """
    needed = {"bid_px", "ask_px", "bid_sz", "ask_sz"}
    if not needed.issubset(tbbo.columns):
        return None

    df = tbbo[["ts_event", "bid_px", "ask_px", "bid_sz", "ask_sz"]].copy()
    df = df.sort_values("ts_event")

    prev_bid = df["bid_px"].shift(1)
    prev_ask = df["ask_px"].shift(1)
    prev_bid_sz = df["bid_sz"].shift(1)
    prev_ask_sz = df["ask_sz"].shift(1)

    e = (
        (df["bid_px"] >= prev_bid).astype(np.float32) * df["bid_sz"]
        - (df["bid_px"] <= prev_bid).astype(np.float32) * prev_bid_sz
        - (df["ask_px"] <= prev_ask).astype(np.float32) * df["ask_sz"]
        + (df["ask_px"] >= prev_ask).astype(np.float32) * prev_ask_sz
    )

    df["_ofi_tick"] = e
    ofi_bar = df.set_index("ts_event")["_ofi_tick"].resample(timestep).sum()

    w5m = bars_m(5)
    features = pd.DataFrame(index=ofi_bar.index)
    features["of_ofi"] = ofi_bar
    roll_mean = ofi_bar.rolling(w5m).mean()
    roll_std = ofi_bar.rolling(w5m).std().replace(0, np.nan)
    features["of_ofi_zscore_60"] = (ofi_bar - roll_mean) / roll_std

    return features.astype(np.float32)


def build_ofi_patch_frame(
    tbbo: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    timestep: str | None = None,
) -> pd.DataFrame | None:
    """Compute only OFI columns, aligned to an existing parquet index."""
    timestep = timestep or get_timestep()
    ofi = _compute_ofi(tbbo, timestep)
    if ofi is None or ofi.empty:
        return None
    return ofi.reindex(target_index)
