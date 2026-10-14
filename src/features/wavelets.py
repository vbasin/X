"""Haar wavelet multi-scale price decomposition features.

v3: Window defaults to 128 bars (matching 5s × 128 = 10.7 min coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from src.common.bar_windows import bars_m


def compute(
    candles: pd.DataFrame,
    price_col: str = "close",
    window: int | None = None,
    scales: list[int] | None = None,
) -> pd.DataFrame:
    """Compute wavelet features from candle/1-min data.

    Parameters
    ----------
    candles : DataFrame with a column named *price_col*.
    price_col : Column containing price values.
    window : Lookback window in bars. Default: ~10.7 min worth of bars.
    scales : Wavelet detail scales to use.
    """
    if window is None:
        # Use ~10.7 min of bars, but must be a power of 2 for Haar
        raw = bars_m(10.7)
        # Round up to next power of 2
        window = 1
        while window < raw:
            window *= 2

    if scales is None:
        scales = [1, 2, 4, 8, 16, 32]

    prices = candles[price_col].values.astype(np.float64)
    n = len(prices)
    log_prices = np.log(np.where(prices > 0, prices, np.nan))

    # Number of decomposition levels needed
    max_level = int(np.log2(max(scales))) + 1

    feature_dict: dict[str, np.ndarray] = {}

    for i in range(window, n):
        segment = log_prices[i - window : i]
        if np.any(np.isnan(segment)):
            for s in scales:
                feature_dict.setdefault(f"wv_detail_{s}", []).append(np.nan)
                feature_dict.setdefault(f"wv_energy_{s}", []).append(np.nan)
            feature_dict.setdefault("wv_approx_slope", []).append(np.nan)
            continue

        coeffs = pywt.wavedec(segment, "haar", level=max_level)
        # coeffs[0] = approximation, coeffs[1:] = details (coarsest to finest)

        for s in scales:
            level_idx = int(np.log2(s)) + 1 if s > 0 else 1
            if level_idx < len(coeffs):
                detail = coeffs[level_idx]
                feature_dict.setdefault(f"wv_detail_{s}", []).append(float(detail[-1]) if len(detail) else 0.0)
                feature_dict.setdefault(f"wv_energy_{s}", []).append(float(np.sum(detail**2)))
            else:
                feature_dict.setdefault(f"wv_detail_{s}", []).append(0.0)
                feature_dict.setdefault(f"wv_energy_{s}", []).append(0.0)

        # Approximation slope (trend direction)
        # Previous version did approx[-1] - approx[-2], which collapses to
        # near-zero on a coarse 2-coeff approximation and was effectively dead.
        # Fit a linear regression across all approximation coeffs to extract a
        # real trend slope.
        approx = coeffs[0]
        if len(approx) >= 2:
            x = np.arange(len(approx), dtype=np.float64)
            slope = float(np.polyfit(x, approx, 1)[0])
            feature_dict.setdefault("wv_approx_slope", []).append(slope)
        else:
            feature_dict.setdefault("wv_approx_slope", []).append(0.0)

    # Build DataFrame aligned to candles index
    features = pd.DataFrame(feature_dict, index=candles.index[window:])
    features = features.reindex(candles.index)

    # Cross-scale momentum: ratio of fine to coarse energy
    if "wv_energy_1" in features.columns and "wv_energy_32" in features.columns:
        features["wv_fine_coarse_ratio"] = (
            features["wv_energy_1"] / features["wv_energy_32"].replace(0, np.nan)
        )

    return features.astype(np.float32)
