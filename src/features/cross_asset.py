"""Cross-asset features: NQ/ES correlation, lead-lag, spread, and beta.

v3: Uses bar_windows() for timestep-independent window sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars_m, get_timestep


def compute(
    nq_bars: pd.DataFrame,
    es_bars: pd.DataFrame,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Compute cross-asset features between NQ and ES futures.

    Parameters
    ----------
    nq_bars : Resampled NQ price bars with 'close' column.
    es_bars : Resampled ES price bars with 'close' column.

    Returns 12 features prefixed ``ca_``.
    """
    timestep = timestep or get_timestep()
    w5m  = bars_m(5)
    w10m = bars_m(10)
    w30m = bars_m(30)

    # Align on common timestamps
    common = nq_bars.index.intersection(es_bars.index)
    if len(common) < w5m:
        return pd.DataFrame(index=nq_bars.index)

    nq = nq_bars.reindex(common)["close"]
    es = es_bars.reindex(common)["close"]

    features = pd.DataFrame(index=common)

    # Returns
    nq_ret = nq.pct_change()
    es_ret = es.pct_change()

    # ── Correlation (multi-window) ──
    for w in (w5m, w10m, w30m):
        features[f"ca_corr_{w}"] = nq_ret.rolling(w).corr(es_ret)

    # ── NQ/ES ratio and z-score ──
    ratio = nq / es.replace(0, np.nan)
    features["ca_ratio"] = ratio
    features[f"ca_ratio_zscore_{w10m}"] = (
        (ratio - ratio.rolling(w10m).mean()) / ratio.rolling(w10m).std().replace(0, np.nan)
    )

    # ── Lead-lag: ES return predicting NQ ──
    features["ca_es_ret_lag1"] = es_ret.shift(1)
    features["ca_es_ret_lag2"] = es_ret.shift(2)

    # ── Divergence: NQ return minus ES return ──
    features["ca_ret_divergence"] = nq_ret - es_ret

    # ── Rolling beta (NQ regressed on ES) ──
    for w in (w10m, w30m):
        cov = nq_ret.rolling(w).cov(es_ret)
        var = es_ret.rolling(w).var()
        features[f"ca_beta_{w}"] = cov / var.replace(0, np.nan)

    # ── Relative strength: NQ cumulative return minus ES ──
    features[f"ca_rel_strength_{w10m}"] = nq_ret.rolling(w10m).sum() - es_ret.rolling(w10m).sum()

    # ── ES momentum ──
    features[f"ca_es_mom_{w5m}"] = es.pct_change(w5m)

    return features.astype(np.float32)
