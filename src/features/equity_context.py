"""Equity-context features from NVDA, TSLA, XLK, SMH.

Captures sector rotation, mega-cap momentum, and tech-leadership signals
that lead or coincide with NQ futures moves.  Produces features prefixed ``eq_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    nq_bars: pd.DataFrame,
    equity_dict: dict[str, pd.DataFrame],
    timestep: str = "10s",
) -> pd.DataFrame:
    """Compute equity-context features.

    Parameters
    ----------
    nq_bars : Resampled NQ price bars with 'close' column.
    equity_dict : Mapping of symbol -> resampled DataFrame with 'close' column.
                  Expected keys: NVDA, TSLA, XLK, SMH (any subset accepted).
    timestep : Resolution (inputs already resampled).

    Returns features prefixed ``eq_``.
    """
    features = pd.DataFrame(index=nq_bars.index)
    nq_ret = nq_bars["close"].pct_change()

    for sym, eq_bars in equity_dict.items():
        common = nq_bars.index.intersection(eq_bars.index)
        if len(common) < 30:
            continue

        eq = eq_bars.reindex(common)["close"]
        eq_ret = eq.pct_change()

        tag = sym.lower()

        # ── Momentum ──
        features[f"eq_{tag}_mom_30"] = eq.pct_change(30).reindex(nq_bars.index)
        features[f"eq_{tag}_mom_60"] = eq.pct_change(60).reindex(nq_bars.index)

        # ── Lead-lag: equity return predicting NQ ──
        features[f"eq_{tag}_ret_lag1"] = eq_ret.shift(1).reindex(nq_bars.index)

        # ── NQ correlation (convergence/divergence) ──
        nq_aligned = nq_ret.reindex(common)
        features[f"eq_{tag}_corr_60"] = nq_aligned.rolling(60).corr(eq_ret).reindex(nq_bars.index)

        # ── Relative strength vs NQ ──
        nq_cum = nq_aligned.rolling(60).sum()
        eq_cum = eq_ret.rolling(60).sum()
        features[f"eq_{tag}_rel_str_60"] = (nq_cum - eq_cum).reindex(nq_bars.index)

    # ── Sector features (if both ETFs present) ──
    if "XLK" in equity_dict and "SMH" in equity_dict:
        xlk_common = nq_bars.index.intersection(equity_dict["XLK"].index)
        smh_common = nq_bars.index.intersection(equity_dict["SMH"].index)
        both_common = xlk_common.intersection(smh_common)
        if len(both_common) >= 30:
            xlk = equity_dict["XLK"].reindex(both_common)["close"]
            smh = equity_dict["SMH"].reindex(both_common)["close"]
            # SMH/XLK ratio: semi vs broad tech rotation
            ratio = smh / xlk.replace(0, np.nan)
            features["eq_smh_xlk_ratio"] = ratio.reindex(nq_bars.index)
            mu = ratio.rolling(60).mean()
            sigma = ratio.rolling(60).std().replace(0, np.nan)
            features["eq_smh_xlk_zscore_60"] = ((ratio - mu) / sigma).reindex(nq_bars.index)

    # ── Breadth: average equity return vs NQ (market leadership) ──
    ret_list = []
    for sym, eq_bars in equity_dict.items():
        if sym in ("XLK", "SMH"):  # use individual stocks for breadth
            continue
        common = nq_bars.index.intersection(eq_bars.index)
        if len(common) >= 30:
            ret_list.append(eq_bars.reindex(common)["close"].pct_change())
    if ret_list:
        avg_eq_ret = pd.concat(ret_list, axis=1).mean(axis=1)
        features["eq_breadth_vs_nq"] = (nq_ret.reindex(avg_eq_ret.index) - avg_eq_ret).reindex(nq_bars.index)

    return features.astype(np.float32)
