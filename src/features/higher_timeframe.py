"""Higher-timeframe context: daily trend, position-in-range, gap, ATR ratio.

Provides multi-day structural context that is constant within each intraday
session but varies day-to-day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    daily_ohlc: pd.DataFrame,
    target_date: str | pd.Timestamp,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute higher-timeframe features and broadcast to intraday index.

    Parameters
    ----------
    daily_ohlc : DataFrame with columns: date, open, high, low, close.
                 Must include at least 20 prior trading days.
    target_date : The date being processed (used to select prior days).
    index : Intraday DatetimeIndex to broadcast features onto.
    """
    features = pd.DataFrame(index=index)

    if daily_ohlc.empty or len(daily_ohlc) < 2:
        features["ht_daily_trend_10"] = np.nan
        features["ht_daily_trend_20"] = np.nan
        features["ht_daily_pos_in_range_20"] = np.nan
        features["ht_gap_pct"] = np.nan
        features["ht_daily_atr_ratio"] = np.nan
        return features.astype(np.float32)

    # Ensure sorted by date
    df = daily_ohlc.sort_values("date").copy()
    target = pd.Timestamp(target_date).date() if not isinstance(target_date, str) else pd.Timestamp(target_date).date()

    # Filter to days before target (strictly prior)
    prior = df[df["date"] < target].tail(20)
    if len(prior) < 2:
        features["ht_daily_trend_10"] = np.nan
        features["ht_daily_trend_20"] = np.nan
        features["ht_daily_pos_in_range_20"] = np.nan
        features["ht_gap_pct"] = np.nan
        features["ht_daily_atr_ratio"] = np.nan
        return features.astype(np.float32)

    closes = prior["close"].values
    highs = prior["high"].values
    lows = prior["low"].values

    # ── Trend: slope of linear regression on closes, normalized by ATR ──
    # Use nanmean so that holiday/partial-session NaN rows don't propagate.
    atr_20 = float(np.nanmean(highs - lows)) if len(highs) > 0 else 1.0
    if np.isnan(atr_20) or atr_20 < 0.01:
        atr_20 = 0.01

    def _trend_slope(closes_subset: np.ndarray) -> float:
        # Drop NaN closes (holidays, early closes) before fitting.
        valid = closes_subset[~np.isnan(closes_subset)]
        n = len(valid)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, valid, 1)[0]
        return float(slope / atr_20)

    ht_trend_10 = _trend_slope(closes[-10:]) if len(closes) >= 10 else _trend_slope(closes)
    ht_trend_20 = _trend_slope(closes)

    # ── Position in range: (yesterday_close - 20d_low) / (20d_high - 20d_low) ──
    range_high = float(np.nanmax(highs))
    range_low = float(np.nanmin(lows))
    range_width = range_high - range_low
    # Use last valid (non-NaN) close in prior window — handles holiday rows.
    valid_closes = closes[~np.isnan(closes)]
    yesterday_close = float(valid_closes[-1]) if len(valid_closes) > 0 else float("nan")
    ht_pos = float((yesterday_close - range_low) / range_width) if (range_width > 0 and not np.isnan(yesterday_close)) else 0.5

    # ── Gap: today's open vs yesterday's close ──
    today_data = df[df["date"] == target]
    if not today_data.empty:
        today_open = today_data["open"].iloc[0]
        ht_gap_pct = float((today_open - yesterday_close) / yesterday_close) if (yesterday_close != 0 and not np.isnan(yesterday_close)) else 0.0
    else:
        ht_gap_pct = 0.0

    # ── ATR ratio: 5-day ATR / 20-day ATR ──
    tr = highs - lows  # simplified true range (NaN where either is NaN)
    atr_5 = float(np.nanmean(tr[-5:])) if len(tr) >= 5 else float(np.nanmean(tr))
    if np.isnan(atr_5):
        atr_5 = atr_20
    ht_atr_ratio = float(atr_5 / atr_20)

    # Broadcast to all intraday bars
    features["ht_daily_trend_10"] = np.float32(ht_trend_10)
    features["ht_daily_trend_20"] = np.float32(ht_trend_20)
    features["ht_daily_pos_in_range_20"] = np.float32(ht_pos)
    features["ht_gap_pct"] = np.float32(ht_gap_pct)
    features["ht_daily_atr_ratio"] = np.float32(ht_atr_ratio)

    return features.astype(np.float32)


def compute_ytd(
    prior_hl: pd.DataFrame,
    candles_1m: pd.DataFrame,
    price: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """YTD session high/low vs *price*.

    Prior completed sessions come from *prior_hl* (columns ``date, high, low``).
    Today's running high/low come from 1-minute candles so the YTD extreme
    can print intraday without using future bars.
    """
    features = pd.DataFrame(index=index)
    px = price.reindex(index).astype("float64")
    ytd_high = np.nan
    ytd_low = np.nan
    if prior_hl is not None and not prior_hl.empty:
        ytd_high = float(np.nanmax(prior_hl["high"].to_numpy(dtype="float64")))
        ytd_low = float(np.nanmin(prior_hl["low"].to_numpy(dtype="float64")))
    if candles_1m is not None and not candles_1m.empty:
        c = candles_1m.copy()
        if c.index.tz is None:
            c.index = c.index.tz_localize("UTC")
        else:
            c.index = c.index.tz_convert("UTC")
        from src.features.closed_bars import shift_closed_1m
        run_h = shift_closed_1m(c["high"].astype("float64").cummax())
        run_l = shift_closed_1m(c["low"].astype("float64").cummin())
        run_h = run_h.reindex(index, method="ffill")
        run_l = run_l.reindex(index, method="ffill")
        if ytd_high == ytd_high:
            run_h = run_h.fillna(ytd_high)
            run_h = np.maximum(run_h, ytd_high)
        if ytd_low == ytd_low:
            run_l = run_l.fillna(ytd_low)
            run_l = np.minimum(run_l, ytd_low)
        yh, yl = run_h, run_l
    else:
        yh = pd.Series(ytd_high, index=index)
        yl = pd.Series(ytd_low, index=index)
    denom = px.replace(0.0, np.nan)
    features["ht_ytd_high_dist"] = (px - yh) / denom
    features["ht_ytd_low_dist"] = (px - yl) / denom
    return features.astype(np.float32)
