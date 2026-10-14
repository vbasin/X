"""Time context features: session blocks, minutes into session, opening range.

v3: Uses bar_windows(). Added 30-minute opening range (Gap #20).
v4: Fixed market_open reference to 08:30 CST (NYSE open) — previously all
    open-anchored features were incorrectly computed from 08:00 (pre-market).
    tc_minutes_into_session now reads -30→0 during 08:00-08:30 pre-market.
    tc_session_progress denominator corrected to 150 min (08:30-11:00).
v5: Removed low-IG cyclical encodings (tc_sin_time, tc_cos_time, tc_sin_dow,
    tc_cos_dow) and coarse blocks (tc_block_open_15m, tc_block_morning).
    Replaced with granular hourly blocks: premkt_30m, open_30m, 9am_60m,
    10am_60m, 11am_60m, 12pm_60m, 1pm_60m, 2pm_60m.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    index: pd.DatetimeIndex,
    candles: pd.DataFrame | None = None,
    close_series: pd.Series | None = None,
    training_start: str = "08:00",
    training_end: str = "11:00",
    market_open: str = "08:30",
    timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """Compute time context features.

    Parameters
    ----------
    index : DatetimeIndex at the output timestep resolution.
    candles : Optional OHLCV data for opening-range features.
    close_series : Price series aligned to *index* for breakout signals.
    training_start, training_end : Data/training-window boundaries (local time).
        The session filter starts at 08:00 to capture pre-market; this should
        NOT be used as the "market open" reference.
    market_open : CME RTH open time (local).  Defaults to "08:30" (NYSE open).
        Used as the zero-point for minutes_into_session, session blocks, and
        opening-range calculations.  Bars before market_open are pre-market
        and will have negative tc_minutes_into_session values.
    timezone : Timezone for session-block encoding.
    """
    ts = index.tz_convert(timezone) if index.tz is not None else index.tz_localize("UTC").tz_convert(timezone)
    features = pd.DataFrame(index=index)

    # Minutes since market open (08:30 CST).  Pre-market bars give negative values.
    open_h, open_m = (int(x) for x in market_open.split(":"))
    market_open_minutes = open_h * 60 + open_m
    bar_minutes = ts.hour * 60 + ts.minute
    features["tc_minutes_into_session"] = (bar_minutes - market_open_minutes).astype(np.float32)

    # Day of week (0=Mon, 4=Fri)
    features["tc_dow"] = ts.dayofweek.astype(np.float32)

    # Session block one-hot.  Boundaries use hardcoded CST whole-hour marks so
    # they remain correct if training_start/training_end is later extended.
    # premkt_30m is anchored to training_start (08:00) → market_open (08:30).
    start_h, start_m = (int(x) for x in training_start.split(":"))
    training_start_minutes = start_h * 60 + start_m
    blocks = {
        "premkt_30m": (training_start_minutes, market_open_minutes),   # 08:00–08:30
        "open_5m":    (market_open_minutes, market_open_minutes + 5),  # 08:30–08:35
        "open_30m":   (market_open_minutes, market_open_minutes + 30), # 08:30–09:00
        "9am_60m":    (9 * 60, 10 * 60),                               # 09:00–10:00
        "10am_60m":   (10 * 60, 11 * 60),                              # 10:00–11:00
        "11am_60m":   (11 * 60, 12 * 60),                              # 11:00–12:00
        "12pm_60m":   (12 * 60, 13 * 60),                              # 12:00–13:00
        "1pm_60m":    (13 * 60, 14 * 60),                              # 13:00–14:00
        "2pm_60m":    (14 * 60, 15 * 60),                              # 14:00–15:00
    }
    for name, (start, end) in blocks.items():
        features[f"tc_block_{name}"] = ((bar_minutes >= start) & (bar_minutes < end)).astype(np.float32)

    # Opening range: running high/low of closed 1-min candles (ADR-041).
    if candles is not None and "high" in candles.columns:
        from src.features.closed_bars import opening_range_running
        or_df = opening_range_running(
            candles, index,
            market_open=market_open, timezone=timezone,
        )
        for col in or_df.columns:
            features[col] = or_df[col].reindex(index)

        # Breakout signals require a price series
        if close_series is not None:
            px = close_series.reindex(index)
            for label in ("5", "15", "30"):
                h_col = f"tc_or{label}_high"
                l_col = f"tc_or{label}_low"
                if h_col in features.columns:
                    features[f"tc_or{label}_breakout_up"] = (px > features[h_col]).astype(np.float32)
                    features[f"tc_or{label}_breakout_down"] = (px < features[l_col]).astype(np.float32)

    # Session progress — fraction of RTH session elapsed (0 at market open, 1 at close).
    # Denominator is market_open → training_end (08:30–11:00 = 150 min).
    # Pre-market bars (< 08:30) are clipped to 0.
    end_h, end_m = (int(x) for x in training_end.split(":"))
    session_len = (end_h * 60 + end_m) - market_open_minutes
    if session_len > 0:
        features["tc_session_progress"] = (
            features["tc_minutes_into_session"].clip(0, session_len) / session_len
        ).astype(np.float32)

    return features
