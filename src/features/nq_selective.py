"""Causal, session-local NQ state for a selective five-minute decision policy.

The input is an ordered print tape with its contemporaneous BBO. Minute bars
are left-closed and become available at their right edge. No label, future
coverage check, or fitted transform belongs in this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.causal import as_of_index


def build_features(tape: pd.DataFrame) -> pd.DataFrame:
    """Return eligible five-minute decisions with ``nq_`` features and metadata.

    ``tape`` must contain one Chicago calendar date, have a UTC DatetimeIndex,
    and carry price/size/side/bid_px/ask_px/bid_sz/ask_sz. Tied timestamps retain
    input exchange order; the last tied quote is the decision-time quote.
    Stops use twice trailing 20-minute mean true range, rounded up to a tick,
    with an eight-point floor and a sixty-point eligibility ceiling.
    """
    required = {"price", "size", "side", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    if not required.issubset(tape.columns):
        raise ValueError(f"missing tape columns: {sorted(required - set(tape.columns))}")
    if not isinstance(tape.index, pd.DatetimeIndex) or tape.index.tz is None:
        raise ValueError("tape requires a timezone-aware DatetimeIndex")
    if tape.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    if not tape.index.is_monotonic_increasing:
        raise ValueError("tape must preserve ascending exchange timestamp order")
    local = tape.index.tz_convert("America/Chicago")
    if len(local.normalize().unique()) != 1:
        raise ValueError("build_features accepts one Chicago calendar date")
    t = tape.copy()
    t.index = t.index.tz_convert("UTC")
    for column in required:
        t[column] = pd.to_numeric(t[column], errors="coerce")
    if not np.isfinite(t["price"]).all() or (t["price"] <= 0).any():
        raise ValueError("trade prices must be finite and positive")
    t["signed_size"] = t["size"] * t["side"]
    t["notional"] = t["price"] * t["size"]
    g = t.resample("1min", closed="left", label="right")
    bars = g.agg(
        open=("price", "first"), high=("price", "max"),
        low=("price", "min"), close=("price", "last"),
        volume=("size", "sum"), delta=("signed_size", "sum"),
        notional=("notional", "sum"),
    )
    # A prefix ending inside a minute cannot make that minute available early.
    bars = bars.loc[bars.index <= t.index[-1]].copy()
    if bars.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    bars["close"] = bars["close"].ffill()
    for column in ("open", "high", "low"):
        bars[column] = bars[column].fillna(bars["close"])
    close = bars["close"]
    previous = close.shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - previous).abs(),
         (bars["low"] - previous).abs()], axis=1,
    ).max(axis=1)
    mean_tr = tr.rolling(20, min_periods=20).mean()
    stop = (np.ceil(2.0 * mean_tr / 0.25) * 0.25).clip(lower=8.0)
    change = close.diff()
    f = pd.DataFrame(index=bars.index)
    for minutes in (1, 5, 15, 30):
        f[f"nq_return_{minutes}m"] = close.diff(minutes) / stop
    for minutes in (5, 20):
        f[f"nq_true_range_{minutes}m"] = tr.rolling(minutes).mean() / stop
        f[f"nq_volatility_{minutes}m"] = change.rolling(minutes).std() / stop
    for minutes in (5, 15, 30):
        travel = change.abs().rolling(minutes).sum().replace(0, np.nan)
        f[f"nq_efficiency_{minutes}m"] = (close.diff(minutes) / travel).fillna(0)
    fast = close.ewm(span=5, adjust=False).mean()
    slow = close.ewm(span=20, adjust=False).mean()
    f["nq_ema_distance"] = (close - slow) / stop
    f["nq_ema_trend"] = (fast - slow) / stop
    f["nq_ema_slope_5m"] = slow.diff(5) / stop
    f["nq_bar_body"] = (close - bars["open"]) / stop
    f["nq_bar_close_location"] = (
        (close - bars["low"]) / (bars["high"] - bars["low"]).replace(0, np.nan)
    ).fillna(0.5)

    day = local[0].normalize()
    cash_open = (day + pd.Timedelta(hours=8, minutes=30)).tz_convert("UTC")
    opening_close = cash_open + pd.Timedelta(minutes=15)
    cash = bars.index > cash_open
    rth_volume = bars["volume"].where(cash, 0).cumsum()
    vwap = bars["notional"].where(cash, 0).cumsum() / rth_volume.replace(0, np.nan)
    rth_high = bars["high"].where(cash).cummax()
    rth_low = bars["low"].where(cash).cummin()
    rth_range = (rth_high - rth_low).replace(0, np.nan)
    f["nq_vwap_distance"] = (close - vwap) / stop
    f["nq_vwap_slope_5m"] = vwap.diff(5) / stop
    f["nq_running_range"] = rth_range / stop
    f["nq_running_position"] = (close - rth_low) / rth_range
    f["nq_running_high_distance"] = (close - rth_high) / stop
    f["nq_running_low_distance"] = (close - rth_low) / stop
    # This fixed slice becomes usable only after its full interval has closed.
    opening = bars.loc[(bars.index > cash_open) & (bars.index <= opening_close)]
    if bars.index[-1] < opening_close or opening.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    opening_high = opening["high"].max()
    opening_low = opening["low"].min()
    opening_width = opening_high - opening_low
    known_opening = bars.index >= opening_close
    f["nq_opening_high_distance"] = ((close - opening_high) / stop).where(known_opening)
    f["nq_opening_low_distance"] = ((close - opening_low) / stop).where(known_opening)
    f["nq_opening_range"] = (opening_width / stop).where(known_opening)
    f["nq_opening_position"] = (
        (close - opening_low) / max(opening_width, 0.25)
    ).where(known_opening)
    for minutes in (1, 5, 15):
        volume = bars["volume"].rolling(minutes).sum()
        f[f"nq_delta_fraction_{minutes}m"] = (
            bars["delta"].rolling(minutes).sum() / volume.replace(0, np.nan)
        ).fillna(0)
    f["nq_volume_ratio_5m"] = (
        bars["volume"].rolling(5).mean() / bars["volume"].rolling(30).mean().replace(0, np.nan)
    )
    f["nq_volume_ratio_1m"] = (
        bars["volume"] / bars["volume"].rolling(20).mean().replace(0, np.nan)
    )
    f["nq_volatility_ratio"] = tr.rolling(5).mean() / mean_tr.replace(0, np.nan)

    end = (day + pd.Timedelta(hours=14, minutes=25)).tz_convert("UTC")
    decisions = pd.date_range(opening_close, min(end, bars.index[-1]), freq="5min")
    if decisions.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    bar_pos = as_of_index(bars.index.asi8, decisions.asi8)
    quotes = t.loc[~t.index.duplicated(keep="last")]
    quote_pos = as_of_index(quotes.index.asi8, decisions.asi8)
    seen = (bar_pos >= 0) & (quote_pos >= 0)
    decisions, bar_pos, quote_pos = decisions[seen], bar_pos[seen], quote_pos[seen]
    result = f.iloc[bar_pos].copy()
    result.index = decisions
    result.index.name = "ts"
    current = quotes.iloc[quote_pos]
    bid = current["bid_px"].to_numpy()
    ask = current["ask_px"].to_numpy()
    bid_size = current["bid_sz"].to_numpy()
    ask_size = current["ask_sz"].to_numpy()
    spread = ask - bid
    depth = bid_size + ask_size
    result["nq_quote_imbalance"] = np.divide(
        bid_size - ask_size, depth, out=np.zeros(len(depth)), where=depth > 0,
    )
    result["nq_spread_points"] = spread
    result["nq_time_of_day"] = (decisions - cash_open).total_seconds() / (390 * 60)
    result["stop_points"] = stop.iloc[bar_pos].to_numpy()
    result["decision_bid"] = bid
    result["decision_ask"] = ask
    result["source_ts"] = current.index
    stale_seconds = (decisions.asi8 - current.index.asi8) / 1e9
    finite = np.isfinite(result.select_dtypes(include="number")).all(axis=1)
    eligible = (
        finite & (result["stop_points"] <= 60.0) & (bid > 0)
        & (spread >= 0.25) & (spread <= 1.0)
        & (stale_seconds >= 0) & (stale_seconds <= 2.0)
        & (bid_size > 0) & (ask_size > 0)
    )
    return result.loc[eligible].copy()
