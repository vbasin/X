"""Causal minute-bar execution for the experimental NQ noise-area policy.

Signals and stop updates use completed bars; orders act on the following bar.
This is a bar execution proxy, not quote-level proof. On an intrabar stop, the
unknown ordering of that bar's extremes prevents exact MAE/MFE reconstruction.
We use its open and stop fill only, omitting potentially post-exit extremes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


_COLUMNS = [
    "entry_decision_ts", "entry_ts", "exit_ts", "side", "entry_price",
    "exit_price", "exit_reason", "stop_initial", "net_usd", "gross_points",
    "mae_points", "mfe_points", "valid",
]


@njit(cache=True)
def _simulate(
    times: np.ndarray, minute: np.ndarray, values: np.ndarray,
    upper: np.ndarray, lower: np.ndarray, vwap: np.ndarray,
    cadence: int, mode: int, use_vwap: bool, max_entries: int,
    cost: float, commission: float,
) -> np.ndarray:
    # d_i, e_i, x_i, exit_at_open, side, entry, exit, reason, initial_stop,
    # gross, net, mae, mfe, valid. Reason: 1 stop, 2 close, 3 missing close.
    result = np.full((max_entries, 14), np.nan)
    position = 0
    pending_side = 0
    decision_i = -1
    entry_i = -1
    exit_i = -1
    entries = 0
    losses = 0
    rows = 0
    entry = stop = initial_stop = mae = mfe = 0.0
    for i in range(len(times)):
        opening, high, low, close = values[i, :4]
        if pending_side != 0:
            position = pending_side
            pending_side = 0
            entry_i = i
            entry = opening + position * cost / 2.0
            entries += 1
            mae = 0.0
            mfe = 0.0
        if position != 0:
            gap_stop = (position == 1 and opening <= stop) or (
                position == -1 and opening >= stop
            )
            hit_stop = gap_stop or (position == 1 and low <= stop) or (
                position == -1 and high >= stop
            )
            if hit_stop:
                raw_exit = opening if gap_stop else stop
                # The open and stop are observable before execution. The other
                # extreme could occur after exit and must not inflate excursions.
                move_open = position * (opening - entry)
                move_exit = position * (raw_exit - entry)
                mae = max(mae, -move_open, -move_exit)
                mfe = max(mfe, move_open, move_exit)
            else:
                adverse = low if position == 1 else high
                favorable = high if position == 1 else low
                mae = max(mae, -position * (adverse - entry))
                mfe = max(mfe, position * (favorable - entry))
                raw_exit = close
            at_end = minute[i] == 390
            if hit_stop or at_end:
                exit_price = raw_exit - position * cost / 2.0
                gross = position * (exit_price - entry)
                net = gross * 20.0 - commission
                result[rows] = np.array([
                    float(decision_i), float(entry_i), float(i), 1.0 if gap_stop else 0.0,
                    float(position), entry, exit_price, 1.0 if hit_stop else 2.0,
                    initial_stop, gross, net, mae, mfe, 1.0,
                ])
                rows += 1
                losses += int(net < 0.0)
                position = 0
                exit_i = i
            else:
                # Updates occur AFTER this bar's trigger checks and become
                # effective only in the next bar. Ratcheting never loosens risk.
                band = upper[i] if position == 1 else lower[i]
                if use_vwap and np.isfinite(vwap[i]):
                    band = max(band, vwap[i]) if position == 1 else min(band, vwap[i])
                if np.isfinite(band):
                    # A sell stop executes at a tick at/below its software
                    # threshold; a buy stop at/above. Never grant fractional ticks.
                    band = np.floor(band * 4.0) / 4.0 if position == 1 else np.ceil(band * 4.0) / 4.0
                    stop = max(stop, band) if position == 1 else min(stop, band)
        if (position == 0 and entries < max_entries and losses < 2
                and i != exit_i and 30 <= minute[i] <= 330
                and minute[i] % cadence == 0):
            # A signal at the final observed bar has no executable next bar.
            # It is not counted as a completed trade or an entry.
            side = 0
            if (mode != -1 and np.isfinite(upper[i]) and close > upper[i]
                    and (not use_vwap or (np.isfinite(vwap[i]) and close > vwap[i]))):
                side = 1
            elif (mode != 1 and np.isfinite(lower[i]) and close < lower[i]
                    and (not use_vwap or (np.isfinite(vwap[i]) and close < vwap[i]))):
                side = -1
            if side != 0:
                stop = upper[i] if side == 1 else lower[i]
                if use_vwap:
                    stop = max(stop, vwap[i]) if side == 1 else min(stop, vwap[i])
                stop = np.floor(stop * 4.0) / 4.0 if side == 1 else np.ceil(stop * 4.0) / 4.0
                initial_stop = stop
                pending_side = side
                decision_i = i
    if position != 0:
        result[rows] = np.array([
            float(decision_i), float(entry_i), -1.0, 0.0, float(position),
            entry, np.nan, 3.0, initial_stop, np.nan, np.nan, mae, mfe, 0.0,
        ])
        rows += 1
    return result[:rows]


def simulate_session(
    bars: pd.DataFrame, upper: np.ndarray, lower: np.ndarray, *,
    cadence_minutes: int = 30, direction_mode: str = "both",
    use_vwap: bool = True, max_entries: int = 3,
    roundtrip_cost_points: float = 1.0, commission: float = 3.36,
) -> pd.DataFrame:
    """Simulate one session of UTC close-stamped minute OHLCV bars.

    ``upper``/``lower`` are aligned, precomputed causal bands; their provenance
    is the caller's responsibility. Only 08:30–15:00 America/Chicago is traded.
    Stops ratchet at every close; decisions occur at cadence multiples after the
    open, from minute 30 through 330. Entries use the NEXT available bar's open.
    ``side`` is +1/-1. Prices include half the proxy round-trip cost per leg;
    ``gross_points`` includes this price friction but excludes commission.
    ``net_usd`` uses $20/point and one round-trip commission. Intrabar stop
    timestamps are the triggering bar's close (an upper bound on actual time).
    Missing final bars leave open trades invalid with NaN realized economics.

    Internal gaps are allowed as requested by the next-available-bar convention;
    unobserved intragap stop paths cannot be reconstructed by this proxy.
    Quote-level validation is required before relying on its execution estimates.
    """
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError("bars require a timezone-aware DatetimeIndex")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("bars must have strictly increasing unique timestamps")
    if cadence_minutes not in (30, 60):
        raise ValueError("cadence_minutes must be 30 or 60")
    modes = {"both": 0, "long": 1, "long_only": 1, "short": -1, "short_only": -1}
    if direction_mode not in modes:
        raise ValueError("direction_mode must be both, long, or short")
    if max_entries < 1 or int(max_entries) != max_entries:
        raise ValueError("max_entries must be a positive integer")
    if (not np.isfinite(roundtrip_cost_points) or roundtrip_cost_points < 0
            or not np.isfinite(commission) or commission < 0):
        raise ValueError("execution costs must be finite and nonnegative")
    upper = np.asarray(upper, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    if upper.shape != (len(bars),) or lower.shape != (len(bars),):
        raise ValueError("bands must be one-dimensional and aligned to bars")
    if len(bars) == 0:
        return pd.DataFrame(columns=_COLUMNS)
    local = bars.index.tz_convert("America/Chicago")
    if len(np.unique(local.date)) != 1:
        raise ValueError("bars must contain exactly one local calendar date")
    if ((local.second != 0) | (local.microsecond != 0) | (local.nanosecond != 0)).any():
        raise ValueError("bars must be stamped on exact minute boundaries")
    minute = np.asarray(local.hour * 60 + local.minute - 510, dtype=np.int64)
    keep = (minute > 0) & (minute <= 390)
    selected = bars.loc[keep, ["open", "high", "low", "close", "volume"]]
    values = selected.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values[:, 4] < 0).any():
        raise ValueError("OHLCV must be finite and volume nonnegative")
    if ((values[:, 1] < np.maximum(values[:, 0], values[:, 3])).any()
            or (values[:, 2] > np.minimum(values[:, 0], values[:, 3])).any()
            or (values[:, 1] < values[:, 2]).any()):
        raise ValueError("OHLC extrema are inconsistent")
    volume = np.cumsum(values[:, 4])
    numerator = np.cumsum(values[:, 1:4].mean(axis=1) * values[:, 4])
    vwap = np.divide(numerator, volume, out=np.full(len(volume), np.nan), where=volume > 0)
    index = selected.index.tz_convert("UTC")
    raw = _simulate(
        index.asi8, minute[keep], values, upper[keep], lower[keep], vwap,
        cadence_minutes, modes[direction_mode], use_vwap, int(max_entries),
        roundtrip_cost_points, commission,
    )
    records = []
    reasons = {1: "stop", 2: "session_close", 3: "missing_session_end"}
    for row in raw:
        exit_ts = pd.NaT if row[2] < 0 else index[int(row[2])]
        if row[3] == 1:
            exit_ts -= pd.Timedelta(minutes=1)
        records.append({
            "entry_decision_ts": index[int(row[0])],
            "entry_ts": index[int(row[1])] - pd.Timedelta(minutes=1),
            "exit_ts": exit_ts, "side": int(row[4]), "entry_price": row[5],
            "exit_price": row[6], "exit_reason": reasons[int(row[7])],
            "stop_initial": row[8], "net_usd": row[10], "gross_points": row[9],
            "mae_points": row[11], "mfe_points": row[12], "valid": bool(row[13]),
        })
    return pd.DataFrame(records, columns=_COLUMNS)
