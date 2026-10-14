"""Quote-event execution for the selective NQ research experiment.

Forward quotes determine outcomes only; callers must form decisions using
information available at their timestamps. One session is supplied per call.
Invalid/censored rows remain visible and have no fabricated economic outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _scan(
    times: np.ndarray, bids: np.ndarray, asks: np.ndarray,
    decisions: np.ndarray, stops: np.ndarray, horizon_ns: int,
    target_r: float, slip: float, latency_ns: int, commission: float,
) -> tuple:
    n = len(decisions) * 2
    # Numeric output columns: entry/exit index, entry/exit price, gross, net,
    # net R, MAE, MFE, largest gap, exit reason, invalid reason.
    out = np.full((n, 12), np.nan)
    grace = 5_000_000_000
    for row in range(n):
        d = row // 2
        side = 1 if row % 2 == 0 else -1
        now, stop = decisions[d], stops[d]
        out[row, 10] = 0
        out[row, 11] = 0
        if not np.isfinite(stop) or stop <= 0:
            out[row, 11] = 1
            continue
        entry = np.searchsorted(times, now + latency_ns)
        if entry == len(times) or times[entry] > now + grace:
            out[row, 11] = 2
            continue
        px = asks[entry] + slip if side == 1 else bids[entry] - slip
        out[row, 0] = entry
        out[row, 2] = px
        end = now + horizon_ns
        trigger, reason = -1, 3
        for j in range(entry, len(times)):
            if times[j] > end:
                break
            mark = bids[j] if side == 1 else asks[j]
            move = side * (mark - px)
            if move <= -stop:
                trigger, reason = j, 1
                break
            if move >= target_r * stop:
                trigger, reason = j, 2
                break
        ready = end + latency_ns
        if trigger >= 0:
            ready = times[trigger] + latency_ns
        exit_i = np.searchsorted(times, ready)
        if trigger >= 0:
            exit_i = max(exit_i, trigger + 1)
        exit_i = max(exit_i, entry + 1)
        out[row, 10] = reason
        # A late observed quote is a real adverse execution for a barrier;
        # a missing time-exit near the scheduled horizon is censored.
        valid = exit_i < len(times)
        if valid and trigger < 0:
            valid = times[exit_i] <= ready + grace
        final_i = exit_i if valid else len(times) - 1
        mae, mfe, gap = 0.0, 0.0, 0.0
        previous = times[entry]
        for j in range(entry, final_i + 1):
            mark = bids[j] if side == 1 else asks[j]
            move = side * (mark - px)
            mae = max(mae, -move)
            mfe = max(mfe, move)
            gap = max(gap, (times[j] - previous) / 1e9)
            previous = times[j]
        out[row, 7], out[row, 8], out[row, 9] = mae, mfe, gap
        if not valid:
            out[row, 11] = 3 if trigger >= 0 else 4
            continue
        exit_px = bids[exit_i] - slip if side == 1 else asks[exit_i] + slip
        gross = side * (exit_px - px)
        out[row, 1], out[row, 3] = exit_i, exit_px
        out[row, 4], out[row, 5] = gross, gross * 20.0 - commission
        out[row, 6] = (gross * 20.0 - commission) / (stop * 20.0)
    return out


def simulate_candidates(
    tape: pd.DataFrame, decisions: pd.DataFrame, *,
    horizon_minutes: float = 30, target_r: float = 2.0,
    slippage_points: float = 0.25, latency_ms: float = 250,
    commission: float = 3.36,
) -> pd.DataFrame:
    """Return long then short outcomes for every timestamp, with a RangeIndex.

    Tape requires ``bid_px``/``ask_px`` (``bid_px_00``/``ask_px_00`` also
    accepted) and a sorted timezone-aware DatetimeIndex; duplicate timestamps
    retain input order. Decisions require a timezone-aware index and
    ``stop_points``. NQ economics use $20 per point and one round-trip fee.
    Both stops and targets trigger market exits; no barrier-price clamp or
    optimistic resting-limit fill is used. Quotes crossing a trigger are
    followed by the first subsequent valid quote at/after trigger + latency.

    Invalid quotes are excluded from execution, never used for filtering the
    decision. Entries absent within five seconds of decision are unfilled.
    Time exits absent within five seconds of horizon + latency are censored.
    Barrier-exit gaps remain executable at the later observed quote and are
    disclosed in ``max_quote_gap_seconds``. Excursions are nonnegative and
    based on executable liquidation quotes, including the entry spread.
    """
    for name, frame in (("tape", tape), ("decisions", decisions)):
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{name} requires a timezone-aware DatetimeIndex")
        if frame.index.hasnans:
            raise ValueError(f"{name} contains missing timestamps")
    if not tape.index.is_monotonic_increasing:
        raise ValueError("tape must be sorted; duplicate event ordering is preserved")
    if not all(np.isfinite(v) for v in (
        horizon_minutes, target_r, slippage_points, latency_ms, commission
    )) or horizon_minutes <= 0 or target_r <= 0 or min(
        slippage_points, latency_ms, commission
    ) < 0:
        raise ValueError("invalid horizon, target, latency, slippage or commission")
    bid_key = "bid_px" if "bid_px" in tape else "bid_px_00"
    ask_key = "ask_px" if "ask_px" in tape else "ask_px_00"
    bids = tape[bid_key].to_numpy(dtype=np.float64)
    asks = tape[ask_key].to_numpy(dtype=np.float64)
    usable = np.isfinite(bids) & np.isfinite(asks) & (bids > 0) & (asks >= bids)
    times = tape.index.as_unit("ns").asi8[usable]
    decision_ns = decisions.index.as_unit("ns").asi8
    stops = decisions["stop_points"].to_numpy(dtype=np.float64)
    result = _scan(
        times, bids[usable], asks[usable], decision_ns, stops,
        int(horizon_minutes * 60 * 1e9), target_r, slippage_points,
        int(latency_ms * 1e6), commission,
    )
    rows = len(result)
    reason_names = np.array(["unfilled", "stop", "target", "time"])
    invalid_names = np.array([
        "", "invalid_stop", "no_entry_within_5s", "missing_barrier_exit",
        "missing_time_exit_within_5s",
    ])
    frame = pd.DataFrame({
        "ts": pd.to_datetime(np.repeat(decision_ns, 2), utc=True),
        "side": np.tile(np.array([1, -1], dtype=np.int8), len(decisions)),
        "stop_points": np.repeat(stops, 2),
    })
    for name, column in (("entry_ts", 0), ("exit_ts", 1)):
        stamps = np.full(rows, np.iinfo(np.int64).min, dtype=np.int64)
        present = np.isfinite(result[:, column])
        stamps[present] = times[result[present, column].astype(np.int64)]
        frame[name] = pd.to_datetime(stamps, utc=True)
    for column, name in enumerate((
        "entry_price", "exit_price", "gross_points", "net_usd", "net_r",
        "mae_points", "mfe_points", "max_quote_gap_seconds",
    ), start=2):
        frame[name] = result[:, column]
    frame["exit_reason"] = reason_names[result[:, 10].astype(np.int64)]
    frame["invalid_reason"] = invalid_names[result[:, 11].astype(np.int64)]
    frame["valid"] = result[:, 11] == 0
    return frame
