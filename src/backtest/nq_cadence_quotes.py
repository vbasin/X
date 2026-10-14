"""Quote replay with cadence-confirmed exits and a fixed emergency stop.

Preserves the minute-stop predecessor; only closed cadence bars change soft
exposure. TBBO remains a trade-sampled execution proxy, not full book replay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


_COLUMNS = ['decision_ts', 'entry_decision_ts', 'entry_ts', 'exit_ts', 'side',
            'entry_price', 'exit_price', 'exit_reason', 'stop_initial', 'net_usd',
            'gross_points', 'mae_points', 'mfe_points', 'valid', 'max_quote_gap_seconds']


@njit(cache=True)
def _replay(qt, bid, ask, bt, minute, close, upper, lower, vwap,
            end, cadence, mode, use_vwap, max_entries, slip, latency, commission, emergency):
    # decision, entry time, exit time, side, entry, exit, reason, stop_initial,
    # net, gross, mae, mfe, valid, gap. Int timestamps stored separately.
    out = np.full((len(bt) + 1, 14), np.nan)
    stamps = np.full((len(bt) + 1, 3), np.iinfo(np.int64).min, dtype=np.int64)
    q = b = rows = entries = losses = 0
    position = pending = 0
    decision = entry_time = 0
    last_exit = np.iinfo(np.int64).min // 2
    last_q = -1
    ready = 0
    exiting = 0
    trigger_index = -1
    stop = initial = entry = mae = mfe = gap = 0.0
    previous_quote = 0
    flattened = False
    huge = np.iinfo(np.int64).max
    while q < len(qt) or b < len(bt) or not flattened:
        next_q = qt[q] if q < len(qt) else huge
        next_b = bt[b] if b < len(bt) else huge
        next_end = end if not flattened else huge
        now = min(next_q, next_b, next_end)
        if now == huge:
            break
        if pending != 0 and now > decision + 5_000_000_000:
            stamps[rows, 0] = decision
            out[rows, 3] = pending
            out[rows, 6] = 3
            out[rows, 7] = initial
            out[rows, 12] = 0
            rows += 1
            pending = 0
        if next_q == now:
            last_q = q
            if pending != 0 and now >= ready:
                position = pending
                pending = 0
                entry_time = now
                entry = ask[q] + slip if position == 1 else bid[q] - slip
                entries += 1
                mae = mfe = gap = 0.0
                previous_quote = now
            if position != 0:
                mark = bid[q] if position == 1 else ask[q]
                move = position * (mark - entry)
                mae = max(mae, -move)
                mfe = max(mfe, move)
                gap = max(gap, (now-previous_quote)/1e9)
                previous_quote = now
                if exiting != 0 and now >= ready and q > trigger_index:
                    price = mark - position * slip
                    gross = position * (price-entry)
                    net = gross*20-commission
                    stamps[rows, 0] = decision
                    stamps[rows, 1] = entry_time
                    stamps[rows, 2] = now
                    out[rows, 3:] = np.array([float(position),entry,price,float(exiting),
                        initial,net,gross,mae,mfe,1.0,gap])
                    rows += 1
                    losses += int(net < 0)
                    position = exiting = 0
                    last_exit = now
                elif exiting == 0 and ((position == 1 and mark <= stop)
                                        or (position == -1 and mark >= stop)):
                    exiting = 1
                    ready = now + latency
                    trigger_index = q
            q += 1
        elif next_end == now:
            flattened = True
            if position != 0 and exiting == 0:
                exiting = 2
                ready = now + latency
                trigger_index = last_q
            # No signal is scheduled this late; defensively cancel pending.
            pending = 0
        else:
            if (position != 0 and exiting == 0 and minute[b] % cadence == 0
                    and 30 <= minute[b] < 390):
                confirmed = ((position == 1 and np.isfinite(upper[b]) and close[b] > upper[b]
                    and (not use_vwap or (np.isfinite(vwap[b]) and close[b] > vwap[b])))
                    or (position == -1 and np.isfinite(lower[b]) and close[b] < lower[b]
                    and (not use_vwap or (np.isfinite(vwap[b]) and close[b] < vwap[b]))))
                if not confirmed:
                    exiting = 6
                    ready = now + latency
                    trigger_index = last_q
            if (position == 0 and pending == 0 and entries < max_entries
                    and losses < 2 and now >= last_exit + 60_000_000_000
                    and 30 <= minute[b] <= 330 and minute[b] % cadence == 0):
                side = 0
                if (mode != -1 and np.isfinite(upper[b]) and close[b] > upper[b]
                        and (not use_vwap or (np.isfinite(vwap[b]) and close[b] > vwap[b]))):
                    side = 1
                elif (mode != 1 and np.isfinite(lower[b]) and close[b] < lower[b]
                        and (not use_vwap or (np.isfinite(vwap[b]) and close[b] < vwap[b]))):
                    side = -1
                if side:
                    opposite = lower[b] if side == 1 else upper[b]
                    if emergency and not np.isfinite(opposite):
                        b += 1
                        continue
                    if emergency:
                        stop = np.floor(opposite / .25) * .25 if side == 1 else np.ceil(opposite / .25) * .25
                        initial = stop
                    else:
                        stop = -np.inf if side == 1 else np.inf
                        initial = np.nan
                    pending = side
                    decision = now
                    ready = now + latency
            b += 1
        if flattened and position == 0 and pending == 0:
            break
    if position != 0:
        stamps[rows, 0] = decision
        stamps[rows, 1] = entry_time
        out[rows, 3:] = np.array([float(position),entry,np.nan,
            4.0 if exiting == 1 else (7.0 if exiting == 6 else 5.0), initial,np.nan,np.nan,mae,mfe,0.0,gap])
        rows += 1
    return out[:rows], stamps[:rows]


def simulate_quote_session(
    tape: pd.DataFrame, bars: pd.DataFrame, upper: np.ndarray, lower: np.ndarray, *,
    cadence_minutes: int = 30, direction_mode: str = 'both', use_vwap: bool = True,
    max_entries: int = 3, slippage_points: float = .25, latency_ms: float = 250,
    commission: float = 3.36, emergency_stop: bool = True,
) -> pd.DataFrame:
    """Replay a closed-bar cadence policy without minute soft-stop updates.

    Entry conditions use the completed cadence close, current causal bands and
    optionally cash HLC3-volume VWAP. At subsequent cadence closes, an existing
    position exits when its own entry condition is false. The soft band is not
    ratcheted. A fixed opposite-band emergency stop may trigger between checks;
    its entry-decision value is floored to a tick for longs and ceiled for shorts.
    Disabling the emergency stop records NaN in ``stop_initial``. Quotes execute
    exits after latency and later than the trigger event. No flip occurs at a
    soft-exit decision; the standard cooldown and daily limits still apply.
    Invalid fills remain visible, and all actual quote gaps are preserved.
    """
    for name, frame in [('tape',tape),('bars',bars)]:
        if not isinstance(frame.index,pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f'{name} requires timezone-aware timestamps')
        if frame.index.hasnans or not frame.index.is_monotonic_increasing:
            raise ValueError(f'{name} timestamps must be sorted and finite')
    if not bars.index.is_unique:
        raise ValueError('bar timestamps must be unique')
    if cadence_minutes not in (30,60):
        raise ValueError('cadence_minutes must be 30 or 60')
    modes = {'both':0,'long_only':1,'long':1,'short_only':-1,'short':-1}
    if direction_mode not in modes:
        raise ValueError('invalid direction_mode')
    if max_entries < 1 or int(max_entries) != max_entries:
        raise ValueError('max_entries must be a positive integer')
    if not all(np.isfinite(x) and x >= 0 for x in [slippage_points,latency_ms,commission]):
        raise ValueError('costs and latency must be finite and nonnegative')
    upper, lower = np.asarray(upper,float), np.asarray(lower,float)
    if upper.shape != (len(bars),) or lower.shape != (len(bars),):
        raise ValueError('bands must align to bars')
    if bars.empty:
        return pd.DataFrame(columns=_COLUMNS)
    local = bars.index.tz_convert('America/Chicago')
    if len(np.unique(local.date)) != 1:
        raise ValueError('bars must contain one local calendar date')
    if ((local.second != 0)|(local.microsecond != 0)|(local.nanosecond != 0)).any():
        raise ValueError('bars must be stamped at minute boundaries')
    if len(tape) and (tape.index.tz_convert('America/Chicago').date != local.date[0]).any():
        raise ValueError('tape and bars must belong to the same local date')
    minute = np.asarray(local.hour*60+local.minute-510,np.int64)
    keep = (minute>0)&(minute<=390)
    values = bars.loc[keep,['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,4]<0).any():
        raise ValueError('OHLCV must be finite and volume nonnegative')
    if ((values[:,1]<np.maximum(values[:,0],values[:,3])).any()
            or (values[:,2]>np.minimum(values[:,0],values[:,3])).any()
            or (values[:,1]<values[:,2]).any()):
        raise ValueError('inconsistent OHLC extrema')
    volume = np.cumsum(values[:,4])
    numerator = np.cumsum(values[:,1:4].mean(axis=1)*values[:,4])
    vwap = np.divide(numerator,volume,out=np.full(len(volume),np.nan),where=volume>0)
    bid = tape['bid_px' if 'bid_px' in tape else 'bid_px_00'].to_numpy(float)
    ask = tape['ask_px' if 'ask_px' in tape else 'ask_px_00'].to_numpy(float)
    valid = np.isfinite(bid)&np.isfinite(ask)&(bid>0)&(ask>=bid)
    end = (local[0].normalize()+pd.Timedelta(hours=15)).tz_convert('UTC').value
    raw, stamps = _replay(tape.index.as_unit('ns').asi8[valid],bid[valid],ask[valid],
        bars.index.as_unit('ns').asi8[keep],minute[keep],values[:,3],upper[keep],lower[keep],
        vwap,end,cadence_minutes,modes[direction_mode],use_vwap,int(max_entries),
        slippage_points,int(latency_ms*1e6),commission,bool(emergency_stop))
    reasons = {1:'emergency_stop',2:'session_close',3:'no_entry_within_5s',
               4:'missing_emergency_exit',5:'missing_session_end',6:'cadence_exit',7:'missing_cadence_exit'}
    result = pd.DataFrame(index=np.arange(len(raw)))
    for i,name in enumerate(['entry_decision_ts','entry_ts','exit_ts']):
        result[name] = pd.to_datetime(stamps[:,i],utc=True)
    result['decision_ts'] = result.entry_decision_ts
    for i,name in [(3,'side'),(4,'entry_price'),(5,'exit_price'),(7,'stop_initial'),
                   (8,'net_usd'),(9,'gross_points'),(10,'mae_points'),(11,'mfe_points'),
                   (13,'max_quote_gap_seconds')]:
        result[name] = raw[:,i]
    result['exit_reason'] = [reasons[int(r)] for r in raw[:,6]]
    result['valid'] = raw[:,12].astype(bool)
    result['side'] = result.side.astype(np.int8)
    return result[_COLUMNS]
