"""Session-shifted, same-clock NQ noise envelopes. No future session inputs."""
from __future__ import annotations
import calendar
import numpy as np
import pandas as pd
import exchange_calendars as xcals


def eligible_sessions(start: str, end: str) -> list[str]:
    """Full NYSE cash days excluding known quarterly expiration week.

    The long history is calendar-front NQ.c.0; recent tapes roll earlier.
    The pre-known expiry-week exclusion avoids that contract mismatch.
    """
    schedule = xcals.get_calendar('XNYS',start=start,end=end).schedule
    days = schedule.index[(schedule['close']-schedule['open']).eq(pd.Timedelta(minutes=390))]
    out=[]
    for day in days:
        expiry = [w[calendar.FRIDAY] for w in calendar.monthcalendar(day.year,day.month) if w[calendar.FRIDAY]][2]
        if day.month in (3,6,9,12) and expiry-4<=day.day<=expiry:
            continue
        out.append(str(day.date()))
    return out


def attach_noise_area(bars: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Join prior-session same-clock scale to cash bars, preserving availability.

    Inputs have UTC close index and OHLCV; session/minute are derived from time.
    Today's open is known after its first minute closes. Incomplete current
    sessions are never excluded using future completion. No overnight levels.
    """
    if not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None:
        raise ValueError('timezone-aware close index required')
    if lookback<2:
        raise ValueError('lookback must be >=2')
    f=bars.copy().sort_index()
    local=f.index.tz_convert('America/Chicago')
    f['session']=local.strftime('%Y-%m-%d')
    f['minute']=local.hour*60+local.minute
    f=f[(f.minute>510)&(f.minute<=900)].copy()
    if f.duplicated(['session','minute']).any():
        raise ValueError('duplicate session/minute bars')
    starts=f[f.minute.eq(511)].set_index('session')['open']
    f['cash_open']=f.session.map(starts)
    f['abs_return']=(f.close/f.cash_open-1).abs()
    # Missing prior-session minutes carry only earlier values within that past
    # session. Never backfill and never use later minutes to fill an earlier one.
    movement=f.pivot(index='session',columns='minute',values='abs_return').reindex(columns=range(511,901)).ffill(axis=1)
    scale=movement.rolling(lookback,min_periods=lookback).mean().shift(1)
    flat=scale.stack(future_stack=True).rename('sigma').reset_index()
    times=f.index
    f=f.reset_index(names='ts').merge(flat,on=['session','minute'],how='left',validate='one_to_one').set_index('ts')
    f['upper']=f.cash_open*(1+f.sigma)
    f['lower']=f.cash_open*(1-f.sigma)
    prior_last=pd.Series(movement.index,index=movement.index).shift(1)
    f['scale_last_session']=f.session.map(prior_last)
    valid=f.sigma.notna()
    if valid.any() and not (f.loc[valid,'scale_last_session']<f.loc[valid,'session']).all():
        raise AssertionError('current/future session in noise scale')
    return f.drop(columns='abs_return')
