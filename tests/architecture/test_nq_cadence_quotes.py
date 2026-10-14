"""Cadence soft exits are distinct from continuous emergency stops."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.nq_cadence_quotes import simulate_quote_session


def ts(clock):
    return pd.Timestamp(f'2026-01-05 {clock}',tz='America/Chicago').tz_convert('UTC')


def bars(clocks, prices):
    x=np.asarray(prices,float)
    return pd.DataFrame({'open':x,'high':x+1,'low':x-1,'close':x,'volume':1.},
                        index=pd.DatetimeIndex([ts(c) for c in clocks]))


def tape(clocks,bids):
    x=np.asarray(bids,float)
    return pd.DataFrame({'bid_px':x,'ask_px':x+.25},
                        index=pd.DatetimeIndex([ts(c) for c in clocks]))


def run(t,b,**kwargs):
    return simulate_quote_session(t,b,np.full(len(b),100.),np.full(len(b),90.),
                                  use_vwap=False,**kwargs)


def test_intraminute_adverse_move_does_not_trigger_soft_exit():
    b=bars(['09:00','09:01','09:30','15:00'],[102,94,102,105])
    t=tape(['09:00:00.250','09:01:00','09:30:00','15:00:00.250'],[102,94,102,105])
    r=run(t,b).iloc[0]
    assert r.exit_reason=='session_close'
    assert r.mae_points==8.5


def test_soft_exit_uses_closed_cadence_and_next_quote_after_latency():
    b=bars(['09:00','09:30','15:00'],[102,99,105])
    t=tape(['09:00:00.250','09:30:00','09:30:00.249','09:30:00.250'],[102,99,98,97])
    r=run(t,b).iloc[0]
    assert r.exit_reason=='cadence_exit'
    assert r.exit_ts==ts('09:30:00.250') and r.exit_price==96.75
    assert r.net_usd==pytest.approx((96.75-102.5)*20-3.36)


def test_emergency_is_fixed_opposite_band_with_tick_rounding():
    b=bars(['09:00','09:01','15:00'],[102,105,105])
    t=tape(['09:00:00.250','09:01:00','09:01:01','09:01:01.250'],[102,92,90,89])
    r=simulate_quote_session(t,b,np.full(3,100.),np.array([90.12,99.,99.]),use_vwap=False).iloc[0]
    assert r.stop_initial==90.
    assert r.exit_ts==ts('09:01:01.250') and r.exit_price==88.75
    assert r.exit_reason=='emergency_stop'


def test_emergency_can_be_disabled_explicitly():
    b=bars(['09:00','09:01','09:30','15:00'],[102,80,102,105])
    t=tape(['09:00:00.250','09:01:00','09:01:00.250','15:00:00.250'],[102,80,79,105])
    r=run(t,b,emergency_stop=False).iloc[0]
    assert np.isnan(r.stop_initial) and r.exit_reason=='session_close'


def test_short_emergency_rounds_up_and_uses_ask():
    b=bars(['09:00','09:01','15:00'],[88,88,88])
    t=tape(['09:00:00.250','09:01:00','09:01:00.250'],[88,100,101])
    r=simulate_quote_session(t,b,np.full(3,100.12),np.full(3,90.),use_vwap=False).iloc[0]
    assert r.side==-1 and r.stop_initial==100.25
    assert r.exit_price==101.5 and r.exit_reason=='emergency_stop'


def test_current_band_can_loosen_at_later_cadence():
    b=bars(['09:00','09:01','09:30','15:00'],[102,110,102,105])
    t=tape(['09:00:00.250','09:01:00','09:30:00','15:00:00.250'],[102,110,102,105])
    r=simulate_quote_session(t,b,np.array([100.,109.,100.,100.]),np.full(4,90.),use_vwap=False).iloc[0]
    assert r.exit_reason=='session_close'


def test_no_flip_at_exit_decision():
    b=bars(['09:00','09:30','10:00','15:00'],[102,88,88,85])
    t=tape(['09:00:00.250','09:30:00.250','10:00:00.250','15:00:00.250'],[102,88,88,85])
    r=run(t,b,emergency_stop=False)
    assert len(r)==2 and r.side.tolist()==[1,-1]
    assert r.iloc[1].entry_decision_ts==ts('10:00')


def test_prefix_and_future_bar_mutations_leave_finished_trade_unchanged():
    b=bars(['09:00','09:30','15:00'],[102,99,105])
    t=tape(['09:00:00.250','09:30:00.250'],[102,99])
    full=run(t,b)
    pd.testing.assert_frame_equal(full,run(t,b.iloc[:2]))
    changed=b.copy();changed.loc[ts('15:00'),['open','high','low','close']]=1000
    pd.testing.assert_frame_equal(full,run(t,changed))


def test_missing_soft_exit_is_not_fabricated():
    b=bars(['09:00','09:30','15:00'],[102,99,105])
    t=tape(['09:00:00.250'],[102])
    r=run(t,b).iloc[0]
    assert not r.valid and r.exit_reason=='missing_cadence_exit'
    assert pd.isna(r.exit_ts) and np.isnan(r.net_usd)


def test_hourly_policy_does_not_react_at_halfhour_check():
    b=bars(['09:30','10:00','10:30','15:00'],[102,95,102,105])
    t=tape(['09:30:00.250','10:00:00.250','10:30:00.250','15:00:00.250'],[102,95,102,105])
    r=run(t,b,cadence_minutes=60).iloc[0]
    assert r.entry_decision_ts==ts('09:30') and r.exit_reason=='session_close'


def test_vwap_soft_condition_only_checked_on_cadence():
    b=bars(['09:00','09:01','09:30','15:00'],[102,200,110,110])
    b.loc[ts('09:00'),['open','high','low']]=[100,102,98]
    t=tape(['09:00:00.250','09:01:00.250','09:30:00.250'],[102,110,110])
    r=simulate_quote_session(t,b,np.full(4,100.),np.full(4,90.)).iloc[0]
    assert r.exit_reason=='cadence_exit' and r.exit_ts==ts('09:30:00.250')


def test_regular_bar_proxy_and_quotes_match_with_equivalent_fills():
    from src.backtest.nq_cadence_area import simulate_session
    index=pd.date_range(ts('09:00'),ts('09:31'),freq='min')
    b=pd.DataFrame({'open':102.,'high':103.,'low':101.,'close':102.,'volume':1.},index=index)
    b.loc[ts('09:30'),['low','close']]=[98.,99.]
    b.loc[ts('09:31'),['open','high','low','close']]=[99.,100.,98.,99.]
    t=tape(['09:00:00.250','09:30:00','09:30:00.250'],[102,98.75,98.75])
    upper=np.full(len(b),100.);lower=np.full(len(b),90.)
    quote=simulate_quote_session(t,b,upper,lower,use_vwap=False)
    proxy=simulate_session(b,upper,lower,use_vwap=False,roundtrip_cost_points=1.)
    assert len(quote)==len(proxy)==1
    for c in ['side','entry_price','exit_price','net_usd','gross_points','stop_initial','exit_reason']:
        assert quote.iloc[0][c]==proxy.iloc[0][c]
