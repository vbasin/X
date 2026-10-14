"""Economic and timing contracts for the independent quote replay."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.nq_noise_quotes import simulate_quote_session


def ts(clock):
    return pd.Timestamp(f'2026-01-05 {clock}',tz='America/Chicago').tz_convert('UTC')


def bars(clocks=('09:00:00','09:01:00','15:00:00'), closes=None):
    closes = np.full(len(clocks),102.) if closes is None else np.asarray(closes,float)
    return pd.DataFrame({'open':closes,'high':closes+1,'low':closes-1,
                         'close':closes,'volume':np.ones(len(clocks))},
                        index=pd.DatetimeIndex([ts(c) for c in clocks]))


def tape(clocks,bids,spread=.25):
    return pd.DataFrame({'bid_px':bids,'ask_px':np.asarray(bids)+spread},
                        index=pd.DatetimeIndex([ts(c) for c in clocks]))


def run(t,b,**kwargs):
    return simulate_quote_session(t,b,np.full(len(b),100.),np.full(len(b),90.),
                                  use_vwap=False,**kwargs)


def test_entry_and_clock_flatten_pay_spread_slippage_commission():
    b=bars()
    t=tape(['09:00:00.249','09:00:00.250','15:00:00.249','15:00:00.250'],[102,103,105,106])
    r=run(t,b).iloc[0]
    assert r.entry_ts == ts('09:00:00.250')
    assert r.exit_ts == ts('15:00:00.250')
    assert r.entry_price == 103.5 and r.exit_price == 105.75
    assert r.net_usd == pytest.approx(2.25*20-3.36)
    assert r.exit_reason == 'session_close' and r.valid


def test_stop_uses_quote_and_preserves_gap_after_latency():
    b=bars()
    t=tape(['09:00:00.250','09:00:10','09:00:10.100','09:00:11'],[102,99,98,94])
    r=run(t,b).iloc[0]
    assert r.exit_ts == ts('09:00:11')
    assert r.exit_price == 93.75
    assert r.exit_reason == 'stop' and r.valid
    assert r.mae_points == pytest.approx(8.5)
    assert r.max_quote_gap_seconds >= 9.75


def test_bar_low_cannot_trigger_quote_stop():
    b=bars()
    b.loc[ts('09:01:00'),'low']=50
    t=tape(['09:00:00.250','09:01:00','15:00:00.250'],[102,102,103])
    r=run(t,b).iloc[0]
    assert r.exit_reason == 'session_close'


def test_bar_close_tightening_triggers_asof_quote_after_close_only():
    b=bars()
    upper=np.array([100.,103.,103.])
    t=tape(['09:00:00.250','09:00:59','09:01:00.249','09:01:00.250'],[102,102,101,100])
    r=simulate_quote_session(t,b,upper,np.full(3,90.),use_vwap=False).iloc[0]
    assert r.entry_price == 102.5
    assert r.stop_initial == 100.
    assert r.exit_ts == ts('09:01:00.250')
    assert r.exit_price == 99.75


def test_stop_ratchet_never_loosens():
    b=bars(('09:00:00','09:01:00','09:02:00','15:00:00'))
    t=tape(['09:00:00.250','09:01:00.500','09:02:00.500','09:02:00.750'],[105,105,100.5,100])
    r=simulate_quote_session(t,b,np.array([100.,101.,90.,90.]),np.full(4,80.),use_vwap=False).iloc[0]
    assert r.exit_ts == ts('09:02:00.750')
    assert r.exit_reason == 'stop'


def test_unfilled_entry_is_explicit_and_not_fabricated():
    b=bars()
    t=tape(['09:00:05.001'],[102])
    r=run(t,b).iloc[0]
    assert not r.valid and r.exit_reason == 'no_entry_within_5s'
    assert pd.isna(r.entry_ts) and np.isnan(r.net_usd)


def test_missing_stop_exit_is_explicit():
    b=bars()
    t=tape(['09:00:00.250','09:00:10'],[102,99])
    r=run(t,b).iloc[0]
    assert not r.valid and r.exit_reason == 'missing_stop_exit'
    assert pd.isna(r.exit_ts) and np.isnan(r.net_usd)


def test_short_fills_on_correct_sides_and_long_only_abstains():
    b=bars(closes=[88,88,88])
    t=tape(['09:00:00.250','15:00:00.250'],[88,86])
    r=run(t,b).iloc[0]
    assert r.side == -1 and r.entry_price == 87.75 and r.exit_price == 86.5
    assert run(t,b,direction_mode='long_only').empty


def test_future_bars_and_quotes_do_not_change_closed_trade_or_initial_signal():
    b=bars()
    t=tape(['09:00:00.250','09:00:10','09:00:10.250','09:01:01'],[102,99,98,102])
    original=run(t,b)
    changed=b.copy()
    changed.loc[changed.index>ts('09:00:00'),['open','high','low','close']]=1000
    other=t.copy()
    other.loc[other.index>ts('09:00:10.250'),:]=5000
    pd.testing.assert_frame_equal(original,run(other,changed))
    prefix=run(t.iloc[:3],b.iloc[:1])
    pd.testing.assert_frame_equal(original,prefix)


def test_vwap_is_session_hlc3_and_causal():
    b=bars(closes=[102,200,200])
    b.loc[ts('09:00:00'),['open','high','low']]=[100,102,98]
    t=tape(['09:00:00.250','09:00:10','09:00:10.250'],[102,100,99])
    a=simulate_quote_session(t,b,np.full(3,100.),np.full(3,90.))
    prefix=simulate_quote_session(t,b.iloc[:1],np.array([100.]),np.array([90.]))
    assert a.iloc[0].stop_initial == pytest.approx(302/3)
    pd.testing.assert_frame_equal(a,prefix)


def test_two_losses_cease_further_entries():
    b=bars(('09:00:00','09:30:00','10:00:00','15:00:00'))
    t=tape(['09:00:00.250','09:00:01','09:00:01.250',
            '09:30:00.250','09:30:01','09:30:01.250',
            '10:00:00.250','15:00:00.250'],[102,99,98,102,99,98,102,110])
    r=run(t,b)
    assert len(r)==2 and (r.net_usd<0).all()


def test_exit_requires_later_quote_even_with_zero_latency():
    b=bars()
    t=tape(['09:00:00','09:00:01','09:00:02','09:00:02'],[102,102,99,98])
    r=run(t,b,latency_ms=0).iloc[0]
    assert r.entry_ts == ts('09:00:01')
    assert r.exit_price == 97.75


def test_missing_final_bar_does_not_hide_clock_flatten():
    b=bars(('09:00:00',))
    t=tape(['09:00:00.250','15:00:00.250'],[102,110])
    r=run(t,b).iloc[0]
    assert r.valid and r.exit_reason=='session_close'


def test_one_minute_cooldown_uses_actual_exit_timestamp():
    b=bars(('09:00:00','09:30:00','10:00:00','15:00:00'))
    t=tape(['09:00:00.250','09:29:10','09:29:10.250',
            '09:30:00.250','10:00:00.250','15:00:00.250'],[102,99,99,105,105,110])
    r=run(t,b)
    assert len(r)==2
    assert r.iloc[1].entry_decision_ts == ts('10:00:00')
