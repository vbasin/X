"""Synthetic integration of closed-bar features, quote replay and economics."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.eval_nq_noise_quotes import read_tape_bars
from src.features.nq_noise_area import attach_noise_area
from src.backtest.nq_noise_quotes import simulate_quote_session
from src.evaluation.economic import (
    Leg,LegRole,OrderType,Trade,compute_trade_ledger,load_assumptions,
)


def stamp(day,clock):
    return pd.Timestamp(f'{day} {clock}',tz='America/Chicago').tz_convert('UTC')


def history():
    rows=[]
    for day,mark in [('2026-01-05',105.),('2026-01-06',106.),
                     ('2026-01-07',120.),('2026-01-08',140.)]:
        for clock,price in [('08:31',100.),('09:00',mark),('09:01',mark+1),('15:00',mark+2)]:
            rows.append(dict(ts=stamp(day,clock),open=price,high=price+1,
                             low=price-1,close=price,volume=10.))
    return pd.DataFrame(rows).set_index('ts')


def test_shared_features_and_quote_results_are_prefix_invariant():
    raw=history()
    full=attach_noise_area(raw,lookback=2)
    cutoff=stamp('2026-01-07','09:00')
    prefix=attach_noise_area(raw.loc[:cutoff],lookback=2)
    pd.testing.assert_frame_equal(full.loc[:cutoff],prefix)
    assert full.loc[cutoff,'sigma']==pytest.approx(.055)
    modified=raw.copy()
    future=modified.index>cutoff
    modified.loc[future,['open','high','low','close']]*=100
    pd.testing.assert_frame_equal(full.loc[:cutoff],attach_noise_area(modified,lookback=2).loc[:cutoff])
    tape=pd.DataFrame({'bid_px':[120.,100.,99.], 'ask_px':[120.25,100.25,99.25]},
        index=pd.DatetimeIndex([stamp('2026-01-07',c) for c in ['09:00:00.250','09:00:10','09:00:10.250']]))
    result=[]
    for f in [full,prefix]:
        b=f[f.session.eq('2026-01-07')]
        result.append(simulate_quote_session(tape,b,b.upper.to_numpy(),b.lower.to_numpy()))
    pd.testing.assert_frame_equal(*result)
    assert len(result[0])==1 and result[0].iloc[0].valid


def test_resolved_quote_fills_match_canonical_nq_accounting():
    f=attach_noise_area(history(),lookback=2)
    b=f[f.session.eq('2026-01-07')]
    tape=pd.DataFrame({'bid_px':[120.,100.,99.], 'ask_px':[120.25,100.25,99.25]},
        index=pd.DatetimeIndex([stamp('2026-01-07',c) for c in ['09:00:00.250','09:00:10','09:00:10.250']]))
    assumptions=load_assumptions('config/execution_costs.yaml','NQ')
    assert assumptions.commission_per_trade==3.36
    for slip in [.25,.5,1.]:
        actual=simulate_quote_session(tape,b,b.upper.to_numpy(),b.lower.to_numpy(),
            slippage_points=slip,commission=assumptions.commission_per_trade)
        canonical=compute_trade_ledger([
            Trade(str(i),int(r.side),
                Leg(OrderType.MARKET,resolved_fill_price=r.entry_price,fill_source='synthetic_quote_replay',ts=str(r.entry_ts)),
                Leg(OrderType.MARKET,role=LegRole.EXIT_TIME,resolved_fill_price=r.exit_price,
                    fill_source='synthetic_quote_replay',ts=str(r.exit_ts)))
            for i,r in enumerate(actual.itertuples())],assumptions)
        np.testing.assert_allclose([r.net_dollars for r in canonical.trades],actual.net_usd,rtol=0,atol=1e-8)


def test_tape_builder_does_not_publish_unclosed_final_minute(tmp_path):
    path=tmp_path/'tape_2026-01-07.parquet'
    source=pd.DataFrame({'price':[100.,101.,102.],'size':[1.,1.,1.]},
        index=pd.DatetimeIndex([stamp('2026-01-07',c) for c in ['08:30:01','08:30:59','08:31:20']]))
    source.to_parquet(path)
    b,_=read_tape_bars(str(path))
    assert (b.index<=source.index[-1]).all()
    assert list(b.index)==[stamp('2026-01-07','08:31')]
