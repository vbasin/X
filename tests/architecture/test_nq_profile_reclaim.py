from datetime import date

import numpy as np
import pandas as pd

from src.backtest.nq_profile_reclaim_quotes import ExecutionConfig, replay_session
from src.features.cash_session_profile import Bound
from src.features.nq_profile_reclaim import ReclaimConfig, cash_minute_state, prior_bounds, reclaim_signals


def _tape(times, prices, *, sides=None, sizes=None, bid=None, ask=None):
    idx = pd.DatetimeIndex(times, tz="UTC")
    n = len(idx)
    return pd.DataFrame({
        "price": prices, "size": sizes or [20] * n, "side": sides or [1] * n,
        "bid_px": bid or [p - .25 for p in prices], "ask_px": ask or [p + .25 for p in prices],
        "bid_sz": [4] * n, "ask_sz": [4] * n,
    }, index=idx)


def test_minute_state_is_right_labelled_and_future_invariant():
    tape = _tape(["2026-05-01 14:30:01", "2026-05-01 14:30:40", "2026-05-01 14:31:01"],
                 [100., 101., 110.])
    first = cash_minute_state(tape, large_print_size=20)
    changed = tape.copy()
    changed.loc[changed.index >= pd.Timestamp("2026-05-01 14:31:00", tz="UTC"), "price"] = 999.
    second = cash_minute_state(changed, large_print_size=20)
    stamp = pd.Timestamp("2026-05-01 14:31:00", tz="UTC")
    assert first.loc[stamp, "close"] == 101.
    pd.testing.assert_series_equal(first.loc[stamp], second.loc[stamp])


def test_prior_bounds_refuse_cross_roll_history():
    # The June roll starts 2026-06-15; all of these rows are from the old lead.
    levels = pd.DataFrame({
        "session": pd.date_range("2026-05-01", periods=25, freq="B"),
        "lo": np.arange(25, dtype=float), "val": np.arange(25, dtype=float) + .25,
        "poc": np.arange(25, dtype=float) + .5, "vah": np.arange(25, dtype=float) + .75,
        "hi": np.arange(25, dtype=float) + 1.,
    })
    assert prior_bounds(levels, date(2026, 6, 16), ReclaimConfig()) == []


def test_reclaim_signal_requires_closed_reclaim_and_outward_large_print():
    idx = pd.date_range("2026-05-01 14:31:00", periods=3, freq="min", tz="UTC")
    minutes = pd.DataFrame({
        "open": [99., 100., 102.], "high": [99.5, 104., 102.], "low": [98.5, 100., 100.],
        "close": [99.5, 102., 100.5], "volume": [100., 100., 100.], "delta": [0., 20., 20.],
        "big_buy": [0., 20., 0.], "big_sell": [0., 0., 0.], "vwap": [99., 100., 101.],
    }, index=idx)
    upper = Bound(100., 101., strength=2., n_levels=3)
    lower = Bound(50., 60., strength=2., n_levels=3)
    got = reclaim_signals(minutes, [upper, lower], ReclaimConfig())
    assert len(got) == 1
    assert got.iloc[0].side == -1
    assert got.iloc[0].decision_ts == idx[-1]
    assert got.iloc[0].target_price == 60.
    minutes["big_buy"] = 0.
    assert reclaim_signals(minutes, [upper, lower], ReclaimConfig()).empty


def test_quote_replay_waits_for_later_bbo_and_keeps_gap_loss_visible():
    decision = pd.Timestamp("2026-05-01 15:01:00", tz="UTC")
    tape = _tape(["2026-05-01 15:01:00.300", "2026-05-01 15:01:02.000", "2026-05-01 15:01:02.300"],
                 [100., 150., 150.], bid=[99.75, 149.75, 149.75], ask=[100., 150., 150.25])
    signals = pd.DataFrame({"decision_ts": [decision], "side": [1], "target_price": [140.]})
    got = replay_session(tape, signals, config=ExecutionConfig(slippage_points=0., commission=0.))
    assert len(got) == 1
    assert got.iloc[0].entry_ts == tape.index[0]
    assert got.iloc[0].exit_ts == tape.index[2]
    assert got.iloc[0].exit_reason == "target"
    assert got.iloc[0].entry_price == 100.
    assert got.iloc[0].exit_price == 149.75

    gap = _tape(["2026-05-01 15:01:00.300", "2026-05-01 15:01:02.000", "2026-05-01 15:01:02.300"],
                [100., 70., 70.], bid=[99.75, 69.75, 69.75], ask=[100., 70., 70.25])
    stopped = replay_session(gap, signals, config=ExecutionConfig(slippage_points=0., commission=0.))
    assert stopped.iloc[0].exit_reason == "stop"
    assert stopped.iloc[0].net_usd < -500.
