"""Causal execution checks; no database or external market data required."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.nq_noise_area import simulate_session


def session(price=110.0):
    index = pd.date_range("2025-01-02 08:31", periods=390, freq="min", tz="America/Chicago").tz_convert("UTC")
    bars = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 10.0}, index=index)
    return bars, np.full(390, 105.0), np.full(390, 95.0)


def test_next_bar_open_and_cash_end_costs():
    bars, upper, lower = session()
    bars.loc[bars.index[30], ["open", "high"]] = [112.0, 113.0]
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.entry_decision_ts == bars.index[29]
    assert trade.entry_ts == bars.index[30] - pd.Timedelta(minutes=1)
    assert trade.entry_price == 112.5  # not the decision close of 110
    assert trade.exit_price == 109.5
    assert trade.gross_points == -3
    assert trade.net_usd == pytest.approx(-63.36)
    assert trade.exit_ts == bars.index[-1]
    assert trade.exit_reason == "session_close"
    assert trade.valid


def test_gap_stop_uses_worse_open_and_not_stop_price():
    bars, upper, lower = session()
    bars.loc[bars.index[31], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.stop_initial == 105
    assert trade.exit_price == 89.5
    assert trade.exit_ts == bars.index[31] - pd.Timedelta(minutes=1)


def test_stop_update_only_applies_to_next_bar():
    bars, upper, lower = session()
    upper[31:] = 109.5
    bars.loc[bars.index[31], "low"] = 106
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.exit_ts == bars.index[32]
    assert trade.exit_price == 109


def test_stop_never_loosens():
    bars, upper, lower = session()
    upper[30] = 108
    upper[31:] = 103
    bars.loc[bars.index[32], "low"] = 107
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.exit_ts == bars.index[32]
    assert trade.exit_price == 107.5


@pytest.mark.parametrize("side", [1, -1])
def test_fractional_initial_stop_rounds_adversely_to_executable_tick(side):
    bars, upper, lower = session(110 if side == 1 else 90)
    if side == 1:
        upper[:] = 105.13
        bars.loc[bars.index[30], "low"] = 104
        expected = 105.0
    else:
        lower[:] = 94.87
        bars.loc[bars.index[30], "high"] = 96
        expected = 95.0
    trade = simulate_session(bars, upper, lower, use_vwap=False,
                             max_entries=1, roundtrip_cost_points=0).iloc[0]
    assert trade.stop_initial == expected
    assert trade.exit_price == expected


@pytest.mark.parametrize("side", [1, -1])
def test_fractional_ratcheted_stop_rounds_adversely_to_executable_tick(side):
    bars, upper, lower = session(110 if side == 1 else 90)
    if side == 1:
        upper[30:] = 108.13
        bars.loc[bars.index[31], "low"] = 107
        expected = 108.0
    else:
        lower[30:] = 91.87
        bars.loc[bars.index[31], "high"] = 93
        expected = 92.0
    trade = simulate_session(bars, upper, lower, use_vwap=False,
                             max_entries=1, roundtrip_cost_points=0).iloc[0]
    assert trade.exit_ts == bars.index[31]
    assert trade.exit_price == expected


def test_two_losses_disable_further_entries():
    bars, upper, lower = session()
    bars.loc[bars.index[[30, 60, 90, 120]], "low"] = 104
    trades = simulate_session(bars, upper, lower, use_vwap=False)
    assert len(trades) == 2
    assert (trades.net_usd < 0).all()
    assert trades.entry_decision_ts.tolist() == [bars.index[29], bars.index[59]]


def test_cannot_reenter_on_stop_exit_decision_boundary():
    bars, upper, lower = session()
    bars.loc[bars.index[59], "low"] = 104
    trades = simulate_session(bars, upper, lower, use_vwap=False)
    assert trades.entry_decision_ts.tolist() == [bars.index[29], bars.index[89]]


def test_short_prices_and_direction_gate():
    bars, upper, lower = session(90)
    trade = simulate_session(bars, upper, lower, use_vwap=False, direction_mode="short").iloc[0]
    assert trade.side == -1
    assert trade.entry_price == 89.5
    assert trade.exit_price == 90.5
    assert trade.net_usd == pytest.approx(-23.36)
    assert simulate_session(bars, upper, lower, use_vwap=False, direction_mode="long").empty


def test_missing_final_close_is_censored():
    bars, upper, lower = session()
    trade = simulate_session(bars.iloc[:-1], upper[:-1], lower[:-1], use_vwap=False).iloc[0]
    assert not trade.valid
    assert trade.exit_reason == "missing_session_end"
    assert pd.isna(trade.exit_ts)
    assert pd.isna(trade.net_usd)
    assert pd.isna(trade.exit_price)


def test_future_suffix_cannot_change_completed_trade():
    bars, upper, lower = session()
    bars.loc[bars.index[32], "low"] = 104
    full = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1)
    prefix = simulate_session(bars.iloc[:40], upper[:40], lower[:40], use_vwap=False, max_entries=1)
    pd.testing.assert_frame_equal(full, prefix)


def test_vwap_gate_is_cumulative_not_final_session_vwap():
    bars, upper, lower = session()
    bars.loc[bars.index[:29], ["open", "high", "low", "close"]] = [120, 121, 119, 120]
    trades = simulate_session(bars, upper, lower, use_vwap=True)
    assert trades.empty  # close remains below accumulated HLC3 VWAP
    assert len(simulate_session(bars, upper, lower, use_vwap=False)) == 1


def test_hourly_cadence_and_entry_cap():
    bars, upper, lower = session()
    bars.loc[bars.index[60], "low"] = 104
    trades = simulate_session(bars, upper, lower, use_vwap=False, cadence_minutes=60, max_entries=1)
    assert len(trades) == 1
    assert trades.iloc[0].entry_decision_ts == bars.index[59]


def test_stop_bar_extremes_after_unknown_exit_do_not_inflate_mfe():
    bars, upper, lower = session()
    bars.loc[bars.index[30], ["high", "low"]] = [500, 104]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.mfe_points == 0
    assert trade.mae_points == 5.5


@pytest.mark.parametrize("problem", ["unordered", "duplicate", "unaligned", "negative_volume"])
def test_invalid_input_fails_closed(problem):
    bars, upper, lower = session()
    if problem == "unordered":
        bars = bars.iloc[::-1]
    elif problem == "duplicate":
        bars.index = pd.DatetimeIndex([bars.index[0]] + list(bars.index[:-1]))
    elif problem == "unaligned":
        upper = upper[:-1]
    else:
        bars.loc[bars.index[0], "volume"] = -1
    with pytest.raises(ValueError):
        simulate_session(bars, upper, lower)
