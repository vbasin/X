"""Causal cadence exits and fixed emergency-stop execution invariants."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.nq_cadence_area import simulate_session


def session(price=110.0):
    index = pd.date_range("2025-01-02 08:31", periods=390, freq="min", tz="America/Chicago").tz_convert("UTC")
    bars = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 10.0}, index=index)
    return bars, np.full(390, 105.0), np.full(390, 95.0)


def test_entry_next_open_and_close_flatten_with_full_cost_path():
    bars, upper, lower = session()
    bars.loc[bars.index[30], ["open", "high"]] = [112, 113]
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.entry_decision_ts == bars.index[29]
    assert trade.entry_ts == bars.index[30] - pd.Timedelta(minutes=1)
    assert trade.entry_price == 112.5
    assert trade.exit_price == 109.5
    assert trade.net_usd == pytest.approx(-63.36)
    assert trade.exit_ts == bars.index[-1]
    assert trade.exit_reason == "session_close"
    assert trade.stop_initial == 95


def test_soft_exit_ignores_intrabar_and_noncadence_failure():
    bars, upper, lower = session()
    bars.loc[bars.index[40], ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.exit_reason == "session_close"
    assert trade.exit_ts == bars.index[-1]


def test_soft_exit_uses_next_open_not_decision_close():
    bars, upper, lower = session()
    bars.loc[bars.index[59], ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    bars.loc[bars.index[60], ["open", "low"]] = [103, 102]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.exit_reason == "cadence_exit"
    assert trade.exit_price == 102.5
    assert trade.exit_ts == bars.index[60] - pd.Timedelta(minutes=1)


def test_soft_boundary_does_not_ratchet():
    bars, upper, lower = session()
    upper[40] = 109.75
    bars.loc[bars.index[59], ["open", "high", "low", "close"]] = [108, 109, 107, 108]
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.exit_reason == "session_close"  # current upper is 105


def test_fixed_emergency_stop_does_not_follow_later_opposite_band():
    bars, upper, lower = session()
    lower[40:] = 107
    bars.loc[bars.index[41], "low"] = 100
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.stop_initial == 95
    assert trade.exit_reason == "session_close"


def test_emergency_gap_uses_worse_actual_open():
    bars, upper, lower = session()
    bars.loc[bars.index[31], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.exit_reason == "emergency_stop"
    assert trade.exit_price == 89.5
    assert trade.exit_ts == bars.index[31] - pd.Timedelta(minutes=1)


def test_pending_market_exit_precedes_emergency_check():
    bars, upper, lower = session()
    bars.loc[bars.index[59], ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    bars.loc[bars.index[60], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.exit_reason == "cadence_exit"
    assert trade.exit_price == 89.5


def test_opposite_signal_exits_without_same_decision_flip():
    bars, upper, lower = session()
    bars.loc[bars.index[59]:, ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    trades = simulate_session(bars, upper, lower, use_vwap=False, emergency_stop=False)
    assert trades.side.tolist() == [1, -1]
    assert trades.iloc[0].exit_reason == "cadence_exit"
    assert trades.iloc[1].entry_decision_ts == bars.index[89]
    assert pd.isna(trades.iloc[0].stop_initial)


def test_emergency_can_be_disabled_without_hidden_intrabar_exit():
    bars, upper, lower = session()
    bars.loc[bars.index[40], "low"] = 50
    trade = simulate_session(bars, upper, lower, use_vwap=False, emergency_stop=False).iloc[0]
    assert trade.exit_reason == "session_close"
    assert pd.isna(trade.stop_initial)
    assert trade.mae_points == 60.5


def test_exits_continue_after_last_allowed_entry_clock():
    bars, upper, lower = session()
    upper[359:] = 115
    trade = simulate_session(bars, upper, lower, use_vwap=False).iloc[0]
    assert trade.exit_reason == "cadence_exit"
    assert trade.exit_ts == bars.index[360] - pd.Timedelta(minutes=1)


def test_missing_final_close_censors_open_position():
    bars, upper, lower = session()
    trade = simulate_session(bars.iloc[:-1], upper[:-1], lower[:-1], use_vwap=False).iloc[0]
    assert not trade.valid
    assert trade.exit_reason == "missing_session_end"
    assert pd.isna(trade.net_usd)
    assert pd.isna(trade.exit_ts)


def test_two_losses_disable_later_entries():
    bars, upper, lower = session()
    bars.loc[bars.index[[30, 60, 90]], "low"] = 94
    trades = simulate_session(bars, upper, lower, use_vwap=False)
    assert len(trades) == 2
    assert (trades.net_usd < 0).all()


@pytest.mark.parametrize("side", [1, -1])
def test_emergency_stop_uses_opposite_band_and_adverse_tick(side):
    bars, upper, lower = session(110 if side == 1 else 90)
    if side == 1:
        lower[:] = 95.13
        bars.loc[bars.index[30], "low"] = 94
        expected = 95
    else:
        upper[:] = 104.87
        bars.loc[bars.index[30], "high"] = 106
        expected = 105
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1,
                             roundtrip_cost_points=0).iloc[0]
    assert trade.stop_initial == expected
    assert trade.exit_price == expected


def test_future_suffix_cannot_change_completed_trade():
    bars, upper, lower = session()
    bars.loc[bars.index[32], "low"] = 94
    full = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1)
    prefix = simulate_session(bars.iloc[:40], upper[:40], lower[:40], use_vwap=False, max_entries=1)
    pd.testing.assert_frame_equal(full, prefix)


def test_hard_stop_excursions_exclude_later_unknown_bar_extreme():
    bars, upper, lower = session()
    bars.loc[bars.index[30], ["high", "low"]] = [500, 94]
    trade = simulate_session(bars, upper, lower, use_vwap=False, max_entries=1).iloc[0]
    assert trade.mfe_points == 0
    assert trade.mae_points == 15.5


def test_hourly_entry_cadence_and_direction_filter():
    bars, upper, lower = session()
    trade = simulate_session(bars, upper, lower, use_vwap=False, cadence_minutes=60).iloc[0]
    assert trade.entry_decision_ts == bars.index[59]
    assert simulate_session(bars, upper, lower, use_vwap=False, direction_mode="short").empty


def test_vwap_is_causal_session_cumulative():
    bars, upper, lower = session()
    bars.loc[bars.index[:29], ["open", "high", "low", "close"]] = [120, 121, 119, 120]
    assert simulate_session(bars, upper, lower, use_vwap=True).empty
    assert len(simulate_session(bars, upper, lower, use_vwap=False)) == 1
