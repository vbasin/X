"""DB-free execution checks with hand-calculated quote paths."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.nq_selective import simulate_candidates


START = pd.Timestamp("2026-01-05T14:30:00Z")


def tape(seconds, bids, asks):
    return pd.DataFrame(
        {"bid_px": bids, "ask_px": asks},
        index=START + pd.to_timedelta(seconds, unit="s"),
    )


def decisions(stop=10.0):
    return pd.DataFrame({"stop_points": [stop]}, index=pd.DatetimeIndex([START]))


def test_long_short_spread_slippage_and_roundtrip_fee():
    result = simulate_candidates(
        tape([0.25, 60.25], [100, 102], [100.5, 102.5]),
        decisions(), horizon_minutes=1,
    )
    long, short = result.iloc[0], result.iloc[1]
    assert result.valid.all()
    assert long.entry_price == 100.75
    assert long.exit_price == 101.75
    assert short.entry_price == 99.75
    assert short.exit_price == 102.75
    assert long.net_usd == pytest.approx(16.64)
    assert short.net_usd == pytest.approx(-63.36)
    assert long.net_r == pytest.approx(16.64 / 200)
    assert long.mae_points == 0.75
    assert short.mae_points == 2.75
    assert long.exit_reason == "time"
    assert long.max_quote_gap_seconds == 60


def test_stop_gap_executes_at_later_quote_without_clamping():
    result = simulate_candidates(
        tape([0.25, 1, 1.1, 1.25, 60.25],
             [100, 96, 95, 93, 93], [100.5, 96.5, 95.5, 93.5, 93.5]),
        decisions(2), horizon_minutes=1,
    )
    long = result.iloc[0]
    assert long.valid
    assert long.exit_reason == "stop"
    assert long.exit_ts == START + pd.Timedelta(seconds=1.25)
    assert long.exit_price == 92.75
    assert long.gross_points == -8
    assert long.gross_points < -long.stop_points


def test_entry_and_target_latency_and_market_target_price():
    result = simulate_candidates(
        tape([0, 0.1, 0.25, 1, 1.1, 1.25, 60.25],
             [80, 90, 100, 105, 106, 104, 104],
             [80.5, 90.5, 100.5, 105.5, 106.5, 104.5, 104.5]),
        decisions(2), horizon_minutes=1,
    )
    long = result.iloc[0]
    assert long.entry_ts == START + pd.Timedelta(milliseconds=250)
    assert long.entry_price == 100.75
    assert long.exit_reason == "target"
    assert long.exit_price == 103.75  # reversal during latency preserved
    assert long.exit_ts == START + pd.Timedelta(seconds=1.25)


def test_missing_exit_is_explicit_and_does_not_flatten_at_last_quote():
    result = simulate_candidates(
        tape([0.25, 30], [100, 101], [100.5, 101.5]),
        decisions(), horizon_minutes=1,
    )
    assert not result.valid.any()
    assert (result.invalid_reason == "missing_time_exit_within_5s").all()
    assert result.exit_ts.isna().all()
    assert result.net_usd.isna().all()
    assert result.entry_price.notna().all()


def test_identical_timestamps_keep_event_order_and_exit_next_event():
    result = simulate_candidates(
        tape([0, 1, 1, 1, 60], [100, 96, 94, 99, 100],
             [100.5, 96.5, 94.5, 99.5, 100.5]),
        decisions(2), horizon_minutes=1, latency_ms=0,
    )
    long = result.iloc[0]
    assert long.exit_reason == "stop"
    assert long.exit_price == 93.75
    assert long.exit_ts == START + pd.Timedelta(seconds=1)


def test_invalid_quotes_ignored_and_late_time_exit_censored():
    result = simulate_candidates(
        tape([0.25, 0.3, 0.4, 0.5, 70],
             [np.nan, 100, 0, 100, 101], [101, 99, 1, 100.5, 101.5]),
        decisions(), horizon_minutes=1,
    )
    assert (result.entry_ts == START + pd.Timedelta(seconds=0.5)).all()
    assert not result.valid.any()
    assert result.net_usd.isna().all()


def test_missing_barrier_execution_is_explicit():
    result = simulate_candidates(
        tape([0.25, 1], [100, 90], [100.5, 90.5]),
        decisions(2), horizon_minutes=1,
    )
    assert result.iloc[0].invalid_reason == "missing_barrier_exit"
    assert not result.iloc[0].valid


def test_entry_deadline_invalid_stop_empty_tape_and_validation():
    quotes = tape([6], [100], [100.5])
    assert (simulate_candidates(quotes, decisions()).invalid_reason
            == "no_entry_within_5s").all()
    assert (simulate_candidates(quotes, decisions(0)).invalid_reason
            == "invalid_stop").all()
    assert not simulate_candidates(quotes.iloc[:0], decisions()).valid.any()
    with pytest.raises(ValueError, match="sorted"):
        simulate_candidates(tape([2, 1], [100, 100], [101, 101]), decisions())
    with pytest.raises(ValueError, match="timezone"):
        naive = decisions()
        naive.index = naive.index.tz_localize(None)
        simulate_candidates(quotes, naive)
