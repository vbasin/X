"""Clock-only comparator execution, population and paired-inference checks."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.compare_nq_cadence_quotes import (
    benchmark_session, paired_bootstrap, select_phase_sources,
)


def quotes(times: list[str], bid: list[float], ask: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(f"2026-07-02 {value}", tz="America/Chicago")
                              for value in times]).tz_convert("UTC")
    return pd.DataFrame({"bid_px": bid, "ask_px": ask}, index=index)


def test_latency_costs_and_chronological_liquidation_marks() -> None:
    tape = quotes(["09:00:00", "09:00:00.249", "09:00:00.250", "10:00:00",
                   "12:00:00", "15:00:00", "15:00:00.250"],
                  [99, 99, 100, 110, 95, 101, 102], [100, 100, 101, 111, 96, 102, 103])
    trade, marks = benchmark_session(tape, "2026-07-02")
    assert trade["valid"]
    assert trade["entry_price"] == 101.25
    assert trade["exit_price"] == 101.75
    assert trade["net_usd"] == pytest.approx(6.64)
    assert marks["terminal_liquidation_usd"] == pytest.approx(6.64)
    assert marks["observed_intratrade_drawdown_usd"] == pytest.approx(300.)
    assert trade["entry_ts"].tz_convert("America/Chicago").strftime("%H:%M:%S.%f") == "09:00:00.250000"


def test_no_entry_after_five_seconds_and_bad_quote_skipped() -> None:
    tape = quotes(["09:00:00.250", "09:00:05.001", "15:00:00.250"],
                  [102, 100, 104], [101, 101, 105])
    trade, marks = benchmark_session(tape, "2026-07-02")
    assert not trade["valid"]
    assert trade["exit_reason"] == "no_entry_within_5s"
    assert marks is None


def test_missing_exit_stays_unresolved() -> None:
    tape = quotes(["09:00:00.250", "15:15:00.001"], [100, 104], [101, 105])
    trade, marks = benchmark_session(tape, "2026-07-02")
    assert not trade["valid"]
    assert trade["exit_reason"] == "missing_session_end"
    assert pd.isna(trade["net_usd"])
    assert marks is None


def test_holdout_inventory_context_is_not_a_trading_population() -> None:
    inventory = [{"session": "2025-06-02"}, {"session": "2026-06-30"}, {"session": "2026-07-02"}]
    selected = select_phase_sources(inventory, ["2026-07-02"], "holdout")
    assert selected == [{"session": "2026-07-02"}]
    with pytest.raises(ValueError, match="reserved phase boundary"):
        select_phase_sources(inventory, ["2026-06-30", "2026-07-02"], "holdout")


def test_paired_bootstrap_resamples_differences_together() -> None:
    baseline = np.arange(20.) * 100
    result = paired_bootstrap(baseline + 5., baseline)
    assert result["paired_mean_daily_95ci"] == [5., 5.]
    assert result["model_minus_benchmark_total_usd"] == 100.
    assert result == paired_bootstrap(baseline + 5., baseline)
