"""Independent ordered liquidation audit using only synthetic quote fixtures."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.audit_nq_cadence_marks import (
    aggregate_drawdown, audit_session, digest, match_fill_event, summarize_marks,
)


def test_drawdown_respects_order_not_unordered_high_minus_low():
    stats = summarize_marks(np.array([-100.0, 500.0, 450.0]))
    assert stats["observed_intratrade_drawdown_usd"] == 100
    assert stats["min_liquidation_usd"] == -100
    assert stats["max_liquidation_usd"] == 500


def test_cross_trade_peak_and_internal_drawdown_match_full_ordered_curve():
    paths = [np.array([300.0, 100.0]), np.array([-100.0, 500.0, 450.0]),
             np.array([200.0, -400.0, -100.0])]
    trades = [{**summarize_marks(path), "net_usd": float(path[-1])} for path in paths]
    result = aggregate_drawdown(trades)
    equity = [0.0]
    realized = 0.0
    for path in paths:
        equity.extend(realized + path)
        realized += path[-1]
    equity = np.asarray(equity)
    assert result["net_usd"] == realized
    assert result["full_observed_liquidation_mark_drawdown_usd"] == pytest.approx(
        np.max(np.maximum.accumulate(equity) - equity))


def test_previous_trade_peak_survives_partial_profit_giveback():
    trades = [{**summarize_marks(np.array([500., 100.])), "net_usd": 100.},
              {**summarize_marks(np.array([-100., 50.])), "net_usd": 50.}]
    assert aggregate_drawdown(trades)["full_observed_liquidation_mark_drawdown_usd"] == 500


def test_duplicate_timestamp_matching_preserves_price_and_order():
    times = np.array([1, 1, 1, 2, 2])
    prices = np.array([99., 100., 100., 102., 103.])
    assert match_fill_event(times, prices, 1, 100.) == (1, 2)
    assert match_fill_event(times, prices, 1, 100., minimum_index=2) == (2, 1)
    assert match_fill_event(times, prices, 2, 103.) == (4, 1)
    with pytest.raises(ValueError):
        match_fill_event(times, prices, 2, 100.)


def test_synthetic_source_hash_fill_prices_and_liquidation_marks(tmp_path):
    decision = pd.Timestamp("2025-01-02T15:00:00Z")
    entry = decision + pd.Timedelta(seconds=1)
    exit_ts = decision + pd.Timedelta(seconds=2)
    index = pd.DatetimeIndex([entry, entry, entry, exit_ts, exit_ts, exit_ts])
    tape = pd.DataFrame({"bid_px": [100., 101., 101., 102., 103., 103.],
                         "ask_px": [100.25, 101.25, 101.25, 102.25, 103.25, 103.25]}, index=index)
    path = tmp_path / "synthetic.parquet"
    tape.to_parquet(path)
    source = {"path": str(path), "sha256": digest(path), "rows": len(tape), "session": "2025-01-02"}
    trade = {"trade_id": 0, "entry_decision_ts": decision, "entry_ts": entry,
             "exit_ts": exit_ts, "side": 1, "entry_price": 101.5,
             "exit_price": 102.75, "net_usd": 21.64}
    rows, verified = audit_session((source, [trade]))
    assert verified["hash_verified"]
    assert len(rows) == 1
    assert rows[0]["quote_events"] == 4
    assert rows[0]["ambiguous_fill_events"] == 2
    assert rows[0]["terminal_liquidation_usd"] == pytest.approx(21.64)
    assert rows[0]["min_liquidation_usd"] == pytest.approx(-18.36)
    assert rows[0]["maximum_observed_quote_gap_seconds"] == 1
    corrupted = {**source, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_session((corrupted, [trade]))


@pytest.mark.parametrize("marks", [[], [np.nan], [np.inf]])
def test_missing_or_nonfinite_marks_are_not_fabricated(marks):
    with pytest.raises(ValueError):
        summarize_marks(np.asarray(marks))


def test_short_liquidation_uses_ask_plus_exit_slip_and_one_fee(tmp_path):
    decision = pd.Timestamp("2025-01-02T15:00:00Z")
    index = pd.DatetimeIndex([decision + pd.Timedelta(seconds=1),
                              decision + pd.Timedelta(seconds=2)])
    tape = pd.DataFrame({"bid_px": [102., 100.], "ask_px": [102.25, 100.25]}, index=index)
    path = tmp_path / "short_synthetic.parquet"
    tape.to_parquet(path)
    source = {"path": str(path), "sha256": digest(path), "rows": 2, "session": "2025-01-02"}
    trade = {"trade_id": 0, "entry_decision_ts": decision, "entry_ts": index[0],
             "exit_ts": index[1], "side": -1, "entry_price": 101.75,
             "exit_price": 100.5, "net_usd": 21.64}
    rows, _ = audit_session((source, [trade]))
    assert rows[0]["terminal_liquidation_usd"] == pytest.approx(21.64)
    assert rows[0]["min_liquidation_usd"] == pytest.approx(-18.36)
    assert rows[0]["ambiguous_fill_events"] == 0
