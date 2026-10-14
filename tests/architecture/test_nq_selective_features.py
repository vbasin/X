"""Synthetic adversarial causality tests; no market files or database needed."""

import numpy as np
import pandas as pd
import pytest

from src.features.nq_selective import build_features


@pytest.fixture(scope="module")
def tape() -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-06 07:00:00", "2026-01-06 15:15:00", freq="1s",
        tz="America/Chicago",
    ).tz_convert("UTC")
    elapsed = np.arange(len(index))
    price = np.round((20000 + elapsed * .002 + 2 * np.sin(elapsed / 40)) * 4) / 4
    return pd.DataFrame(
        {"price": price, "size": 1 + elapsed % 7,
         "side": np.where(elapsed % 3 == 0, -1, 1),
         "bid_px": price - .25, "ask_px": price + .25,
         "bid_sz": 10 + elapsed % 5, "ask_sz": 12 + elapsed % 3},
        index=index.rename("ts"),
    )


def timestamp(clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-06 {clock}", tz="America/Chicago").tz_convert("UTC")


@pytest.mark.parametrize("clock", ["08:45:00", "09:32:40", "11:00:00", "14:25:00"])
def test_every_returned_prefix_row_equals_full_session(tape: pd.DataFrame, clock: str) -> None:
    full = build_features(tape)
    cutoff = timestamp(clock)
    prefix = build_features(tape.loc[:cutoff])
    assert not prefix.empty
    pd.testing.assert_frame_equal(prefix, full.loc[:cutoff], check_freq=False)
    assert (prefix["source_ts"] <= prefix.index).all()
    assert prefix.index.max() <= cutoff


def test_adversarial_future_prices_volume_and_quotes_do_not_change_past(tape: pd.DataFrame) -> None:
    cutoff = timestamp("10:15:00")
    changed = tape.copy()
    future = changed.index > cutoff
    changed.loc[future, "price"] *= 5
    changed.loc[future, "size"] *= 1000
    changed.loc[future, "side"] = -1
    changed.loc[future, "bid_px"] += 3000
    changed.loc[future, "ask_px"] += 4000
    changed.loc[future, "bid_sz"] *= 100
    expected = build_features(tape).loc[:cutoff]
    actual = build_features(changed).loc[:cutoff]
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_print_exactly_at_close_belongs_to_next_bar(tape: pd.DataFrame) -> None:
    decision = timestamp("09:30:00")
    changed = tape.copy()
    # Keep the BBO fixed: this changes only the first trade of the next minute.
    changed.loc[decision, "price"] += 1000
    changed.loc[decision, "size"] = 1000000
    expected = build_features(tape).loc[decision]
    actual = build_features(changed).loc[decision]
    pd.testing.assert_series_equal(actual, expected)


def test_incomplete_first_opening_range_never_yields_decision(tape: pd.DataFrame) -> None:
    result = build_features(tape.loc[:timestamp("08:44:59")])
    assert result.empty


def test_stale_decision_quote_cannot_be_repaired_by_future_print(tape: pd.DataFrame) -> None:
    decision = timestamp("09:30:00")
    stale = tape.loc[~((tape.index > decision - pd.Timedelta(seconds=3)) & (tape.index <= decision))]
    assert decision in build_features(tape).index
    assert decision not in build_features(stale).index
    before = decision - pd.Timedelta(minutes=5)
    pd.testing.assert_frame_equal(
        build_features(stale).loc[:before], build_features(tape).loc[:before], check_freq=False,
    )


def test_duplicate_timestamp_uses_last_observed_quote(tape: pd.DataFrame) -> None:
    decision = timestamp("09:30:00")
    replacement = tape.loc[[decision]].copy()
    replacement.loc[:, "bid_sz"] = 100
    replacement.loc[:, "ask_sz"] = 1
    tied = pd.concat([tape.loc[:decision], replacement, tape.loc[tape.index > decision]])
    result = build_features(tied)
    assert result.loc[decision, "nq_quote_imbalance"] == pytest.approx(99 / 101)
    assert result.loc[decision, "source_ts"] == decision


def test_no_next_day_state_is_accepted(tape: pd.DataFrame) -> None:
    tomorrow = tape.iloc[:1].copy()
    tomorrow.index += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="one Chicago calendar date"):
        build_features(pd.concat([tape, tomorrow]))
