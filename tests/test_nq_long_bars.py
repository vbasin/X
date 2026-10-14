"""Availability and input-integrity checks for long-history research bars."""
import numpy as np
import pandas as pd
import pytest

from scripts.research.build_nq_long_bars import aggregate_session


def sample() -> pd.DataFrame:
    index = pd.to_datetime(["2020-06-01T13:30:00Z", "2020-06-01T13:30:59Z",
                            "2020-06-01T13:31:00Z", "2020-06-01T13:33:00Z"])
    price = np.array([100.0, 101.0, 102.0, 103.0])
    return pd.DataFrame({"open": price, "high": price + .25,
                         "low": price - .25, "close": price, "volume": 1}, index=index)


def test_bar_close_availability_and_exclusive_boundary() -> None:
    bars = aggregate_session(sample(), "2020-06-01")
    first = bars.iloc[0]
    assert bars.index[0] == pd.Timestamp("2020-06-01T13:31:00Z")
    assert first.bar_open_ts == pd.Timestamp("2020-06-01T13:30:00Z")
    assert first.close == 101.0  # 13:31:00 input second belongs to NEXT minute.
    assert first.high == 101.25
    assert first.volume == 2
    assert first.minute_ct == 510


def test_prefix_invariance_at_availability_cutoff() -> None:
    raw = sample()
    earlier = aggregate_session(raw.iloc[:2], "2020-06-01")
    changed = raw.copy()
    changed.loc[changed.index[2:], ["open", "high", "low", "close"]] += 500
    later = aggregate_session(changed, "2020-06-01")
    pd.testing.assert_frame_equal(earlier, later.loc[earlier.index], check_freq=False)


def test_missing_minutes_are_not_backfilled() -> None:
    bars = aggregate_session(sample(), "2020-06-01")
    assert pd.Timestamp("2020-06-01T13:33:00Z") not in bars.index
    assert len(bars) == 3


@pytest.mark.parametrize("case", ["duplicate", "unsorted", "ohlc", "wrong_date", "nan"])
def test_invalid_sources_fail_closed(case: str) -> None:
    raw = sample()
    session = "2020-06-01"
    if case == "duplicate":
        raw = pd.concat([raw, raw.iloc[[-1]]])
    elif case == "unsorted":
        raw = raw.iloc[::-1]
    elif case == "ohlc":
        raw.loc[raw.index[0], "high"] = 90
    elif case == "wrong_date":
        session = "2020-06-02"
    elif case == "nan":
        raw.loc[raw.index[0], "close"] = np.nan
    with pytest.raises(ValueError):
        aggregate_session(raw, session)
