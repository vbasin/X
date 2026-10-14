"""Causal and calendar invariants for the independent noise-envelope research."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.nq_noise_area import attach_noise_area, eligible_sessions


def history() -> pd.DataFrame:
    """Small deterministic sessions containing all cash minutes and side buffers."""
    frames = []
    for day_number, day in enumerate(pd.bdate_range("2024-01-02", periods=20)):
        index = pd.date_range(f"{day.date()} 08:30", periods=392, freq="min",
                              tz="America/Chicago").tz_convert("UTC")
        opening = 100.0 + day_number
        close = opening + np.arange(len(index)) * (day_number + 1) / 1000
        frame = pd.DataFrame({"open": opening, "high": close + 1,
                              "low": opening - 1, "close": close, "volume": 10},
                             index=index)
        # Match the real prepared-parquet schema, including fields overwritten
        # by the feature function and its named DatetimeIndex.
        frame.index.name = "ts"
        frame["session"] = str(day.date())
        frame["bar_open_ts"] = index - pd.Timedelta(minutes=1)
        frame["minute_ct"] = (frame.bar_open_ts.dt.tz_convert("America/Chicago").dt.hour * 60
                              + frame.bar_open_ts.dt.tz_convert("America/Chicago").dt.minute)
        frame["observed_seconds"] = 60
        frames.append(frame)
    return pd.concat(frames)


def test_future_sessions_cannot_change_prior_envelopes() -> None:
    raw = history()
    cutoff = raw.session.unique()[16]
    baseline = attach_noise_area(raw)
    changed = raw.copy()
    future = changed.session.gt(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 7
    changed.loc[future, "volume"] *= 19
    mutated = attach_noise_area(changed)
    pd.testing.assert_frame_equal(baseline.loc[baseline.session.le(cutoff)],
                                  mutated.loc[mutated.session.le(cutoff)])
    prefix = attach_noise_area(raw.loc[raw.session.le(cutoff)])
    pd.testing.assert_frame_equal(prefix, baseline.loc[prefix.index])


def test_current_session_prices_cannot_change_its_learned_sigma() -> None:
    raw = history()
    today = raw.session.unique()[16]
    baseline = attach_noise_area(raw)
    changed = raw.copy()
    current = changed.session.eq(today)
    # Even the cash-open observation may change the envelope center but must
    # never change the scale learned from completed previous sessions.
    changed.loc[current, ["open", "high", "low", "close"]] *= 4
    mutated = attach_noise_area(changed)
    original_today = baseline.loc[baseline.session.eq(today)]
    changed_today = mutated.loc[mutated.session.eq(today)]
    assert original_today.sigma.notna().all()
    pd.testing.assert_series_equal(original_today.sigma, changed_today.sigma)
    np.testing.assert_allclose(changed_today.upper, original_today.upper * 4)
    np.testing.assert_allclose(changed_today.lower, original_today.lower * 4)


def test_later_current_minutes_cannot_change_prior_intraday_envelopes() -> None:
    raw = history()
    today = raw.session.unique()[16]
    cutoff = pd.Timestamp(f"{today} 11:00", tz="America/Chicago").tz_convert("UTC")
    baseline = attach_noise_area(raw)
    changed = raw.copy()
    changed.loc[changed.index > cutoff, ["high", "low", "close"]] *= 3
    mutated = attach_noise_area(changed)
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], mutated.loc[:cutoff])
    # A truncated current session must produce the same features as the prefix
    # of a complete session, even though its completion is not yet knowable.
    prefix = attach_noise_area(raw.loc[:cutoff])
    pd.testing.assert_frame_equal(prefix, baseline.loc[prefix.index])


def test_fourteen_prior_sessions_and_no_current_observation() -> None:
    raw = history()
    features = attach_noise_area(raw, lookback=14)
    sessions = features.session.unique()
    assert features.loc[features.session.isin(sessions[:14]), "sigma"].isna().all()
    minute = 540
    first14 = features.loc[features.session.isin(sessions[:14]) & features.minute.eq(minute)]
    expected = (first14.close / first14.cash_open - 1).abs().mean()
    row = features.loc[features.session.eq(sessions[14]) & features.minute.eq(minute)].iloc[0]
    assert row.sigma == pytest.approx(expected)
    assert row.scale_last_session == sessions[13]


def test_first_minute_available_at_0831_and_missing_open_is_not_backfilled() -> None:
    raw = history()
    features = attach_noise_area(raw)
    assert features.minute.min() == 511
    assert features.minute.max() == 900
    first = features.iloc[0]
    assert features.index[0].tz_convert("America/Chicago").strftime("%H:%M") == "08:31"
    assert first.bar_open_ts.tz_convert("America/Chicago").strftime("%H:%M") == "08:30"
    assert first.cash_open == raw.loc[features.index[0], "open"]
    latest = raw.session.unique()[-1]
    local = raw.index.tz_convert("America/Chicago")
    missing_open = raw.loc[~(raw.session.eq(latest) & (local.hour == 8) & (local.minute == 31))]
    missing_features = attach_noise_area(missing_open)
    today = missing_features.loc[missing_features.session.eq(latest)]
    assert today.cash_open.isna().all()
    assert today.upper.isna().all()


def test_calendar_excludes_known_expiry_week_holidays_and_short_days() -> None:
    days = set(eligible_sessions("2024-01-01", "2024-12-31"))
    for expiry_monday in ["2024-03-11", "2024-06-17", "2024-09-16", "2024-12-16"]:
        expiry_week = pd.date_range(expiry_monday, periods=5, freq="D").strftime("%Y-%m-%d")
        assert not set(expiry_week) & days
    assert {"2024-03-08", "2024-03-18", "2024-07-02", "2024-07-05"} <= days
    assert not {"2024-01-01", "2024-03-29", "2024-07-03", "2024-07-04",
                "2024-11-28", "2024-11-29", "2024-12-24", "2024-12-25"} & days


def test_duplicate_session_minutes_fail_loudly() -> None:
    raw = history()
    bad = pd.concat([raw, raw.iloc[[1]]])
    with pytest.raises(ValueError, match="duplicate session/minute"):
        attach_noise_area(bad)
