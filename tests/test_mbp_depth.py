"""5s mean MBP-10 depth imbalance (mbp_*)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features.mbp_depth import (
    MBP_COLS,
    aggregate_timestep,
    compute,
    event_depth,
    level_sum_sql,
)


def _events(ts, bid_sz0, ask_sz0, bid_sz1=None, ask_sz1=None):
    n = len(ts)
    df = pd.DataFrame({
        "bid_sz_00": bid_sz0,
        "ask_sz_00": ask_sz0,
        "bid_sz_01": bid_sz1 if bid_sz1 is not None else np.zeros(n),
        "ask_sz_01": ask_sz1 if ask_sz1 is not None else np.zeros(n),
    }, index=pd.DatetimeIndex(ts, tz="UTC"))
    return df


def test_level_sum_sql_coalesce_ten_levels():
    bid = level_sum_sql("bid")
    assert "COALESCE(bid_sz_00, 0)::float8" in bid
    assert "bid_sz_09" in bid
    assert "ask_" not in bid
    with pytest.raises(ValueError):
        level_sum_sql("mid")


def test_event_depth_sums_null_as_zero_and_skips_log_on_zero():
    ts = pd.date_range("2026-08-20 13:30:00", periods=3, freq="1s", tz="UTC")
    df = _events(ts, [10, 0, 40], [10, 10, 20], [5, None, 0], [0, 0, 0])
    # pandas None in float col → NaN, treated as 0
    df.loc[ts[1], "bid_sz_01"] = np.nan
    ev = event_depth(df)
    np.testing.assert_allclose(ev["bid_count"].to_numpy(), [15, 0, 40])
    np.testing.assert_allclose(ev["ask_count"].to_numpy(), [10, 10, 20])
    np.testing.assert_allclose(ev["imbalance_diff"].to_numpy(), [5, -10, 20])
    assert ev["imbalance_log"].iloc[1] != ev["imbalance_log"].iloc[1]  # NaN
    np.testing.assert_allclose(ev["imbalance_log"].iloc[0], math.log(15) - math.log(10))
    np.testing.assert_allclose(ev["imbalance_log"].iloc[2], math.log(40) - math.log(20))


def test_aggregate_5s_mean_uses_own_n_for_log():
    ts = pd.date_range("2026-08-20 13:30:00", periods=4, freq="1s", tz="UTC")
    df = _events(ts, [10, 20, 0, 40], [10, 10, 10, 10])
    bars = aggregate_timestep(event_depth(df), "5s")
    row = bars.iloc[0]
    np.testing.assert_allclose(row["bid_count"], (10 + 20 + 0 + 40) / 4)
    np.testing.assert_allclose(row["ask_count"], 10.0)
    np.testing.assert_allclose(row["imbalance_diff"], (0 + 10 + -10 + 30) / 4)
    expected_log = (
        (math.log(10) - math.log(10))
        + (math.log(20) - math.log(10))
        + (math.log(40) - math.log(10))
    ) / 3
    np.testing.assert_allclose(row["imbalance_log"], expected_log)


def test_compute_renames_and_reindexes():
    ts = pd.date_range("2026-08-20 13:30:00", periods=2, freq="5s", tz="UTC")
    bars = pd.DataFrame({
        "ts": ts,
        "bid_count": [12.0, 8.0],
        "ask_count": [10.0, 10.0],
        "imbalance_diff": [2.0, -2.0],
        "imbalance_log": [0.1, -0.2],
    })
    grid = pd.date_range("2026-08-20 13:30:00", periods=3, freq="5s", tz="UTC")
    out = compute(bars, grid=grid)
    assert list(out.columns) == list(MBP_COLS)
    assert len(out) == 3
    np.testing.assert_allclose(out["mbp_bid_count"].iloc[0], 12.0)
    assert np.isnan(out["mbp_bid_count"].iloc[2])


def test_compute_empty_with_grid_is_all_nan():
    grid = pd.date_range("2026-08-20 13:30:00", periods=4, freq="5s", tz="UTC")
    out = compute(pd.DataFrame(), grid=grid)
    assert list(out.columns) == list(MBP_COLS)
    assert out.isna().all().all()
