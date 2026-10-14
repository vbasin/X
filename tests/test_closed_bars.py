"""Closed-bar visibility: 1-min shift, opening range running, live 1m drop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.candle_structure import compute, compute_multitimeframe
from src.features.closed_bars import (
    align_closed_1m_to_grid,
    drop_incomplete_1m,
    opening_range_running,
    shift_closed_1m,
)
from src.features.live_runtime import tbbo_to_1m_candles
from src.features.time_context import compute as tc_compute
from src.features.wavelets import compute as wv_compute


def _1m_session(n=40, start="2026-06-02 13:30", close_up=True):
    """UTC 1-min candles. 13:30 UTC = 08:30 America/Chicago (CDT)."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.arange(n, dtype="float64") * (0.5 if close_up else -0.5)
    open_ = close - 0.25
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 10.0},
        index=idx,
    )


def test_15m_clv_no_lookahead():
    idx = pd.date_range("2026-06-02 14:00", periods=16, freq="1min", tz="UTC")
    close = np.full(16, 100.0)
    high = np.full(16, 100.2)
    low = np.full(16, 99.8)
    high[14] = 110.0
    close[14] = 110.0
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    feats = compute_multitimeframe(candles, timeframes=["15min"])
    before = feats.loc[idx[idx < pd.Timestamp("2026-06-02 14:15", tz="UTC")], "cs_15m_clv"]
    assert before.isna().all() or (before.abs() < 0.5).all()
    at_close = feats.loc[pd.Timestamp("2026-06-02 14:15", tz="UTC"), "cs_15m_clv"]
    assert float(at_close) > 0.8


def test_30m_clv_no_lookahead():
    idx = pd.date_range("2026-06-02 14:00", periods=31, freq="1min", tz="UTC")
    close = np.full(31, 100.0)
    high = np.full(31, 100.2)
    low = np.full(31, 99.8)
    high[29] = 110.0
    close[29] = 110.0
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    feats = compute_multitimeframe(candles, timeframes=["30min"])
    before = feats.loc[idx[idx < pd.Timestamp("2026-06-02 14:30", tz="UTC")], "cs_30m_clv"]
    assert before.isna().all() or (before.abs() < 0.5).all()
    at_close = feats.loc[pd.Timestamp("2026-06-02 14:30", tz="UTC"), "cs_30m_clv"]
    assert float(at_close) > 0.8


def test_or_nan_before_rth():
    candles = _1m_session(n=40)
    grid = pd.date_range("2026-06-02 13:00:00", periods=12 * 30, freq="5s", tz="UTC")
    or_ = opening_range_running(candles, grid)
    pre = or_.loc[pd.Timestamp("2026-06-02 13:05:00", tz="UTC")]
    assert np.isnan(pre["tc_or5_high"])
    assert np.isnan(pre["tc_or15_high"])
    assert np.isnan(pre["tc_or30_high"])


def test_or_running_during_window():
    """08:32 CT sees closed 08:30 minute only — not a spike at 08:34."""
    idx = pd.date_range("2026-06-02 13:30", periods=10, freq="1min", tz="UTC")
    close = np.full(10, 100.0)
    high = np.full(10, 100.5)
    low = np.full(10, 99.5)
    high[0] = 101.0   # 13:30 = 08:30 CT, visible at 13:31
    high[4] = 110.0   # 13:34 = 08:34 CT, visible at 13:35
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    grid = pd.date_range("2026-06-02 13:30:00", periods=12 * 10, freq="5s", tz="UTC")
    or_ = opening_range_running(candles, grid)
    at_832 = or_.loc[pd.Timestamp("2026-06-02 13:32:00", tz="UTC"), "tc_or5_high"]
    assert float(at_832) == 101.0
    at_834 = or_.loc[pd.Timestamp("2026-06-02 13:34:00", tz="UTC"), "tc_or5_high"]
    assert float(at_834) == 101.0


def test_or_frozen_after_window():
    idx = pd.date_range("2026-06-02 13:30", periods=20, freq="1min", tz="UTC")
    close = np.full(20, 100.0)
    high = np.full(20, 100.5)
    low = np.full(20, 99.5)
    high[0] = 101.0
    high[2] = 103.0   # 13:32, in 5m OR, visible 13:33
    high[10] = 120.0  # 13:40, after 5m OR — must not enter OR5
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    grid = pd.date_range("2026-06-02 13:30:00", periods=12 * 20, freq="5s", tz="UTC")
    or_ = opening_range_running(candles, grid)
    after = or_.loc[pd.Timestamp("2026-06-02 13:36:00", tz="UTC"), "tc_or5_high"]
    later = or_.loc[pd.Timestamp("2026-06-02 13:45:00", tz="UTC"), "tc_or5_high"]
    assert float(after) == 103.0
    assert float(later) == 103.0


def test_cs_clv_1m_visible_at_close():
    idx = pd.date_range("2026-06-02 13:08", periods=5, freq="1min", tz="UTC")
    # 13:09 candle: close near high. 13:10 candle: close near low (forming).
    high = np.array([100.2, 100.2, 100.2, 110.0, 100.2])
    low = np.array([99.8, 99.8, 99.8, 99.8, 99.8])
    close = np.array([100.0, 100.1, 100.15, 99.85, 100.0])
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    raw = compute(candles)
    grid = pd.date_range("2026-06-02 13:08:00", periods=5 * 12, freq="5s", tz="UTC")
    aligned = align_closed_1m_to_grid(raw[["cs_clv"]], grid)
    t_mid = pd.Timestamp("2026-06-02 13:10:25", tz="UTC")
    t_prev_close = pd.Timestamp("2026-06-02 13:10:00", tz="UTC")
    got = float(aligned.loc[t_mid, "cs_clv"])
    prev = float(raw.loc[pd.Timestamp("2026-06-02 13:09:00", tz="UTC"), "cs_clv"])
    forming = float(raw.loc[pd.Timestamp("2026-06-02 13:10:00", tz="UTC"), "cs_clv"])
    assert np.isclose(got, prev)
    assert not np.isclose(got, forming)
    assert np.isclose(float(aligned.loc[t_prev_close, "cs_clv"]), prev)


def test_wavelets_not_shifted():
    candles = _1m_session(n=160, start="2026-06-02 12:00")
    wv = wv_compute(candles)
    t0 = pd.Timestamp("2026-06-02 14:10:00", tz="UTC")
    t_mid = pd.Timestamp("2026-06-02 14:10:25", tz="UTC")
    grid = pd.date_range("2026-06-02 14:00:00", periods=12 * 20, freq="5s", tz="UTC")
    aligned = wv.reindex(grid, method="ffill")
    assert t0 in wv.index
    assert np.isfinite(wv.loc[t0, "wv_detail_1"])
    assert np.isclose(
        float(aligned.loc[t_mid, "wv_detail_1"]),
        float(wv.loc[t0, "wv_detail_1"]),
        equal_nan=True,
    )


def test_live_drops_incomplete_1m():
    idx = pd.date_range("2026-06-02 14:00:00", periods=90, freq="1s", tz="UTC")
    tbbo = pd.DataFrame({"ts_event": idx, "price": 100.0 + np.arange(90), "size": 1.0})
    # Last print 14:01:29 — forming 14:01 bucket must not appear.
    got = tbbo_to_1m_candles(tbbo)
    assert got.index.max() == pd.Timestamp("2026-06-02 14:00:00", tz="UTC")
    raw = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0},
        index=pd.date_range("2026-06-02 14:00", periods=3, freq="1min", tz="UTC"),
    )
    closed = drop_incomplete_1m(
        raw, as_of=pd.Timestamp("2026-06-02 14:01:25", tz="UTC"),
    )
    assert closed.index.max() == pd.Timestamp("2026-06-02 14:00:00", tz="UTC")
    closed_late = drop_incomplete_1m(
        raw, as_of=pd.Timestamp("2026-06-02 14:02:00", tz="UTC"),
    )
    assert closed_late.index.max() == pd.Timestamp("2026-06-02 14:01:00", tz="UTC")


def test_time_context_uses_running_or():
    idx = pd.date_range("2026-06-02 13:30", periods=10, freq="1min", tz="UTC")
    high = np.full(10, 100.5)
    high[0] = 101.0
    high[4] = 110.0
    candles = pd.DataFrame({
        "open": 100.0, "high": high, "low": 99.5, "close": 100.0, "volume": 1.0,
    }, index=idx)
    grid = pd.date_range("2026-06-02 13:00:00", periods=12 * 50, freq="5s", tz="UTC")
    close = pd.Series(100.0, index=grid)
    feats = tc_compute(grid, candles=candles, close_series=close)
    pre = feats.loc[pd.Timestamp("2026-06-02 13:05:00", tz="UTC"), "tc_or5_high"]
    mid = feats.loc[pd.Timestamp("2026-06-02 13:32:00", tz="UTC"), "tc_or5_high"]
    assert np.isnan(pre)
    assert float(mid) == 101.0


def test_shift_closed_1m_empty():
    empty = pd.DataFrame()
    assert shift_closed_1m(empty).empty
