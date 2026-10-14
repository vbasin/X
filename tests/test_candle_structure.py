"""Candle-structure v5.4: 5m/10m CLV, range_ma_6, 15s/30s CLV from TBBO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.candle_structure import (
    CS_5S_OHLC_COLS,
    CS_PATCH_COLS,
    build_cs_patch_frame,
    compute,
    compute_5s_ohlc,
    compute_multitimeframe,
    compute_subminute_clv,
)


def _1m_candles(n=40, start="2026-06-02 14:00", close_up=True):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.arange(n, dtype="float64") * (0.5 if close_up else -0.5)
    open_ = close - 0.25
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 10.0},
        index=idx,
    )


def test_compute_adds_range_ma_6():
    feats = compute(_1m_candles(20))
    assert "cs_range_ma_6" in feats.columns
    rng = _1m_candles(20)
    expect = (rng["high"] - rng["low"]).rolling(6).mean()
    got = feats["cs_range_ma_6"].to_numpy(dtype="float64")
    np.testing.assert_allclose(got[5:], expect.to_numpy()[5:], rtol=1e-5, atol=1e-5)


def test_multitf_emits_5m_10m_clv_and_keeps_15m_30m_body():
    feats = compute_multitimeframe(_1m_candles(40))
    for col in ("cs_5m_clv", "cs_10m_clv", "cs_15m_clv", "cs_30m_clv"):
        assert col in feats.columns, col
    assert "cs_15m_body_range" in feats.columns
    assert "cs_30m_body_range" in feats.columns
    assert "cs_5m_body_range" not in feats.columns
    assert "cs_10m_body_range" not in feats.columns


def test_5m_clv_no_lookahead():
    """A spike only in the last minute of a 5m bucket must not leak earlier."""
    idx = pd.date_range("2026-06-02 14:00", periods=10, freq="1min", tz="UTC")
    close = np.full(10, 100.0)
    high = np.full(10, 100.2)
    low = np.full(10, 99.8)
    # Minute labeled 14:04 is inside [14:00, 14:05); spike high at 14:04.
    high[4] = 110.0
    close[4] = 110.0
    candles = pd.DataFrame(
        {"open": 100.0, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    feats = compute_multitimeframe(candles, timeframes=["5min"])
    # label=right stamps the completed [14:00, 14:05) bar at 14:05.
    before = feats.loc[idx[idx < pd.Timestamp("2026-06-02 14:05", tz="UTC")], "cs_5m_clv"]
    assert before.isna().all() or (before.abs() < 0.5).all()
    at_close = feats.loc[pd.Timestamp("2026-06-02 14:05", tz="UTC"), "cs_5m_clv"]
    assert at_close > 0.8


def test_subminute_clv_from_tbbo_and_no_lookahead():
    # 60 seconds of 1s prints, flat then a late spike inside the first 15s.
    idx = pd.date_range("2026-06-02 14:00:00", periods=60, freq="1s", tz="UTC")
    price = np.full(60, 100.0)
    price[14] = 110.0  # last second of [14:00:00, 14:00:15)
    tbbo = pd.DataFrame({"ts_event": idx, "price": price, "size": 1.0})
    feats = compute_subminute_clv(tbbo, timestep="5s")
    assert "cs_15s_clv" in feats.columns
    assert "cs_30s_clv" in feats.columns
    # 5s slots before 14:00:15 must not see the 110 print.
    early = feats.loc[feats.index < pd.Timestamp("2026-06-02 14:00:15", tz="UTC"), "cs_15s_clv"]
    assert early.isna().all()
    at_15 = feats.loc[pd.Timestamp("2026-06-02 14:00:15", tz="UTC"), "cs_15s_clv"]
    assert float(at_15) > 0.8


def test_subminute_empty_tbbo():
    assert compute_subminute_clv(pd.DataFrame()).empty


def _run_lengths(s: pd.Series) -> list[int]:
    vals = s.to_numpy()
    if len(vals) == 0:
        return []
    lengths = []
    n = 1
    for i in range(1, len(vals)):
        a, b = vals[i - 1], vals[i]
        same = (a == b) or (pd.isna(a) and pd.isna(b))
        if same:
            n += 1
        else:
            if not pd.isna(vals[i - 1]):
                lengths.append(n)
            n = 1
    if not pd.isna(vals[-1]):
        lengths.append(n)
    return lengths


def test_stale_forward_hold_15s_30s_3_and_6_bars():
    """15s CLV holds 3×5s bars; 30s CLV holds 6×5s bars."""
    idx = pd.date_range("2026-06-02 14:00:00", periods=120, freq="1s", tz="UTC")
    price = np.full(120, 100.0)
    # Distinct close at the end of each 15s bucket so consecutive holds differ.
    for t in range(14, 120, 15):
        price[t] = 100.0 + t
    tbbo = pd.DataFrame({"ts_event": idx, "price": price, "size": 1.0})
    grid = pd.date_range("2026-06-02 14:00:00", periods=24, freq="5s", tz="UTC")
    feats = compute_subminute_clv(tbbo, timestep="5s", grid=grid)
    holds_15 = _run_lengths(feats["cs_15s_clv"])
    holds_30 = _run_lengths(feats["cs_30s_clv"])
    assert holds_15, "no finite 15s holds"
    assert holds_30, "no finite 30s holds"
    # Interior holds should be exactly 3 and 6; the last incomplete bucket may be shorter.
    assert all(h == 3 for h in holds_15[:-1]), holds_15
    assert all(h == 6 for h in holds_30[:-1]), holds_30


def test_stale_forward_hold_5m_10m_on_5s_grid():
    """5m CLV holds 60×5s bars; 10m CLV holds 120×5s bars."""
    candles = _1m_candles(40, start="2026-06-02 14:00")
    # Make each 1m close unique so adjacent 5m/10m buckets differ.
    candles = candles.copy()
    candles["close"] = 100.0 + np.arange(len(candles))
    candles["high"] = candles["close"] + 0.2
    candles["low"] = candles[["open", "close"]].min(axis=1) - 0.1
    grid = pd.date_range("2026-06-02 14:00:00", periods=40 * 12, freq="5s", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": pd.date_range("2026-06-02 14:00:00", periods=40 * 60, freq="1s", tz="UTC"),
        "price": 100.0,
        "size": 1.0,
    })
    out = build_cs_patch_frame(candles, tbbo, grid, timestep="5s")
    assert list(out.columns) == list(CS_PATCH_COLS)
    holds_5 = _run_lengths(out["cs_5m_clv"])
    holds_10 = _run_lengths(out["cs_10m_clv"])
    assert holds_5, "no finite 5m holds"
    assert holds_10, "no finite 10m holds"
    assert all(h == 60 for h in holds_5[:-1]), holds_5
    assert all(h == 120 for h in holds_10[:-1]), holds_10


def test_5s_ohlc_from_ticks_and_catalog_close():
    ts = pd.DatetimeIndex([
        pd.Timestamp("2026-08-25 13:30:00", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:01", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:04", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:10", tz="UTC"),
    ])
    tbbo = pd.DataFrame({
        "ts_event": ts,
        "price": [100.0, 101.0, 99.5, 102.0],
        "size": [2.0, 3.0, 1.0, 4.0],
    })
    grid = pd.date_range("2026-08-25 13:30:00", periods=3, freq="5s", tz="UTC")
    close = pd.Series([99.5, 99.5, 102.0], index=grid)  # bar0 last, bar1 ffill, bar2 last
    out = compute_5s_ohlc(tbbo, grid=grid, timestep="5s", close=close)
    assert list(out.columns) == list(CS_5S_OHLC_COLS)
    np.testing.assert_allclose(out["cs_open"].iloc[0], 100.0)
    np.testing.assert_allclose(out["cs_high"].iloc[0], 101.0)
    np.testing.assert_allclose(out["cs_low"].iloc[0], 99.5)
    np.testing.assert_allclose(out["cs_close"].to_numpy(), close.to_numpy())
    np.testing.assert_allclose(out["cs_volume"].iloc[0], 6.0)
    np.testing.assert_allclose(out["cs_range"].iloc[0], 1.5)
    np.testing.assert_allclose(out["cs_direction"].iloc[0], -1.0)
    np.testing.assert_allclose(out["cs_return"].iloc[0], (99.5 - 100.0) / 100.0)
    # Empty middle bar: flat at last close, volume 0.
    np.testing.assert_allclose(out["cs_open"].iloc[1], 99.5)
    np.testing.assert_allclose(out["cs_high"].iloc[1], 99.5)
    np.testing.assert_allclose(out["cs_low"].iloc[1], 99.5)
    np.testing.assert_allclose(out["cs_volume"].iloc[1], 0.0)
    np.testing.assert_allclose(out["cs_range"].iloc[1], 0.0)
    np.testing.assert_allclose(out["cs_direction"].iloc[1], 0.0)
    np.testing.assert_allclose(out["cs_return"].iloc[1], 0.0)


def test_5s_ohlc_high_low_include_catalog_close():
    ts = pd.DatetimeIndex([
        pd.Timestamp("2026-08-25 13:30:00", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:01", tz="UTC"),
    ])
    tbbo = pd.DataFrame({"ts_event": ts, "price": [100.0, 100.5], "size": [1.0, 1.0]})
    grid = pd.date_range("2026-08-25 13:30:00", periods=1, freq="5s", tz="UTC")
    close = pd.Series([101.0], index=grid)  # outside tick last
    out = compute_5s_ohlc(tbbo, grid=grid, timestep="5s", close=close)
    np.testing.assert_allclose(out["cs_high"].iloc[0], 101.0)
    np.testing.assert_allclose(out["cs_low"].iloc[0], 100.0)
