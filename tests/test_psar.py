"""Closed-candle Parabolic SAR: 4 parquet cols, 5s pierce, 60s flip hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.psar import (
    DEFAULT_ATR_PERIOD,
    FLIP_BARS,
    PS_COLS,
    PS_RETIRED_COLS,
    closed_ohlc,
    compute,
    normalize_timeframe,
    sar_on_closed,
    tf_label,
)


def _1m(n=20, start="2026-06-02 14:00", close_step=0.5):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.arange(n, dtype="float64") * close_step
    open_ = close - 0.25
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def _tbbo_from_5s(grid: pd.DatetimeIndex, price) -> pd.DataFrame:
    """One print at each 5s left edge (fills that bar's high=low=price)."""
    px = np.asarray(price, dtype="float64")
    if px.ndim == 0 or px.size == 1:
        px = np.full(len(grid), float(px.reshape(-1)[0]))
    return pd.DataFrame({"ts_event": grid, "price": px})


def _pine_sar(high, low, close, start=0.02, increment=0.02, maximum=0.2):
    """Line-for-line Pine ``if bar_index > 0`` SAR / nextBarSAR."""
    n = len(close)
    sar = np.full(n, np.nan)
    nxt = np.full(n, np.nan)
    up_arr = np.full(n, np.nan)
    af_arr = np.full(n, np.nan)
    uptrend = False
    ep = np.nan
    af = start
    next_bar = np.nan
    for bar_index in range(1, n):
        first_trend_bar = False
        cur = next_bar
        if bar_index == 1:
            if close[1] > close[0]:
                uptrend = True
                ep = high[1]
                prev_sar = low[0]
                prev_ep = high[1]
            else:
                uptrend = False
                ep = low[1]
                prev_sar = high[0]
                prev_ep = low[1]
            first_trend_bar = True
            cur = prev_sar + start * (prev_ep - prev_sar)
        if (uptrend and cur > low[bar_index]) or ((not uptrend) and cur < high[bar_index]):
            first_trend_bar = True
            uptrend = not uptrend
            cur = min(ep, low[bar_index]) if uptrend else max(ep, high[bar_index])
            ep = high[bar_index] if uptrend else low[bar_index]
            af = start
        if not first_trend_bar:
            if uptrend and high[bar_index] > ep:
                ep = high[bar_index]
                af = min(af + increment, maximum)
            if (not uptrend) and low[bar_index] < ep:
                ep = low[bar_index]
                af = min(af + increment, maximum)
        if uptrend:
            cur = min(cur, low[bar_index - 1])
            if bar_index > 1:
                cur = min(cur, low[bar_index - 2])
        else:
            cur = max(cur, high[bar_index - 1])
            if bar_index > 1:
                cur = max(cur, high[bar_index - 2])
        next_bar = cur + af * (ep - cur)
        sar[bar_index] = cur
        nxt[bar_index] = next_bar
        up_arr[bar_index] = 1.0 if uptrend else -1.0
        af_arr[bar_index] = af
    return sar, nxt, up_arr, af_arr


def test_public_schema_is_four_columns():
    assert tf_label("1min") == "1m"
    assert normalize_timeframe("1m") == "1min"
    assert PS_COLS == (
        "ps_1m_next_dist_atr",
        "ps_1m_trend",
        "ps_lh_dist_sar",
        "ps_flip",
    )
    assert "ps_1m_sar" in PS_RETIRED_COLS
    grid = pd.date_range("2026-06-02 14:00:00", periods=12, freq="5s", tz="UTC")
    out = compute(pd.DataFrame(), grid=grid)
    assert list(out.columns) == list(PS_COLS)
    assert len(out) == 12
    assert out[["ps_1m_next_dist_atr", "ps_1m_trend", "ps_lh_dist_sar"]].isna().all().all()
    assert (out["ps_flip"] == 0.0).all()


def test_init_bar_matches_hand_calc():
    idx = pd.date_range("2026-06-02 14:00", periods=3, freq="1min", tz="UTC")
    ohlc = pd.DataFrame(
        {
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=idx + pd.Timedelta(minutes=1),
    )
    out = sar_on_closed(ohlc)
    assert np.isnan(out["sar"].iloc[0])
    np.testing.assert_allclose(out["sar"].iloc[1], 99.0)
    np.testing.assert_allclose(out["next_sar"].iloc[1], 99.06)
    assert out["trend"].iloc[1] == 1.0
    np.testing.assert_allclose(out["af"].iloc[1], 0.02)
    np.testing.assert_allclose(out["sar"].iloc[2], 99.0)
    np.testing.assert_allclose(out["next_sar"].iloc[2], 99.0 + 0.04 * (103.0 - 99.0))


def test_matches_pine_replica_up_and_flip():
    rng = np.random.default_rng(0)
    n = 80
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.8, n))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    low[40] = close[40] - 25.0
    high[40] = close[40] - 5.0
    close[40] = close[40] - 15.0
    idx = pd.date_range("2026-06-02 14:00", periods=n, freq="1min", tz="UTC")
    ohlc = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)
    got = sar_on_closed(ohlc)
    exp_sar, exp_nxt, exp_tr, exp_af = _pine_sar(high, low, close)
    np.testing.assert_allclose(got["sar"].to_numpy(), exp_sar, equal_nan=True)
    np.testing.assert_allclose(got["next_sar"].to_numpy(), exp_nxt, equal_nan=True)
    np.testing.assert_allclose(got["trend"].to_numpy(), exp_tr, equal_nan=True)
    np.testing.assert_allclose(got["af"].to_numpy(), exp_af, equal_nan=True)


def test_1m_trend_and_next_dist_visible_only_at_close():
    candles = _1m(n=12, close_step=1.0)
    grid = pd.date_range("2026-06-02 14:00:00", periods=48, freq="5s", tz="UTC")
    out = compute(candles, grid=grid, tbbo=_tbbo_from_5s(grid, 105.0))
    before = out.loc[
        out.index < pd.Timestamp("2026-06-02 14:02:00", tz="UTC"), "ps_1m_trend"
    ]
    assert before.isna().all()
    at_close = out.loc[pd.Timestamp("2026-06-02 14:02:00", tz="UTC"), "ps_1m_trend"]
    assert at_close == 1.0
    held = out.loc[pd.Timestamp("2026-06-02 14:02:25", tz="UTC"), "ps_1m_trend"]
    assert held == at_close
    # next_dist_atr needs ATR(8); by 14:08 it is finite
    t8 = pd.Timestamp("2026-06-02 14:08:00", tz="UTC")
    grid8 = pd.date_range("2026-06-02 14:00:00", periods=12 * 12, freq="5s", tz="UTC")
    out8 = compute(candles, grid=grid8, tbbo=_tbbo_from_5s(grid8, 110.0))
    assert np.isfinite(out8.loc[t8, "ps_1m_next_dist_atr"])
    assert DEFAULT_ATR_PERIOD == 8


def test_forming_minute_not_visible_on_grid():
    candles = _1m(n=3, close_step=1.0)
    grid = pd.date_range("2026-06-02 14:00:00", periods=30, freq="5s", tz="UTC")
    out = compute(candles, grid=grid, tbbo=_tbbo_from_5s(grid, 104.0))
    assert np.isfinite(out.loc[pd.Timestamp("2026-06-02 14:02:00", tz="UTC"), "ps_1m_trend"])
    last = out.loc[grid.max(), "ps_1m_trend"]
    at_2 = out.loc[pd.Timestamp("2026-06-02 14:02:00", tz="UTC"), "ps_1m_trend"]
    assert last == at_2


def test_lh_dist_uptrend_is_5s_low_minus_sar():
    candles = _1m(n=8, close_step=0.5)
    closed = closed_ohlc(candles, "1min")
    state = sar_on_closed(closed)
    t = pd.Timestamp("2026-06-02 14:03:00", tz="UTC")
    sar = float(state.loc[t, "next_sar"])
    trend = float(state.loc[t, "trend"])
    assert trend == 1.0
    grid = pd.date_range("2026-06-02 14:00:00", periods=12 * 8, freq="5s", tz="UTC")
    # 5s bar at 14:03:10: one print well above SAR, one wick below
    ts_hi = pd.Timestamp("2026-06-02 14:03:10", tz="UTC")
    ts_lo = pd.Timestamp("2026-06-02 14:03:12", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": [ts_hi, ts_lo],
        "price": [sar + 5.0, sar - 2.0],
    })
    out = compute(candles, grid=grid, tbbo=tbbo)
    dist = float(out.loc[ts_hi, "ps_lh_dist_sar"])
    np.testing.assert_allclose(dist, (sar - 2.0) - sar)
    assert dist < 0.0
    assert out.loc[ts_hi, "ps_flip"] == 1.0


def test_lh_dist_downtrend_is_sar_minus_5s_high():
    idx = pd.date_range("2026-06-02 14:00", periods=8, freq="1min", tz="UTC")
    close = 100.0 - np.arange(8, dtype="float64")
    candles = pd.DataFrame({
        "open": close + 0.25,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
    }, index=idx)
    closed = closed_ohlc(candles, "1min")
    state = sar_on_closed(closed)
    t = pd.Timestamp("2026-06-02 14:03:00", tz="UTC")
    sar = float(state.loc[t, "next_sar"])
    assert state.loc[t, "trend"] == -1.0
    grid = pd.date_range("2026-06-02 14:00:00", periods=12 * 8, freq="5s", tz="UTC")
    ts = pd.Timestamp("2026-06-02 14:03:10", tz="UTC")
    tbbo = pd.DataFrame({"ts_event": [ts], "price": [sar + 3.0]})
    out = compute(candles, grid=grid, tbbo=tbbo)
    dist = float(out.loc[ts, "ps_lh_dist_sar"])
    np.testing.assert_allclose(dist, sar - (sar + 3.0))
    assert dist < 0.0


def test_flip_holds_twelve_5s_bars():
    candles = _1m(n=8, close_step=0.5)
    closed = closed_ohlc(candles, "1min")
    state = sar_on_closed(closed)
    t0 = pd.Timestamp("2026-06-02 14:03:00", tz="UTC")
    sar = float(state.loc[t0, "next_sar"])
    grid = pd.date_range("2026-06-02 14:00:00", periods=12 * 8, freq="5s", tz="UTC")
    pierce = pd.Timestamp("2026-06-02 14:03:10", tz="UTC")
    # quiet tape at sar+10, one pierce print
    px = np.full(len(grid), sar + 10.0)
    px[grid.get_loc(pierce)] = sar - 1.0
    out = compute(candles, grid=grid, tbbo=_tbbo_from_5s(grid, px))
    assert FLIP_BARS == 12
    assert out.loc[pierce, "ps_flip"] == 1.0
    last_hold = pierce + pd.Timedelta(seconds=5 * 11)
    after = pierce + pd.Timedelta(seconds=5 * 12)
    assert out.loc[last_hold, "ps_flip"] == 1.0
    assert out.loc[after, "ps_flip"] == 0.0
    before = pierce - pd.Timedelta(seconds=5)
    assert out.loc[before, "ps_flip"] == 0.0


def test_closed_ohlc_1m_shifts_left_label():
    candles = _1m(n=3)
    closed = closed_ohlc(candles, "1min")
    assert closed.index[0] == pd.Timestamp("2026-06-02 14:01:00", tz="UTC")
    np.testing.assert_allclose(closed["close"].iloc[0], candles["close"].iloc[0])


def test_next_dist_does_not_use_future_1m_bars():
    full = _1m(n=30, close_step=0.4)
    prefix = full.iloc[:-5]
    grid = pd.date_range("2026-06-02 14:00:00", periods=20 * 12, freq="5s", tz="UTC")
    a = compute(full, grid=grid, tbbo=_tbbo_from_5s(grid, 110.0))
    b = compute(prefix, grid=grid, tbbo=_tbbo_from_5s(grid, 110.0))
    t = pd.Timestamp("2026-06-02 14:10:00", tz="UTC")
    np.testing.assert_allclose(
        float(a.loc[t, "ps_1m_trend"]), float(b.loc[t, "ps_1m_trend"]),
    )
    np.testing.assert_allclose(
        float(a.loc[t, "ps_1m_next_dist_atr"]),
        float(b.loc[t, "ps_1m_next_dist_atr"]),
    )


def test_downtrend_init_when_close_does_not_rise():
    idx = pd.date_range("2026-06-02 14:00", periods=3, freq="1min", tz="UTC")
    ohlc = pd.DataFrame(
        {
            "high": [101.0, 100.5, 99.5],
            "low": [99.0, 98.0, 97.0],
            "close": [100.5, 99.0, 98.0],
        },
        index=idx + pd.Timedelta(minutes=1),
    )
    out = sar_on_closed(ohlc)
    assert out["trend"].iloc[1] == -1.0
    np.testing.assert_allclose(out["sar"].iloc[1], 101.0)
    np.testing.assert_allclose(out["next_sar"].iloc[1], 101.0 + 0.02 * (98.0 - 101.0))
