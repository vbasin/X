"""Closed-candle volume footprint: wick/body volume, 1m POC, 5-bar EMA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volume_footprint import (
    VF_COLS,
    compute,
    compute_footprints,
    describe_regions,
    feature_names,
    tf_label,
    viz_payload,
)


def _tbbo(ts, price, size=None, side=None):
    n = len(ts)
    return pd.DataFrame({
        "ts_event": ts,
        "price": price,
        "size": np.ones(n, dtype="float64") if size is None else size,
        "side": (["B"] * n) if side is None else side,
    })


def test_tf_label_and_column_count():
    assert tf_label("1min") == "1m"
    assert tf_label("5min") == "5m"
    names = feature_names(("1min", "5min"))
    assert names == VF_COLS
    assert len(VF_COLS) == 20
    assert VF_COLS[0] == "vf_1m_upper_vol_frac"
    assert "vf_1m_poc_loc" in VF_COLS
    assert "vf_1m_poc_ema_5" in VF_COLS
    assert "vf_5m_body_delta_scaled" in VF_COLS
    assert "vf_5m_poc_loc" not in VF_COLS
    assert all(c.startswith("vf_") for c in VF_COLS)
    assert len(feature_names(("1min",))) == 11
    assert len(feature_names(("5min",))) == 9


def test_describe_regions_wick_body_delta_and_scaled():
    # Chronological: open 100, high 101 (ask 40), low 99.5 (bid 10), close 100.5 (ask 30)
    # Body [100, 100.5] also gets the open print (ask 20).
    ticks = np.array([100.00, 101.00, 99.50, 100.50])
    size = np.array([20.0, 40.0, 10.0, 30.0])
    ask = np.array([20.0, 40.0, 0.0, 30.0])
    bid = np.array([0.0, 0.0, 10.0, 0.0])
    feats = describe_regions(ticks, size, ask, bid)
    np.testing.assert_allclose(feats["upper_vol_frac"], 0.40)
    np.testing.assert_allclose(feats["body_vol_frac"], 0.50)
    np.testing.assert_allclose(feats["lower_vol_frac"], 0.10)
    np.testing.assert_allclose(
        feats["upper_vol_frac"] + feats["body_vol_frac"] + feats["lower_vol_frac"],
        1.0,
    )
    np.testing.assert_allclose(feats["upper_delta"], 40.0)   # all ask
    np.testing.assert_allclose(feats["body_delta"], 50.0)
    np.testing.assert_allclose(feats["lower_delta"], -10.0)  # all bid
    np.testing.assert_allclose(feats["upper_delta_scaled"], 0.40 * (40.0 / 40.0))
    np.testing.assert_allclose(feats["body_delta_scaled"], 0.50 * (50.0 / 50.0))
    np.testing.assert_allclose(feats["lower_delta_scaled"], 0.10 * (-10.0 / 10.0))
    # POC is 101 (40 lots); range 99.5–101
    np.testing.assert_allclose(feats["poc"], 101.00)
    np.testing.assert_allclose(feats["poc_loc"], (101.00 - 99.50) / (101.00 - 99.50))


def test_describe_regions_body_includes_open_and_close():
    ticks = np.array([100.00, 100.25, 100.50])
    size = np.array([5.0, 1.0, 4.0])
    ask = size.copy()
    bid = np.zeros(3)
    feats = describe_regions(ticks, size, ask, bid)
    # No wicks: high=close=100.50, low=open=100.00 → 100% body
    np.testing.assert_allclose(feats["body_vol_frac"], 1.0)
    np.testing.assert_allclose(feats["upper_vol_frac"], 0.0)
    np.testing.assert_allclose(feats["lower_vol_frac"], 0.0)
    np.testing.assert_allclose(feats["upper_delta_scaled"], 0.0)
    np.testing.assert_allclose(feats["lower_delta_scaled"], 0.0)


def test_describe_regions_zero_upper_wick_scaled_is_zero():
    ticks = np.array([100.00, 99.50, 100.00])
    size = np.array([2.0, 3.0, 5.0])
    ask = np.array([2.0, 0.0, 5.0])
    bid = np.array([0.0, 3.0, 0.0])
    feats = describe_regions(ticks, size, ask, bid)
    assert feats["upper_vol_frac"] == 0.0
    assert feats["upper_delta"] == 0.0
    assert feats["upper_delta_scaled"] == 0.0
    np.testing.assert_allclose(feats["lower_vol_frac"], 0.3)
    np.testing.assert_allclose(feats["body_vol_frac"], 0.7)


def test_compute_empty_tbbo_has_schema():
    grid = pd.date_range("2026-06-02 14:00:00", periods=12, freq="5s", tz="UTC")
    out = compute(pd.DataFrame(), grid=grid)
    assert list(out.columns) == list(VF_COLS)
    assert len(out) == 12
    assert out.isna().all().all()


def test_1m_visible_only_at_close():
    idx = pd.date_range("2026-06-02 14:00:00", periods=60, freq="1s", tz="UTC")
    tbbo = _tbbo(idx, np.full(60, 101.00), size=np.full(60, 2.0), side=["B"] * 60)
    grid = pd.date_range("2026-06-02 14:00:00", periods=24, freq="5s", tz="UTC")
    out = compute(tbbo, grid=grid, timeframes=("1min",))
    before = out.loc[
        out.index < pd.Timestamp("2026-06-02 14:01:00", tz="UTC"),
        "vf_1m_body_vol_frac",
    ]
    assert before.isna().all()
    at_close = out.loc[pd.Timestamp("2026-06-02 14:01:00", tz="UTC"), "vf_1m_body_vol_frac"]
    assert float(at_close) == 1.0
    poc = out.loc[pd.Timestamp("2026-06-02 14:01:00", tz="UTC"), "vf_1m_poc_loc"]
    assert float(poc) == 0.5
    later = out.loc[pd.Timestamp("2026-06-02 14:01:10", tz="UTC"), "vf_1m_body_vol_frac"]
    assert float(later) == 1.0


def test_5m_no_lookahead_on_late_upper_wick():
    idx = pd.date_range("2026-06-02 14:00:00", periods=300, freq="1s", tz="UTC")
    price = np.full(300, 100.00)
    # Last 20s of [14:00, 14:05): spike to 110 then back so close stays 100.
    price[-20:-1] = 110.00
    size = np.full(300, 1.0)
    size[-20:-1] = 10.0
    tbbo = _tbbo(idx, price, size=size, side=["B"] * 300)
    grid = pd.date_range("2026-06-02 14:00:00", periods=12 * 6, freq="5s", tz="UTC")
    out = compute(tbbo, grid=grid, timeframes=("5min",))
    before = out.loc[
        out.index < pd.Timestamp("2026-06-02 14:05:00", tz="UTC"),
        "vf_5m_upper_vol_frac",
    ]
    assert before.isna().all()
    at_close = float(out.loc[pd.Timestamp("2026-06-02 14:05:00", tz="UTC"), "vf_5m_upper_vol_frac"])
    assert at_close > 0.3
    held = float(out.loc[pd.Timestamp("2026-06-02 14:05:25", tz="UTC"), "vf_5m_upper_vol_frac"])
    assert held == at_close
    assert "vf_5m_poc_loc" not in out.columns


def test_1m_and_5m_columns_together():
    idx = pd.date_range("2026-06-02 14:00:00", periods=300, freq="1s", tz="UTC")
    tbbo = _tbbo(idx, np.full(300, 100.25), size=np.full(300, 3.0), side=["A"] * 300)
    grid = pd.date_range("2026-06-02 14:00:00", periods=72, freq="5s", tz="UTC")
    out = compute(tbbo, grid=grid)
    assert list(out.columns) == list(VF_COLS)
    assert float(out.loc[pd.Timestamp("2026-06-02 14:01:00", tz="UTC"), "vf_1m_body_vol_frac"]) == 1.0
    assert np.isnan(out.loc[pd.Timestamp("2026-06-02 14:01:00", tz="UTC"), "vf_5m_body_vol_frac"])
    assert float(out.loc[pd.Timestamp("2026-06-02 14:05:00", tz="UTC"), "vf_5m_body_vol_frac"]) == 1.0


def test_1m_poc_ema_5_matches_ewm_span_5():
    # Five 1-min candles; POC at low then high so loc is 0,0,0,0,1.
    rows = []
    for i, loc_price in enumerate([100.00, 100.00, 100.00, 100.00, 101.00]):
        start = pd.Timestamp("2026-06-02 14:00:00", tz="UTC") + pd.Timedelta(minutes=i)
        ts = pd.date_range(start, periods=2, freq="20s")
        # Two prints at the same price → poc_loc = 0.5 if we don't create a range.
        # Give every bar a 100–101 range but dump volume at loc_price.
        prices = [100.00, loc_price, 101.00]
        times = [start, start + pd.Timedelta(seconds=20), start + pd.Timedelta(seconds=40)]
        sizes = [1.0, 20.0, 1.0]
        rows.append(_tbbo(times, prices, size=sizes, side=["B"] * 3))
    tbbo = pd.concat(rows, ignore_index=True)
    out = compute(tbbo, timeframes=("1min",))
    loc = out["vf_1m_poc_loc"].dropna()
    assert len(loc) == 5
    np.testing.assert_allclose(loc.iloc[:4], 0.0, atol=1e-6)
    np.testing.assert_allclose(loc.iloc[4], 1.0, atol=1e-6)
    ema = out["vf_1m_poc_ema_5"].dropna()
    assert len(ema) == 1  # min_periods=5
    expect = loc.ewm(span=5, adjust=False, min_periods=5).mean().iloc[-1]
    np.testing.assert_allclose(float(ema.iloc[-1]), float(expect), rtol=1e-5)


def test_viz_payload_closed_bar_and_body_bands():
    idx = pd.date_range("2026-06-02 14:00:00", periods=60, freq="1s", tz="UTC")
    payload = viz_payload(_tbbo(idx, np.full(60, 100.0)), timeframe="1min")
    assert payload["closed_bar"] is True
    assert payload["label"] == "1m"
    assert payload["n"] == 1
    c = payload["candles"][0]
    assert c["t_close"].endswith("Z")
    assert "body_low" in c and "ask" in c
    assert "concentrations" not in c


def test_compute_footprints_right_labels_partial_minute():
    """Prints in [14:00, 14:00:30) still stamp at the 14:01 close."""
    idx = pd.date_range("2026-06-02 14:00:00", periods=30, freq="1s", tz="UTC")
    fps = compute_footprints(_tbbo(idx, np.full(30, 100.0)), "1min")
    assert len(fps) == 1
    assert fps[0].close_ts == pd.Timestamp("2026-06-02 14:01:00", tz="UTC")
    np.testing.assert_allclose(fps[0].features["body_vol_frac"], 1.0)
