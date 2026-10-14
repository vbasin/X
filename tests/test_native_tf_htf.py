"""Native 1h / 1d HTF context hive (ADR-091)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.features.native_tf_htf import (
    CS_COLS,
    FEATURE_HTF_COLS,
    PS_COLS,
    RV_COLS,
    SR_COLS,
    TR_COLS,
    ZZ_COLS,
    bar_end_from_left,
    compute_htf_features,
    empty_frame,
    select_lead_month,
)
from src.search.custom_test import parse_feature_catalog


def _hourly(n: int = 80, start: str = "2026-08-03 13:00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    px = 20000.0 + np.cumsum(rng.normal(2.0, 8.0, size=n))
    high = px + rng.uniform(2.0, 15.0, size=n)
    low = px - rng.uniform(2.0, 15.0, size=n)
    open_ = np.concatenate([[px[0]], px[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(high, np.maximum(open_, px)),
        "low": np.minimum(low, np.minimum(open_, px)),
        "close": px,
        "volume": rng.integers(100, 5000, size=n).astype("float64"),
    }, index=idx)


def test_empty_has_schema():
    out = empty_frame()
    assert list(out.columns) == ["bar_end_ts", *FEATURE_HTF_COLS]
    assert out.empty
    assert compute_htf_features(pd.DataFrame(), timeframe="1h").empty


def test_bar_end_1h_is_left_plus_one_hour():
    idx = pd.DatetimeIndex(["2026-08-25 13:00:00"], tz="UTC")
    got = bar_end_from_left(idx, "1h")
    assert got[0] == pd.Timestamp("2026-08-25 14:00:00", tz="UTC")


def test_bar_end_1d_is_next_chicago_1700_across_dst():
    # Session open 2026-03-07 17:00 CST (UTC-6) → close 2026-03-08 17:00 CDT (UTC-5).
    left = pd.DatetimeIndex(["2026-03-07 23:00:00"], tz="UTC")
    got = bar_end_from_left(left, "1d")
    assert got[0] == pd.Timestamp("2026-03-08 22:00:00", tz="UTC")


def test_compute_emits_all_group_columns():
    out = compute_htf_features(_hourly(90), timeframe="1h")
    assert not out.empty
    assert len(out) == 90
    for col in FEATURE_HTF_COLS:
        assert col in out.columns
    assert out["bar_end_ts"].iloc[0] == pd.Timestamp("2026-08-03 14:00:00", tz="UTC")
    assert out["cs_close"].notna().all()
    assert out["cs_atr_14"].notna().any()
    assert set(pd.Series(out["tr_ema_stack"]).dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert out["rv_realized_vol_5"].notna().any()
    assert out["sr_pivot_dist"].notna().any()
    assert len(FEATURE_HTF_COLS) == (
        len(CS_COLS) + len(TR_COLS) + len(PS_COLS)
        + len(ZZ_COLS) + len(RV_COLS) + len(SR_COLS)
    )
    assert len(FEATURE_HTF_COLS) == 45


def test_1h_prior_levels_are_previous_session_not_previous_hour():
    # Two Globex sessions of three hours each (13:00–16:00 UTC = 08:00–11:00 CT).
    a = pd.date_range("2026-08-25 13:00:00", periods=3, freq="1h", tz="UTC")
    b = pd.date_range("2026-08-26 13:00:00", periods=3, freq="1h", tz="UTC")
    idx = a.append(b)
    close = np.array([100.0, 101.0, 110.0, 120.0, 121.0, 122.0])
    high = np.array([101.0, 102.0, 115.0, 125.0, 126.0, 127.0])
    low = np.array([99.0, 100.0, 109.0, 119.0, 120.0, 121.0])
    candles = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(6),
    }, index=idx)
    out = compute_htf_features(candles, timeframe="1h")
    # Last bar of session B: prior session high 115, close 110.
    last = out.iloc[-1]
    px = 122.0
    np.testing.assert_allclose(last["sr_prior_high_dist"], (px - 115.0) / px, rtol=1e-5)
    np.testing.assert_allclose(last["sr_prior_low_dist"], (px - 99.0) / px, rtol=1e-5)
    np.testing.assert_allclose(last["sr_prior_close_dist"], (px - 110.0) / px, rtol=1e-5)
    # Pivot PP = (115 + 99 + 110) / 3
    pp = (115.0 + 99.0 + 110.0) / 3.0
    np.testing.assert_allclose(last["sr_pivot_dist"], (px - pp) / px, rtol=1e-5)


def test_1d_gap_is_open_vs_prev_close():
    idx = pd.date_range("2026-08-03 22:00:00", periods=5, freq="1D", tz="UTC")
    close = np.array([100.0, 110.0, 105.0, 120.0, 118.0])
    open_ = np.array([99.0, 112.0, 104.0, 121.0, 119.0])
    candles = pd.DataFrame({
        "open": open_,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": np.ones(5),
    }, index=idx)
    out = compute_htf_features(candles, timeframe="1d")
    # Row 1: gap (112 - 100) / 100
    np.testing.assert_allclose(out["tr_gap"].iloc[1], 0.12, rtol=1e-5)


def test_fib_levels_sit_inside_swing():
    n = 60
    idx = pd.date_range("2026-08-03 13:00:00", periods=n, freq="1h", tz="UTC")
    # Clean impulse 100 → 200 so ZZ has a defined range.
    close = np.linspace(100.0, 200.0, n)
    candles = pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.ones(n),
    }, index=idx)
    out = compute_htf_features(candles, timeframe="1h")
    row = out.iloc[-1]
    # Distances to 38.2 / 50 / 61.8: close near the high ⇒ 382 dist > 500 > 618
    # (all positive if close is above those retracement levels).
    if np.isfinite(row["sr_fib_382_dist"]) and np.isfinite(row["sr_fib_618_dist"]):
        assert row["sr_fib_382_dist"] > row["sr_fib_500_dist"]
        assert row["sr_fib_500_dist"] > row["sr_fib_618_dist"]


def test_select_lead_month_prefers_active_contract():
    from src.common.contract_calendar import active_contract

    bucket = pd.Timestamp("2026-08-25 13:00:00", tz="UTC")
    want = active_contract("NQ", date(2026, 8, 25))
    other = "NQZ6" if want != "NQZ6" else "NQU6"
    raw = pd.DataFrame({
        "bucket": [bucket, bucket],
        "symbol": [other, want],
        "open": [1.0, 2.0],
        "high": [1.0, 3.0],
        "low": [1.0, 2.0],
        "close": [1.0, 2.5],
        "volume": [9999.0, 10.0],
    })
    got = select_lead_month(raw)
    assert len(got) == 1
    assert got["symbol"].iloc[0] == want
    assert got["close"].iloc[0] == 2.5


def test_catalogs_parse_six_groups():
    for store in ("1h", "1d"):
        parsed = parse_feature_catalog(store=store)
        prefixes = [g["prefix"] for g in parsed["groups"]]
        assert prefixes == ["cs_", "tr_", "ps_", "zz_", "rv_", "sr_"]
        names = {f["name"] for f in parsed["features"]}
        assert names == set(FEATURE_HTF_COLS)
        assert "sr_fib_618_dist" in names
        assert "tr_ema_stack" in names
        assert "cs_atr_14" in names
        assert "vw_vwap_dist" not in names


def test_store_views_native_htf():
    from src.features.stores import views_for_store

    assert views_for_store("1h")["native_1h"] == [
        "cs_", "tr_", "ps_", "zz_", "rv_", "sr_",
    ]
    assert views_for_store("1d")["native_1d"] == [
        "cs_", "tr_", "ps_", "zz_", "rv_", "sr_",
    ]
