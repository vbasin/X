"""Native 5-minute feature hive (ADR-090)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.native_tf_5m import (
    CS_COLS,
    FEATURE_5M_COLS,
    PS_COLS,
    RV_COLS,
    VF_COLS,
    VP_COLS,
    VW_COLS,
    ZZ_COLS,
    candles_5m_from_tbbo,
    compute_5m_features,
)
from src.search.custom_test import parse_feature_catalog


def _tbbo(n_min: int = 180, start: str = "2026-08-25 13:30:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_min * 12, freq="5s", tz="UTC")
    rng = np.random.default_rng(11)
    px = 20000.0 + np.cumsum(rng.normal(0, 0.8, size=len(idx)))
    side = np.where(rng.random(len(idx)) > 0.5, "B", "A")
    return pd.DataFrame({
        "ts_event": idx,
        "price": px,
        "size": rng.integers(1, 8, size=len(idx)).astype("float64"),
        "side": side,
        "bid_px": px - 0.25,
        "ask_px": px + 0.25,
        "bid_sz": np.full(len(idx), 10.0),
        "ask_sz": np.full(len(idx), 10.0),
    })


def test_empty_tbbo_has_schema():
    out = compute_5m_features(pd.DataFrame())
    assert list(out.columns) == ["bar_end_ts", *FEATURE_5M_COLS]
    assert out.empty


def test_bar_end_ts_is_closed_5m_label():
    ts = pd.date_range("2026-08-25 13:30:00", periods=12, freq="25s", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": ts,
        "price": [100.0, 101, 99, 100.5, 100.2, 100.4, 100.1, 100.3, 100.6, 100.7, 99.8, 100.9],
        "size": np.ones(12),
        "side": ["B"] * 12,
        "bid_px": np.full(12, 99.75),
        "ask_px": np.full(12, 100.25),
        "bid_sz": np.full(12, 5.0),
        "ask_sz": np.full(12, 5.0),
    })
    out = compute_5m_features(tbbo)
    assert len(out) == 1
    assert out["bar_end_ts"].iloc[0] == pd.Timestamp("2026-08-25 13:35:00", tz="UTC")
    assert out["cs_open"].iloc[0] == 100.0
    assert out["cs_high"].iloc[0] == 101.0
    assert out["cs_low"].iloc[0] == 99.0
    assert out["cs_close"].iloc[0] == 100.9


def test_candles_drop_empty_buckets():
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:31:00",
        "2026-08-25 13:40:00",
    ], utc=True)
    tbbo = pd.DataFrame({
        "ts_event": ts,
        "price": [1.0, 2.0, 3.0],
        "size": [1.0, 1.0, 1.0],
        "side": ["B", "B", "A"],
        "bid_px": [1, 2, 3],
        "ask_px": [1.25, 2.25, 3.25],
        "bid_sz": [1, 1, 1],
        "ask_sz": [1, 1, 1],
    })
    c = candles_5m_from_tbbo(tbbo)
    assert len(c) == 2
    assert list(c.index) == [
        pd.Timestamp("2026-08-25 13:30:00", tz="UTC"),
        pd.Timestamp("2026-08-25 13:40:00", tz="UTC"),
    ]


def test_compute_emits_all_group_columns():
    out = compute_5m_features(_tbbo(240))
    assert not out.empty
    for col in FEATURE_5M_COLS:
        assert col in out.columns
    assert out["cs_close"].notna().all()
    assert out["vw_vwap_dist"].notna().any()
    assert out["vw_bb_upper_dist"].notna().any()
    assert out["vw_bb_lower_dist"].notna().any()
    assert out["cs_atr_5"].notna().any()
    assert out["rv_realized_vol_5"].notna().any()
    assert set(out["ps_5m_trend"].dropna().unique()).issubset({-1.0, 1.0})
    assert set(out["zz_signal"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert out["vf_5m_body_vol_frac"].notna().any()
    assert "vf_5m_poc_loc" in out.columns
    assert "iv_atm_mean" not in out.columns
    assert len(FEATURE_5M_COLS) == (
        len(CS_COLS) + len(VP_COLS) + len(VW_COLS) + len(PS_COLS)
        + len(ZZ_COLS) + len(VF_COLS) + len(RV_COLS)
    )
    assert len(FEATURE_5M_COLS) == 53


def test_vwap_bands_are_two_stdev_from_vwap():
    """Session VWAP ± 2σ: close at VWAP ⇒ upper_dist = −lower_dist = −2σ/VWAP."""
    idx = pd.date_range("2026-08-25 13:30:00", periods=60, freq="5s", tz="UTC")
    px = np.full(60, 100.0)
    px[30:] = 104.0
    tbbo = pd.DataFrame({
        "ts_event": idx,
        "price": px,
        "size": np.ones(60),
        "side": ["B"] * 60,
        "bid_px": px - 0.25,
        "ask_px": px + 0.25,
        "bid_sz": np.ones(60),
        "ask_sz": np.ones(60),
    })
    out = compute_5m_features(tbbo)
    assert not out.empty
    row = out.iloc[-1]
    dist = float(row["vw_vwap_dist"])
    up = float(row["vw_bb_upper_dist"])
    lo = float(row["vw_bb_lower_dist"])
    # (close − (vwap+2σ))/vwap + (close − (vwap−2σ))/vwap = 2 * vw_vwap_dist
    np.testing.assert_allclose(up + lo, 2.0 * dist, rtol=1e-5, atol=1e-6)
    # Upper band is above VWAP when σ > 0, so upper dist < vwap dist.
    assert up < dist


def test_catalog_5m_parses_seven_groups():
    parsed = parse_feature_catalog(store="5m")
    prefixes = [g["prefix"] for g in parsed["groups"]]
    assert prefixes == ["cs_", "vp_", "vw_", "ps_", "zz_", "vf_", "rv_", "svp_"]
    names = {f["name"] for f in parsed["features"]}
    assert set(FEATURE_5M_COLS) <= names
    assert "svp_poc_dist" in names
    assert "vw_bb_upper_dist" in names and "vw_bb_lower_dist" in names
    assert "cs_atr_5" in names
    assert "zz_signal" in names
    assert "so_fast_d_9_3" not in names


def test_store_view_native_5m():
    from src.features.stores import views_for_store
    v = views_for_store("5m")
    assert v["native_5m"] == ["cs_", "vp_", "vw_", "ps_", "zz_", "vf_", "rv_", "svp_"]
