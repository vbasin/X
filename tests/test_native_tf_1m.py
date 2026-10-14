"""Native 1-minute feature hive (ADR-085)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.native_tf_1m import (
    CS_COLS,
    FEATURE_1M_COLS,
    MS_COLS,
    PS_COLS,
    RV_COLS,
    SO_COLS,
    VP_COLS,
    VW_COLS,
    candles_1m_from_tbbo,
    compute_1m_features,
)
from src.search.custom_test import parse_feature_catalog


def _tbbo(n_min: int = 90, start: str = "2026-08-25 13:30:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_min * 12, freq="5s", tz="UTC")
    rng = np.random.default_rng(7)
    px = 20000.0 + np.cumsum(rng.normal(0, 0.5, size=len(idx)))
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
    out = compute_1m_features(pd.DataFrame())
    assert list(out.columns) == ["bar_end_ts", *FEATURE_1M_COLS]
    assert out.empty


def test_bar_end_ts_is_closed_1m_label():
    ts = pd.date_range("2026-08-25 13:30:00", periods=8, freq="5s", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": ts,
        "price": [100.0, 101, 99, 100.5, 100.2, 100.4, 100.1, 100.3],
        "size": np.ones(8),
        "side": ["B"] * 8,
        "bid_px": np.full(8, 99.75),
        "ask_px": np.full(8, 100.25),
        "bid_sz": np.full(8, 5.0),
        "ask_sz": np.full(8, 5.0),
    })
    out = compute_1m_features(tbbo)
    assert len(out) == 1
    assert out["bar_end_ts"].iloc[0] == pd.Timestamp("2026-08-25 13:31:00", tz="UTC")
    assert out["cs_open"].iloc[0] == 100.0
    assert out["cs_high"].iloc[0] == 101.0
    assert out["cs_low"].iloc[0] == 99.0
    assert out["cs_close"].iloc[0] == 100.3


def test_candles_drop_empty_minutes():
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:30",
        "2026-08-25 13:32:00",
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
    c = candles_1m_from_tbbo(tbbo)
    assert len(c) == 2
    assert list(c.index) == [
        pd.Timestamp("2026-08-25 13:30:00", tz="UTC"),
        pd.Timestamp("2026-08-25 13:32:00", tz="UTC"),
    ]


def test_compute_emits_all_group_columns():
    out = compute_1m_features(_tbbo(90))
    assert not out.empty
    for col in FEATURE_1M_COLS:
        assert col in out.columns
    assert out["cs_close"].notna().all()
    assert out["vw_vwap_dist"].notna().any()
    assert out["rv_realized_vol_60"].notna().any()
    assert set(out["ps_1m_trend"].dropna().unique()).issubset({-1.0, 1.0})
    assert "iv_atm_mean" not in out.columns
    assert len(SO_COLS) + len(CS_COLS) + len(PS_COLS) + len(
        VW_COLS) + len(RV_COLS) + len(VP_COLS) + len(MS_COLS) == 71


def test_catalog_1m_parses_eight_groups():
    parsed = parse_feature_catalog(store="1m")
    prefixes = [g["prefix"] for g in parsed["groups"]]
    assert prefixes == ["so_", "cs_", "ps_", "vw_", "rv_", "vp_", "ms_"]
    names = {f["name"] for f in parsed["features"]}
    assert names == set(FEATURE_1M_COLS)
    assert "iv_atm_mean" not in names
    assert "of_ofi" not in names


def test_store_view_native_1m():
    from src.features.stores import views_for_store
    v = views_for_store("1m")
    assert v["native_1m"] == ["so_", "cs_", "ps_", "vw_", "rv_", "vp_", "ms_"]
