"""Slim 1h/1d regime pack (ADR-095)."""

from __future__ import annotations

import pytest
import pandas as pd

from src.features.htf_context import (
    CONTEXT_1D, CONTEXT_1H, CONTEXT_COLS, HTF_PACK_ID,
    asof_attach_context, wants_htf_on_store,
)
from src.objectives.loader import load_objective


def test_pack_is_regime_only():
    assert HTF_PACK_ID == "regime_v1"
    assert "cs_open" not in CONTEXT_COLS
    assert "cs_atr_14" not in CONTEXT_COLS
    assert "sr_fib_382_dist" not in CONTEXT_COLS
    assert "sr_pivot_dist" not in CONTEXT_COLS
    assert "tr_ema_stack" in CONTEXT_1H and "tr_ema_stack" in CONTEXT_1D
    assert "sr_prior_high_dist" in CONTEXT_1D
    assert len(CONTEXT_1H) + len(CONTEXT_1D) <= 16


def test_wants_htf_on_predictive_stores_only():
    obj = load_objective("nq-50pt-econ-lg-st")
    assert wants_htf_on_store(obj, "15s")
    assert wants_htf_on_store(obj, "1m")
    assert wants_htf_on_store(obj, "5m")
    assert not wants_htf_on_store(obj, "1h")
    assert not wants_htf_on_store(obj, "5s")
    assert not wants_htf_on_store(load_objective("nq-50pt-econ"), "15s")


def test_asof_attach_prefixes_and_is_causal():
    grid = pd.date_range("2026-01-02 15:00:00", periods=4, freq="15s", tz="UTC")
    aligned = pd.DataFrame({"ts": grid, "p": [1, 2, 3, 4]})
    h1 = pd.DataFrame({
        "ts": [pd.Timestamp("2026-01-02 14:00:00", tz="UTC"),
               pd.Timestamp("2026-01-02 15:00:00", tz="UTC")],
        "tr_ema_stack": [1.0, -1.0],
        "rv_atr_ratio": [0.8, 1.2],
        "tr_slope_10": [0.1, 0.2],
        "tr_gap": [0.0, 0.01],
        "zz_signal": [1.0, -1.0],
        "rv_vol_ratio": [1.0, 1.1],
        "sr_pos_in_range_20": [0.4, 0.6],
    })
    out = asof_attach_context(aligned, {"1h": h1})
    assert "h1_tr_ema_stack" in out.columns
    assert "h1_cs_atr_14" not in out.columns
    row = out.loc[out["ts"] == pd.Timestamp("2026-01-02 15:00:15", tz="UTC")].iloc[0]
    assert row["h1_tr_ema_stack"] == pytest.approx(-1.0)
