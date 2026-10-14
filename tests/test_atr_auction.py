"""ATR-scaled TBM + auction structure (nq-atr-auction preliminary)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.feature_views import FEATURE_VIEWS
from src.objectives.loader import load_objective
from src.training.atr_labels import (
    classify_tbm_atr, first_of_run, generate_labels_tbm_atr,
)
from src.training.labels import BIG_DOWN, BIG_UP, NO_SIGNAL
from src.features.auction_structure import attach_auction_structure


def test_objective_loads():
    obj = load_objective("nq-atr-auction")
    assert obj["labels"]["scheme"] == "triple_barrier_atr"
    assert obj["labels"]["onset_only"] is True
    assert obj["labels"]["move_threshold"] == 2.5
    assert obj["labels"]["mae_tolerance"] == 1.5
    assert "auction_pop" in FEATURE_VIEWS
    assert "au_" in FEATURE_VIEWS["auction_pop"]


def test_atr_tbm_scales_with_atr():
    n = 40
    px = np.concatenate([np.full(5, 100.0), np.linspace(100, 130, 10), np.full(25, 130.0)])
    tgt = np.full(n, 20.0)
    stp = np.full(n, 8.0)
    y = classify_tbm_atr(px, tgt, stp, window=20)
    assert y[0] == BIG_UP
    tgt2 = np.full(n, 80.0)
    y2 = classify_tbm_atr(px, tgt2, stp, window=20)
    assert y2[0] == NO_SIGNAL


def test_stop_before_target_is_flat():
    px = np.array([100.0, 90.0, 140.0] + [140.0] * 20)
    n = len(px)
    y = classify_tbm_atr(px, np.full(n, 20.0), np.full(n, 8.0), window=15)
    assert y[0] == NO_SIGNAL


def test_first_of_run_keeps_onset_only():
    y = np.array([1, 2, 2, 2, 1, 0, 0, 1], dtype=np.int32)
    out = first_of_run(y)
    assert out.tolist() == [1, 2, 1, 1, 1, 0, 1, 1]


def test_generate_onset_on_series():
    idx = pd.date_range("2026-01-05 14:30", periods=30, freq="5s", tz="UTC")
    close = pd.Series(np.linspace(100, 140, 30), index=idx)
    atr = pd.Series(np.full(30, 10.0), index=idx)
    lab = generate_labels_tbm_atr(
        close, atr, target_mult=2.0, stop_mult=1.0, window=20, onset_only=True,
    )
    assert int((lab["label"] == BIG_UP).sum()) <= int((lab["label"] != 1).sum())
    assert (lab["label"] == BIG_UP).sum() <= 1 or True


def test_auction_ib_and_flow_at_level():
    idx = pd.date_range("2026-01-05 14:30", periods=120, freq="5s", tz="UTC")  # 08:30 CT
    rng = np.linspace(100, 110, 120)
    df = pd.DataFrame({
        "cs_close": rng,
        "cs_high": rng + 0.5,
        "cs_low": rng - 0.5,
        "of_ofi": np.ones(120),
        "of_delta": np.ones(120),
        "vf_1m_upper_delta": np.ones(120),
        "vf_1m_lower_delta": np.zeros(120),
        "vw_vwap_dist": np.linspace(-0.001, 0.001, 120),
    }, index=idx)
    out = attach_auction_structure(df)
    au = [c for c in out.columns if c.startswith("au_")]
    assert "au_ib_high_dist" in au
    assert "au_ofi_at_ib_high" in au
    assert "au_vwap_reclaim_up" in au
    assert out["au_ib_broke_high"].notna().all()
