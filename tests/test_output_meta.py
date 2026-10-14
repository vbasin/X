"""ADR-033 output_meta: ts-align, vector features, logistic gate, objective load."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.objectives.loader import load_objective
from src.search.output_meta import (
    align_on_ts, build_X_meta, compose, proposed_side, topology_output_meta,
)


def _oof(name: str, n: int, start: int = 0, *, seq_drop: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    ts = pd.date_range("2026-01-01", periods=n, freq="5s", tz="UTC")[seq_drop:]
    m = len(ts)
    dirs = rng.choice(["up", "down", "flat"], size=m, p=[0.2, 0.2, 0.6])
    truth = rng.choice(["up", "down", "flat"], size=m, p=[0.15, 0.15, 0.7])
    p_up = rng.random(m)
    p_dn = rng.random(m)
    p_fl = rng.random(m)
    s = p_up + p_dn + p_fl
    return pd.DataFrame({
        "ts": ts,
        "direction": dirs,
        "confidence": rng.random(m),
        "p_down": p_dn / s, "p_flat": p_fl / s, "p_up": p_up / s,
        "direction_true": truth,
    })


def test_objective_loads_output_meta():
    obj = load_objective("nq-50pt-xgb-tcn")
    assert obj["objective_id"] == "nq-50pt-xgb-tcn"
    cfg = topology_output_meta(obj)
    assert cfg is not None
    assert cfg["bases"] == ["xgboost", "tcn"]
    assert cfg["meta"]["family"] == "logistic_regression"
    assert cfg["meta"]["embargo_bars"] == 120
    assert obj["labels"]["move_threshold"] == 50
    assert obj["search"]["feature_views"]["tcn"] == "momentum_edge"


def test_econ_untouched():
    obj = load_objective("nq-50pt-econ")
    assert obj["search"]["topology"]["mode"] == "primary_meta"
    assert topology_output_meta(obj) is None
    assert obj["search"]["feature_views"]["tcn"] == "sequence_baseline"


def test_align_on_ts_inner_join_unequal_lengths():
    a = _oof("a", 40, seq_drop=0)
    b = _oof("b", 40, seq_drop=8)
    aligned = align_on_ts({"tree": a, "seq": b}, ["tree", "seq"])
    assert len(aligned) == 32
    assert aligned["ts"].is_monotonic_increasing
    assert "tree__direction" in aligned.columns
    assert "seq__p_up" in aligned.columns


def test_x_meta_is_output_vectors_only():
    a = _oof("a", 20)
    b = _oof("b", 20)
    aligned = align_on_ts({"m1": a, "m2": b}, ["m1", "m2"])
    X, names = build_X_meta(aligned, ["m1", "m2"])
    assert X.shape == (20, 14)  # 6*2 + n_directional + agree_side
    assert all(not n.startswith("of_") and not n.startswith("rv_") for n in names)
    assert names.count("m1__p_up") == 1


def test_proposed_side_conflict_is_none():
    ts = pd.date_range("2026-01-01", periods=3, freq="5s", tz="UTC")
    a = pd.DataFrame({"ts": ts, "direction": ["up", "up", "flat"],
                      "confidence": [0.8, 0.8, 0.4],
                      "p_down": 0.1, "p_flat": 0.1, "p_up": 0.8,
                      "direction_true": ["up", "down", "flat"]})
    b = pd.DataFrame({"ts": ts, "direction": ["up", "down", "flat"],
                      "confidence": [0.7, 0.9, 0.3],
                      "p_down": [0.1, 0.8, 0.2], "p_flat": 0.1, "p_up": [0.8, 0.1, 0.2],
                      "direction_true": ["up", "down", "flat"]})
    aligned = align_on_ts({"m1": a, "m2": b}, ["m1", "m2"])
    side = proposed_side(aligned, ["m1", "m2"])
    assert list(side) == ["up", None, None]


def test_compose_runs_logistic_gate():
    a = _oof("aa", 80)
    b = _oof("bb", 80, seq_drop=5)
    out = compose({"aa": a, "bb": b}, bases=["aa", "bb"],
                  approve_threshold=0.5, n_folds=3, embargo_bars=2,
                  meta_family="logistic_regression")
    assert "meta_prob" in out.columns
    assert "proposed_side" in out.columns
    assert set(out["direction"].unique()) <= {"up", "down", "flat"}
    assert len(out) == 75


def test_sequence_windows_use_actual_width():
    from src.models.sequence_windows import make_windows, trim_warmup
    X = np.zeros((20, 7), dtype="float32")
    W = make_windows(X, 8)
    assert W.shape == (13, 8, 7)
    y = np.arange(20)
    assert list(trim_warmup(y, 8)) == list(range(7, 20))
