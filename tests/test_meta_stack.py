"""Tests for meta_stack topology (signed-evidence binary LR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.search.meta_stack import (
    compose,
    fit_final,
    signed_features,
    topology_meta_stack,
)


def _synthetic_bases(n: int = 400, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Two bases whose signed p_up-p_down aligns with a deterministic direction.

    Encodes a learnable relationship:
      - direction_true = 'up'   when a>0 and b>0
      - direction_true = 'down' when a<0 and b<0
      - direction_true = 'flat' otherwise
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-07-01", periods=n, freq="5s", tz="UTC")
    a = rng.uniform(-0.9, 0.9, n)
    b = rng.uniform(-0.9, 0.9, n)
    truth = np.where((a > 0.2) & (b > 0.2), "up",
             np.where((a < -0.2) & (b < -0.2), "down", "flat"))

    def _mk(sig: np.ndarray) -> pd.DataFrame:
        p_up = 0.5 + 0.5 * np.clip(sig, -1, 1) * 0.9
        p_down = 1.0 - p_up - 1e-6
        p_flat = 1.0 - p_up - p_down
        direction = np.where(p_up > 0.55, "up", np.where(p_down > 0.55, "down", "flat"))
        confidence = np.maximum(p_up, np.maximum(p_down, p_flat))
        return pd.DataFrame({
            "ts": ts, "direction": direction, "confidence": confidence,
            "p_down": p_down, "p_flat": p_flat, "p_up": p_up,
            "direction_true": truth,
        })

    return {"xgboost": _mk(a), "tcn": _mk(b)}


def test_topology_meta_stack_parses_yaml_shape():
    obj = {"search": {"topology": {
        "mode": "meta_stack",
        "bases": ["xgboost", "tcn"],
        "meta": {"scheme": "walk_forward_OOF", "feature": "linear_weights",
                 "approve_threshold": 0.55},
    }}}
    cfg = topology_meta_stack(obj)
    assert cfg is not None
    assert cfg["bases"] == ["xgboost", "tcn"]
    assert cfg["meta"]["scheme"] == "walk_forward_OOF"
    assert cfg["meta"]["feature"] == "linear_weights"
    assert cfg["meta"]["approve_threshold"] == 0.55
    assert cfg["meta"]["family"] == "logistic_regression"


def test_topology_meta_stack_none_for_other_modes():
    assert topology_meta_stack({"search": {"topology": {"mode": "output_meta"}}}) is None
    assert topology_meta_stack({"search": {"topology": {"mode": "single"}}}) is None
    assert topology_meta_stack({}) is None


def test_signed_features_linear_range():
    oofs = _synthetic_bases(n=200)
    from src.search.output_meta import align_on_ts
    aligned = align_on_ts(oofs, ["xgboost", "tcn"])
    X, names = signed_features(aligned, ["xgboost", "tcn"], "linear_weights")
    assert X.shape == (200, 2)
    assert names == ["xgboost__s_linear_weights", "tcn__s_linear_weights"]
    assert X.min() >= -1.0 - 1e-6 and X.max() <= 1.0 + 1e-6


def test_signed_features_log_finite():
    oofs = _synthetic_bases(n=200)
    from src.search.output_meta import align_on_ts
    aligned = align_on_ts(oofs, ["xgboost", "tcn"])
    X, _ = signed_features(aligned, ["xgboost", "tcn"], "log_weights")
    assert X.shape == (200, 2)
    assert np.all(np.isfinite(X))  # eps prevents log(0)


def test_walk_forward_oof_block0_is_warmup():
    """Block 0 of walk_forward_OOF must have no meta fires (proba stays neutral)."""
    oofs = _synthetic_bases(n=500)
    out = compose(oofs, bases=["xgboost", "tcn"], scheme="walk_forward_OOF",
                  feature="linear_weights", approve_threshold=0.55, n_folds=5)
    edges = np.linspace(0, len(out), 6, dtype=int)
    block0 = out.iloc[edges[0]:edges[1]]
    # Block 0 rows have meta_finite=False → all flat
    assert (~block0["meta_finite"]).all()
    assert (block0["direction"] == "flat").all()
    # Later blocks should have at least some finite proba
    tail = out.iloc[edges[1]:]
    assert tail["meta_finite"].any()


def test_cross_fit_covers_all_rows():
    """cross_fit should produce finite proba for every row (subject to class balance)."""
    oofs = _synthetic_bases(n=500)
    out = compose(oofs, bases=["xgboost", "tcn"], scheme="cross_fit",
                  feature="linear_weights", approve_threshold=0.55, n_folds=5)
    assert out["meta_finite"].mean() > 0.9   # some folds may be class-degenerate


def test_symmetric_fire_rule():
    """Direction=up requires p_up>=thr; direction=down requires p_up<=1-thr; else flat."""
    oofs = _synthetic_bases(n=600)
    out = compose(oofs, bases=["xgboost", "tcn"], scheme="cross_fit",
                  feature="linear_weights", approve_threshold=0.55, n_folds=5)
    up_rows = out[out["direction"] == "up"]
    dn_rows = out[out["direction"] == "down"]
    fl_rows = out[out["direction"] == "flat"]
    assert (up_rows["meta_p_up"] >= 0.55 - 1e-9).all()
    assert (dn_rows["meta_p_up"] <= 0.45 + 1e-9).all()
    # Flat rows are either warmup (meta_finite False) OR 0.45 < p_up < 0.55
    fl_finite = fl_rows[fl_rows["meta_finite"]]
    assert ((fl_finite["meta_p_up"] > 0.45) & (fl_finite["meta_p_up"] < 0.55)).all()


def test_confidence_matches_side():
    """confidence = p_up for up rows, 1 - p_up for down rows, 0 for flat."""
    oofs = _synthetic_bases(n=400)
    out = compose(oofs, bases=["xgboost", "tcn"], scheme="cross_fit",
                  feature="linear_weights", approve_threshold=0.55, n_folds=5)
    up_rows = out[out["direction"] == "up"]
    dn_rows = out[out["direction"] == "down"]
    fl_rows = out[out["direction"] == "flat"]
    np.testing.assert_allclose(up_rows["confidence"].to_numpy(),
                               up_rows["meta_p_up"].to_numpy())
    np.testing.assert_allclose(dn_rows["confidence"].to_numpy(),
                               1.0 - dn_rows["meta_p_up"].to_numpy())
    assert (fl_rows["confidence"] == 0.0).all()


def test_fit_final_returns_per_base_weights():
    oofs = _synthetic_bases(n=400)
    fitted = fit_final(oofs, bases=["xgboost", "tcn"], feature="linear_weights")
    assert set(fitted["weights"].keys()) == {"xgboost__s_linear_weights",
                                              "tcn__s_linear_weights"}
    assert isinstance(fitted["bias"], float)
    # On the synthetic data, both bases point the same way as truth → positive weights
    assert fitted["weights"]["xgboost__s_linear_weights"] > 0
    assert fitted["weights"]["tcn__s_linear_weights"] > 0


def test_learns_to_downweight_broken_base():
    """When one base is pure noise, the LR should downweight it relative to the signal."""
    rng = np.random.default_rng(42)
    n = 800
    ts = pd.date_range("2025-07-01", periods=n, freq="5s", tz="UTC")
    a = rng.uniform(-0.9, 0.9, n)
    noise = rng.uniform(-0.9, 0.9, n)
    truth = np.where(a > 0.3, "up", np.where(a < -0.3, "down", "flat"))

    def _mk(sig):
        p_up = 0.5 + 0.5 * np.clip(sig, -1, 1) * 0.9
        p_down = 1.0 - p_up - 1e-6
        return pd.DataFrame({
            "ts": ts,
            "direction": np.where(p_up > 0.55, "up", np.where(p_down > 0.55, "down", "flat")),
            "confidence": np.maximum(p_up, p_down),
            "p_down": p_down, "p_flat": np.zeros(n) + 1e-6, "p_up": p_up,
            "direction_true": truth,
        })

    fitted = fit_final({"xgboost": _mk(a), "tcn": _mk(noise)},
                       bases=["xgboost", "tcn"], feature="linear_weights")
    w_signal = abs(fitted["weights"]["xgboost__s_linear_weights"])
    w_noise = abs(fitted["weights"]["tcn__s_linear_weights"])
    assert w_signal > w_noise, (
        f"expected signal base to outweigh noise base — got signal={w_signal:.3f} noise={w_noise:.3f}"
    )
