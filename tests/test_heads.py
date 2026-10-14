"""Tests for shared hazard / quantile heads (src/models/heads.py, ADR-053)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.heads import (
    CR_NEITHER,
    CR_STOP,
    CR_TARGET,
    DEFAULT_N_BUCKETS,
    DEFAULT_QUANTILES,
    EXCURSION_COLS,
    NO_TOUCH_CLASS,
    attach_hazard_heads,
    compile_hazard_model,
    decode_hazard_class,
    encode_cr_sequence,
    encode_hazard_class,
    encode_pathlabel_frame,
    integrate_hazard_probs,
    masses_from_logits,
    n_hazard_classes,
    n_quantile_outputs,
    p_first_from_cr_probs,
    pinball_loss,
    serving_from_cr_logits,
    serving_from_logits,
)
from src.training.path_labels import CENSORED, STOP, TARGET


# ── codec ─────────────────────────────────────────────────────────────────────

def test_class_count_is_no_touch_plus_two_sides_of_buckets():
    assert n_hazard_classes(20) == 41
    assert n_hazard_classes(6) == 13
    assert n_quantile_outputs() == 12


def test_censored_encodes_to_class_zero_regardless_of_sentinel_bucket():
    assert encode_hazard_class(CENSORED, DEFAULT_N_BUCKETS).item() == NO_TOUCH_CLASS
    assert encode_hazard_class(CENSORED, 0).item() == NO_TOUCH_CLASS


def test_target_and_stop_buckets_do_not_overlap():
    t0 = encode_hazard_class(TARGET, 0).item()
    t_last = encode_hazard_class(TARGET, 19).item()
    s0 = encode_hazard_class(STOP, 0).item()
    s_last = encode_hazard_class(STOP, 19).item()
    assert t0 == 1 and t_last == 20
    assert s0 == 21 and s_last == 40
    assert len({t0, t_last, s0, s_last, 0}) == 5


def test_encode_decode_roundtrip():
    events = np.array([CENSORED, TARGET, STOP, TARGET, STOP], dtype=np.int8)
    buckets = np.array([20, 0, 0, 19, 7], dtype=np.int16)
    ids = encode_hazard_class(events, buckets)
    ev2, bu2 = decode_hazard_class(ids)
    np.testing.assert_array_equal(ev2, events)
    np.testing.assert_array_equal(bu2, buckets)


def test_touched_row_with_sentinel_bucket_is_rejected():
    with pytest.raises(ValueError, match="bucket"):
        encode_hazard_class(TARGET, DEFAULT_N_BUCKETS)


def test_encode_pathlabel_frame_matches_event_columns():
    n = 4
    frame = pd.DataFrame({
        "ts": pd.date_range("2026-03-02 14:30:00+00:00", periods=n, freq="s"),
        "event_up": [CENSORED, TARGET, STOP, TARGET],
        "bucket_up": [20, 2, 0, 19],
        "event_down": [STOP, CENSORED, TARGET, CENSORED],
        "bucket_down": [4, 20, 1, 20],
        "mfe_up_r": [0.1, 0.2, 0.3, 0.4],
        "mae_up_r": [0.0, 0.1, 0.2, 0.3],
        "mfe_down_r": [0.5, 0.0, 0.1, 0.0],
        "mae_down_r": [0.2, 0.2, 0.2, 0.2],
    })
    packed = encode_pathlabel_frame(frame)
    assert packed["y_up"].tolist() == [
        0,
        encode_hazard_class(TARGET, 2).item(),
        encode_hazard_class(STOP, 0).item(),
        encode_hazard_class(TARGET, 19).item(),
    ]
    assert packed["y_down"][0] == encode_hazard_class(STOP, 4).item()
    assert packed["y_quantile"].shape == (4, 4)
    np.testing.assert_allclose(packed["y_quantile"][:, 0], frame["mfe_up_r"])


# ── integrate + serving ───────────────────────────────────────────────────────

def _one_hot(cls: int, n_cls: int = 41) -> np.ndarray:
    p = np.zeros(n_cls, dtype="float64")
    p[cls] = 1.0
    return p


def test_integrate_one_hot_target_bucket_is_pure_target():
    cls = encode_hazard_class(TARGET, 5).item()
    p_t, p_s, p_c = integrate_hazard_probs(_one_hot(cls)[None, :])
    assert p_t.item() == pytest.approx(1.0)
    assert p_s.item() == pytest.approx(0.0)
    assert p_c.item() == pytest.approx(0.0)


def test_integrate_one_hot_stop_and_censored():
    p_t, p_s, p_c = integrate_hazard_probs(
        _one_hot(encode_hazard_class(STOP, 3).item())[None, :])
    assert p_s.item() == pytest.approx(1.0) and p_t.item() == pytest.approx(0.0)
    p_t, p_s, p_c = integrate_hazard_probs(_one_hot(0)[None, :])
    assert p_c.item() == pytest.approx(1.0)


def test_three_masses_sum_to_one_on_a_random_simplex():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(8, n_hazard_classes()))
    masses = masses_from_logits(logits, logits)
    np.testing.assert_allclose(
        masses.p_target_up + masses.p_stop_up + masses.p_censored_up, 1.0, atol=1e-12)


def test_serving_picks_the_larger_target_mass():
    n_cls = n_hazard_classes()
    up = np.zeros((2, n_cls))
    down = np.zeros((2, n_cls))
    # row 0: up-target bucket 0 vs down-censored → up
    up[0, encode_hazard_class(TARGET, 0)] = 20.0
    down[0, 0] = 20.0
    # row 1: down-target vs up-stop → down
    up[1, encode_hazard_class(STOP, 0)] = 20.0
    down[1, encode_hazard_class(TARGET, 2)] = 20.0
    df = serving_from_logits(up, down, ts=pd.date_range(
        "2026-03-02", periods=2, freq="s", tz="UTC"))
    assert df["direction"].tolist() == ["up", "down"]
    assert (df["confidence"] > 0.9).all()
    assert set(df.columns) >= {"ts", "direction", "confidence"}


def test_cr_censored_is_neither_on_every_bucket():
    y, mask = encode_cr_sequence(np.array([CENSORED]), np.array([20]))
    assert y.shape == (1, 20)
    assert (y == CR_NEITHER).all()
    assert (mask == 1.0).all()


def test_cr_target_masks_after_exit_bucket():
    y, mask = encode_cr_sequence(np.array([TARGET]), np.array([2]))
    assert y[0, 2] == CR_TARGET
    assert (y[0, :2] == CR_NEITHER).all()
    assert (mask[0, :3] == 1.0).all()
    assert (mask[0, 3:] == 0.0).all()


def test_cr_stop_at_bucket_zero():
    y, mask = encode_cr_sequence(np.array([STOP]), np.array([0]))
    assert y[0, 0] == CR_STOP
    assert mask[0, 0] == 1.0
    assert (mask[0, 1:] == 0.0).all()


def test_cr_integrate_certain_target_at_t0():
    p = np.zeros((1, 20, 3))
    p[0, 0, CR_TARGET] = 1.0
    p[0, 1:, CR_NEITHER] = 1.0
    pt, ps, pn = p_first_from_cr_probs(p)
    assert pt.item() == pytest.approx(1.0)
    assert ps.item() == pytest.approx(0.0)


def test_cr_peaked_target_every_bucket_stays_in_unit_interval():
    """Old 1e-8 neither-floor leaked survival and made P(target-first) > 1."""
    p = np.zeros((3, 20, 3))
    p[..., CR_TARGET] = 1.0
    pt, ps, pn = p_first_from_cr_probs(p)
    assert (pt >= 0.0).all() and (pt <= 1.0).all()
    np.testing.assert_allclose(pt, 1.0, atol=1e-12)
    np.testing.assert_allclose(ps, 0.0, atol=1e-12)
    np.testing.assert_allclose(pt + ps + pn, 1.0, atol=1e-12)


def test_serving_from_cr_extreme_logits_in_unit_interval():
    up = np.full((4, 20, 3), -20.0)
    up[..., CR_TARGET] = 1e6
    down = np.full((4, 20, 3), -20.0)
    down[..., CR_STOP] = 1e6
    df = serving_from_cr_logits(up, down, ts=pd.date_range(
        "2026-03-02", periods=4, freq="s", tz="UTC"), plugin_name="path_tcn")
    assert df["confidence"].between(0.0, 1.0).all()
    assert (df["direction"] == "up").all()


def test_serving_from_cr_nan_logits_is_flat_zero():
    nan = np.full((1, 20, 3), np.nan)
    df = serving_from_cr_logits(nan, nan, plugin_name="path_tcn")
    assert df["direction"].iloc[0] == "flat"
    assert df["confidence"].iloc[0] == 0.0


def test_inference_r_gate_zeros_sub_floor_rows():
    from src.models.heads import MIN_INFERENCE_R, apply_inference_r_gate, serving_from_logits

    n_cls = n_hazard_classes()
    logits = np.zeros((3, n_cls))
    logits[:, encode_hazard_class(TARGET, 0)] = 20.0
    df = serving_from_logits(logits, np.zeros((3, n_cls)))
    assert (df["direction"] == "up").all()
    gated = apply_inference_r_gate(df, np.array([80.0, 10.0, np.nan]))
    assert list(gated["direction"]) == ["up", "flat", "flat"]
    assert gated["confidence"].iloc[1] == 0.0
    assert MIN_INFERENCE_R == pytest.approx(67.3)


def test_equal_target_mass_is_flat_not_a_coin_flip():
    n_cls = n_hazard_classes()
    logits = np.zeros((1, n_cls))
    logits[0, encode_hazard_class(TARGET, 1)] = 3.0
    df = serving_from_logits(logits, logits)
    assert df["direction"].iloc[0] == "flat"
    assert 0.0 <= df["confidence"].iloc[0] <= 1.0


# ── pinball ───────────────────────────────────────────────────────────────────

def test_perfect_quantile_prediction_is_zero_pinball():
    y = np.array([[0.2, 0.1, 0.3, 0.05]])
    # (1, 4, 3) identical across quantiles
    pred = np.repeat(y[..., None], 3, axis=-1)
    assert pinball_loss(y, pred) == pytest.approx(0.0)


def test_overpredicting_is_penalised_more_at_low_quantile():
    y = np.array([[1.0]])
    over = np.array([[[2.0, 2.0, 2.0]]])   # all quantiles = 2
    under = np.array([[[0.0, 0.0, 0.0]]])
    # pinball(q) for error e=y-pred: q*e if e>0 (under), (q-1)*e if e<0 (over)
    # over at q=0.1 costs 0.9; under at q=0.1 costs 0.1 — over is costlier
    # averaged over 3 quantiles the mean still ranks over > under here? Check q=0.1 only.
    assert pinball_loss(y, np.array([[[2.0]]]), quantiles=(0.1,)) > \
        pinball_loss(y, np.array([[[0.0]]]), quantiles=(0.1,))
    assert pinball_loss(y, over) > 0.0 and pinball_loss(y, under) > 0.0


def test_flat_target_major_pred_matches_shaped():
    y = np.array([[0.2, 0.1, 0.3, 0.05]])
    shaped = np.arange(12, dtype="float64").reshape(1, 4, 3) / 20.0
    flat = shaped.reshape(1, 12)
    assert pinball_loss(y, flat) == pytest.approx(pinball_loss(y, shaped))


# ── Keras attach / compile ────────────────────────────────────────────────────

def test_attach_hazard_heads_last_dim_is_41():
    from tensorflow import keras

    inp = keras.Input(shape=(16,))
    up, down = attach_hazard_heads(inp)
    assert tuple(up.shape) == (None, 41)
    assert tuple(down.shape) == (None, 41)
    keras.backend.clear_session()


def test_compiled_hazard_model_fits_encoded_labels():
    from tensorflow import keras

    keras.backend.clear_session()
    rng = np.random.default_rng(1)
    n, d, n_cls = 32, 8, n_hazard_classes()
    x = rng.normal(size=(n, d)).astype("float32")
    y_up = rng.integers(0, n_cls, size=n, dtype=np.int32)
    y_down = rng.integers(0, n_cls, size=n, dtype=np.int32)
    inp = keras.Input(shape=(d,))
    emb = keras.layers.Dense(8, activation="relu")(inp)
    model = compile_hazard_model(inp, emb)
    hist = model.fit(
        x, {"y_up": y_up, "y_down": y_down},
        epochs=1, batch_size=16, verbose=0,
    )
    assert np.isfinite(hist.history["loss"][0])
    logits = model.predict(x, verbose=0)
    df = serving_from_logits(logits["y_up"], logits["y_down"])
    assert len(df) == n
    assert set(df["direction"]).issubset({"up", "down", "flat"})
    keras.backend.clear_session()


def test_compiled_model_with_quantile_head_accepts_excursion_targets():
    from tensorflow import keras

    keras.backend.clear_session()
    rng = np.random.default_rng(2)
    n, d = 16, 4
    x = rng.normal(size=(n, d)).astype("float32")
    y_up = rng.integers(0, 41, size=n, dtype=np.int32)
    y_down = rng.integers(0, 41, size=n, dtype=np.int32)
    y_q = rng.random(size=(n, len(EXCURSION_COLS))).astype("float32")
    inp = keras.Input(shape=(d,))
    model = compile_hazard_model(inp, inp, include_quantile=True)
    hist = model.fit(
        x, {"y_up": y_up, "y_down": y_down, "y_quantile": y_q},
        epochs=1, batch_size=8, verbose=0,
    )
    assert np.isfinite(hist.history["loss"][0])
    keras.backend.clear_session()
