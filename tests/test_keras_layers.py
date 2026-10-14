"""Serializable Keras layers + first-party .keras load (ADR-059 / ADR-077)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.serving.bundle import load_serving_bundle

_LAB_TCN = Path(
    "/opt/vtech/data/research/custom_test/nq-50pt-econ/"
    "20260822T050238Z/models/tcn_so_mbp_vix"
)
_LAB_LSTM_TBM_OPT = Path(
    "/opt/vtech/data/research/custom_test/nq-50pt-tbm-opt/"
    "20260827T212219Z/models/attention_lstm"
)


def _inject_keras15_initializer_axes(src: Path, dest: Path) -> None:
    """Rewrite GlorotUniform configs as Keras 3.15 would (input_axes/output_axes)."""
    with zipfile.ZipFile(src) as zin:
        data = {name: zin.read(name) for name in zin.namelist()}
    cfg = json.loads(data["config.json"])

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "GlorotUniform" and isinstance(obj.get("config"), dict):
                obj["config"].setdefault("input_axes", None)
                obj["config"].setdefault("output_axes", None)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(cfg)
    data["config.json"] = json.dumps(cfg).encode()
    with zipfile.ZipFile(dest, "w") as zout:
        for name, blob in data.items():
            zout.writestr(name, blob)


def test_dual_side_fraction_is_relative_long_mass():
    from tensorflow import keras
    from src.models.keras_layers import DualSideFraction

    keras.backend.clear_session()
    pl = keras.Input(shape=(1,), name="p_long")
    ps = keras.Input(shape=(1,), name="p_short")
    frac = DualSideFraction(name="side_rank")([pl, ps])
    model = keras.Model([pl, ps], frac)
    out = model.predict(
        {"p_long": np.array([[0.8], [0.2], [0.5]], dtype="float32"),
         "p_short": np.array([[0.2], [0.8], [0.5]], dtype="float32")},
        verbose=0,
    )
    np.testing.assert_allclose(out.reshape(-1), [0.8, 0.2, 0.5], atol=1e-5)
    keras.backend.clear_session()


def test_last_timestep_selects_final_bar():
    from tensorflow import keras
    from src.models.keras_layers import LastTimestep

    keras.backend.clear_session()
    x = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    layer = LastTimestep(name="last_step")
    out = layer(x).numpy()
    np.testing.assert_array_equal(out, x[:, -1, :])
    keras.backend.clear_session()


def test_tcn_pooling_last_roundtrip_safe_mode(tmp_path):
    """New TCN pooling=last graphs load under Keras 3 safe_mode=True."""
    from tensorflow import keras
    from src.models.tcn_plugin import _build_tcn

    keras.backend.clear_session()
    model = _build_tcn(
        seq_len=8, n_features=3,
        p={"filters": 4, "kernel_size": 2, "dilations": [1], "dropout": 0.0,
           "pooling": "last"},
    )
    assert any(l.__class__.__name__ == "LastTimestep" for l in model.layers)
    path = tmp_path / "tcn_last.keras"
    model.save(path)
    loaded = keras.models.load_model(path, safe_mode=True)
    x = np.zeros((2, 8, 3), dtype="float32")
    a, b = model.predict(x, verbose=0), loaded.predict(x, verbose=0)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    keras.backend.clear_session()


def test_legacy_python_lambda_blocked_then_loads_via_serving(tmp_path):
    """Existing Custom Test TCN files used Lambda(lambda t: t[:, -1, :])."""
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.backend.clear_session()
    inp = keras.Input(shape=(4, 2))
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(inp)
    out = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    dest = tmp_path / "bundle"
    dest.mkdir()
    model.save(dest / "model.keras")
    (dest / "feature_schema.json").write_text(json.dumps({"feature_cols": ["a", "b"]}))
    (dest / "finalize_manifest.json").write_text(json.dumps({
        "family": "tcn",
        "candidate_id": "tcn_so_mbp_vix",
        "spec_params": {"pooling": "last", "seq_len": 4},
        "kind": "sequence",
        "seq_len": 4,
    }))

    with pytest.raises(ValueError, match="Lambda"):
        keras.models.load_model(dest / "model.keras")

    bundle = load_serving_bundle(dest, validate=False)
    assert bundle["family"] == "tcn"
    pred = bundle["booster"].predict(np.zeros((2, 4, 2), dtype="float32"), verbose=0)
    assert pred.shape == (2, 3)
    keras.backend.clear_session()


def test_patch_hazard_roundtrip_no_lambda(tmp_path):
    from tensorflow import keras
    from src.models.patch_hazard_plugin import PatchHazardPlugin

    keras.backend.clear_session()
    plugin = PatchHazardPlugin()
    plugin._params = {
        "seq_len": 16, "patch_len": 8, "d_model": 8, "n_heads": 2, "n_layers": 1,
        "ff_dim": 16, "include_quantile": False, "dropout": 0.0,
    }
    cols = [f"vw_{i}" for i in range(2)] + [f"of_{i}" for i in range(2)] + [
        f"rv_{i}" for i in range(2)] + [f"iv_{i}" for i in range(2)]
    model = plugin._rebuild(8, cols)
    assert not any(l.__class__.__name__ == "Lambda" for l in model.layers)
    path = tmp_path / "patch.keras"
    model.save(path)
    loaded = keras.models.load_model(path, safe_mode=True)
    x = np.zeros((2, 16, 8), dtype="float32")
    out0, out1 = model.predict(x, verbose=0), loaded.predict(x, verbose=0)

    def _as_arrays(pred):
        if isinstance(pred, dict):
            return [pred[k] for k in sorted(pred)]
        if isinstance(pred, (list, tuple)):
            return list(pred)
        return [pred]

    for a, b in zip(_as_arrays(out0), _as_arrays(out1)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    keras.backend.clear_session()


@pytest.mark.skipif(not (_LAB_TCN / "model.keras").exists(), reason="lab TCN artifact not on disk")
def test_load_lab_tcn_so_mbp_vix():
    bundle = load_serving_bundle(_LAB_TCN, validate=False)
    assert bundle["family"] == "tcn"
    names = [l.name for l in bundle["booster"].layers]
    assert "last_step" in names
    seq_len = int(bundle.get("seq_len") or 128)
    n_feat = len(bundle["feature_cols"])
    pred = bundle["booster"].predict(
        np.zeros((2, seq_len, n_feat), dtype="float32"), verbose=0)
    assert pred.shape == (2, 3)


def test_load_keras15_glorot_uniform_axes(tmp_path):
    """CPU Keras 3.13 must load GPU 3.15 LSTM files (ADR-077)."""
    import keras
    from tensorflow.keras import layers
    from src.models.keras_layers import load_keras_model

    keras.backend.clear_session()
    inp = layers.Input(shape=(4, 2))
    x = layers.LSTM(4, return_sequences=False)(inp)
    out = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    raw = tmp_path / "lstm.keras"
    model.save(raw)
    patched = tmp_path / "lstm_k15.keras"
    _inject_keras15_initializer_axes(raw, patched)

    with pytest.raises(TypeError, match="input_axes"):
        keras.models.load_model(patched, safe_mode=False)

    loaded = load_keras_model(patched, compile=False)
    x = np.zeros((2, 4, 2), dtype="float32")
    a, b = model.predict(x, verbose=0), loaded.predict(x, verbose=0)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    keras.backend.clear_session()
    # Patch must restore the classmethod descriptor (not a bound method).
    layers.LSTM(2)(layers.Input(shape=(3, 2)))
    keras.backend.clear_session()


@pytest.mark.skipif(
    not (_LAB_LSTM_TBM_OPT / "model.keras").exists(),
    reason="lab tbm-opt attention_lstm artifact not on disk",
)
def test_load_lab_tbm_opt_attention_lstm():
    bundle = load_serving_bundle(_LAB_LSTM_TBM_OPT, validate=False)
    assert bundle["family"] == "attention_lstm"
    seq_len = int(bundle.get("seq_len") or 128)
    n_feat = len(bundle["feature_cols"])
    pred = bundle["booster"].predict(
        np.zeros((2, seq_len, n_feat), dtype="float32"), verbose=0)
    assert pred.shape == (2, 3)
    assert abs(float(pred[0].sum()) - 1.0) < 1e-5
