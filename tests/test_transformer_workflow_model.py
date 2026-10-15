"""Transformer regressions required by the isolated end-to-end workflow."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.models.base import ModelSpec
from src.models.transformer_plugin import TransformerPlugin

pytestmark = pytest.mark.ml_workflow

PARAMS = {
    "seq_len": 4, "d_model": 8, "n_heads": 2, "n_layers": 1,
    "n_features": 3, "epochs": 1, "batch_size": 4, "dropout": 0.0,
    "learning_rate": 0.003,
}


@pytest.mark.parametrize("invalid", [
    {"seq_len": 0}, {"n_heads": 0}, {"d_model": 7}, {"n_layers": -1},
    {"batch_size": 0}, {"epochs": 1.5}, {"n_features": True},
    {"dropout": 1.0}, {"dropout": float("nan")},
    {"learning_rate": 0.0}, {"learning_rate": float("inf")},
    {"head": "unknown"},
])
def test_invalid_model_design_fails_before_tensorflow(invalid: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TransformerPlugin().build(ModelSpec("transformer", {**PARAMS, **invalid}), {})


def test_position_embedding_trains_and_survives_safe_serving_reload(tmp_path) -> None:
    tf = pytest.importorskip("tensorflow")
    from src.common.vm_config import get_vm_config
    from src.models.keras_layers import load_keras_model

    workers = get_vm_config().workers_for(mem_per_worker_gb=1.0)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(workers)
        tf.config.threading.set_inter_op_parallelism_threads(workers)
    except RuntimeError:
        pass  # Another explicit test scope may already have initialized TF.
    tf.keras.utils.set_random_seed(11)
    model = TransformerPlugin().build(ModelSpec("transformer", dict(PARAMS)), {})
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.003)
    embedding = model.get_layer("position_embedding")
    assert any(weight is embedding.embeddings for weight in model.trainable_weights)
    before = embedding.get_weights()[0].copy()
    x = np.random.default_rng(11).normal(size=(8, 4, 3)).astype("float32")
    model.train_on_batch(x, np.arange(8, dtype="int32") % 3)
    assert not np.array_equal(before, embedding.get_weights()[0])
    expected = model(x, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    restored = load_keras_model(path, safe_mode=True)
    np.testing.assert_allclose(restored(x, training=False).numpy(), expected, atol=1e-7)
    np.testing.assert_array_equal(
        restored.get_layer("position_embedding").get_weights()[0],
        embedding.get_weights()[0],
    )


def test_dual_binary_honors_learning_rate() -> None:
    pytest.importorskip("tensorflow")
    model = TransformerPlugin().build(
        ModelSpec("transformer", {**PARAMS, "head": "dual_binary"}), {},
    )
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.003)


def _fake_trainer(monkeypatch, objective=None):
    import src.models.transformer_plugin as module

    models = []

    class Model:
        def fit(self, data, **kwargs):
            self.data = data
            self.kwargs = kwargs
            return SimpleNamespace(history={"loss": [0.5]})

    def rebuild(self, n_features):
        result = Model()
        models.append(result)
        return result

    monkeypatch.setattr(TransformerPlugin, "_rebuild", rebuild)
    monkeypatch.setattr(module, "train_callbacks", lambda **kw: [])
    monkeypatch.setattr(
        module, "batched_sequence",
        lambda X, y, batch_size, sample_weight=None: (X, y, sample_weight),
    )
    plugin = TransformerPlugin()
    built = plugin.build(ModelSpec("transformer", dict(PARAMS)), objective or {})
    return plugin, built, models


def test_cost_loss_initializes_train_and_validation_targets(monkeypatch) -> None:
    objective = {
        "custom_optimization": True,
        "cost_matrix": {"costs": {
            "down": [0, 1, 2], "flat": [1, 0, 1], "up": [2, 1, 0],
        }},
    }
    plugin, model, models = _fake_trainer(monkeypatch, objective)
    x = np.arange(36, dtype="float32").reshape(12, 3)
    y = np.arange(12) % 3
    callback = object()
    fitted = plugin.fit(model, [{
        "X_train": x, "y_train": y, "X_val": x, "y_val": y,
    }], callbacks=[callback])
    np.testing.assert_array_equal(models[-1].data[1], y[3:])
    np.testing.assert_array_equal(models[-1].kwargs["validation_data"][1], y[3:])
    assert models[-1].data[2] is None
    assert models[-1].kwargs["callbacks"] == [callback]
    assert fitted.meta["history"] == [{"loss": [0.5]}]


def test_three_dimensional_folds_have_independent_models(monkeypatch) -> None:
    plugin, model, models = _fake_trainer(monkeypatch)
    fold = {"X_train": np.ones((8, 4, 3)), "y_train": np.arange(8) % 3}
    fitted = plugin.fit(model, [fold, fold])
    assert len(models) == 3  # build() plus one fresh model per fold.
    assert len({id(item) for item in models}) == 3
    assert fitted.artifact is models[-1]
    assert fitted.meta["scaler"] is None


def test_fold_shapes_and_finite_values_are_required(monkeypatch) -> None:
    plugin, model, _ = _fake_trainer(monkeypatch)
    with pytest.raises(ValueError, match="at least one"):
        plugin.fit(model, [])
    with pytest.raises(ValueError, match="sequence width"):
        plugin.fit(model, [{"X_train": np.ones((8, 2, 3)), "y_train": np.ones(8)}])
    with pytest.raises(ValueError, match="finite"):
        plugin.fit(model, [{"X_train": np.full((8, 3), np.nan), "y_train": np.ones(8)}])
    with pytest.raises(ValueError, match="row counts"):
        plugin.fit(model, [{"X_train": np.ones((8, 3)), "y_train": np.ones(7)}])
    with pytest.raises(ValueError, match="require y_val"):
        plugin.fit(model, [{
            "X_train": np.ones((8, 3)), "y_train": np.ones(8),
            "X_val": np.ones((4, 3)),
        }])
