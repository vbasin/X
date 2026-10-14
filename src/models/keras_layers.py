"""Serializable Keras layers for DL plugins.

Keras 3 refuses to deserialize ``Lambda`` layers whose ``function`` is a
Python lambda (``safe_mode=True`` by default). Custom Test TCN
``pooling=last`` artifacts hit that on WebUI Backtest. Use these layers in
graphs; load first-party ``.keras`` files through :func:`load_keras_model`.

Use the standalone ``keras`` package (not ``tensorflow.keras``) so
``keras.saving.register_keras_serializable`` exists on TF 2.21 + Keras 3.

GPU Custom Test LSTM files are Keras 3.15; the CPU venv is 3.13. Serving
drops unknown initializer keys (``input_axes`` / ``output_axes``) so those
artifacts load without bumping the control-plane Keras pin (ADR-077).
"""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Any, Iterator

import keras
from keras import layers


@keras.saving.register_keras_serializable(package="vtech")
class LastTimestep(layers.Layer):
    """Select the last causal timestep: ``(B, T, F) -> (B, F)``."""

    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, Any]:
        return (input_shape[0], input_shape[-1])

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@keras.saving.register_keras_serializable(package="vtech")
class GatherFeatures(layers.Layer):
    """Gather named feature indices on the last axis: ``(B, T, F) -> (B, T, K)``."""

    def __init__(self, indices: list[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]

    def call(self, inputs):
        return keras.ops.take(inputs, self.indices, axis=-1)

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, Any, int]:
        return (input_shape[0], input_shape[1], len(self.indices))

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "indices": list(self.indices)}


@keras.saving.register_keras_serializable(package="vtech")
class ZerosTokens(layers.Layer):
    """Dummy patch tokens when a stream has no columns: ``(B, n_patches, 1)``."""

    def __init__(self, n_patches: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_patches = int(n_patches)

    def call(self, inputs):
        batch = keras.ops.shape(inputs)[0]
        return keras.ops.zeros((batch, self.n_patches, 1), dtype=inputs.dtype)

    def compute_output_shape(self, input_shape: Any) -> tuple[Any, int, int]:
        return (input_shape[0], self.n_patches, 1)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "n_patches": self.n_patches}


@keras.saving.register_keras_serializable(package="vtech")
class DualSideFraction(layers.Layer):
    """Relative long mass ``p_long / (p_long + p_short)`` (ADR-088).

    Independent dual-binary sigmoids do not compete. This scalar is the
    2-way share used as a dense path-side regularizer on the decision
    heads. Serving still reads ``p_long`` / ``p_short`` only.
    """

    def call(self, inputs):
        pl, ps = inputs
        pl = keras.ops.clip(pl, 1e-6, 1.0 - 1e-6)
        ps = keras.ops.clip(ps, 1e-6, 1.0 - 1e-6)
        return pl / (pl + ps)

    def compute_output_shape(self, input_shape: Any) -> Any:
        return input_shape[0]

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@keras.saving.register_keras_serializable(package="vtech")
class ExpectedCostLoss(keras.losses.Loss):
    """Softmax expected cost: ``sum_k C[true, k] * p_k`` (ADR-072).

    ``costs`` is a 3×3 list in down/flat/up order (rows = true, cols = pred).
    The model output is a 3-class probability vector (already softmaxed).
    """

    def __init__(
        self,
        costs: list[list[float]],
        name: str = "expected_cost",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.costs = [[float(x) for x in row] for row in costs]
        if len(self.costs) != 3 or any(len(r) != 3 for r in self.costs):
            raise ValueError("ExpectedCostLoss.costs must be a 3x3 matrix")

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y = keras.ops.cast(keras.ops.reshape(y_true, [-1]), "int32")
        C = keras.ops.convert_to_tensor(self.costs, dtype=y_pred.dtype)
        row = keras.ops.take(C, y, axis=0)
        return keras.ops.sum(row * y_pred, axis=-1)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "costs": self.costs}


def keras_custom_objects() -> dict[str, Any]:
    """Name map for ``load_model(..., custom_objects=...)``."""
    from src.models.attention_lstm import AttentionLayer

    objs: dict[str, Any] = {
        "LastTimestep": LastTimestep,
        "GatherFeatures": GatherFeatures,
        "ZerosTokens": ZerosTokens,
        "ExpectedCostLoss": ExpectedCostLoss,
        "DualSideFraction": DualSideFraction,
        "AttentionLayer": AttentionLayer,
    }
    # Keras 3 registered-class ids use ``package>ClassName``.
    for cls in (
        LastTimestep, GatherFeatures, ZerosTokens, ExpectedCostLoss, DualSideFraction,
    ):
        objs[f"vtech>{cls.__name__}"] = cls
    return objs


@contextlib.contextmanager
def _tolerate_newer_initializer_config() -> Iterator[None]:
    """Drop Keras 3.15+ initializer keys unknown to older Keras (ADR-077).

    ``GlorotUniform.get_config()`` on 3.15 includes ``input_axes`` /
    ``output_axes``. ``Initializer.from_config`` on 3.13 does ``cls(**config)``
    and raises ``TypeError``, which WebUI Backtest surfaces as HTTP 500.
    """
    from keras.src.initializers.initializer import Initializer

    # ``__dict__`` keeps the classmethod descriptor. ``Initializer.from_config``
    # is a bound method; assigning that back makes GlorotUniform.from_config
    # instantiate the base Initializer (NotImplementedError on __call__).
    orig = Initializer.__dict__["from_config"]

    @classmethod
    def _from_config(cls, config):  # noqa: ANN001
        cfg = dict(config)
        cfg.pop("dtype", None)
        try:
            return cls(**cfg)
        except TypeError:
            params = inspect.signature(cls.__init__).parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                raise
            allowed = {name for name in params if name != "self"}
            return cls(**{k: v for k, v in cfg.items() if k in allowed})

    Initializer.from_config = _from_config
    try:
        yield
    finally:
        Initializer.from_config = orig


def _short_keras_load_error(exc: BaseException) -> str:
    """Innermost Keras deserialize line — the wrapper repeats the same TypeError."""
    msg = str(exc).strip()
    marker = "Exception encountered: "
    if marker in msg:
        msg = msg.rsplit(marker, 1)[-1].strip()
    return msg.splitlines()[0][:400] if msg else type(exc).__name__


def load_keras_model(
    path: str | Path,
    *,
    compile: bool = True,
    custom_objects: dict[str, Any] | None = None,
    safe_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a first-party ``.keras`` artifact.

    Serving / Backtest / Live only load models we trained. Keras 3's default
    ``safe_mode=True`` blocks Python-lambda ``Lambda`` layers already on disk
    (TCN ``pooling=last`` Custom Test bundles). New graphs use serializable
    layers; ``safe_mode=False`` remains required for those older files.

    GPU 3.15 LSTM artifacts also carry initializer keys 3.13 rejects; those
    are dropped for the duration of this call (ADR-077).
    """
    objs = keras_custom_objects()
    if custom_objects:
        objs.update(custom_objects)
    try:
        with _tolerate_newer_initializer_config():
            return keras.models.load_model(
                path,
                custom_objects=objs,
                compile=compile,
                safe_mode=safe_mode,
                **kwargs,
            )
    except TypeError as exc:
        raise ValueError(
            f"failed to load keras model {path} "
            f"(Keras {keras.__version__}): {_short_keras_load_error(exc)}"
        ) from exc
