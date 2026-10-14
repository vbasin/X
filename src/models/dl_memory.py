"""GPU-memory safety for sequence DL families (LSTM, TCN, …).

Keras converts whole numpy inputs into one device constant (``_EagerConst``).
An uncapped fold window (e.g. 1.45M × 128 × F float32) then OOMs the A10
host→GPU copy with ``Dst tensor is not initialized`` (LSTM 2026-07-11, TCN
2026-08-15). A ``Sequence`` keeps arrays on the host and copies one batch per
step. Validation is monitoring-only for most plugins → stride-cap it;
predictions stream in bounded chunks.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

VAL_CAP = int(os.environ.get("VTECH_DL_VAL_CAP", "50000"))
PREDICT_CHUNK = int(os.environ.get("VTECH_DL_PREDICT_CHUNK", "65536"))


def stride_indices(n: int, cap: int) -> np.ndarray | None:
    """Evenly-spaced indices of length *cap*, or None if *n* already fits."""
    if n <= cap:
        return None
    return np.linspace(0, n - 1, cap).astype(int)


def stride_cap(X: np.ndarray, y: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly-strided subsample to *cap* rows (keeps temporal spread)."""
    idx = stride_indices(len(X), cap)
    if idx is None:
        return X, y
    return X[idx], y[idx]


def _n_rows(X: Any) -> int:
    if isinstance(X, dict):
        return int(len(next(iter(X.values()))))
    return int(len(X))


def _slice_rows(X: Any, s: slice) -> Any:
    if isinstance(X, dict):
        return {k: v[s] for k, v in X.items()}
    return X[s]


def predict_chunked(model: Any, X: Any, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """Bounded-memory predict: stream *X* through the model in row chunks.

    Dual-binary Keras graphs return a dict of heads; those are stacked to
    ``(N, 2)`` ``[p_long, p_short]`` (ADR-080). *X* may be an ndarray or a
    Keras multi-input dict (ADR-082).
    """
    n = _n_rows(X)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    from src.models.dual_binary import stack_head_predict
    outs = [stack_head_predict(model.predict(_slice_rows(X, slice(i, i + chunk)), verbose=0))
            for i in range(0, n, chunk)]
    return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


def early_stop(*, patience: int = 10, monitor: str = "val_loss") -> Any:
    """Keras EarlyStopping shared by sequence DL plugins (LSTM, TCN)."""
    from tensorflow.keras.callbacks import EarlyStopping
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)


def train_callbacks(
    *,
    fold_i: int,
    n_folds: int,
    n_epochs: int,
    has_val: bool,
    patience: int = 10,
    monitor: str = "val_loss",
) -> list[Any] | None:
    """EarlyStopping (when val exists) + ADR-073 epoch heartbeat. ``None`` if empty."""
    cbs: list[Any] = []
    if has_val:
        cbs.append(early_stop(patience=patience, monitor=monitor))
    from src.training.job_progress import keras_epoch_callback
    cbs.extend(keras_epoch_callback(fold_i=fold_i, n_folds=n_folds, n_epochs=n_epochs))
    return cbs or None


def batched_sequence(
    X: np.ndarray,
    y: Any,
    batch_size: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Keras ``Sequence`` yielding per-batch slices of host arrays.

    Passing whole numpy arrays to ``model.fit`` makes Keras convert them into ONE
    device constant — a multi-GB fold window then fails the host→GPU copy.
    Optional *sample_weight* is yielded as the third tuple element (class balance).
    *y* may be an ndarray or a dict of ndarrays (multi-head hazard).
    """
    import tensorflow as tf

    class _Seq(tf.keras.utils.Sequence):
        def __init__(self) -> None:
            super().__init__()
            self.n = _n_rows(X)

        def __len__(self) -> int:
            return max(1, (self.n + batch_size - 1) // batch_size)

        def __getitem__(self, i: int):
            s = slice(i * batch_size, min((i + 1) * batch_size, self.n))
            xb = _slice_rows(X, s)
            yb = {k: v[s] for k, v in y.items()} if isinstance(y, dict) else y[s]
            if sample_weight is None:
                return xb, yb
            if isinstance(sample_weight, dict):
                keys = yb.keys() if isinstance(yb, dict) else sample_weight.keys()
                return xb, yb, {k: sample_weight[k][s] for k in keys}
            if isinstance(yb, dict):
                return xb, yb, {k: sample_weight[s] for k in yb}
            return xb, yb, sample_weight[s]

    return _Seq()
