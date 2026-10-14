"""Shared heads on an encoder embedding ([ADR-053](../../docs/decisions/ADR-053-vol-normalized-path-labels-and-encoder-ablation.md)).

The encoder is a black box that emits a vector per bar (or a pooled vector). These
heads turn that vector into the dense path-label targets and, at inference, into the
serving contract ``{direction, confidence}``.

**Competing-risks hazard (canonical first head).** Per side the joint class is
``{no-touch} ∪ (target × bucket) ∪ (stop × bucket)``. With ``n_buckets=20`` that is
41 classes. Cross-entropy on that joint is defined on *every* bar, including
``usable=False`` and sparse-flat rows — that is what closes the sample-count gap
the 3-class ``{0,1,2}`` label cannot.

Layout (class id)::

    0                         no-touch / censored
    1 .. n_buckets            target in bucket ``k = id - 1``
    n_buckets+1 .. 2*n_buckets  stop in bucket ``k = id - 1 - n_buckets``

Censored parquet rows store ``bucket = n_buckets`` as a sentinel; encoding maps
those to class 0 and never teaches a real bucket.

**Quantile head.** Pinball at ``{0.1, 0.5, 0.9}`` on
``(mfe_up_r, mae_up_r, mfe_down_r, mae_down_r)``. Output is flattened target-major
(12 units). SSL (masked-bar reconstruction) is not in this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.base import (
    CANONICAL_CONFIDENCE,
    CANONICAL_DIRECTION,
    CANONICAL_TS,
    validate_canonical,
)
from src.training.path_labels import CENSORED, STOP, TARGET

DEFAULT_N_BUCKETS = 20
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
EXCURSION_COLS: tuple[str, ...] = ("mfe_up_r", "mae_up_r", "mfe_down_r", "mae_down_r")
# Live +50/−6 is the 2026 geometry: fire only when 0.743 × R ≥ 50 (ADR-053).
MIN_INFERENCE_R = 67.3

NO_TOUCH_CLASS = 0

# Discrete-time competing-risks per bucket (recovered spec).
# 0 = neither-yet, 1 = target-first this bucket, 2 = stop-first this bucket.
CR_NEITHER, CR_TARGET, CR_STOP = 0, 1, 2
N_CR_CLASSES = 3


def n_hazard_classes(n_buckets: int = DEFAULT_N_BUCKETS) -> int:
    """``1 + 2 * n_buckets`` — no-touch plus target/stop per bucket."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    return 1 + 2 * n_buckets


def n_quantile_outputs(n_quantiles: int = len(DEFAULT_QUANTILES)) -> int:
    return len(EXCURSION_COLS) * int(n_quantiles)


def encode_cr_sequence(
    event: Any,
    bucket: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket competing-risks labels and a right-censor mask.

    Returns ``y`` ``(N, n_buckets)`` in ``{neither, target, stop}`` and
    ``mask`` ``(N, n_buckets)`` (0 after the observed exit). A fully censored
    episode is ``neither`` on every bucket with mask 1.
    """
    event_arr = np.asarray(event)
    bucket_arr = np.asarray(bucket)
    n = event_arr.shape[0]
    t = np.arange(n_buckets, dtype=np.int32)[None, :]
    censored = event_arr == CENSORED
    k = np.where(
        censored,
        n_buckets,
        np.clip(bucket_arr, 0, n_buckets - 1),
    ).astype(np.int32)[:, None]
    y = np.zeros((n, n_buckets), dtype=np.int32)
    at = (t == k) & (~censored[:, None])
    y = np.where(at & (event_arr[:, None] == TARGET), CR_TARGET, y)
    y = np.where(at & (event_arr[:, None] == STOP), CR_STOP, y)
    mask = (t <= k).astype(np.float32)
    mask = np.where(censored[:, None], 1.0, mask)
    # t > k already 0 from t <= k when not censored (k < n_buckets)
    return y, mask


def p_first_from_cr_probs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate per-bucket softmax ``(..., K, 3)`` → target-first / stop-first / none.

    Do **not** floor ``P(neither)`` away from 0: that leaks survival through later
    buckets and can make ``P(target-first) > 1`` (failed the serving contract after
    a full A10 fit). Zero neither-mass is a real peaked softmax; later terms must
    vanish. Non-finite / denormal inputs are projected onto the 3-simplex first.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    p = np.where(np.isfinite(p), p, 0.0)
    mass = p.sum(axis=-1, keepdims=True)
    p = np.divide(p, mass, out=np.full_like(p, 1.0 / N_CR_CLASSES), where=mass > 0)
    p_n = p[..., CR_NEITHER]
    p_t = p[..., CR_TARGET]
    p_s = p[..., CR_STOP]
    with np.errstate(divide="ignore"):
        log_n = np.where(p_n > 0.0, np.log(np.clip(p_n, 1e-300, 1.0)), -np.inf)
    csum = np.cumsum(log_n, axis=-1)
    zeros = np.zeros(csum.shape[:-1] + (1,), dtype=csum.dtype)
    log_surv = np.concatenate([zeros, csum[..., :-1]], axis=-1)
    surv = np.exp(log_surv)
    p_target = np.nansum(surv * p_t, axis=-1)
    p_stop = np.nansum(surv * p_s, axis=-1)
    p_none = np.exp(csum[..., -1])
    p_none = np.where(np.isfinite(p_none), p_none, 0.0)
    tot = p_target + p_stop + p_none
    tot = np.where(tot > 0.0, tot, 1.0)
    p_target = np.clip(p_target / tot, 0.0, 1.0)
    p_stop = np.clip(p_stop / tot, 0.0, 1.0)
    p_none = np.clip(p_none / tot, 0.0, 1.0)
    return p_target, p_stop, p_none


def serving_from_cr_logits(
    logits_up: np.ndarray,
    logits_down: np.ndarray,
    ts: np.ndarray | None = None,
    *,
    plugin_name: str = "path_tcn",
) -> pd.DataFrame:
    """Softmax over {neither, target, stop} per bucket, then integrate."""
    lu = np.asarray(logits_up, dtype=np.float64)
    ld = np.asarray(logits_down, dtype=np.float64)
    p_up = _softmax(lu, axis=-1)
    p_dn = _softmax(ld, axis=-1)
    tu, su, cu = p_first_from_cr_probs(p_up)
    td, sd, cd = p_first_from_cr_probs(p_dn)
    df = serving_from_masses(
        HazardMasses(tu, su, cu, td, sd, cd),
        ts=ts, plugin_name=plugin_name,
    )
    if lu.size:
        bad = (
            ~np.isfinite(lu).reshape(lu.shape[0], -1).all(axis=1)
            | ~np.isfinite(ld).reshape(ld.shape[0], -1).all(axis=1)
        )
        if bad.any():
            df.loc[bad, CANONICAL_DIRECTION] = "flat"
            df.loc[bad, CANONICAL_CONFIDENCE] = 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Joint-class codec
# ══════════════════════════════════════════════════════════════════════════════

def encode_hazard_class(
    event: Any,
    bucket: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> np.ndarray:
    """Map ``(event, bucket)`` from a path-label row to the joint class id.

    ``event`` is ``0`` censored / ``1`` target / ``2`` stop. Touches must have
    ``bucket ∈ [0, n_buckets)``; censored rows may carry the parquet sentinel
    ``bucket = n_buckets`` and are encoded as class 0 regardless.
    """
    event_arr = np.asarray(event)
    bucket_arr = np.asarray(bucket)
    if event_arr.shape != bucket_arr.shape:
        raise ValueError("event and bucket must have the same shape")
    touched = event_arr != CENSORED
    if np.any(touched & ((bucket_arr < 0) | (bucket_arr >= n_buckets))):
        raise ValueError(
            "touched rows must have bucket in [0, n_buckets); "
            f"n_buckets={n_buckets}"
        )
    out = np.zeros(event_arr.shape, dtype=np.int32)
    target = event_arr == TARGET
    stop = event_arr == STOP
    out[target] = 1 + bucket_arr[target].astype(np.int32)
    out[stop] = 1 + n_buckets + bucket_arr[stop].astype(np.int32)
    return out


def decode_hazard_class(
    cls: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_hazard_class`. Censored → ``bucket = n_buckets``."""
    ids = np.asarray(cls, dtype=np.int32)
    n_cls = n_hazard_classes(n_buckets)
    if np.any((ids < 0) | (ids >= n_cls)):
        raise ValueError(f"class id out of range [0, {n_cls})")
    event = np.full(ids.shape, CENSORED, dtype=np.int8)
    bucket = np.full(ids.shape, n_buckets, dtype=np.int16)
    target = (ids >= 1) & (ids <= n_buckets)
    stop = (ids > n_buckets) & (ids < n_cls)
    event[target] = TARGET
    event[stop] = STOP
    bucket[target] = (ids[target] - 1).astype(np.int16)
    bucket[stop] = (ids[stop] - 1 - n_buckets).astype(np.int16)
    return event, bucket


def encode_pathlabel_frame(
    frame: pd.DataFrame,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> dict[str, np.ndarray]:
    """Pack a ``label_session`` / parquet frame into training arrays.

    Returns ``y_up``, ``y_down`` (joint class ids) and ``y_quantile``
    ``(N, 4)`` in :data:`EXCURSION_COLS` order. ``ts`` is passed through.
    """
    missing = [c for c in ("event_up", "bucket_up", "event_down", "bucket_down")
               if c not in frame.columns]
    if missing:
        raise KeyError(f"path-label frame missing {missing}")
    out: dict[str, np.ndarray] = {
        "y_up": encode_hazard_class(frame["event_up"], frame["bucket_up"],
                                    n_buckets=n_buckets),
        "y_down": encode_hazard_class(frame["event_down"], frame["bucket_down"],
                                     n_buckets=n_buckets),
    }
    if "ts" in frame.columns:
        out["ts"] = np.asarray(frame["ts"])
    if all(c in frame.columns for c in EXCURSION_COLS):
        out["y_quantile"] = np.column_stack(
            [frame[c].to_numpy(dtype="float64") for c in EXCURSION_COLS]
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Inference reduction
# ══════════════════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = np.where(np.isfinite(z), z, 0.0)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -80.0, 80.0))
    s = e.sum(axis=axis, keepdims=True)
    return np.divide(e, s, out=np.full_like(e, 1.0 / z.shape[axis]), where=s > 0)


def integrate_hazard_probs(
    probs: np.ndarray,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum a softmax over buckets → ``(p_target, p_stop, p_censored)``.

    ``probs`` has last-dim ``n_hazard_classes``. The three masses sum to 1.
    """
    n_cls = n_hazard_classes(n_buckets)
    if probs.shape[-1] != n_cls:
        raise ValueError(f"expected last dim {n_cls}, got {probs.shape[-1]}")
    p_censored = probs[..., NO_TOUCH_CLASS]
    p_target = probs[..., 1:1 + n_buckets].sum(axis=-1)
    p_stop = probs[..., 1 + n_buckets:].sum(axis=-1)
    return p_target, p_stop, p_censored


@dataclass(frozen=True)
class HazardMasses:
    """Integrated per-side barrier probabilities (each triple sums to 1)."""

    p_target_up: np.ndarray
    p_stop_up: np.ndarray
    p_censored_up: np.ndarray
    p_target_down: np.ndarray
    p_stop_down: np.ndarray
    p_censored_down: np.ndarray


def masses_from_logits(
    logits_up: np.ndarray,
    logits_down: np.ndarray,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> HazardMasses:
    """Softmax each side, then integrate over buckets."""
    p_tu, p_su, p_cu = integrate_hazard_probs(
        _softmax(np.asarray(logits_up)), n_buckets=n_buckets)
    p_td, p_sd, p_cd = integrate_hazard_probs(
        _softmax(np.asarray(logits_down)), n_buckets=n_buckets)
    return HazardMasses(p_tu, p_su, p_cu, p_td, p_sd, p_cd)


def serving_from_masses(
    masses: HazardMasses,
    ts: np.ndarray | None = None,
    *,
    plugin_name: str = "hazard",
) -> pd.DataFrame:
    """``direction = sign(P(target_up) − P(target_down))``; confidence is the max.

    Ties (equal target mass) are ``flat``. No serving-contract change versus
    existing plugins: columns ``ts``, ``direction``, ``confidence``. Extra
    masses are attached for diagnostics; they are not required at serve time.
    """
    p_up = np.asarray(masses.p_target_up, dtype="float64")
    p_down = np.asarray(masses.p_target_down, dtype="float64")
    p_up = np.where(np.isfinite(p_up), np.clip(p_up, 0.0, 1.0), 0.0)
    p_down = np.where(np.isfinite(p_down), np.clip(p_down, 0.0, 1.0), 0.0)
    direction = np.where(
        p_up > p_down, "up",
        np.where(p_down > p_up, "down", "flat"),
    )
    confidence = np.clip(np.maximum(p_up, p_down), 0.0, 1.0)
    n = p_up.shape[0]
    if ts is None:
        ts_col = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
    else:
        ts_col = pd.to_datetime(ts, utc=True)
    df = pd.DataFrame({
        CANONICAL_TS: ts_col,
        CANONICAL_DIRECTION: direction,
        CANONICAL_CONFIDENCE: confidence,
        "p_target_up": p_up,
        "p_target_down": p_down,
        "p_stop_up": np.asarray(masses.p_stop_up, dtype="float64"),
        "p_stop_down": np.asarray(masses.p_stop_down, dtype="float64"),
        "p_censored_up": np.asarray(masses.p_censored_up, dtype="float64"),
        "p_censored_down": np.asarray(masses.p_censored_down, dtype="float64"),
    })
    validate_canonical(df, plugin_name=plugin_name)
    return df


def serving_from_logits(
    logits_up: np.ndarray,
    logits_down: np.ndarray,
    ts: np.ndarray | None = None,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
    plugin_name: str = "hazard",
) -> pd.DataFrame:
    return serving_from_masses(
        masses_from_logits(logits_up, logits_down, n_buckets=n_buckets),
        ts=ts, plugin_name=plugin_name,
    )


def apply_inference_r_gate(
    frame: pd.DataFrame,
    r_value: np.ndarray | None,
    *,
    min_r: float = MIN_INFERENCE_R,
) -> pd.DataFrame:
    """Zero out bars whose causal R is below the live +50/−6 geometry.

    Fail-closed on non-finite R. ``r_value is None`` leaves *frame* unchanged
    (callers that do not have a yardstick yet — e.g. ``plugin.predict(X)``).
    """
    if r_value is None or min_r <= 0:
        return frame
    r = np.asarray(r_value, dtype="float64")
    if len(r) != len(frame):
        raise ValueError(f"r_value length {len(r)} != frame length {len(frame)}")
    out = frame.copy()
    blocked = ~np.isfinite(r) | (r < float(min_r))
    if blocked.any():
        out.loc[blocked, CANONICAL_DIRECTION] = "flat"
        out.loc[blocked, CANONICAL_CONFIDENCE] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Pinball (quantile) loss — numpy, used as the spec / test oracle
# ══════════════════════════════════════════════════════════════════════════════

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> float:
    """Mean pinball loss.

    ``y_true`` is ``(..., T)``. ``y_pred`` is either the same shape as
    ``y_true`` broadcast against ``quantiles``, or ``(..., T, Q)``, or flat
    ``(..., T*Q)`` in target-major order matching :func:`attach_quantile_head`.
    """
    y_true_arr = np.asarray(y_true, dtype="float64")
    y_pred_arr = np.asarray(y_pred, dtype="float64")
    q = np.asarray(quantiles, dtype="float64")
    t = y_true_arr.shape[-1]
    nq = q.shape[0]
    if y_pred_arr.shape == y_true_arr.shape:
        y_pred_arr = y_pred_arr[..., None]
    elif y_pred_arr.shape[:-1] == y_true_arr.shape and y_pred_arr.shape[-1] == nq:
        pass
    elif y_pred_arr.shape[:-1] == y_true_arr.shape[:-1] and y_pred_arr.shape[-1] == t * nq:
        y_pred_arr = y_pred_arr.reshape(*y_true_arr.shape, nq)
    else:
        raise ValueError(
            f"y_pred shape {y_pred_arr.shape} incompatible with "
            f"y_true {y_true_arr.shape} and {nq} quantiles"
        )
    err = y_true_arr[..., None] - y_pred_arr
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


# ══════════════════════════════════════════════════════════════════════════════
# Keras attach points — imported lazily so numpy tests do not load TF
# ══════════════════════════════════════════════════════════════════════════════

def attach_hazard_heads(
    embedding: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
    name: str = "hazard",
) -> tuple[Any, Any]:
    """Linear joint-class logits on the last dim of *embedding*.

    Returns ``(logits_up, logits_down)``, each of width
    :func:`n_hazard_classes`. Works on ``(batch, d)`` or ``(batch, time, d)``.
    """
    from tensorflow.keras import layers

    n_cls = n_hazard_classes(n_buckets)
    logits_up = layers.Dense(n_cls, name=f"{name}_up_logits")(embedding)
    logits_down = layers.Dense(n_cls, name=f"{name}_down_logits")(embedding)
    return logits_up, logits_down


def attach_quantile_head(
    embedding: Any,
    *,
    n_quantiles: int = len(DEFAULT_QUANTILES),
    name: str = "quantile",
) -> Any:
    """Flattened pinball outputs, target-major, width ``4 * n_quantiles``."""
    from tensorflow.keras import layers

    return layers.Dense(n_quantile_outputs(n_quantiles), name=f"{name}_raw")(embedding)


def compile_hazard_model(
    inputs: Any,
    embedding: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
    include_quantile: bool = False,
    n_quantiles: int = len(DEFAULT_QUANTILES),
    learning_rate: float = 1e-3,
) -> Any:
    """Wrap *embedding* with the shared heads and compile.

    Outputs are a dict so ``fit({...}, {...})`` matches
    :func:`encode_pathlabel_frame` keys ``y_up`` / ``y_down`` / ``y_quantile``.
    Hazard uses logits + ``from_logits=True`` sparse CE. Quantile uses a Keras
    pinball that matches :func:`pinball_loss`.
    """
    from tensorflow import keras

    logits_up, logits_down = attach_hazard_heads(embedding, n_buckets=n_buckets)
    outputs: dict[str, Any] = {
        "y_up": logits_up,
        "y_down": logits_down,
    }
    losses: dict[str, Any] = {
        "y_up": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "y_down": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    }
    if include_quantile:
        outputs["y_quantile"] = attach_quantile_head(
            embedding, n_quantiles=n_quantiles)
        if n_quantiles != len(DEFAULT_QUANTILES):
            raise ValueError(
                f"n_quantiles must be {len(DEFAULT_QUANTILES)} (got {n_quantiles})")
        losses["y_quantile"] = _PinballLoss(quantiles=DEFAULT_QUANTILES)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=losses)
    return model


def attach_cr_heads(
    embedding: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
    name: str = "hazard",
) -> tuple[Any, Any]:
    """Per-bucket 3-way logits ``(n_buckets, 3)`` for each side."""
    from tensorflow.keras import layers

    n_out = int(n_buckets) * N_CR_CLASSES
    raw_up = layers.Dense(n_out, name=f"{name}_up_raw")(embedding)
    raw_dn = layers.Dense(n_out, name=f"{name}_down_raw")(embedding)
    logits_up = layers.Reshape((int(n_buckets), N_CR_CLASSES), name="y_up")(raw_up)
    logits_down = layers.Reshape((int(n_buckets), N_CR_CLASSES), name="y_down")(raw_dn)
    return logits_up, logits_down


def compile_cr_hazard_model(
    inputs: Any,
    embedding: Any,
    *,
    n_buckets: int = DEFAULT_N_BUCKETS,
    include_quantile: bool = True,
    n_quantiles: int = len(DEFAULT_QUANTILES),
    learning_rate: float = 1e-3,
) -> Any:
    """Competing-risks NLL (per-bucket 3-way) + optional pinball."""
    from tensorflow import keras

    logits_up, logits_down = attach_cr_heads(embedding, n_buckets=n_buckets)
    outputs: dict[str, Any] = {"y_up": logits_up, "y_down": logits_down}
    ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    losses: dict[str, Any] = {"y_up": ce, "y_down": ce}
    if include_quantile:
        outputs["y_quantile"] = attach_quantile_head(
            embedding, n_quantiles=n_quantiles)
        losses["y_quantile"] = _PinballLoss(quantiles=DEFAULT_QUANTILES)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=losses)
    return model


class _PinballLoss:
    """Keras loss: mean pinball, ``y_pred`` flat target-major."""

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.quantiles = tuple(float(q) for q in quantiles)

    def get_config(self) -> dict[str, Any]:
        return {"quantiles": list(self.quantiles)}

    def __call__(self, y_true: Any, y_pred: Any) -> Any:
        import tensorflow as tf

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nq = len(self.quantiles)
        pred = tf.reshape(y_pred, tf.concat([tf.shape(y_true), [nq]], 0))
        q = tf.constant(self.quantiles, dtype=y_true.dtype)
        err = tf.expand_dims(y_true, -1) - pred
        loss = tf.maximum(q * err, (q - 1.0) * err)
        return tf.reduce_mean(loss)
