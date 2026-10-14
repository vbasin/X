"""Dual-binary head helpers (ADR-076).

Independent P(long) / P(short) on the same TBM event. Distinct from
``xgboost_ovr`` (up-vs-rest / down-vs-rest on 3-class labels).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import CANONICAL_CONFIDENCE, CANONICAL_DIRECTION, CANONICAL_TS
from src.features.target_geometry import HEAD_LONG_COLS, HEAD_SHORT_COLS
from src.training.aux_labels import (
    LONG_AUX_MAP, SHORT_AUX_MAP, aux_loss_weight, iter_aux_map,
)
from src.training.labels import BIG_DOWN, BIG_UP

HEAD_MULTICLASS = "multiclass"
HEAD_OVR = "ovr"
HEAD_DUAL_BINARY = "dual_binary"
HEADS = (HEAD_MULTICLASS, HEAD_OVR, HEAD_DUAL_BINARY)
# Library default event cut (ADR-089). nq-50pt-econ-lg-st overrides via
# ``labels.fire`` (0.40, ADR-095). Trade iff max(p_long, p_short) ≥ fire
# AND side = argmax. Ties stay flat. 3-class dir_threshold is a different knob.
DUAL_BINARY_FIRE = 0.1


def dual_binary_fire(
    objective: dict[str, Any] | None = None,
    default: float = DUAL_BINARY_FIRE,
) -> float:
    """Event cut from ``labels.fire``, else ``kpi.tau``, else *default*."""
    obj = objective or {}
    lab = obj.get("labels") or {}
    if lab.get("fire") is not None:
        return float(lab["fire"])
    kpi = obj.get("kpi") or {}
    if kpi.get("tau") is not None:
        return float(kpi["tau"])
    return float(default)

SEQ_INPUT_NAME = "input"
LONG_CTX_NAME = "long_ctx"
SHORT_CTX_NAME = "short_ctx"


def as_dual_binary_y(y: np.ndarray) -> np.ndarray:
    """``(N, 2)`` float32 columns ``[y_long, y_short]``.

    Accepts already-2d dual-binary labels or 1d 3-class ``{0,1,2}``.
    """
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"dual-binary y expected shape (N, 2), got {arr.shape}")
        return arr.astype("float32", copy=False)
    arr = arr.reshape(-1)
    out = np.zeros((arr.size, 2), dtype="float32")
    out[:, 0] = (arr == BIG_UP).astype("float32")
    out[:, 1] = (arr == BIG_DOWN).astype("float32")
    return out


def true_direction_from_dual(y: np.ndarray) -> np.ndarray:
    """Map dual-binary y to ``up`` / ``down`` / ``flat`` (never both-1)."""
    yy = as_dual_binary_y(y)
    long = yy[:, 0] >= 0.5
    short = yy[:, 1] >= 0.5
    return np.where(long, "up", np.where(short, "down", "flat")).astype(object)


def dual_binary_to_canonical(
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    fire: float = DUAL_BINARY_FIRE,
) -> pd.DataFrame:
    """Canonical OOF plus ``p_long`` / ``p_short``.

    Default (ADR-089 / ADR-095): long when ``p_long > p_short`` and
    ``p_long ≥ fire`` (event = max of the two heads); short symmetric;
    else flat. Library default ``fire=0.1``; ``nq-50pt-econ-lg-st`` sets
    ``labels.fire: 0.40``. ``fire=0`` is argmax only.
    ``confidence = max(p_long, p_short)``.
    """
    pl = np.clip(np.asarray(p_long, dtype="float64").reshape(-1), 0.0, 1.0)
    ps = np.clip(np.asarray(p_short, dtype="float64").reshape(-1), 0.0, 1.0)
    thr = float(fire)
    up = (pl > ps) & (pl >= thr)
    down = (ps > pl) & (ps >= thr)
    direction = np.where(up, "up", np.where(down, "down", "flat")).astype(object)
    conf = np.maximum(pl, ps)
    df = pd.DataFrame({
        CANONICAL_TS: pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION: direction,
        CANONICAL_CONFIDENCE: conf,
        "p_long": pl,
        "p_short": ps,
        "p_max": conf,
        "spread": pl - ps,
    })
    if y_true is not None:
        df["direction_true"] = true_direction_from_dual(y_true)
    return df


def stack_head_predict(pred: Any) -> np.ndarray:
    """Normalize a Keras ``predict`` result to ``(N, C)``.

    Dual-binary graphs return ``{"p_long": (N,1), "p_short": (N,1)}`` (ADR-080).
    3-class softmax stays ``(N, 3)``.
    """
    if isinstance(pred, dict):
        if "p_long" in pred and "p_short" in pred:
            pl = np.asarray(pred["p_long"], dtype="float32").reshape(-1)
            ps = np.asarray(pred["p_short"], dtype="float32").reshape(-1)
            return np.column_stack([pl, ps])
        keys = list(pred)
        cols = [np.asarray(pred[k], dtype="float32").reshape(-1) for k in keys]
        return np.column_stack(cols)
    if isinstance(pred, (list, tuple)):
        arrs = [np.asarray(p, dtype="float32").reshape(-1) for p in pred]
        if len(arrs) == 1:
            a = np.asarray(pred[0])
            return a if a.ndim == 2 else a.reshape(-1, 1)
        return np.column_stack(arrs)
    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def y_as_head_dict(y: np.ndarray) -> dict[str, np.ndarray]:
    """``(N, 2)`` dual-binary y → Keras multi-output dict."""
    yy = as_dual_binary_y(y)
    return {"p_long": yy[:, 0], "p_short": yy[:, 1]}


def dual_sample_weight(
    y: np.ndarray, class_balance: Any, mask: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    """Per-head sample weights (same helper as XGB dual-binary).

    *mask* (true = loss-active) restricts class-frequency to session /
    contiguous windows so Globex overnight flats do not set the weights
    (ADR-104).
    """
    from src.models.tabular_plugins import _make_sample_weight

    yy = as_dual_binary_y(y)
    sw_l = _make_sample_weight(
        yy[:, 0].astype("int32"), class_balance, mask=mask)
    sw_s = _make_sample_weight(
        yy[:, 1].astype("int32"), class_balance, mask=mask)
    if sw_l is None and sw_s is None:
        return None
    ones = np.ones(len(yy), dtype="float64")
    return {
        "p_long": sw_l if sw_l is not None else ones,
        "p_short": sw_s if sw_s is not None else ones,
    }


def side_feature_index(cols: list[str] | tuple[str, ...] | None, side: str) -> list[int]:
    """Column indices *side* is allowed to see (drops the opposite lg/st pair)."""
    names = list(cols or [])
    drop = set(HEAD_SHORT_COLS if side == "long" else HEAD_LONG_COLS)
    return [i for i, c in enumerate(names) if c not in drop]


def _is_slow_ctx(name: str) -> bool:
    """1h/1d HTF is piecewise-constant over a 15s seq_len=60 window (ADR-104)."""
    return str(name).startswith("h1_") or str(name).startswith("d1_")


def head_feature_spec(cols: list[str] | tuple[str, ...] | None) -> dict[str, list[int]]:
    """Split catalog columns into shared sequence vs last-bar head context.

    ``lg_*`` / ``st_*`` stay head-private. ``h1_*`` / ``d1_*`` are last-bar
    context on **both** heads (not the 15s tape).
    """
    names = list(cols or [])
    long_i = [i for i, c in enumerate(names)
              if c in HEAD_LONG_COLS or _is_slow_ctx(c)]
    short_i = [i for i, c in enumerate(names)
               if c in HEAD_SHORT_COLS or _is_slow_ctx(c)]
    head = set(long_i + short_i)
    shared_i = [i for i in range(len(names)) if i not in head]
    return {"shared": shared_i, "long": long_i, "short": short_i}


def split_head_columns(
    X: np.ndarray,
    spec: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """``(X_shared, X_long_ctx, X_short_ctx)``. Ctx is None when that side has 0 cols."""
    arr = np.asarray(X)
    shared = spec.get("shared") or []
    long_i = spec.get("long") or []
    short_i = spec.get("short") or []
    if not long_i and not short_i:
        return arr, None, None
    xs = arr[:, shared] if shared else arr
    xl = arr[:, long_i] if long_i else None
    xh = arr[:, short_i] if short_i else None
    return xs, xl, xh


def pack_sequence_inputs(
    seq: np.ndarray,
    long_ctx: np.ndarray | None = None,
    short_ctx: np.ndarray | None = None,
) -> Any:
    """Keras multi-input dict, or the sequence array when there is no head context."""
    if long_ctx is None and short_ctx is None:
        return seq
    out: dict[str, np.ndarray] = {SEQ_INPUT_NAME: seq}
    if long_ctx is not None:
        out[LONG_CTX_NAME] = np.asarray(long_ctx, dtype="float32")
    if short_ctx is not None:
        out[SHORT_CTX_NAME] = np.asarray(short_ctx, dtype="float32")
    return out


def keras_input_names(model: Any) -> list[str]:
    """Input tensor names for a Keras model (Keras 3 often omits ``input_names``)."""
    names = list(getattr(model, "input_names", None) or [])
    if names:
        return [_strip_tensor_name(n) for n in names]
    inputs = getattr(model, "inputs", None)
    if inputs is None:
        return []
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    out: list[str] = []
    for t in inputs:
        n = _strip_tensor_name(getattr(t, "name", "") or "")
        if n:
            out.append(n)
    return out


def _strip_tensor_name(name: Any) -> str:
    s = str(name).split(":")[0]
    return s.split("/")[-1]


def window_pack_for_model(
    X: np.ndarray,
    seq_len: int,
    feature_cols: list[str] | tuple[str, ...] | None,
    model: Any | None = None,
) -> Any:
    """Window a scaled 2-D feature matrix for a sequence Keras ``predict``.

    Single-input graphs stay ``(N, T, F)``. Dual-binary graphs with last-bar
    ``long_ctx`` / ``short_ctx`` (ADR-082) return the same dict
    ``AttentionLSTMPlugin._window_pack`` used at train time: shared columns
    as the sequence, ``lg_*`` / ``st_*`` as the current-bar context vectors.
    """
    from src.models.sequence_windows import make_windows, trim_warmup

    names = keras_input_names(model) if model is not None else []
    want_ctx = LONG_CTX_NAME in names or SHORT_CTX_NAME in names
    arr = np.asarray(X)
    if not want_ctx:
        return make_windows(arr, seq_len)
    spec = head_feature_spec(feature_cols)
    xs, xl, xh = split_head_columns(arr, spec)
    Xw = make_windows(xs if getattr(xs, "size", 0) else arr, seq_len)
    long_c = trim_warmup(xl, seq_len) if xl is not None else None
    short_c = trim_warmup(xh, seq_len) if xh is not None else None
    packed = pack_sequence_inputs(Xw, long_c, short_c)
    if not isinstance(packed, dict):
        raise ValueError(
            f"Keras model expects inputs {names} but feature_cols "
            f"{list(feature_cols or [])} have no lg_*/st_* context columns")
    missing = [n for n in names if n not in packed]
    if missing:
        raise ValueError(
            f"Keras model expects inputs {names}; packed {list(packed)} "
            f"(missing {missing}). Need lg_* / st_* in the feature frame.")
    return {k: packed[k] for k in names if k in packed}


def compile_dual_binary_model(
    inputs: Any,
    embedding: Any,
    *,
    optimizer: Any = "adam",
    learning_rate: float | None = None,
    name: str | None = None,
    long_context: Any = None,
    short_context: Any = None,
    aux: bool = False,
    aux_columns: tuple[str, ...] | None = None,
) -> Any:
    """Shared encoder + named ``p_long`` / ``p_short`` sigmoid heads, independent BCE.

    Optional last-bar *long_context* / *short_context* tensors are concatenated
    only onto that side's private trunk (lg_* never reach the short head).
    ``aux=True`` adds Huber regression outputs for path-quality labels
    (ADR-082 / ADR-084 / ADR-086) and a ``side_rank`` output
    ``p_long / (p_long + p_short)`` trained on which side had the larger
    10m MFE (ADR-088). Canonical serving still reads only ``p_long`` /
    ``p_short``. *aux_columns* (from ``labels.aux``) subsets the parquet
    sources; omitted = the full aux set when ``aux=True``.
    Short ``forward_return_*`` heads train on ``-z(return)``.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    def _trunk(emb, ctx, tag: str):
        x = emb
        if ctx is not None:
            x = layers.Concatenate(name=f"{tag}_concat")([emb, ctx])
        if ctx is not None or aux:
            x = layers.Dense(32, activation="relu", name=f"{tag}_priv")(x)
        return x

    long_h = _trunk(embedding, long_context, "long")
    short_h = _trunk(embedding, short_context, "short")
    p_long = layers.Dense(1, activation="sigmoid", name="p_long")(long_h)
    p_short = layers.Dense(1, activation="sigmoid", name="p_short")(short_h)
    outputs: dict[str, Any] = {"p_long": p_long, "p_short": p_short}
    losses: dict[str, Any] = {
        "p_long": "binary_crossentropy",
        "p_short": "binary_crossentropy",
    }
    weights: dict[str, float] = {"p_long": 1.0, "p_short": 1.0}
    if aux:
        huber = keras.losses.Huber(delta=1.0)
        want = None if aux_columns is None else set(aux_columns)
        for k_name, src, _sign in iter_aux_map(LONG_AUX_MAP):
            if want is not None and src not in want:
                continue
            outputs[k_name] = layers.Dense(1, name=k_name)(long_h)
            losses[k_name] = huber
            weights[k_name] = aux_loss_weight(k_name)
        for k_name, src, _sign in iter_aux_map(SHORT_AUX_MAP):
            if want is not None and src not in want:
                continue
            outputs[k_name] = layers.Dense(1, name=k_name)(short_h)
            losses[k_name] = huber
            weights[k_name] = aux_loss_weight(k_name)
        from src.models.keras_layers import DualSideFraction
        from src.training.aux_labels import (
            SIDE_RANK_LOSS_WEIGHT, SIDE_RANK_NAME,
            STOP_FIRST_LONG, STOP_FIRST_SHORT, stop_first_enabled,
        )
        outputs[SIDE_RANK_NAME] = DualSideFraction(name=SIDE_RANK_NAME)(
            [p_long, p_short])
        losses[SIDE_RANK_NAME] = "binary_crossentropy"
        weights[SIDE_RANK_NAME] = SIDE_RANK_LOSS_WEIGHT
        if stop_first_enabled(aux_columns):
            outputs[STOP_FIRST_LONG] = layers.Dense(
                1, activation="sigmoid", name=STOP_FIRST_LONG)(long_h)
            outputs[STOP_FIRST_SHORT] = layers.Dense(
                1, activation="sigmoid", name=STOP_FIRST_SHORT)(short_h)
            losses[STOP_FIRST_LONG] = "binary_crossentropy"
            losses[STOP_FIRST_SHORT] = "binary_crossentropy"
            weights[STOP_FIRST_LONG] = aux_loss_weight(STOP_FIRST_LONG)
            weights[STOP_FIRST_SHORT] = aux_loss_weight(STOP_FIRST_SHORT)
    model = keras.Model(inputs, outputs, name=name)
    if learning_rate is not None and (optimizer == "adam" or optimizer is None):
        opt: Any = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
    model.compile(optimizer=opt, loss=losses, loss_weights=weights)
    return model


def keras_proba_to_canonical(
    proba: Any,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    fire: float = DUAL_BINARY_FIRE,
) -> pd.DataFrame:
    """``predict`` dict / ``(N, 2)`` dual-binary or ``(N, 3)`` softmax → canonical OOF."""
    arr = stack_head_predict(proba)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return dual_binary_to_canonical(
            arr[:, 0], arr[:, 1], ts=ts, y_true=y_true, fire=fire)
    idx = arr.argmax(axis=1)
    conf = arr.max(axis=1).astype("float64")
    labels = {0: "down", 1: "flat", 2: "up"}
    df = pd.DataFrame({
        CANONICAL_TS: pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION: [labels.get(int(i), "flat") for i in idx],
        CANONICAL_CONFIDENCE: conf,
    })
    df[CANONICAL_DIRECTION] = df[CANONICAL_DIRECTION].astype("object")
    if y_true is not None:
        yt = np.asarray(y_true)
        if yt.ndim == 2:
            df["direction_true"] = true_direction_from_dual(yt)
        else:
            df["direction_true"] = [labels.get(int(y), "flat") for y in yt]
    return df


def is_dual_binary_frame(df: pd.DataFrame | None) -> bool:
    if df is None or not hasattr(df, "columns"):
        return False
    return "p_long" in df.columns and "p_short" in df.columns


def head_binary_report(
    y: np.ndarray,
    p: np.ndarray,
    *,
    fire: float = 0.5,
) -> dict[str, Any]:
    """Per-head binary metrics on independent P(event)."""
    from sklearn.metrics import (
        average_precision_score, confusion_matrix, roc_auc_score,
    )

    yt = (np.asarray(y).reshape(-1) >= 0.5).astype("int32")
    pr = np.clip(np.asarray(p, dtype="float64").reshape(-1), 0.0, 1.0)
    pred = (pr >= float(fire)).astype("int32")
    n_pos = int(yt.sum())
    n = int(yt.size)
    out: dict[str, Any] = {
        "n": n,
        "n_pos": n_pos,
        "n_pred_pos": int(pred.sum()),
        "p_mean": float(pr.mean()) if n else None,
        "p_max": float(pr.max()) if n else None,
        "p_min": float(pr.min()) if n else None,
        "fire": float(fire),
        "confusion_at_fire": {
            "labels": ["neg", "pos"],
            "rows_true_cols_pred": confusion_matrix(
                yt, pred, labels=[0, 1]).tolist() if n else [[0, 0], [0, 0]],
        },
    }
    if n and n_pos > 0 and n_pos < n:
        out["roc_auc"] = float(roc_auc_score(yt, pr))
        out["average_precision"] = float(average_precision_score(yt, pr))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out


def dual_binary_head_block(df: pd.DataFrame, *, fire: float = 0.5) -> dict[str, Any]:
    """Top-level diagnostics block for a dual-binary OOF frame."""
    y_long = (df["direction_true"].astype(str) == "up").to_numpy()
    y_short = (df["direction_true"].astype(str) == "down").to_numpy()
    return {
        "head": "dual_binary",
        "labels": ["long", "short"],
        "heads": {
            "long": head_binary_report(y_long, df["p_long"].to_numpy(), fire=fire),
            "short": head_binary_report(y_short, df["p_short"].to_numpy(), fire=fire),
        },
        "canonical_labels": ["down", "flat", "up"],
    }
