"""Walk-forward fit/predict for encoder + shared hazard heads (ADR-053)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.base import FittedModel, ModelSpec, validate_canonical
from src.models.dl_memory import VAL_CAP, batched_sequence, train_callbacks
from src.models.heads import (
    MIN_INFERENCE_R,
    apply_inference_r_gate,
    compile_hazard_model,
    serving_from_logits,
)
from src.models.sequence_windows import make_windows, trim_warmup
from src.models.tabular_plugins import _make_sample_weight

logger = logging.getLogger("vtech.models.hazard_fit")

RebuildFn = Callable[[int, list[str] | None], Any]


def _as_dict_outputs(raw: Any) -> dict[str, np.ndarray]:
    if isinstance(raw, dict):
        return {k: np.asarray(v) for k, v in raw.items()}
    raise TypeError(f"hazard model.predict returned {type(raw)}, expected dict")


def predict_hazard_chunked(model: Any, X: np.ndarray, chunk: int = 65536) -> dict[str, np.ndarray]:
    if len(X) == 0:
        return {}
    parts: list[dict[str, np.ndarray]] = []
    for i in range(0, len(X), chunk):
        parts.append(_as_dict_outputs(model.predict(X[i:i + chunk], verbose=0)))
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _pack_y(fold: dict[str, Any], split: str, n: int) -> dict[str, np.ndarray] | None:
    up = fold.get(f"y_up_{split}")
    down = fold.get(f"y_down_{split}")
    if up is None or down is None:
        return None
    if len(up) != n or len(down) != n:
        raise ValueError(f"y_up_{split}/y_down_{split} length mismatch vs X_{split}")
    out: dict[str, np.ndarray] = {"y_up": np.asarray(up), "y_down": np.asarray(down)}
    q = fold.get(f"y_quantile_{split}")
    if q is not None and len(q) == n:
        out["y_quantile"] = np.asarray(q, dtype="float32")
    return out


def fit_hazard_folds(
    *,
    rebuild: RebuildFn,
    train_folds: list[dict[str, np.ndarray]],
    params: dict[str, Any],
    plugin_name: str,
    spec: ModelSpec,
) -> FittedModel:
    """Per-fold RobustScaler + rebuild + Sequence-fed hazard training."""
    from sklearn.preprocessing import RobustScaler

    seq_len = int(params.get("seq_len", 128))
    epochs = int(params.get("epochs", 15))
    batch_size = int(params.get("batch_size", 64))
    class_balance = params.get("class_balance", "sqrt")
    include_quantile = bool(params.get("include_quantile", True))
    min_r = float(params.get("inference_r_min", MIN_INFERENCE_R) or 0.0)

    oof_frames: list[pd.DataFrame] = []
    last_scaler = None
    last_model: Any = None
    last_cols: list[str] | None = None
    n_folds = len(train_folds)
    from src.training.job_progress import checkpoint_fold, note_train_start

    for k, fold in enumerate(train_folds):
        scaler = RobustScaler()
        X_tr_2d = scaler.fit_transform(fold["X_train"]).astype("float32")
        X_val_2d = (scaler.transform(fold["X_val"]).astype("float32")
                    if len(fold["X_val"]) else fold["X_val"])
        actual_f = int(X_tr_2d.shape[-1])
        cols = list(fold.get("feature_cols") or [])
        last_cols = cols
        last_model = rebuild(actual_f, cols)
        last_scaler = scaler

        y_tr = _pack_y(fold, "train", len(fold["y_train"]))
        if y_tr is None:
            raise ValueError(
                f"{plugin_name}: fold is missing y_up_train/y_down_train — "
                "use labels.scheme=path_vol_normalized so fold_cache joins parquet_pathlabels"
            )
        if not include_quantile:
            y_tr.pop("y_quantile", None)

        X_tr = make_windows(X_tr_2d, seq_len)
        y_tr_w = {name: trim_warmup(arr, seq_len) for name, arr in y_tr.items()}
        y_sparse = trim_warmup(fold["y_train"], seq_len)
        sw = _make_sample_weight(y_sparse, class_balance)

        val_data = None
        if len(X_val_2d):
            y_val = _pack_y(fold, "val", len(fold["y_val"]))
            if y_val is not None:
                if not include_quantile:
                    y_val.pop("y_quantile", None)
                X_val = make_windows(X_val_2d, seq_len)
                y_val_w = {name: trim_warmup(arr, seq_len) for name, arr in y_val.items()}
                # stride_cap is (X, y) for a single y; cap X and each y head together
                if len(X_val) > VAL_CAP:
                    idx = np.linspace(0, len(X_val) - 1, VAL_CAP).astype(int)
                    X_val = X_val[idx]
                    y_val_w = {n: a[idx] for n, a in y_val_w.items()}
                val_data = batched_sequence(X_val, y_val_w, batch_size)

        logger.info(
            "hazard fit plugin=%s fold=%d X_tr=%s batch=%d class_balance=%s quantile=%s",
            plugin_name, k, X_tr.shape, batch_size, class_balance, include_quantile,
        )
        note_train_start(k, n_folds, n_epochs=epochs)
        last_model.fit(
            batched_sequence(X_tr, y_tr_w, batch_size, sample_weight=sw),
            validation_data=val_data,
            epochs=epochs,
            verbose=0,
            callbacks=train_callbacks(
                fold_i=k, n_folds=n_folds, n_epochs=epochs,
                has_val=val_data is not None,
            ),
        )
        if "X_oof" in fold and last_model is not None:
            X_oof_2d = scaler.transform(fold["X_oof"]).astype("float32")
            X_oof = make_windows(X_oof_2d, seq_len)
            ts_oof = trim_warmup(fold.get("ts_oof"), seq_len)
            outs = predict_hazard_chunked(last_model, X_oof)
            df = serving_from_logits(
                outs["y_up"], outs["y_down"], ts=ts_oof, plugin_name=plugin_name,
            )
            r_oof = fold.get("r_value_oof")
            if r_oof is not None:
                r_oof = trim_warmup(np.asarray(r_oof), seq_len)
            df = apply_inference_r_gate(df, r_oof, min_r=min_r)
            validate_canonical(df, plugin_name=plugin_name)
            oof_frames.append(df)
            checkpoint_fold(k, n_folds, df)
        else:
            checkpoint_fold(k, n_folds, None)

    oof = pd.concat(oof_frames) if oof_frames else None
    return FittedModel(
        spec=spec,
        artifact=last_model,
        oof_preds=oof,
        meta={
            "seq_len": seq_len,
            "scaler": last_scaler,
            "feature_cols": last_cols,
            "inference_r_min": min_r,
        },
    )


def predict_hazard(
    fitted: FittedModel,
    X: np.ndarray,
    *,
    plugin_name: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    seq_len = int((fitted.meta or {}).get("seq_len") or params.get("seq_len", 128))
    scaler = (fitted.meta or {}).get("scaler")
    X2 = scaler.transform(X).astype("float32") if scaler is not None else X
    Xw = make_windows(X2, seq_len)
    outs = predict_hazard_chunked(fitted.artifact, Xw)
    df = serving_from_logits(outs["y_up"], outs["y_down"], plugin_name=plugin_name)
    validate_canonical(df, plugin_name=plugin_name)
    return df


def compile_on_embedding(
    inputs: Any,
    embedding: Any,
    *,
    include_quantile: bool = True,
) -> Any:
    return compile_hazard_model(inputs, embedding, include_quantile=include_quantile)
