"""TCN (Temporal Convolutional Network) plugin — GPU family.

Pure Keras implementation using dilated causal Conv1D + residual blocks.
No external tcn package required.

Reference: Bai et al. 2018 (Paper 5 in research corpus) — TCN consistently
outperforms LSTM on long-memory sequential tasks.

Fold builder emits 2-D ``(N, F)``; this plugin windows to ``(N', seq_len, F)``
where ``F`` is the assigned feature count (never a hardcoded n_features).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import FittedModel, ModelSpec, validate_canonical
from src.models.dl_memory import VAL_CAP, batched_sequence, predict_chunked, stride_cap, train_callbacks
from src.models.sequence_windows import make_windows, trim_warmup
from src.models.tabular_plugins import _make_sample_weight

logger = logging.getLogger("vtech.models.tcn_plugin")


class TCNPlugin:
    name: str = "tcn"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._params = dict(spec.params or {})
        self._obj_doc = objective
        self._head = str(self._params.get("head") or "multiclass")
        p = self._params
        # Placeholder width; fit() rebuilds from actual X.shape[-1] after the view.
        n_features = int(p.get("n_features") or 1)
        return _build_tcn(
            seq_len=int(p.get("seq_len", 128)), n_features=n_features, p=p,
            objective=objective,
        )

    def _rebuild(self, n_features: int) -> Any:
        p = getattr(self, "_params", {})
        return _build_tcn(
            seq_len=int(p.get("seq_len", 128)), n_features=int(n_features), p=p,
            objective=getattr(self, "_obj_doc", None),
        )

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        if objective is not None:
            self._obj_doc = objective
        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 128))
        epochs = int(p.get("epochs", 15))
        batch_size = int(p.get("batch_size", 64))
        # Same knob as XGBoost: unweighted CE on ~88% flat TBM labels is a
        # constant-flat fixed point. Default sqrt inverse-frequency.
        # Skipped when the objective uses the 3x3 expected-cost loss (ADR-072).
        from src.training.cost_matrix import cost_matrix_array
        class_balance = p.get("class_balance", "sqrt")
        dual = str(getattr(self, "_head", p.get("head") or "multiclass")) == "dual_binary"
        cost_on = (not dual) and cost_matrix_array(getattr(self, "_obj_doc", None)) is not None

        oof_frames: list[pd.DataFrame] = []
        last_scaler = None
        last_model = model
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start
        for k, fold in enumerate(train_folds):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_tr_2d = scaler.fit_transform(fold["X_train"]).astype("float32")
            X_val_2d = (scaler.transform(fold["X_val"]).astype("float32")
                        if len(fold["X_val"]) else fold["X_val"])
            actual_f = int(X_tr_2d.shape[-1])
            # Fresh weights each fold — continued training on a collapsed fold-0
            # model otherwise poisons later OOF.
            last_model = self._rebuild(actual_f)
            last_scaler = scaler

            X_tr = make_windows(X_tr_2d, seq_len)
            y_tr = trim_warmup(fold["y_train"], seq_len)
            X_val = make_windows(X_val_2d, seq_len) if len(X_val_2d) else X_val_2d
            y_val = trim_warmup(fold["y_val"], seq_len)
            if dual:
                from src.models.dual_binary import as_dual_binary_y
                y_tr = as_dual_binary_y(y_tr)
                y_val = as_dual_binary_y(y_val)
            if len(X_val):
                X_val, y_val = stride_cap(X_val, y_val, VAL_CAP)
            if cost_on:
                sw = None
            elif dual:
                from src.models.dual_binary import dual_sample_weight, y_as_head_dict
                sw = dual_sample_weight(y_tr, class_balance)
            else:
                sw = _make_sample_weight(y_tr, class_balance)
            if dual:
                from src.models.dual_binary import y_as_head_dict
                y_fit = y_as_head_dict(y_tr)
                val_y = y_as_head_dict(y_val) if len(X_val) else y_val
            else:
                y_fit = y_tr
                val_y = y_val
            val_data = batched_sequence(X_val, val_y, batch_size) if len(X_val) else None
            logger.info(
                "fit fold=%d X_tr=%s X_val=%s batch=%d class_balance=%s cost_matrix=%s "
                "dual=%s (Sequence-fed, RobustScaler)",
                k, X_tr.shape, getattr(X_val, "shape", None), batch_size,
                None if cost_on else class_balance, cost_on, dual)
            note_train_start(k, n_folds, n_epochs=epochs)
            last_model.fit(
                batched_sequence(X_tr, y_fit, batch_size, sample_weight=sw),
                validation_data=val_data,
                epochs=epochs,
                verbose=0,
                callbacks=train_callbacks(
                    fold_i=k, n_folds=n_folds, n_epochs=epochs,
                    has_val=val_data is not None,
                ),
            )
            if "X_oof" in fold:
                X_oof_2d = scaler.transform(fold["X_oof"]).astype("float32")
                X_oof = make_windows(X_oof_2d, seq_len)
                y_oof = trim_warmup(fold.get("y_oof"), seq_len)
                if dual:
                    from src.models.dual_binary import as_dual_binary_y
                    y_oof = as_dual_binary_y(y_oof) if y_oof is not None else y_oof
                ts_oof = trim_warmup(fold.get("ts_oof"), seq_len)
                oof_frames.append(_proba_to_canonical(
                    predict_chunked(last_model, X_oof),
                    ts=ts_oof,
                    y_true=y_oof,
                ))
                checkpoint_fold(k, n_folds, oof_frames[-1])
            else:
                checkpoint_fold(k, n_folds, None)
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=p),
            artifact=last_model, oof_preds=oof,
            meta={"seq_len": seq_len, "scaler": last_scaler,
                  "head": str(getattr(self, "_head", p.get("head") or "multiclass"))},
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        seq_len = int((fitted.meta or {}).get("seq_len")
                      or getattr(self, "_params", {}).get("seq_len", 128))
        scaler = (fitted.meta or {}).get("scaler")
        X2 = scaler.transform(X).astype("float32") if scaler is not None else X
        Xw = make_windows(X2, seq_len)
        proba = predict_chunked(fitted.artifact, Xw)
        df = _proba_to_canonical(proba)
        validate_canonical(df, plugin_name=self.name)
        return df


def tcn_embedding(*, seq_len: int, n_features: int, p: dict[str, Any]) -> tuple[Any, Any]:
    """Causal dilated TCN trunk ending in a Dense(64) embedding (no softmax).

    Shared by the 3-class TCN plugin and the hazard-head Encoder A (ADR-053).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    filters = p.get("filters", 64)
    kernel_size = p.get("kernel_size", 3)
    dilations = p.get("dilations", [1, 2, 4, 8])
    dropout = p.get("dropout", 0.2)

    inputs = keras.Input(shape=(int(seq_len), int(n_features)))
    x = inputs
    for d in dilations:
        skip = x
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=d, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1)(skip)
        x = layers.Add()([x, skip])
    # "gap" (default): mean over the 128-bar window — discards *when* in the
    # lookback. "last": current causal timestep only (path_tcn style).
    # LastTimestep is a registered Layer — Python-lambda Lambda cannot be
    # deserialized under Keras 3 safe_mode (ADR-059).
    pool = str(p.get("pooling") or "gap").strip().lower()
    if pool == "last":
        from src.models.keras_layers import LastTimestep
        x = LastTimestep(name="last_step")(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return inputs, x


def _build_tcn(
    *,
    seq_len: int,
    n_features: int,
    p: dict[str, Any],
    objective: dict[str, Any] | None = None,
) -> Any:
    from tensorflow import keras
    from tensorflow.keras import layers
    from src.training.cost_matrix import keras_compile_loss

    inputs, emb = tcn_embedding(seq_len=seq_len, n_features=n_features, p=p)
    if str((p or {}).get("head") or "multiclass") == "dual_binary":
        from src.models.dual_binary import compile_dual_binary_model
        return compile_dual_binary_model(inputs, emb, name="tcn")
    outputs = layers.Dense(3, activation="softmax")(emb)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=keras_compile_loss(objective),
        metrics=["accuracy"],
    )
    return model


def _proba_to_canonical(
    proba: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    from src.models.dual_binary import keras_proba_to_canonical, stack_head_predict
    arr = stack_head_predict(proba)
    df = keras_proba_to_canonical(arr, ts=ts, y_true=y_true)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        df["p_down"] = arr[:, 0].astype("float64")
        df["p_flat"] = arr[:, 1].astype("float64")
        df["p_up"] = arr[:, 2].astype("float64")
    return df



