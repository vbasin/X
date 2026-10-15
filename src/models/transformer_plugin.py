"""Transformer (encoder) plugin — GPU family.

Encoder-only Transformer using Keras MultiHeadAttention + positional encoding.
Avoids decoder-only/pure-transformer architectures flagged as underperforming
on LOB tasks in Papers 7, 9; encoder-only with global-average-pooling is safer.

Architecture:
    Input → Positional encoding → N × (MHA + Add&Norm + FFN + Add&Norm)
          → GlobalAveragePooling1D → Dense trunk → softmax(3) or dual-binary heads
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

logger = logging.getLogger("vtech.models.transformer_plugin")


class TransformerPlugin:
    name: str = "transformer"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._params = dict(spec.params or {})
        self._validate_params()
        self._obj_doc = objective
        self._head = str(self._params.get("head") or "multiclass")
        n_features = int(self._params.get("n_features") or 1)
        return self._rebuild(n_features)

    def _validate_params(self) -> None:
        p = self._params
        for key, default in (("d_model", 64), ("n_heads", 4), ("n_layers", 2),
                             ("seq_len", 128), ("epochs", 15), ("batch_size", 64),
                             ("n_features", 1)):
            value = p.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"transformer {key} must be a positive integer")
        if p.get("d_model", 64) % p.get("n_heads", 4):
            raise ValueError("transformer d_model must be divisible by n_heads")
        dropout = float(p.get("dropout", 0.1))
        if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("transformer dropout must be finite and in [0, 1)")
        learning_rate = float(p.get("learning_rate", 1e-4))
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("transformer learning_rate must be positive and finite")
        if str(p.get("head") or "multiclass") not in {"multiclass", "dual_binary"}:
            raise ValueError("transformer head must be multiclass or dual_binary")

    def _rebuild(self, n_features: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers
        from src.training.cost_matrix import keras_compile_loss

        p = getattr(self, "_params", {})
        d_model = p.get("d_model", 64)
        n_heads = p.get("n_heads", 4)
        n_layers = p.get("n_layers", 2)
        dropout = p.get("dropout", 0.1)
        seq_len = p.get("seq_len", 128)
        learning_rate = float(p.get("learning_rate", 1e-4))
        ff_dim = d_model * 4
        dual = str(getattr(self, "_head", p.get("head") or "multiclass")) == "dual_binary"

        inputs = keras.Input(shape=(seq_len, int(n_features)))
        x = layers.Dense(d_model)(inputs)
        # Symbolic indices keep the Embedding in the functional graph. Calling
        # it on an eager tf.range previously froze an untracked random tensor.
        positions = keras.ops.cast(
            keras.ops.cumsum(keras.ops.ones_like(inputs[:, :, 0]), axis=1) - 1,
            "int32",
        )
        position_values = layers.Embedding(
            input_dim=seq_len, output_dim=d_model, name="position_embedding",
        )(positions)
        x = layers.Add()([x, position_values])
        for _ in range(n_layers):
            attn = layers.MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
            )(x, x)
            x = layers.LayerNormalization()(layers.Add()([x, attn]))
            ffn = layers.Dense(ff_dim, activation="relu")(x)
            ffn = layers.Dropout(dropout)(ffn)
            ffn = layers.Dense(d_model)(ffn)
            x = layers.LayerNormalization()(layers.Add()([x, ffn]))
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout)(x)
        if dual:
            from src.models.dual_binary import compile_dual_binary_model
            return compile_dual_binary_model(
                inputs, x, learning_rate=learning_rate, name="transformer")
        outputs = layers.Dense(3, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras_compile_loss(getattr(self, "_obj_doc", None)),
            metrics=["accuracy"],
        )
        return model

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
        callbacks: list[Any] | None = None,
    ) -> FittedModel:
        """Fit independent folds; optional callbacks provide deadline/tracking hooks."""
        if not train_folds:
            raise ValueError("transformer requires at least one training fold")
        if objective is not None:
            self._obj_doc = objective
        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 128))
        epochs = int(p.get("epochs", 15))
        batch_size = int(p.get("batch_size", 64))
        class_balance = p.get("class_balance", "sqrt")
        dual = str(getattr(self, "_head", p.get("head") or "multiclass")) == "dual_binary"
        from src.training.cost_matrix import cost_matrix_array
        cost_on = (not dual) and cost_matrix_array(getattr(self, "_obj_doc", None)) is not None

        oof_frames: list[pd.DataFrame] = []
        histories: list[dict[str, list[float]]] = []
        last_scaler = None
        last_model = model
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start
        for k, fold in enumerate(train_folds):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_tr_raw = np.asarray(fold["X_train"])
            if X_tr_raw.ndim not in (2, 3) or not len(X_tr_raw):
                raise ValueError("transformer X_train must be a nonempty 2-D or 3-D array")
            if not np.isfinite(X_tr_raw).all():
                raise ValueError("transformer X_train must contain only finite values")
            if len(fold["y_train"]) != len(X_tr_raw):
                raise ValueError("transformer X_train/y_train row counts differ")
            for split in ("val", "oof"):
                X_split = fold.get(f"X_{split}")
                y_split = fold.get(f"y_{split}")
                if X_split is not None and len(X_split):
                    if (X_split.ndim != X_tr_raw.ndim or X_split.shape[1:] != X_tr_raw.shape[1:]
                            or not np.isfinite(X_split).all()):
                        raise ValueError(f"transformer X_{split} shape or values are invalid")
                    if y_split is not None and len(y_split) != len(X_split):
                        raise ValueError(f"transformer X_{split}/y_{split} row counts differ")
                    if split == "val" and y_split is None:
                        raise ValueError("transformer validation features require y_val")
            if getattr(X_tr_raw, "ndim", 0) == 2:
                X_tr_2d = scaler.fit_transform(X_tr_raw).astype("float32")
                X_val_raw = fold.get("X_val")
                X_val_2d = (scaler.transform(X_val_raw).astype("float32")
                            if X_val_raw is not None and len(X_val_raw) else X_val_raw)
                last_model = self._rebuild(int(X_tr_2d.shape[-1]))
                last_scaler = scaler
                X_tr = make_windows(X_tr_2d, seq_len)
                y_tr = trim_warmup(fold["y_train"], seq_len)
                X_val = make_windows(X_val_2d, seq_len) if X_val_2d is not None and len(X_val_2d) else X_val_2d
                y_val = trim_warmup(fold.get("y_val"), seq_len) if X_val is not None and len(X_val) else fold.get("y_val")
            else:
                if X_tr_raw.shape[1] != seq_len:
                    raise ValueError("transformer 3-D X_train sequence width differs from seq_len")
                # Pre-windowed folds also require independent model/optimizer
                # state. Reusing the build() model leaked weights across folds.
                last_model = self._rebuild(int(X_tr_raw.shape[-1]))
                last_scaler = None
                X_tr, y_tr = X_tr_raw, fold["y_train"]
                X_val, y_val = fold.get("X_val"), fold.get("y_val")
            if dual:
                from src.models.dual_binary import as_dual_binary_y
                y_tr = as_dual_binary_y(y_tr)
                if y_val is not None and len(y_val):
                    y_val = as_dual_binary_y(y_val)
            if X_val is not None and len(X_val):
                X_val, y_val = stride_cap(X_val, y_val, VAL_CAP)
            if cost_on:
                sw = None
                y_fit = y_tr
                val_y = y_val
            elif dual:
                from src.models.dual_binary import dual_sample_weight, y_as_head_dict
                sw = dual_sample_weight(y_tr, class_balance)
                y_fit = y_as_head_dict(y_tr)
                val_y = y_as_head_dict(y_val) if y_val is not None and len(y_val) else y_val
            else:
                sw = _make_sample_weight(y_tr, class_balance)
                y_fit = y_tr
                val_y = y_val
            val_data = batched_sequence(X_val, val_y, batch_size) if X_val is not None and len(X_val) else None
            logger.info("fit fold=%d X_tr=%s dual=%s", k, getattr(X_tr, "shape", None), dual)
            note_train_start(k, n_folds, n_epochs=epochs)
            fold_callbacks = list(train_callbacks(
                fold_i=k, n_folds=n_folds, n_epochs=epochs,
                has_val=val_data is not None,
            ) or []) + list(callbacks or [])
            history = last_model.fit(
                batched_sequence(X_tr, y_fit, batch_size, sample_weight=sw),
                validation_data=val_data,
                epochs=epochs, verbose=0,
                callbacks=fold_callbacks,
            )
            histories.append({
                key: [float(value) for value in values]
                for key, values in history.history.items()
            })
            if "X_oof" in fold and fold["X_oof"] is not None and len(fold["X_oof"]):
                Xo = fold["X_oof"]
                if last_scaler is not None and getattr(Xo, "ndim", 0) == 2:
                    Xo = make_windows(last_scaler.transform(Xo).astype("float32"), seq_len)
                    yo = trim_warmup(fold.get("y_oof"), seq_len)
                    ts_oof = trim_warmup(fold.get("ts_oof"), seq_len)
                else:
                    yo = fold.get("y_oof")
                    ts_oof = fold.get("ts_oof")
                if dual:
                    from src.models.dual_binary import as_dual_binary_y
                    yo = as_dual_binary_y(yo) if yo is not None else yo
                from src.models.dual_binary import keras_proba_to_canonical
                oof_frames.append(keras_proba_to_canonical(
                    predict_chunked(last_model, Xo), ts=ts_oof, y_true=yo))
                checkpoint_fold(k, n_folds, oof_frames[-1])
            else:
                checkpoint_fold(k, n_folds, None)
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=p),
            artifact=last_model, oof_preds=oof,
            meta={"seq_len": seq_len, "scaler": last_scaler,
                  "history": histories,
                  "head": str(getattr(self, "_head", p.get("head") or "multiclass"))},
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        seq_len = int((fitted.meta or {}).get("seq_len")
                      or getattr(self, "_params", {}).get("seq_len", 128))
        scaler = (fitted.meta or {}).get("scaler")
        X2 = X
        if scaler is not None and getattr(X, "ndim", 0) == 2:
            X2 = scaler.transform(X).astype("float32")
        X_w = make_windows(X2, seq_len) if getattr(X2, "ndim", 0) == 2 else X2
        from src.models.dual_binary import keras_proba_to_canonical
        df = keras_proba_to_canonical(predict_chunked(fitted.artifact, X_w))
        validate_canonical(df, plugin_name=self.name)
        return df
