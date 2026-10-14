"""1s-path TCN + shared hazard heads (ADR-053 Encoder A, recovered spec).

10 dilated causal layers, dilations ``1…512``, kernel 2 → receptive field 1,024 s.
Reads ATR-normalised OHLC features from :mod:`src.search.path1s_folds`, not the
5s engineered matrix. Last causal timestep is the embedding (not GAP).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import FittedModel, ModelSpec, validate_canonical
from src.models.heads import (
    MIN_INFERENCE_R,
    apply_inference_r_gate,
    compile_cr_hazard_model,
    decode_hazard_class,
    encode_cr_sequence,
    serving_from_cr_logits,
)
from src.models.path_windows import (
    PathWindowSequence,
    as_keras_sequence,
    entry_pairs,
    gather_field,
    gather_windows,
)
from src.models.tabular_plugins import _make_sample_weight
from src.search.path1s_folds import PathSession

logger = logging.getLogger("vtech.models.path_tcn")

PATH_TCN_DILATIONS: tuple[int, ...] = tuple(2**i for i in range(10))  # 1..512


def path_tcn_embedding(*, seq_len: int, n_features: int, p: dict[str, Any]) -> tuple[Any, Any]:
    from tensorflow import keras
    from tensorflow.keras import layers

    filters = int(p.get("filters", 64))
    kernel_size = int(p.get("kernel_size", 2))
    dropout = float(p.get("dropout", 0.1))
    dilations = list(p.get("dilations") or PATH_TCN_DILATIONS)

    inputs = keras.Input(shape=(int(seq_len), int(n_features)))
    x = inputs
    for d in dilations:
        skip = x
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=int(d), activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters, kernel_size, padding="causal",
                          dilation_rate=int(d), activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        if skip.shape[-1] != filters:
            skip = layers.Conv1D(filters, 1)(skip)
        x = layers.Add()([x, skip])
    from src.models.keras_layers import LastTimestep
    last = LastTimestep(name="last_step")(x)
    emb = layers.Dense(64, activation="relu")(last)
    emb = layers.Dropout(dropout)(emb)
    return inputs, emb


class PathTCNPlugin:
    name: str = "path_tcn"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._params = dict(spec.params or {})
        p = self._params
        n_features = int(p.get("n_features") or 6)
        return self._rebuild(n_features)

    def _rebuild(self, n_features: int) -> Any:
        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 1024))
        inputs, emb = path_tcn_embedding(seq_len=seq_len, n_features=int(n_features), p=p)
        return compile_cr_hazard_model(
            inputs, emb,
            include_quantile=bool(p.get("include_quantile", True)),
        )

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, Any]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        from sklearn.preprocessing import RobustScaler

        from src.models.dl_memory import VAL_CAP, train_callbacks
        from src.models.hazard_fit import predict_hazard_chunked

        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 1024))
        epochs = int(p.get("epochs", 15))
        batch_size = int(p.get("batch_size", 64))
        class_balance = p.get("class_balance", "sqrt")
        include_quantile = bool(p.get("include_quantile", True))
        min_r = float(p.get("inference_r_min", MIN_INFERENCE_R) or 0.0)
        y_fields = ("cr_up", "cr_down") + (("y_quantile",) if include_quantile else ())

        oof_frames: list[pd.DataFrame] = []
        last_model: Any = None
        last_scaler = None
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start

        for k, fold in enumerate(train_folds):
            train_s: list[PathSession] = list(fold.get("sessions_train") or [])
            if not train_s:
                raise ValueError(
                    "path_tcn requires sessions_train — use search.input=path1s "
                    "(1s path folds, not the 5s feature join)"
                )
            scaler = RobustScaler()
            scaler.fit(_sample_rows(train_s))
            last_scaler = scaler
            n_features = int(train_s[0].X.shape[-1])
            last_model = self._rebuild(n_features)

            tr_scaled = [_ensure_cr(_scale_session(s, scaler)) for s in train_s]
            seq_tr = PathWindowSequence(
                tr_scaled, seq_len, batch_size, y_fields=y_fields, shuffle=True,
            )
            y_sparse = gather_field(tr_scaled, [(int(a), int(b)) for a, b in seq_tr.pairs], "y")
            sw = _make_sample_weight(y_sparse, class_balance)
            if sw is not None:
                seq_tr.sample_weight = sw

            val_data = None
            val_s = list(fold.get("sessions_val") or [])
            if val_s:
                val_scaled = [_ensure_cr(_scale_session(s, scaler)) for s in val_s]
                seq_val = PathWindowSequence(
                    val_scaled, seq_len, batch_size, y_fields=y_fields, shuffle=False,
                )
                if len(seq_val.pairs) > VAL_CAP:
                    seq_val.pairs = seq_val.pairs[
                        np.linspace(0, len(seq_val.pairs) - 1, VAL_CAP).astype(int)
                    ]
                if len(seq_val.pairs):
                    val_data = as_keras_sequence(seq_val)

            logger.info(
                "path_tcn fit fold=%d train_windows=%d seq_len=%d batch=%d",
                k, len(seq_tr.pairs), seq_len, batch_size,
            )
            note_train_start(k, n_folds, n_epochs=epochs)
            last_model.fit(
                as_keras_sequence(seq_tr),
                validation_data=val_data,
                epochs=epochs,
                verbose=0,
                callbacks=train_callbacks(
                    fold_i=k, n_folds=n_folds, n_epochs=epochs,
                    has_val=val_data is not None,
                ),
            )

            oof_s = list(fold.get("sessions_oof") or [])
            if oof_s and last_model is not None:
                oof_scaled = [_ensure_cr(_scale_session(s, scaler)) for s in oof_s]
                pairs = entry_pairs(oof_scaled, seq_len)
                if pairs:
                    Xw = gather_windows(oof_scaled, pairs, seq_len)
                    outs = predict_hazard_chunked(last_model, Xw)
                    ts_e = np.array([
                        oof_scaled[si].ts[oof_scaled[si].entry_idx[ei]]
                        for si, ei in pairs
                    ])
                    df = serving_from_cr_logits(
                        outs["y_up"], outs["y_down"], ts=ts_e, plugin_name=self.name,
                    )
                    r = np.array([
                        oof_scaled[si].r_value[ei] for si, ei in pairs
                    ], dtype="float64")
                    df = apply_inference_r_gate(df, r, min_r=min_r)
                    validate_canonical(df, plugin_name=self.name)
                    oof_frames.append(df)
                    checkpoint_fold(k, n_folds, df)
                else:
                    checkpoint_fold(k, n_folds, None)
            else:
                checkpoint_fold(k, n_folds, None)

        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=p),
            artifact=last_model,
            oof_preds=oof,
            meta={"seq_len": seq_len, "scaler": last_scaler, "inference_r_min": min_r},
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        """Window a contiguous 1s feature matrix ``(T, F)`` and score after warmup."""
        from src.models.hazard_fit import predict_hazard_chunked

        p = getattr(self, "_params", {}) or (fitted.spec.params if fitted.spec else {})
        seq_len = int((fitted.meta or {}).get("seq_len") or p.get("seq_len", 1024))
        scaler = (fitted.meta or {}).get("scaler")
        X2 = scaler.transform(X).astype("float32") if scaler is not None else np.asarray(X, dtype="float32")
        if len(X2) < seq_len:
            return serving_from_cr_logits(
                np.zeros((0, 20, 3)), np.zeros((0, 20, 3)), plugin_name=self.name,
            )
        idx = np.lib.stride_tricks.sliding_window_view(np.arange(len(X2)), seq_len)
        Xw = X2[idx]
        outs = predict_hazard_chunked(fitted.artifact, Xw)
        df = serving_from_cr_logits(outs["y_up"], outs["y_down"], plugin_name=self.name)
        validate_canonical(df, plugin_name=self.name)
        return df


def _sample_rows(sessions: list[PathSession], cap_per: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(0)
    chunks = []
    for s in sessions:
        n = len(s.X)
        if n == 0:
            continue
        take = min(n, cap_per)
        idx = rng.choice(n, size=take, replace=False)
        chunks.append(s.X[idx])
    if not chunks:
        raise ValueError("no path rows to fit RobustScaler")
    return np.concatenate(chunks, axis=0)


def _scale_session(s: PathSession, scaler) -> PathSession:
    return replace(s, X=scaler.transform(s.X).astype("float32"))


def _ensure_cr(s: PathSession) -> PathSession:
    if s.cr_up is not None and len(s.cr_up) == len(s.y_up):
        return s
    ev, bu = decode_hazard_class(s.y_up)
    cr_up, mask_up = encode_cr_sequence(ev, bu)
    evd, bud = decode_hazard_class(s.y_down)
    cr_dn, mask_dn = encode_cr_sequence(evd, bud)
    return replace(s, cr_up=cr_up, cr_down=cr_dn,
                   cr_mask_up=mask_up, cr_mask_down=mask_dn)
