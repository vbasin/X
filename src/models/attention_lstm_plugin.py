"""Attention-LSTM plugin — wraps the existing model as a ModelPlugin.

This is a thin adapter; the implementation stays in ``src/models/attention_lstm.py``.
New model families follow this same pattern: one ``*_plugin.py`` file per family.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import (
    FittedModel,
    ModelSpec,
    validate_canonical,
)
from src.models.dl_memory import (
    VAL_CAP, batched_sequence, predict_chunked, stride_indices, train_callbacks,
)
from src.models.sequence_windows import (
    make_windows, trim_warmup, window_contiguous_mask,
)
from src.models.tabular_plugins import _make_sample_weight

logger = logging.getLogger("vtech.models.attention_lstm_plugin")


class AttentionLSTMPlugin:
    """ModelPlugin adapter for the existing Attention-LSTM (Detection-3-class).

    Handles 2-D fold arrays (shape N × F) emitted by the search controller's
    fold builder by converting them to sliding sequence windows (N-seq_len+1 ×
    seq_len × F) before training. If the actual feature count after feature
    selection differs from the ``n_features`` param, the model is rebuilt with
    the correct shape automatically.
    """

    name: str = "attention_lstm"

    # Populated in build(); used to rebuild with the correct feature count in fit().
    _params: dict[str, Any]

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        """Build (but do not train) an Attention-LSTM from *spec*."""
        from src.models.attention_lstm import build_attention_lstm
        from src.training.cost_matrix import keras_compile_loss

        self._params = dict(spec.params)   # store for potential rebuild in fit()
        self._obj_doc = objective
        self._head = str(spec.params.get("head") or "multiclass")
        n_features = spec.params.get("n_features", 154)
        seq_len    = spec.params.get("seq_len", 128)
        dual = self._head == "dual_binary"
        return build_attention_lstm(
            input_shape=(seq_len, n_features),
            lstm_units=spec.params.get("lstm_units", [64, 32]),
            attention_heads=spec.params.get("attention_heads", 4),
            dense_units=spec.params.get("dense_units", [64, 32]),
            dropout=spec.params.get("dropout", 0.2),
            num_classes=2 if dual else 3,
            output_activation="sigmoid" if dual else "softmax",
            loss=None if dual else keras_compile_loss(objective),
            head=self._head,
            n_long_ctx=int(getattr(self, "_n_long_ctx", 0) or 0),
            n_short_ctx=int(getattr(self, "_n_short_ctx", 0) or 0),
            aux=bool(getattr(self, "_aux", False)),
            aux_columns=getattr(self, "_aux_cols", None) if getattr(self, "_aux", False) else None,
        )

    def _rebuild(self, n_features: int) -> Any:
        """Fresh compiled model for *n_features* — new weights each walk-forward fold."""
        from src.models.attention_lstm import build_attention_lstm
        from src.training.cost_matrix import keras_compile_loss
        p = getattr(self, "_params", {})
        seq_len = int(p.get("seq_len", 128))
        head = str(getattr(self, "_head", p.get("head") or "multiclass"))
        dual = head == "dual_binary"
        n_long = int(getattr(self, "_n_long_ctx", 0) or 0)
        n_short = int(getattr(self, "_n_short_ctx", 0) or 0)
        aux = bool(getattr(self, "_aux", False))
        logger.info(
            "attention_lstm: rebuild n_features=%d seq_len=%d dual=%s "
            "n_long_ctx=%d n_short_ctx=%d aux=%s",
            n_features, seq_len, dual, n_long, n_short, aux)
        return build_attention_lstm(
            input_shape=(seq_len, int(n_features)),
            lstm_units=p.get("lstm_units", [64, 32]),
            attention_heads=p.get("attention_heads", 4),
            dense_units=p.get("dense_units", [64, 32]),
            dropout=p.get("dropout", 0.2),
            num_classes=2 if dual else 3,
            output_activation="sigmoid" if dual else "softmax",
            loss=None if dual else keras_compile_loss(getattr(self, "_obj_doc", None)),
            head=head,
            n_long_ctx=n_long,
            n_short_ctx=n_short,
            aux=aux,
            aux_columns=getattr(self, "_aux_cols", None) if aux else None,
        )

    def _rebuild_for_features(self, model: Any, actual_n_features: int) -> Any:
        """Compat wrapper — always rebuilds (ignores *model* weights)."""
        return self._rebuild(actual_n_features)

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        """Train on walk-forward folds; returns OOF predictions with ``direction_true``.

        Accepts 2-D (N, F) fold arrays (windowed here) or already 3-D sequences.
        Each fold gets a fresh model + RobustScaler so fold-0 collapse cannot
        poison later OOF (same contract as TCN).
        """
        del model  # rebuilt per fold from actual F
        if objective is not None:
            self._obj_doc = objective
        params = getattr(self, "_params", {})
        seq_len = int(params.get("seq_len", 128))
        epochs = int(params.get("epochs", 15))
        batch_size = int(params.get("batch_size", 128))
        # Same as TCN: unweighted CE on ~87% flat is an always-flat fixed point.
        # Skipped when the objective uses the 3x3 expected-cost loss (ADR-072).
        from src.training.cost_matrix import cost_matrix_array
        class_balance = params.get("class_balance", "sqrt")
        cost_on = cost_matrix_array(getattr(self, "_obj_doc", None)) is not None

        oof_frames: list[pd.DataFrame] = []
        last_scaler = None
        last_model: Any = None
        n_folds = len(train_folds)
        from src.training.job_progress import checkpoint_fold, note_train_start
        for k, fold in enumerate(train_folds):
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_tr_2d = scaler.fit_transform(fold["X_train"]).astype("float32")
            X_val_raw = fold.get("X_val")
            if X_val_raw is not None and len(X_val_raw):
                X_val_2d = scaler.transform(X_val_raw).astype("float32")
            else:
                X_val_2d = X_val_raw
            dual = str(getattr(self, "_head", "multiclass")) == "dual_binary"
            cols = list(fold.get("feature_cols") or [])
            spec = {"shared": list(range(X_tr_2d.shape[-1])), "long": [], "short": []}
            if dual and cols and len(cols) == int(X_tr_2d.shape[-1]):
                from src.models.dual_binary import head_feature_spec
                spec = head_feature_spec(cols)
            self._head_spec = spec
            from src.training.aux_labels import aux_from_objective
            req = aux_from_objective(getattr(self, "_obj_doc", None))
            aux_tr = fold.get("y_aux_train")
            self._aux_cols = req
            self._aux = bool(
                dual and req and aux_tr is not None and getattr(aux_tr, "size", 0))
            self._n_long_ctx = len(spec.get("long") or [])
            self._n_short_ctx = len(spec.get("short") or [])

            from src.models.dual_binary import pack_sequence_inputs, split_head_columns
            X_tr_s, X_tr_l, X_tr_h = split_head_columns(X_tr_2d, spec)
            actual_f = int(X_tr_s.shape[-1]) if X_tr_s.ndim == 2 else int(X_tr_2d.shape[-1])
            last_model = self._rebuild(actual_f)
            last_scaler = scaler

            X_tr = make_windows(X_tr_s, seq_len)
            y_raw = fold["y_train"]
            if dual:
                from src.models.dual_binary import as_dual_binary_y, y_as_head_dict
                y_raw = as_dual_binary_y(y_raw)
            y_tr = trim_warmup(y_raw, seq_len)
            long_tr = trim_warmup(X_tr_l, seq_len) if X_tr_l is not None else None
            short_tr = trim_warmup(X_tr_h, seq_len) if X_tr_h is not None else None
            x_fit = pack_sequence_inputs(X_tr, long_tr, short_tr)

            aux_mu = aux_sd = None
            aux_tr_w = None
            if self._aux:
                from src.training.aux_labels import (
                    aux_sample_weight, pack_aux_outputs, zscore_fit,
                )
                aux_tr_w = trim_warmup(aux_tr, seq_len)
                aux_mu, aux_sd = zscore_fit(aux_tr_w)
                self._aux_mu = aux_mu
                self._aux_sd = aux_sd

            if X_val_2d is not None and len(X_val_2d):
                X_val_s, X_val_l, X_val_h = split_head_columns(X_val_2d, spec)
                X_val = make_windows(X_val_s, seq_len)
                yv = fold["y_val"]
                if dual:
                    from src.models.dual_binary import as_dual_binary_y
                    yv = as_dual_binary_y(yv)
                y_val = trim_warmup(yv, seq_len)
                long_v = trim_warmup(X_val_l, seq_len) if X_val_l is not None else None
                short_v = trim_warmup(X_val_h, seq_len) if X_val_h is not None else None
                cap_idx = stride_indices(len(X_val), VAL_CAP)
                if cap_idx is not None:
                    X_val = X_val[cap_idx]
                    y_val = y_val[cap_idx]
                    if long_v is not None:
                        long_v = long_v[cap_idx]
                    if short_v is not None:
                        short_v = short_v[cap_idx]
                x_val = pack_sequence_inputs(X_val, long_v, short_v)
                val_y = y_as_head_dict(y_val) if dual else y_val
                if self._aux:
                    from src.training.aux_labels import pack_aux_outputs
                    aux_v = fold.get("y_aux_val")
                    if aux_v is not None and getattr(aux_v, "size", 0):
                        aux_v = trim_warmup(aux_v, seq_len)
                        if cap_idx is not None:
                            aux_v = aux_v[cap_idx]
                        val_y = {**val_y, **pack_aux_outputs(
                            aux_v, mu=aux_mu, sd=aux_sd, columns=self._aux_cols)}
                val_keep = np.ones(len(y_val), dtype="float64")
                vs = fold.get("w_session_val")
                if vs is not None and getattr(vs, "size", 0):
                    vw = trim_warmup(np.asarray(vs, dtype="float64"), seq_len)
                    if cap_idx is not None:
                        vw = vw[cap_idx]
                    val_keep = val_keep * vw
                vgap = window_contiguous_mask(fold.get("ts_val"), seq_len)
                if vgap is not None and vgap.size:
                    if cap_idx is not None:
                        vgap = vgap[cap_idx]
                    val_keep = val_keep * vgap.astype("float64")
                val_data = batched_sequence(
                    x_val, val_y, batch_size, sample_weight=val_keep)
            else:
                val_data = None
            keep = np.ones(len(y_tr), dtype="float64")
            sess_w = fold.get("w_session_train")
            if sess_w is not None and getattr(sess_w, "size", 0):
                keep = keep * trim_warmup(np.asarray(sess_w, dtype="float64"), seq_len)
            gap_ok = window_contiguous_mask(fold.get("ts_train"), seq_len)
            if gap_ok is not None and gap_ok.size:
                keep = keep * gap_ok.astype("float64")
            onset = fold.get("w_onset_train")
            if onset is not None and getattr(onset, "size", 0):
                keep = keep * trim_warmup(np.asarray(onset, dtype="float64"), seq_len)
            active = keep > 0
            if cost_on:
                sw = None
            elif dual:
                from src.models.dual_binary import dual_sample_weight
                sw = dual_sample_weight(y_tr, class_balance, mask=active)
            else:
                sw = _make_sample_weight(y_tr, class_balance, mask=active)
            y_fit = y_as_head_dict(y_tr) if dual else y_tr
            if dual and self._aux:
                from src.training.aux_labels import aux_sample_weight, pack_aux_outputs
                y_fit = {**y_fit, **pack_aux_outputs(
                    aux_tr_w, mu=aux_mu, sd=aux_sd, columns=self._aux_cols)}
                aux_w = aux_sample_weight(aux_tr_w, columns=self._aux_cols)
                if sw is None:
                    ones = np.ones(len(y_tr), dtype="float64")
                    sw = {"p_long": ones, "p_short": ones}
                sw = {**sw, **aux_w}
            if sw is None:
                sw = keep
            elif isinstance(sw, dict):
                sw = {k: np.asarray(v, dtype="float64") * keep for k, v in sw.items()}
            else:
                sw = np.asarray(sw, dtype="float64") * keep
            logger.info(
                "fit fold=%d X_tr=%s X_val=%s batch=%d class_balance=%s cost_matrix=%s "
                "dual=%s aux=%s n_long_ctx=%d n_short_ctx=%d "
                "(Sequence-fed, RobustScaler, cuDNN)",
                k, X_tr.shape, None if val_data is None else (len(X_val),),
                batch_size, None if cost_on else class_balance, cost_on, dual,
                self._aux, self._n_long_ctx, self._n_short_ctx)
            note_train_start(k, n_folds, n_epochs=epochs)
            last_model.fit(
                batched_sequence(x_fit, y_fit, batch_size, sample_weight=sw),
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
                x_oof = self._window_pack(X_oof_2d, spec, seq_len)
                yo = fold.get("y_oof")
                if str(getattr(self, "_head", "multiclass")) == "dual_binary":
                    from src.models.dual_binary import as_dual_binary_y
                    yo = as_dual_binary_y(yo) if yo is not None else yo
                y_oof = trim_warmup(yo, seq_len)
                ts_oof = trim_warmup(fold.get("ts_oof"), seq_len)
                oof_frames.append(self._proba_to_canonical(
                    predict_chunked(last_model, x_oof),
                    ts=ts_oof,
                    y_true=y_oof,
                ))
                checkpoint_fold(k, n_folds, oof_frames[-1])
            else:
                checkpoint_fold(k, n_folds, None)

        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(
            spec=ModelSpec(family=self.name, params=params),
            artifact=last_model, oof_preds=oof,
            meta={"seq_len": seq_len, "scaler": last_scaler,
                  "head": str(getattr(self, "_head", "multiclass")),
                  "head_spec": spec,
                  "aux": bool(getattr(self, "_aux", False)),
                  "feature_cols": list(fold.get("feature_cols") or [])},
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        """Predict on *X* (2-D or 3-D) and return the canonical frame.

        2-D arrays are RobustScaled (last-fold scaler) then windowed; only bars
        with a full look-back window are returned.
        """
        seq_len = int((fitted.meta or {}).get("seq_len")
                      or getattr(self, "_params", {}).get("seq_len", 128))
        scaler = (fitted.meta or {}).get("scaler")
        spec = (fitted.meta or {}).get("head_spec") or getattr(self, "_head_spec", None)
        X2 = X
        if scaler is not None and getattr(X, "ndim", 0) == 2:
            X2 = scaler.transform(X).astype("float32")
        if getattr(X2, "ndim", 0) == 2:
            x_in = self._window_pack(X2, spec, seq_len)
        else:
            x_in = X2
        proba = predict_chunked(fitted.artifact, x_in)
        df = self._proba_to_canonical(proba)
        validate_canonical(df, plugin_name=self.name)
        return df

    @staticmethod
    def _window_pack(X2: np.ndarray, spec: dict | None, seq_len: int):
        from src.models.dual_binary import pack_sequence_inputs, split_head_columns
        if not spec:
            return make_windows(X2, seq_len)
        xs, xl, xh = split_head_columns(X2, spec)
        Xw = make_windows(xs if xs.size else X2, seq_len)
        long_c = trim_warmup(xl, seq_len) if xl is not None else None
        short_c = trim_warmup(xh, seq_len) if xh is not None else None
        return pack_sequence_inputs(Xw, long_c, short_c)

    def _proba_to_canonical(
        self,
        proba: np.ndarray,
        ts: np.ndarray | None = None,
        y_true: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Convert a (N, 3) softmax array to the canonical prediction frame.

        Pass *y_true* (integer label array, 0=down/1=flat/2=up) to include
        the ``direction_true`` column required for OOF scoring.
        """
        from src.models.dual_binary import dual_binary_fire, keras_proba_to_canonical
        return keras_proba_to_canonical(
            proba, ts=ts, y_true=y_true,
            fire=dual_binary_fire(getattr(self, "_obj_doc", None)),
        )
