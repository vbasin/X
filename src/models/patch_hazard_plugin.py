"""Four-stream patch Transformer + shared hazard heads (ADR-053 Encoder B).

Encodes the **same 5s feature view as the TCN**, split by prefix (price /
order-flow / volume-profile / options-context) and patched along time
(default 8 × 5s bars per token). Empty streams emit a dummy token so the
ablation can share `sequence_baseline`. This is **not** a 1s-candle encoder —
`parquet_path1s` is the label path only. Optional 1s OHLC as a second input
is on hold.

Defaults size the encoder at ~0.9 M params (d_model=128, 4 layers, ff=512).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.base import FittedModel, ModelSpec, validate_canonical
from src.models.hazard_fit import compile_on_embedding, fit_hazard_folds, predict_hazard

STREAMS: dict[str, tuple[str, ...]] = {
    "price": ("vw_", "cs_", "ht_", "dc_"),
    "flow": ("of_", "bp_", "tl_", "ss_", "ms_", "lt_", "ta_"),
    "volume": ("vp_", "rv_", "sd_", "vf_"),
    "options": ("iv_", "gx_", "ca_"),
}


class PatchHazardPlugin:
    name: str = "patch_hazard"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        self._params = dict(spec.params or {})
        p = self._params
        n_features = int(p.get("n_features") or 1)
        return self._rebuild(n_features, None)

    def _rebuild(self, n_features: int, feature_cols: list[str] | None) -> Any:
        p = getattr(self, "_params", {})
        return _build_patch_encoder(
            seq_len=int(p.get("seq_len", 128)),
            n_features=int(n_features),
            feature_cols=feature_cols,
            p=p,
        )

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        p = getattr(self, "_params", {})
        return fit_hazard_folds(
            rebuild=self._rebuild,
            train_folds=train_folds,
            params=p,
            plugin_name=self.name,
            spec=ModelSpec(family=self.name, params=p),
        )

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        df = predict_hazard(
            fitted, X, plugin_name=self.name,
            params=getattr(self, "_params", {}) or (fitted.spec.params if fitted.spec else {}),
        )
        validate_canonical(df, plugin_name=self.name)
        return df


def _stream_indices(feature_cols: list[str] | None, n_features: int) -> dict[str, list[int]]:
    if not feature_cols or len(feature_cols) != n_features:
        # Equal chunks if the fold did not carry names (unit tests / placeholder build).
        q, rem = divmod(n_features, 4)
        out: dict[str, list[int]] = {}
        start = 0
        for i, name in enumerate(STREAMS):
            n = q + (1 if i < rem else 0)
            out[name] = list(range(start, start + n))
            start += n
        return out
    out = {}
    for name, prefixes in STREAMS.items():
        out[name] = [i for i, c in enumerate(feature_cols)
                     if any(c.startswith(p) for p in prefixes)]
    return out


def _build_patch_encoder(
    *,
    seq_len: int,
    n_features: int,
    feature_cols: list[str] | None,
    p: dict[str, Any],
) -> Any:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    from src.models.hazard_fit import compile_on_embedding

    d_model = int(p.get("d_model", 128))
    n_heads = int(p.get("n_heads", 4))
    n_layers = int(p.get("n_layers", 4))
    dropout = float(p.get("dropout", 0.1))
    patch_len = int(p.get("patch_len", 8))
    ff_dim = int(p.get("ff_dim", 512))
    include_quantile = bool(p.get("include_quantile", True))
    seq_len = int(seq_len)
    if seq_len % patch_len:
        raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
    n_patches = seq_len // patch_len
    streams = _stream_indices(feature_cols, n_features)

    from src.models.keras_layers import GatherFeatures, ZerosTokens

    inputs = keras.Input(shape=(seq_len, int(n_features)))
    tokens: list[Any] = []
    for sname, idx in streams.items():
        if idx:
            gathered = GatherFeatures(idx, name=f"{sname}_gather")(inputs)
            tok = layers.Conv1D(
                d_model, patch_len, strides=patch_len, padding="valid",
                name=f"{sname}_patch",
            )(gathered)
        else:
            dummy = ZerosTokens(n_patches, name=f"{sname}_zeros")(inputs)
            tok = layers.Dense(d_model, name=f"{sname}_dummy")(dummy)
        tokens.append(tok)

    x = layers.Concatenate(axis=1, name="patch_concat")(tokens)
    n_tok = n_patches * len(STREAMS)
    pos = layers.Embedding(n_tok, d_model, name="pos")(tf.range(0, n_tok))
    x = x + pos
    for i in range(n_layers):
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
            name=f"mha_{i}",
        )(x, x)
        x = layers.LayerNormalization(name=f"ln1_{i}")(layers.Add()([x, attn]))
        ffn = layers.Dense(ff_dim, activation="relu", name=f"ff1_{i}")(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(d_model, name=f"ff2_{i}")(ffn)
        x = layers.LayerNormalization(name=f"ln2_{i}")(layers.Add()([x, ffn]))
    emb = layers.GlobalAveragePooling1D()(x)
    emb = layers.Dense(64, activation="relu")(emb)
    emb = layers.Dropout(dropout)(emb)
    return compile_on_embedding(inputs, emb, include_quantile=include_quantile)
