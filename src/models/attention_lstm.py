"""Attention-LSTM model definition (TensorFlow/Keras)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("vtech.models.attention_lstm")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class AttentionLayer(layers.Layer):
    """Multi-head self-attention layer."""

    def __init__(self, units: int, num_heads: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads

    def build(self, input_shape: tf.TensorShape) -> None:
        d_in = int(input_shape[-1])
        self.W_q = self.add_weight(name="W_q", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_k = self.add_weight(name="W_k", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_v = self.add_weight(name="W_v", shape=(d_in, self.units), initializer="glorot_uniform")
        self.W_o = self.add_weight(name="W_o", shape=(self.units, self.units), initializer="glorot_uniform")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Use -1 for batch dim so XLA sees one fixed shape regardless of tail-batch size.
        # seq_len: prefer static Python int (known at build time); fall back to tf.shape.
        seq_len: int | tf.Tensor = inputs.shape[1] or tf.shape(inputs)[1]

        Q = tf.reshape(tf.matmul(inputs, self.W_q), [-1, seq_len, self.num_heads, self.head_dim])
        K = tf.reshape(tf.matmul(inputs, self.W_k), [-1, seq_len, self.num_heads, self.head_dim])
        V = tf.reshape(tf.matmul(inputs, self.W_v), [-1, seq_len, self.num_heads, self.head_dim])

        Q, K, V = (tf.transpose(t, [0, 2, 1, 3]) for t in (Q, K, V))

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)

        context = tf.reshape(tf.transpose(context, [0, 2, 1, 3]), [-1, seq_len, self.units])
        return tf.matmul(context, self.W_o)

    def get_config(self) -> dict:
        return {**super().get_config(), "units": self.units, "num_heads": self.num_heads}


def focal_loss(gamma: float = 2.0):
    """Focal loss for sparse integer multi-class labels.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=2.0 down-weights easy correctly-classified examples by (1-p_t)^2,
    concentrating gradient on hard misclassified examples.  A well-predicted
    NO_SIGNAL bar at p=0.97 contributes (0.03)^2 = 0.09% of its normal loss;
    a misclassified directional bar at p=0.02 keeps 98% of its loss.

    IMPORTANT: This function returns a PER-SAMPLE loss tensor of shape
    (batch_size,), NOT a scalar.  Keras requires per-sample losses so that
    ``model.fit(class_weight=...)`` can correctly reweight each sample before
    reducing to a scalar.  If tf.reduce_mean() is called inside here, Keras
    receives a scalar and class_weight only scales the gradient *magnitude*
    (not direction), effectively silencing the minority-class rebalancing.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        # Gather p_t: predicted probability for the true class
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(batch_size), y_true_int], axis=1)
        p_t = tf.gather_nd(y_pred, indices)
        p_t = tf.clip_by_value(p_t, 1e-7, 1.0)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        # Return per-sample losses (shape: [batch_size]) so Keras can apply
        # class_weight correctly to change gradient direction, not just scale.
        return focal_weight * (-tf.math.log(p_t))

    loss_fn.__name__ = f"focal_loss_g{gamma}"
    return loss_fn


def build_attention_lstm(
    input_shape: tuple[int, int],
    num_classes: int = 3,
    lstm_units: list[int] | None = None,
    attention_heads: int = 4,
    dense_units: list[int] | None = None,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    loss: Any = None,
    output_activation: str | None = None,
    head: str = "multiclass",
    n_long_ctx: int = 0,
    n_short_ctx: int = 0,
    aux: bool = False,
    aux_columns: tuple[str, ...] | None = None,
) -> Model:
    """Build and compile an Attention-LSTM classifier.

    Architecture
    ------------
    Input → [LSTM layers] → [Attention] → LayerNorm → GlobalPool → Dense → head

    ``head=dual_binary`` (ADR-080 / ADR-082): shared encoder on the sequence,
    optional last-bar ``long_ctx`` / ``short_ctx`` (lg_* / st_* only on that
    head), two named sigmoid outputs ``p_long`` / ``p_short``, independent BCE.
    ``aux=True`` adds Huber path-quality heads (objective ``labels.aux``,
    ADR-084) and a ``side_rank`` regularizer on ``p_long`` vs ``p_short``
    (ADR-088). Otherwise 3-class softmax.

    ``loss`` overrides focal / CE when the objective opts into expected-cost
    (ADR-072). Pass a Keras loss instance or a string name. Ignored for
    dual-binary (always BCE per head).
    """
    if lstm_units is None:
        lstm_units = [64, 32]
    if dense_units is None:
        dense_units = [64, 32]
    dual = str(head) == "dual_binary"

    inputs = layers.Input(shape=input_shape, name="input")
    x = inputs

    # Stacked LSTM layers — all but last return sequences
    # attention_heads=0 means no attention; last LSTM returns a single vector.
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1) or (attention_heads > 0)
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=dropout,
            unroll=False,  # cuDNN LSTM; unroll=True was a 10× A10 stall (ADR-043)
            name=f"lstm_{i}",
        )(x)

    if attention_heads > 0:
        att_units = lstm_units[-1]
        # Round up to nearest multiple of attention_heads
        att_units = ((att_units + attention_heads - 1) // attention_heads) * attention_heads
        x = AttentionLayer(att_units, attention_heads, name="attention")(x)
        x = layers.LayerNormalization(name="layer_norm")(x)
        x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # Dense trunk (shared by both heads)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    if dual:
        from src.models.dual_binary import (
            LONG_CTX_NAME, SHORT_CTX_NAME, compile_dual_binary_model,
        )
        model_inputs: Any = inputs
        long_ctx = short_ctx = None
        extra: list[Any] = [inputs]
        if int(n_long_ctx) > 0:
            long_ctx = layers.Input(shape=(int(n_long_ctx),), name=LONG_CTX_NAME)
            extra.append(long_ctx)
        if int(n_short_ctx) > 0:
            short_ctx = layers.Input(shape=(int(n_short_ctx),), name=SHORT_CTX_NAME)
            extra.append(short_ctx)
        if len(extra) > 1:
            model_inputs = extra
        model = compile_dual_binary_model(
            model_inputs, x, learning_rate=learning_rate, name="attention_lstm",
            long_context=long_ctx, short_context=short_ctx, aux=bool(aux),
            aux_columns=aux_columns,
        )
        logger.info("model_built", extra={
            "input": input_shape, "params": model.count_params(),
            "head": "dual_binary", "aux": bool(aux),
            "n_long_ctx": int(n_long_ctx), "n_short_ctx": int(n_short_ctx),
        })
        return model

    act = output_activation or ("sigmoid" if int(num_classes) == 2 else "softmax")
    outputs = layers.Dense(num_classes, activation=act, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="attention_lstm")
    if loss is not None:
        loss_fn = loss
    elif act == "sigmoid":
        loss_fn = "binary_crossentropy"
    else:
        loss_fn = focal_loss(focal_gamma) if use_focal_loss else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    logger.info("model_built", extra={"input": input_shape, "params": model.count_params()})
    return model
