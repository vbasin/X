"""Reinforcement Learning plugin — GPU, policy-gradient (REINFORCE).

The RL approach is fundamentally different from all other families:

    state   = feature vector at bar t
    action  = {0: down-signal, 1: no-signal (flat), 2: up-signal}
    reward  = derived from the objective's directional-move labels
              (positive for a correct directional call with valid MAE;
               negative for a wrong direction or a false positive)

Training uses REINFORCE (Monte-Carlo policy gradient) with a learned
baseline (actor-critic light: a single value head sharing the policy
backbone).  The policy network is a small MLP over the flat feature vector;
no sequence dimension is used at this stage (temporal context is implicit
in the engineered features).  A future extension can swap in an LSTM encoder.

Reward shaping (bootstrapped from supervised labels)
-----------------------------------------------------
The RL agent does not simulate a trading engine at this stage — it uses the
existing walk-forward labels as a proxy reward signal:

    label == action AND label != NO_SIGNAL  → +1.0  (correct directional call)
    label == NO_SIGNAL AND action == FLAT   → +0.5  (correct abstention)
    label == NO_SIGNAL AND action != FLAT   →  0.0  (false positive)
    label == DIRECTION AND action == FLAT   → -0.3  (missed signal)
    action != label AND both directional    → -1.0  (wrong direction)

This is a supervised-RL hybrid suitable for offline training on historical
labels.  A live reward signal (real P&L) can replace this in production.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import (
    CANONICAL_CONFIDENCE, CANONICAL_DIRECTION, CANONICAL_TS,
    FittedModel, ModelSpec, validate_canonical,
)

logger = logging.getLogger("vtech.models.rl_plugin")

_IDX_TO_DIR     = {0: "down", 1: "flat", 2: "up"}
_DIR_FROM_LABEL = {0: "down", 1: "flat", 2: "up"}

# Reward matrix: reward[true_label][predicted_action]
_REWARD: dict[int, dict[int, float]] = {
    0: {0: +1.0, 1: -0.3, 2: -1.0},   # true=down
    1: {0:  0.0, 1: +0.5, 2:  0.0},   # true=no_signal (flat)
    2: {0: -1.0, 1: -0.3, 2: +1.0},   # true=up
}


class RLPlugin:
    name: str = "rl"

    def build(self, spec: ModelSpec, objective: dict[str, Any]) -> Any:
        """Build the policy + value network (actor-critic, shared backbone)."""
        from tensorflow import keras
        from tensorflow.keras import layers

        p = spec.params
        n_features  = p.get("n_features", 154)
        seq_len     = p.get("seq_len", 128)
        hidden      = p.get("hidden_units", [256, 128])

        # Input: flat features (seq_len × n_features flattened)
        state_dim = seq_len * n_features
        inp = keras.Input(shape=(state_dim,), name="state")
        x = inp
        for units in hidden:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(p.get("dropout", 0.1))(x)

        # Dual heads
        policy_logits = layers.Dense(3, name="policy")(x)            # → softmax externally
        value         = layers.Dense(1, name="value", activation=None)(x)  # baseline

        model = keras.Model(inputs=inp, outputs=[policy_logits, value])
        return model

    def fit(
        self,
        model: Any,
        train_folds: list[dict[str, np.ndarray]],
        *,
        objective: dict[str, Any] | None = None,
    ) -> FittedModel:
        import tensorflow as tf
        from sklearn.preprocessing import RobustScaler

        p_lr         = 3e-4
        p_gamma      = 0.99          # discount
        p_epochs     = 50
        p_batch      = 512

        optimizer = tf.keras.optimizers.Adam(learning_rate=p_lr)
        oof_frames: list[pd.DataFrame] = []
        scaler: RobustScaler | None = None

        for fold in train_folds:
            X_tr  = _flatten(fold["X_train"])
            y_tr  = fold["y_train"]
            X_oof = _flatten(fold.get("X_oof", fold["X_val"]))
            y_oof = fold.get("y_oof", fold["y_val"])

            scaler = RobustScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_oof = scaler.transform(X_oof)

            # Precompute reward matrix: shape (N, 3)
            reward_mat = np.array(
                [[_REWARD[int(y)][a] for a in range(3)] for y in y_tr],
                dtype=np.float32,
            )

            ds = (
                tf.data.Dataset.from_tensor_slices((
                    X_tr.astype(np.float32), reward_mat
                ))
                .shuffle(len(X_tr))
                .batch(p_batch)
            )

            for epoch in range(p_epochs):
                epoch_loss = 0.0
                for X_b, R_b in ds:
                    with tf.GradientTape() as tape:
                        logits, values = model(X_b, training=True)
                        probs   = tf.nn.softmax(logits)
                        actions = tf.argmax(probs, axis=1)
                        # Expected reward under current policy
                        expected_r = tf.reduce_sum(probs * R_b, axis=1)
                        # Advantage: expected reward − baseline
                        advantage  = expected_r - tf.squeeze(values, axis=1)
                        # Policy loss: negative expected reward weighted by advantage
                        log_probs   = tf.nn.log_softmax(logits)
                        chosen_lp   = tf.reduce_sum(
                            log_probs * tf.one_hot(actions, 3), axis=1
                        )
                        policy_loss = -tf.reduce_mean(
                            tf.stop_gradient(advantage) * chosen_lp
                        )
                        # Value loss: MSE of baseline vs expected reward
                        value_loss = tf.reduce_mean(
                            tf.square(expected_r - tf.squeeze(values, axis=1))
                        )
                        # Entropy bonus (encourages exploration)
                        entropy    = -tf.reduce_mean(
                            tf.reduce_sum(probs * log_probs, axis=1)
                        )
                        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, model.trainable_variables)
                    )
                    epoch_loss += float(loss)

            # OOF predictions
            logits, _ = model(X_oof.astype(np.float32), training=False)
            import tensorflow as _tf
            proba = _tf.nn.softmax(logits).numpy()
            oof_frames.append(_proba_to_canonical(proba, y_true=y_oof))

        artifact = {"model": model, "scaler": scaler}
        oof = pd.concat(oof_frames) if oof_frames else None
        return FittedModel(spec=ModelSpec(family=self.name), artifact=artifact, oof_preds=oof)

    def predict(self, fitted: FittedModel, X: np.ndarray) -> pd.DataFrame:
        import tensorflow as tf
        art    = fitted.artifact
        X_sc   = art["scaler"].transform(_flatten(X)).astype(np.float32)
        logits, _ = art["model"](X_sc, training=False)
        proba  = tf.nn.softmax(logits).numpy()
        df     = _proba_to_canonical(proba)
        validate_canonical(df, plugin_name=self.name)
        return df


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def _proba_to_canonical(
    proba: np.ndarray,
    ts: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    idx  = proba.argmax(axis=1)
    conf = proba.max(axis=1).astype("float64")
    df = pd.DataFrame({
        CANONICAL_TS:         pd.to_datetime(ts, utc=True) if ts is not None else pd.NaT,
        CANONICAL_DIRECTION:  [_IDX_TO_DIR[i] for i in idx],
        CANONICAL_CONFIDENCE: conf,
    })
    if y_true is not None:
        df["direction_true"] = [_DIR_FROM_LABEL.get(int(y), "flat") for y in y_true]
    return df
