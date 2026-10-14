"""End-to-end training pipeline for the detection-2 model.

Run as: python -m src.training.trainer --start=YYYY-MM-DD --end=YYYY-MM-DD
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common.vm_config import get_vm_config
from src.training.config import MLConfig
from src.training.data_pipeline import DataPipeline
from src.training.evaluate import evaluate_model, save_metrics
from src.training.labels import generate_labels

logger = logging.getLogger("vtech.training.trainer")

# 3-class softmax: raw labels from generate_labels used directly.
# {0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP} — all three classes receive full gradient.
CLASS_NAMES = ["BIG_DOWN", "NO_SIGNAL", "BIG_UP"]
NO_SIGNAL = 1  # raw label value; used only for signal-count logging


def _epoch_ig_groups(
    model: "tf.keras.Model",
    X_cls: "np.ndarray",
    cls_idx: int,
    feature_cols: list[str],
    n_steps: int = 8,
    max_bars: int = 128,
) -> list[tuple[str, float]]:
    """Fast per-epoch IG group importance — small val sample, coarse steps.

    Called from _DirectionalF1Callback.on_epoch_end after each epoch.
    Returns top-4 feature groups by mean absolute IG for the given class.
    """
    if len(X_cls) == 0:
        return []
    X_s = X_cls[:max_bars]
    seq_len, n_feats = int(X_s.shape[1]), int(X_s.shape[2])
    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps, dtype=np.float32)
    x_batch = tf.constant(X_s, dtype=tf.float32)
    x_interp = (
        x_batch[:, tf.newaxis, :, :]
        * tf.constant(alphas)[tf.newaxis, :, tf.newaxis, tf.newaxis]
    )
    x_flat = tf.reshape(x_interp, (-1, seq_len, n_feats))
    with tf.GradientTape() as tape:
        tape.watch(x_flat)
        preds = model(x_flat, training=False)
        target = tf.reduce_sum(preds[:, cls_idx])
    grads = tape.gradient(target, x_flat)
    if grads is None:
        return []
    grads = tf.reshape(grads, (len(X_s), n_steps, seq_len, n_feats))
    abs_feat = np.mean(
        np.abs((x_batch * tf.reduce_mean(grads, axis=1)).numpy()), axis=(0, 1)
    )
    feat_ranked = sorted(zip(feature_cols, abs_feat.tolist()), key=lambda x: -x[1])
    return feat_ranked[:5]


class _DirectionalF1Callback(tf.keras.callbacks.Callback):
    """Computes val_directional_f1 = (F1_BIG_DOWN + F1_BIG_UP) / 2 per epoch.

    Must appear *before* EarlyStopping in the callbacks list so the metric is
    available in logs when EarlyStopping.on_epoch_end runs.
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, feature_cols: list[str], use_ig: bool = True, group: str = "") -> None:
        super().__init__()
        self._X = X_val
        self._y = y_val
        self._group = group
        self._feature_cols = feature_cols
        self._use_ig = use_ig

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        import time as _time
        from sklearn.metrics import f1_score as _f1
        _cb_t0 = _time.perf_counter()

        _pred_t0 = _time.perf_counter()
        y_pred = np.argmax(
            self.model.predict(self._X, batch_size=2048, verbose=0), axis=1
        )
        _pred_s = _time.perf_counter() - _pred_t0

        f1_bd = _f1(self._y, y_pred, labels=[0], average="macro", zero_division=0)
        f1_bu = _f1(self._y, y_pred, labels=[2], average="macro", zero_division=0)
        dir_f1 = float((f1_bd + f1_bu) / 2.0)
        if logs is not None:
            logs["val_directional_f1"] = dir_f1
        logger.info(
            "directional_f1  group=%s  epoch=%d  f1_bd=%.4f  f1_bu=%.4f  val_directional_f1=%.4f",
            self._group, epoch + 1, f1_bd, f1_bu, dir_f1,
        )
        _ig_s = 0.0
        if self._use_ig:
            try:
                _ig_t0 = _time.perf_counter()
                grps_bu = _epoch_ig_groups(self.model, self._X[self._y == 2], 2, self._feature_cols)
                grps_bd = _epoch_ig_groups(self.model, self._X[self._y == 0], 0, self._feature_cols)
                _ig_s = _time.perf_counter() - _ig_t0
                bu_str = ",".join(f"{g}:{v:.4f}" for g, v in grps_bu)
                bd_str = ",".join(f"{g}:{v:.4f}" for g, v in grps_bd)
                logger.info(
                    "ig_epoch  epoch=%d  BU_top=%s  BD_top=%s",
                    epoch + 1, bu_str, bd_str,
                )
            except Exception:
                pass
        _cb_total_s = _time.perf_counter() - _cb_t0
        logger.info(
            "callback_timing  epoch=%d  cb_predict_s=%.2f  cb_ig_s=%.2f  cb_total_s=%.2f  val_n=%d",
            epoch + 1, _pred_s, _ig_s, _cb_total_s, len(self._X),
        )


def _mlf_p08(p_at_tau: list, direction: str) -> tuple[float, int]:
    """Return (precision, n_signals) at tau=0.80 for the given direction."""
    entry = next(
        (p for p in p_at_tau if abs(p.get("tau", 0) - 0.8) < 0.01 and p.get("direction") == direction),
        None,
    )
    if entry is None:
        return 0.0, 0
    prec = entry.get("precision", 0.0)
    return (float(prec) if prec == prec else 0.0), int(entry.get("n_signals", 0))


class _MLflowEpochCallback(tf.keras.callbacks.Callback):
    """Logs per-epoch training metrics to the active MLflow run.

    No-op when MLflow is unavailable or no run is active. Must be placed
    *after* _DirectionalF1Callback in the callbacks list so that
    ``val_directional_f1`` is already populated in ``logs``.
    """

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not logs:
            return
        try:
            import mlflow as _mlf
            if _mlf.active_run():
                for k, v in logs.items():
                    try:
                        _mlf.log_metric(k, float(v), step=epoch)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass


FEATURE_GROUP_PREFIXES = {
    "book_pressure": "bp_",
    "order_flow": "of_",
    "trade_location": "tl_",
    "same_side": "ss_",
    "large_trade_cvd": "lt_",
    "trade_arrival": "ta_",
    "microstructure": "ms_",
    "realized_vol": "rv_",
    "sub_bar_dynamics": "sd_",
    "volume_profile": "vp_",
    "volume_footprint": "vf_",
    "iv_surface": "iv_",
    "dealer_gex": "gx_",
    "daily_context": "dc_",
    "wavelets": "wv_",
    "candle_structure": "cs_",
    "time_context": "tc_",
    "vwap": "vw_",
    "cross_asset": "ca_",
    "economic_calendar": "ec_",
    "hawkes_clustering": "hp_",
    "higher_timeframe": "ht_",
}


def _group_permutation_importance(
    model: "tf.keras.Model",
    X_test: "np.ndarray",
    y_test: "np.ndarray",
    feature_cols: list[str],
    tau: float = 0.80,
) -> dict:
    """Zero out each feature group in X_test and measure drop in directional precision@tau.

    Uses data already in memory — no disk I/O. Called after evaluate_model() in step 9b.
    Returns a dict suitable for JSON serialisation that is merged into training_result.json.
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}

    def _prec_at_tau(probs: np.ndarray, y: np.ndarray, cls_idx: int) -> tuple[float, int]:
        mask = probs[:, cls_idx] >= tau
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0
        return float((y[mask] == cls_idx).mean()), n

    base_probs = model.predict(X_test, batch_size=1024, verbose=0)
    base_up, base_n_up = _prec_at_tau(base_probs, y_test, 2)
    base_dn, base_n_dn = _prec_at_tau(base_probs, y_test, 0)

    def _run_group(args: tuple[str, str]) -> dict | None:
        group_name, prefix = args
        idxs = [feat_to_idx[f] for f in feature_cols if f.startswith(prefix) and f in feat_to_idx]
        if not idxs:
            return None
        X_perm = X_test.copy()
        X_perm[:, :, idxs] = 0.0
        probs = model.predict(X_perm, batch_size=1024, verbose=0)
        up, n_up = _prec_at_tau(probs, y_test, 2)
        dn, n_dn = _prec_at_tau(probs, y_test, 0)
        logger.info(
            "perm_importance_group",
            extra={"group": group_name, "delta_prec_up": round(base_up - up, 4),
                   "delta_prec_dn": round(base_dn - dn, 4)},
        )
        return {
            "group": group_name,
            "n_feats": len(idxs),
            "delta_prec_up": round(base_up - up, 4),
            "delta_prec_dn": round(base_dn - dn, 4),
            "prec_up_zeroed": round(up, 4),
            "prec_dn_zeroed": round(dn, 4),
            "n_up_zeroed": n_up,
            "n_dn_zeroed": n_dn,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = list(pool.map(_run_group, sorted(FEATURE_GROUP_PREFIXES.items())))
    groups: list[dict] = [r for r in raw if r is not None]

    groups.sort(key=lambda r: -(r["delta_prec_up"] + r["delta_prec_dn"]))
    return {
        "baseline_prec_up": round(base_up, 4),
        "baseline_n_up": base_n_up,
        "baseline_prec_dn": round(base_dn, 4),
        "baseline_n_dn": base_n_dn,
        "tau": tau,
        "groups": groups,
    }


def _integrated_gradients_importance(
    model: "tf.keras.Model",
    X_test: "np.ndarray",
    y_test: "np.ndarray",
    feature_cols: list[str],
    n_steps: int = 64,
) -> dict:
    """Per-feature Integrated Gradients importance on the test set.

    Computes IG w.r.t. BIG_UP logit on true BIG_UP bars (y==2) and
    w.r.t. BIG_DOWN logit on true BIG_DOWN bars (y==0).
    Uses in-memory X_test — no disk I/O.  Returned dict is merged into
    training_result.json automatically via save_metrics().
    """
    seq_len = X_test.shape[1]
    n_feats = X_test.shape[2]

    try:
        # get_vm_config() is lru_cache'd — clear so we get a fresh /proc/meminfo
        # reading *after* training data (X_train/val/test) is already in memory.
        get_vm_config.cache_clear()
        _vm = get_vm_config()
        _usable_gb = max(0.0, _vm.available_ram_gb - 4.0) * 0.40   # 4 GB overhead; 40% of the rest
        bytes_per_seq = n_steps * seq_len * n_feats * 4 * 8
        bsz = max(32, min(3500, int(_usable_gb * 1e9 / bytes_per_seq)))
    except Exception:
        bsz = 32

    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps, dtype=np.float32)

    def _ig_raw_chunk(X_cls: np.ndarray, cls_idx: int) -> np.ndarray:
        """Run GradientTape on a chunk of bars; returns raw (N, S, F) IG array."""
        if len(X_cls) == 0:
            return np.zeros((0, seq_len, n_feats), dtype=np.float32)
        all_ig: list[np.ndarray] = []
        for start in range(0, len(X_cls), bsz):
            x_np = X_cls[start : start + bsz]
            actual_b = len(x_np)
            x_batch = tf.constant(x_np, dtype=tf.float32)
            x_interp = (
                x_batch[:, tf.newaxis, :, :]
                * tf.constant(alphas)[tf.newaxis, :, tf.newaxis, tf.newaxis]
            )
            x_flat = tf.reshape(x_interp, (-1, seq_len, n_feats))
            with tf.GradientTape() as tape:
                tape.watch(x_flat)
                preds = model(x_flat, training=False)
                target = tf.reduce_sum(preds[:, cls_idx])
            grads = tape.gradient(target, x_flat)
            grads = tf.reshape(grads, (actual_b, n_steps, seq_len, n_feats))
            avg_grads = tf.reduce_mean(grads, axis=1)
            all_ig.append((x_batch * avg_grads).numpy())
        return np.concatenate(all_ig, axis=0)  # (N, S, F)

    def _ig_aggregate(ig_all: np.ndarray, X_cls: np.ndarray, cls_idx: int) -> dict:
        """Compute per-feature stats, groups, and top-features from raw IG array."""
        if ig_all.shape[0] == 0:
            return {}
        abs_feat = np.mean(np.abs(ig_all), axis=(0, 1))  # (n_feats,)
        abs_ts   = np.mean(np.abs(ig_all), axis=(0, 2))  # (seq_len,)

        feat_to_idx = {f: i for i, f in enumerate(feature_cols)}
        feat_ranked = sorted(zip(feature_cols, abs_feat.tolist()), key=lambda x: -x[1])

        groups: list[dict] = []
        for grp, prefix in sorted(FEATURE_GROUP_PREFIXES.items()):
            idxs = [feat_to_idx[f] for f in feature_cols
                    if f.startswith(prefix) and f in feat_to_idx]
            if not idxs:
                continue
            groups.append({
                "group": grp,
                "n_feats": len(idxs),
                "mean_abs_ig": round(float(abs_feat[idxs].mean()), 6),
            })
        groups.sort(key=lambda g: -g["mean_abs_ig"])

        for rank, (feat, val) in enumerate(feat_ranked[:10], 1):
            logger.info("ig_top_feature", extra={
                "class": "BIG_UP" if cls_idx == 2 else "BIG_DOWN",
                "rank": rank, "feature": feat, "mean_abs_ig": round(float(val), 6),
            })

        return {
            "n_bars": int(len(X_cls)),
            "top_features": [
                {"feature": f, "mean_abs_ig": round(float(v), 6)}
                for f, v in feat_ranked[:30]
            ],
            "groups": groups,
            "timestep_profile": [round(float(v), 6) for v in abs_ts.tolist()],
        }

    mask_up = y_test == 2
    mask_dn = y_test == 0
    X_up = X_test[mask_up]
    X_dn = X_test[mask_dn]
    # Split each class in half so all 4 chunks run in parallel.
    mid_up = max(1, len(X_up) // 2)
    mid_dn = max(1, len(X_dn) // 2)
    logger.info("ig_importance_start", extra={
        "n_up": int(len(X_up)), "n_dn": int(len(X_dn)),
        "n_steps": n_steps, "batch_size": bsz, "workers": 4,
    })
    result: dict = {"n_steps": n_steps, "batch_size": bsz}
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_up1 = pool.submit(_ig_raw_chunk, X_up[:mid_up],   2)
        fut_up2 = pool.submit(_ig_raw_chunk, X_up[mid_up:],   2)
        fut_dn1 = pool.submit(_ig_raw_chunk, X_dn[:mid_dn],   0)
        fut_dn2 = pool.submit(_ig_raw_chunk, X_dn[mid_dn:],   0)
        ig_up = np.concatenate([fut_up1.result(), fut_up2.result()], axis=0)
        ig_dn = np.concatenate([fut_dn1.result(), fut_dn2.result()], axis=0)
    result["BIG_UP"]   = _ig_aggregate(ig_up, X_up, 2)
    result["BIG_DOWN"] = _ig_aggregate(ig_dn, X_dn, 0)
    return result


class Trainer:
    """Train a detection-2 attention-LSTM end-to-end."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config

    def run(
        self,
        start_date: str,
        end_date: str,
        save_dir: str | None = None,
        data_dir: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        eval_only: bool = False,
    ) -> dict:
        """Execute full pipeline and return metrics dict.

        When *test_start* / *test_end* are supplied the data in
        [start_date, end_date] is used as train+val (ratio split) and the
        separate [test_start, test_end] window is used as held-out test.
        This enables explicit walk-forward fold evaluation.

        When *eval_only* is True, steps 7-8 (build + fit) are skipped and the
        existing ``save_dir/model.keras`` checkpoint is loaded for evaluation
        only. Used to recover test metrics + IG when a previous training run
        crashed in step 9.
        """
        import os
        from src.common.repro import set_global_seed, log_experiment, hash_dict
        from src.common.data_manifest import schema_for_dataframe

        seed = set_global_seed()
        t0 = time.time()
        save_path = Path(save_dir or "checkpoints/latest")
        save_path.mkdir(parents=True, exist_ok=True)

        # ── MLflow: open tracking run ─────────────────────────────────────────
        _mlf_active = False
        if self.config.mlflow.enabled:
            try:
                import mlflow as _mlf
                _mlf.set_tracking_uri(self.config.mlflow.tracking_uri)
                _mlf.set_experiment(self.config.mlflow.experiment_name)
                _mlf.start_run(run_name=save_path.name)
                _mlf_active = True
                mc = self.config.model
                tc = self.config.training
                lc = self.config.labels
                dc = self.config.data
                _mlf.log_params({
                    "train_start": start_date,
                    "train_end": end_date,
                    "test_start": test_start or "",
                    "test_end": test_end or "",
                    "model.lstm_units": str(mc.lstm_units),
                    "model.attention_heads": mc.attention_heads,
                    "model.dense_units": str(mc.dense_units),
                    "model.dense_dropout": mc.dense_dropout,
                    "model.use_attention": mc.attention_heads > 0,
                    "training.batch_size": tc.batch_size,
                    "training.learning_rate": tc.learning_rate,
                    "training.early_stop_patience": tc.early_stop_patience,
                    "training.use_balanced_sampling": tc.use_balanced_sampling,
                    "training.use_focal_loss": tc.use_focal_loss,
                    "training.min_steps_per_epoch": tc.min_steps_per_epoch,
                    "labels.big_move_threshold": lc.big_move_threshold,
                    "labels.mae_tolerance": lc.mae_tolerance,
                    "labels.forward_windows": str(lc.forward_windows),
                    "data.scaler_type": dc.scaler_type,
                    "data.training_start": dc.training_start,
                    "data.training_end": dc.training_end,
                })
                logger.info("mlflow_run_started  run_name=%s  uri=%s", save_path.name, self.config.mlflow.tracking_uri)
            except Exception as _mlf_err:
                logger.warning("mlflow_init_failed  err=%s", _mlf_err)
                _mlf_active = False

        data_dir = data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")
        pipeline = DataPipeline(data_dir, self.config.data)

        # 1. Load features
        logger.info("step", extra={"n": 1, "action": "load_features"})
        features_df = pipeline.load_features(start_date, end_date)
        if test_start and test_end:
            logger.info("walk_forward_mode", extra={"train_end": end_date, "test_start": test_start, "test_end": test_end})
            test_df_raw = pipeline.load_features(test_start, test_end)
        else:
            test_df_raw = None
        feature_cols = [c for c in features_df.columns if not c.startswith("label") and not c.startswith("_")]

        enabled = self.config.features.enabled_groups
        disabled_prefixes = [
            FEATURE_GROUP_PREFIXES[grp]
            for grp, on in enabled.items()
            if not on and grp in FEATURE_GROUP_PREFIXES
        ]
        if disabled_prefixes:
            feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in disabled_prefixes)]
        disabled_feats = set(self.config.features.disabled_features or [])
        if disabled_feats:
            unknown = disabled_feats - set(feature_cols)
            if unknown:
                logger.warning("disabled_features_not_found", extra={"cols": sorted(unknown)})
            feature_cols = [c for c in feature_cols if c not in disabled_feats]
        if not feature_cols:
            raise ValueError("All feature groups disabled — nothing to train on")

        if _mlf_active:
            try:
                import mlflow as _mlf
                _mlf.log_param("n_features", len(feature_cols))
            except Exception:
                pass

        # 2. Generate detection-2 labels
        logger.info("step", extra={"n": 2, "action": "generate_labels"})
        if "_close" not in features_df.columns:
            raise ValueError("Feature Parquet missing '_close' column.")
        prices = features_df["_close"]
        labels_df = generate_labels(prices, self.config.labels)

        full_df = pd.concat([features_df[feature_cols], labels_df], axis=1).dropna(subset=["label"])

        # Log raw class distribution: NO_SIGNAL count is informational only
        dist = full_df["label"].value_counts().sort_index()
        names = {0: "BIG_DOWN", 1: "NO_SIGNAL", 2: "BIG_UP"}
        for cls_id, count in dist.items():
            logger.info("class_dist", extra={
                "class": names.get(int(cls_id), str(cls_id)),
                "count": int(count),
                "pct": f"{count / len(full_df) * 100:.1f}%",
            })

        logger.info("samples", extra={"n": len(full_df), "features": len(feature_cols)})

        # 3. Temporal split
        logger.info("step", extra={"n": 3, "action": "split"})
        if test_df_raw is not None:
            # Walk-forward: train+val from full_df (ratio split), test from separate window
            test_features = [c for c in test_df_raw.columns if not c.startswith("label") and not c.startswith("_")]
            enabled_test = [c for c in test_features if c in feature_cols]
            test_labels_raw = generate_labels(test_df_raw["_close"], self.config.labels)
            test_full = pd.concat([test_df_raw[enabled_test], test_labels_raw], axis=1).dropna(subset=["label"])
            # Use only first 85% of full_df as train+val (reserve nothing as internal test)
            n = len(full_df)
            tv_end = int(n * (self.config.data.train_ratio + self.config.data.val_ratio))
            tv_df = full_df.iloc[:tv_end]
            tv_n = len(tv_df)
            tv_train_end = int(tv_n * (self.config.data.train_ratio / (self.config.data.train_ratio + self.config.data.val_ratio)))
            train_df = tv_df.iloc[:tv_train_end].copy()
            val_df   = tv_df.iloc[tv_train_end:].copy()
            test_df  = test_full.copy()
        else:
            train_df, val_df, test_df = [s.copy() for s in pipeline.split_temporal(full_df)]

        # 4. Preprocess — outlier clipping + scaling
        logger.info("step", extra={"n": 4, "action": "preprocess"})
        train_mean = train_df[feature_cols].mean()
        train_std = train_df[feature_cols].std().replace(0, 1)
        z_threshold = 5.0
        for df_part in [train_df, val_df, test_df]:
            clipped = df_part[feature_cols].clip(
                lower=(train_mean - z_threshold * train_std),
                upper=(train_mean + z_threshold * train_std),
                axis=1,
            )
            df_part[feature_cols] = clipped

        scaler_type = self.config.data.scaler_type
        pipeline.fit_scaler(train_df, feature_cols, scaler_type=scaler_type)
        train_s = pipeline.transform(train_df)
        val_s = pipeline.transform(val_df)
        test_s = pipeline.transform(test_df)

        for df_part in [train_s, val_s, test_s]:
            df_part[feature_cols] = df_part[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 5. Sequences
        logger.info("step", extra={"n": 5, "action": "sequences"})
        seq_len = self.config.model.sequence_length
        # In eval-only mode skip building X_train/X_val — they are 12+ GB and
        # never used; only X_test is needed for evaluation and IG.
        if not eval_only:
            X_train, y_train_raw = DataPipeline.create_sequences(
                train_s[feature_cols].values, train_s["label"].values, seq_len)
            X_val, y_val_raw = DataPipeline.create_sequences(
                val_s[feature_cols].values, val_s["label"].values, seq_len)
        else:
            X_train = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
            y_train_raw = np.empty(0, dtype=np.int32)
            X_val = np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
            y_val_raw = np.empty(0, dtype=np.int32)
        X_test, y_test_raw = DataPipeline.create_sequences(
            test_s[feature_cols].values, test_s["label"].values, seq_len)
        logger.info("sequence_shapes", extra={
            "train": list(X_train.shape), "val": list(X_val.shape), "test": list(X_test.shape),
        })
        if not eval_only and X_train.shape[0] == 0:
            raise ValueError(f"No training sequences — need at least {seq_len} rows in training split")

        # 6. 3-class labels — raw {0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP} used directly.
        # Inverse-frequency class weights give BIG_DOWN and BIG_UP 15x more gradient
        # pressure than NO_SIGNAL, preventing the majority-class collapse.
        y_train = y_train_raw.astype(np.int32)
        y_val   = y_val_raw.astype(np.int32)
        y_test  = y_test_raw.astype(np.int32)

        w = self.config.model.class_weight_directional
        class_weight = {0: w, 1: 1.0, 2: w}  # BIG_DOWN=w, NO_SIGNAL=1, BIG_UP=w

        n_signal_train = int(np.sum(y_train_raw != NO_SIGNAL))
        n_signal_val   = int(np.sum(y_val_raw   != NO_SIGNAL))
        n_signal_test  = int(np.sum(y_test_raw  != NO_SIGNAL))
        logger.info("signal_universe", extra={
            "train_signal": n_signal_train,
            "train_total": int(len(y_train_raw)),
            "train_signal_pct": f"{100 * n_signal_train / max(len(y_train_raw), 1):.2f}%",
            "test_signal": n_signal_test,
            "test_total": int(len(y_test_raw)),
        })
        if n_signal_train == 0 and not eval_only:
            raise ValueError(
                "No directional signals in training split — try lowering "
                "big_move_threshold or raising mae_tolerance"
            )

        # 7. Build model (2-class softmax)
        if not eval_only:
            logger.info("step", extra={"n": 7, "action": "build_model"})
            model = self._build_model((X_train.shape[1], X_train.shape[2]))
        else:
            logger.info("step", extra={"n": 7, "action": "skip_build_model_eval_only"})
            model = None  # will be loaded from best_ckpt in step 9

        # 8. Train (skipped in eval-only mode)
        if eval_only:
            logger.info("step", extra={"n": 8, "action": "skip_train_eval_only"})
            history = None
        else:
            logger.info("step", extra={"n": 8, "action": "train"})
            from tensorflow.keras.callbacks import (
                EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler,
            )
            import math as _math

            _MONITOR = "val_directional_f1"
            _tcfg = self.config.training

            # ── LR schedule: warmup + cosine decay OR legacy plateau ──────────
            _lr_schedule = getattr(_tcfg, "lr_schedule", "plateau")
            _warmup_eps  = getattr(_tcfg, "lr_warmup_epochs", 5)
            _peak_lr     = _tcfg.learning_rate
            _min_lr      = _tcfg.min_lr
            _total_eps   = _tcfg.epochs

            if _lr_schedule == "cosine":
                # Warmup: linear ramp from min_lr → peak_lr over warmup_epochs.
                # Decay: cosine anneal from peak_lr → min_lr over remaining epochs.
                def _cosine_lr_fn(epoch: int) -> float:
                    if epoch < _warmup_eps:
                        # Linear warmup (epoch is 0-indexed here)
                        return _min_lr + (_peak_lr - _min_lr) * (epoch + 1) / _warmup_eps
                    # Cosine decay phase
                    decay_eps = max(_total_eps - _warmup_eps, 1)
                    progress  = (epoch - _warmup_eps) / decay_eps   # 0.0 → 1.0
                    cosine    = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                    return _min_lr + (_peak_lr - _min_lr) * cosine

                _lr_callback = LearningRateScheduler(
                    _cosine_lr_fn, verbose=0
                )
                logger.info(
                    "lr_schedule  mode=cosine  warmup_epochs=%d  peak_lr=%.2e  min_lr=%.2e  total_epochs=%d",
                    _warmup_eps, _peak_lr, _min_lr, _total_eps,
                )
            else:
                _lr_callback = ReduceLROnPlateau(
                    monitor=_MONITOR,
                    mode="max",
                    patience=_tcfg.lr_reduce_patience,
                    factor=_tcfg.lr_reduce_factor,
                    min_lr=_min_lr,
                )
                logger.info("lr_schedule  mode=plateau")

            # Ablation mode: exactly one group enabled → tag log lines with its name.
            # Normal training (all groups on): empty string → backward-compatible log format.
            _enabled_groups = [g for g, on in self.config.features.enabled_groups.items() if on]
            _active_group = _enabled_groups[0] if len(_enabled_groups) == 1 else ""

            callbacks = [
                # Must be first: computes val_directional_f1 before EarlyStopping reads logs
                _DirectionalF1Callback(X_val, y_val, feature_cols, use_ig=_tcfg.use_ig, group=_active_group),
                # Second: log all epoch metrics (including val_directional_f1) to MLflow
                _MLflowEpochCallback(),
                EarlyStopping(
                    monitor=_MONITOR,
                    mode="max",
                    patience=_tcfg.early_stop_patience,
                    min_delta=_tcfg.early_stop_min_delta,
                    restore_best_weights=False,  # v013: prevent locking in epoch-1 collapse when monitor never improves; ModelCheckpoint(save_best_only=True) still persists best weights to disk for evaluation
                ),
                _lr_callback,
                ModelCheckpoint(
                    str(save_path / "model.keras"),
                    save_best_only=True,
                    monitor=_MONITOR,
                    mode="max",
                ),
            ]

            # Trim X_train to a batch-size multiple so XLA compiles exactly one kernel shape.
            # train_sequences: use config cap if set, otherwise calculate from data length.
            _bs = self.config.training.batch_size
            _cap = self.config.training.train_sequences
            if _cap is not None:
                n_trim = (min(len(X_train), _cap) // _bs) * _bs
            else:
                n_trim = (len(X_train) // _bs) * _bs
            if n_trim < len(X_train):
                logger.info(
                    "trim_train_sequences  original=%d  trimmed=%d  dropped=%d",
                    len(X_train), n_trim, len(X_train) - n_trim,
                )

            use_balanced = getattr(self.config.training, "use_balanced_sampling", False)
            if use_balanced:
                # v013: Balanced sampling — per_class_count = min(n_BD, n_BU). NS is
                # downsampled to the same count via tf.data.sample_from_datasets with
                # weights=[1/3, 1/3, 1/3]. Causality preserved: each sequence is an
                # already-closed 128-bar look-back window; only sequence-level ordering
                # within the train pool changes.
                import tensorflow as tf

                idx_bd = np.where(y_train == 0)[0]
                idx_ns = np.where(y_train == 1)[0]
                idx_bu = np.where(y_train == 2)[0]
                per_class = int(min(len(idx_bd), len(idx_bu)))
                if per_class < _bs // 3:
                    raise ValueError(
                        f"Balanced sampling needs per_class >= batch_size/3 ({_bs//3}); "
                        f"got per_class={per_class} (n_BD={len(idx_bd)}, n_BU={len(idx_bu)})"
                    )
                logger.info("balanced_sampling", extra={
                    "per_class": per_class, "total": 3 * per_class,
                    "n_bd": int(len(idx_bd)), "n_ns": int(len(idx_ns)), "n_bu": int(len(idx_bu)),
                })

                # Cap idx_ns to per_class * 3 to avoid creating a huge
                # X_train[idx_ns] copy (NS is majority class; the full copy
                # can exceed 10 GB on 31 GB VMs while X_train is also in memory).
                # steps_per_epoch = (3*per_class)//batch_size is unchanged, so the
                # number of NS samples consumed per epoch is identical to before.
                # Sampling weights remain 1/3 each so class balance is unchanged;
                # we simply limit which NS sequences are in the shuffle pool.
                _ns_orig = len(idx_ns)
                if _ns_orig > per_class * 3:
                    rng_ns = np.random.default_rng(seed)
                    idx_ns = rng_ns.choice(idx_ns, size=per_class * 3, replace=False)
                    logger.info(
                        "ns_pool_capped  original=%d  capped=%d  per_class=%d",
                        _ns_orig, len(idx_ns), per_class,
                    )

                def _make_ds(indices: np.ndarray) -> "tf.data.Dataset":
                    X_sub = X_train[indices].astype(np.float32, copy=False)  # noqa: F821 - closure; deleted after fit
                    y_sub = y_train[indices].astype(np.int32, copy=False)  # noqa: F821 - closure; deleted after fit
                    return (
                        tf.data.Dataset.from_tensor_slices((X_sub, y_sub))
                        .shuffle(buffer_size=len(indices), seed=seed, reshuffle_each_iteration=True)
                        .repeat()
                    )

                ds_train = tf.data.Dataset.sample_from_datasets(
                    [_make_ds(idx_bd), _make_ds(idx_ns), _make_ds(idx_bu)],
                    weights=[1.0 / 3, 1.0 / 3, 1.0 / 3],
                    seed=seed,
                ).batch(_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

                # Floor at min_steps_per_epoch so each epoch has enough gradient updates
                # for the LR scheduler and early stopping to function when per_class is small.
                _min_steps = getattr(self.config.training, "min_steps_per_epoch", 50)
                steps_per_epoch = max((3 * per_class) // _bs, _min_steps)
                ds_val = (
                    tf.data.Dataset.from_tensor_slices(
                        (X_val.astype(np.float32, copy=False), y_val.astype(np.int32, copy=False))
                    ).batch(_bs).prefetch(tf.data.AUTOTUNE)
                )

                history = model.fit(
                    ds_train,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=ds_val,
                    epochs=self.config.training.epochs,
                    callbacks=callbacks,
                    verbose=1,
                )
            else:
                history = model.fit(
                    X_train[:n_trim], y_train[:n_trim],
                    class_weight=class_weight,
                    validation_data=(X_val, y_val),
                    epochs=self.config.training.epochs,
                    batch_size=_bs,
                    callbacks=callbacks,
                    verbose=1,
                )

        # 9. Evaluate on full test set (all bars); log signal counts separately
        # v013: EarlyStopping uses restore_best_weights=False so `model` holds final-epoch
        # weights here. Reload best-by-val_directional_f1 checkpoint that ModelCheckpoint
        # wrote during fit, so evaluation/IG reflect the best model, not the last one.

        # Free training/validation arrays + TF datasets before post-training
        # analysis — X_train alone is ~12 GB; keeping it through perm-importance
        # (which calls X_test.copy() per group) pushes past 31 GB VM headroom.
        _n_train_seq = int(X_train.shape[0]) if X_train is not None and X_train.size > 0 else 0
        import gc as _gc
        try:
            del X_train, y_train, X_val, y_val
        except NameError:
            pass
        try:
            del ds_train, ds_val
        except NameError:
            pass
        _gc.collect()

        logger.info("step", extra={"n": 9, "action": "evaluate"})
        best_ckpt = save_path / "model.keras"
        if eval_only and not best_ckpt.exists():
            raise FileNotFoundError(
                f"eval_only=True but no checkpoint at {best_ckpt}. "
                "Run training first or point --save-dir at a directory with model.keras."
            )
        if best_ckpt.exists():
            from src.models.attention_lstm import AttentionLayer, focal_loss
            from src.models.keras_layers import load_keras_model
            # AttentionLayer is a custom layer (not @register_keras_serializable),
            # so it must be passed explicitly. focal_loss is only needed when the
            # checkpoint was saved with compile=True under focal loss; we load
            # with compile=False so it's optional, but include for safety.
            custom_objects: dict = {"AttentionLayer": AttentionLayer}
            if getattr(self.config.training, "use_focal_loss", False):
                custom_objects["focal_loss_g2.0"] = focal_loss(2.0)
            model = load_keras_model(str(best_ckpt), custom_objects=custom_objects, compile=False)
            # Recompile for evaluate_model() — only needs loss+metrics, not optimizer state.
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info("loaded_best_checkpoint", extra={"path": str(best_ckpt)})
        metrics = evaluate_model(model, X_test, y_test, class_names=CLASS_NAMES)
        metrics["training_time_seconds"] = time.time() - t0
        if history is not None:
            metrics["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        else:
            metrics["history"] = {}
            metrics["eval_only"] = True
        metrics["n_signal_test"] = n_signal_test
        metrics["n_signal_train"] = n_signal_train

        # 9b. Group permutation importance — in-memory, no disk I/O
        logger.info("step", extra={"n": "9b", "action": "permutation_importance"})
        metrics["feature_group_importance"] = _group_permutation_importance(
            model, X_test, y_test, feature_cols
        )

        # 9c. Integrated Gradients — per-feature importance, ~30s in-memory
        logger.info("step", extra={"n": "9c", "action": "integrated_gradients"})
        metrics["integrated_gradients"] = _integrated_gradients_importance(
            model, X_test, y_test, feature_cols
        )

        # 10. Save artifacts
        # v013: do NOT re-save model.keras here — ModelCheckpoint already wrote the
        # best-by-val_directional_f1 weights during fit, and we just evaluated that
        # checkpoint. Overwriting with the in-memory model would replace best with last.
        # eval_only runs write to eval_result.json so they never overwrite training history.
        logger.info("step", extra={"n": 10, "action": "save"})
        result_filename = "eval_result.json" if eval_only else "training_result.json"
        save_metrics(metrics, save_path, filename=result_filename)
        with open(save_path / "features.json", "w") as f:
            json.dump({"features": feature_cols, "scaler": scaler_type}, f, indent=2)

        train_schema = schema_for_dataframe(features_df[feature_cols])
        with open(save_path / "feature_schema.json", "w") as f:
            json.dump(train_schema, f, indent=2)

        config_dict = {
            "model": vars(self.config.model),
            "training": vars(self.config.training),
            "data": vars(self.config.data),
            "labels": vars(self.config.labels),
            "features_enabled": self.config.features.enabled_groups,
        }
        with open(save_path / "config_snapshot.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        log_experiment({
            "model_dir": str(save_path),
            "seed": seed,
            "start_date": start_date,
            "end_date": end_date,
            "test_start": test_start or "",
            "test_end": test_end or "",
            "config_hash": hash_dict(config_dict),
            "n_features": len(feature_cols),
            "n_train_seq": _n_train_seq,
            "n_test_seq": int(X_test.shape[0]),
            "test_accuracy": float(metrics.get("test_accuracy", 0.0)),
            "test_f1_macro": float(metrics.get("test_f1_macro", 0.0)),
            "training_seconds": float(metrics.get("training_time_seconds", 0.0)),
        })

        # ── MLflow: log final metrics, precision-at-tau curves, artifacts ────
        if _mlf_active:
            try:
                import mlflow as _mlf
                f1pc = metrics.get("test_f1_per_class", {})
                crep = metrics.get("classification_report", {})
                p_at_tau = metrics.get("precision_at_tau", [])
                p08_up, n_p08_up = _mlf_p08(p_at_tau, "BIG_UP")
                p08_dn, n_p08_dn = _mlf_p08(p_at_tau, "BIG_DOWN")
                hist_vdf1 = metrics.get("history", {}).get("val_directional_f1", [])
                best_vdf1 = max(hist_vdf1) if hist_vdf1 else 0.0
                best_vdf1_ep = (int(hist_vdf1.index(max(hist_vdf1))) + 1) if hist_vdf1 else 0
                _mlf.log_metrics({
                    "test_f1_macro":       float(metrics.get("test_f1_macro", 0.0)),
                    "test_f1_bd":          float(f1pc.get("BIG_DOWN", 0.0)),
                    "test_f1_bu":          float(f1pc.get("BIG_UP", 0.0)),
                    "test_f1_ns":          float(f1pc.get("NO_SIGNAL", 0.0)),
                    "test_prec_bd":        float(crep.get("BIG_DOWN", {}).get("precision", 0.0)),
                    "test_recall_bd":      float(crep.get("BIG_DOWN", {}).get("recall", 0.0)),
                    "test_prec_bu":        float(crep.get("BIG_UP", {}).get("precision", 0.0)),
                    "test_recall_bu":      float(crep.get("BIG_UP", {}).get("recall", 0.0)),
                    "test_accuracy":       float(metrics.get("test_accuracy", 0.0)),
                    "n_signal_test":       float(n_signal_test),
                    "n_signal_train":      float(n_signal_train),
                    "best_val_dir_f1":     float(best_vdf1),
                    "best_val_dir_f1_ep":  float(best_vdf1_ep),
                    "epochs_run":          float(len(hist_vdf1)),
                    "test_p08_up":         float(p08_up),
                    "test_p08_dn":         float(p08_dn),
                    "n_signals_p08_up":    float(n_p08_up),
                    "n_signals_p08_dn":    float(n_p08_dn),
                    "training_seconds":    float(metrics.get("training_time_seconds", 0.0)),
                })
                # Precision-at-tau curves (step = int(tau*100) for axis alignment)
                for pt in p_at_tau:
                    _step = int(round(pt["tau"] * 100))
                    _prec = pt.get("precision", float("nan"))
                    _nsig = float(pt.get("n_signals", 0))
                    if pt["direction"] == "BIG_UP":
                        if _prec == _prec:  # not NaN
                            _mlf.log_metric("prec_at_tau_bu", float(_prec), step=_step)
                        _mlf.log_metric("n_signals_at_tau_bu", _nsig, step=_step)
                    else:
                        if _prec == _prec:
                            _mlf.log_metric("prec_at_tau_bd", float(_prec), step=_step)
                        _mlf.log_metric("n_signals_at_tau_bd", _nsig, step=_step)
                # Artifacts
                for _art in (result_filename, "features.json", "config_snapshot.json", "feature_schema.json"):
                    _ap = save_path / _art
                    if _ap.exists():
                        _mlf.log_artifact(str(_ap))
                # Log model with TF flavor so it appears in the Models column
                if model is not None and not eval_only:
                    import mlflow.tensorflow as _mlf_tf
                    from src.models.attention_lstm import AttentionLayer, focal_loss as _focal_loss
                    _custom_obj = {"AttentionLayer": AttentionLayer}
                    if getattr(self.config.training, "use_focal_loss", False):
                        _custom_obj["focal_loss_g2.0"] = _focal_loss(2.0)
                    _mlf_tf.log_model(
                        model,
                        artifact_path="model",
                        custom_objects=_custom_obj,
                    )
            except Exception as _mlf_err:
                logger.warning("mlflow_log_final_failed  err=%s", _mlf_err)
            finally:
                try:
                    import mlflow as _mlf
                    _mlf.end_run()
                    logger.info("mlflow_run_ended  run_name=%s", save_path.name)
                except Exception:
                    pass

        logger.info("done", extra={
            "accuracy": metrics["test_accuracy"],
            "f1_macro": metrics["test_f1_macro"],
            "seconds": metrics["training_time_seconds"],
        })
        return metrics

    def _build_model(self, input_shape: tuple[int, int]):
        from src.models.attention_lstm import build_attention_lstm

        cfg = self.config.model
        tcfg = self.config.training
        return build_attention_lstm(
            input_shape=input_shape,
            num_classes=cfg.num_classes,
            lstm_units=cfg.lstm_units,
            attention_heads=cfg.attention_heads,
            dense_units=cfg.dense_units,
            dropout=cfg.dense_dropout,
            learning_rate=tcfg.learning_rate,
            use_focal_loss=getattr(tcfg, "use_focal_loss", False),
            focal_gamma=getattr(tcfg, "focal_loss_gamma", 2.0),
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train NQ detection-2 model")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--test-start", default=None, help="Walk-forward test start date YYYY-MM-DD")
    parser.add_argument("--test-end", default=None, help="Walk-forward test end date YYYY-MM-DD")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load existing save_dir/model.keras and run evaluation + IG only")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    config = MLConfig.from_env()
    if config.training.intraop_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(config.training.intraop_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    trainer = Trainer(config)
    trainer.run(
        args.start, args.end,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        test_start=args.test_start,
        test_end=args.test_end,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
