"""LSTM + XGBoost stacking meta-learner.

Loads saved LSTM fold checkpoints, re-runs the fold's preprocessing pipeline
to reproduce the exact scaled inputs seen during LSTM training, then generates
out-of-fold (OOF) probabilities for each fold's held-out test window.

Meta-features per bar:  [scaled_last_bar_features (F), lstm_probs (3)]
                        shape: (N, F + 3),  F ≈ 154

An XGBoost meta-learner is trained on the stacked OOF dataset and evaluated
on a final held-out test window that lies beyond all fold boundaries.

trainer.py and the LSTM Trainer class are NOT imported or modified.

Run (after training fold checkpoints with launch_v015.sh):
    python -m src.training.stack_trainer \\
        --fold1-dir  data/legacy/checkpoints/model_v015_fold1 \\
        --fold1-train-start 2025-05-15 --fold1-train-end 2025-09-30 \\
        --fold1-test-start  2025-10-01 --fold1-test-end  2025-10-31 \\
        --fold2-dir  data/legacy/checkpoints/model_v015_fold2 \\
        --fold2-train-start 2025-05-15 --fold2-train-end 2025-10-31 \\
        --fold2-test-start  2025-11-01 --fold2-test-end  2025-11-28 \\
        --fold3-dir  data/legacy/checkpoints/model_v015_fold3 \\
        --fold3-train-start 2025-05-15 --fold3-train-end 2026-01-30 \\
        --fold3-test-start  2026-02-01 --fold3-test-end  2026-04-01 \\
        --final-test-start 2026-04-06 --final-test-end 2026-06-04 \\
        --save-dir data/legacy/checkpoints/stack_v001
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from xgboost import XGBClassifier

from src.training.config import MLConfig
from src.training.data_pipeline import DataPipeline
from src.training.evaluate import CLASS_NAMES, _precision_at_tau_sweep, save_metrics
from src.training.labels import generate_labels

logger = logging.getLogger("vtech.training.stack_trainer")

_FEATURE_GROUP_PREFIXES: dict[str, str] = {
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

_BD = 0
_NS = 1
_BU = 2


@dataclass
class FoldDef:
    """One walk-forward fold: LSTM checkpoint + date windows."""

    lstm_checkpoint_dir: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    name: str = ""


def _resolve_feature_cols(features_df: pd.DataFrame, config: MLConfig) -> list[str]:
    """Filter feature columns — identical logic to trainer.py and xgb_trainer.py."""
    cols = [
        c for c in features_df.columns
        if not c.startswith("label") and not c.startswith("_")
    ]
    enabled = config.features.enabled_groups
    disabled_prefixes = [
        _FEATURE_GROUP_PREFIXES[grp]
        for grp, on in enabled.items()
        if not on and grp in _FEATURE_GROUP_PREFIXES
    ]
    if disabled_prefixes:
        cols = [c for c in cols if not any(c.startswith(p) for p in disabled_prefixes)]
    disabled_feats = set(config.features.disabled_features or [])
    if disabled_feats:
        cols = [c for c in cols if c not in disabled_feats]
    return cols


def _clip_outliers_inplace(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_mean: pd.Series,
    train_std: pd.Series,
    z: float = 5.0,
) -> None:
    df[feature_cols] = df[feature_cols].clip(
        lower=(train_mean - z * train_std),
        upper=(train_mean + z * train_std),
        axis=1,
    )


def _load_lstm_model(checkpoint_dir: str | Path):
    """Load a saved Keras LSTM checkpoint.

    Imports TensorFlow lazily so the module can be imported without GPU init
    when only the meta-learner training path is exercised.
    """
    from src.models.attention_lstm import AttentionLayer
    from src.models.keras_layers import load_keras_model

    model_path = Path(checkpoint_dir) / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {model_path}\n"
            "Train the LSTM folds first with launch_v015.sh before running the "
            "stack trainer."
        )
    model = load_keras_model(
        str(model_path),
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )
    logger.info("lstm_model_loaded  path=%s", str(model_path))
    return model


def _build_fold_oof(
    fold: FoldDef,
    config: MLConfig,
    data_dir: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OOF meta-features for one fold.

    Preprocessing steps mirror trainer.py exactly:
      1. Load train window → refit scaler on train split
      2. Load test window  → apply same clip + scale
      3. Create sequences  → run LSTM inference → probs
      4. Return (X_meta, y_meta) where
             X_meta = np.hstack([X_seq[:, -1, :], lstm_probs])  shape (N, F+3)
             y_meta = y_seq                                       shape (N,)

    The scaler is refitted per fold so each fold's preprocessing exactly
    matches the scaler that was used during LSTM training for that fold.
    """
    pipeline = DataPipeline(data_dir, config.data)

    # Load train data (needed only to refit the scaler).
    logger.info(
        "fold_oof_load_train  fold=%s  train_start=%s  train_end=%s",
        fold.name,
        fold.train_start,
        fold.train_end,
    )
    train_raw = pipeline.load_features(fold.train_start, fold.train_end)
    feature_cols = _resolve_feature_cols(train_raw, config)

    train_labels = generate_labels(train_raw["_close"], config.labels)
    train_df = pd.concat(
        [train_raw[feature_cols], train_labels], axis=1
    ).dropna(subset=["label"])

    # Use the same 70%/15% ratio split to identify the training subset for scaler.
    n = len(train_df)
    tv_ratio = config.data.train_ratio + config.data.val_ratio
    tv_end = int(n * tv_ratio)
    tv_df = train_df.iloc[:tv_end]
    train_split = config.data.train_ratio / tv_ratio
    scaler_fit_df = tv_df.iloc[: int(len(tv_df) * train_split)].copy()

    # Compute outlier clip stats from the train subset.
    train_mean = scaler_fit_df[feature_cols].mean()
    train_std = scaler_fit_df[feature_cols].std().replace(0, 1)
    _clip_outliers_inplace(scaler_fit_df, feature_cols, train_mean, train_std)

    # Fit scaler on the same train subset.
    pipeline.fit_scaler(scaler_fit_df, feature_cols, scaler_type=config.data.scaler_type)

    # Load and preprocess test window.
    logger.info(
        "fold_oof_load_test  fold=%s  test_start=%s  test_end=%s",
        fold.name,
        fold.test_start,
        fold.test_end,
    )
    test_raw = pipeline.load_features(fold.test_start, fold.test_end)
    test_feature_cols = [c for c in test_raw.columns if c in feature_cols]
    test_labels = generate_labels(test_raw["_close"], config.labels)
    test_df = pd.concat(
        [test_raw[test_feature_cols], test_labels], axis=1
    ).dropna(subset=["label"]).copy()

    _clip_outliers_inplace(test_df, feature_cols, train_mean, train_std)
    test_scaled = pipeline.transform(test_df)
    test_scaled[feature_cols] = (
        test_scaled[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Build sequences.
    X_seq, y_seq = DataPipeline.create_sequences(
        test_scaled[feature_cols].values,
        test_scaled["label"].values,
        seq_len,
    )
    if X_seq.shape[0] == 0:
        logger.warning("fold_empty_sequences  fold=%s", fold.name)
        n_feats = len(feature_cols)
        return (
            np.empty((0, n_feats + 3), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    # LSTM inference.
    lstm_model = _load_lstm_model(fold.lstm_checkpoint_dir)
    lstm_probs = lstm_model.predict(X_seq, batch_size=2048, verbose=0)  # (N, 3)

    # Meta-features: last bar of each sequence (already scaled) + LSTM probs.
    X_last_bar = X_seq[:, -1, :]  # (N, F)
    X_meta = np.hstack([X_last_bar, lstm_probs]).astype(np.float32)  # (N, F+3)

    logger.info(
        "fold_oof_built  fold=%s  n_oof=%d  meta_features=%d",
        fold.name,
        X_meta.shape[0],
        X_meta.shape[1],
    )
    return X_meta, y_seq.astype(np.int32)


def _build_meta_weights(y: np.ndarray, directional_weight: float = 4.0) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    weights[y == _BD] = directional_weight
    weights[y == _BU] = directional_weight
    return weights


def _evaluate_stack(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Same evaluation schema as _evaluate_xgb() in xgb_trainer.py."""
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    labels_list = [0, 1, 2]

    accuracy = float((y_pred == y_test).mean())
    loss = float(log_loss(y_test, probs, labels=labels_list))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_pc = f1_score(
        y_test, y_pred, average=None, labels=labels_list, zero_division=0
    ).tolist()
    f1_bd = float(f1_score(y_test, y_pred, labels=[_BD], average="macro", zero_division=0))
    f1_bu = float(f1_score(y_test, y_pred, labels=[_BU], average="macro", zero_division=0))
    dir_f1 = (f1_bd + f1_bu) / 2.0
    cm = confusion_matrix(y_test, y_pred, labels=labels_list).tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        labels=labels_list,
        output_dict=True,
        zero_division=0,
    )
    logger.info(
        "evaluation  accuracy=%.4f  f1_macro=%.4f  directional_f1=%.4f",
        accuracy,
        f1_macro,
        dir_f1,
    )
    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "test_f1_macro": f1_macro,
        "test_directional_f1": dir_f1,
        "test_f1_per_class": dict(zip(CLASS_NAMES, f1_pc)),
        "confusion_matrix": cm,
        "classification_report": report,
        "precision_at_tau": _precision_at_tau_sweep(probs, y_test),
    }


class StackTrainer:
    """Train a LSTM+XGBoost stacking meta-learner."""

    def __init__(self, config: MLConfig) -> None:
        self.config = config

    def run(
        self,
        folds: list[FoldDef],
        final_test_start: str,
        final_test_end: str,
        save_dir: str | None = None,
        data_dir: str | None = None,
    ) -> dict:
        """Build OOF meta-dataset, train XGBoost meta, evaluate on final test window.

        Fold checkpoints must exist (model.keras in each fold's checkpoint dir).
        Final test window must lie beyond all fold test windows to avoid leakage.

        The last fold in *folds* is used for final-test-window LSTM inference;
        it should be the fold with the largest training window (most recent).
        """
        from src.common.repro import set_global_seed

        set_global_seed()
        t0 = time.time()
        save_path = Path(save_dir or "checkpoints/stack_latest")
        save_path.mkdir(parents=True, exist_ok=True)
        data_dir = data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")

        seq_len = self.config.model.sequence_length
        directional_w = float(self.config.model.class_weight_directional)

        # ── 1. Build OOF meta-dataset ───────────────────────────────────────
        logger.info("step  n=1  action=build_oof_meta_dataset")
        X_oof_parts: list[np.ndarray] = []
        y_oof_parts: list[np.ndarray] = []

        for fold in folds:
            X_meta, y_meta = _build_fold_oof(fold, self.config, data_dir, seq_len)
            if X_meta.shape[0] == 0:
                logger.warning("skipping_empty_fold  fold=%s", fold.name)
                continue
            X_oof_parts.append(X_meta)
            y_oof_parts.append(y_meta)

        if not X_oof_parts:
            raise RuntimeError(
                "No OOF data was generated. Ensure LSTM fold checkpoints exist "
                "in the specified --fold*-dir paths."
            )

        # Hold out the last fold's OOF data as meta-val for XGBoost early stopping.
        if len(X_oof_parts) >= 2:
            X_meta_train = np.concatenate(X_oof_parts[:-1], axis=0)
            y_meta_train = np.concatenate(y_oof_parts[:-1], axis=0)
            X_meta_val = X_oof_parts[-1]
            y_meta_val = y_oof_parts[-1]
        else:
            # Only one fold available — use full OOF for train, no early stopping val.
            X_meta_train = X_oof_parts[0]
            y_meta_train = y_oof_parts[0]
            X_meta_val = X_oof_parts[0]
            y_meta_val = y_oof_parts[0]
            logger.warning("single_fold_no_meta_val — early stopping will overfit")

        sample_weight = _build_meta_weights(y_meta_train, directional_w)

        logger.info(
            "oof_shapes  meta_train=%s  meta_val=%s  n_meta_features=%d",
            list(X_meta_train.shape),
            list(X_meta_val.shape),
            X_meta_train.shape[1],
        )

        # ── 2. Train XGBoost meta-learner ───────────────────────────────────
        logger.info("step  n=2  action=train_meta_xgboost")
        meta_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=4,          # shallow: meta-learner should be simple
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.80,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device="cuda",
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            n_jobs=-1,
            random_state=42,
            verbosity=1,
        )
        meta_model.fit(
            X_meta_train,
            y_meta_train,
            sample_weight=sample_weight,
            eval_set=[(X_meta_val, y_meta_val)],
            verbose=50,
        )
        logger.info(
            "meta_training_done  best_iteration=%d", int(meta_model.best_iteration)
        )

        # ── 3. Save meta model + preprocessing artifacts ────────────────────
        logger.info("step  n=3  action=save_meta_model")
        meta_model.save_model(str(save_path / "stack_xgb.json"))

        # Rebuild last fold's scaler/clip stats for serving-time inference.
        last_fold = folds[-1]
        _pipeline_last = DataPipeline(data_dir, self.config.data)
        _train_raw_last = _pipeline_last.load_features(
            last_fold.train_start, last_fold.train_end
        )
        _feat_cols_last = _resolve_feature_cols(_train_raw_last, self.config)
        _lbl_last = generate_labels(_train_raw_last["_close"], self.config.labels)
        _full_last = pd.concat(
            [_train_raw_last[_feat_cols_last], _lbl_last], axis=1
        ).dropna(subset=["label"])

        n_last = len(_full_last)
        tv_ratio = self.config.data.train_ratio + self.config.data.val_ratio
        tv_end_last = int(n_last * tv_ratio)
        scaler_fit_last = _full_last.iloc[:tv_end_last]
        train_end_last = int(
            len(scaler_fit_last)
            * self.config.data.train_ratio
            / tv_ratio
        )
        scaler_fit_last = scaler_fit_last.iloc[:train_end_last].copy()

        _m_last = scaler_fit_last[_feat_cols_last].mean()
        _s_last = scaler_fit_last[_feat_cols_last].std().replace(0, 1)
        _clip_outliers_inplace(scaler_fit_last, _feat_cols_last, _m_last, _s_last)
        _pipeline_last.fit_scaler(
            scaler_fit_last, _feat_cols_last, self.config.data.scaler_type
        )

        joblib.dump(_pipeline_last.scaler, str(save_path / "meta_scaler.pkl"))
        with open(save_path / "meta_clip_stats.json", "w") as fh:
            json.dump(
                {"mean": _m_last.to_dict(), "std": _s_last.to_dict()},
                fh,
                indent=2,
            )
        with open(save_path / "meta_feature_cols.json", "w") as fh:
            json.dump(
                {
                    "feature_cols": _feat_cols_last,
                    "n_features": len(_feat_cols_last),
                    "n_meta_features": X_meta_train.shape[1],
                },
                fh,
                indent=2,
            )
        logger.info("artifacts_saved  save_dir=%s", str(save_path))

        # ── 4. Final held-out test evaluation ──────────────────────────────
        logger.info("step  n=4  action=final_test_evaluation")
        logger.info(
            "final_test_window  start=%s  end=%s  lstm_fold=%s",
            final_test_start,
            final_test_end,
            last_fold.name,
        )
        # Re-run preprocessing using last fold's fitted pipeline (built above).
        final_test_raw = _pipeline_last.load_features(final_test_start, final_test_end)
        final_test_feat_cols = [
            c for c in final_test_raw.columns if c in _feat_cols_last
        ]
        final_test_labels = generate_labels(
            final_test_raw["_close"], self.config.labels
        )
        final_test_df = pd.concat(
            [final_test_raw[final_test_feat_cols], final_test_labels], axis=1
        ).dropna(subset=["label"]).copy()

        _clip_outliers_inplace(final_test_df, _feat_cols_last, _m_last, _s_last)
        final_test_scaled = _pipeline_last.transform(final_test_df)
        final_test_scaled[_feat_cols_last] = (
            final_test_scaled[_feat_cols_last]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        X_final_seq, y_final = DataPipeline.create_sequences(
            final_test_scaled[_feat_cols_last].values,
            final_test_scaled["label"].values,
            seq_len,
        )
        if X_final_seq.shape[0] == 0:
            raise ValueError(
                f"No sequences in final test window {final_test_start}–{final_test_end}. "
                f"Need at least {seq_len} rows."
            )

        lstm_final = _load_lstm_model(last_fold.lstm_checkpoint_dir)
        lstm_probs_final = lstm_final.predict(
            X_final_seq, batch_size=2048, verbose=0
        )
        X_final_meta = np.hstack(
            [X_final_seq[:, -1, :], lstm_probs_final]
        ).astype(np.float32)

        # ── 5. Evaluate stacking model on final test ────────────────────────
        logger.info("step  n=5  action=evaluate")
        metrics = _evaluate_stack(
            meta_model, X_final_meta, y_final.astype(np.int32)
        )
        metrics["best_iteration"] = int(meta_model.best_iteration)
        metrics["n_oof_train"] = int(len(X_meta_train))
        metrics["n_oof_val"] = int(len(X_meta_val))
        metrics["n_final_test"] = int(len(X_final_meta))
        metrics["n_meta_features"] = int(X_meta_train.shape[1])
        metrics["n_lstm_features"] = len(_feat_cols_last)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        metrics["final_test_start"] = final_test_start
        metrics["final_test_end"] = final_test_end
        metrics["folds"] = [
            {
                "name": f.name,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            for f in folds
        ]

        # ── 6. Save metrics ─────────────────────────────────────────────────
        save_metrics(metrics, save_path, filename="training_result.json")
        logger.info(
            "done  f1_macro=%.4f  directional_f1=%.4f  elapsed_s=%.1f",
            metrics["test_f1_macro"],
            metrics["test_directional_f1"],
            metrics["elapsed_s"],
        )
        return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Train LSTM+XGBoost stacking meta-learner"
    )

    # Fold 1
    parser.add_argument(
        "--fold1-dir", default="data/legacy/checkpoints/model_v015_fold1"
    )
    parser.add_argument("--fold1-train-start", default="2025-05-15")
    parser.add_argument("--fold1-train-end",   default="2025-09-30")
    parser.add_argument("--fold1-test-start",  default="2025-10-01")
    parser.add_argument("--fold1-test-end",    default="2025-10-31")

    # Fold 2
    parser.add_argument(
        "--fold2-dir", default="data/legacy/checkpoints/model_v015_fold2"
    )
    parser.add_argument("--fold2-train-start", default="2025-05-15")
    parser.add_argument("--fold2-train-end",   default="2025-10-31")
    parser.add_argument("--fold2-test-start",  default="2025-11-01")
    parser.add_argument("--fold2-test-end",    default="2025-11-28")

    # Fold 3
    parser.add_argument(
        "--fold3-dir", default="data/legacy/checkpoints/model_v015_fold3"
    )
    parser.add_argument("--fold3-train-start", default="2025-05-15")
    parser.add_argument("--fold3-train-end",   default="2026-01-30")
    parser.add_argument("--fold3-test-start",  default="2026-02-01")
    parser.add_argument("--fold3-test-end",    default="2026-04-01")

    # Final held-out test window (must be beyond all fold test windows).
    parser.add_argument("--final-test-start", default="2026-04-06")
    parser.add_argument(
        "--final-test-end", required=True, help="Final test end YYYY-MM-DD"
    )

    parser.add_argument("--save-dir", default="data/legacy/checkpoints/stack_v001")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    _folds = [
        FoldDef(
            lstm_checkpoint_dir=args.fold1_dir,
            train_start=args.fold1_train_start,
            train_end=args.fold1_train_end,
            test_start=args.fold1_test_start,
            test_end=args.fold1_test_end,
            name="fold1",
        ),
        FoldDef(
            lstm_checkpoint_dir=args.fold2_dir,
            train_start=args.fold2_train_start,
            train_end=args.fold2_train_end,
            test_start=args.fold2_test_start,
            test_end=args.fold2_test_end,
            name="fold2",
        ),
        FoldDef(
            lstm_checkpoint_dir=args.fold3_dir,
            train_start=args.fold3_train_start,
            train_end=args.fold3_train_end,
            test_start=args.fold3_test_start,
            test_end=args.fold3_test_end,
            name="fold3",
        ),
    ]

    config = MLConfig.from_env()
    trainer = StackTrainer(config)
    trainer.run(
        folds=_folds,
        final_test_start=args.final_test_start,
        final_test_end=args.final_test_end,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
    )
