"""Standalone XGBoost trainer for the detection-2 big-move classification problem.

Shares data loading, label generation, splitting, and evaluation utilities with
the LSTM trainer but does NOT import from trainer.py.  XGBoost is scale-invariant
so RobustScaler is omitted; outlier clipping (z=5.0) is retained for consistency.

Run:
    python -m src.training.xgb_trainer \\
        --start 2025-05-15 --end 2026-01-30 \\
        --test-start 2026-02-01 --test-end 2026-04-01 \\
        --save-dir /home/opc/VTECH/data/legacy/checkpoints/xgb_v001
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

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

logger = logging.getLogger("vtech.training.xgb_trainer")

# Mirrors FEATURE_GROUP_PREFIXES in trainer.py — no import coupling.
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

_BD = 0  # BIG_DOWN label
_NS = 1  # NO_SIGNAL label
_BU = 2  # BIG_UP label


def _resolve_feature_cols(features_df: pd.DataFrame, config: MLConfig) -> list[str]:
    """Filter feature columns via enabled_groups / disabled_features — mirrors trainer.py."""
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
    """Clip feature columns to ±z standard deviations in-place."""
    df[feature_cols] = df[feature_cols].clip(
        lower=(train_mean - z * train_std),
        upper=(train_mean + z * train_std),
        axis=1,
    )


def _build_sample_weights(y: np.ndarray, directional_weight: float = 4.0) -> np.ndarray:
    """Per-sample weights: BD=w, NS=1.0, BU=w (mirrors class_weight_directional)."""
    weights = np.ones(len(y), dtype=np.float32)
    weights[y == _BD] = directional_weight
    weights[y == _BU] = directional_weight
    return weights


def _evaluate_xgb(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate XGBoost model — output schema matches evaluate_model() in evaluate.py."""
    probs = model.predict_proba(X_test)  # (N, 3)
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


def _xgb_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
) -> dict:
    """Group-level + top-30 feature importance using XGBoost native gain scores.

    ``model.feature_importances_`` is a numpy array aligned with input columns
    in fit() order — no f0/f1/... remapping needed.
    """
    importances: np.ndarray = model.feature_importances_  # (n_features,)
    ranked = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])

    groups: list[dict] = []
    for grp, prefix in sorted(_FEATURE_GROUP_PREFIXES.items()):
        idxs = [i for i, f in enumerate(feature_cols) if f.startswith(prefix)]
        if not idxs:
            continue
        groups.append(
            {
                "group": grp,
                "n_feats": len(idxs),
                "mean_gain": round(float(importances[idxs].mean()), 6),
            }
        )
    groups.sort(key=lambda g: -g["mean_gain"])

    return {
        "top_features": [
            {"feature": f, "gain": round(float(v), 6)} for f, v in ranked[:30]
        ],
        "groups": groups,
    }


class XGBTrainer:
    """Train a 3-class XGBoost model on the detection-2 flat feature set."""

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
    ) -> dict:
        """Execute the full pipeline and return a metrics dict.

        When *test_start* / *test_end* are provided the window
        [start_date, end_date] is used for train+val and the separate
        [test_start, test_end] window for held-out evaluation (walk-forward
        mode, matching trainer.py behaviour).
        """
        from src.common.repro import set_global_seed

        set_global_seed()
        t0 = time.time()
        save_path = Path(save_dir or "checkpoints/xgb_latest")
        save_path.mkdir(parents=True, exist_ok=True)
        data_dir = data_dir or os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")
        pipeline = DataPipeline(data_dir, self.config.data)

        # 1. Load features ──────────────────────────────────────────────────
        logger.info("step  n=1  action=load_features")
        features_df = pipeline.load_features(start_date, end_date)
        test_df_raw = None
        if test_start and test_end:
            test_df_raw = pipeline.load_features(test_start, test_end)
            logger.info(
                "walk_forward_mode  train_end=%s  test_start=%s  test_end=%s",
                end_date,
                test_start,
                test_end,
            )

        # 2. Resolve feature columns ────────────────────────────────────────
        logger.info("step  n=2  action=resolve_feature_cols")
        feature_cols = _resolve_feature_cols(features_df, self.config)
        if not feature_cols:
            raise ValueError(
                "No feature columns remain after applying enabled_groups filter."
            )
        logger.info("feature_cols_resolved  n_features=%d", len(feature_cols))

        # 3. Generate detection-2 labels ────────────────────────────────────
        logger.info("step  n=3  action=generate_labels")
        if "_close" not in features_df.columns:
            raise ValueError(
                "Parquet missing '_close' column required for label generation."
            )
        labels_df = generate_labels(features_df["_close"], self.config.labels)
        full_df = pd.concat(
            [features_df[feature_cols], labels_df], axis=1
        ).dropna(subset=["label"])

        dist = full_df["label"].value_counts().sort_index()
        _names = {0: "BIG_DOWN", 1: "NO_SIGNAL", 2: "BIG_UP"}
        for cls_id, count in dist.items():
            logger.info(
                "class_dist  class=%s  count=%d  pct=%.1f%%",
                _names.get(int(cls_id), str(cls_id)),
                int(count),
                count / len(full_df) * 100,
            )

        # 4. Temporal split ─────────────────────────────────────────────────
        logger.info("step  n=4  action=split")
        if test_df_raw is not None:
            test_feature_cols = [c for c in test_df_raw.columns if c in feature_cols]
            test_labels_raw = generate_labels(
                test_df_raw["_close"], self.config.labels
            )
            test_full = pd.concat(
                [test_df_raw[test_feature_cols], test_labels_raw], axis=1
            ).dropna(subset=["label"])
            # Split train window into train / val using configured ratios.
            n = len(full_df)
            tv_ratio = self.config.data.train_ratio + self.config.data.val_ratio
            tv_end = int(n * tv_ratio)
            tv_df = full_df.iloc[:tv_end]
            train_split = self.config.data.train_ratio / tv_ratio
            train_df = tv_df.iloc[: int(len(tv_df) * train_split)].copy()
            val_df = tv_df.iloc[int(len(tv_df) * train_split) :].copy()
            test_df = test_full.copy()
        else:
            train_df, val_df, test_df = [
                s.copy() for s in pipeline.split_temporal(full_df)
            ]

        logger.info(
            "split_sizes  train=%d  val=%d  test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # 5. Outlier clipping — z=5.0, trained on train_df statistics ───────
        logger.info("step  n=5  action=clip_outliers")
        train_mean = train_df[feature_cols].mean()
        train_std = train_df[feature_cols].std().replace(0, 1)
        for df_part in [train_df, val_df, test_df]:
            _clip_outliers_inplace(df_part, feature_cols, train_mean, train_std)

        # 6. Flat feature matrices (no scaler, no sequences) ────────────────
        logger.info("step  n=6  action=build_arrays")

        def _safe_values(df: pd.DataFrame) -> np.ndarray:
            return (
                df[feature_cols]
                .fillna(0)
                .replace([np.inf, -np.inf], 0)
                .values.astype(np.float32)
            )

        X_train = _safe_values(train_df)
        X_val = _safe_values(val_df)
        X_test = _safe_values(test_df)
        y_train = train_df["label"].values.astype(np.int32)
        y_val = val_df["label"].values.astype(np.int32)
        y_test = test_df["label"].values.astype(np.int32)

        logger.info(
            "array_shapes  train=%s  val=%s  test=%s",
            list(X_train.shape),
            list(X_val.shape),
            list(X_test.shape),
        )

        # 7. Sample weights ─────────────────────────────────────────────────
        directional_w = float(self.config.model.class_weight_directional)
        sample_weight = _build_sample_weights(y_train, directional_w)

        # 8. Train XGBClassifier ────────────────────────────────────────────
        logger.info("step  n=8  action=train_xgboost")
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.70,
            min_child_weight=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device="cuda",
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42,
            verbosity=1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        logger.info("training_done  best_iteration=%d", int(model.best_iteration))

        # 9. Save model + feature metadata ──────────────────────────────────
        logger.info("step  n=9  action=save_model")
        model_path = save_path / "xgb_model.json"
        model.save_model(str(model_path))
        with open(save_path / "features.json", "w") as fh:
            json.dump(
                {
                    "features": feature_cols,
                    "n_features": len(feature_cols),
                    "scaler": "none",
                },
                fh,
                indent=2,
            )
        logger.info("model_saved  path=%s", str(model_path))

        # 10. Evaluate + importance + save metrics ──────────────────────────
        logger.info("step  n=10  action=evaluate")
        metrics = _evaluate_xgb(model, X_test, y_test)
        metrics["feature_importance"] = _xgb_feature_importance(model, feature_cols)
        metrics["best_iteration"] = int(model.best_iteration)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        metrics["train_start"] = start_date
        metrics["train_end"] = end_date
        metrics["test_start"] = test_start or ""
        metrics["test_end"] = test_end or ""
        metrics["n_train"] = int(len(X_train))
        metrics["n_val"] = int(len(X_val))
        metrics["n_test"] = int(len(X_test))
        metrics["n_features"] = len(feature_cols)
        metrics["directional_weight"] = directional_w

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
    parser = argparse.ArgumentParser(description="Train XGBoost NQ detection-2 model")
    parser.add_argument(
        "--start", required=True, help="Training start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", required=True, help="Training end date YYYY-MM-DD"
    )
    parser.add_argument(
        "--test-start", default=None, help="Walk-forward test start YYYY-MM-DD"
    )
    parser.add_argument(
        "--test-end", default=None, help="Walk-forward test end YYYY-MM-DD"
    )
    parser.add_argument(
        "--save-dir", default=None, help="Directory to save model + metrics"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root data directory (overrides VTECH_DATA_DIR)",
    )
    args = parser.parse_args()

    config = MLConfig.from_env()
    trainer = XGBTrainer(config)
    trainer.run(
        start_date=args.start,
        end_date=args.end,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        test_start=args.test_start,
        test_end=args.test_end,
    )
