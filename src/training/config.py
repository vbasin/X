"""ML configuration dataclasses for the Databento pipeline.

Detection-2 architecture: every bar emits one of {BIG_DOWN, NO_SIGNAL, BIG_UP}.
NO_SIGNAL bars are masked during training via sample_weight=0; the model
output layer is a 2-class softmax over {BIG_DOWN, BIG_UP}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields as _dc_fields, MISSING as _MISSING
from pathlib import Path

logger = logging.getLogger("vtech.training.config")


def _fd(cls, name: str):
    """Return the dataclass field default — single source of truth for all fallbacks."""
    f = next(fi for fi in _dc_fields(cls) if fi.name == name)
    return f.default if f.default is not _MISSING else f.default_factory()


def _from_section(cls, section: dict):
    """Construct a dataclass from a YAML section; field defaults are the fallback."""
    return cls(**{fi.name: section.get(fi.name, _fd(cls, fi.name)) for fi in _dc_fields(cls)})


@dataclass
class FeatureConfig:
    wavelet_window: int = 128
    wavelet_scales: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    max_spread_pct: float = 0.25
    min_option_price: float = 0.25
    large_trade_threshold: int = 10  # contracts — threshold for lt_ features
    enabled_groups: dict[str, bool] = field(default_factory=lambda: {
        "book_pressure": True,
        "order_flow": True,
        "trade_location": True,
        "same_side": True,
        "large_trade_cvd": True,
        "microstructure": True,
        "trade_arrival": True,
        "realized_vol": True,
        "sub_bar_dynamics": True,
        "volume_profile": True,
        "iv_surface": True,
        "dealer_gex": True,
        "daily_context": True,
        "wavelets": True,
        "candle_structure": True,
        "time_context": True,
        "vwap": True,
        "cross_asset": True,
        "economic_calendar": True,
        "hawkes_clustering": True,
        "higher_timeframe": True,
    })
    disabled_features: list[str] = field(default_factory=list)
    # Individual feature columns to exclude from training regardless of their
    # group toggle. Exact column names (e.g. "bp_spread_mean"). Applied after
    # enabled_groups filtering. Full catalog: docs/reference/FEATURE_CATALOG.md


@dataclass
class DataConfig:
    feature_timestep: str = "5s"
    lookback_minutes: int = 128
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    training_start: str = "08:00"
    training_end: str = "11:00"
    timezone: str = "America/Chicago"
    scaler_type: str = "robust"  # "standard", "robust", or "minmax"


@dataclass
class LabelConfig:
    """Label parameters (scheme selected by ``scheme``).

    ``detection_2`` (default): a bar is BIG_UP if the next ``forward_windows[0]``
    bars contain an upward excursion of at least ``big_move_threshold`` AND the
    worst adverse downward excursion stays below ``mae_tolerance`` (order-agnostic).
    ``triple_barrier``: first-touch — BIG_UP iff the +threshold profit barrier is
    touched before the -mae_tolerance stop barrier and before the window end.
    Symmetric for BIG_DOWN. Everything else is NO_SIGNAL (masked).
    """

    big_move_threshold: float = 50.0
    mae_tolerance: float = 5.0
    forward_windows: list[int] = field(default_factory=lambda: [60, 120, 180, 360])
    scheme: str = "detection_2"
    target_observation: str = "candle_close"


@dataclass
class ModelConfig:
    sequence_length: int = 128
    lstm_units: list[int] = field(default_factory=lambda: [128])
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.1
    attention_heads: int = 4  # 0 = no attention; N = multi-head attention with N heads
    dense_units: list[int] = field(default_factory=lambda: [64])
    dense_dropout: float = 0.3
    num_classes: int = 3  # 3-class: {BIG_DOWN=0, NO_SIGNAL=1, BIG_UP=2}
    class_weight_directional: float = 15.0  # inverse-frequency weight for BIG_DOWN and BIG_UP


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 0.001
    early_stop_patience: int = 12    # v011: raised from 5 → 12 (ADR-011)
    early_stop_min_delta: float = 0.001
    lr_reduce_patience: int = 5      # v011: raised from 3 → 5 (ADR-011)
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-5
    save_best_only: bool = True
    use_focal_loss: bool = True      # v011: focal loss replaces CE (ADR-011)
    focal_loss_gamma: float = 2.0    # standard Lin et al. 2017 value
    intraop_threads: int = 0         # 0 = TF default (all cores); set to N when N folds run in parallel
    train_sequences: int | None = None  # cap training sequences before fit; None = calculate from data
    use_balanced_sampling: bool = False  # v013: when true, build a tf.data pipeline that draws 1/3 BD, 1/3 NS, 1/3 BU per batch; per-class count = min(n_BD, n_BU). Replaces class_weight.
    min_steps_per_epoch: int = 50    # v013 fix: floor for steps_per_epoch in balanced sampling; prevents tiny epochs when per_class is small
    lr_schedule: str = "plateau"     # "plateau" = ReduceLROnPlateau (legacy); "cosine" = warmup + cosine decay
    lr_warmup_epochs: int = 5        # epochs over which LR ramps from min_lr → learning_rate (cosine schedule only)
    use_ig: bool = True              # set False to skip per-epoch IG group logging (speeds up HPO/ablation trials ~30-40%)


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://127.0.0.1:5000"  # overridden by MLFLOW_TRACKING_URI env var if set
    experiment_name: str = "nq-50pt-intraday"
    enabled: bool = True


def _labels_from_objective(objective_id: str | None) -> LabelConfig:
    """Build LabelConfig from the trade objective doc (single source of truth).

    The objective's ``labels.move_threshold`` / ``labels.mae_tolerance`` map to
    ``LabelConfig.big_move_threshold`` / ``mae_tolerance``. ``forward_windows`` is a
    training-only knob (not objective-defining) and keeps the dataclass default.

    ``config.yaml`` no longer names an objective, so when ``objective_id`` is
    ``None`` the active objective is auto-discovered from ``config/objectives/``
    (``resolve_active_objective_id``). An explicit ``objective_id`` still wins if
    one is ever supplied. Falls back to dataclass defaults if no objective can be
    resolved or loaded, so callers never crash on a missing/invalid objective.
    """
    base = LabelConfig()
    try:
        from src.objectives.loader import load_objective, resolve_active_objective_id

        oid = objective_id or resolve_active_objective_id()
        if not oid:
            logger.warning(
                "no objective found in config/objectives/ — using LabelConfig defaults"
            )
            return base
        obj = load_objective(str(oid))
    except Exception as exc:  # noqa: BLE001 — config load must never hard-fail
        logger.warning(
            "could not resolve/load objective (%s) — using LabelConfig defaults", exc
        )
        return base
    lbl = obj.get("labels") or {}
    return LabelConfig(
        big_move_threshold=float(lbl.get("move_threshold", base.big_move_threshold)),
        mae_tolerance=float(lbl.get("mae_tolerance", base.mae_tolerance)),
        forward_windows=base.forward_windows,
        scheme=str(lbl.get("scheme", base.scheme)),
        target_observation=str(lbl.get("target_observation", base.target_observation)),
    )


@dataclass
class MLConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @classmethod
    def from_env(cls, config_path: Path | None = None) -> MLConfig:
        from src.common.config import _load_yaml

        d = _load_yaml(config_path or Path("/opt/vtech/config.yaml"))

        # features — merge YAML enabled_groups over the dataclass defaults so
        # any group absent from the YAML still defaults to True (enabled).
        feat_section = dict(d.get("features", {}) or {})
        yaml_groups = feat_section.get("enabled_groups", {}) or {}
        default_groups = FeatureConfig().enabled_groups
        feat_section["enabled_groups"] = {**default_groups, **yaml_groups}
        features = _from_section(FeatureConfig, feat_section)

        # labels — SINGLE SOURCE OF TRUTH is the trade objective doc under
        # config/objectives/. config.yaml is objective-agnostic: it names no
        # objective and carries no label params (so they cannot drift). The
        # active objective is auto-discovered when exactly one exists; an
        # explicit `objective_id` key still overrides discovery if present.
        labels = _labels_from_objective(d.get("objective_id"))

        return cls(
            features=features,
            data=    _from_section(DataConfig,     d.get("data",     {}) or {}),
            labels=  labels,
            model=   _from_section(ModelConfig,    d.get("model",    {}) or {}),
            training=_from_section(TrainingConfig, d.get("training", {}) or {}),
            mlflow=  _from_section(MLflowConfig,   d.get("mlflow",   {}) or {}),
        )
