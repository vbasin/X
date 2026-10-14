"""Model Catalog — Phase 23 W4 (declared here alongside W3; W4 adds more families).

The catalog is the declarative registry of candidate model *types* the search
controller selects from.  It is a knowledge/context layer — completely separate
from ``data/models/model_registry/`` which holds trained artifacts.

Structure: learning paradigm → family → CatalogEntry.

Every entry carries:
  name            — unique key matching ModelPlugin.name
  family_group    — top-level paradigm bucket
  compute_class   — "cpu" | "gpu" | "either"
  cost_class      — "cheap" | "moderate" | "expensive" (relative training cost;
                    used for cost estimation + try-order by the Model Scout, W5)
  needs_labels    — True for supervised families
  needs_sequence  — True for sequence/temporal models (LSTM, TCN, …)
  feature_view    — optional named curated feature subset (src/models/feature_views.py);
                    None = full matrix. Per-objective override:
                    objective["search"]["feature_views"][family]
  params_default  — default ModelSpec.params for a quick first-run

Adding a new family:
  1. Implement the plugin in src/models/<name>_plugin.py
  2. Add a CatalogEntry here
  3. Set compute_class so the dispatcher routes it correctly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    name:             str
    display_name:     str
    family_group:     str   # "sequence_dl" | "tree_ensemble" | "probabilistic" | "regime" | "linear" | "ensemble"
    compute_class:    str   # "cpu" | "gpu" | "either"
    needs_labels:     bool
    needs_sequence:   bool
    cost_class:       str   = "moderate"   # "cheap" | "moderate" | "expensive"
    feature_view:     str | None = None    # named curated feature subset (feature_views.py)
    feature_selection: dict[str, Any] | None = None  # explicit feature pick for USER-OVERRIDE
                                           # candidates (manual mode). One of:
                                           #   {"view": "<name>"} | {"prefixes": [...]} |
                                           #   {"columns": [...]}. None = fall back to
                                           # feature_view / full matrix. See feature_views.py.
    search_max_rows:  int | None = None    # train-fold row cap. Trees: random seed 42.
                                           # Sequence: last-N contiguous bars (ADR-043).
                                           # None = all rows. Multi-base topology still
                                           # drops the cap so bases share train rows.
    search_space:     dict[str, Any] = field(default_factory=dict)   # W8 per-family HPO space
    params_default:   dict[str, Any] = field(default_factory=dict)
    params_help:      dict[str, str] = field(default_factory=dict)  # Custom Test param tooltips
    plugin_family:    str | None = None    # USER-OVERRIDE: the registered plugin family to load
                                           # when `name` is a custom candidate id (lets several
                                           # manual candidates share one family, e.g. two XGBoost
                                           # configs). None = `name` is itself the family key.
    notes:            str  = ""

    @property
    def family_key(self) -> str:
        """Registered plugin/catalog family to build this candidate with.

        Equals ``name`` for catalogued entries; for user-override candidates whose
        ``name`` is a unique candidate id, ``plugin_family`` carries the real family.
        """
        return self.plugin_family or self.name


# ── Registered families ───────────────────────────────────────────────────────

_CATALOG: list[CatalogEntry] = [
    # ── Sequence DL (GPU) ────────────────────────────────────────────────────
    CatalogEntry(
        name="attention_lstm",
        display_name="Attention-LSTM (3-class softmax)",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        search_max_rows=100_000,
        search_space={
            "class_balance": {"type": "categorical",
                              "choices": [None, "balanced", "sqrt"]},
        },
        params_default={"lstm_units": [64, 32], "attention_heads": 4,
                        "dense_units": [64, 32], "dropout": 0.2,
                        "seq_len": 128, "epochs": 15,
                        "batch_size": 128, "class_balance": "sqrt"},
        params_help={
            "seq_len": "Lookback in 5-second bars. Each sample is this many consecutive bars ending at the decision bar. 128 = 10.7 minutes. Input shape is (seq_len, F) where F is the count of features you assigned on the board.",
            "epochs": "Max training passes per walk-forward fold (early-stop patience 10). Last GPU Custom Test used 15.",
            "class_balance": "Train-time loss weights. sqrt: a rare up/down bar counts ~4× a flat bar. none = unweighted (always-flat collapse).",
        },
        notes="cuDNN LSTM; class_balance=sqrt sample weights (ADR-044); per-fold RobustScaler + rebuild",
    ),
    CatalogEntry(
        name="tcn",
        display_name="Temporal Convolutional Network",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        search_max_rows=100_000,
        params_default={"filters": 64, "kernel_size": 3, "dilations": [1,2,4,8],
                        "dropout": 0.2, "seq_len": 128, "epochs": 15,
                        "class_balance": "sqrt"},
        params_help={
            "seq_len": "Lookback in 5-second bars. 128 = 10.7 minutes of history per sample.",
        },
        notes="Often beats LSTM on long-memory tasks (Papers 5, 6 in research corpus)",
    ),
    CatalogEntry(
        name="tcn_close",
        display_name="TCN on 5s close log-return",
        family_group="sequence_dl",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="moderate",
        plugin_family="tcn",
        search_max_rows=100_000,
        feature_selection={"columns": ["_close"]},
        params_default={"filters": 64, "kernel_size": 3, "dilations": [1, 2, 4, 8],
                        "dropout": 0.2, "seq_len": 128, "epochs": 5,
                        "batch_size": 256, "class_balance": "sqrt",
                        "pooling": "last"},
        params_help={
            "seq_len": "Lookback in 5-second bars. 128 = 10.7 minutes of close log-returns.",
            "epochs": "Cap at 5 for a CPU smoke; raise for a real fit.",
        },
        notes=(
            "1-channel causal log-return of parquet _close (not 1s path, not full OHLCV). "
            "Labels stay 5s-close TBM. CPU-runnable; same TCNPlugin trunk as tcn."
        ),
    ),
    CatalogEntry(
        name="tcn_hazard",
        display_name="TCN + competing-risks hazard heads",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        feature_view="sequence_baseline",
        search_max_rows=100_000,
        params_default={"filters": 64, "kernel_size": 3, "dilations": [1, 2, 4, 8],
                        "dropout": 0.2, "seq_len": 128, "epochs": 15,
                        "batch_size": 64, "class_balance": "sqrt",
                        "include_quantile": True, "inference_r_min": 67.3},
        params_help={
            "seq_len": "Lookback in 5-second bars. 128 = 10.7 minutes. Softmax is replaced by 41-class hazard heads per side (ADR-053).",
            "inference_r_min": "OOF bars with causal R below this are emitted flat. 67.3 = live +50/−6 geometry.",
        },
        notes="5s-feature TCN+heads (not the 1s-path ablation). Prefer path_tcn.",
    ),
    CatalogEntry(
        name="path_tcn",
        display_name="1s-path TCN + hazard heads",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        search_max_rows=None,
        params_default={
            "filters": 64, "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "dropout": 0.1, "seq_len": 1024, "epochs": 15, "batch_size": 64,
            "class_balance": "sqrt", "include_quantile": True,
            "inference_r_min": 67.3,
        },
        params_help={
            "seq_len": "1-second lookback. 1024 = 17.1 min, matching dilations 1…512 (RF 1024 s).",
            "dilations": "Ten causal layers 1…512. kernel_size=2 → RF = 1024 seconds.",
        },
        notes="Encoder A (recovered spec): ATR-normalised parquet_path1s, last-step embedding, shared heads. No VIX.",
    ),
    CatalogEntry(
        name="patch_hazard",
        display_name="Patch Transformer + competing-risks hazard heads",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        feature_view="sequence_baseline",
        search_max_rows=100_000,
        params_default={"d_model": 128, "n_heads": 4, "n_layers": 4, "ff_dim": 512,
                        "patch_len": 8, "dropout": 0.1, "seq_len": 128, "epochs": 15,
                        "batch_size": 64, "class_balance": "sqrt",
                        "include_quantile": True, "inference_r_min": 67.3},
        params_help={
            "seq_len": "Lookback in 5-second bars. Must be divisible by patch_len (default 8).",
            "d_model": "Transformer width. 128 × 4 layers × ff=512 is ~0.9M params.",
        },
        notes="Encoder B: 4-stream patch tokeniser (price/flow/volume/options) + 4-layer Transformer + shared heads. 1s path patches not yet wired.",
    ),
    CatalogEntry(
        name="transformer",
        display_name="Transformer (encoder, small)",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        cost_class="expensive",
        search_max_rows=100_000,
        params_default={"d_model": 64, "n_heads": 4, "n_layers": 2,
                        "dropout": 0.1, "seq_len": 128, "epochs": 15},
        notes="Avoid pure decoder-only; encoder-only or TFT recommended (ADR notes)",
    ),
    # ── Tree ensembles (CPU) ─────────────────────────────────────────────────
    CatalogEntry(
        name="xgboost",
        display_name="XGBoost classifier",
        family_group="tree_ensemble",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        search_max_rows=150_000,
        search_space={
            "n_estimators":     {"type": "int",   "low": 150, "high": 500, "step": 50},
            "max_depth":        {"type": "int",   "low": 4,   "high": 12},
            "learning_rate":    {"type": "float", "low": 0.01, "high": 0.20, "log": True},
            "subsample":        {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "class_balance":    {"type": "categorical", "choices": [None, "balanced", "sqrt"]},
            "dir_threshold":    {"type": "float", "low": 0.30, "high": 0.60},
        },
        params_default={"n_estimators": 150, "max_depth": 4,
                        "learning_rate": 0.05, "subsample": 0.8,
                        "colsample_bytree": 0.8, "use_label_encoder": False,
                        "eval_metric": "mlogloss",
                        "class_balance": "sqrt", "dir_threshold": 0.5},
    ),
    CatalogEntry(
        name="xgboost_ovr",
        display_name="XGBoost one-vs-rest (up vs rest + down vs rest)",
        family_group="tree_ensemble",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        plugin_family="xgboost",
        search_max_rows=None,
        params_default={"n_estimators": 150, "max_depth": 4,
                        "learning_rate": 0.05, "subsample": 0.8,
                        "colsample_bytree": 0.8, "use_label_encoder": False,
                        "eval_metric": "logloss",
                        "class_balance": "sqrt", "dir_threshold": 0.55,
                        "head": "ovr"},
        notes="Two binary XGBs; P(up) and P(down) are independent. ADR-045.",
    ),
    CatalogEntry(
        name="lightgbm",
        display_name="LightGBM classifier",
        family_group="tree_ensemble",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        search_max_rows=150_000,
        search_space={
            "n_estimators":  {"type": "int",   "low": 150, "high": 500, "step": 50},
            "max_depth":     {"type": "int",   "low": 4,   "high": 12},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
            "num_leaves":    {"type": "int",   "low": 15,  "high": 127},
            "subsample":     {"type": "float", "low": 0.6, "high": 1.0},
            "class_balance": {"type": "categorical", "choices": [None, "balanced", "sqrt"]},
            "dir_threshold": {"type": "float", "low": 0.30, "high": 0.60},
        },
        params_default={"n_estimators": 150, "max_depth": 4,
                        "learning_rate": 0.05, "subsample": 0.8,
                        "num_leaves": 31,
                        "class_balance": "sqrt", "dir_threshold": 0.5},
    ),
    CatalogEntry(
        name="random_forest",
        display_name="Random Forest classifier",
        family_group="tree_ensemble",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        search_max_rows=150_000,
        search_space={
            "n_estimators":  {"type": "int", "low": 200, "high": 600, "step": 100},
            "max_depth":     {"type": "int", "low": 6,   "high": 20},
            "class_balance": {"type": "categorical", "choices": [None, "balanced", "sqrt"]},
            "dir_threshold": {"type": "float", "low": 0.30, "high": 0.60},
        },
        params_default={"n_estimators": 200, "max_depth": 8, "n_jobs": -1,
                        "class_balance": "sqrt", "dir_threshold": 0.5},
    ),
    # ── Probabilistic / statistical (CPU) ────────────────────────────────────
    CatalogEntry(
        name="hmm",
        display_name="Hidden Markov Model (regime detector)",
        family_group="probabilistic",
        compute_class="cpu",
        needs_labels=False,
        needs_sequence=True,
        feature_view="regime_core",
        search_max_rows=50_000,
        params_default={"n_components": 4, "covariance_type": "diag", "n_iter": 100},
        notes="Unsupervised regime detector; scored as a gate/router via canonical adapter.",
    ),
    CatalogEntry(
        name="gmm",
        display_name="Gaussian Mixture Model (regime)",
        family_group="probabilistic",
        compute_class="cpu",
        needs_labels=False,
        needs_sequence=False,
        cost_class="cheap",
        feature_view="regime_core",
        search_max_rows=50_000,
        params_default={"n_components": 4, "covariance_type": "diag",
                        "reg_covar": 1e-4, "max_iter": 200},
    ),
    # ── Linear baseline (CPU) ────────────────────────────────────────────────
    CatalogEntry(
        name="logistic_regression",
        display_name="Logistic Regression (baseline)",
        family_group="linear",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        cost_class="cheap",
        search_max_rows=200_000,
        params_default={"C": 1.0, "max_iter": 500, "class_weight": "balanced"},
        notes="Cheap calibrated baseline for sanity-checking the leaderboard",
    ),
    # ── SVM (CPU) ─────────────────────────────────────────────────────────────
    CatalogEntry(
        name="svm",
        display_name="SVM (LinearSVC + Platt calibration)",
        family_group="linear",
        compute_class="cpu",
        needs_labels=True,
        needs_sequence=False,
        cost_class="moderate",
        search_max_rows=150_000,
        params_default={"C": 1.0, "kernel": "linear", "large_scale": True,
                        "class_weight": "balanced"},
        notes="Always uses LinearSVC (large_scale=True). RBF is O(n^2) on 1.4M rows.",
    ),
    # ── CNN (GPU) ─────────────────────────────────────────────────────────────
    CatalogEntry(
        name="cnn",
        display_name="1-D Convolutional Neural Network",
        family_group="sequence_dl",
        compute_class="gpu",
        needs_labels=True,
        needs_sequence=True,
        search_max_rows=100_000,
        params_default={"filters": 64, "kernel_size": 3, "n_blocks": 3,
                        "dropout": 0.2, "seq_len": 128, "n_features": 154, "epochs": 15},
        notes="Faster than LSTM; good baseline for feature-sequence tasks",
    ),
    # ── Clustering regime (CPU) ───────────────────────────────────────────────
    CatalogEntry(
        name="kmeans",
        display_name="K-Means clustering (regime)",
        family_group="regime",
        compute_class="cpu",
        needs_labels=False,
        needs_sequence=False,
        cost_class="cheap",
        feature_view="regime_core",
        search_max_rows=50_000,
        params_default={"n_clusters": 6, "n_init": 10, "max_iter": 300},
        notes="Centroid-based regime detection; bias derived from empirical label distribution per cluster",
    ),
    CatalogEntry(
        name="dbscan",
        display_name="DBSCAN density clustering (regime)",
        family_group="regime",
        compute_class="cpu",
        needs_labels=False,
        needs_sequence=False,
        feature_view="regime_core",
        search_max_rows=20_000,
        params_default={"eps": 0.5, "min_samples": 5, "n_jobs": -1},
        notes="Discovers k automatically; inference via 1-NN to nearest centroid; noise points -> flat",
    ),
    # ── Reinforcement Learning (GPU) ──────────────────────────────────────────
    CatalogEntry(
        name="rl",
        display_name="Reinforcement Learning (REINFORCE, policy-gradient)",
        family_group="reinforcement_learning",
        compute_class="gpu",
        needs_labels=False,
        needs_sequence=False,
        cost_class="expensive",
        search_max_rows=100_000,
        params_default={"n_features": 154, "seq_len": 128,
                        "hidden_units": [256, 128], "dropout": 0.1, "epochs": 15},
        notes="REINFORCE actor-critic; reward bootstrapped from directional labels. "
              "Action space: {0=down, 1=flat, 2=up}.",
    ),
]

# Indexed for fast lookup
_BY_NAME: dict[str, CatalogEntry] = {e.name: e for e in _CATALOG}


def get_catalog() -> list[CatalogEntry]:
    """Return all registered model-family catalog entries."""
    return list(_CATALOG)


def get_entry(name: str) -> CatalogEntry:
    """Return the catalog entry for *name*; raises KeyError if not found."""
    if name not in _BY_NAME:
        raise KeyError(f"Model family '{name}' not in catalog")
    return _BY_NAME[name]
