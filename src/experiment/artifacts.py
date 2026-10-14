"""Canonical run-output contract: one name per semantic artifact class.

Run directory (the existing, adopted layout -- no extra nesting)::

    research/runs/<objective_id>/<run_ts>/

Every class below has ONE canonical filename, a minimum-content rule, and an
applicability rule computed from the run's own ``RunManifest``. A class whose
responsibility did not run is *not applicable* -- never "missing".

Filename criteria (enforced by tests): a canonical name describes semantic
content, so it stays valid for one-stage, multi-stage, gated, ungated or
ensemble strategies. It never contains a stage number, a model family, a
version suffix or a temporary experiment label.

Existing files are NOT renamed, copied or removed. Where an existing file
already realises a class (e.g. ``oof_xgboost.parquet`` realises
``oof_predictions``), the manifest's ``artifacts`` map records that mapping
instead of duplicating a large artifact to satisfy naming symmetry.
``COMPATIBILITY_ARTIFACTS`` classifies the existing files explicitly.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pathlib import Path

from src.experiment.schemas import OOF_PREDICTION_COLUMNS

#: Filenames. Objective runs write ``manifest.json`` only at run level.
#: ``run_layout`` imports these names and resolves paths; it does not define
#: a second contract.
MANIFEST_JSON = "manifest.json"
MODEL_JSON = "model.json"
MODEL_ARTIFACT = "model.ubj"

#: Fitted-weight filenames, one semantic class. A run keeps whichever file its
#: fit actually wrote -- the extension names the format needed to read the
#: weights back, not a version or a stage. ``MODEL_ARTIFACT`` is the default a
#: new XGBoost run writes; a Keras or pickled fit writes its own from this set.
MODEL_ARTIFACT_FILENAMES: tuple[str, ...] = (
    "model.ubj",            # XGBoost booster
    "model.keras",          # attention-LSTM / TCN
    "model.pkl",            # pickled tree model
    "meta.pkl",             # meta estimator; a live bundle is meta.json + meta.pkl
    "meta_estimator.pkl",   # tf_oof_meta's estimator, not a live serving bundle
)
OOF_PREDICTIONS = "oof_predictions.parquet"
POPULATION_CACHE = "training_matrix.parquet"  # optional; one population, not one per feature subset
METRICS_JSON = "metrics.json"
FOLD_METRICS_JSON = "fold_metrics.json"
SELECTED_POPULATION_JSON = "selected_population.json"
TRADE_LEDGER = "trade_ledger.parquet"
ECONOMIC_METRICS_JSON = "economic_metrics.json"
COST_SENSITIVITY_JSON = "cost_sensitivity.json"
REPORT_DIR = "report"
SUMMARY_MD = "summary.md"
SHAP_IMPORTANCE = "shap_importance.csv"
SHAP_BAR = "shap_bar.png"
SHAP_BEESWARM = "shap_beeswarm.png"
ANALYSIS_VIEW = "analysis_view.parquet"  # temporary Data Wrangler materialization
HPO_JSON = "hpo.json"                    # one Optuna study, beside manifest.json
TRIALS_SUMMARY = "trials_summary.parquet"
TRIAL_JSON = "trial.json"                # one trial, inside that trial's model folder

#: Old run-level files. Readers still accept them. New objective runs do not write them.
LEGACY_RUN_JSON = "run.json"
LEGACY_CONTRACT_JSON = "contract.json"
LEGACY_FOLDS_JSON = "folds.json"

#: Columns a compact OOF always keeps when they are present. Feature columns
#: are everything else the model trained on, and they are not stored on the OOF.
OOF_KEY_COLUMNS: tuple[str, ...] = ("ts", "session", "side")
OOF_PREDICTION_PREFIXES: tuple[str, ...] = ("p_", "tau_", "mv_")
OOF_OUTCOME_COLUMNS: frozenset[str] = frozenset({
    "fold", "weight", "reached", "t_move", "t_event", "t_reach", "t_stop", "t_fill",
    "mfe_pts", "mae_pts", "price", "r_value", "event", "pnl_pts", "pnl_r",
    "tgt_dist_pts", "tgt_dist_r", "stp_dist_pts", "stp_dist_r",
    "fill_px", "decision_price", "stop_px", "tgt_level", "tgt_rank",
})

#: model.json records one fitted configuration. The feature list is this
#: model's; another model on the same population names its own columns.
MODEL_JSON_FIELDS: tuple[str, ...] = (
    "model_id", "model_type", "hyperparameters", "target",
    "feature_columns", "inputs",
)

#: How a joined feature or context row says when its value became known.
#: ``timestamp``: the time column is that instant. ``bar_close``: the time
#: column is the bar open and the value is known only after ``bar_duration``.
#: ``session_close``: the row summarizes a session and is known only at that
#: session's close. Predictions are decision-time scores (``timestamp``).
AVAILABILITY_TIMESTAMP = "timestamp"
AVAILABILITY_BAR_CLOSE = "bar_close"
AVAILABILITY_SESSION_CLOSE = "session_close"
FEATURE_AVAILABILITY: frozenset[str] = frozenset({
    AVAILABILITY_TIMESTAMP, AVAILABILITY_BAR_CLOSE, AVAILABILITY_SESSION_CLOSE,
})


def _nonempty(v: Any) -> bool:
    return not (v is None or v == {} or v == [] or v == ())


def _hpo_study(m: Any) -> bool:
    """True when this run is an Optuna study.

    The marker is ``config_snapshot.study.kind == "hpo"``. The snapshot still
    holds the shared objective. Each trial's hyperparameters stay in that
    trial's ``model.json``, so ``config_snapshot_resolved`` stays false.
    """
    snap = getattr(m, "config_snapshot", None)
    if not isinstance(snap, Mapping):
        return False
    study = snap.get("study")
    return isinstance(study, Mapping) and study.get("kind") == "hpo"


def _ran(responsibility: str) -> Callable[[Any], bool]:
    """Applicable iff the manifest declares it exercised this responsibility.

    Deliberately NOT inferred from ``execution_assumptions``: a run records
    its cost assumptions as configuration even when it performed no economic
    evaluation (e.g. a Track B custom_test search), and treating that as
    "economics ran" would demand artifacts for a step that never happened.
    """
    def check(m: Any) -> bool:
        return any(k == responsibility or k.startswith(responsibility + ":")
                   for k in (m.responsibility_versions or {}))
    return check


def _calibrated(m: Any) -> bool:
    c = m.calibration or {}
    return _nonempty(c) and str(c.get("method", "")).lower() not in ("", "none")


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_class: str
    filename: str                      # canonical name for NEW writes
    fmt: str                           # json | parquet | md | embedded
    applies: Callable[[Any], bool]     # derived from the RunManifest
    required_content: tuple[str, ...]  # minimum top-level keys / columns
    description: str
    comparable: bool = False           # diffed leaf-by-leaf by compare_results


ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("manifest", "manifest.json", "json", lambda m: True,
                 ("run_id", "objective_id", "code", "datasets", "config_snapshot",
                  "validation", "seeds", "execution_assumptions",
                  "responsibility_versions", "artifacts", "evidence_level",
                  "missing_evidence"),
                 "Run/objective identity, provenance, artifact inventory, evidence completeness."),
    ArtifactSpec("config_snapshot", "manifest.json#config_snapshot", "embedded", lambda m: True,
                 (), "Configuration recorded for this run. Embedded in the manifest (never a "
                     "pointer to the source YAML); no second file. The class name matches the "
                     "manifest field it realises -- it is NOT named 'resolved_config': whether "
                     "this snapshot is genuinely the fully-resolved configuration is a separate, "
                     "explicit claim (RunManifest.config_snapshot_resolved), which can be False "
                     "for a non-empty snapshot. See docs/EXPERIMENTS.md §1."),
    ArtifactSpec("metrics", "metrics.json", "json", lambda m: True,
                 ("n_samples", "metrics"),
                 "Aggregate metrics (schemas.RunMetrics). n_samples names the population.",
                 comparable=True),
    ArtifactSpec("fold_metrics", "fold_metrics.json", "json",
                 lambda m: m.validation is not None, ("folds",),
                 "Per-fold schemas.FoldMetrics records (fold_id, train/test bounds, "
                 "n_train, n_test, metrics).", comparable=True),
    ArtifactSpec("oof_predictions", "oof_predictions.parquet", "parquet",
                 lambda m: m.validation is not None,
                 OOF_PREDICTION_COLUMNS,
                 "Out-of-fold predictions. Row identity = decision ts; fold_id "
                 "recommended. Tied to the run by living in its directory + manifest map."),
    ArtifactSpec("selected_population", "selected_population.json", "json",
                 lambda m: _nonempty(m.decision_policy),
                 ("n_raw", "n_selected", "selection_rule"),
                 "Decision/gating selection (schemas.SelectedPopulation); detects "
                 "population change between runs.", comparable=True),
    ArtifactSpec("calibration", "calibration.json", "json", _calibrated,
                 ("method", "n_samples", "bucket_edges", "predicted_rate",
                  "observed_rate", "brier_score"),
                 "schemas.CalibrationResult. Not applicable when calibration.method is none.",
                 comparable=True),
    ArtifactSpec("trade_ledger", "trade_ledger.parquet", "parquet",
                 _ran("economic_evaluation"),
                 ("trade_id", "direction", "quantity", "entry_order_type",
                  "exit_order_type", "entry_price", "exit_price", "entry_ts",
                  "exit_ts", "gross_dollars", "commission", "net_dollars"),
                 "economic.TradeResult rows. Assumptions (incl. economic_contract_unit "
                 "and underlying) live in economic_metrics.json."),
    ArtifactSpec("economic_metrics", "economic_metrics.json", "json",
                 _ran("economic_evaluation"),
                 ("assumptions", "n_trades", "gross_dollars", "total_commission",
                  "net_dollars"),
                 "economic.TradeLedger aggregates, never separated from their assumptions.",
                 comparable=True),
    ArtifactSpec("cost_sensitivity", "cost_sensitivity.json", "json",
                 _ran("economic_evaluation"), (),
                 "Scenario label -> economic_metrics-shaped ledger summary.",
                 comparable=True),
    ArtifactSpec("feature_importance", "feature_importance/", "embedded",
                 _ran("feature_importance"), (),
                 "Realised today by src/evaluation/shap_explain.py's SHAP/ directory; "
                 "reused, not re-specified."),
    ArtifactSpec("effective_sample_size", "effective_sample_size.json", "json",
                 lambda m: False, ("n_trades", "n_eff", "method"),
                 "Dependence-aware n_eff. DEFERRED: never applicable until implemented."),
    ArtifactSpec("report", "report.md", "md", lambda m: False, (),
                 "Human-readable conclusions. Recommended; not required, so its "
                 "absence is not reported as missing evidence."),
    ArtifactSpec(
        "model", "model.json", "json",
        lambda m: any(bool(p.get("models")) for p in (getattr(m, "passes", None) or ())),
        ("model_id", "model_type", "hyperparameters", "target"),
        "One fitted model: type, hyperparameters, target, the feature columns "
        "that model trained on, and the as-of join spec for its inputs. The file "
        "lives under pass_<n>/model_<id>/; the manifest lists each relative path."),
    ArtifactSpec(
        "hpo_study", "hpo.json", "json", _hpo_study,
        ("sampler", "seed", "search_space", "metric", "n_trials",
         "pass_n", "selected_model_id"),
        "One Optuna study on this run. About ten trials share one last-range "
        "fold. selected_model_id points at the winning pass_<n>/model_<id>/."),
    ArtifactSpec(
        "trials_summary", "trials_summary.parquet", "parquet", _hpo_study,
        (),
        "One row per Optuna trial: trial_number, model_id, status, score, params."),
)

BY_CLASS: Mapping[str, ArtifactSpec] = {a.artifact_class: a for a in ARTIFACTS}


#: Existing files in Track B run directories, classified. Nothing here is
#: renamed or removed. ``realises`` names the canonical class a file already
#: satisfies, so the manifest can map to it instead of duplicating it.
COMPATIBILITY_ARTIFACTS: tuple[tuple[str, str, str | None], ...] = (
    # pattern, status, realises
    ("run.json", "compatibility_consumed", None),        # old runs; manifest absorbs it
    ("contract.json", "compatibility_consumed", None),   # old runs; config_snapshot absorbs it
    ("search_result.json", "compatibility_consumed", None),
    ("folds.json", "compatibility_consumed", None),      # old runs; validation absorbs the design
    ("spec.manual.yaml", "compatibility_consumed", None),
    ("topology.yaml", "compatibility_consumed", None),
    ("label_contract.json", "canonical", None),          # dataset lineage (label_store)
    ("oof_*.parquet", "compatibility_consumed", "oof_predictions"),
    ("trades.parquet", "superseded_migration_candidate", "trade_ledger"),
    ("SHAP", "canonical", "feature_importance"),
    ("SHAP/*", "canonical", "feature_importance"),
    ("diagnostics_*.json", "implementation_supplemental", None),
    ("diagnostics.json", "implementation_supplemental", None),
    # Written by scripts/research/migrate_custom_test.py: which source bundle a
    # migrated model folder came from, and that bundle's own finalize record.
    # Per-model detail only -- the run-level commit lives on manifest.code.
    ("migration_provenance.json", "implementation_supplemental", None),
    ("progress*.json", "implementation_supplemental", None),  # in-flight telemetry
    ("backtest_report.json", "superseded_migration_candidate", "economic_metrics"),
    ("models", "implementation_supplemental", None),
    ("training_matrix.parquet", "optional_cache", None),
    ("*/training_matrix.parquet", "optional_cache", None),
    ("analysis_view.parquet", "temporary", None),
    ("*/analysis_view.parquet", "temporary", None),
    ("trial.json", "canonical", None),
    ("*/trial.json", "canonical", None),
    # Track B serving-bundle files. A migrated model folder keeps them beside
    # the weights: scaler.pkl is applied before the model at replay, and
    # feature_schema.json / finalize_manifest.json stay until
    # src/serving/bundle.py reads model.json instead.
    ("scaler.pkl", "canonical", None),
    ("feature_schema.json", "compatibility_consumed", "model"),
    ("finalize_manifest.json", "compatibility_consumed", "model"),
    ("compose.json", "canonical", None),
    ("inference_policy.json", "canonical", None),
    ("meta.json", "canonical", None),
    ("archetype_stats.json", "implementation_supplemental", None),
    ("folds", "implementation_supplemental", None),
    ("bases", "implementation_supplemental", None),
    # Report set. shap_explain.py's multi-class output and the eval scripts'
    # prose summary land under the model's report/ directory.
    ("summary.md", "canonical", "report"),
    ("shap_values_sample.npz", "reproducible_derived", "feature_importance"),
    ("shap_*", "canonical", "feature_importance"),
    ("SHAP_SUMMARY.md", "canonical", "feature_importance"),
)


def classify_existing(relpath: str) -> tuple[str, str | None]:
    """(status, realised canonical class) for an existing run-dir file.

    A pattern matches the path as given or its final component, so a file
    inside a Track B bundle (``models/<candidate>/model.keras``) classifies
    the same as the migrated copy beside ``model.json``.
    """
    name = Path(relpath).name
    for spec in ARTIFACTS:
        if relpath == spec.filename or name == spec.filename:
            return "canonical", spec.artifact_class
    if name in MODEL_ARTIFACT_FILENAMES:
        return "canonical", "model_artifact"
    for pattern, status, realises in COMPATIBILITY_ARTIFACTS:
        if fnmatch.fnmatch(relpath, pattern) or fnmatch.fnmatch(name, pattern):
            return status, realises
    return "unclassified", None


def applicable(manifest: Any) -> dict[str, bool]:
    """Canonical class -> whether this run is expected to produce it."""
    return {a.artifact_class: bool(a.applies(manifest)) for a in ARTIFACTS}


def validate_content(spec: ArtifactSpec, payload: Any) -> list[str]:
    """Minimum-content check. Returns missing top-level keys (empty = valid)."""
    if not spec.required_content or spec.fmt != "json":
        return []
    if not isinstance(payload, Mapping):
        return list(spec.required_content)
    return [k for k in spec.required_content if k not in payload]


def _prediction_column(name: str) -> bool:
    return name.startswith(OOF_PREDICTION_PREFIXES) or name.startswith("y_")


def compact_oof_columns(columns: list[str], feature_columns: list[str]) -> list[str]:
    """OOF columns with the model's feature columns removed.

    Keeps row keys, the fold id, outcome fields, predictions, and train-derived
    thresholds. A feature that is also a prediction or an outcome is kept.
    """
    features = set(feature_columns)
    kept: list[str] = []
    for name in columns:
        if name in features and name not in OOF_OUTCOME_COLUMNS and not _prediction_column(name):
            if name not in OOF_KEY_COLUMNS and name not in {"fold", "weight"}:
                continue
        kept.append(name)
    return kept


def _input_sources(inputs: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    block = inputs.get(kind) or []
    if isinstance(block, Mapping):
        return [block]
    if isinstance(block, (list, tuple)):
        return [item for item in block if isinstance(item, Mapping)]
    return []


def _source_spec_errors(source: Mapping[str, Any], kind: str) -> list[str]:
    """A feature or context source must say when each row became known."""
    avail = source.get("availability")
    if kind == "predictions":
        if avail not in (None, AVAILABILITY_TIMESTAMP):
            return ["predictions.availability"]
        return []
    if avail not in FEATURE_AVAILABILITY:
        return [f"{kind}.availability"]
    errors: list[str] = []
    if avail == AVAILABILITY_BAR_CLOSE and not source.get("bar_duration"):
        errors.append(f"{kind}.bar_duration")
    if avail == AVAILABILITY_SESSION_CLOSE:
        if not source.get("session_close"):
            errors.append(f"{kind}.session_close")
        if not source.get("timezone"):
            errors.append(f"{kind}.timezone")
    return errors


def validate_model_inputs(inputs: Mapping[str, Any] | None) -> list[str]:
    """Missing pieces of a model's join spec. Empty means the spec is usable.

    A spec names labels, feature datasets, optional context, optional upstream
    predictions, join keys, filters, and (when the population was materialized
    once) the optional cache. It does not require ``training_matrix.parquet``.

    Every feature and context source declares ``availability`` so the join can
    refuse a value that is knowable only after the decision timestamp. Labels
    are not given an availability: the outcome may use the forward path.
    """
    if not isinstance(inputs, Mapping) or not inputs:
        return ["inputs"]
    missing: list[str] = []
    if "join_keys" not in inputs:
        missing.append("join_keys")
    has_source = any(inputs.get(key) for key in ("labels", "features", "context", "predictions"))
    if not has_source and not inputs.get("population_cache"):
        missing.append("labels|features|population_cache")
    joined = False
    for kind in ("features", "context", "predictions"):
        sources = _input_sources(inputs, kind)
        if sources:
            joined = True
        for source in sources:
            missing.extend(_source_spec_errors(source, kind))
    keys = inputs.get("join_keys") or []
    if joined and "ts" not in keys:
        missing.append("join_keys.ts")
    return missing
