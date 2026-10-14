"""Paths for one objective run.

Artifact names and the join/OOF rules live in
``src.experiment.artifacts``. This module resolves paths and
allocates model ids. It does not define a second filename contract.

One objective run::

    research/runs/<objective_id>/<run_ts>/
        manifest.json                         passes, resolved objective, fold design
        pass_<n>/training_matrix.parquet      optional cache of one row population
        pass_<n>/model_<model_id>/
            model.json                        type, hyperparameters, target, features, join
            oof_predictions.parquet           keys, outcomes, predictions; no feature columns
            metrics.json
            fold_metrics.json
            report/summary.md
            report/shap_*

An Optuna study is the same run, written by
``scripts/search/hpo_model_search.py``. ``manifest.validation`` records the
folds that study scored. It does not force ``last_range`` / ``n_folds=1``.
The study writes trial metadata, not a per-trial OOF or a population cache::

    research/runs/<objective_id>/<hpo_run_ts>/
        manifest.json
        hpo.json
        trials_summary.parquet
        pass_<n>/model_<model_id>/
            trial.json
            model.json
            metrics.json

``model_id`` names one configuration. A second feature subset is another
``model.json`` over the same population, not another matrix. The run records
the label contract id and does not write ``applies_to`` onto that contract.

``run.json``, ``contract.json`` and ``folds.json`` are read for runs that
have no manifest. New runs do not write them.

This module resolves paths. It holds no model-type, strategy or pass-role
vocabulary: model identity and ``model.json`` content are
``src.experiment.model_identity``; the cascade's own pass roles and
feature views are ``src.training.cascade_run_spec``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiment.model_identity import validate_model_id, write_json
from src.experiment.artifacts import (
    COST_SENSITIVITY_JSON,
    ECONOMIC_METRICS_JSON,
    FOLD_METRICS_JSON,
    LEGACY_CONTRACT_JSON as CONTRACT_JSON,
    LEGACY_FOLDS_JSON as FOLDS_JSON,
    LEGACY_RUN_JSON as RUN_JSON,
    METRICS_JSON,
    MODEL_ARTIFACT,
    MODEL_JSON,
    HPO_JSON,
    OOF_PREDICTIONS,
    POPULATION_CACHE as TRAINING_MATRIX,
    TRIAL_JSON,
    TRIALS_SUMMARY,
    REPORT_DIR,
    SELECTED_POPULATION_JSON,
    SHAP_BAR,
    SHAP_BEESWARM,
    SHAP_IMPORTANCE,
    SUMMARY_MD,
    TRADE_LEDGER,
)
from src.common.data_paths import runs_dir

def new_run_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def objective_dir(objective_id: str) -> Path:
    return runs_dir() / objective_id


def run_dir(objective_id: str, run_ts: str) -> Path:
    return objective_dir(objective_id) / run_ts


def pass_dir(objective_id: str, run_ts: str, pass_n: int) -> Path:
    n = int(pass_n)
    if n < 1:
        raise ValueError(f"pass must be >= 1, got {pass_n}")
    return run_dir(objective_id, run_ts) / f"pass_{n}"


def model_dir(objective_id: str, run_ts: str, pass_n: int, model_id: str,
              model_type: str | None = None) -> Path:
    mid = validate_model_id(model_id, model_type)
    return pass_dir(objective_id, run_ts, pass_n) / f"model_{mid}"


def population_cache_path(objective_id: str, run_ts: str, pass_n: int) -> Path:
    """Optional single materialization of one pass's row population.

    Shared by every model on that pass. A different feature list does not
    get its own file.
    """
    return pass_dir(objective_id, run_ts, pass_n) / TRAINING_MATRIX


def training_matrix_path(objective_id: str, run_ts: str, pass_n: int, model_id: str) -> Path:
    """Legacy per-model cache. New writes use :func:`population_cache_path`."""
    return model_dir(objective_id, run_ts, pass_n, model_id) / TRAINING_MATRIX


def oof_predictions_path(objective_id: str, run_ts: str, pass_n: int, model_id: str) -> Path:
    return model_dir(objective_id, run_ts, pass_n, model_id) / OOF_PREDICTIONS


def report_dir(objective_id: str, run_ts: str, pass_n: int, model_id: str) -> Path:
    return model_dir(objective_id, run_ts, pass_n, model_id) / REPORT_DIR


def hpo_json_path(objective_id: str, run_ts: str) -> Path:
    return run_dir(objective_id, run_ts) / HPO_JSON


def trials_summary_path(objective_id: str, run_ts: str) -> Path:
    return run_dir(objective_id, run_ts) / TRIALS_SUMMARY


def resolve_model(objective_id: str, run_ts: str, pass_n: int, model_id: str) -> dict[str, Path]:
    """Every standard file for one model. Callers use the paths they produce."""
    d = model_dir(objective_id, run_ts, pass_n, model_id)
    report = d / REPORT_DIR
    return {
        "model_dir": d,
        "training_matrix": d / TRAINING_MATRIX,
        "oof_predictions": d / OOF_PREDICTIONS,
        "model_json": d / MODEL_JSON,
        "trial": d / TRIAL_JSON,
        "model_artifact": d / MODEL_ARTIFACT,
        "metrics": d / METRICS_JSON,
        "fold_metrics": d / FOLD_METRICS_JSON,
        "selected_population": d / SELECTED_POPULATION_JSON,
        "trade_ledger": d / TRADE_LEDGER,
        "economic_metrics": d / ECONOMIC_METRICS_JSON,
        "cost_sensitivity": d / COST_SENSITIVITY_JSON,
        "report_dir": report,
        "summary": report / SUMMARY_MD,
        "shap_importance": report / SHAP_IMPORTANCE,
        "shap_bar": report / SHAP_BAR,
        "shap_beeswarm": report / SHAP_BEESWARM,
    }


def read_run(objective_id: str, run_ts: str) -> dict[str, Any]:
    """Pass list from ``manifest.json``, or from a legacy ``run.json``."""
    from src.experiment.manifest import read_run_record
    return read_run_record(run_dir(objective_id, run_ts))


def merge_run_json(
    objective_id: str,
    run_ts: str,
    pass_entry: dict[str, Any],
    *,
    config_snapshot: dict[str, Any] | None = None,
    config_snapshot_resolved: bool | None = None,
    validation: dict[str, Any] | None = None,
    seeds: dict[str, int] | None = None,
    execution_assumptions: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    code: Any = None,
) -> Path:
    """Insert or update one pass inside ``manifest.json``.

    Does not write ``run.json``, ``contract.json``, or ``folds.json``.
    Does not open the label contract. Models are matched by ``model_id``.
    """
    from src.experiment.manifest import merge_pass
    return merge_pass(
        run_dir(objective_id, run_ts),
        objective_id=objective_id,
        run_ts=run_ts,
        pass_entry=pass_entry,
        config_snapshot=config_snapshot,
        config_snapshot_resolved=config_snapshot_resolved,
        validation=validation,
        seeds=seeds,
        execution_assumptions=execution_assumptions,
        extra=extra,
        code=code,
    )


def previous_model(objective_id: str, run_ts: str, pass_n: int) -> tuple[int, str]:
    """The single model recorded on the pass before ``pass_n``."""
    doc = read_run(objective_id, run_ts)
    prev_n = int(pass_n) - 1
    if prev_n < 1:
        raise FileNotFoundError(f"pass {pass_n} has no previous pass")
    prev = next((p for p in doc.get("passes") or [] if int(p.get("pass", -1)) == prev_n), None)
    if prev is None:
        raise FileNotFoundError(
            f"pass {prev_n} is not in {objective_id}/{run_ts}/manifest.json")
    models = list(prev.get("models") or [])
    if len(models) != 1:
        raise FileNotFoundError(
            f"pass {prev_n} has {len(models)} models; name the upstream model")
    return prev_n, str(models[0]["model_id"])


def resolve_population_cache(objective_id: str, run_ts: str, pass_n: int) -> Path | None:
    """Pass-level population cache, or a legacy copy inside the model folder."""
    cache = population_cache_path(objective_id, run_ts, pass_n)
    if cache.is_file():
        return cache
    try:
        doc = read_run(objective_id, run_ts)
    except FileNotFoundError:
        return None
    entry = next((p for p in doc.get("passes") or [] if int(p.get("pass", -1)) == int(pass_n)), None)
    if entry is None:
        return None
    root = run_dir(objective_id, run_ts)
    recorded = entry.get("population_cache")
    if recorded:
        path = root / str(recorded)
        if path.is_file():
            return path
    for model in entry.get("models") or []:
        mid = model.get("model_id")
        if mid:
            legacy = training_matrix_path(objective_id, run_ts, pass_n, str(mid))
            if legacy.is_file():
                return legacy
        rel = model.get("training_matrix")
        if rel:
            path = root / str(rel)
            if path.is_file():
                return path
    return None


def previous_oof_core(objective_id: str, run_ts: str, pass_n: int) -> tuple[Path, list[str]]:
    """Path and column names of the previous pass's compact OOF."""
    prev_n, mid = previous_model(objective_id, run_ts, pass_n)
    path = oof_predictions_path(objective_id, run_ts, prev_n, mid)
    meta_path = model_dir(objective_id, run_ts, prev_n, mid) / MODEL_JSON
    cols: list[str] | None = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        cols = meta.get("oof_columns") or meta.get("oof_core_columns")
    if not cols:
        if not path.is_file():
            raise FileNotFoundError(meta_path if meta_path else path)
        import pyarrow.parquet as pq
        cols = list(pq.read_schema(path).names)
    return path, list(cols)


def previous_training_matrix(objective_id: str, run_ts: str, pass_n: int) -> Path:
    """Previous pass's population cache, including a legacy per-model file."""
    prev_n, _mid = previous_model(objective_id, run_ts, pass_n)
    found = resolve_population_cache(objective_id, run_ts, prev_n)
    if found is None:
        raise FileNotFoundError(population_cache_path(objective_id, run_ts, prev_n))
    return found


def read_previous_oof(objective_id: str, run_ts: str, pass_n: int):
    """Previous pass's OOF projected to ``oof_core_columns``."""
    import pandas as pd
    path, cols = previous_oof_core(objective_id, run_ts, pass_n)
    return pd.read_parquet(path, columns=cols)


def empty_index(objective_id: str) -> dict[str, Any]:
    return {
        "objective_id": objective_id,
        "leaderboard": [],
        "staged": [],
        "promoted": None,
    }


def write_index(objective_id: str, doc: dict[str, Any] | None = None) -> Path:
    return write_json(objective_dir(objective_id) / "index.json", doc or empty_index(objective_id))


def compact_parquets(sources: list[Path], dest: Path) -> int:
    """Stream ``sources`` into one zstd parquet at ``dest``. Returns row count."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    files = [p for p in sources if p.is_file()]
    if not files:
        raise FileNotFoundError("no parquet files to compact")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    dataset = ds.dataset([str(p) for p in files], format="parquet")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in dataset.to_batches():
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
        if writer is None:
            raise FileNotFoundError("parquet sources were empty")
        writer.close()
        writer = None
        tmp.replace(dest)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists() and not dest.exists():
            tmp.unlink()
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# One resolver for every research run, whichever tree it lives in
# ══════════════════════════════════════════════════════════════════════════════
#
# research/runs/ is the primary output for every NEW run. The other two roots
# are read-only history:
#
#   research/search/       still referenced by staged/registered bundles, so it
#                          stays a live compatibility source until those models
#                          are repointed or retired
#
# research/custom_test/ is GONE: every fitted run was migrated into
# research/runs/ and the directory was deleted (2026-09-23). It is still listed
# below only so a machine that has not run the migration keeps working; on a
# machine where it does not exist it is skipped.
#
# Consumers call resolve_run_dir()/run_roots() instead of each learning the
# layout, so retiring a root later is one edit here rather than a sweep.

def run_roots() -> tuple[Path, ...]:
    """Every root a research run may live under, primary (new writes) first.

    A root that does not exist is skipped, so retiring one needs no other edit.
    """
    from src.common.data_paths import custom_test_dir, search_dir

    roots = [runs_dir()]
    for legacy in (custom_test_dir(), search_dir()):
        if legacy not in roots and legacy.is_dir():
            roots.append(legacy)
    return tuple(roots)


def new_run_dir(objective_id: str, run_ts: str) -> Path:
    """Where a NEW run of any workflow is written. Always ``research/runs/``."""
    return run_dir(objective_id, run_ts)


def resolve_run_dir(objective_id: str, run_ts: str) -> Path:
    """An existing run's directory, searching the canonical root first.

    Raises ``FileNotFoundError`` naming every root tried, rather than returning
    a path that does not exist -- a silently wrong run directory is how an
    evaluation ends up scoring the wrong model.
    """
    tried: list[Path] = []
    for root in run_roots():
        candidate = root / str(objective_id) / str(run_ts)
        tried.append(candidate)
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no run directory for {objective_id}/{run_ts}; tried "
        + ", ".join(str(p) for p in tried)
    )


def find_run_dir(objective_id: str, run_ts: str) -> Path | None:
    """:func:`resolve_run_dir` without raising."""
    try:
        return resolve_run_dir(objective_id, run_ts)
    except FileNotFoundError:
        return None


def iter_run_dirs(*, objective_id: str | None = None) -> list[Path]:
    """Every run directory across all roots, newest run_ts first.

    A run that exists in more than one root (a migrated Track B run) is
    returned once, from the canonical root.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Path] = []
    for root in run_roots():
        if not root.is_dir():
            continue
        for objective in sorted(p for p in root.iterdir() if p.is_dir()):
            if objective_id is not None and objective.name != objective_id:
                continue
            for run in sorted(objective.iterdir(), reverse=True):
                if not run.is_dir() or run.name.startswith("_"):
                    continue
                key = (objective.name, run.name)
                if key in seen:
                    continue
                seen.add(key)
                out.append(run)
    return out
