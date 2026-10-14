"""Leaderboard writer — Phase 23 W2.

Appends a candidate's scored result to the leaderboard for a given
(objective_id, contract_version) pair.  The leaderboard is a JSONL file at::

    data/models/leaders/<objective_id>/leaderboard_v<contract_version>.jsonl

and a companion ``latest_leaders.json`` that always holds the current
top-k by primary precision. Frozen eval contracts stay under
``data/evaluation/<objective_id>/``.

Both are also surfaced as MLflow metrics on the parent Phase-23 run if MLflow
is available.

Usage::

    from src.evaluation.leaderboard import record_result

    record_result(
        contract=contract,
        candidate_id="xgboost_v1",
        model_family="xgboost",
        score_result=contract.score(predictions, ground_truth),
        spec=spec,
        mlflow_run_id=run.info.run_id,   # optional
    )
"""

from __future__ import annotations

from src.common.data_paths import leaders_dir

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.evaluation.leaderboard")

_TOP_K = 10  # entries in latest_leaders.json


def record_result(
    *,
    objective_id: str,
    contract_version: str,
    candidate_id: str,
    model_family: str,
    score_result: dict[str, Any],
    spec_params: dict[str, Any] | None = None,
    ensemble_members: list[str] | None = None,
    mlflow_run_id: str | None = None,
    notes: str = "",
) -> Path:
    """Append one scored candidate to the leaderboard.

    Parameters
    ----------
    objective_id, contract_version:
        Must match the contract used to produce *score_result*.
    candidate_id:
        Unique identifier for this candidate, e.g. ``"xgboost_v1_fold3"``.
    model_family:
        Catalog family key, e.g. ``"xgboost"``.
    score_result:
        Output of ``EvalContract.score()``.
    spec_params:
        Hyperparameter dict from the candidate's ``ModelSpec``.
    ensemble_members:
        For ensemble candidates, the base-model candidate_ids used.
    mlflow_run_id:
        Optional MLflow run to cross-link.
    notes:
        Free-form string (e.g. "OOF stacking, top-3 bases").

    Returns
    -------
    Path
        The leaderboard JSONL file path.
    """
    entry: dict[str, Any] = {
        "recorded_at":      datetime.now(timezone.utc).isoformat(),
        "objective_id":     objective_id,
        "contract_version": contract_version,
        "candidate_id":     candidate_id,
        "model_family":     model_family,
        "primary_precision": score_result.get("primary_precision"),
        "primary_dir_f1":   score_result.get("primary_dir_f1"),
        "primary_tau":      score_result.get("primary_tau"),
        "n_signals":        (score_result.get("per_tau") or [{}])[0].get("n_signals"),
        "full_scores":      score_result,
        "spec_params":      spec_params or {},
        "ensemble_members": ensemble_members or [],
        "mlflow_run_id":    mlflow_run_id,
        "notes":            notes,
    }

    lb_path = _leaderboard_path(objective_id, contract_version)
    lb_path.parent.mkdir(parents=True, exist_ok=True)

    with lb_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(
        "leaderboard_entry",
        extra={
            "candidate_id": candidate_id,
            "precision": entry["primary_precision"],
            "dir_f1": entry["primary_dir_f1"],
        },
    )

    _refresh_leaders(objective_id, contract_version)
    _log_to_mlflow(entry, mlflow_run_id)

    return lb_path


def load_leaderboard(objective_id: str, contract_version: str) -> list[dict[str, Any]]:
    """Return all entries for this (objective_id, contract_version), newest first."""
    lb_path = _leaderboard_path(objective_id, contract_version)
    if not lb_path.exists():
        return []
    entries = [json.loads(line) for line in lb_path.read_text().splitlines() if line.strip()]
    return sorted(entries, key=lambda e: e.get("recorded_at", ""), reverse=True)


def get_leaders(
    objective_id: str,
    contract_version: str,
    k: int = _TOP_K,
) -> list[dict[str, Any]]:
    """Return the top-*k* candidates by primary precision for this contract."""
    all_entries = load_leaderboard(objective_id, contract_version)
    # Best result per candidate_id (in case of multiple re-runs)
    by_id: dict[str, dict] = {}
    for e in all_entries:
        cid = e["candidate_id"]
        prec = e.get("primary_precision") or -1.0
        if prec > (by_id.get(cid, {}).get("primary_precision") or -1.0):
            by_id[cid] = e
    return sorted(by_id.values(), key=lambda e: e.get("primary_precision") or -1.0, reverse=True)[:k]


def champion(objective_id: str, contract_version: str) -> dict[str, Any] | None:
    """Return the top entry on the leaderboard, or None if empty."""
    leaders = get_leaders(objective_id, contract_version, k=1)
    return leaders[0] if leaders else None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _leaderboard_path(objective_id: str, contract_version: str) -> Path:
    return leaders_dir() / objective_id / f"leaderboard_v{contract_version}.jsonl"


def _refresh_leaders(objective_id: str, contract_version: str) -> None:
    leaders = get_leaders(objective_id, contract_version)
    out = leaders_dir() / objective_id / "latest_leaders.json"
    out.write_text(json.dumps(
        {
            "objective_id":     objective_id,
            "contract_version": contract_version,
            "updated_at":       datetime.now(timezone.utc).isoformat(),
            "leaders":          leaders,
        },
        indent=2,
    ))


def _log_to_mlflow(entry: dict[str, Any], run_id: str | None) -> None:
    """Log primary KPIs as MLflow metrics on *run_id* if MLflow is reachable."""
    if run_id is None:
        return
    try:
        import mlflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                k: v for k, v in {
                    "precision_at_tau": entry.get("primary_precision"),
                    "directional_f1":   entry.get("primary_dir_f1"),
                }.items() if v is not None
            })
    except Exception as exc:
        logger.debug("mlflow_log_skipped: %s", exc)
