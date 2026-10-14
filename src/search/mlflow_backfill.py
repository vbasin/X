"""Backfill MLflow from persisted search runs — ADR-024 observability.

Replays ``data/research/search/<objective_id>/<run_ts>/search_result.json`` files into the
``search/<objective_id>`` MLflow experiment via the controller's wave logger, so
historical waves (run before MLflow logging existed) are browsable in the UI
exactly like new ones. Idempotence guard: a wave whose ``run_ts`` tag already
exists in the experiment is skipped.

Usage::

    python -m src.search.mlflow_backfill --objective nq-50pt-tbm            # all runs
    python -m src.search.mlflow_backfill --objective nq-50pt-tbm --run <dir>
"""

from __future__ import annotations

from src.common.data_paths import data_root, search_dir

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("vtech.search.mlflow_backfill")

_DATA_ROOT = data_root()


def _existing_wave_ts(objective_id: str) -> set[str]:
    """run_ts tags already present in the search/<objective_id> experiment."""
    try:
        import mlflow
        from src.training.config import MLflowConfig
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               MLflowConfig().tracking_uri))
        exp = mlflow.get_experiment_by_name(f"search/{objective_id}")
        if exp is None:
            return set()
        runs = mlflow.search_runs(
            [exp.experiment_id], filter_string="tags.experiment_type != 'finalize'",
            output_format="list")
        return {r.data.tags.get("run_ts", "") for r in runs}
    except Exception as exc:  # noqa: BLE001
        logger.warning("mlflow_lookup_failed (%s) — proceeding without dedupe", exc)
        return set()


def backfill(objective_id: str, run_dir: str | None = None) -> int:
    from src.objectives.loader import load_objective
    from src.search.controller import CandidateResult, _log_wave_to_mlflow

    objective = load_objective(objective_id)
    # historical backfill: resolve across roots, canonical first
    from src.common.run_layout import run_roots
    root = next((r / objective_id for r in run_roots()
                 if (r / objective_id).is_dir()), search_dir() / objective_id)
    run_dirs = ([Path(run_dir)] if run_dir else
                sorted(d for d in root.glob("*Z") if (d / "search_result.json").exists()))
    seen = _existing_wave_ts(objective_id)

    n = 0
    for rd in run_dirs:
        summary = json.loads((rd / "search_result.json").read_text())
        run_ts = summary.get("run_ts", rd.name)
        if run_ts in seen:
            print(f"⏭  {rd.name} — already in MLflow, skipping")
            continue
        results = [
            CandidateResult(
                candidate_id=c["candidate_id"], family=c["family"],
                spec_params=c.get("spec_params") or {}, status=c.get("status", "ok"),
                score=c.get("score") or {}, compute_class=c.get("compute_class", "cpu"),
                elapsed_s=float(c.get("elapsed_s") or 0.0),
                feature_selection=c.get("feature_selection"),
            )
            for c in summary.get("candidates", [])
        ]
        _log_wave_to_mlflow(results, objective, run_ts,
                            experiment_type=summary.get("experiment_type", "model_selection"))
        ok = sum(1 for r in results if r.status == "ok")
        print(f"✅ {rd.name} — {ok}/{len(results)} candidates logged")
        n += 1
    print(f"Backfill complete: {n} wave(s) logged to MLflow experiment search/{objective_id}")
    return n


def _cli() -> None:
    p = argparse.ArgumentParser(description="Backfill MLflow from persisted search runs")
    p.add_argument("--objective", required=True)
    p.add_argument("--run", default="", help="Single run dir (default: all runs)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    backfill(args.objective, args.run or None)


if __name__ == "__main__":
    _cli()
