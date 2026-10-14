"""T2 evidence adapters — read-only internal-experimental sources (ADR-025 §2).

NOT WIRED INTO PASS B (2026-07-15 directive). Pass B now integrates only user-confirmed
PROVEN-SUCCESS research (the T3 journal — see src.scoping.journal.load_proven_evidence);
un-confirmed experimental data (a Scoping "champion" from these adapters is a starting base
model, not a finalized success) must not drive it. These adapters are retained as reusable,
read-only queries — a future synthesis-stage warm-start hint could consume them as explicitly
NON-authoritative context — but nothing in the scoping loop calls them today.

Machine-generated, no gate. Four thin adapters over existing subsystems (never re-implement)
plus one aggregate digest:

    query_results_index    warm-start priors from prior searches (src.search.results_index)
    query_leaderboard      current standings vs the eval contract (src.evaluation.leaderboard)
    query_episodes_patterns  mined success/failure patterns (src.evaluation.{episodes,patterns})
    query_mlflow           run history from MLflow experiment search/<objective_id>

Every adapter degrades to an {"error": …} payload — evidence being absent is itself
information for the agents, never a crash.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("vtech.scoping.evidence")


def query_results_index(objective_id: str, match: str = "base") -> dict[str, Any]:
    """Warm-start priors aggregated from prior search runs over similar objectives."""
    try:
        from src.objectives.loader import load_objective
        from src.search.results_index import ResultsIndex
        return ResultsIndex().query_priors(load_objective(objective_id), match=match)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def query_leaderboard(objective_id: str, k: int = 10) -> list[dict[str, Any]] | dict[str, Any]:
    """Top standings for this objective under its latest eval contract."""
    try:
        from src.evaluation.contract import EvalContract
        from src.evaluation.leaderboard import get_leaders
        c = EvalContract.load_latest(objective_id)
        version = getattr(c, "contract_version", "") if c else ""
        if not version:
            return {"error": "no eval contract yet for this objective"}
        return get_leaders(objective_id, version, k=k)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def query_episodes_patterns(objective_id: str) -> dict[str, Any]:
    """Mined outcome patterns (win/loss separations, blind spots, ToD effects) from the
    latest recorded episodes — the outcome-report if present, else mined on the fly."""
    try:
        from src.evaluation.episodes import latest_outcome_reports, load_episodes
        reports = latest_outcome_reports(objective_id, limit=1)
        if reports:
            return reports[0]
        ep = load_episodes(objective_id)
        if ep is None or ep.empty:
            return {"error": "no recorded episodes for this objective"}
        from src.evaluation.patterns import mine_patterns
        return mine_patterns(ep)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def query_mlflow(objective_id: str, max_runs: int = 25) -> list[dict[str, Any]] | dict[str, Any]:
    """Compact rows from the ``search/<objective_id>`` MLflow experiment, best first."""
    try:
        import mlflow
        from src.training.config import MLflowConfig
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLflowConfig().tracking_uri))
        df = mlflow.search_runs(
            experiment_names=[f"search/{objective_id}"],
            filter_string="tags.candidate_id != ''",     # nested candidate runs only
            order_by=["metrics.precision_at_tau DESC"],
            max_results=max_runs,
        )
        if df is None or df.empty:
            return {"error": "no MLflow runs for this objective"}
        cols = {
            "run_id": "run_id",
            "tags.candidate_id": "candidate_id", "tags.family": "family",
            "tags.run_ts": "run_ts",
            "metrics.precision_at_tau": "precision_at_tau",
            "metrics.directional_f1": "directional_f1",
            "metrics.elapsed_s": "elapsed_s",
        }
        keep = [c for c in cols if c in df.columns]
        out = df[keep].rename(columns={c: cols[c] for c in keep})
        return out.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def t2_digest(objective_id: str, *, mlflow_max_runs: int = 25) -> dict[str, Any]:
    """The Pass-B1 evidence bundle: all four T2 sources in one compact dict."""
    return {
        "tier": "T2",
        "objective_id": objective_id,
        "results_index_priors": query_results_index(objective_id),
        "leaderboard": query_leaderboard(objective_id),
        "outcome_patterns": query_episodes_patterns(objective_id),
        "mlflow_runs": query_mlflow(objective_id, max_runs=mlflow_max_runs),
    }
