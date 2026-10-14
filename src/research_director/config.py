"""Research Director config loader — ``RDConfig.load()`` (referenced by ``run.py``).

Merges the program context (``config.yaml``) with the RD control-plane knobs
(``config/research_director.yaml``). The cognitive LOOP knobs (fields, caps, providers, cost,
memory) live under the ``rd_loop:`` block and are loaded separately by
``src.research_director.orchestrator.load_rd_cfg`` — this config only carries what the cycle DRIVER
(program identity, objective, model registry, Governor, MLflow) needs.

Pure (no DB/LLM/network), so it is safe to load in ``--dry-run`` mode. Reuses ``pyyaml``.
"""

from __future__ import annotations

from src.common.data_paths import model_registry_dir

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path("/opt/vtech")


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml  # noqa: PLC0415
    return yaml.safe_load(path.read_text()) or {}


@dataclass(frozen=True)
class RDConfig:
    """Cycle driver config. Single pin: ``objective_id`` from ``config.yaml`` only."""

    objective_id: str
    # Alias of objective_id for journal/path keys that still say "program"
    program_id: str
    model_name: str  # informational only — baseline is registry LATEST, not name-matched
    program_objectives: dict[str, Any]
    labels: dict[str, Any]
    registry_dir: str
    external_search_enabled: bool
    mlflow_enabled: bool
    mlflow_tracking_uri: str | None
    mlflow_experiment_name: str
    gov_min_gap_s: int
    gov_act_cap_24h_s: int
    gov_usd_cap_day: float
    gov_usd_cap_week: float

    @classmethod
    def load(cls, *, rd_path: Path | None = None, app_path: Path | None = None) -> "RDConfig":
        from src.common.config_paths import runtime_path as _runtime_path
        rd = _yaml(rd_path or _runtime_path("research_director.yaml"))
        app = _yaml(app_path or _REPO / "config.yaml")

        research = rd.get("research", {}) or {}
        mlflow_rd = rd.get("mlflow", {}) or {}
        gov = rd.get("governor", {}) or {}

        app_mlflow = app.get("mlflow", {}) or {}
        paths = app.get("paths", {}) or {}
        labels = app.get("labels", {}) or {}

        # Single pin: config.yaml objective_id (no env override — SSOT).
        objective_id = str(app.get("objective_id") or "nq-50pt-econ")
        program_objectives = {
            "objectives": [objective_id],
            "success_thresholds": app.get("success thresholds", {}),
            "labels": labels,
            "primary_metric": "precision_at_tau",
        }

        return cls(
            objective_id=objective_id,
            program_id=objective_id,  # no separate program concept
            model_name="",  # unused for baseline resolution
            program_objectives=program_objectives,
            labels=labels,
            registry_dir=paths.get("model_registry_dir", str(model_registry_dir())),
            external_search_enabled=bool(research.get("external_search_enabled", True)),
            mlflow_enabled=bool(mlflow_rd.get("log_iterations", True)),
            mlflow_tracking_uri=app_mlflow.get("tracking_uri"),
            mlflow_experiment_name=app_mlflow.get("experiment_name", "nq-50pt-intraday"),
            gov_min_gap_s=int(gov.get("min_gap_s", 1800)),
            gov_act_cap_24h_s=int(gov.get("act_cap_24h_s", 4 * 3600)),
            gov_usd_cap_day=float(gov.get("usd_cap_day", 20.0)),
            gov_usd_cap_week=float(gov.get("usd_cap_week", 80.0)),
        )
