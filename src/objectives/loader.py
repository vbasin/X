"""Objective loader — reads and validates a Trade Objective YAML against its JSON schema.

Usage::

    from src.objectives.loader import load_objective
    obj = load_objective("nq-50pt")          # searches config/objectives/<id>.yaml
    obj = load_objective("/full/path/to/file.yaml")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("vtech.objectives.loader")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OBJECTIVES_DIR = _REPO_ROOT / "config" / "objectives"
_SCHEMA_PATH = _OBJECTIVES_DIR / "objective.schema.json"


def _get_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text())


def load_objective(objective_id_or_path: str) -> dict[str, Any]:
    """Load and validate a Trade Objective.

    Parameters
    ----------
    objective_id_or_path:
        Either a bare objective ID (e.g. ``"nq-50pt"``) that resolves to
        ``config/objectives/<id>.yaml``, or an absolute/relative YAML path.

    Returns
    -------
    dict
        Parsed and schema-validated objective.

    Raises
    ------
    FileNotFoundError
        If the YAML file cannot be found.
    jsonschema.ValidationError
        If the YAML does not conform to ``objective.schema.json``.
    """
    p = Path(objective_id_or_path)
    if not p.suffix:
        # treat as a bare id
        p = _OBJECTIVES_DIR / f"{objective_id_or_path}.yaml"
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Objective file not found: {p}")

    obj: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    _validate(obj, p)
    _validate_champion_gate(obj, p)
    _validate_custom_optimization(obj, p)
    logger.info("objective_loaded", extra={"path": str(p), "objective_id": obj.get("objective_id")})
    return obj


def _validate(obj: dict[str, Any], source: Path) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        logger.warning("jsonschema not installed — objective schema validation skipped")
        return
    schema = _get_schema()
    errs = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(e.path))
    if errs:
        msgs = "; ".join(f"{list(e.path)}: {e.message}" for e in errs)
        raise ValueError(f"Objective {source} failed schema validation: {msgs}")


def _validate_champion_gate(obj: dict[str, Any], source: Path) -> None:
    """``champion_gate: true`` requires a defined ``kpi.requirement`` (ADR-071)."""
    kpi = obj.get("kpi") or {}
    if kpi.get("champion_gate") is not True:
        return
    from src.evaluation.kpi_criteria import requirement_conditions
    if not requirement_conditions(obj):
        raise ValueError(
            f"Objective {source} sets kpi.champion_gate: true but kpi.requirement "
            "has no conditions — declare the gate in YAML (do not rely on a code default)"
        )


def _validate_custom_optimization(obj: dict[str, Any], source: Path) -> None:
    """``custom_optimization: true`` requires a full 3×3 ``cost_matrix`` (ADR-072)."""
    if obj.get("custom_optimization") is not True:
        return
    from src.training.cost_matrix import parse_cost_matrix
    try:
        parse_cost_matrix(obj.get("cost_matrix"))
    except ValueError as exc:
        raise ValueError(f"Objective {source} custom_optimization is invalid: {exc}") from exc


def resolve_active_objective_id() -> str | None:
    """Discover the active objective when config.yaml does not name one.

    config.yaml is objective-agnostic (it carries no ``objective_id``), so the
    active objective is inferred from ``config/objectives/``: if exactly one
    objective YAML exists, its id (file stem) is returned. Returns ``None`` when
    the directory holds no objective. Raises ``ValueError`` if more than one
    objective is present, so a caller can never silently bind to the wrong one.
    """
    candidates = sorted(_OBJECTIVES_DIR.glob("*.yaml"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Multiple objective files in config/objectives/ "
            f"({[p.stem for p in candidates]}); cannot auto-select. "
            "Name one via `objective_id` in config.yaml."
        )
    return candidates[0].stem
