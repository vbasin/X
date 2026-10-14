"""Model identity and ``model.json`` content.

``src/common/run_layout.py`` resolves *where* a model's files live. This
module decides *what* identifies one model and what its metadata file
contains -- the two concerns were previously mixed into the path resolver,
which put model-type and stage vocabulary into a module whose only job is
directory arithmetic.

Filenames and the join/OOF rules stay in
``src.experiment.artifacts``. Nothing here defines a filename.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.experiment.artifacts import (
    MODEL_JSON,
    REPORT_DIR,
    compact_oof_columns,
)

#: Folder names that would make every model of one family share a directory.
#: This is a denylist for *naming*, not a model registry: an id that is only
#: the algorithm cannot distinguish two fits of that algorithm in one pass.
_TYPE_IDS = frozenset({
    "xgb", "xgboost", "lgbm", "lightgbm", "catboost", "torch", "keras",
})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return value
    try:
        import numpy as np
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
    except ImportError:
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str) + "\n")
    return path


def validate_model_id(model_id: str, model_type: str | None = None) -> str:
    """Reject a model id that is only the algorithm name."""
    mid = str(model_id).strip()
    banned = set(_TYPE_IDS)
    if model_type:
        banned.add(str(model_type).strip().lower())
    if (not mid or "/" in mid or "\\" in mid or mid.startswith(".")
            or mid.lower() in banned):
        raise ValueError(
            "model_id must name this trained model and must not be the model "
            f"type (got {model_id!r})"
        )
    return mid


def _canonicalize(value: Any, *, sort_string_lists: bool) -> Any:
    """JSON-ready value with dict keys sorted.

    String lists inside a feature spec are sets of names: order is not a
    different configuration. Hyperparameter lists keep their order.
    """
    value = _jsonable(value)
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item, sort_string_lists=sort_string_lists)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        items = [_canonicalize(item, sort_string_lists=sort_string_lists) for item in value]
        if sort_string_lists and all(isinstance(item, str) for item in items):
            return sorted(items)
        return items
    return value


def allocate_model_id(
    objective_id: str,
    run_ts: str,
    pass_n: int,
    model_type: str,
    hyperparameters: Mapping[str, Any],
    feature_spec: Any = None,
    target: str | None = None,
) -> str:
    """Stable id for one model configuration inside a pass.

    The same configuration always hashes to the same 12 hex characters.
    A different hyperparameter, feature view, or target name does not, so
    two fits of one model type in one pass do not share a folder.

    ``run_ts`` is the run id. Output paths and any other clock time are not
    inputs. Pass ``--model-id`` to override this hash; this function does
    not consult it.
    """
    if not isinstance(hyperparameters, Mapping):
        raise TypeError("hyperparameters must be a mapping of the resolved fit")
    payload = {
        "objective_id": str(objective_id),
        "run_ts": str(run_ts),
        "pass_n": int(pass_n),
        "model_type": str(model_type),
        "hyperparameters": _canonicalize(dict(hyperparameters), sort_string_lists=False),
        "feature_spec": _canonicalize(feature_spec, sort_string_lists=True),
        "target": None if target is None else str(target),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def write_model_json(
    directory: Path,
    *,
    model_id: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    label_contract_id: str | None = None,
    oof_core_columns: list[str] | None = None,
    oof_columns: list[str] | None = None,
    feature_spec: Any = None,
    feature_columns: list[str] | None = None,
    inputs: dict[str, Any] | None = None,
    target: str | None = None,
) -> Path:
    """Model identity. ``inputs`` is the join spec; ``feature_columns`` is this fit's list."""
    from src.experiment.artifacts import validate_model_inputs
    payload: dict[str, Any] = {
        "model_id": model_id,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
    }
    if target is not None:
        payload["target"] = target
    if feature_spec is not None:
        payload["feature_spec"] = feature_spec
    if feature_columns is not None:
        payload["feature_columns"] = list(feature_columns)
    if inputs is not None:
        missing = validate_model_inputs(inputs)
        if missing:
            raise ValueError(f"model inputs missing {missing}")
        payload["inputs"] = inputs
    if label_contract_id:
        payload["label_contract_id"] = label_contract_id
    stored = oof_columns if oof_columns is not None else oof_core_columns
    if stored:
        payload["oof_columns"] = list(stored)
        payload["oof_core_columns"] = list(stored)
    return write_json(directory / MODEL_JSON, payload)


def model_report_dir(directory: Path) -> Path:
    """Create this model's ``report/`` directory and return it.

    Callers create it when they are about to write a report into it. A run
    does not carry an empty ``report/`` as a placeholder -- see
    ``docs/STORAGE_CONTRACT.md`` §21.
    """
    report = directory / REPORT_DIR
    report.mkdir(parents=True, exist_ok=True)
    return report


def oof_core_columns(
    oof_columns: list[str],
    matrix_columns: list[str],
    label_columns: list[str],
) -> list[str]:
    """OOF columns with feature columns removed.

    Prediction columns stay even when the matrix also carries them. The next
    pass reads those predictions, then joins its own features.
    """
    labels = set(label_columns)
    features = [
        name for name in matrix_columns
        if name not in labels and not name.startswith(("p_", "tau_", "mv_"))
    ]
    return compact_oof_columns(list(oof_columns), features)
