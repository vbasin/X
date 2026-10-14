"""What the cascade's own passes and fits are called.

This is Track C implementation vocabulary -- pass roles, the XGBoost knobs
that identify a cascade fit, and the two feature views its builders share.
It deliberately does NOT live in ``src/common/run_layout.py``: a path
resolver that knows the words "move", "target" or "xgboost" has stopped
being a path resolver, and ADR-136 keeps strategy vocabulary out of the
canonical layer. Another strategy names its own passes and its own views.
"""

from __future__ import annotations

from typing import Any

#: What the cascade writers record on the manifest pass list. Other objectives
#: pass their own. Roles describe what a pass decides, not its position.
CASCADE_PASS: dict[int, dict[str, str]] = {
    1: {
        "role": "directional_screen",
        "purpose": (
            "Score each decision-grid row for whether price makes a favourable "
            "excursion of at least the objective's move threshold before the "
            "adverse excursion, within the horizon. This pass does not choose "
            "a price target or a stop."
        ),
    },
    2: {
        "role": "target_reach",
        "purpose": (
            "On the rows the previous pass scored, score whether the chosen "
            "target is reached before that target's stop. The upstream input "
            "is the previous pass's out-of-fold prediction columns only. "
            "This pass reads that file for the gate, then joins its own labels, "
            "features, and context. It does not reuse the previous pass's "
            "feature columns as its matrix."
        ),
    },
}


def xgb_hyperparameters(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    eval_metric: str = "logloss",
) -> dict[str, Any]:
    """Hyperparameters that identify an XGBoost fit.

    Per-fold ``scale_pos_weight`` and ``n_jobs`` are not part of the id.
    One is recomputed from the training labels. The other is the machine.
    """
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "seed": int(seed),
        "eval_metric": str(eval_metric),
    }


def move_feature_spec(*, drop_order_flow: bool = False) -> dict[str, Any]:
    """Feature view shared by the move-matrix build and the move eval."""
    return {
        "view": "numeric_non_leak",
        "drop_order_flow": bool(drop_order_flow),
        "exclude": ["r_value"],
    }


def target_feature_spec(*, context: str) -> dict[str, Any]:
    """Feature view shared by the target-matrix build and the target eval.

    ``context`` is the matrix build's column view (``curated``, ``all``,
    or ``none``), not a list of column names.
    """
    return {
        "view": "numeric_non_leak",
        "context": str(context),
    }
