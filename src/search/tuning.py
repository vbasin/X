"""Per-candidate HPO inner loop — Phase 23 W8.

Optional bounded tuning pass that runs *before the champion gate judges a candidate*, so a
genuinely capable family is not discarded for a fixable reason (e.g. xgboost collapsing to
NO_SIGNAL purely because it had no class weighting — the W3/W8 finding).

Deterministic Optuna search over the family's **declarative search space** (owned by the
catalog, ``CatalogEntry.search_space``), scored on the same **full walk-forward OOF** and the
same **precision-first, dual-preferring** objective used by ``scripts/hpo_model_search.py`` —
so tuning optimises the exact quantity the gate's ``requirement`` cares about (per-direction
precision@τ), never a fire-on-everything ``directional_f1`` proxy.

``tune_candidate`` returns the best parameter overrides; the controller merges them onto the
catalog defaults for the final fit. Off by default (``tune_trials=0``); the controller wires
it via the ``search.tune_trials`` knob.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

logger = logging.getLogger("vtech.search.tuning")

DEFAULT_PRECISION_BAR = 0.85
DEFAULT_MIN_SIGNALS = 25


def suggest_params(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Interpret a declarative catalog search space into Optuna ``suggest_*`` calls."""
    out: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec.get("type")
        if t == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif t == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown search-space type '{t}' for param '{name}'")
    return out


def dual_preferring_score(
    kpi: dict[str, Any], *, bar: float = DEFAULT_PRECISION_BAR, floor: int = DEFAULT_MIN_SIGNALS,
) -> tuple[float, str]:
    """Tiered, dual-direction-preferring objective (precision-first; recall/F1 ignored).

    Strict tiers (base 30/20/10) so a dual model always beats a single specialist, which
    always beats a below-bar model:

      DUAL  (30): both up & down precision@τ >= bar with n>=floor each → maximise avg + volume.
      ONE   (20): exactly one direction clears the bar (specialist).
      BELOW (10): a direction fires >= floor but below the bar → climb.
      NONE  (<0): nothing fires >= floor → climb toward the floor.
    """
    def _dir_ok(p, n: int) -> bool:
        return n >= floor and p is not None and not _isnan(p)

    up_p, up_n = kpi.get("precision_up"),   int(kpi.get("n_up") or 0)
    dn_p, dn_n = kpi.get("precision_down"), int(kpi.get("n_down") or 0)
    up_ok, dn_ok = _dir_ok(up_p, up_n), _dir_ok(dn_p, dn_n)
    up_bar = up_ok and float(up_p) >= bar
    dn_bar = dn_ok and float(dn_p) >= bar

    if up_bar and dn_bar:
        return 30.0 + (float(up_p) + float(dn_p)) / 2 + 0.15 * math.log10(up_n + dn_n), "dual"
    if up_bar or dn_bar:
        p, n = (float(up_p), up_n) if up_bar else (float(dn_p), dn_n)
        return 20.0 + p + 0.15 * math.log10(n), ("up" if up_bar else "down")
    elig = []
    if up_ok:
        elig.append((float(up_p), up_n))
    if dn_ok:
        elig.append((float(dn_p), dn_n))
    if elig:
        bp, bn = max(elig)
        return 10.0 + bp + 0.15 * math.log10(bn), "below_bar"
    reached = max(up_n, dn_n)
    return -1.0 + 0.5 * min(reached / floor, 1.0), "none"


def tune_candidate(
    entry: Any,                       # CatalogEntry (duck-typed: .name, .search_space, .params_default)
    objective: dict[str, Any],
    folds: list[dict[str, Any]],
    *,
    n_trials: int,
    tau: float | None = None,
    bar: float = DEFAULT_PRECISION_BAR,
    floor: int = DEFAULT_MIN_SIGNALS,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Run a bounded Optuna search over ``entry.search_space`` on the given walk-forward folds.

    Returns ``{"best_params", "best_value", "best_mode", "n_trials"}`` (best_params = the tuned
    overrides to merge onto the catalog defaults), or ``None`` when tuning is off / unavailable /
    the family has no search space.
    """
    if n_trials <= 0 or not getattr(entry, "search_space", None):
        return None
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — skipping tuning for %s", entry.name)
        return None

    from src.evaluation._kpi import canonical_trade_kpi
    from src.evaluation.kpi_criteria import primary_tau
    from src.models.base import ModelSpec
    from src.search.controller import _load_plugin

    tau = tau if tau is not None else primary_tau(objective)
    plugin = _load_plugin(entry.family_key)
    defaults = dict(entry.params_default)

    def objective_fn(trial: "optuna.Trial") -> float:
        params = {**defaults, **suggest_params(trial, entry.search_space)}
        spec = ModelSpec(family=entry.family_key, params=params)
        model = plugin.build(spec, objective)
        fitted = plugin.fit(model, folds, objective=objective)
        oof = fitted.oof_preds
        if oof is None or "direction_true" not in oof.columns:
            return -1.0
        gt = pd.DataFrame({"direction_true": oof["direction_true"]})
        kpi = canonical_trade_kpi(oof, gt, objective, tau=tau)
        score, mode = dual_preferring_score(kpi, bar=bar, floor=floor)
        trial.set_user_attr("mode", mode)
        for k in ("precision_up", "precision_down", "n_up", "n_down", "precision_at_tau"):
            trial.set_user_attr(k, kpi.get(k))
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_fn, n_trials=n_trials)

    bt = study.best_trial
    logger.info("tune_candidate family=%s trials=%d best=%.4f mode=%s",
                entry.name, len(study.trials), study.best_value, bt.user_attrs.get("mode"))
    return {
        "best_params": dict(bt.params),
        "best_value": float(study.best_value),
        "best_mode": bt.user_attrs.get("mode"),
        "n_trials": len(study.trials),
    }


def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True
