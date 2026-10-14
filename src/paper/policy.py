"""Paper policy from objective YAML + serving dir_threshold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.common.data_paths import model_staging_dir
from src.objectives.loader import load_objective

EntryStyle = Literal["last", "join", "cross"]


@dataclass(frozen=True)
class PaperPolicy:
    """Live-paper rules: limit entry + OCO bracket (points)."""

    objective_id: str
    target_points: float
    stop_points: float
    dir_threshold: float
    cost_per_trade: float
    slippage_points: float
    max_hold_seconds: float | None
    eod_flat: bool
    max_open: int
    entry_style: EntryStyle
    entry_timeout_seconds: float
    session_tz: str
    trading_start: str  # HH:MM local — no new entries before
    trading_end: str  # HH:MM local — flat / no new entries after
    model_path: str | None
    symbol_parent: str  # e.g. NQ.FUT


def _horizon_to_seconds(horizon: str | None) -> float | None:
    if not horizon:
        return None
    s = str(horizon).strip().lower()
    if s[-1:].isalpha():
        n, u = float(s[:-1]), s[-1]
        mult = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(u, 1.0)
        return max(1.0, n * mult)
    return max(1.0, float(s))


def load_paper_policy(
    *,
    objective_id: str | None = None,
    model_path: str | Path | None = None,
    dir_threshold: float | None = None,
    entry_style: EntryStyle = "last",
    entry_timeout_seconds: float = 60.0,
) -> PaperPolicy:
    """Build policy from objective labels/costs (+ optional serving bundle thr)."""
    import yaml
    from pathlib import Path as P

    oid = objective_id
    if oid is None:
        cfg = yaml.safe_load((P(__file__).resolve().parents[2] / "config.yaml").read_text()) or {}
        oid = str(cfg.get("objective_id") or "nq-50pt-econ")
    obj = load_objective(oid)
    labels = obj.get("labels") or {}
    costs = obj.get("costs") or {}
    session = obj.get("session") or {}
    horizon = (obj.get("horizon") or {}).get("max_trade_horizon")

    thr = dir_threshold
    mpath = str(model_path) if model_path else None
    if mpath:
        try:
            from src.serving.bundle import load_serving_bundle
            bundle = load_serving_bundle(mpath, validate=False)
            if thr is None:
                thr = float(bundle.get("dir_thr", 0.5))
        except Exception:
            pass
    if thr is None:
        thr = 0.5

    slip = 0.0
    if str(costs.get("slippage_model", "")).startswith("fixed"):
        slip = 1.0
    slip = float(costs.get("slippage_points", slip))

    return PaperPolicy(
        objective_id=oid,
        target_points=float(labels.get("move_threshold", 50)),
        stop_points=float(labels.get("mae_tolerance", 6)),
        dir_threshold=float(thr),
        cost_per_trade=float(costs.get("cost_per_trade", 0)),
        slippage_points=slip,
        max_hold_seconds=_horizon_to_seconds(horizon),
        eod_flat=True,
        max_open=1,
        entry_style=entry_style,
        entry_timeout_seconds=float(entry_timeout_seconds),
        session_tz=str(session.get("timezone") or "America/Chicago"),
        trading_start=str(session.get("trading_start") or "08:30"),
        trading_end=str(session.get("trading_end") or "14:45"),
        model_path=mpath,
        symbol_parent="NQ.FUT",
    )


def resolve_default_model(objective_id: str) -> Path | None:
    """Registry LATEST for *objective_id*, else newest staging finalize bundle."""
    try:
        from src.serving.promote import resolve_latest_bundle
        reg = resolve_latest_bundle(objective_id)
        if reg is not None:
            return reg
    except Exception:
        pass
    root = model_staging_dir() / objective_id
    if not root.is_dir():
        return None
    best: Path | None = None
    best_m = -1.0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not (d / "feature_schema.json").exists():
            continue
        if not any((d / n).exists() for n in ("model.ubj", "model.pkl", "model.joblib")):
            continue
        m = d.stat().st_mtime
        if m > best_m:
            best_m, best = m, d
    return best
