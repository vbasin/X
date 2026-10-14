"""15s dual-binary labels from existing 5s TBM labels (ADR-066)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.training.aux_labels import attach_aux_labels
from src.training.config import LabelConfig
from src.training.data_pipeline import ensure_utc_index
from src.training.labels import encode_dual_binary, generate_labels, on_timestep_grid

DECISION_TIMESTEP_S = 15
SRC_TIMESTEP_S = 5


def tbm_dual_binary_15s(
    close: pd.Series,
    *,
    move_threshold: float = 50.0,
    mae_tolerance: float = 6.0,
    horizon_s: int = 600,
    src_step_s: int = SRC_TIMESTEP_S,
    decision_step_s: int = DECISION_TIMESTEP_S,
    scheme: str = "triple_barrier",
    target_observation: str = "candle_close",
    target_high: pd.Series | None = None,
    target_low: pd.Series | None = None,
) -> pd.DataFrame:
    """Return ``y_long`` and ``y_short`` on the 15s grid. Both columns on every row."""
    close = pd.Series(close, copy=False)
    close.index = pd.to_datetime(close.index, utc=True)
    window = max(1, int(horizon_s) // max(1, int(src_step_s)))
    cfg = LabelConfig(
        big_move_threshold=float(move_threshold),
        mae_tolerance=float(mae_tolerance),
        forward_windows=[window],
        scheme=scheme,
        target_observation=target_observation,
    )
    y = generate_labels(close, cfg, target_high=target_high, target_low=target_low)["label"]
    grid = on_timestep_grid(y.index, decision_step_s)
    out = encode_dual_binary(y.loc[grid])
    # Aux is computed on the 5s path (same as TBM) then aligned to the 15s grid.
    out = attach_aux_labels(
        out, close,
        move_threshold=float(move_threshold),
        horizon_s=int(horizon_s),
        src_step_s=int(src_step_s),
        target_observation=target_observation,
        target_high=target_high,
        target_low=target_low,
    )
    out.index.name = "ts"
    return out


def labels_from_5s_close(
    close: pd.Series,
    *,
    target_high: pd.Series | None = None,
    target_low: pd.Series | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return tbm_dual_binary_15s(
        close, target_high=target_high, target_low=target_low, **kwargs)


def labels_from_5s_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    df = ensure_utc_index(pd.read_parquet(path, columns=["_close"]))
    if "_close" not in df.columns:
        raise ValueError(f"{path} has no _close column")
    return labels_from_5s_close(df["_close"], **kwargs)


def label_params_from_objective(objective: dict[str, Any]) -> dict[str, Any]:
    lbl = objective.get("labels") or {}
    hor = objective.get("horizon") or {}
    raw = str(hor.get("max_trade_horizon") or "10m").strip().lower()
    if raw.endswith("m"):
        horizon_s = int(float(raw[:-1]) * 60)
    elif raw.endswith("s"):
        horizon_s = int(raw[:-1])
    else:
        horizon_s = 600
    return {
        "move_threshold": float(lbl.get("move_threshold", 50.0)),
        "mae_tolerance": float(lbl.get("mae_tolerance", 6.0)),
        "horizon_s": horizon_s,
        "scheme": str(lbl.get("scheme") or "triple_barrier"),
        "target_observation": str(lbl.get("target_observation") or "candle_close"),
    }
