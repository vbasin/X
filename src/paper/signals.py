"""Model signals for paper — same serving path as Live tab.

Signals are **not** fill prices. Fills come only from the TBBO tick stream.
This module decides *when* to place a limit entry.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.data_paths import native_tf_dir

logger = logging.getLogger("vtech.paper.signals")


def load_feature_day(day: str, parquet_dir: Path | None = None) -> pd.DataFrame:
    if parquet_dir is not None:
        path = Path(parquet_dir) / f"features_{day}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "ts" in df.columns:
                    df = df.set_index("ts")
                elif "bar_end_ts" in df.columns:
                    df = df.set_index("bar_end_ts")
                df.index = pd.to_datetime(df.index, utc=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df.sort_index()
    from src.features.hive_5s import load_session_day

    df = load_session_day(day)
    if df is None or df.empty:
        raise FileNotFoundError(
            f"no 5s features for {day} under {native_tf_dir('5s')}"
        )
    return df.sort_index()


def score_feature_frame(df: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """Full-day canonical predictions (direction / confidence / ts)."""
    from src.serving.bundle import load_serving_bundle
    from src.serving.predict import predict_canonical

    bundle = load_serving_bundle(model_path, validate=False)
    preds = predict_canonical(df, bundle)
    preds = preds.set_index(pd.DatetimeIndex(preds["ts"]))
    return preds


def latest_signal(
    model_path: str | Path,
    *,
    day: str | None = None,
    rebuild_live: bool = False,
) -> dict[str, Any]:
    """Score the last feature bar (optionally rebuild today from TimescaleDB)."""
    from src.serving.bundle import load_serving_bundle
    from src.serving.predict import predict_canonical

    session = day or datetime.now(timezone.utc).date().isoformat()
    path: Path | None = None

    if rebuild_live:
        try:
            from src.features.engine import FeatureEngine
            built = FeatureEngine().build_date(date.fromisoformat(session))
            path = Path(built) if built else None
        except Exception as exc:
            logger.warning("paper_signal_rebuild_failed: %s", exc)

    if path is not None and path.exists() and path.suffix == ".parquet" and path.name.startswith("features_") and len(path.stem) == len("features_YYYY-MM-DD"):
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" in df.columns:
                df = df.set_index("ts")
            elif "bar_end_ts" in df.columns:
                df = df.set_index("bar_end_ts")
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        try:
            df = load_feature_day(session)
        except FileNotFoundError:
            return {"ok": False, "error": f"no features for {session}", "day": session}
    if df.empty:
        return {"ok": False, "error": "empty feature frame", "day": session}

    bundle = load_serving_bundle(model_path, validate=False)
    last = df.iloc[[-1]]
    pred = predict_canonical(last, bundle).iloc[0]
    return {
        "ok": True,
        "day": session,
        "ts": str(pred["ts"]),
        "direction": str(pred["direction"]),
        "confidence": float(pred["confidence"]),
        "n_bars": len(df),
        "model": str(model_path),
        "feature_path": str(path),
    }


class SignalCursor:
    """Walk precomputed predictions in time as ticks advance (replay mode)."""

    def __init__(self, preds: pd.DataFrame) -> None:
        self._preds = preds.sort_index()
        self._idx = 0
        self._times = list(self._preds.index)
        self._n = len(self._times)

    def signals_through(self, ts: datetime) -> list[tuple[datetime, str, float]]:
        """Return new (ts, direction, conf) rows with bar_ts <= market *ts*."""
        out: list[tuple[datetime, str, float]] = []
        while self._idx < self._n and self._times[self._idx] <= ts:
            t = self._times[self._idx]
            row = self._preds.iloc[self._idx]
            out.append((
                t.to_pydatetime() if hasattr(t, "to_pydatetime") else t,
                str(row["direction"]),
                float(row["confidence"]),
            ))
            self._idx += 1
        return out
