"""Slim 1h / 1d regime pack for predictive bases and OOF meta (ADR-095).

Hive files still hold the 45-col HTF catalog (ADR-091). Consumers only
asof-join the columns that normalize a 50-point / 6-MAE event: vol regime,
trend, range position, prior-session location. Raw OHLC, Fib, floor pivots,
and SAR distances stay on disk and are not features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.features.htf_context")

HTF_PACK_ID = "regime_v1"
_PREDICTIVE_STORES = frozenset({"15s", "1m", "5m"})
_CONTEXT_TF = frozenset({"1h", "1d"})

# 1h: this Globex hour's trend / vol / location in the last ~session.
CONTEXT_1H: tuple[str, ...] = (
    "tr_ema_stack",
    "tr_slope_10",
    "tr_gap",
    "zz_signal",
    "rv_atr_ratio",
    "rv_vol_ratio",
    "sr_pos_in_range_20",
)
# 1d: prior Globex session regime + overnight gap + prior high/low.
CONTEXT_1D: tuple[str, ...] = (
    "tr_ema_stack",
    "tr_gap",
    "zz_signal",
    "rv_atr_ratio",
    "sr_prior_high_dist",
    "sr_prior_low_dist",
)
CONTEXT_BY_TF: dict[str, tuple[str, ...]] = {
    "1h": CONTEXT_1H,
    "1d": CONTEXT_1D,
}
CTX_PREFIX: dict[str, str] = {"1h": "h1_", "1d": "d1_"}
# Union, unique, 1h then 1d. Tests / catalogs that want "the pack".
CONTEXT_COLS: tuple[str, ...] = tuple(dict.fromkeys((*CONTEXT_1H, *CONTEXT_1D)))

_TF_TOLERANCE = {
    "1h": pd.Timedelta("65min"),
    "1d": pd.Timedelta("26h"),
}


def context_cols_for(tf: str) -> tuple[str, ...]:
    return CONTEXT_BY_TF.get(str(tf).strip(), ())


def wants_htf_on_store(objective: dict[str, Any] | None, store: str) -> bool:
    """True when a predictive native-TF store should see the slim HTF pack."""
    if str(store or "").strip() not in _PREDICTIVE_STORES:
        return False
    topo = ((objective or {}).get("search") or {}).get("topology") or {}
    for c in list(topo.get("context") or []):
        if not isinstance(c, dict):
            continue
        if str(c.get("native_tf") or "").strip() in _CONTEXT_TF:
            return True
    return False


def load_htf_frames(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """``{tf: DataFrame(ts, *context_cols)}`` clipped to [*start* − 5d, *end*]."""
    from src.features.stores import load_feature_frame, store_available

    pad0 = (pd.Timestamp(start) - pd.Timedelta(days=5)).date()
    pad1 = pd.Timestamp(end).date()
    out: dict[str, pd.DataFrame] = {}
    for tf in ("1h", "1d"):
        if not store_available(tf):
            logger.warning("htf_context: store %s unavailable — skip", tf)
            continue
        df = load_feature_frame(tf, pad0, pad1)
        if df is None or df.empty:
            continue
        want = [c for c in context_cols_for(tf) if c in df.columns]
        if not want:
            continue
        piece = df[want].copy()
        piece.index.name = "ts"
        out[tf] = piece.reset_index()
    return out


def asof_attach_context(
    aligned: pd.DataFrame,
    context: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Causal asof of slim HTF onto a frame that already has ``ts``."""
    if not context:
        return aligned
    out = aligned
    for tf, frame in context.items():
        tf = str(tf).strip()
        if tf not in _CONTEXT_TF or frame is None or len(frame) == 0:
            continue
        ctx = frame.copy()
        if "ts" not in ctx.columns:
            if isinstance(ctx.index, pd.DatetimeIndex):
                ctx = ctx.reset_index()
                if ctx.columns[0] != "ts":
                    ctx = ctx.rename(columns={ctx.columns[0]: "ts"})
            else:
                continue
        ctx["ts"] = pd.to_datetime(ctx["ts"], utc=True)
        ctx = ctx.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts")
        cols = [c for c in context_cols_for(tf) if c in ctx.columns
                and pd.api.types.is_numeric_dtype(ctx[c])]
        if not cols:
            continue
        prefix = CTX_PREFIX.get(tf, f"{tf}_")
        out = _asof_right(
            out, ctx, cols=cols,
            tolerance=_TF_TOLERANCE[tf], prefix=prefix,
        )
    return out


def attach_htf_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add prefixed slim HTF columns onto a UTC DatetimeIndex feature frame."""
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, utc=True)
    elif "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True)
    else:
        return df
    try:
        context = load_htf_frames(ts.min(), ts.max())
    except Exception as exc:
        logger.warning("htf_context: load failed: %s", exc)
        return df
    if not context:
        return df
    left = pd.DataFrame({"ts": ts})
    joined = asof_attach_context(left, context)
    extra = [c for c in joined.columns if c != "ts" and c not in df.columns]
    if not extra:
        return df
    block = joined[extra].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = df.copy()
    for c in extra:
        out[c] = block[c].to_numpy()
    return out


def _asof_right(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    cols: list[str],
    tolerance: pd.Timedelta,
    prefix: str,
) -> pd.DataFrame:
    piece = right[["ts", *cols]].copy()
    piece = piece.rename(columns={c: f"{prefix}{c}" for c in cols})
    left_s = left.sort_values("ts")
    piece_s = piece.sort_values("ts")
    return pd.merge_asof(
        left_s, piece_s, on="ts", direction="backward", tolerance=tolerance,
    )
