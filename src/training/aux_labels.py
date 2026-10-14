"""Path-quality auxiliary labels for dual-binary heads (ADR-082).

Computed on the same close series as TBM (5s for both 5s and 15s stores;
15s rows are a grid subsample). Primary ``y_long`` / ``y_short`` stay the
first-touch event; these columns are dense regression targets on every bar.

Units: MFE / MAE / forward returns in NQ points; ``Time_to_TGT_*`` in seconds
to first ±threshold touch. No touch inside the horizon → NaN (not the
window cap). Huber zero-weights those rows (ADR-098).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd

AUX_LABEL_COLS: tuple[str, ...] = (
    "MFE_long_10m",
    "MFE_short_10m",
    "MAE_b4_TGTorMFE_long_10m",
    "MAE_b4_TGTorMFE_short_10m",
    "forward_return_1m",
    "forward_return_5m",
    "forward_return_10m",
    "MAE_before_target_or_MFE",
    "Time_to_TGT_long",
    "Time_to_TGT_short",
)

# Keras output name → parquet column → sign applied after per-fold z-score
# (ADR-086). Sign −1 makes short_forward_return_* = −z(forward_return_*), so
# a +20 pt rally is a positive long target and a negative short target.
# Shared MAE_before_target_or_MFE is parquet-only (each trunk already has
# MAE_b4_TGTorMFE_* for its side).
LONG_AUX_MAP: tuple[tuple[str, str, float], ...] = (
    ("MFE_long_10m", "MFE_long_10m", 1.0),
    ("MAE_b4_TGTorMFE_long_10m", "MAE_b4_TGTorMFE_long_10m", 1.0),
    ("Time_to_TGT_long", "Time_to_TGT_long", 1.0),
    ("long_forward_return_1m", "forward_return_1m", 1.0),
    ("long_forward_return_5m", "forward_return_5m", 1.0),
    ("long_forward_return_10m", "forward_return_10m", 1.0),
)
SHORT_AUX_MAP: tuple[tuple[str, str, float], ...] = (
    ("MFE_short_10m", "MFE_short_10m", 1.0),
    ("MAE_b4_TGTorMFE_short_10m", "MAE_b4_TGTorMFE_short_10m", 1.0),
    ("Time_to_TGT_short", "Time_to_TGT_short", 1.0),
    ("short_forward_return_1m", "forward_return_1m", -1.0),
    ("short_forward_return_5m", "forward_return_5m", -1.0),
    ("short_forward_return_10m", "forward_return_10m", -1.0),
)


def iter_aux_map(
    *maps: tuple[tuple[str, str, float], ...],
) -> Iterator[tuple[str, str, float]]:
    """Yield ``(keras_name, parquet_col, sign)`` from one or more aux maps."""
    rows = maps or (LONG_AUX_MAP, SHORT_AUX_MAP)
    for mapping in rows:
        for item in mapping:
            name, src, sign = item[0], item[1], float(item[2])
            if sign not in (-1.0, 1.0):
                raise ValueError(f"aux map sign for {name} must be ±1, got {sign}")
            yield name, src, sign

PRIMARY_LOSS_WEIGHT = 1.0
DIR_AUX_LOSS_WEIGHT = 0.15
SHARED_AUX_LOSS_WEIGHT = 0.05
# Dense path-side ranking on p_long vs p_short (ADR-088). Not a serve head.
SIDE_RANK_NAME = "side_rank"
SIDE_RANK_LOSS_WEIGHT = 0.3
# Binary competing-risk: MAE on that side reached the 6-pt stop first (ADR-095).
STOP_FIRST_LONG = "long_stop_first"
STOP_FIRST_SHORT = "short_stop_first"
STOP_FIRST_LOSS_WEIGHT = 0.25
STOP_FIRST_MAE_COL: dict[str, str] = {
    STOP_FIRST_LONG: "MAE_b4_TGTorMFE_long_10m",
    STOP_FIRST_SHORT: "MAE_b4_TGTorMFE_short_10m",
}
DEFAULT_MAE_TOLERANCE = 6.0
# Sample weight = clip(|MFE_long − MFE_short| / this, 0, 1). A 50-pt path
# winner counts fully; 1-pt chop barely moves the gradient. Rows are not dropped.
SIDE_RANK_SCALE_PTS = 50.0

_DIR_AUX_NAMES = {
    "MFE_long_10m", "MFE_short_10m",
    "MAE_b4_TGTorMFE_long_10m", "MAE_b4_TGTorMFE_short_10m",
    "Time_to_TGT_long", "Time_to_TGT_short",
}


def aux_from_objective(objective: dict[str, Any] | None) -> tuple[str, ...] | None:
    """Opt-in aux columns from ``labels.aux``, or ``None`` if the objective omits them.

    Omitted / empty / false → no aux heads (3-class LSTMs, other dual-binary
    objectives). ``true`` → the full :data:`AUX_LABEL_COLS` set. A list must be
    a subset of those names (unknown names raise).
    """
    raw = ((objective or {}).get("labels") or {}).get("aux")
    if raw is None or raw is False:
        return None
    if raw is True:
        return AUX_LABEL_COLS
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"labels.aux must be a list of column names, got {type(raw).__name__}")
    if not raw:
        return None
    names = [str(x) for x in raw]
    unknown = [n for n in names if n not in AUX_LABEL_COLS]
    if unknown:
        raise ValueError(
            f"labels.aux unknown columns: {unknown}; allowed: {list(AUX_LABEL_COLS)}")
    want = set(names)
    return tuple(c for c in AUX_LABEL_COLS if c in want)


def aux_loss_weight(output_name: str) -> float:
    if output_name in ("p_long", "p_short"):
        return PRIMARY_LOSS_WEIGHT
    if output_name == SIDE_RANK_NAME:
        return SIDE_RANK_LOSS_WEIGHT
    if output_name in (STOP_FIRST_LONG, STOP_FIRST_SHORT):
        return STOP_FIRST_LOSS_WEIGHT
    if output_name in _DIR_AUX_NAMES:
        return DIR_AUX_LOSS_WEIGHT
    return SHARED_AUX_LOSS_WEIGHT


def stop_first_enabled(columns: tuple[str, ...] | list[str] | None) -> bool:
    """True when MAE-before columns are in the requested aux set (or full set)."""
    if columns is None:
        return True
    want = set(columns)
    return all(src in want for src in STOP_FIRST_MAE_COL.values())


def path_side_rank_target(aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense path-side target from ``MFE_long_10m`` vs ``MFE_short_10m``.

    ``y = 1`` when long MFE strictly exceeds short MFE (else 0). Sample
    weight is ``clip(|Δ| / 50, 0, 1)`` so a TBM-sized excursion counts as
    1 and a 1-pt chop is ~0.02. Ties / NaN get weight 0 (``y`` is unused).
    """
    a = np.asarray(aux, dtype="float64")
    i_l = AUX_LABEL_COLS.index("MFE_long_10m")
    i_s = AUX_LABEL_COLS.index("MFE_short_10m")
    if a.ndim != 2 or a.shape[1] <= max(i_l, i_s):
        n = int(a.shape[0]) if a.ndim >= 1 else 0
        return np.zeros((n, 1), dtype="float32"), np.zeros(n, dtype="float64")
    delta = a[:, i_l] - a[:, i_s]
    y = (delta > 0.0).astype("float32").reshape(-1, 1)
    mag = np.abs(delta)
    w = np.where(
        np.isfinite(delta) & (mag > 0.0),
        np.clip(mag / SIDE_RANK_SCALE_PTS, 0.0, 1.0),
        0.0,
    )
    return y, w.astype("float64")


def _bars(horizon_s: int, src_step_s: int) -> int:
    return max(1, int(horizon_s) // max(1, int(src_step_s)))


def compute_aux_labels(
    close: pd.Series,
    *,
    move_threshold: float = 50.0,
    horizon_s: int = 600,
    src_step_s: int = 5,
    target_observation: str = "candle_close",
    target_high: pd.Series | None = None,
    target_low: pd.Series | None = None,
) -> pd.DataFrame:
    """Dense path-quality columns aligned with *close*.

    MFE is the peak favorable close excursion in ``(t, t+horizon]``.
    ``MAE_b4_TGTorMFE_*`` is the peak adverse close excursion from *t* until
    the earlier of (first target touch on that side, the bar of that side's
    MFE). ``MAE_before_target_or_MFE`` is the MAE of the side with the larger
    MFE. Forward returns are close-to-close point deltas.
    """
    close = pd.Series(close, copy=False)
    close.index = pd.to_datetime(close.index, utc=True)
    p = pd.to_numeric(close, errors="coerce").to_numpy(dtype="float64")
    mode = str(target_observation or "candle_close").lower()
    if mode not in {"candle_close", "candle_hilo", "tbbo"}:
        raise ValueError(f"unsupported target_observation={target_observation!r}")
    if mode == "candle_close":
        high = low = p
    else:
        if target_high is None or target_low is None:
            raise ValueError(f"target_observation={mode!r} requires target_high and target_low")
        high = pd.to_numeric(target_high.reindex(close.index), errors="coerce").to_numpy(dtype="float64")
        low = pd.to_numeric(target_low.reindex(close.index), errors="coerce").to_numpy(dtype="float64")
    n = int(p.size)
    win = _bars(horizon_s, src_step_s)
    step = max(1, int(src_step_s))
    thr = float(move_threshold)
    b1 = _bars(60, step)
    b5 = _bars(300, step)
    b10 = _bars(600, step)

    out = {c: np.full(n, np.nan, dtype="float64") for c in AUX_LABEL_COLS}
    if n < 2:
        return pd.DataFrame(out, index=close.index)

    # Pad so every t has a (win,)-shaped future; tail is NaN.
    pad = np.full(win, np.nan, dtype="float64")
    ext = np.concatenate([p, pad])
    from numpy.lib.stride_tricks import sliding_window_view

    view = sliding_window_view(ext, win + 1)[:n]
    future = view[:, 1:]
    rel = future - p[:, None]
    high_ext = np.concatenate([high, pad])
    low_ext = np.concatenate([low, pad])
    high_rel = sliding_window_view(high_ext, win + 1)[:n, 1:] - p[:, None]
    low_rel = sliding_window_view(low_ext, win + 1)[:n, 1:] - p[:, None]
    finite = np.isfinite(rel)
    has_future = finite.any(axis=1)

    rel_l = np.where(finite, rel, -np.inf)
    rel_s = np.where(finite, -rel, -np.inf)
    mfe_l = np.where(has_future, np.fmax(rel_l.max(axis=1), 0.0), np.nan)
    mfe_s = np.where(has_future, np.fmax(rel_s.max(axis=1), 0.0), np.nan)

    hit_l = finite & (high_rel >= thr)
    hit_s = finite & (low_rel <= -thr)
    any_l = hit_l.any(axis=1)
    any_s = hit_s.any(axis=1)
    k_l = np.argmax(hit_l, axis=1)
    k_s = np.argmax(hit_s, axis=1)

    # argmax on all-False is 0; gated by any_*. Timeout is NaN, never the
    # horizon cap (the window end is not a target — ADR-098).
    ttt_l = np.where(any_l, (k_l + 1).astype("float64") * step, np.nan)
    ttt_s = np.where(any_s, (k_s + 1).astype("float64") * step, np.nan)
    ttt_l = np.where(has_future & any_l, ttt_l, np.nan)
    ttt_s = np.where(has_future & any_s, ttt_s, np.nan)

    i_mfe_l = np.argmax(rel_l, axis=1)
    i_mfe_s = np.argmax(rel_s, axis=1)
    end_l = np.where(any_l, np.minimum(k_l, i_mfe_l), i_mfe_l)
    end_s = np.where(any_s, np.minimum(k_s, i_mfe_s), i_mfe_s)

    k = np.arange(win, dtype=np.int32)[None, :]
    mask_l = finite & (k <= end_l[:, None])
    mask_s = finite & (k <= end_s[:, None])
    adverse_l = np.where(mask_l, -rel, -np.inf)
    adverse_s = np.where(mask_s, rel, -np.inf)
    mae_l = np.where(mask_l.any(axis=1), np.fmax(adverse_l.max(axis=1), 0.0), np.nan)
    mae_s = np.where(mask_s.any(axis=1), np.fmax(adverse_s.max(axis=1), 0.0), np.nan)

    idx = np.arange(n)
    for bars, col in ((b1, "forward_return_1m"), (b5, "forward_return_5m"),
                      (b10, "forward_return_10m")):
        ok = (idx + bars < n) & np.isfinite(p) & np.isfinite(p[np.clip(idx + bars, 0, n - 1)])
        dest = idx[ok]
        out[col][dest] = p[dest + bars] - p[dest]

    out["MFE_long_10m"] = mfe_l
    out["MFE_short_10m"] = mfe_s
    out["MAE_b4_TGTorMFE_long_10m"] = mae_l
    out["MAE_b4_TGTorMFE_short_10m"] = mae_s
    out["Time_to_TGT_long"] = ttt_l
    out["Time_to_TGT_short"] = ttt_s
    prefer_long = np.where(np.isnan(mfe_l) & np.isnan(mfe_s), False, mfe_l >= mfe_s)
    out["MAE_before_target_or_MFE"] = np.where(prefer_long, mae_l, mae_s)

    frame = pd.DataFrame(out, index=close.index)
    return frame.astype("float32")


def sanitize_time_to_tgt(
    df: pd.DataFrame,
    *,
    horizon_s: float = 600.0,
) -> pd.DataFrame:
    """NaN ``Time_to_TGT_*`` unless that side actually printed ±threshold.

    Stale parquets stored the window cap (600) on timeouts. The cap is not a
    target. A value ≥ *horizon_s* with no confirming hit is NaN. Last-bar
    genuine hits (y_long/y_short or 3-class BIG_UP/BIG_DOWN) keep their time.
    """
    if df is None or df.empty:
        return df
    hor = float(horizon_s) if horizon_s and horizon_s > 0 else 600.0
    out = df
    wrote = False

    def _hit_long(frame: pd.DataFrame) -> pd.Series | None:
        if "y_long" in frame.columns:
            return pd.to_numeric(frame["y_long"], errors="coerce").fillna(0.0) >= 0.5
        if "label" in frame.columns:
            from src.training.labels import BIG_UP
            return pd.to_numeric(frame["label"], errors="coerce") == BIG_UP
        return None

    def _hit_short(frame: pd.DataFrame) -> pd.Series | None:
        if "y_short" in frame.columns:
            return pd.to_numeric(frame["y_short"], errors="coerce").fillna(0.0) >= 0.5
        if "label" in frame.columns:
            from src.training.labels import BIG_DOWN
            return pd.to_numeric(frame["label"], errors="coerce") == BIG_DOWN
        return None

    for col, hit_fn in (
        ("Time_to_TGT_long", _hit_long),
        ("Time_to_TGT_short", _hit_short),
    ):
        if col not in df.columns:
            continue
        t = pd.to_numeric(df[col], errors="coerce")
        hit = hit_fn(df)
        if hit is not None:
            # Timeouts stored as 600 (or any ≥ horizon) on non-hits → NaN.
            keep = t.notna() & (hit.to_numpy() | (t.to_numpy() < hor))
            t = t.where(keep, np.nan)
        else:
            t = t.where(t.notna() & (t < hor), np.nan)
        if not wrote:
            out = df.copy()
            wrote = True
        out[col] = t
    return out


def attach_aux_labels(
    frame: pd.DataFrame,
    close: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    """Left-join aux columns onto *frame* (same index)."""
    aux = compute_aux_labels(close, **kwargs)
    out = frame.copy()
    for c in AUX_LABEL_COLS:
        out[c] = aux[c].reindex(out.index)
    return sanitize_time_to_tgt(out, horizon_s=float(kwargs.get("horizon_s") or 600))


def aux_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """``(N, 10)`` float32 in :data:`AUX_LABEL_COLS` order, or None if missing."""
    missing = [c for c in AUX_LABEL_COLS if c not in df.columns]
    if missing:
        return None
    clean = sanitize_time_to_tgt(df)
    return clean[list(AUX_LABEL_COLS)].to_numpy(dtype="float32")


def zscore_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean / std ignoring NaN. Std floored at 1e-6."""
    a = np.asarray(arr, dtype="float64")
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-6), sd, 1.0)
    return mu.astype("float32"), sd.astype("float32")


def zscore_apply(arr: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype="float32")
    return (a - mu.astype("float32")) / sd.astype("float32")


def pack_aux_outputs(
    aux: np.ndarray | None,
    *,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    columns: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Keras y-dict for aux heads. NaN → 0 (pair with zero sample weight)."""
    if aux is None:
        return {}
    z = aux if mu is None else zscore_apply(aux, mu, sd)  # type: ignore[arg-type]
    col = {c: i for i, c in enumerate(AUX_LABEL_COLS)}
    want = None if columns is None else set(columns)
    out: dict[str, np.ndarray] = {}
    for name, src, sign in iter_aux_map():
        if want is not None and src not in want:
            continue
        v = np.asarray(z[:, col[src]], dtype="float32") * np.float32(sign)
        out[name] = np.where(np.isfinite(v), v, 0.0).reshape(-1, 1)
    # Path-side ranking uses raw MFE points, not the z-scored Huber targets.
    y_side, _w = path_side_rank_target(aux)
    out[SIDE_RANK_NAME] = y_side
    out.update(_pack_stop_first(aux, columns=columns))
    return out


def _pack_stop_first(
    aux: np.ndarray,
    *,
    columns: tuple[str, ...] | None,
    mae_tolerance: float = DEFAULT_MAE_TOLERANCE,
) -> dict[str, np.ndarray]:
    if not stop_first_enabled(columns):
        return {}
    a = np.asarray(aux, dtype="float64")
    col = {c: i for i, c in enumerate(AUX_LABEL_COLS)}
    out: dict[str, np.ndarray] = {}
    thr = float(mae_tolerance)
    for name, src in STOP_FIRST_MAE_COL.items():
        i = col.get(src)
        if i is None or a.ndim != 2 or a.shape[1] <= i:
            continue
        v = a[:, i]
        y = (np.isfinite(v) & (v >= thr)).astype("float32").reshape(-1, 1)
        out[name] = y
    return out


def aux_sample_weight(
    aux: np.ndarray | None,
    columns: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    if aux is None:
        return {}
    a = np.asarray(aux, dtype="float32")
    col = {c: i for i, c in enumerate(AUX_LABEL_COLS)}
    want = None if columns is None else set(columns)
    out: dict[str, np.ndarray] = {}
    for name, src, _sign in iter_aux_map():
        if want is not None and src not in want:
            continue
        v = a[:, col[src]]
        out[name] = np.isfinite(v).astype("float64")
    _y, w_side = path_side_rank_target(aux)
    out[SIDE_RANK_NAME] = w_side
    if stop_first_enabled(columns):
        for name, src in STOP_FIRST_MAE_COL.items():
            i = col.get(src)
            if i is None or a.ndim != 2 or a.shape[1] <= i:
                continue
            out[name] = np.isfinite(a[:, i]).astype("float64")
    return out


def near_miss_sample_weight(
    y_side: np.ndarray,
    aux: np.ndarray | None,
    *,
    side: str,
    scale_pts: float = 50.0,
) -> np.ndarray | None:
    """Upweight negative rows whose MFE almost reached the 50-pt target.

    Positives already have MFE ≥ 50 (TBM definition); they stay at weight 1.
    A 40-pt failed path counts ~1.8× a 2-pt chop. Used by dual-binary trees
    (ADR-095); LSTM keeps Huber MFE heads instead.
    """
    if aux is None or getattr(aux, "size", 0) == 0:
        return None
    col = "MFE_long_10m" if str(side) == "long" else "MFE_short_10m"
    i = AUX_LABEL_COLS.index(col)
    a = np.asarray(aux, dtype="float64")
    y = np.asarray(y_side).reshape(-1)
    if a.ndim != 2 or a.shape[1] <= i or a.shape[0] != y.size:
        return None
    mag = np.clip(np.nan_to_num(a[:, i], nan=0.0) / float(scale_pts), 0.0, 1.0)
    w = np.ones(y.size, dtype="float64")
    neg = y < 0.5
    w[neg] *= 1.0 + mag[neg]
    return w
