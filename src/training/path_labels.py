"""Honest first-touch labels and dense training targets from the 1-second path.

Replaces the close-only label chain for any geometry whose stop is comparable to a bar's
range. `generate_labels_tbm` (:mod:`src.training.labels`) compares **5s closes** to both
barriers, so a bar that traded 9 points against the entry and closed +1 is labelled a
clean winner ([ADR-052](../../docs/decisions/ADR-052-path-realised-account-simulator.md)).
Here first-touch is resolved on the 1s high/low path from ``parquet_path1s``.

Two things this module adds beyond honest ``{0,1,2}`` labels:

**Vol-normalised barriers.** A fixed +50/−6 geometry is not the same event across eras.
Measured on real 1s paths (08:30–11:00 CT), the directional rate at 50/6 is **0.000% for
2011–2017** and 15.5% in 2026, because the median 10-minute range grew from ~6 to ~67
points. In 2011–2013 the 6-point *stop* is wider than the median 10-minute range, so the
geometry is incoherent, not merely rare. ``mode="vol_normalized"`` expresses both barriers
as multiples of a causal volatility yardstick ``R`` (see :func:`session_r`), making 2011
and 2026 the same problem in normalised units.

**A microstructure floor.** Normalisation does not rescue every era: tick size has been
0.25 points since 1999 while the barrier is relative. A normalised 2011 stop is 2.2 ticks
— inside the spread. ``min_stop_ticks`` marks such sessions ``usable=False`` rather than
silently training on fills that cannot happen.

**Dense targets.** Alongside the sparse ``{0,1,2}`` label, every entry bar also carries
discrete-time competing-risks targets (``event_*``/``bucket_*``) and MFE/MAE excursions in
R units. The sparse label gives one supervised value per episode; the hazard buckets give
~``n_buckets`` and the excursions are defined on *every* bar including flat ones. That is
what lets tens of millions of 1s bars train an encoder that ~5k directional episodes
cannot.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from datetime import time as dtime
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from src.features.session_path import (
    parse_hhmm as _parse_hhmm,
    prepare_path,
    session_grid as _session_grid,
)

logger = logging.getLogger("vtech.training.path_labels")

# Label encoding — identical to src.training.labels so downstream consumers are unchanged.
BIG_DOWN, NO_SIGNAL, BIG_UP = 0, 1, 2

# Competing-risks event codes.
CENSORED, TARGET, STOP = 0, 1, 2

_MODES = ("fixed", "vol_normalized")

# 2026 geometry expressed in R (median 10-minute range) — the reference the vol-normalised
# mode reproduces in every era. Derived from measured 1s paths: 50/67.3 and 6/67.3.
DEFAULT_TARGET_R = 0.743
DEFAULT_STOP_R = 0.0891


@dataclass(frozen=True)
class PathLabelConfig:
    """Barrier geometry, volatility yardstick, microstructure floor and target encoding."""

    mode: str = "vol_normalized"

    # fixed mode
    target_points: float = 50.0
    stop_points: float = 6.0

    # vol_normalized mode — barriers as multiples of R
    target_r: float = DEFAULT_TARGET_R
    stop_r: float = DEFAULT_STOP_R

    horizon_seconds: int = 600

    # volatility yardstick: R = median of the trailing sessions' median window range
    vol_window_seconds: int = 600
    vol_lookback_sessions: int = 5

    # microstructure floor
    tick_size: float = 0.25
    min_stop_ticks: float = 8.0

    # dense targets
    n_buckets: int = 20

    # entry window (America/Chicago); the grid must extend horizon_seconds past entry_end
    entry_start: str = "08:30"
    entry_end: str = "11:00"

    tie_policy: str = "stop_first"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        if self.tie_policy not in ("stop_first", "target_first"):
            raise ValueError("tie_policy must be 'stop_first' or 'target_first'")
        if self.horizon_seconds < 1 or self.n_buckets < 1:
            raise ValueError("horizon_seconds and n_buckets must be >= 1")
        if self.horizon_seconds % self.n_buckets:
            raise ValueError("horizon_seconds must be divisible by n_buckets")
        for name in ("target_points", "stop_points", "target_r", "stop_r", "tick_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mode == "fixed" and self.stop_points >= self.target_points:
            raise ValueError("stop_points must be < target_points (labels would overlap)")
        if self.mode == "vol_normalized" and self.stop_r >= self.target_r:
            raise ValueError("stop_r must be < target_r (labels would overlap)")
        if self.vol_lookback_sessions < 1:
            raise ValueError("vol_lookback_sessions must be >= 1")

    @property
    def bucket_seconds(self) -> int:
        return self.horizon_seconds // self.n_buckets


# ══════════════════════════════════════════════════════════════════════════════
# Volatility yardstick
# ══════════════════════════════════════════════════════════════════════════════

def session_r(grid: pd.DataFrame, cfg: PathLabelConfig) -> float:
    """Median rolling-window range during this session's entry window, in points.

    The per-session contribution to ``R``. Computed on non-overlapping
    ``vol_window_seconds`` blocks so it is a plain median of independent ranges rather
    than an autocorrelated rolling statistic.
    """
    ct = grid.index.tz_convert("America/Chicago")
    win = (ct.time >= _parse_hhmm(cfg.entry_start)) & (ct.time < _parse_hhmm(cfg.entry_end))
    if win.sum() < cfg.vol_window_seconds:
        return float("nan")
    high = grid["high"].to_numpy()[win]
    low = grid["low"].to_numpy()[win]
    w = cfg.vol_window_seconds
    n = len(high) // w
    if n < 1:
        return float("nan")
    blocks_hi = high[: n * w].reshape(n, w).max(axis=1)
    blocks_lo = low[: n * w].reshape(n, w).min(axis=1)
    return float(np.median(blocks_hi - blocks_lo))


# ══════════════════════════════════════════════════════════════════════════════
# First-touch scan with per-entry barrier distances
# ══════════════════════════════════════════════════════════════════════════════

def _first_touch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: np.ndarray,
    side: int,
    target: np.ndarray,
    stop: np.ndarray,
    *,
    horizon: int,
    last_idx: int,
    tie_policy: str,
) -> dict[str, np.ndarray]:
    """First-touch outcome per entry, with **per-entry** barrier distances.

    Differs from :func:`src.backtest.portfolio_sim._resolve_exits` in two ways that matter
    for labelling: barriers are arrays (vol-normalised geometry varies bar to bar), and
    there is no slippage or commission — a label must describe the market, not a fill.

    ``side`` is ``+1`` long / ``-1`` short. Returns ``event`` (0 censored / 1 target /
    2 stop), ``offset`` (seconds held; ``horizon`` when censored), ``mfe``/``mae`` in
    points, and ``ambiguous`` (one bar straddled both barriers).
    """
    n = len(entry_idx)
    ep = close[entry_idx]
    tgt = ep + side * target
    stp = ep - side * stop
    fav = high if side > 0 else low
    adv = low if side > 0 else high

    event = np.full(n, CENSORED, dtype=np.int8)
    offset = np.full(n, horizon, dtype=np.int32)
    mfe = np.zeros(n, dtype="float64")
    mae = np.zeros(n, dtype="float64")
    ambiguous = np.zeros(n, dtype=bool)

    limit = np.minimum(horizon, last_idx - entry_idx).astype(np.int64)
    for k in range(1, horizon + 1):
        live = (event == CENSORED) & (limit >= k)
        if not live.any():
            break
        j = entry_idx[live] + k
        f, a = fav[j], adv[j]
        e = ep[live]
        mfe[live] = np.maximum(mfe[live], side * (f - e))
        mae[live] = np.maximum(mae[live], side * (e - a))

        t_hit = side * (f - tgt[live]) >= 0
        s_hit = side * (a - stp[live]) <= 0
        idx_live = np.flatnonzero(live)
        if tie_policy == "target_first":
            take_t, take_s = t_hit, s_hit & ~t_hit
        else:
            take_s, take_t = s_hit, t_hit & ~s_hit
        for mask, code in ((take_t, TARGET), (take_s, STOP)):
            if mask.any():
                sel = idx_live[mask]
                event[sel] = code
                offset[sel] = k
        both = t_hit & s_hit
        if both.any():
            ambiguous[idx_live[both]] = True

    # Entries whose window is truncated by the end of the grid are censored, not labelled.
    truncated = (event == CENSORED) & (limit < horizon)
    return {"event": event, "offset": offset, "mfe": mfe, "mae": mae,
            "ambiguous": ambiguous, "truncated": truncated}


# ══════════════════════════════════════════════════════════════════════════════
# Per-session labelling
# ══════════════════════════════════════════════════════════════════════════════

def label_session(
    grid: pd.DataFrame,
    r_value: float,
    cfg: PathLabelConfig,
) -> pd.DataFrame:
    """Label one session's entry window. *r_value* must come from **prior** sessions."""
    ct = grid.index.tz_convert("America/Chicago")
    win = ((ct.time >= _parse_hhmm(cfg.entry_start)) & (ct.time < _parse_hhmm(cfg.entry_end)))
    entry_idx = np.flatnonzero(win)
    if len(entry_idx) == 0:
        return _empty_frame()

    if cfg.mode == "fixed":
        tp, sp = cfg.target_points, cfg.stop_points
    else:
        if not np.isfinite(r_value) or r_value <= 0:
            return _empty_frame()
        tp, sp = cfg.target_r * r_value, cfg.stop_r * r_value

    stop_ticks = sp / cfg.tick_size
    usable = bool(stop_ticks >= cfg.min_stop_ticks)

    high = grid["high"].to_numpy()
    low = grid["low"].to_numpy()
    close = grid["close"].to_numpy()
    last_idx = len(grid) - 1
    tgt_arr = np.full(len(entry_idx), tp, dtype="float64")
    stp_arr = np.full(len(entry_idx), sp, dtype="float64")

    up = _first_touch(high, low, close, entry_idx, 1, tgt_arr, stp_arr,
                      horizon=cfg.horizon_seconds, last_idx=last_idx,
                      tie_policy=cfg.tie_policy)
    dn = _first_touch(high, low, close, entry_idx, -1, tgt_arr, stp_arr,
                      horizon=cfg.horizon_seconds, last_idx=last_idx,
                      tie_policy=cfg.tie_policy)

    label = np.full(len(entry_idx), NO_SIGNAL, dtype=np.int32)
    label[up["event"] == TARGET] = BIG_UP
    label[dn["event"] == TARGET] = BIG_DOWN

    bs = cfg.bucket_seconds
    out = pd.DataFrame({
        "ts": grid.index[entry_idx],
        "label": label,
        # competing-risks targets: which barrier, and in which time bucket
        "event_up": up["event"], "bucket_up": np.minimum(
            (up["offset"] - 1) // bs, cfg.n_buckets - 1).astype(np.int16),
        "event_down": dn["event"], "bucket_down": np.minimum(
            (dn["offset"] - 1) // bs, cfg.n_buckets - 1).astype(np.int16),
        "offset_up": up["offset"], "offset_down": dn["offset"],
        # dense excursion targets, in points and in R units
        "mfe_up_points": up["mfe"], "mae_up_points": up["mae"],
        "mfe_down_points": dn["mfe"], "mae_down_points": dn["mae"],
        # geometry actually applied — self-describing, so a mixed-era dataset is auditable
        "target_points": tp, "stop_points": sp, "r_value": r_value,
        "stop_ticks": stop_ticks, "usable": usable,
        "ambiguous": up["ambiguous"] | dn["ambiguous"],
        "truncated": up["truncated"] | dn["truncated"],
    })
    scale = r_value if (cfg.mode == "vol_normalized" and np.isfinite(r_value)
                        and r_value > 0) else tp / cfg.target_r
    for c in ("mfe_up", "mae_up", "mfe_down", "mae_down"):
        out[f"{c}_r"] = out[f"{c}_points"] / scale
    # A censored entry has no observed barrier — the hazard head must not learn a bucket.
    out.loc[out["event_up"] == CENSORED, "bucket_up"] = cfg.n_buckets
    out.loc[out["event_down"] == CENSORED, "bucket_down"] = cfg.n_buckets
    return out


def _empty_frame() -> pd.DataFrame:
    cols = {"ts": "datetime64[ns, UTC]", "label": "int32",
            "event_up": "int8", "bucket_up": "int16", "event_down": "int8",
            "bucket_down": "int16", "offset_up": "int32", "offset_down": "int32",
            "mfe_up_points": "float64", "mae_up_points": "float64",
            "mfe_down_points": "float64", "mae_down_points": "float64",
            "target_points": "float64", "stop_points": "float64", "r_value": "float64",
            "stop_ticks": "float64", "usable": "bool", "ambiguous": "bool",
            "truncated": "bool", "mfe_up_r": "float64", "mae_up_r": "float64",
            "mfe_down_r": "float64", "mae_down_r": "float64"}
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})


# ══════════════════════════════════════════════════════════════════════════════
# Streaming builder
# ══════════════════════════════════════════════════════════════════════════════

def iter_session_labels(
    sessions: Iterable[tuple[Any, pd.DataFrame]],
    cfg: PathLabelConfig | None = None,
) -> Iterator[tuple[Any, pd.DataFrame]]:
    """Label sessions in chronological order, carrying a causal volatility yardstick.

    ``R`` for session *d* is the median of the previous ``vol_lookback_sessions``
    sessions' :func:`session_r` — prior sessions only, so a barrier is knowable at the
    open. The first ``vol_lookback_sessions`` sessions yield nothing in
    ``vol_normalized`` mode: there is no history to set a barrier from.
    """
    cfg = cfg or PathLabelConfig()
    history: deque[float] = deque(maxlen=cfg.vol_lookback_sessions)
    for key, raw in sessions:
        grid = _session_grid(prepare_path(raw))
        r_today = session_r(grid, cfg)
        r_prior = (float(np.median(history))
                   if len(history) == cfg.vol_lookback_sessions else float("nan"))
        if cfg.mode == "fixed" or np.isfinite(r_prior):
            frame = label_session(grid, r_prior, cfg)
            if not frame.empty:
                yield key, frame
        if np.isfinite(r_today):
            history.append(r_today)


def summarise(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Per-year label geography: directional rate, geometry applied, usability.

    This is the report that decides which era can supervise a geometry at all.
    """
    rows = []
    for f in frames:
        if f.empty:
            continue
        rows.append({
            "year": int(pd.to_datetime(f["ts"].iloc[0]).year),
            "n_entries": len(f),
            "up_rate": float((f["label"] == BIG_UP).mean()),
            "down_rate": float((f["label"] == BIG_DOWN).mean()),
            "dir_rate": float((f["label"] != NO_SIGNAL).mean()),
            "target_points": float(f["target_points"].iloc[0]),
            "stop_points": float(f["stop_points"].iloc[0]),
            "stop_ticks": float(f["stop_ticks"].iloc[0]),
            "usable": bool(f["usable"].iloc[0]),
            "ambiguous_rate": float(f["ambiguous"].mean()),
            "censored_up_rate": float((f["event_up"] == CENSORED).mean()),
        })
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    agg = d.groupby("year").agg(
        sessions=("n_entries", "size"), entries=("n_entries", "sum"),
        dir_rate=("dir_rate", "mean"), up_rate=("up_rate", "mean"),
        down_rate=("down_rate", "mean"), target_points=("target_points", "mean"),
        stop_points=("stop_points", "mean"), stop_ticks=("stop_ticks", "mean"),
        usable_frac=("usable", "mean"), ambiguous_rate=("ambiguous_rate", "mean"),
    ).reset_index()
    return agg
