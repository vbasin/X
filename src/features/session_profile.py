"""Persisted daily volume profiles — the cascade's structural SSOT.

Both cascade builders used to rebuild every prior session's profile inside their
own session loop and carry it forward in a ``deque``, which is why each parallel
shard had to re-derive its own warmup before it could emit a single row. That
work is identical in every shard, so it is done once and written here.

Two artifacts:

* ``session_levels.parquet`` — one row per session (``lo/val/poc/vah/hi``, open,
  close, volume, delta, shape, ``r_value``). A few hundred rows, so a 20-session
  lookback is one read rather than 20 tape rebuilds.
* ``profile_<session>.parquet`` — the full volume- and delta-at-price histogram
  at the native 0.25-point NQ tick.

Levels come from :func:`src.features.tape.tape_levels`, so they carry ADR-129
tick parity with the WebUI. The bar-smeared histogram in
:mod:`src.features.prior_volume_profile` is deliberately **not** used: spreading
a bar's volume across its ``[low, high]`` is the approximation ADR-127 retired
for this path.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data_paths import layout
from src.features.cash_session_profile import SessionLevels
from src.features.tape import TAPE_PROFILE_TICK_SIZE, tape_levels
from src.training.auction_levels import ProfileConfig
from src.training.move_label import cash_slice, session_r_from_tape

logger = logging.getLogger("vtech.features.session_profile")

CASH_START, CASH_END = "08:30", "15:00"

# Shape codes match src/features/prior_volume_profile.py so a consumer reading
# either source interprets the column the same way.
SHAPE_D = 0.0        # balanced — one fat centre
SHAPE_P = 1.0        # mass high, thin tail below
SHAPE_b = 2.0        # mass low, thin tail above
SHAPE_DOUBLE = 3.0   # two separated modes

LEVELS_COLUMNS = (
    "session", "lo", "val", "poc", "vah", "hi", "open", "close",
    "volume", "delta", "n_prints", "va_width_pts", "shape", "r_value",
)


# ── paths ────────────────────────────────────────────────────────────────────

def session_profile_dir(root: Path | None = None) -> Path:
    if root is not None:
        return root / "sources" / "features" / "session_profile"
    return layout("session_profile")


def session_levels_path(root: Path | None = None) -> Path:
    return session_profile_dir(root) / "session_levels.parquet"


def profile_path(session: date, root: Path | None = None) -> Path:
    return session_profile_dir(root) / f"profile_{session.isoformat()}.parquet"


# ── primitives ───────────────────────────────────────────────────────────────

def tick_histogram(tape: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(prices, volume, delta)`` at the native tick — one bin per traded price."""
    px = np.round(tape["price"].to_numpy(dtype="float64") / TAPE_PROFILE_TICK_SIZE)
    px *= TAPE_PROFILE_TICK_SIZE
    size = tape["size"].to_numpy(dtype="float64")
    side = tape["side"].to_numpy(dtype="float64")
    prices, inverse = np.unique(px, return_inverse=True)
    volume = np.bincount(inverse, weights=size, minlength=prices.size)
    delta = np.bincount(inverse, weights=size * side, minlength=prices.size)
    return prices, volume, delta


def profile_shape(prices: np.ndarray, volume: np.ndarray) -> float:
    """Classify the profile as D / P / b / double distribution.

    Double distribution wins when a second mode holds at least 55% of the peak
    and is separated from it by a valley under 35% of the smaller peak. Failing
    that, the volume-weighted skew about the mean decides P vs b vs D.
    """
    total = float(volume.sum())
    if total <= 0 or prices.size < 5:
        return SHAPE_D

    # Smooth over a span of the price range, so single-tick spikes are not modes.
    k = max(3, prices.size // 12)
    if k % 2 == 0:
        k += 1
    sm = np.convolve(volume, np.ones(k) / k, mode="same")

    peak_i = int(np.argmax(sm))
    peak_v = float(sm[peak_i])
    if peak_v > 0:
        min_sep = 0.25 * sm.size
        for j in range(sm.size):
            if abs(j - peak_i) < min_sep or sm[j] < 0.55 * peak_v:
                continue
            a, b = (peak_i, j) if peak_i < j else (j, peak_i)
            if float(sm[a:b + 1].min()) < 0.35 * min(peak_v, float(sm[j])):
                return SHAPE_DOUBLE

    mean = float((prices * volume).sum() / total)
    var = float((volume * (prices - mean) ** 2).sum() / total)
    if var <= 0:
        return SHAPE_D
    skew = float((volume * (prices - mean) ** 3).sum() / total) / (var ** 1.5)
    if skew < -0.45:
        return SHAPE_P
    if skew > 0.45:
        return SHAPE_b
    return SHAPE_D


# ── build ────────────────────────────────────────────────────────────────────

def build_one(tape: pd.DataFrame, session: date, cfg: ProfileConfig | None = None
              ) -> tuple[dict, pd.DataFrame] | None:
    """Levels row and tick histogram for one session's cash prints."""
    cfg = cfg or ProfileConfig()
    cash = cash_slice(tape, CASH_START, CASH_END)
    if cash.empty:
        return None
    levels = tape_levels(cash, pd.Timestamp(session), cfg)
    if levels is None:
        return None

    prices, volume, delta = tick_histogram(cash)
    px = cash["price"].to_numpy(dtype="float64")
    row = {
        "session": pd.Timestamp(session),
        "lo": float(levels.lo), "val": float(levels.val), "poc": float(levels.poc),
        "vah": float(levels.vah), "hi": float(levels.hi),
        "open": float(px[0]), "close": float(px[-1]),
        "volume": float(volume.sum()), "delta": float(delta.sum()),
        "n_prints": int(len(cash)),
        "va_width_pts": float(levels.vah - levels.val),
        "shape": profile_shape(prices, volume),
        "r_value": float(session_r_from_tape(cash)),
    }
    hist = pd.DataFrame({
        "price": prices.astype("float32"),
        "volume": volume.astype("float32"),
        "delta": delta.astype("float32"),
    })
    return row, hist


def write_profile(session: date, hist: pd.DataFrame, root: Path | None = None) -> Path:
    p = profile_path(session, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(p, index=False, compression="zstd")
    return p


def write_levels(rows: list[dict], root: Path | None = None) -> Path:
    """Write (or merge into) ``session_levels.parquet``, newest build winning."""
    p = session_levels_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(LEVELS_COLUMNS))
    if p.exists():
        old = pd.read_parquet(p)
        df = pd.concat([old[~old["session"].isin(df["session"])], df],
                       ignore_index=True)
    df = df.sort_values("session", kind="mergesort").reset_index(drop=True)
    df.to_parquet(p, index=False, compression="zstd")
    return p


# ── read ─────────────────────────────────────────────────────────────────────

def load_levels(root: Path | None = None) -> pd.DataFrame:
    """Every built session's levels, oldest first."""
    p = session_levels_path(root)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run scripts/features/build_session_profiles.py first")
    df = pd.read_parquet(p)
    df["session"] = pd.to_datetime(df["session"])
    return df.sort_values("session", kind="mergesort").reset_index(drop=True)


def load_profile(session: date, root: Path | None = None) -> pd.DataFrame | None:
    p = profile_path(session, root)
    return pd.read_parquet(p) if p.exists() else None


def history_before(levels: pd.DataFrame, session: date, n: int) -> list[SessionLevels]:
    """The ``n`` closed sessions before *session*, **newest first**.

    Shaped for :func:`src.features.cash_session_profile.build_bounds`, which
    decays level weight by position in this list.
    """
    prior = levels[levels["session"] < pd.Timestamp(session)]
    if prior.empty:
        return []
    take = prior.tail(n).iloc[::-1]
    return [
        SessionLevels(session=r.session, lo=float(r.lo), val=float(r.val),
                      poc=float(r.poc), vah=float(r.vah), hi=float(r.hi))
        for r in take.itertuples(index=False)
    ]


def prior_row(levels: pd.DataFrame, session: date) -> pd.Series | None:
    """The immediately preceding closed session's levels row."""
    prior = levels[levels["session"] < pd.Timestamp(session)]
    return None if prior.empty else prior.iloc[-1]


def r_before(levels: pd.DataFrame, session: date, n: int) -> float:
    """Median ``r_value`` over the ``n`` sessions before *session*.

    ATR-like reading of current conditions; NaN until a full window exists, which
    is what keeps early sessions out of the matrices.
    """
    prior = levels[levels["session"] < pd.Timestamp(session)]
    vals = prior["r_value"].tail(n).to_numpy(dtype="float64")
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if vals.size == n else float("nan")
