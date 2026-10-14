"""Rules-based protective stop from recent auction structure.

A stop is not a free parameter to be searched over. Once a directional move is
identified, the question "where is price unlikely to briefly pull back to?" has a
structural answer, and searching every price level for the one with the best
realised outcome is how a backtest learns the past instead of the market.

Three candidate floors are read off the recent tape (mirrored for shorts):

1. **Absorption shelf.** A large *sell*-aggressor print in the **lower wick of an
   up candle** is someone filling aggressive selling without letting price drop —
   a resting bid wall. The deepest such print in the lookback is the price the
   book has demonstrably defended.
2. **Wick floor.** The lowest low of the recent candles: where pullbacks have
   actually been stopping, wall or no wall.
3. **Zone floor.** The far side of the level bound price is leaving. A breakout
   that comes back through the whole zone was not a breakout.

``policy`` decides which becomes the stop. ``tightest`` takes the highest floor —
the best reward:risk still backed by structure; ``safest`` takes the lowest —
the most room.

**Measured result (2026-06-01 → 06-10, 5s cadence, ``probe_stop_rule.py``).** Every
structural policy beats a *distance-matched random* control by only 1.0–1.8
percentage points of survival, and absorption is inside noise (−0.01 to +0.02
depending on ``large_order_size``). Survival tracks distance almost perfectly:
0.41 R → 67%, 0.64 R → 77%, whatever the structure underneath. The conclusion is
that **distance is doing the work, not the structure**, which is why ``fixed`` exists
and is the honest default. The structural policies are kept so the finding stays
reproducible rather than asserted.

Nothing here looks forward: every input is a print or a bound that existed at or
before the decision timestamp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.training.stop_rule")

LONG, SHORT = 1, -1
POLICIES = ("tightest", "safest", "wick", "absorption", "zone", "fixed")


@dataclass(frozen=True)
class StopConfig:
    """Lookback depth, wall definition and distance guards."""

    lookback_bars: int = 5          # recent candles scanned for walls and wicks
    bar_interval: str = "1min"
    buffer_pts: float = 1.0         # placed this far beyond the chosen floor
    large_order_size: int = 20      # a print this size or larger can be a wall
    min_dist_r: float = 0.10        # closer than this and the stop is spread noise
    max_dist_r: float = 1.00        # further than this and the trade is untradeable
    fixed_dist_r: float = 0.45      # distance used by the ``fixed`` policy
    policy: str = "fixed"

    def __post_init__(self) -> None:
        if self.lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        if self.buffer_pts <= 0:
            raise ValueError("buffer_pts must be positive")
        if self.large_order_size < 1:
            raise ValueError("large_order_size must be >= 1")
        if self.min_dist_r >= self.max_dist_r:
            raise ValueError("min_dist_r must be < max_dist_r")
        if not self.min_dist_r <= self.fixed_dist_r <= self.max_dist_r:
            raise ValueError("fixed_dist_r must lie in [min_dist_r, max_dist_r]")
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")


@dataclass
class StopBars:
    """Per-candle structure needed by the stop rule, precomputed once per session.

    Arrays are aligned to ``index`` (candle end timestamps). ``bid_wall`` /
    ``ask_wall`` hold the absorption shelf price for each candle, or NaN where no
    large print qualified.
    """

    index: pd.DatetimeIndex
    high: np.ndarray
    low: np.ndarray
    bid_wall: np.ndarray
    ask_wall: np.ndarray


def build_stop_bars(tape: pd.DataFrame, cfg: StopConfig) -> StopBars:
    """Candle highs/lows and absorption shelves from the print tape.

    A print is a **bid wall** when it is sell-aggressor, at least
    ``large_order_size``, and sits in the lower wick of an up candle — the
    bullish-absorption cell of the flow-state taxonomy. Ask walls mirror it.
    """
    floor = tape.index.floor(cfg.bar_interval)
    g = tape.groupby(floor)["price"]
    bars = pd.DataFrame({"open": g.first(), "high": g.max(),
                         "low": g.min(), "close": g.last()})
    # Candle end labels the bar, matching the native-TF convention.
    bars.index = bars.index + pd.Timedelta(cfg.bar_interval)

    px = tape["price"].to_numpy(dtype="float64")
    sz = tape["size"].to_numpy(dtype="float64")
    sd = tape["side"].to_numpy(dtype="float64")
    key = floor + pd.Timedelta(cfg.bar_interval)
    o = bars["open"].reindex(key).to_numpy(dtype="float64")
    c = bars["close"].reindex(key).to_numpy(dtype="float64")
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)

    big = sz >= cfg.large_order_size
    up = c > o
    down = c < o
    bid_hit = big & (sd < 0) & (px < body_lo) & up      # sell absorbed in an up-candle wick
    ask_hit = big & (sd > 0) & (px > body_hi) & down    # buy absorbed in a down-candle wick

    bid_wall = (pd.Series(np.where(bid_hit, px, np.nan), index=key)
                .groupby(level=0).min().reindex(bars.index).to_numpy(dtype="float64"))
    ask_wall = (pd.Series(np.where(ask_hit, px, np.nan), index=key)
                .groupby(level=0).max().reindex(bars.index).to_numpy(dtype="float64"))
    return StopBars(index=bars.index,
                    high=bars["high"].to_numpy(dtype="float64"),
                    low=bars["low"].to_numpy(dtype="float64"),
                    bid_wall=bid_wall, ask_wall=ask_wall)


def stop_candidates(sb: StopBars, ts: pd.Timestamp, side: int,
                    zone_edge: float | None, cfg: StopConfig
                    ) -> dict[str, float]:
    """The three structural floors (or ceilings, for a short) at *ts*.

    Only candles that **closed** at or before *ts* are read, so the result is
    knowable at the decision. Missing candidates come back as NaN.
    """
    j = int(np.searchsorted(sb.index.values, ts.to_datetime64(), side="right"))
    if j == 0:
        return {"wick": np.nan, "absorption": np.nan, "zone": _f(zone_edge)}
    i = max(0, j - cfg.lookback_bars)
    if side == LONG:
        wick = float(np.min(sb.low[i:j]))
        wall = sb.bid_wall[i:j]
    else:
        wick = float(np.max(sb.high[i:j]))
        wall = sb.ask_wall[i:j]
    finite = wall[np.isfinite(wall)]
    absorption = float(finite.min() if side == LONG else finite.max()) if finite.size else np.nan
    return {"wick": wick, "absorption": absorption, "zone": _f(zone_edge)}


def resting_stop(cands: dict[str, float], side: int, entry: float,
                 r_value: float, cfg: StopConfig) -> float:
    """Resolve the candidates to one stop price, or NaN if none is tradeable.

    ``tightest`` / ``safest`` reduce over whatever candidates exist; the named
    policies use one and fail closed when it is missing; ``fixed`` ignores structure
    entirely and places the stop at ``fixed_dist_r`` from the entry. The buffer is
    applied beyond the chosen floor, then distance guards reject stops that are
    inside spread noise or too wide to risk.
    """
    r = max(r_value, 1e-9)
    if cfg.policy == "fixed":
        d = cfg.fixed_dist_r * r
        stop = entry - d if side == LONG else entry + d
        return float(stop)

    if cfg.policy in ("wick", "absorption", "zone"):
        chosen = cands.get(cfg.policy, np.nan)
    else:
        vals = np.array([v for v in cands.values() if np.isfinite(v)], dtype="float64")
        if vals.size == 0:
            return np.nan
        want_high = (cfg.policy == "tightest") == (side == LONG)
        chosen = float(vals.max() if want_high else vals.min())
    if not np.isfinite(chosen):
        return np.nan

    stop = chosen - cfg.buffer_pts if side == LONG else chosen + cfg.buffer_pts
    dist = abs(entry - stop)
    if dist < cfg.min_dist_r * r or dist > cfg.max_dist_r * r:
        return np.nan
    # A stop on the wrong side of the entry is not a stop.
    if (side == LONG and stop >= entry) or (side == SHORT and stop <= entry):
        return np.nan
    return float(stop)


def _f(x: float | None) -> float:
    return float(x) if x is not None and np.isfinite(x) else np.nan


def stop_candidates_vec(sb: StopBars, ts: np.ndarray, side: int,
                        zone_edge: np.ndarray, cfg: StopConfig) -> dict[str, np.ndarray]:
    """:func:`stop_candidates` for every decision at once.

    The per-decision window ``[j - lookback_bars, j)`` is variable only at the
    session edges, so the wick floor and absorption shelf come from a sparse table
    over the candle arrays rather than a slice per decision. NaN walls are mapped
    to the neutral extreme before the reduction and mapped back afterwards.
    """
    n = sb.index.size
    j = np.searchsorted(sb.index.values, ts, side="right")
    i = np.maximum(j - cfg.lookback_bars, 0)
    lo_i = np.clip(i, 0, max(n - 1, 0))
    hi_i = np.clip(j - 1, 0, max(n - 1, 0))
    valid = j > 0

    if side == LONG:
        wick_tbl = _sparse(sb.low, is_max=False)
        wick = _range(wick_tbl, lo_i, hi_i, is_max=False)
        wall_src = np.where(np.isfinite(sb.bid_wall), sb.bid_wall, np.inf)
        wall = _range(_sparse(wall_src, is_max=False), lo_i, hi_i, is_max=False)
        wall = np.where(np.isinf(wall), np.nan, wall)
    else:
        wick = _range(_sparse(sb.high, is_max=True), lo_i, hi_i, is_max=True)
        wall_src = np.where(np.isfinite(sb.ask_wall), sb.ask_wall, -np.inf)
        wall = _range(_sparse(wall_src, is_max=True), lo_i, hi_i, is_max=True)
        wall = np.where(np.isinf(wall), np.nan, wall)

    return {"wick": np.where(valid, wick, np.nan),
            "absorption": np.where(valid, wall, np.nan),
            "zone": zone_edge.astype("float64")}


def resting_stop_vec(cands: dict[str, np.ndarray], side: int, entry: np.ndarray,
                     r_value: float, cfg: StopConfig) -> np.ndarray:
    """:func:`resting_stop` for every decision at once; NaN where untradeable."""
    r = max(r_value, 1e-9)
    if cfg.policy == "fixed":
        d = cfg.fixed_dist_r * r
        return entry - d if side == LONG else entry + d

    if cfg.policy in ("wick", "absorption", "zone"):
        chosen = cands[cfg.policy].astype("float64")
    else:
        stack = np.vstack([cands["wick"], cands["absorption"], cands["zone"]])
        want_high = (cfg.policy == "tightest") == (side == LONG)
        with np.errstate(invalid="ignore"):
            chosen = np.nanmax(stack, axis=0) if want_high else np.nanmin(stack, axis=0)

    stop = chosen - cfg.buffer_pts if side == LONG else chosen + cfg.buffer_pts
    dist = np.abs(entry - stop)
    bad = (~np.isfinite(stop)) | (dist < cfg.min_dist_r * r) | (dist > cfg.max_dist_r * r)
    bad |= (stop >= entry) if side == LONG else (stop <= entry)
    return np.where(bad, np.nan, stop)


def _sparse(a: np.ndarray, is_max: bool) -> np.ndarray:
    """Sparse table over a candle array for O(1) variable-window reductions."""
    n = a.size
    levels = max(1, int(np.floor(np.log2(max(n, 1)))) + 1)
    t = np.empty((levels, n), dtype="float64")
    t[0] = a
    for k in range(1, levels):
        w = 1 << (k - 1)
        m = n - (1 << k) + 1
        if m <= 0:
            t[k] = t[k - 1]
            continue
        op = np.maximum if is_max else np.minimum
        op(t[k - 1, :m], t[k - 1, w:w + m], out=t[k, :m])
        t[k, m:] = t[k - 1, m:]
    return t


def _range(t: np.ndarray, l: np.ndarray, r: np.ndarray, is_max: bool) -> np.ndarray:
    span = np.maximum(r - l + 1, 1)
    k = np.floor(np.log2(span)).astype(np.int64)
    a = t[k, l]
    b = t[k, np.maximum(r - (1 << k) + 1, 0)]
    return np.maximum(a, b) if is_max else np.minimum(a, b)
