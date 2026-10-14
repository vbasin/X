"""Cash-session volume-profile levels, resolved into bounds by KDE peak finding.

A **cash session** is 08:30-15:00 America/Chicago. (09:30-16:00 Eastern; the
Central end is 15:00, and CME equity-index futures are halted 15:15-16:00 CT so
no 16:00 CT data exists.) Levels are emitted only for a session that has
**closed**, so nothing here can see the session it is used to trade.

Each closed session yields five levels — ``lo``, ``val``, ``poc``, ``vah``,
``hi``. Levels from many sessions agree: the same price gets revisited, and a
price several sessions settled on is stronger than one touched once. Grouping
them is a two-step job:

1. **The density decides grouping.** Each level is deposited on a tick grid as a
   point mass weighted by kind (value-area edges above the POC above the
   extremes) and decayed by session age; the sum is smoothed with a Gaussian of
   width ``sigma_pts``. Every local maximum is a bound centre.
2. **Each level joins its nearest peak by price**, and the bound spans
   ``min(member prices) .. max(member prices)``, widened by ``buffer_pts``.

So the edges are always *real observed levels* and the ``strength`` /
``n_levels`` / kind breakdown are sums over the levels that actually raised the
peak — auditable in exactly the way gap merging was.

What changes is that the grouping decision is **local**. Single-linkage gap
merging was transitive — A joins B and B joins C no matter how far A is from C —
and produced an observed 435-point bound (5.8x the median session range) from 45
levels whose largest internal gap was one point under the merge threshold. A
level joining its nearest density peak cannot chain: two distant clusters belong
to different peaks however many intermediates sit between them.

Resolution is set by ``sigma_pts`` in absolute points, so nothing here depends on
R, on the price level, or on the prevailing volatility regime. Volatility decides
how far away a level is worth aiming at, which is a Layer-1 question; it does not
decide whether two prices are the same level.

Entry is a resting order rather than a fill at the decision price. A decision bar
usually sits *inside* a bound, and inside a zone price is pinned rather than
directional, so :func:`entry_levels` places the order at the **far edge of the
containing bound** — the first price at which the level has demonstrably failed
to hold. Nothing is traded until price gets there.

The trading logic those bounds support is breakout-and-continuation: price is
inside or between bounds, and the next bound in the direction of travel is the
natural target while the one behind is the natural stop. When price has left the
whole clustered range (above the highest ``hi`` or below the lowest ``lo``),
:func:`levels_for_price` walks the lookback further back in ``extend_step``
blocks until it finds bounds on the open side — otherwise there would be nothing
to aim at exactly when a breakout is running.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dtime

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.features.cash_session_profile")

CASH_START = "08:30"
CASH_END = "15:00"

# Value-area edges are where responsive trade turns; the POC is where it settles;
# the extremes are the least-tested of the five.
LEVEL_WEIGHTS: dict[str, float] = {
    "vah": 1.00, "val": 1.00, "poc": 0.70, "hi": 0.40, "lo": 0.40,
}
LEVEL_KINDS = ("lo", "val", "poc", "vah", "hi")


@dataclass(frozen=True)
class LevelConfig:
    """Level-density resolution and lookback policy.

    Every length here is in **index points** and none of it is a function of R.
    Volatility decides how far away a target is worth aiming at, which is a
    Layer-1 question (:func:`bound_candidates`); it does not decide whether two
    prices are the same level, which is what this config governs.

    ``sigma_pts`` is the only resolution knob and it sets what a bound *means*.
    Measured on 2026-06-25 (100 levels from 20 sessions, R = 74.75):

    ======  ======  ===========  ==============  ========
    sigma   bounds  singletons   >= 3 members    max width
    ======  ======  ===========  ==============  ========
    1.0        90       80              0          0.05 R
    **5.0**    62       35              9        **0.30 R**
    8.0        46       17             14          0.63 R
    12.0       36       12             17          1.16 R
    ======  ======  ===========  ==============  ========

    At 1.0 a bound is one level and multi-session agreement never registers; by
    12.0 zones are wide enough to reintroduce the width problem that killed gap
    merging. 5.0 is the default: enough smoothing for genuine agreement to form
    a zone, narrow enough that no bound approaches a session's range.
    """

    lookback_sessions: int = 20
    sigma_pts: float = 5.0          # KDE bandwidth; sets what counts as one level
    grid_step: float = 0.25         # density grid pitch = tick size
    kde_truncate: float = 4.0       # kernel support, in sigma
    min_peak_mass: float = 0.10     # peaks lighter than this are not levels
    buffer_pts: float = 1.0         # widening either side of the member span
    entry_offset_pts: float = 1.0   # confirmation offset when between bounds
    extend_step: int = 5            # sessions added per probe when price is outside
    max_lookback_sessions: int = 120
    recency_halflife: float = 10.0  # sessions
    value_area_pct: float = 0.70
    bin_size: float = 1.0           # volume-profile bin used by session_levels

    def __post_init__(self) -> None:
        for name in ("sigma_pts", "grid_step", "kde_truncate", "bin_size",
                     "recency_halflife", "buffer_pts", "entry_offset_pts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lookback_sessions < 1 or self.extend_step < 1:
            raise ValueError("lookback_sessions and extend_step must be >= 1")
        if not 0 < self.value_area_pct < 1:
            raise ValueError("value_area_pct must be in (0, 1)")
        if self.sigma_pts < self.grid_step:
            raise ValueError("sigma_pts must be at least one grid step")
        if self.min_peak_mass < 0:
            raise ValueError("min_peak_mass must be >= 0")



@dataclass(frozen=True)
class SessionLevels:
    """The five levels of one closed cash session."""

    session: pd.Timestamp
    lo: float
    val: float
    poc: float
    vah: float
    hi: float

    def items(self) -> list[tuple[str, float]]:
        return [(k, float(getattr(self, k))) for k in LEVEL_KINDS]


@dataclass
class Bound:
    """A merged price zone built from one or more session levels."""

    lo: float
    hi: float
    strength: float
    n_levels: int
    kinds: dict[str, int] = field(default_factory=dict)
    newest_age: int = 0
    # (kind, price, session) for each member level that raised this peak —
    # chart overlay / Axis labels. ``session`` is the cash-session end date
    # of the closed session that produced the level.
    members: list[tuple[str, float, pd.Timestamp]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def contains(self, price: float) -> bool:
        return self.lo <= price <= self.hi


def cash_session_slice(day: pd.DataFrame, start: str = CASH_START,
                       end: str = CASH_END) -> pd.DataFrame:
    """Rows whose Chicago time is in ``[start, end)``."""
    ct = day.index.tz_convert("America/Chicago")
    h0, m0 = (int(x) for x in start.split(":"))
    h1, m1 = (int(x) for x in end.split(":"))
    return day.loc[(ct.time >= dtime(h0, m0)) & (ct.time < dtime(h1, m1))]


def session_levels(day: pd.DataFrame, session: pd.Timestamp,
                   cfg: LevelConfig) -> SessionLevels | None:
    """Extract ``lo/val/poc/vah/hi`` from one **closed** cash session."""
    cash = cash_session_slice(day)
    if cash.empty:
        return None
    high = cash["high"].to_numpy(dtype="float64")
    low = cash["low"].to_numpy(dtype="float64")
    vol = cash["volume"].to_numpy(dtype="float64")

    lo_bin = int(np.floor(low.min() / cfg.bin_size))
    hi_bin = int(np.floor(high.max() / cfg.bin_size))
    n = hi_bin - lo_bin + 1
    b_lo = np.floor(low / cfg.bin_size).astype(np.int64) - lo_bin
    b_hi = np.floor(high / cfg.bin_size).astype(np.int64) - lo_bin
    span = (b_hi - b_lo + 1).astype("float64")
    diff = np.zeros(n + 1, dtype="float64")
    np.add.at(diff, b_lo, vol / span)
    np.add.at(diff, b_hi + 1, -vol / span)
    prof = np.cumsum(diff[:-1])
    if prof.sum() <= 0:
        return None

    prices = (np.arange(lo_bin, hi_bin + 1) + 0.5) * cfg.bin_size
    poc_i = int(np.argmax(prof))
    lo_i = hi_i = poc_i
    acc, target = prof[poc_i], cfg.value_area_pct * prof.sum()
    while acc < target and (lo_i > 0 or hi_i < n - 1):
        below = prof[lo_i - 1] if lo_i > 0 else -1.0
        above = prof[hi_i + 1] if hi_i < n - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += prof[hi_i]
        else:
            lo_i -= 1
            acc += prof[lo_i]
    return SessionLevels(session=session, lo=float(low.min()), val=float(prices[lo_i]),
                         poc=float(prices[poc_i]), vah=float(prices[hi_i]),
                         hi=float(high.max()))


def build_bounds(history: list[SessionLevels], cfg: LevelConfig,
                 n_sessions: int | None = None) -> list[Bound]:
    """Bounds from KDE grouping, with edges taken from the grouped levels.

    *history* is newest-first. The density decides **grouping only**: each level
    is a point mass of ``LEVEL_WEIGHTS[kind] * 0.5 ** (age / recency_halflife)``
    on a ``grid_step`` grid, the sum is convolved with a Gaussian of width
    ``sigma_pts``, and every local maximum becomes a bound. Each level then joins
    its **nearest peak by price**, and the bound spans
    ``min(member prices) .. max(member prices)`` widened by ``buffer_pts``.

    Edges are therefore always real observed levels, exactly as under gap
    merging, but the grouping decision is local: a level joins the peak it is
    closest to, so two distant clusters cannot fuse through a chain of
    intermediates the way single linkage allowed.
    """
    take = history[: n_sessions or cfg.lookback_sessions]
    prices: list[float] = []
    kinds: list[str] = []
    weights: list[float] = []
    ages: list[int] = []
    sessions: list[pd.Timestamp] = []
    for age, sl in enumerate(take):
        decay = 0.5 ** (age / cfg.recency_halflife)
        for kind, price in sl.items():
            if np.isfinite(price):
                prices.append(price)
                kinds.append(kind)
                weights.append(LEVEL_WEIGHTS[kind] * decay)
                ages.append(age)
                sessions.append(pd.Timestamp(sl.session))
    if not prices:
        return []

    px = np.asarray(prices, dtype="float64")
    wt = np.asarray(weights, dtype="float64")
    pad = cfg.kde_truncate * cfg.sigma_pts
    origin = float(px.min()) - pad
    n_bins = int(np.ceil((float(px.max()) + pad - origin) / cfg.grid_step)) + 1
    slot = np.clip(np.round((px - origin) / cfg.grid_step).astype(np.int64), 0, n_bins - 1)
    mass = np.zeros(n_bins, dtype="float64")
    np.add.at(mass, slot, wt)

    peaks = _peaks(_smooth(mass, cfg))
    if peaks.size == 0:
        return []
    assign = _nearest_peak(px, origin + peaks * cfg.grid_step)

    groups: list[tuple[np.ndarray, float]] = []
    for j in range(peaks.size):
        sel = np.flatnonzero(assign == j)
        if sel.size and wt[sel].sum() >= cfg.min_peak_mass:
            groups.append((sel, float(wt[sel].sum())))
    if not groups:
        return []

    # Nearest-peak assignment splits at the midpoint between peaks, so the raw
    # member spans are already disjoint and ordered; only the buffer can make
    # them overlap.
    raw = [(float(px[sel].min()), float(px[sel].max())) for sel, _ in groups]
    out: list[Bound] = []
    for j, ((sel, strength), (lo_raw, hi_raw)) in enumerate(zip(groups, raw)):
        left = (lo_raw - raw[j - 1][1]) / 2 if j else np.inf
        right = (raw[j + 1][0] - hi_raw) / 2 if j + 1 < len(raw) else np.inf
        counts: dict[str, int] = {}
        members: list[tuple[str, float, pd.Timestamp]] = []
        for i in sel:
            counts[kinds[i]] = counts.get(kinds[i], 0) + 1
            members.append((kinds[i], float(px[i]), sessions[i]))
        members.sort(key=lambda m: (m[1], m[0], m[2]))
        out.append(Bound(lo=lo_raw - min(cfg.buffer_pts, left),
                         hi=hi_raw + min(cfg.buffer_pts, right),
                         strength=strength, n_levels=int(sel.size), kinds=counts,
                         newest_age=int(min(ages[i] for i in sel)),
                         members=members))
    return out


def _smooth(mass: np.ndarray, cfg: LevelConfig) -> np.ndarray:
    """Gaussian convolution, normalised so total mass is preserved."""
    sigma_bins = cfg.sigma_pts / cfg.grid_step
    half = int(np.ceil(cfg.kde_truncate * sigma_bins))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_bins) ** 2)
    return np.convolve(mass, k / k.sum(), mode="same")


def _peaks(dens: np.ndarray) -> np.ndarray:
    """Local maxima of the density.

    A plateau is resolved to its left-hand bin (``>`` on the left, ``>=`` on the
    right), so a flat top yields one peak rather than none.
    """
    if dens.size < 3:
        return np.array([], dtype=np.int64)
    interior = (dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])
    return np.flatnonzero(interior) + 1


def _nearest_peak(px: np.ndarray, peak_px: np.ndarray) -> np.ndarray:
    """Index of the closest peak price to each level price."""
    i = np.searchsorted(peak_px, px)
    lo = np.clip(i - 1, 0, peak_px.size - 1)
    hi = np.clip(i, 0, peak_px.size - 1)
    return np.where(np.abs(px - peak_px[lo]) <= np.abs(peak_px[hi] - px), lo, hi)


def levels_for_range(history: list[SessionLevels], lo: float, hi: float,
                     cfg: LevelConfig) -> tuple[list[Bound], int]:
    """Bounds that cover ``[lo, hi]``, extending the lookback if they do not.

    Returns ``(bounds, sessions_used)``. When the range sits above every bound
    (or below every bound) the lookback grows by ``extend_step`` until the
    clustered span covers both ends or ``max_lookback_sessions`` is reached —
    a breakout must still have somewhere to aim. A single price is the special
    case ``lo == hi`` (:func:`levels_for_price`).
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        n = min(cfg.lookback_sessions, len(history))
        return build_bounds(history, cfg, n), n
    if hi < lo:
        lo, hi = hi, lo
    n = min(cfg.lookback_sessions, len(history))
    while True:
        bounds = build_bounds(history, cfg, n)
        if not bounds:
            return [], n
        span_lo = min(b.lo for b in bounds)
        span_hi = max(b.hi for b in bounds)
        need_above = hi > span_hi
        need_below = lo < span_lo
        if not (need_above or need_below):
            return bounds, n
        if n >= min(cfg.max_lookback_sessions, len(history)):
            return bounds, n
        n = min(n + cfg.extend_step, cfg.max_lookback_sessions, len(history))


def levels_for_price(history: list[SessionLevels], price: float,
                     cfg: LevelConfig) -> tuple[list[Bound], int]:
    """Bounds that bracket *price*, extending the lookback if it does not.

    Returns ``(bounds, sessions_used)``. When price sits above every bound (or
    below every bound) the lookback grows by ``extend_step`` until a bound exists
    on the open side or ``max_lookback_sessions`` is reached — a breakout must
    still have somewhere to aim.
    """
    return levels_for_range(history, price, price, cfg)


def next_bounds(bounds: list[Bound], price: float
                ) -> tuple[Bound | None, Bound | None, Bound | None]:
    """``(inside, next_above, next_below)`` relative to *price*."""
    inside = next((b for b in bounds if b.contains(price)), None)
    above = [b for b in bounds if b.lo > price]
    below = [b for b in bounds if b.hi < price]
    return (inside,
            min(above, key=lambda b: b.lo) if above else None,
            max(below, key=lambda b: b.hi) if below else None)


def entry_levels(bounds: list[Bound], price: float,
                 cfg: LevelConfig) -> tuple[float, float]:
    """Resting-order prices ``(long, short)`` for a decision taken at *price*.

    Filling at the decision price assumes the trade is worth taking wherever
    price happens to be, but a decision bar is usually *inside* a bound, and
    inside a zone the auction is rotating rather than travelling: the level is a
    magnet, and an immediate fill buys the middle of the rotation. So the order
    rests at the **far edge of the containing bound** in the direction of the
    trade, which is the first price at which that level has failed to hold.

    Between bounds there is no magnet to clear, so the order rests
    ``entry_offset_pts`` away instead \u2014 enough to require the move to continue
    rather than to reverse into the fill.
    """
    inside, _, _ = next_bounds(bounds, price)
    off = cfg.entry_offset_pts
    if inside is None:
        return price + off, price - off
    # max/min guard the degenerate case of price sitting exactly on an edge.
    return max(inside.hi, price + off), min(inside.lo, price - off)


# ══════════════════════════════════════════════════════════════════════════════
# Interaction counters — how the current session has treated each bound
# ══════════════════════════════════════════════════════════════════════════════

def bound_interaction_matrices(prices: np.ndarray, bounds: list[Bound]
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative ``(tests, breaks, rejects)`` per row per bound, shape ``(n, m)``.

    A **test** is an entry into the zone from outside. A **break** is a traverse
    that exits the far side. A **rejection** is an entry that leaves back the way
    it came — the event a trader reads as the level holding. Counts are
    cumulative, so row *i* reflects only what had happened by *i*.
    """
    n, m = len(prices), len(bounds)
    tests = np.zeros((n, m), dtype=np.int32)
    breaks = np.zeros((n, m), dtype=np.int32)
    rejects = np.zeros((n, m), dtype=np.int32)
    if n == 0 or m == 0:
        return tests, breaks, rejects

    for j, b in enumerate(bounds):
        zone = np.where(prices > b.hi, 1, np.where(prices < b.lo, -1, 0)).astype(np.int8)
        prev, now = zone[:-1], zone[1:]

        enter = (prev != 0) & (now == 0)          # stepped into the zone
        exit_ = (prev == 0) & (now != 0)          # stepped out of it
        through = (prev != 0) & (now != 0) & (now != prev)   # gapped clean across

        # Which side the zone was entered from, carried forward to the exit that
        # resolves it. ffill replaces the sequential `entered_from` accumulator.
        src = np.where(enter, prev, np.nan)
        src = pd.Series(src).ffill().to_numpy()

        resolved = exit_ & ~np.isnan(src)
        broke = resolved & (now == -src)
        rejected = resolved & (now == src)

        t = np.zeros(n, dtype=np.int32)
        br = np.zeros(n, dtype=np.int32)
        rj = np.zeros(n, dtype=np.int32)
        t[1:] = enter
        br[1:] = broke | through
        rj[1:] = rejected
        tests[:, j] = np.cumsum(t)
        breaks[:, j] = np.cumsum(br)
        rejects[:, j] = np.cumsum(rj)
    return tests, breaks, rejects


def interactions_at(mats: tuple[np.ndarray, np.ndarray, np.ndarray],
                    bounds: list[Bound], row: int) -> dict[int, tuple[int, int, int]]:
    """Snapshot the per-bound counters at *row*, keyed by bound identity."""
    tests, breaks, rejects = mats
    if not bounds or row < 0 or row >= tests.shape[0]:
        return {}
    return {id(b): (int(tests[row, j]), int(breaks[row, j]), int(rejects[row, j]))
            for j, b in enumerate(bounds)}


def bound_interactions(prices: np.ndarray, bounds: list[Bound]) -> pd.DataFrame:
    """Session-wide cumulative test/break/reject totals across all bounds."""
    tests, breaks, rejects = bound_interaction_matrices(prices, bounds)
    if tests.size == 0:
        return _empty_interactions(len(prices))
    return pd.DataFrame({
        "bi_tests_total": tests.sum(axis=1),
        "bi_breaks_total": breaks.sum(axis=1),
        "bi_rejects_total": rejects.sum(axis=1),
    })


def _empty_interactions(n: int) -> pd.DataFrame:
    return pd.DataFrame({"bi_tests_total": np.zeros(n, dtype=np.int32),
                         "bi_breaks_total": np.zeros(n, dtype=np.int32),
                         "bi_rejects_total": np.zeros(n, dtype=np.int32)})


def bound_candidates(
    bounds: list[Bound],
    entry_price: float,
    r_value: float,
    *,
    max_per_side: int = 10,
    min_dist_r: float = 0.05,
    max_dist_r: float = 3.0,
    interactions: dict[int, tuple[int, int, int]] | None = None,
    ladder_r: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """Candidate exits from the bound structure, as the Layer-0 contract.

    The tradeable price of a bound is its **near edge** — the first price at
    which the zone is reached — so a bound above contributes ``lo`` and one below
    contributes ``hi``. A bound containing the entry is skipped: price is already
    inside it, so it is neither a target nor a stop.

    Structural levels alone give the optimiser too little to choose between —
    measured at ~5 candidates per bar against ~110 for a free grid, which cut net
    expectancy from 16.3 to 11.2 points per trade. ``ladder_r`` therefore adds
    plain volatility-multiple levels flagged ``is_structural = 0``, so geometry
    (especially stop distance, which sets both reward-to-risk and tradeable size)
    stays free while the auction still supplies the levels worth aiming at. The
    flag lets the model discover which kind matters rather than assuming it.

    Descriptors are the auction properties of the zone (how many sessions agreed,
    which level kinds, how wide, how it has been treated today), never an
    absolute price.
    """
    if not np.isfinite(r_value) or r_value <= 0:
        return _empty_candidates()

    ordered = sorted(bounds, key=lambda b: b.lo)
    rows: list[dict[str, float]] = []
    for i, b in enumerate(ordered):
        if b.contains(entry_price):
            continue
        above = b.lo > entry_price
        level = b.lo if above else b.hi
        dist_r = abs(level - entry_price) / r_value
        if not (min_dist_r <= dist_r <= max_dist_r):
            continue
        # Runway past the zone: gap to the next bound further from entry.
        nxt = ordered[i + 1] if (above and i + 1 < len(ordered)) else (
            ordered[i - 1] if (not above and i > 0) else None)
        room = (abs((nxt.lo if above else nxt.hi) - (b.hi if above else b.lo)) / r_value
                if nxt is not None else max_dist_r)
        t, br, rj = (interactions or {}).get(id(b), (0, 0, 0))
        vv = b.kinds.get("vah", 0) + b.kinds.get("val", 0)
        rows.append({
            "level": level, "above": above, "is_structural": 1.0,
            "dist_pts": abs(level - entry_price), "dist_r": dist_r,
            "strength": b.strength, "n_levels": float(b.n_levels),
            "width_r": (b.hi - b.lo) / r_value,
            "newest_age": float(b.newest_age),
            "frac_value_edges": vv / max(b.n_levels, 1),
            "frac_poc": b.kinds.get("poc", 0) / max(b.n_levels, 1),
            "room_beyond": room,
            "tests": float(t), "breaks": float(br), "rejects": float(rj),
            # A zone broken more often than it has held is weaker overhead.
            "hold_ratio": rj / max(t, 1),
        })

    for mult in ladder_r:
        if not (min_dist_r <= mult <= max_dist_r):
            continue
        for above in (True, False):
            lvl = entry_price + (mult * r_value if above else -mult * r_value)
            rows.append({
                "level": lvl, "above": above, "is_structural": 0.0,
                "dist_pts": mult * r_value, "dist_r": mult,
                "strength": 0.0, "n_levels": 0.0, "width_r": 0.0,
                "newest_age": np.nan, "frac_value_edges": 0.0, "frac_poc": 0.0,
                "room_beyond": np.nan, "tests": 0.0, "breaks": 0.0,
                "rejects": 0.0, "hold_ratio": 0.0,
            })

    if not rows:
        return _empty_candidates()
    df = pd.DataFrame(rows)
    # Structural and ladder levels compete for different jobs, so they get
    # separate budgets: a shared cap silently evicts the whole ladder as soon as
    # the bound set is rich enough, and with it the contrast the model needs to
    # tell an agreed level from an arbitrary volatility multiple.
    keep = [g.nlargest(max_per_side, "strength")
            for _, g in df.groupby(["above", "is_structural"], sort=False)]
    return pd.concat(keep, ignore_index=True)


def _empty_candidates() -> pd.DataFrame:
    cols = {"level": "float64", "above": "bool", "is_structural": "float64",
            "dist_pts": "float64",
            "dist_r": "float64", "strength": "float64", "n_levels": "float64",
            "width_r": "float64", "newest_age": "float64",
            "frac_value_edges": "float64", "frac_poc": "float64",
            "room_beyond": "float64", "tests": "float64", "breaks": "float64",
            "rejects": "float64", "hold_ratio": "float64"}
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})


def bound_features(bounds: list[Bound], price: float, r_value: float,
                   interactions: dict[int, tuple[int, int, int]] | None = None,
                   ) -> dict[str, float]:
    """Where price sits in the bound structure, for the entry-state vector."""
    inside, above, below = next_bounds(bounds, price)
    out: dict[str, float] = {
        "lv_n_bounds": float(len(bounds)),
        "lv_inside_bound": 1.0 if inside is not None else 0.0,
        "lv_inside_strength": inside.strength if inside else 0.0,
    }
    for tag, b in (("above", above), ("below", below)):
        if b is None or r_value <= 0:
            out[f"lv_dist_{tag}_r"] = np.nan
            out[f"lv_strength_{tag}"] = np.nan
            out[f"lv_n_levels_{tag}"] = np.nan
            out[f"lv_width_{tag}_r"] = np.nan
        else:
            edge = b.lo if tag == "above" else b.hi
            out[f"lv_dist_{tag}_r"] = abs(edge - price) / r_value
            out[f"lv_strength_{tag}"] = b.strength
            out[f"lv_n_levels_{tag}"] = float(b.n_levels)
            out[f"lv_width_{tag}_r"] = (b.hi - b.lo) / r_value
    if above is not None and below is not None and r_value > 0:
        span = (above.lo - below.hi)
        out["lv_channel_width_r"] = span / r_value
        out["lv_pos_in_channel"] = ((price - below.hi) / span) if span > 0 else 0.5
    else:
        out["lv_channel_width_r"] = np.nan
        out["lv_pos_in_channel"] = np.nan
    if interactions:
        t, b_, r_ = interactions.get(id(inside) if inside else -1, (0, 0, 0))
        out["lv_inside_tests"] = float(t)
        out["lv_inside_breaks"] = float(b_)
        out["lv_inside_rejects"] = float(r_)
    return out
