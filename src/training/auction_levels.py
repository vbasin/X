"""Layer 0/1 — auction candidate levels and their descriptors.

Candidate exits are **prices**, generated mechanically from the volume- and
delta-at-price structure of prior sessions, and described by *continuous
properties* rather than a categorical level name. A model trained on
``vol_at_level`` / ``path_volume_between`` / ``n_sessions_confluence`` learns the
physics of a node; a model trained on ``level_type == "VAH"`` memorises a label
and cannot generalise to a level nobody enumerated.

Pipeline:

1. :func:`session_profile` — volume-at-price and signed-delta-at-price for one
   session, from the 1s path. Each 1s bar spreads its volume across the bins its
   range covers (difference-array + cumsum, exact and O(n)).
2. :func:`composite_profile` — recency-weighted stack of the trailing
   ``lookback_sessions`` profiles.
3. :func:`find_nodes` — local maxima (HVN) and minima (LVN) of the composite.
4. :func:`select_candidates` — top-K per side inside a distance band, with the
   Layer-1 descriptors attached.

**Delta requires TBBO.** ``parquet_path1s`` carries OHLCV only. A CLV proxy for
aggressor side was measured against true ``of_delta`` on 60 overlap sessions and
fails at the level that matters: per-bar sign agreement 65%, session CVD median
r 0.50 but p10 **-0.45**, and delta-at-price profile correlation **r ~ 0.00**
(ADR-123). At 1s, CLV is dominated by bid-ask bounce, and aggregating by *price*
rather than time does not cancel it. :func:`session_profile` therefore takes an
explicit ``delta`` array; callers supply true TBBO delta via
:func:`allocate_tf_delta`. Without it the delta channel is left at zero rather
than filled with noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dtime

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.training.auction_levels")

DEFAULT_BIN_SIZE = 1.0          # index points per profile bin
DEFAULT_LOOKBACK_SESSIONS = 30
DEFAULT_MAX_CANDIDATES = 10     # per side, per role


@dataclass(frozen=True)
class ProfileConfig:
    """Profile resolution, lookback depth and candidate selection band."""

    bin_size: float = DEFAULT_BIN_SIZE
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS
    recency_halflife: float = 10.0      # sessions; weight = 0.5 ** (age / halflife)
    node_window_bins: int = 5           # local-extremum half-width
    value_area_pct: float = 0.70
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    # Candidate distance band, in R units. Nearer than min is spread noise;
    # further than max is unreachable inside one session.
    min_dist_r: float = 0.05
    max_dist_r: float = 3.0
    confluence_windows: tuple[int, ...] = (5, 10, 30)
    confluence_tol_bins: int = 2

    def __post_init__(self) -> None:
        if self.bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be >= 1")
        if not 0 < self.value_area_pct < 1:
            raise ValueError("value_area_pct must be in (0, 1)")
        if self.min_dist_r >= self.max_dist_r:
            raise ValueError("min_dist_r must be < max_dist_r")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")


@dataclass
class Profile:
    """Volume/delta by price bin over a contiguous bin range."""

    lo_bin: int
    bin_size: float
    volume: np.ndarray
    delta: np.ndarray

    @property
    def hi_bin(self) -> int:
        return self.lo_bin + len(self.volume) - 1

    def prices(self) -> np.ndarray:
        """Bin centre prices."""
        return (np.arange(self.lo_bin, self.hi_bin + 1) + 0.5) * self.bin_size

    def bin_of(self, price: float | np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(price, dtype="float64") / self.bin_size).astype(np.int64)

    def value_at(self, price: float | np.ndarray, which: str = "volume") -> np.ndarray:
        """Profile value at *price*, 0 outside the profile's range."""
        arr = self.volume if which == "volume" else self.delta
        idx = self.bin_of(price) - self.lo_bin
        out = np.zeros(np.shape(idx), dtype="float64")
        ok = (idx >= 0) & (idx < len(arr))
        out[ok] = arr[idx[ok]]
        return out


def allocate_tf_delta(
    day: pd.DataFrame,
    tf_delta: pd.Series,
    timeframe: str = "15s",
) -> np.ndarray:
    """Spread true aggressor delta from closed ``timeframe`` bars onto 1s rows.

    Each 1s row receives its parent bar's signed delta in proportion to the
    volume it contributed, so the sign comes from TBBO and only the intra-bar
    price attribution is approximated. ``tf_delta`` is indexed by bar-end
    timestamp, matching the native-TF hive convention (left-closed ``[T, T+w)``
    labelled ``T+w``). Rows with no covering bar get 0.
    """
    v = day["volume"].to_numpy(dtype="float64")
    # Left-closed [T, T+w) labelled T+w: ceil() would map a row already on a
    # boundary to its own timestamp, i.e. into the previous bar.
    parent = day.index.floor(timeframe) + pd.Timedelta(timeframe)
    bar_vol = pd.Series(v, index=parent).groupby(level=0).transform("sum").to_numpy()
    d = tf_delta.reindex(parent).to_numpy(dtype="float64")
    share = np.divide(v, bar_vol, out=np.zeros_like(v), where=bar_vol > 0)
    return np.nan_to_num(d) * share


def session_profile(day: pd.DataFrame, cfg: ProfileConfig,
                    delta: np.ndarray | None = None) -> Profile:
    """Volume- and delta-at-price for one session's 1s OHLCV frame.

    Each bar's volume is spread uniformly across the bins its ``[low, high]``
    range covers. Implemented as a difference array plus ``cumsum`` so the spread
    is exact without a per-bar Python loop.

    *delta* is per-row signed volume from TBBO (see :func:`allocate_tf_delta`).
    Omitted, the delta channel stays zero — a CLV estimate is uncorrelated with
    true delta at price (ADR-123) and a zero channel is honest where noise is not.
    """
    h = day["high"].to_numpy(dtype="float64")
    l = day["low"].to_numpy(dtype="float64")
    v = day["volume"].to_numpy(dtype="float64")

    lo_bin = int(np.floor(l.min() / cfg.bin_size))
    hi_bin = int(np.floor(h.max() / cfg.bin_size))
    n = hi_bin - lo_bin + 1

    b_lo = np.floor(l / cfg.bin_size).astype(np.int64) - lo_bin
    b_hi = np.floor(h / cfg.bin_size).astype(np.int64) - lo_bin
    span = (b_hi - b_lo + 1).astype("float64")

    dlt = np.zeros_like(v) if delta is None else np.asarray(delta, dtype="float64")
    vol_diff = np.zeros(n + 1, dtype="float64")
    dlt_diff = np.zeros(n + 1, dtype="float64")
    np.add.at(vol_diff, b_lo, v / span)
    np.add.at(vol_diff, b_hi + 1, -v / span)
    np.add.at(dlt_diff, b_lo, dlt / span)
    np.add.at(dlt_diff, b_hi + 1, -dlt / span)

    return Profile(lo_bin=lo_bin, bin_size=cfg.bin_size,
                   volume=np.cumsum(vol_diff[:-1]), delta=np.cumsum(dlt_diff[:-1]))


def composite_profile(profiles: list[Profile], cfg: ProfileConfig) -> Profile:
    """Recency-weighted stack of session profiles (index 0 = most recent)."""
    if not profiles:
        raise ValueError("composite_profile needs at least one profile")
    lo = min(p.lo_bin for p in profiles)
    hi = max(p.hi_bin for p in profiles)
    n = hi - lo + 1
    vol = np.zeros(n, dtype="float64")
    dlt = np.zeros(n, dtype="float64")
    for age, p in enumerate(profiles):
        w = 0.5 ** (age / cfg.recency_halflife)
        off = p.lo_bin - lo
        vol[off:off + len(p.volume)] += w * p.volume
        dlt[off:off + len(p.delta)] += w * p.delta
    return Profile(lo_bin=lo, bin_size=cfg.bin_size, volume=vol, delta=dlt)


def find_nodes(profile: Profile, cfg: ProfileConfig) -> tuple[np.ndarray, np.ndarray]:
    """Bin indices of high-volume nodes (HVN) and low-volume nodes (LVN).

    A bin is an HVN when it is the maximum of the window centred on it; LVN is
    the mirrored test on a positive-volume bin. Both are returned as *absolute*
    bin indices.
    """
    v = profile.volume
    if len(v) < 2 * cfg.node_window_bins + 1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    w = cfg.node_window_bins
    padded = np.pad(v, w, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1)
    is_max = (v >= win.max(axis=1)) & (v > 0)
    is_min = (v <= win.min(axis=1)) & (v > 0)
    return (np.flatnonzero(is_max) + profile.lo_bin,
            np.flatnonzero(is_min) + profile.lo_bin)


def value_area(profile: Profile, cfg: ProfileConfig) -> tuple[float, float, float]:
    """``(poc, va_low, va_high)`` prices by the standard expand-from-POC rule."""
    v = profile.volume
    total = v.sum()
    prices = profile.prices()
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    poc = int(np.argmax(v))
    lo = hi = poc
    acc = v[poc]
    target = cfg.value_area_pct * total
    while acc < target and (lo > 0 or hi < len(v) - 1):
        below = v[lo - 1] if lo > 0 else -1.0
        above = v[hi + 1] if hi < len(v) - 1 else -1.0
        if above >= below:
            hi += 1
            acc += v[hi]
        else:
            lo -= 1
            acc += v[lo]
    return float(prices[poc]), float(prices[lo]), float(prices[hi])


def _confluence(node_bins: np.ndarray, per_session_nodes: list[np.ndarray],
                windows: tuple[int, ...], tol: int) -> dict[str, np.ndarray]:
    """For each candidate bin, how many of the last *w* sessions had a node near it."""
    out: dict[str, np.ndarray] = {}
    for w in windows:
        counts = np.zeros(len(node_bins), dtype=np.int16)
        for nodes in per_session_nodes[:w]:
            if len(nodes) == 0:
                continue
            pos = np.searchsorted(nodes, node_bins)
            for shift in (0, -1):
                j = np.clip(pos + shift, 0, len(nodes) - 1)
                counts += (np.abs(nodes[j] - node_bins) <= tol).astype(np.int16)
        out[f"n_sessions_confluence_{w}"] = np.minimum(counts, w)
    return out


def _path_volume(profile: Profile, entry_price: float,
                 levels: np.ndarray) -> np.ndarray:
    """Composite volume sitting strictly between entry and each level.

    The auction-native quantity: volume you must trade through to reach a level.
    A thin path is easy passage; a thick one is where price stalls.
    """
    cum = np.concatenate([[0.0], np.cumsum(profile.volume)])
    e_bin = int(profile.bin_of(entry_price)) - profile.lo_bin
    l_bin = profile.bin_of(levels) - profile.lo_bin
    n = len(profile.volume)
    e = np.clip(e_bin, 0, n)
    lo = np.clip(np.minimum(l_bin, e), 0, n)
    hi = np.clip(np.maximum(l_bin, e), 0, n)
    return cum[hi] - cum[lo]


def union_nodes(*profiles: Profile | None, cfg: ProfileConfig) -> np.ndarray:
    """Node bins found on each profile independently, then unioned.

    Nodes are searched per profile rather than on the summed profile because the
    prior-30-session composite carries ~30x the volume of a developing session:
    summed first, today's nodes are never local maxima and the candidate set goes
    blind exactly when price trends out of the historical range.
    """
    found = []
    for p in profiles:
        if p is None:
            continue
        hvn, lvn = find_nodes(p, cfg)
        if len(hvn) or len(lvn):
            found.append(np.concatenate([hvn, lvn]))
    if not found:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(found))


def select_candidates(
    profile: Profile,
    per_session_nodes: list[np.ndarray],
    entry_price: float,
    r_value: float,
    cfg: ProfileConfig,
    *,
    developing: Profile | None = None,
    node_bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Top-K candidate levels above and below *entry_price*, with descriptors.

    Returns one row per candidate carrying only continuous properties — there is
    deliberately no ``level_type`` column. ``max_candidates`` caps each side: the
    Layer-3 optimiser takes an argmax over these, so the count is a regulariser
    on that selection, not merely a compute knob.

    *node_bins* lets the caller supply a pre-unioned node set (see
    :func:`union_nodes`); omitted, nodes come from *profile* alone.
    """
    if node_bins is None:
        node_bins = union_nodes(profile, developing, cfg=cfg)
    if len(node_bins) == 0 or not np.isfinite(r_value) or r_value <= 0:
        return _empty_candidates()

    prices = (node_bins + 0.5) * cfg.bin_size
    dist = prices - entry_price
    dist_r = np.abs(dist) / r_value
    band = (dist_r >= cfg.min_dist_r) & (dist_r <= cfg.max_dist_r)
    if not band.any():
        return _empty_candidates()

    node_bins, prices, dist, dist_r = node_bins[band], prices[band], dist[band], dist_r[band]
    vol = profile.value_at(prices, "volume")
    dlt = profile.value_at(prices, "delta")
    vol_total = profile.volume.sum()
    strength = (np.searchsorted(np.sort(profile.volume), vol) / max(1, len(profile.volume)))

    df = pd.DataFrame({
        "level": prices,
        "bin": node_bins,
        "dist_pts": np.abs(dist),
        "dist_r": dist_r,
        "above": dist > 0,
        "vol_at_level": vol / vol_total if vol_total > 0 else vol,
        "delta_at_level": dlt / vol_total if vol_total > 0 else dlt,
        "node_strength": strength,
        "path_volume_between": _path_volume(profile, entry_price, prices) / vol_total
                               if vol_total > 0 else 0.0,
    })
    for k, v in _confluence(node_bins, per_session_nodes,
                            cfg.confluence_windows, cfg.confluence_tol_bins).items():
        df[k] = v

    src = developing if developing is not None else profile
    poc, val, vah = value_area(src, cfg)
    va_width = max(vah - val, cfg.bin_size)
    df["position_vs_value"] = (df["level"] - poc) / va_width

    # Runway if the level breaks: gap to the next candidate further out.
    df = df.sort_values("level").reset_index(drop=True)
    fwd = df["level"].diff().shift(-1).abs()
    bwd = df["level"].diff().abs()
    df["room_beyond"] = np.where(df["above"], fwd, bwd) / r_value
    df["room_beyond"] = df["room_beyond"].fillna(cfg.max_dist_r)

    keep = []
    for above in (True, False):
        side = df[df["above"] == above]
        keep.append(side.nlargest(cfg.max_candidates, "node_strength"))
    return pd.concat(keep, ignore_index=True) if keep else _empty_candidates()


def _empty_candidates() -> pd.DataFrame:
    cols = {"level": "float64", "bin": "int64", "dist_pts": "float64", "dist_r": "float64",
            "above": "bool", "vol_at_level": "float64", "delta_at_level": "float64",
            "node_strength": "float64", "path_volume_between": "float64",
            "position_vs_value": "float64", "room_beyond": "float64"}
    df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})
    for w in ProfileConfig().confluence_windows:
        df[f"n_sessions_confluence_{w}"] = pd.Series(dtype="int16")
    return df


def rth_slice(day: pd.DataFrame, start: str = "08:30", end: str = "15:00") -> pd.DataFrame:
    """Rows whose Chicago time falls in ``[start, end)``."""
    ct = day.index.tz_convert("America/Chicago")
    h0, m0 = (int(x) for x in start.split(":"))
    h1, m1 = (int(x) for x in end.split(":"))
    mask = (ct.time >= dtime(h0, m0)) & (ct.time < dtime(h1, m1))
    return day.loc[mask]
