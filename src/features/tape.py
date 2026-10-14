"""Print-level tape: exact volume-at-price, delta-at-price and path resolution.

The tape is the cache written by ``scripts/data/build_tape.py``: one row per CME
print, carrying ``price``, ``size`` and ``side`` (aggressor, ``+1`` buy / ``-1``
sell), ordered by ``ts_event`` then exchange ``sequence``.

Three things become *observed* that were previously reconstructed from 1-second
OHLCV bars:

* **Volume at price.** A bar's volume no longer needs spreading uniformly across
  the bins its ``[low, high]`` covers — each print lands on one price.
* **Delta at price.** Aggressor side arrives per print, so the delta channel of a
  profile is exact rather than a 15-second aggregate redistributed by volume share
  (:func:`src.training.auction_levels.allocate_tf_delta`).
* **First touch.** Barrier resolution on 1s bars cannot order two levels traded in
  the same second, and falls back to ``tie_policy``. The print sequence orders
  them directly.

:func:`tape_to_path` derives a 1-second OHLCV frame from the tape so existing
consumers keep working unchanged; :func:`tape_profile` and :func:`tape_levels`
replace their bar-based counterparts where accuracy matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data_paths import parquet_tape_dir, price_tape_dir
from src.training.auction_levels import Profile, ProfileConfig

logger = logging.getLogger("vtech.features.tape")

TAPE_COLS = ("price", "size", "side")
TAPE_PROFILE_TICK_SIZE = 0.25


def list_tape_files(symbol: str = "NQ") -> list[Path]:
    """Session tape files. ``sources/price_tape/<symbol>`` overrides a same-named
    legacy file under ``sources/features/parquet_tape``.
    """
    by_name: dict[str, Path] = {}
    legacy = parquet_tape_dir()
    if legacy.is_dir():
        for p in legacy.glob("tape_*.parquet"):
            by_name[p.name] = p
    current = price_tape_dir(symbol)
    if current.is_dir():
        for p in current.glob("tape_*.parquet"):
            by_name[p.name] = p
    return [by_name[k] for k in sorted(by_name)]


def tape_path(session: date, root: Path | None = None, *, symbol: str = "NQ") -> Path:
    name = f"tape_{session.isoformat()}.parquet"
    if root is not None:
        return Path(root) / name
    current = price_tape_dir(symbol) / name
    if current.exists():
        return current
    return parquet_tape_dir() / name


def load_tape(session: date, root: Path | None = None) -> pd.DataFrame | None:
    """One session's prints, or ``None`` if the tape was never built."""
    p = tape_path(session, root)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def tape_to_path(tape: pd.DataFrame) -> pd.DataFrame:
    """1-second OHLCV bars from prints — a drop-in for ``parquet_path1s`` frames.

    Seconds with no print are absent, matching the exchange-native convention the
    path consumers already expect (no trade means no excursion).
    """
    g = tape.groupby(tape.index.floor("1s"))
    out = pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["size"].sum().astype("int64"),
    })
    out.index.name = "ts"
    return out


def tape_delta_1s(tape: pd.DataFrame) -> pd.Series:
    """Exact signed volume per second — replaces ``allocate_tf_delta``."""
    signed = tape["size"].to_numpy(dtype="float64") * tape["side"].to_numpy(dtype="float64")
    return pd.Series(signed, index=tape.index.floor("1s")).groupby(level=0).sum()


def print_ofi(tape: pd.DataFrame) -> np.ndarray:
    """Cont (2014) order-flow imbalance per print, from the quote at the print.

    OFI is a *book* measure — it counts size added to and pulled from the best bid
    and offer — so it is not a restatement of trade delta. Computing it needs the
    BBO snapshot attached to each print, which is why the tape stores
    ``bid_px/ask_px/bid_sz/ask_sz``. The first print of a session has no previous
    quote and is NaN.
    """
    bid_px = tape["bid_px"].to_numpy(dtype="float64")
    ask_px = tape["ask_px"].to_numpy(dtype="float64")
    bid_sz = tape["bid_sz"].to_numpy(dtype="float64")
    ask_sz = tape["ask_sz"].to_numpy(dtype="float64")
    n = bid_px.size
    if n == 0:
        return np.empty(0, dtype="float64")

    prev_bid = np.concatenate(([np.nan], bid_px[:-1]))
    prev_ask = np.concatenate(([np.nan], ask_px[:-1]))
    prev_bid_sz = np.concatenate(([np.nan], bid_sz[:-1]))
    prev_ask_sz = np.concatenate(([np.nan], ask_sz[:-1]))
    ofi = (np.where(bid_px >= prev_bid, bid_sz, 0.0)
           - np.where(bid_px <= prev_bid, prev_bid_sz, 0.0)
           - np.where(ask_px <= prev_ask, ask_sz, 0.0)
           + np.where(ask_px >= prev_ask, prev_ask_sz, 0.0))
    ofi[0] = np.nan
    return ofi


def tape_flow_bars(tape: pd.DataFrame, timestep: str = "15s") -> pd.DataFrame:
    """Bar-end indexed ``of_delta``/``of_ofi``/``of_cvd``/``of_ofi_cum``/``cs_close``.

    Shaped like a slice of the native-TF hive so it drops straight into
    :func:`src.training.entry_state.build_entry_state`, but computed from this
    session's prints on the lead contract rather than read from a store built
    under a different symbol filter.
    """
    if tape.empty:
        return pd.DataFrame()
    bucket = tape.index.floor(timestep)
    signed = tape["size"].to_numpy(dtype="float64") * tape["side"].to_numpy(dtype="float64")
    ofi = print_ofi(tape)
    g = pd.DataFrame({"signed": signed, "ofi": ofi,
                      "price": tape["price"].to_numpy(dtype="float64")},
                     index=bucket).groupby(level=0)
    out = pd.DataFrame({
        "of_delta": g["signed"].sum(),
        "of_ofi": g["ofi"].sum(),
        "cs_close": g["price"].last(),
    })
    out.index = out.index + pd.Timedelta(timestep)
    out.index.name = "bar_end_ts"
    out["of_cvd"] = out["of_delta"].cumsum()
    out["of_ofi_cum"] = out["of_ofi"].cumsum()
    return out


def tape_profile(tape: pd.DataFrame, cfg: ProfileConfig) -> Profile:
    """Volume- and delta-at-price from prints.

    No uniform spreading: every print contributes its whole size to the single bin
    containing its price, and its signed size to that bin's delta.
    """
    px = tape["price"].to_numpy(dtype="float64")
    sz = tape["size"].to_numpy(dtype="float64")
    sd = tape["side"].to_numpy(dtype="float64")

    bins = np.floor(px / cfg.bin_size).astype(np.int64)
    lo_bin = int(bins.min())
    n = int(bins.max()) - lo_bin + 1
    idx = bins - lo_bin

    vol = np.zeros(n, dtype="float64")
    dlt = np.zeros(n, dtype="float64")
    np.add.at(vol, idx, sz)
    np.add.at(dlt, idx, sz * sd)
    return Profile(lo_bin=lo_bin, bin_size=cfg.bin_size, volume=vol, delta=dlt)


@dataclass(frozen=True)
class TapeLevels:
    """Value-area geometry of one session, computed from prints."""

    session: pd.Timestamp
    lo: float
    val: float
    poc: float
    vah: float
    hi: float


def tape_levels(tape: pd.DataFrame, session: pd.Timestamp, cfg: ProfileConfig
                ) -> TapeLevels | None:
    """``lo/val/poc/vah/hi`` from an exact tick-level print profile.

    The model uses the same 0.25-point tick profile and tie rule as the WebUI.
    ``cfg`` remains in the signature for callers that also build legacy profiles;
    session levels deliberately use the NQ tick rather than its ``bin_size``.
    """
    if tape.empty:
        return None
    raw_px = tape["price"].to_numpy(dtype="float64")
    sizes = tape["size"].to_numpy(dtype="float64")
    ticks = np.round(raw_px / TAPE_PROFILE_TICK_SIZE) * TAPE_PROFILE_TICK_SIZE
    prices, inverse = np.unique(ticks, return_inverse=True)
    volumes = np.bincount(inverse, weights=sizes, minlength=prices.size)
    if volumes.sum() <= 0:
        return None
    poc, vah, val = tick_value_area(prices, volumes, cfg.value_area_pct)
    return TapeLevels(session=session,
                      lo=float(raw_px.min()), val=val, poc=poc, vah=vah,
                      hi=float(raw_px.max()))


def tick_value_area(prices: np.ndarray, volumes: np.ndarray,
                    va_pct: float = 0.70) -> tuple[float, float, float]:
    """Return ``(POC, VAH, VAL)`` using the WebUI's tick expansion rule."""
    px = np.asarray(prices, dtype="float64")
    vol = np.asarray(volumes, dtype="float64")
    ok = np.isfinite(px) & np.isfinite(vol) & (vol > 0)
    if not bool(ok.any()):
        return (float("nan"), float("nan"), float("nan"))
    order = np.argsort(px[ok], kind="mergesort")
    px, vol = px[ok][order], vol[ok][order]
    poc_i = int(np.argmax(vol))
    target = float(vol.sum()) * float(va_pct)
    lo = hi = poc_i
    acc = float(vol[poc_i])
    while acc + 1e-12 < target and (lo > 0 or hi < len(vol) - 1):
        above = float(vol[hi + 1]) if hi + 1 < len(vol) else -1.0
        below = float(vol[lo - 1]) if lo > 0 else -1.0
        if above > below:
            hi += 1
            acc += above
        elif below > above:
            lo -= 1
            acc += below
        else:
            if above >= 0:
                hi += 1
                acc += above
            if below >= 0:
                lo -= 1
                acc += below
    return float(px[poc_i]), float(px[hi]), float(px[lo])


def first_touch(prices: np.ndarray, start: int, upper: float, lower: float
                ) -> tuple[int, int]:
    """First print index at or beyond either barrier, scanning forward from *start*.

    Returns ``(idx, which)`` with ``which`` ``+1`` upper, ``-1`` lower, ``0`` neither
    before the end of the array. Because prints are ordered by ``(ts_event,
    sequence)``, a barrier pair touched inside the same second resolves by true
    execution order — there is no tie policy to apply.
    """
    seg = prices[start:]
    up = np.flatnonzero(seg >= upper)
    dn = np.flatnonzero(seg <= lower)
    i_up = int(up[0]) if up.size else -1
    i_dn = int(dn[0]) if dn.size else -1
    if i_up < 0 and i_dn < 0:
        return len(prices) - 1, 0
    if i_dn < 0 or (0 <= i_up < i_dn):
        return start + i_up, 1
    return start + i_dn, -1


def excursions(prices: np.ndarray, start: int, end: int, entry: float
               ) -> tuple[float, float]:
    """``(mfe_up, mfe_down)`` in points over ``prices[start:end+1]`` versus *entry*.

    Both are reported as non-negative magnitudes: how far price ran above the
    entry, and how far below. A directional move label reads one as favourable and
    the other as adverse depending on side.
    """
    seg = prices[start:end + 1]
    if seg.size == 0:
        return 0.0, 0.0
    return float(max(seg.max() - entry, 0.0)), float(max(entry - seg.min(), 0.0))


def sparse_tables(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse tables for O(1) range max and range min over the print array.

    Prices are exact in float32 at index-future scale (30,000.25 needs 17 bits of
    mantissa against 24 available), which halves the tables against float64.
    """
    n = px.size
    levels = max(1, int(np.floor(np.log2(n))) + 1)
    hi = np.empty((levels, n), dtype="float32")
    lo = np.empty((levels, n), dtype="float32")
    hi[0] = px
    lo[0] = px
    for j in range(1, levels):
        w = 1 << (j - 1)
        m = n - (1 << j) + 1
        if m <= 0:
            hi[j] = hi[j - 1]
            lo[j] = lo[j - 1]
            continue
        np.maximum(hi[j - 1, :m], hi[j - 1, w:w + m], out=hi[j, :m])
        np.minimum(lo[j - 1, :m], lo[j - 1, w:w + m], out=lo[j, :m])
        hi[j, m:] = hi[j - 1, m:]
        lo[j, m:] = lo[j - 1, m:]
    return hi, lo


def rmq(table: np.ndarray, l: np.ndarray, r: np.ndarray, is_max: bool) -> np.ndarray:
    """Range max/min over ``[l, r]`` inclusive, vectorised across all queries."""
    span = np.maximum(r - l + 1, 1)
    j = np.floor(np.log2(span)).astype(np.int64)
    a = table[j, l]
    b = table[j, np.maximum(r - (1 << j) + 1, 0)]
    return np.maximum(a, b) if is_max else np.minimum(a, b)


def first_cross(table: np.ndarray, start: np.ndarray, stop: np.ndarray,
                level: np.ndarray, upward: bool) -> np.ndarray:
    """First index in ``[start, stop]`` where price crosses *level*, else ``-1``.

    *table* is the max table for an upward cross, the min table for a downward
    one. The running extreme since ``start`` is monotone, so the crossing index is
    found by a binary search that runs vectorised across every query at once —
    ``log2(n)`` range queries total rather than one scan per query.
    """
    ext = rmq(table, start, stop, upward).astype("float64")
    hit = ext >= level if upward else ext <= level
    out = np.full(start.size, -1, dtype="int64")
    if not hit.any():
        return out
    lo_i = start[hit].copy()
    hi_i = stop[hit].copy()
    s0 = start[hit]
    tgt = level[hit]
    while np.any(lo_i < hi_i):
        mid = (lo_i + hi_i) // 2
        e = rmq(table, s0, mid, upward).astype("float64")
        ok = e >= tgt if upward else e <= tgt
        hi_i = np.where(ok, mid, hi_i)
        lo_i = np.where(ok, lo_i, mid + 1)
    out[hit] = lo_i
    return out
