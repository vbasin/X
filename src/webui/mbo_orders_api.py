"""Live Chart "Order map" data: large resting orders + reload chains from `databento.mbo`.

`assemble` is pure (event/candle frames in, JSON-able dict out) and unit-tested without a
database; `order_map` is the thin DB wrapper the API route calls. `order_events` is the
per-order row lookup behind a double-click on a drawn order. Lifecycle / outcome / chain
logic lives in `src.webui.mbo_orders` and is shared with `scripts/research/mbo_large_orders.py`.

Timestamps in the payload are unix seconds (UTC). `databento.mbo` is a rolling ~1-month window
served ~8 h behind real time, so the payload always carries the data coverage and whether the
requested window was clamped to it.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.webui.mbo_orders import (CANDLE_SQL, CLEARS_SQL, ICEBERG_SQL, ORDERS_SQL,
                                  RESTING_SNAPSHOT_SQL,
                                  SNAP_BOUNDARIES_SQL, build_segments, classify, find_chains,
                                  seed_from_snapshot)

MAX_SPAN_H = 5 * 24        # longest window computed in one request (newest 5 days kept)
QUERY_MS = 90_000          # 5-day NQU6 ORDERS_SQL with instrument_id ~37s
ICEBERG_MS = 45_000        # 5-day iceberg scan ~14s; on timeout the map still returns large orders
ICE_ID_CAP = 200           # extra order_id JOIN without an index; 5-day June has ~2900 icebergs
ICEBERG_SPAN_H = 36        # skip ICEBERG_SQL on longer windows; 5-day d-CTE never returns
DETAIL_PAD = timedelta(seconds=2)   # ts_recv slack around a segment so the boundary event is included
DETAIL_ROW_CAP = 20_000
DETAIL_MS = 90_000
_NQ_OUTRIGHT = re.compile(r"^NQ[FGHJKMNQUVXZ][0-9]$")
# `databento.mbo` is partitioned on ts_recv and has no order_id index, so a bare
# order_id scan would read every chunk. The predicate is order_id plus this
# ts_recv window; the row order the user asked for is ts_event, then sequence.
# The action CASE is only a tie-break when those two are equal (same packet).
ORDER_EVENTS_SQL = """
SELECT ts_recv, ts_event, rtype, publisher_id, instrument_id,
       action, side, price, size, channel_id, order_id, flags,
       ts_in_delta, sequence, symbol, dataset
FROM databento.mbo
WHERE order_id = %(order_id)s
  AND symbol = %(symbol)s
  AND ts_recv >= %(t0)s
  AND ts_recv < %(t1)s
ORDER BY ts_event, sequence,
         CASE action WHEN 'A' THEN 0 WHEN 'F' THEN 1 WHEN 'M' THEN 2 ELSE 3 END
LIMIT %(lim)s
"""
_CACHE_MAX = 6
_cache: "OrderedDict[tuple, tuple[pd.DataFrame, pd.DataFrame, float]]" = OrderedDict()
_cov_cache: dict[str, Any] = {"t": 0.0, "v": None}
_lock = threading.Lock()   # one heavy computation at a time
_iid_cache: dict[str, int] = {}


class OrderMapBusy(Exception):
    """Another order-map window is already computing; the client should retry."""


def _f(x: Any, nd: int = 3) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def parse_ts(value: str | float | int | datetime) -> datetime:
    """ISO-8601 (Z or offset; naive = UTC) or unix seconds -> aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        try:
            dt = datetime.fromtimestamp(float(s), tz=timezone.utc)
        except ValueError:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"not an ISO datetime or unix seconds: {value!r}") from exc
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def assemble(events: pd.DataFrame, candles: pd.DataFrame, t1: datetime, *, min_size: int,
             approach: float, chain_n: int, chain_dt: float, chain_min_secs: float,
             chain_min_abs: int, break_pts: float, seg: pd.DataFrame | None = None
             ) -> dict[str, Any]:
    """Events + 1-min trade candles -> order lines, reload chains and outcome summary."""
    if seg is None:
        ev = events.copy()
        if not ev.empty:
            ev["ts_recv"] = pd.to_datetime(ev["ts_recv"], utc=True)
        seg = build_segments(ev, pd.Timestamp(t1))
    candles = candles.copy()
    if not candles.empty:
        candles["t"] = pd.to_datetime(candles["t"], utc=True)
    if seg.empty:
        return {"orders": [], "chains": [], "summary": {"n_orders": 0, "outcomes": {}}}
    seg = seg.copy()
    for c in ("t_start", "t_end"):
        seg[c] = pd.to_datetime(seg[c], utc=True)
    # Reserve (iceberg) test, exact and per visit to a price level: this segment took more than it
    # ever displayed there. `filled`/`disp` come from build_segments, so absorbed can never
    # contradict the fill count shown beside it. Kept regardless of displayed size -- filtering on
    # that size is precisely what hid these orders (ADR-149).
    if {"filled", "disp"} <= set(seg.columns):
        f_ = seg["filled"].fillna(0)
        d_ = seg["disp"].fillna(0)
        l_ = seg["left"].fillna(0) if "left" in seg.columns else 0
        # Conservation test for hidden size, and the only formulation that does not depend on the
        # order of records inside a packet: an order that displayed at most `disp` here, gave up
        # `filled`, and still had `left` showing when it finished, must have held filled + left.
        # More than it ever showed => the surplus was hidden.
        #
        # This replaces an earlier "a Modify put the size back up after a fill" test, which was
        # wrong for a subtle reason worth keeping: CME packs several F and M records for one order
        # into a single (ts_recv, sequence), and their relative order within that packet is
        # arbitrary. ORDERS_SQL breaks such ties by size ascending, so the largest Modify always
        # sorted last and looked like a replenishment. Order 6882403938353 -- a plain 50-lot ask
        # eaten down 49, 48, 44, ... absorbing exactly the 50 it showed -- was badged on that
        # artifact alone. The conservation test gives 50 + 0 > 50 = false, correctly.
        #
        # `filled > 0` stays as a floor: nothing executed here means nothing was proven here.
        ice = (f_ > 0) & ((f_ + l_) > d_) & ((f_ >= min_size) | (d_ >= min_size))
        seg["ice_abs"] = np.where(ice, f_, np.nan)
        seg["ice_disp"] = np.where(ice, d_, np.nan)
    else:
        seg["ice_abs"] = np.nan
        seg["ice_disp"] = np.nan
    # Classify only what is displayed.
    shown = seg[(seg["size0"] >= min_size) | seg["ice_abs"].notna()].copy()
    if candles.empty:
        shown["touched"], shown["dist_end"], shown["outcome"] = False, np.nan, None
        chains = pd.DataFrame()
    else:
        shown = classify(shown, candles, approach) if len(shown) else shown.assign(
            touched=False, dist_end=np.nan, outcome=None)
        chains = find_chains(seg, candles, chain_dt, chain_n, break_pts,
                             chain_min_secs, chain_min_abs)
    orders = [{
        "i": int(r.order_id), "s": r.side, "p": _f(r.price, 4), "z": int(r.size0),
        "a": _f(r.t_start.timestamp()), "b": _f(r.t_end.timestamp()), "e": r.end,
        "f": int(r.filled) if pd.notna(r.filled) else 0, "pre": bool(r.pre),
        "o": (r.outcome if isinstance(r.outcome, str) else None), "d": _f(r.dist_end, 2),
        # Reserve order: `ia` is what it actually absorbed, `id_` what it ever displayed.
        **({"ice": True, "ia": int(r.ice_abs), "idisp": int(r.ice_disp)}
           if getattr(r, "ice_abs", None) is not None and pd.notna(getattr(r, "ice_abs", None))
           else {}),
    } for r in shown.itertuples(index=False)]
    ch = []
    for r in chains.itertuples(index=False) if not chains.empty else []:
        ch.append({"s": r.side, "p": _f(r.price, 4), "a": _f(r.t_start.timestamp()),
                   "b": _f(r.t_end.timestamp()), "n": int(r.orders), "clip": int(r.clip),
                   "abs": int(r.absorbed), "pinned": bool(r.pinned), "after": r.after,
                   "mfe": _f(r.mfe5m, 2), "raw": int(chains.attrs.get("raw", len(chains)))})
    final = shown[shown["end"] != "moved"]
    if "ice_abs" in final.columns:
        ice_rows = final[final["ice_abs"].notna()]
        n_ice = int(ice_rows["order_id"].nunique())
        ice_abs = int(ice_rows.drop_duplicates("order_id")["ice_abs"].sum())
    else:
        n_ice, ice_abs = 0, 0      # caller passed a frame predating iceberg annotation
    return {
        "orders": orders, "chains": ch,
        "summary": {"n_orders": int(final["order_id"].nunique()), "n_segments": len(orders),
                    "outcomes": {str(k): int(v) for k, v in final["outcome"].value_counts().items()},
                    "n_icebergs": n_ice, "ice_absorbed": ice_abs},
    }


def _query():  # late import: live_chart pulls heavy deps and is only needed with a DB
    from src.webui.live_chart import _query as q
    return q


def coverage() -> dict[str, Any]:
    """{min, max} unix seconds of databento.mbo (chunk catalog; cached 60 s).

    ``min(ts_recv)`` / ``max(ts_recv)`` on the hypertable decompresses every chunk when
    compression orderby is ``ts_event, sequence`` (the partition column is only a trailing
    orderby key). Chunk range_start / range_end are the same bounds the planner uses.
    """
    now = time.time()
    if _cov_cache["v"] is not None and now - _cov_cache["t"] < 60:
        return _cov_cache["v"]
    q = _query()
    df = q(
        "SELECT min(range_start) AS lo, max(range_end) AS hi "
        "FROM timescaledb_information.chunks "
        "WHERE hypertable_schema = 'databento' AND hypertable_name = 'mbo'",
        (), timeout_ms=10_000,
    )
    lo = df["lo"].iloc[0] if not df.empty else None
    hi_excl = df["hi"].iloc[0] if not df.empty else None
    if lo is None or hi_excl is None or pd.isna(lo) or pd.isna(hi_excl):
        df = q("SELECT (SELECT min(ts_recv) FROM databento.mbo) AS lo, "
               "(SELECT max(ts_recv) FROM databento.mbo) AS hi", (), timeout_ms=20_000)
        lo, hi = df["lo"].iloc[0], df["hi"].iloc[0]
        v = {"min": _f(pd.Timestamp(lo).timestamp()) if pd.notna(lo) else None,
             "max": _f(pd.Timestamp(hi).timestamp()) if pd.notna(hi) else None}
    else:
        # range_end is exclusive (next chunk start). Callers add 1 µs to cov["max"].
        hi = pd.Timestamp(hi_excl) - pd.Timedelta(microseconds=1)
        v = {"min": _f(pd.Timestamp(lo).timestamp()), "max": _f(hi.timestamp())}
    _cov_cache.update(t=now, v=v)
    return v


_sym_cache: dict[tuple[str, str], str | None] = {}


def _pick_symbol(t0: datetime, t1: datetime) -> str | None:
    """Lead NQ outright in ``[t0, t1)``.

    Prefers ``mbo_resting_orders_v2`` (tiny) over counting trades in ``databento.mbo``.
    A COUNT(*) of action='T' on compressed chunks ordered by ts_event times out in 60s
    on a 5-day window, which 504s the whole order map before any overlay SQL runs.
    """
    k = (t0.isoformat(), t1.isoformat())
    if k in _sym_cache:
        return _sym_cache[k]
    q = _query()
    p = {"t0": t0, "t1": t1}
    df = q(
        "SELECT symbol FROM databento.mbo_resting_orders_v2 "
        "WHERE snapshot_ts > %(t0)s AND snapshot_ts <= %(t1)s "
        "AND symbol ~ '^NQ[FGHJKMNQUVXZ][0-9]$' "
        "GROUP BY 1 ORDER BY count(*) DESC LIMIT 1",
        p, timeout_ms=15_000,
    )
    if df.empty:
        sql = ("SELECT symbol FROM databento.mbo WHERE ts_recv >= %(t0)s AND ts_recv < %(t1)s "
               "AND action = 'T' AND symbol ~ '^NQ[FGHJKMNQUVXZ][0-9]$' LIMIT 1")
        for lo in (max(t0, t1 - timedelta(hours=2)), t0):
            df = q(sql, {"t0": lo, "t1": t1}, timeout_ms=30_000)
            if not df.empty:
                break
    sym = str(df["symbol"].iloc[0]) if not df.empty else None
    if len(_sym_cache) > 256:
        _sym_cache.clear()
    _sym_cache[k] = sym
    return sym


def _instrument_id(sym: str, t0: datetime, t1: datetime) -> int | None:
    """compress_segmentby key for ``sym``. Cached; bounded to the requested window."""
    if sym in _iid_cache:
        return _iid_cache[sym]
    q = _query()
    df = q(
        "SELECT instrument_id FROM databento.mbo "
        "WHERE symbol = %(symbol)s AND ts_recv >= %(t0)s AND ts_recv < %(t1)s "
        "AND ts_event >= %(t0)s AND ts_event < %(t1)s LIMIT 1",
        {"symbol": sym, "t0": t0, "t1": t1},
        timeout_ms=15_000,
    )
    iid = int(df["instrument_id"].iloc[0]) if not df.empty else None
    if iid is not None:
        _iid_cache[sym] = iid
    return iid


def cap_span(t0: datetime, t1: datetime) -> tuple[datetime, bool]:
    """Keep the newest ``MAX_SPAN_H`` of ``[t0, t1)``. The flag is true when the start moved."""
    if t1 - t0 > timedelta(hours=MAX_SPAN_H):
        return t1 - timedelta(hours=MAX_SPAN_H), True
    return t0, False


def order_map(start: str | float, end: str | float, *, symbol: str | None = None,
              min_size: int = 20, clip_min: int = 5, approach: float = 10.0, chain_n: int = 3,
              chain_dt: float = 2.0, chain_min_secs: float = 10.0, chain_min_abs: int = 50,
              break_pts: float = 1.0) -> dict[str, Any]:
    """Order-map payload for [start, end) clamped to MBO coverage (newest 5 days kept)."""
    import psycopg
    try:
        return _order_map(
            start, end, symbol=symbol, min_size=min_size, clip_min=clip_min,
            approach=approach, chain_n=chain_n, chain_dt=chain_dt,
            chain_min_secs=chain_min_secs, chain_min_abs=chain_min_abs,
            break_pts=break_pts,
        )
    except OrderMapBusy:
        raise
    except psycopg.errors.QueryCanceled as exc:
        # The symbol probe and the map queries share a statement timeout.
        # Leaving QueryCanceled uncaught becomes a bare 500, which the chart
        # shows as "500 Internal Server Error" with no time-limit detail.
        raise TimeoutError("MBO order map exceeded the time limit") from exc


def _order_map(start: str | float, end: str | float, *, symbol: str | None = None,
               min_size: int = 20, clip_min: int = 5, approach: float = 10.0, chain_n: int = 3,
               chain_dt: float = 2.0, chain_min_secs: float = 10.0, chain_min_abs: int = 50,
               break_pts: float = 1.0) -> dict[str, Any]:
    if not (1 <= min_size <= 100_000 and 1 <= clip_min <= 1000):
        raise ValueError("min_size / clip_min out of range")
    t0, t1 = parse_ts(start), parse_ts(end)
    cov = coverage()
    if cov["min"] is None:
        raise LookupError("databento.mbo is empty")
    lo = datetime.fromtimestamp(cov["min"], tz=timezone.utc)
    hi = datetime.fromtimestamp(cov["max"], tz=timezone.utc) + timedelta(microseconds=1)
    req0, req1 = t0, t1
    t0, t1 = max(t0, lo), min(t1, hi)
    meta = {"data_min": cov["min"], "data_max": cov["max"], "req_start": req0.timestamp(),
            "req_end": req1.timestamp(), "clamped": (t0, t1) != (req0, req1), "truncated": False,
            "max_span_h": MAX_SPAN_H}
    if t1 <= t0:
        return {**meta, "symbol": symbol, "start": None, "end": None, "orders": [], "chains": [],
                "summary": {"n_orders": 0, "outcomes": {}}, "note": "no MBO data in that window"}
    t0, meta["truncated"] = cap_span(t0, t1)
    # The chart's window is 'now'-relative and refreshes constantly; snap the start to a 30-min
    # boundary so repeated requests share one cache entry (extra history before the visible start
    # is simply off-screen).
    t0 = t0.replace(minute=(t0.minute // 30) * 30, second=0, microsecond=0)
    # Fetch at min_size, not clip_min. clip_min=5 on a 5-day window pulled 369k events and
    # spent ~85s in build_segments; the chart aborted with 0 bytes and no error. Chains still
    # run on the fetched set (reload clips below min_size are a 2h-window concern).
    sym = symbol or _pick_symbol(t0, t1)
    if not sym:
        return {**meta, "symbol": None, "start": t0.timestamp(), "end": t1.timestamp(),
                "orders": [], "chains": [], "summary": {"n_orders": 0, "outcomes": {}},
                "note": "no NQ trades in that window"}
    key = (sym, t0.isoformat(), t1.isoformat(), min_size)
    took, hit, notes = 0.0, True, []
    if not _lock.acquire(blocking=False):
        got = _cache.get(key)
        if got is None:
            raise OrderMapBusy(
                "MBO order map is still computing another window; retry shortly"
            )
        seg, candles, _ = got
    else:
        try:
            got = _cache.get(key)
            if got is None:
                hit, t_a = False, time.time()
                iid = _instrument_id(sym, t0, t1)
                if iid is None:
                    return {**meta, "symbol": sym, "start": t0.timestamp(), "end": t1.timestamp(),
                            "orders": [], "chains": [], "summary": {"n_orders": 0, "outcomes": {}},
                            "note": "no NQ trades in that window", "cache_hit": False,
                            "compute_s": round(time.time() - t_a, 1)}
                p = {"symbol": sym, "t0": t0, "t1": t1, "min_size": min_size, "instrument_id": iid}
                q = _query()
                # ADR-147: orders already resting when the window opened have no Add inside
                # [t0, t1) for ORDERS_SQL's `big` CTE to find on its own -- find the nearest
                # resting-book checkpoint at or before t0 and widen the real-event query back to
                # it, so build_segments() sees a continuous stream from a known state instead of
                # only what started fresh inside the display window.
                snap = q(RESTING_SNAPSHOT_SQL, p, timeout_ms=QUERY_MS)
                span_h = (t1 - t0).total_seconds() / 3600.0
                long_win = span_h > ICEBERG_SPAN_H
                # 5-day windows: do not widen to the resting checkpoint or JOIN thousands of
                # extra order_ids (Aug 24–29 sat in that JOIN past 3 min). Seed Adds are still
                # concat'd below so the book at t0 is drawn.
                if long_win:
                    q_start = t0
                    notes.append(
                        "icebergs: displayed-size only on windows longer than "
                        f"{int(ICEBERG_SPAN_H)}h"
                    )
                else:
                    q_start = snap["snapshot_ts"].min().to_pydatetime() if not snap.empty else t0
                cw = {**p, "t0": q_start}
                seed_ids = {int(i) for i in snap["order_id"]} if not snap.empty else set()
                ice_ids: set[int] = set()
                import psycopg
                with ThreadPoolExecutor(max_workers=5) as pool:
                    f_cd = pool.submit(q, CANDLE_SQL, p, timeout_ms=QUERY_MS)
                    f_cl = pool.submit(q, CLEARS_SQL, cw, timeout_ms=QUERY_MS)
                    f_bd = pool.submit(q, SNAP_BOUNDARIES_SQL, cw, timeout_ms=QUERY_MS)
                    f_ice = None
                    if not long_win:
                        f_ice = pool.submit(
                            q, ICEBERG_SQL, {**p, "min_size": min_size}, timeout_ms=ICEBERG_MS)
                    if f_ice is not None:
                        try:
                            ice = f_ice.result()
                        except psycopg.errors.QueryCanceled:
                            ice = pd.DataFrame()
                            notes.append("icebergs skipped (scan timed out)")
                        ice_ids = {int(i) for i in ice["order_id"]} if not ice.empty else set()
                        if len(ice_ids) > ICE_ID_CAP:
                            notes.append(
                                f"icebergs omitted ({len(ice_ids)} in window; "
                                f"showing displayed size ≥ {min_size})"
                            )
                            ice_ids = set()
                    extra = [] if long_win else sorted(ice_ids | seed_ids)
                    f_ev = pool.submit(
                        q, ORDERS_SQL, {**p, "t0": q_start, "extra_ids": extra},
                        timeout_ms=QUERY_MS)
                    candles, r_cl, r_bd, ev = (
                        f_cd.result(), f_cl.result(), f_bd.result(), f_ev.result())
                if not ev.empty:
                    ev["ts_recv"] = pd.to_datetime(ev["ts_recv"], utc=True)
                seed = seed_from_snapshot(snap)
                if not seed.empty:
                    seed["ts_recv"] = pd.to_datetime(seed["ts_recv"], utc=True)
                if not seed.empty:
                    ev = pd.concat([seed, ev], ignore_index=True) if not ev.empty else seed
                # ADR-149: exchange-side removals over the *widened* range, so an order that was
                # purged at a session close stops being drawn as resting from there on. Both were
                # fetched above, in parallel with the iceberg scan.
                _ts = pd.Series(dtype="datetime64[ns, UTC]")
                clears = pd.to_datetime(r_cl.get("ts_recv", _ts), utc=True).tolist()
                bounds = pd.to_datetime(r_bd.get("ts_recv", _ts), utc=True).tolist()
                seg = build_segments(ev, pd.Timestamp(t1), clears=clears, boundaries=bounds)
                # NOTE: ICEBERG_SQL is only used to decide which order_ids to FETCH (their displayed
                # size is below min_size, so ORDERS_SQL's own filter can never reach them). The badge
                # itself is derived in assemble() from build_segments' per-segment `filled` and `disp`,
                # which are exact per visit to a level -- the SQL totals are per (order, price) and so
                # still double-count an order that leaves a level and comes back (ADR-149).
                took = time.time() - t_a
                _cache[key] = got = (seg, candles, took)
                while len(_cache) > _CACHE_MAX:
                    _cache.popitem(last=False)
            else:
                _cache.move_to_end(key)
            seg, candles, _ = got
        finally:
            _lock.release()
    out = assemble(pd.DataFrame(), candles, t1, min_size=min_size, approach=approach,
                   chain_n=chain_n, chain_dt=chain_dt, chain_min_secs=chain_min_secs,
                   chain_min_abs=chain_min_abs, break_pts=break_pts, seg=seg)
    if notes and not out.get("note"):
        out["note"] = "; ".join(notes)
    return {**meta, "symbol": sym, "start": t0.timestamp(), "end": t1.timestamp(),
            "cache_hit": hit, "compute_s": round(took, 1),
            "params": {"min_size": min_size, "clip_min": clip_min, "approach": approach,
                       "chain_n": chain_n, "chain_dt": chain_dt,
                       "chain_min_secs": chain_min_secs, "chain_min_abs": chain_min_abs},
            **out}


def order_event_window(start: datetime, end: datetime, *, coverage_lo: datetime,
                       coverage_hi: datetime) -> tuple[datetime, datetime, bool]:
    """ts_recv bounds for one order's displayed life.

    Pads the requested ``[start, end]`` by ``DETAIL_PAD``, clamps it to MBO coverage,
    and keeps at most ``MAX_SPAN_H`` (the newest part). The flag is true when that
    cap moved the start. An empty intersection returns ``(t, t, False)``."""
    a, b = start - DETAIL_PAD, end + DETAIL_PAD
    if b < a:
        a, b = b, a
    a, b = max(a, coverage_lo), min(b, coverage_hi)
    if b <= a:
        return a, a, False
    if b - a > timedelta(hours=MAX_SPAN_H):
        return b - timedelta(hours=MAX_SPAN_H), b, True
    return a, b, False


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def _txt(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def event_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One JSON object per MBO record. ``order_id`` is a string so it cannot round."""
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for r in frame.itertuples(index=False):
        oid = _int(r.order_id)
        rows.append({
            "ts_recv": _iso(r.ts_recv),
            "ts_event": _iso(r.ts_event),
            "rtype": _int(r.rtype),
            "publisher_id": _int(r.publisher_id),
            "instrument_id": _int(r.instrument_id),
            "action": _txt(r.action),
            "side": _txt(r.side),
            "price": _f(r.price, 4),
            "size": _int(r.size),
            "channel_id": _int(r.channel_id),
            "order_id": None if oid is None else str(oid),
            "flags": _int(r.flags),
            "ts_in_delta": _int(r.ts_in_delta),
            "sequence": _int(r.sequence),
            "symbol": _txt(r.symbol),
            "dataset": _txt(r.dataset),
        })
    return rows


def order_events(order_id: int, start: str | float, end: str | float, *,
                 symbol: str) -> dict[str, Any]:
    """Every ``databento.mbo`` row for one order, ordered by ts_event, sequence.

    The scan is limited to the order's displayed life (padded, clamped to coverage,
    newest 5 days). A wider bare ``order_id`` predicate would decompress the whole
    hypertable.
    """
    try:
        oid = int(order_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("order_id must be an integer") from exc
    if oid <= 0:
        raise ValueError("order_id must be a positive integer")
    sym = str(symbol or "").strip()
    if not _NQ_OUTRIGHT.fullmatch(sym):
        raise ValueError("symbol must be an NQ outright (e.g. NQZ6)")
    t0_req, t1_req = parse_ts(start), parse_ts(end)
    cov = coverage()
    if cov["min"] is None:
        raise LookupError("databento.mbo is empty")
    lo = datetime.fromtimestamp(cov["min"], tz=timezone.utc)
    hi = datetime.fromtimestamp(cov["max"], tz=timezone.utc) + timedelta(microseconds=1)
    t0, t1, truncated = order_event_window(t0_req, t1_req, coverage_lo=lo, coverage_hi=hi)
    out: dict[str, Any] = {
        "order_id": str(oid), "symbol": sym,
        "start": None if t1 <= t0 else t0.timestamp(),
        "end": None if t1 <= t0 else t1.timestamp(),
        "truncated": truncated, "row_cap": False, "max_span_h": MAX_SPAN_H,
        "n": 0, "rows": [],
    }
    if t1 <= t0:
        out["note"] = "no MBO coverage in that window"
        return out
    import psycopg
    try:
        frame = _query()(ORDER_EVENTS_SQL, {
            "order_id": oid, "symbol": sym, "t0": t0, "t1": t1,
            "lim": DETAIL_ROW_CAP + 1,
        }, timeout_ms=DETAIL_MS)
    except psycopg.errors.QueryCanceled as exc:
        raise TimeoutError(
            "MBO order lookup exceeded the time limit — the order's window is too wide"
        ) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise LookupError("databento.mbo is not installed") from exc
    capped = len(frame) > DETAIL_ROW_CAP
    if capped:
        frame = frame.iloc[:DETAIL_ROW_CAP]
    rows = event_rows(frame)
    out.update(row_cap=capped, n=len(rows), rows=rows)
    return out
