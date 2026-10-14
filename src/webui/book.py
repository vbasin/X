"""Historic MBP-10 order-book reader for the Border Visualizer.

Reads the posterior TimescaleDB hypertable ``databento.mbp_10`` (ingested
Databento L2). Each row already carries the full top-10 book after the event,
so the visualizer does not reconstruct from deltas.

Windows are hard-capped: a full RTH scan is 13–19M compressed rows and is too
slow for the console. Downsample with ``time_bucket`` + ``last()`` (measured
~0.8s for 5 minutes × 10 levels on compressed chunks).
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator
from zoneinfo import ZoneInfo

N_LEVELS = 10
TICK = 0.25
MAX_SPAN_SECONDS = 30 * 60
MAX_BUCKETS = 3600
TARGET_BUCKETS = 3000
STMT_TIMEOUT_MS = 20_000
SNAPSHOT_LOOKBACK = timedelta(seconds=5)
SNAPSHOT_LOOKBACK_WIDE = timedelta(seconds=30)
CT = ZoneInfo("America/Chicago")

# Console aliases → raw CME symbol stored in databento.mbp_10.
_SYMBOL_ALIASES = {
    "NQ": "NQU6",
    "/NQ": "NQU6",
    "NQ.FUT": "NQU6",
    "NQU6": "NQU6",
}

_SYMBOL_RE_OK = re.compile(r"^[A-Za-z0-9./_-]{1,16}$")

_NICE_BUCKET_MS = (
    10, 20, 25, 50, 100, 200, 250, 500,
    1_000, 2_000, 2_500, 5_000, 10_000, 15_000, 30_000, 60_000,
)

_HEAT_KEYS: list[str] = []
_HEAT_LAST_SQL: list[str] = []
for _i in range(N_LEVELS):
    for _side in ("bid", "ask"):
        for _kind in ("px", "sz"):
            _col = f"{_side}_{_kind}_{_i:02d}"
            _HEAT_KEYS.append(_col)
            _HEAT_LAST_SQL.append(f"last({_col}, ts_recv) AS {_col}")
for _col in ("action", "side", "price", "size"):
    _HEAT_KEYS.append(_col)
    _HEAT_LAST_SQL.append(f"last({_col}, ts_recv) AS {_col}")

_SNAP_LEVELS: list[str] = []
for _i in range(N_LEVELS):
    for _side in ("bid", "ask"):
        for _kind in ("px", "sz", "ct"):
            _SNAP_LEVELS.append(f"{_side}_{_kind}_{_i:02d}")


def parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    s = (value or "").strip()
    if not s:
        raise ValueError("timestamp is required")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s or not _SYMBOL_RE_OK.match(s):
        raise ValueError("invalid symbol")
    if s in _SYMBOL_ALIASES:
        return _SYMBOL_ALIASES[s]
    return _SYMBOL_ALIASES.get(s.lstrip("/"), s.lstrip("/"))


def auto_bucket_ms(start: datetime, end: datetime, target: int = TARGET_BUCKETS) -> int:
    span_ms = max(1.0, (end - start).total_seconds() * 1000.0)
    raw = span_ms / max(1, target)
    for nice in _NICE_BUCKET_MS:
        if nice >= raw:
            return int(nice)
    return int(_NICE_BUCKET_MS[-1])


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp_span(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if end <= start:
        raise ValueError("end must be after start")
    span = (end - start).total_seconds()
    if span > MAX_SPAN_SECONDS:
        raise ValueError(
            f"window too wide ({span:.0f}s); max is {MAX_SPAN_SECONDS // 60} minutes "
            "(MBP-10 is 13–19M rows/day — zoom in, then scrub)"
        )
    if span < 0.05:
        raise ValueError("window too short")
    return start, end


@contextmanager
def _connect() -> Iterator[Any]:
    import psycopg
    from src.common.db import get_dsn

    with psycopg.connect(
        get_dsn(),
        connect_timeout=8,
        options=f"-c statement_timeout={STMT_TIMEOUT_MS}",
    ) as conn:
        yield conn


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    import psycopg

    try:
        with _connect() as conn:
            return conn.execute(sql, params).fetchall()
    except psycopg.errors.QueryCanceled as exc:
        raise TimeoutError(
            "MBP-10 query exceeded 20s — pick a shorter window"
        ) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise LookupError("databento.mbp_10 is not installed") from exc


def _display_symbol(symbol: str) -> str:
    if symbol == "NQU6":
        return "NQ"
    if symbol.endswith(("H", "M", "U", "Z")) and len(symbol) >= 4:
        return symbol[:2] if symbol[:2].isalpha() else symbol
    return symbol


def meta() -> dict[str, Any]:
    """Chunk-backed coverage for the visualizer. Never COUNT(*) the hypertable."""
    try:
        chunks = _query(
            "SELECT range_start, range_end, is_compressed "
            "FROM timescaledb_information.chunks "
            "WHERE hypertable_schema = 'databento' AND hypertable_name = 'mbp_10' "
            "ORDER BY range_start"
        )
    except Exception as exc:  # noqa: BLE001 — console degrades when DB is down
        return {"available": False, "error": str(exc), "table": "databento.mbp_10"}
    if not chunks:
        return {
            "available": False,
            "error": "databento.mbp_10 has no chunks",
            "table": "databento.mbp_10",
        }
    range_start = chunks[0][0]
    range_end = chunks[-1][1]
    dates = []
    seen: set[str] = set()
    for rs, _re, _comp in chunks:
        d = rs.astimezone(timezone.utc).date().isoformat()
        if d not in seen:
            seen.add(d)
            dates.append(d)
    last_end = chunks[-1][1]
    probe_end = last_end
    probe_start = last_end - timedelta(minutes=2)
    try:
        sym_rows = _query(
            "SELECT DISTINCT symbol FROM databento.mbp_10 "
            "WHERE ts_recv >= %s AND ts_recv < %s",
            (probe_start, probe_end),
        )
        if not sym_rows:
            first_start = chunks[-1][0]
            sym_rows = _query(
                "SELECT DISTINCT symbol FROM databento.mbp_10 "
                "WHERE ts_recv >= %s AND ts_recv < %s",
                (first_start, first_start + timedelta(minutes=2)),
            )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc), "table": "databento.mbp_10"}
    symbols = []
    for (sym,) in sym_rows:
        if not sym:
            continue
        symbols.append({
            "symbol": sym,
            "display": _display_symbol(sym),
            "label": f"{_display_symbol(sym)} ({sym})",
        })
    if not symbols:
        symbols = [{"symbol": "NQU6", "display": "NQ", "label": "NQ (NQU6)"}]
    last_avail = (range_end - timedelta(microseconds=1)).astimezone(CT)
    default_day = last_avail.date()
    default_start = datetime(
        default_day.year, default_day.month, default_day.day, 8, 30, tzinfo=CT
    ).astimezone(timezone.utc)
    default_end = default_start + timedelta(minutes=5)
    lo = range_start if isinstance(range_start, datetime) else default_start
    hi = range_end if isinstance(range_end, datetime) else default_end
    if default_end > hi:
        default_end = hi
        default_start = default_end - timedelta(minutes=5)
    if default_start < lo:
        default_start = lo
        default_end = min(hi, default_start + timedelta(minutes=5))
    return {
        "available": True,
        "table": "databento.mbp_10",
        "schema": "mbp-10",
        "n_chunks": len(chunks),
        "compressed_chunks": sum(1 for r in chunks if r[2]),
        "range_start": _iso(range_start),
        "range_end": _iso(range_end - timedelta(microseconds=1)),
        "dates": dates,
        "symbols": symbols,
        "tick": TICK,
        "levels": N_LEVELS,
        "max_span_seconds": MAX_SPAN_SECONDS,
        "default": {
            "symbol": symbols[0]["symbol"],
            "start": _iso(default_start),
            "end": _iso(default_end),
            "bucket_ms": auto_bucket_ms(default_start, default_end),
        },
    }


def heatmap(
    symbol: str,
    start: str,
    end: str,
    bucket_ms: int | None = None,
) -> dict[str, Any]:
    """Downsampled 10-level book snapshots over ``[start, end)``."""
    raw_symbol = resolve_symbol(symbol)
    t0 = parse_ts(start)
    t1 = parse_ts(end)
    t0, t1 = _clamp_span(t0, t1)
    span_ms = (t1 - t0).total_seconds() * 1000.0
    if bucket_ms is None:
        bms = auto_bucket_ms(t0, t1)
    else:
        try:
            bms = int(bucket_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("bucket_ms must be an integer") from exc
        if bms < 10 or bms > 60_000:
            raise ValueError("bucket_ms must be between 10 and 60000")
    n_est = span_ms / bms
    if n_est > MAX_BUCKETS:
        bms = auto_bucket_ms(t0, t1, target=MAX_BUCKETS)
    interval = f"{bms} milliseconds"
    select_list = ",\n       ".join(_HEAT_LAST_SQL)
    sql = (
        "SELECT time_bucket(%s::interval, ts_recv) AS bucket,\n"
        f"       {select_list}\n"
        "FROM databento.mbp_10\n"
        "WHERE ts_recv >= %s AND ts_recv < %s\n"
        "  AND symbol = %s\n"
        "GROUP BY 1\n"
        "ORDER BY 1"
    )
    rows = _query(sql, (interval, t0, t1, raw_symbol))
    packed = pack_heatmap_rows(rows)
    packed.update({
        "symbol": raw_symbol,
        "display": _display_symbol(raw_symbol),
        "start": _iso(t0),
        "end": _iso(t1),
        "bucket_ms": bms,
        "levels": N_LEVELS,
        "tick": TICK,
        "n": packed["n"],
    })
    return packed


def pack_heatmap_rows(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Pack ``time_bucket`` + last() rows into parallel arrays for the canvas."""
    n = len(rows)
    n_lv = N_LEVELS
    t_ms: list[int] = []
    bid_px: list[float | None] = [None] * (n * n_lv)
    ask_px: list[float | None] = [None] * (n * n_lv)
    bid_sz: list[int | None] = [None] * (n * n_lv)
    ask_sz: list[int | None] = [None] * (n * n_lv)
    action: list[str | None] = [None] * n
    side: list[str | None] = [None] * n
    price: list[float | None] = [None] * n
    size: list[int | None] = [None] * n
    key_i = {k: i for i, k in enumerate(_HEAT_KEYS)}
    for i, row in enumerate(rows):
        bucket = row[0]
        vals = row[1:]
        if bucket is None:
            t_ms.append(0)
        elif isinstance(bucket, datetime):
            t_ms.append(int(bucket.timestamp() * 1000))
        else:
            t_ms.append(int(bucket))
        base = i * n_lv
        for lv in range(n_lv):
            bp = vals[key_i[f"bid_px_{lv:02d}"]]
            ap = vals[key_i[f"ask_px_{lv:02d}"]]
            bs = vals[key_i[f"bid_sz_{lv:02d}"]]
            az = vals[key_i[f"ask_sz_{lv:02d}"]]
            bid_px[base + lv] = float(bp) if bp is not None else None
            ask_px[base + lv] = float(ap) if ap is not None else None
            bid_sz[base + lv] = int(bs) if bs is not None else None
            ask_sz[base + lv] = int(az) if az is not None else None
        act = vals[key_i["action"]]
        sd = vals[key_i["side"]]
        pr = vals[key_i["price"]]
        sz = vals[key_i["size"]]
        action[i] = (str(act).strip() or None) if act is not None else None
        side[i] = (str(sd).strip() or None) if sd is not None else None
        price[i] = float(pr) if pr is not None else None
        size[i] = int(sz) if sz is not None else None
    return {
        "n": n,
        "t_ms": t_ms,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "bid_sz": bid_sz,
        "ask_sz": ask_sz,
        "action": action,
        "side": side,
        "price": price,
        "size": size,
    }


def snapshot(symbol: str, ts: str) -> dict[str, Any]:
    """Last full book at or before ``ts`` (bounded lookback, partition-friendly)."""
    raw_symbol = resolve_symbol(symbol)
    at = parse_ts(ts)
    row = _snapshot_row(raw_symbol, at, SNAPSHOT_LOOKBACK)
    if row is None:
        row = _snapshot_row(raw_symbol, at, SNAPSHOT_LOOKBACK_WIDE)
    if row is None:
        return {
            "symbol": raw_symbol,
            "display": _display_symbol(raw_symbol),
            "ts": _iso(at),
            "found": False,
        }
    return format_snapshot_row(row, raw_symbol)


def _snapshot_row(symbol: str, at: datetime, lookback: timedelta) -> tuple[Any, ...] | None:
    cols = (
        "ts_recv, ts_event, instrument_id, symbol, action, side, depth, "
        "price, size, flags, sequence, " + ", ".join(_SNAP_LEVELS)
    )
    sql = (
        f"SELECT {cols} FROM databento.mbp_10 "
        "WHERE symbol = %s AND ts_recv > %s AND ts_recv <= %s "
        "ORDER BY ts_recv DESC, sequence DESC LIMIT 1"
    )
    rows = _query(sql, (symbol, at - lookback, at))
    return rows[0] if rows else None


def imbalance_stats(bid: list[Any], ask: list[Any]) -> dict[str, Any]:
    """Depth-sum imbalance at one book snapshot (10 levels).

    ``bid_count`` / ``ask_count`` are the sum of resting size (``sz``), not
    order counts. ``imbalance_log`` is ``ln(bid) − ln(ask)`` and is ``None``
    when either side is empty (log of 0 is undefined).
    """
    def _sum(side: list[Any]) -> int:
        total = 0
        for x in side or []:
            v = x.get("sz") if isinstance(x, dict) else x
            if v is not None:
                total += int(v)
        return total

    bid_count = _sum(bid)
    ask_count = _sum(ask)
    logv = None
    if bid_count > 0 and ask_count > 0:
        logv = math.log(bid_count) - math.log(ask_count)
    return {
        "bid_count": bid_count,
        "ask_count": ask_count,
        "imbalance_diff": bid_count - ask_count,
        "imbalance_log": logv,
    }


def aggregate_imbalance(
    t_ms: list[int],
    bid_counts: list[float],
    ask_counts: list[float],
    at_ms: int,
    period_ms: int,
) -> dict[str, Any]:
    """Mean per-snapshot depth in the epoch-aligned bucket containing ``at_ms``.

    ``period_ms <= 0`` is point mode: the last sample with ``t <= at_ms``.
    ``bid_count`` / ``ask_count`` on the result are means of the per-snapshot
    L0–L9 size sums. Diff and log use those means.
    """
    n_in = min(len(t_ms), len(bid_counts), len(ask_counts))
    if n_in == 0:
        return {
            "bid_count": 0, "ask_count": 0, "imbalance_diff": 0,
            "imbalance_log": None, "n": 0, "period_ms": period_ms,
            "start_ms": at_ms, "end_ms": at_ms,
        }
    if period_ms <= 0:
        start = end = at_ms
        pick = 0
        for i in range(n_in):
            if t_ms[i] <= at_ms:
                pick = i
        bid_m = float(bid_counts[pick] or 0)
        ask_m = float(ask_counts[pick] or 0)
        n = 1
    else:
        start = (int(at_ms) // int(period_ms)) * int(period_ms)
        end = start + int(period_ms)
        bid_sum = 0.0
        ask_sum = 0.0
        n = 0
        for i in range(n_in):
            t = t_ms[i]
            if t < start:
                continue
            if t >= end:
                break
            bid_sum += float(bid_counts[i] or 0)
            ask_sum += float(ask_counts[i] or 0)
            n += 1
        bid_m = bid_sum / n if n else 0.0
        ask_m = ask_sum / n if n else 0.0
    logv = None
    if bid_m > 0 and ask_m > 0:
        logv = math.log(bid_m) - math.log(ask_m)
    return {
        "bid_count": bid_m,
        "ask_count": ask_m,
        "imbalance_diff": bid_m - ask_m,
        "imbalance_log": logv,
        "n": n,
        "period_ms": period_ms,
        "start_ms": start,
        "end_ms": end,
    }


def format_snapshot_row(row: tuple[Any, ...], symbol: str | None = None) -> dict[str, Any]:
    """Turn a snapshot SELECT row into the ladder JSON the UI renders."""
    ts_recv, ts_event, instrument_id, sym, action, side, depth, price, size, flags, sequence = row[:11]
    levels = row[11:]
    bid: list[dict[str, Any]] = []
    ask: list[dict[str, Any]] = []
    for lv in range(N_LEVELS):
        off = lv * 6
        bid.append({
            "px": _f(levels[off + 0]),
            "sz": _i(levels[off + 1]),
            "ct": _i(levels[off + 2]),
        })
        ask.append({
            "px": _f(levels[off + 3]),
            "sz": _i(levels[off + 4]),
            "ct": _i(levels[off + 5]),
        })
    bid0, ask0 = bid[0].get("px"), ask[0].get("px")
    mid = (bid0 + ask0) / 2.0 if bid0 is not None and ask0 is not None else None
    spread = (ask0 - bid0) if bid0 is not None and ask0 is not None else None
    bsz, asz = bid[0].get("sz") or 0, ask[0].get("sz") or 0
    denom = bsz + asz
    imbalance = ((bsz - asz) / denom) if denom else None
    raw = symbol or (sym or "")
    return {
        "found": True,
        "symbol": raw,
        "display": _display_symbol(raw),
        "ts_recv": _iso(ts_recv),
        "ts_event": _iso(ts_event),
        "instrument_id": instrument_id,
        "action": (str(action).strip() if action is not None else None),
        "side": (str(side).strip() if side is not None else None),
        "depth": depth,
        "price": _f(price),
        "size": _i(size),
        "flags": flags,
        "sequence": sequence,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "imbalance": imbalance,
        **imbalance_stats(bid, ask),
        "tick": TICK,
        "levels": N_LEVELS,
    }


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


def _i(v: Any) -> int | None:
    return int(v) if v is not None else None
