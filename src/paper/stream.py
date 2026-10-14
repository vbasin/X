"""Databento Live (and DB replay) market data feeds for paper execution.

Live path opens a **dedicated** Databento Live session (NQ.FUT tbbo only) so
fills are driven by the same print stream as real trading — not 5s parquet
closes. Replay path reads ``databento.tbbo`` from TimescaleDB in event order
for offline true-path tests.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.paper.engine import MarketTick

logger = logging.getLogger("vtech.paper.stream")

_PX_SCALE = 1e9  # FIXED_PRICE_SCALE fallback
_NQ_FUT_RE = re.compile(r"^NQ[FGHJKMNQUVXZ][0-9]$")
_CT = ZoneInfo("America/Chicago")


def _ns_to_dt(ns: int) -> datetime | None:
    if ns is None or ns == 0:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def _px(raw: int | float | None) -> float | None:
    if raw is None:
        return None
    try:
        from databento_dbn import FIXED_PRICE_SCALE
        scale = float(FIXED_PRICE_SCALE)
    except Exception:
        scale = _PX_SCALE
    v = float(raw) / scale
    # UNDF / bad prices often at extreme fixed-point values
    if v <= 0 or v > 1e7:
        return None
    return v


def tick_from_tbbo_msg(rec: Any, symbol: str = "") -> MarketTick | None:
    """Convert a Databento TBBO/MBP-1 record into a MarketTick."""
    price = _px(getattr(rec, "price", None))
    if price is None:
        # try pretty_price if available
        try:
            pp = float(rec.pretty_price)
            if pp > 0:
                price = pp
        except Exception:
            return None
    if price is None:
        return None
    ts = _ns_to_dt(int(getattr(rec, "ts_event", 0) or 0))
    if ts is None:
        ts = datetime.now(timezone.utc)
    bid = ask = None
    levels = getattr(rec, "levels", None)
    if levels:
        lvl = levels[0]
        bid = _px(getattr(lvl, "bid_px", None))
        ask = _px(getattr(lvl, "ask_px", None))
    return MarketTick(
        ts=ts,
        price=float(price),
        size=int(getattr(rec, "size", 0) or 0),
        bid=bid,
        ask=ask,
        symbol=symbol or "",
        instrument_id=int(getattr(rec, "instrument_id", 0) or 0) or None,
    )


def run_databento_live(
    on_tick: Callable[[MarketTick], None],
    *,
    dataset: str | None = None,
    symbols: str = "NQ.FUT",
    stype_in: str = "parent",
    primary_only: bool = True,
    stop_flag: Callable[[], bool] | None = None,
) -> None:
    """Block: subscribe to Databento Live tbbo and invoke *on_tick* per print.

    Requires ``DATABENTO_API_KEY`` (same as ``vtech-live-streamer``).
    """
    import databento as db
    from src.acquisition.schemas import DATASET

    ds = dataset or DATASET
    client = db.Live(
        reconnect_policy="reconnect",
        heartbeat_interval_s=15,
        slow_reader_behavior="warn",
    )
    client.subscribe(dataset=ds, schema="tbbo", symbols=symbols, stype_in=stype_in)

    sym_cache: dict[int, str] = {}

    def _resolve(iid: int) -> str:
        if iid in sym_cache:
            return sym_cache[iid]
        live_sym = client.symbology_map.get(iid, "")
        if live_sym:
            sym_cache[iid] = str(live_sym)
            return str(live_sym)
        return ""

    def _primary_today() -> str | None:
        try:
            from src.common.contract_calendar import active_contract
            return active_contract("NQ", datetime.now(_CT).date())
        except Exception:
            return None

    primary = _primary_today() if primary_only else None
    logger.info(
        "paper_live_subscribe dataset=%s symbols=%s primary=%s",
        ds, symbols, primary,
    )

    def _cb(record: db.DBNRecord) -> None:
        if stop_flag and stop_flag():
            return
        # TBBO / MBP1 trade prints
        rtype = type(record).__name__
        if rtype not in ("TBBOMsg", "MBP1Msg") and not hasattr(record, "price"):
            return
        if not hasattr(record, "price") or not hasattr(record, "instrument_id"):
            return
        # skip pure book updates without a trade print (size 0 and no price move)
        try:
            sz = int(getattr(record, "size", 0) or 0)
        except Exception:
            sz = 0
        # Still accept size>=0 trades; TBBO always has a price at the trade
        sym = _resolve(int(record.instrument_id))
        if primary_only and primary and sym and not sym.startswith(primary[:2]):
            # allow NQH6 etc. matching primary root month
            if sym != primary and not (primary and sym == primary):
                # strict: only lead month symbol
                if _NQ_FUT_RE.match(sym) and primary and sym != primary:
                    return
        if primary_only and primary and _NQ_FUT_RE.match(sym or "") and sym != primary:
            return
        tick = tick_from_tbbo_msg(record, symbol=sym)
        if tick is None:
            return
        if sz <= 0 and tick.price <= 0:
            return
        on_tick(tick)

    def _err(exc: Exception) -> None:
        logger.exception("paper_live_callback_error: %s", exc)

    client.add_callback(record_callback=_cb, exception_callback=_err)
    client.start()
    try:
        while True:
            if stop_flag and stop_flag():
                break
            time.sleep(0.5)
            # block_for_close would hang forever; poll is_connected if available
            if hasattr(client, "is_connected") and not client.is_connected():
                logger.warning("paper_live_disconnected")
                time.sleep(2)
    finally:
        try:
            client.stop()
        except Exception:
            pass


def iter_tbbo_from_db(
    day: str | date,
    *,
    symbol: str | None = None,
    limit: int | None = None,
) -> Iterator[MarketTick]:
    """Yield TBBO trade prints for *day* from TimescaleDB (event order).

    Used for offline true-path paper tests without a live market session.
    """
    import psycopg
    from src.common.db import get_dsn
    from src.common.contract_calendar import active_contract

    d = date.fromisoformat(str(day)[:10]) if not isinstance(day, date) else day
    primary = symbol or active_contract("NQ", d)
    # session window: CT calendar day in UTC bounds (wide — filter by CT date in SQL)
    sql = """
        SELECT ts_event, price, size, bid_px, ask_px, symbol, instrument_id
        FROM databento.tbbo
        WHERE ts_event >= %s::date
          AND ts_event < (%s::date + interval '1 day')
          AND symbol = %s
          AND price IS NOT NULL
          AND price > 0
        ORDER BY ts_event ASC
    """
    params: list[Any] = [d.isoformat(), d.isoformat(), primary]
    if limit:
        sql += " LIMIT %s"
        params.append(int(limit))

    with psycopg.connect(get_dsn()) as conn:
        with conn.cursor(name="paper_tbbo_replay") as cur:
            cur.itersize = 5000
            cur.execute(sql, params)
            for row in cur:
                ts_event, price, size, bid, ask, sym, iid = row
                if ts_event is None or price is None:
                    continue
                if ts_event.tzinfo is None:
                    ts_event = ts_event.replace(tzinfo=timezone.utc)
                yield MarketTick(
                    ts=ts_event,
                    price=float(price),
                    size=int(size or 0),
                    bid=float(bid) if bid is not None else None,
                    ask=float(ask) if ask is not None else None,
                    symbol=str(sym or primary),
                    instrument_id=int(iid) if iid is not None else None,
                )
