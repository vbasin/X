"""Tests for src/webui/mbo_orders_api.py — pure payload assembly, no database."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.webui.mbo_orders_api import assemble, parse_ts

T0 = pd.Timestamp("2026-09-24 13:30:00", tz="UTC")
T_END = (T0 + pd.Timedelta(hours=1)).to_pydatetime()
KW = dict(min_size=20, approach=10.0, chain_n=3, chain_dt=2.0, chain_min_secs=10.0,
          chain_min_abs=50, break_pts=1.0)


def _ev(rows):
    return pd.DataFrame(
        [(T0 + pd.Timedelta(seconds=r[0]), r[1], r[2], r[3], r[4], r[5], r[6], False) for r in rows],
        columns=["ts_recv", "sequence", "order_id", "action", "side", "size", "price", "snap"])


def _candles(n=5, lo=90.0, hi=110.0):
    t = pd.date_range(T0, periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"t": t, "o": (lo + hi) / 2, "h": hi, "l": lo, "c": (lo + hi) / 2, "v": 1})


def test_payload_is_json_and_filters_by_size():
    ev = _ev([(0, 1, 1, "A", "B", 30, 100.0), (5, 2, 1, "F", "B", 30, 100.0), (5, 3, 1, "C", "B", 30, 100.0),
              (0, 4, 2, "A", "A", 8, 105.0), (9, 5, 2, "C", "A", 8, 105.0)])
    out = assemble(ev, _candles(), T_END, **KW)
    json.dumps(out)                                        # no NaN / numpy scalars
    assert [o["i"] for o in out["orders"]] == [1]          # the 8-lot order is below min_size
    o = out["orders"][0]
    assert o["s"] == "B" and o["z"] == 30 and o["e"] == "filled" and o["o"] == "FILLED"
    assert o["b"] - o["a"] == pytest.approx(5.0)
    assert out["summary"]["outcomes"] == {"FILLED": 1}


def test_min_size_is_a_display_filter_only_small_orders_still_feed_chains():
    rows, t, seq = [], 0.0, 10
    for i in range(5):                                     # 5 x 10-lot reload chain, below min_size 20
        oid = 100 + i
        rows += [(t, seq, oid, "A", "B", 10, 100.0), (t + 8, seq + 1, oid, "F", "B", 10, 100.0),
                 (t + 8, seq + 2, oid, "C", "B", 10, 100.0)]
        t += 8.5
        seq += 3
    out = assemble(_ev(rows), _candles(), T_END, **KW)
    assert out["orders"] == []                             # nothing >= 20 lots to draw
    assert len(out["chains"]) == 1 and out["chains"][0]["n"] == 5 and out["chains"][0]["abs"] == 50


def test_empty_events_give_empty_payload():
    out = assemble(_ev([]), _candles(), T_END, **KW)
    assert out["orders"] == [] and out["chains"] == [] and out["summary"]["n_orders"] == 0


def test_missing_candles_still_returns_orders_without_outcomes():
    ev = _ev([(0, 1, 1, "A", "B", 30, 100.0), (5, 2, 1, "C", "B", 30, 100.0)])
    out = assemble(ev, _candles().iloc[0:0], T_END, **KW)
    assert out["orders"][0]["o"] is None and out["orders"][0]["d"] is None
    json.dumps(out)


def test_cap_span_keeps_newest_five_days():
    from datetime import timedelta

    from src.webui.mbo_orders_api import MAX_SPAN_H, cap_span

    assert MAX_SPAN_H == 5 * 24
    t1 = datetime(2026, 9, 30, tzinfo=timezone.utc)
    t0 = t1 - timedelta(days=10)
    start, truncated = cap_span(t0, t1)
    assert truncated is True
    assert start == t1 - timedelta(hours=120)
    short = t1 - timedelta(days=2)
    start2, truncated2 = cap_span(short, t1)
    assert truncated2 is False and start2 == short


def test_parse_ts_accepts_iso_and_epoch():
    a = parse_ts("2026-09-24T13:30:00Z")
    assert a == datetime(2026, 9, 24, 13, 30, tzinfo=timezone.utc)
    assert parse_ts(str(a.timestamp())) == a
    assert parse_ts("2026-09-24T13:30:00") == a            # naive = UTC
    with pytest.raises(ValueError):
        parse_ts("tomorrow-ish")


def test_an_iceberg_is_kept_even_though_its_displayed_size_is_below_min_size():
    """ADR-149: filtering on displayed size is exactly what hid these orders. A 1-lot display that
    absorbed 60 must survive a min_size=20 filter and be labelled with what it really absorbed."""
    import pandas as pd
    from src.webui.mbo_orders_api import assemble
    t0 = pd.Timestamp("2026-09-24 13:30:00", tz="UTC")
    seg = pd.DataFrame([{
        "order_id": 7, "side": "A", "price": 29300.0, "size0": 1, "t_start": t0,
        "t_end": t0 + pd.Timedelta(minutes=5), "pre": False, "end": "cancelled", "filled": 60,
        "disp": 1,      # never showed more than 1 lot, yet took 60 -> hidden reserve
    }])
    out = assemble(pd.DataFrame(), pd.DataFrame(), t0 + pd.Timedelta(hours=1), min_size=20,
                   approach=10.0, chain_n=3, chain_dt=2, chain_min_secs=10, chain_min_abs=50,
                   break_pts=1, seg=seg)
    assert len(out["orders"]) == 1, "a 1-lot display must not be filtered out by min_size=20"
    o = out["orders"][0]
    assert o["ice"] is True and o["ia"] == 60 and o["idisp"] == 1
    assert out["summary"]["n_icebergs"] == 1 and out["summary"]["ice_absorbed"] == 60


def test_order_event_window_pads_short_life_and_caps_at_five_days():
    from datetime import timedelta

    from src.webui.mbo_orders_api import ORDER_EVENTS_SQL, order_event_window

    lo = datetime(2026, 8, 23, tzinfo=timezone.utc)
    hi = datetime(2026, 9, 30, tzinfo=timezone.utc)
    a = datetime(2026, 9, 29, 20, 0, tzinfo=timezone.utc)
    b = datetime(2026, 9, 29, 20, 10, tzinfo=timezone.utc)
    t0, t1, truncated = order_event_window(a, b, coverage_lo=lo, coverage_hi=hi)
    assert truncated is False
    assert t0 == a - timedelta(seconds=2)
    assert t1 == b + timedelta(seconds=2)

    long_a = hi - timedelta(days=12)
    t0, t1, truncated = order_event_window(
        long_a, hi - timedelta(seconds=1), coverage_lo=lo, coverage_hi=hi)
    assert truncated is True
    assert t1 - t0 == timedelta(hours=120)
    assert t1 <= hi

    t0, t1, truncated = order_event_window(
        hi + timedelta(days=1), hi + timedelta(days=2), coverage_lo=lo, coverage_hi=hi)
    assert truncated is False and t1 <= t0

    sql = " ".join(ORDER_EVENTS_SQL.split())
    assert "WHERE order_id = %(order_id)s" in sql
    assert "ORDER BY ts_event, sequence" in sql
    assert "ts_recv >= %(t0)s" in sql


def test_event_rows_keep_order_id_and_microseconds():
    from src.webui.mbo_orders_api import event_rows

    df = pd.DataFrame([{
        "ts_recv": pd.Timestamp("2026-09-29 20:00:00.123456", tz="UTC"),
        "ts_event": pd.Timestamp("2026-09-29 20:00:00.123000", tz="UTC"),
        "rtype": 160, "publisher_id": 1, "instrument_id": 12345,
        "action": "F", "side": "B", "price": 25100.25, "size": 3,
        "channel_id": 4, "order_id": 6882403938353, "flags": 128,
        "ts_in_delta": 7000, "sequence": 42, "symbol": "NQZ6", "dataset": "GLBX.MDP3",
    }])
    rows = event_rows(df)
    assert rows[0]["order_id"] == "6882403938353"
    assert rows[0]["ts_event"].startswith("2026-09-29T20:00:00.123000")
    assert rows[0]["price"] == 25100.25
    assert rows[0]["sequence"] == 42
    assert rows[0]["action"] == "F" and rows[0]["flags"] == 128
    assert event_rows(df.iloc[0:0]) == []


def test_order_map_statement_timeout_is_timeouterror(monkeypatch):
    import psycopg

    from src.webui import mbo_orders_api as api

    monkeypatch.setattr(api, "coverage", lambda: {
        "min": datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp(),
        "max": datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp(),
    })

    def _canceled(*_a, **_k):
        raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(api, "_pick_symbol", _canceled)
    with pytest.raises(TimeoutError, match="time limit"):
        api.order_map("2026-09-29T14:00:00+00:00", "2026-09-29T16:00:00+00:00")


def test_a_plain_small_order_is_still_filtered_out_by_min_size():
    import pandas as pd
    from src.webui.mbo_orders_api import assemble
    t0 = pd.Timestamp("2026-09-24 13:30:00", tz="UTC")
    seg = pd.DataFrame([{
        "order_id": 8, "side": "A", "price": 29300.0, "size0": 1, "t_start": t0,
        "t_end": t0 + pd.Timedelta(minutes=5), "pre": False, "end": "cancelled", "filled": 1,
        "disp": 1,      # took exactly what it showed -> ordinary order
    }])
    out = assemble(pd.DataFrame(), pd.DataFrame(), t0 + pd.Timedelta(hours=1), min_size=20,
                   approach=10.0, chain_n=3, chain_dt=2, chain_min_secs=10, chain_min_abs=50,
                   break_pts=1, seg=seg)
    assert out["orders"] == [] and out["summary"]["n_icebergs"] == 0


def test_mbo_sql_filters_ts_event_for_compressed_orderby():
    from src.webui.mbo_orders import CANDLE_SQL, ICEBERG_SQL, ORDERS_SQL

    for sql in (ORDERS_SQL, ICEBERG_SQL, CANDLE_SQL):
        assert "ts_recv >= %(t0)s" in sql
        assert "ts_event >= %(t0)s" in sql
        assert "ts_event < %(t1)s" in sql
        assert "instrument_id = %(instrument_id)s" in sql


def test_order_map_busy_when_lock_held(monkeypatch):
    from src.webui import mbo_orders_api as api

    monkeypatch.setattr(api, "coverage", lambda: {
        "min": datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp(),
        "max": datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp(),
    })
    assert api._lock.acquire(blocking=False)
    try:
        with pytest.raises(api.OrderMapBusy, match="still computing"):
            api.order_map(
                "2026-09-29T14:00:00+00:00", "2026-09-29T16:00:00+00:00", symbol="NQU6")
    finally:
        api._lock.release()


def test_iceberg_timeout_is_softer_than_orders():
    from src.webui.mbo_orders_api import ICE_ID_CAP, ICEBERG_MS, ICEBERG_SPAN_H, QUERY_MS

    assert ICEBERG_MS < QUERY_MS
    assert QUERY_MS <= 90_000
    assert ICE_ID_CAP == 200
    assert ICEBERG_SPAN_H == 36
