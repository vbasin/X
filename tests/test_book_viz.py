"""Border Visualizer — MBP-10 heatmap/snapshot API and packing."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.webui import book as bookviz
from src.webui import datasources as ds
from src.webui.app import create_app
from src.webui.book import N_LEVELS, _HEAT_KEYS, _SNAP_LEVELS
from src.webui.jobs import JobManager

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "SCOPING_ROOT", tmp_path / "scoping")
    monkeypatch.setattr(ds, "SEARCH_ROOT", tmp_path / "search")
    jm = JobManager(root=tmp_path / "jobs")
    with TestClient(create_app(jobs=jm, enable_live_worker=False), base_url="http://127.0.0.1") as c:
        yield c


def _heat_row(ts: datetime, bid0: float = 24750.0, ask0: float = 24750.25):
    vals = []
    for k in _HEAT_KEYS:
        if k.startswith("bid_px_"):
            lv = int(k[-2:])
            vals.append(bid0 - 0.25 * lv)
        elif k.startswith("ask_px_"):
            lv = int(k[-2:])
            vals.append(ask0 + 0.25 * lv)
        elif "_sz_" in k:
            vals.append(10 + int(k[-2:]))
        elif k == "action":
            vals.append("T")
        elif k == "side":
            vals.append("B")
        elif k == "price":
            vals.append(bid0)
        elif k == "size":
            vals.append(3)
        else:
            vals.append(None)
    return (ts, *vals)


def test_resolve_symbol_aliases():
    assert bookviz.resolve_symbol("nq") == "NQU6"
    assert bookviz.resolve_symbol("/NQ") == "NQU6"
    assert bookviz.resolve_symbol("NQU6") == "NQU6"
    assert bookviz.resolve_symbol("NQ.FUT") == "NQU6"
    with pytest.raises(ValueError):
        bookviz.resolve_symbol("bad symbol")
    with pytest.raises(ValueError):
        bookviz.resolve_symbol("")


def test_parse_ts_naive_is_utc():
    dt = bookviz.parse_ts("2026-08-20T13:30:00")
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timedelta(0)
    assert bookviz.parse_ts("2026-08-20T13:30:00Z") == dt
    with pytest.raises(ValueError):
        bookviz.parse_ts("not-a-time")


def test_auto_bucket_ms_snaps_to_nice():
    start = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    assert bookviz.auto_bucket_ms(start, start + timedelta(minutes=5)) == 100
    assert bookviz.auto_bucket_ms(start, start + timedelta(minutes=15)) == 500
    assert bookviz.auto_bucket_ms(start, start + timedelta(minutes=30)) >= 500


def test_clamp_span_rejects_wide_window():
    start = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="too wide"):
        bookviz._clamp_span(start, start + timedelta(hours=2))
    with pytest.raises(ValueError, match="after start"):
        bookviz._clamp_span(start, start)
    bookviz._clamp_span(start, start + timedelta(minutes=30))


def test_pack_heatmap_rows_parallel_arrays():
    ts = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
    packed = bookviz.pack_heatmap_rows([_heat_row(ts), _heat_row(ts + timedelta(milliseconds=100))])
    assert packed["n"] == 2
    assert len(packed["t_ms"]) == 2
    assert len(packed["bid_px"]) == 2 * N_LEVELS
    assert packed["bid_px"][0] == 24750.0
    assert packed["ask_px"][0] == 24750.25
    assert packed["bid_px"][9] == 24750.0 - 0.25 * 9
    assert packed["action"] == ["T", "T"]
    assert packed["size"] == [3, 3]


def test_format_snapshot_row_ladder():
    ts = datetime(2026, 8, 20, 13, 30, 1, tzinfo=timezone.utc)
    levels = []
    for i in range(N_LEVELS):
        levels.extend([
            24750.0 - 0.25 * i, 10 + i, 2,          # bid px,sz,ct
            24750.25 + 0.25 * i, 11 + i, 3,         # ask px,sz,ct
        ])
    assert len(_SNAP_LEVELS) == 60
    row = (
        ts, ts, 12345, "NQU6", "T", "B", 0,
        24750.0, 4, 128, 99, *levels,
    )
    out = bookviz.format_snapshot_row(row)
    assert out["found"] is True
    assert out["symbol"] == "NQU6"
    assert out["display"] == "NQ"
    assert out["bid"][0]["px"] == 24750.0
    assert out["ask"][0]["px"] == 24750.25
    assert out["spread"] == pytest.approx(0.25)
    assert out["mid"] == pytest.approx(24750.125)
    assert out["bid"][9]["sz"] == 19
    assert out["ask"][9]["ct"] == 3
    # bid sz = 10+11+...+19 = 145; ask sz = 11+12+...+20 = 155
    assert out["bid_count"] == 145
    assert out["ask_count"] == 155
    assert out["imbalance_diff"] == -10
    assert out["imbalance_log"] == pytest.approx(math.log(145) - math.log(155))


def test_heatmap_uses_time_bucket_and_alias(monkeypatch):
    captured: dict = {}

    def fake_query(sql, params=()):
        captured["sql"] = sql
        captured["params"] = params
        ts = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc)
        return [_heat_row(ts)]

    monkeypatch.setattr(bookviz, "_query", fake_query)
    out = bookviz.heatmap(
        "NQ", "2026-08-20T13:30:00Z", "2026-08-20T13:35:00Z", bucket_ms=100,
    )
    assert out["symbol"] == "NQU6"
    assert out["display"] == "NQ"
    assert out["n"] == 1
    assert out["bucket_ms"] == 100
    assert "time_bucket" in captured["sql"]
    assert "last(bid_px_00" in captured["sql"]
    assert captured["params"][-1] == "NQU6"
    assert captured["params"][0] == "100 milliseconds"


def test_heatmap_rejects_wide_without_query(monkeypatch):
    monkeypatch.setattr(bookviz, "_query", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no query")))
    with pytest.raises(ValueError, match="too wide"):
        bookviz.heatmap("NQU6", "2026-08-20T13:30:00Z", "2026-08-20T15:00:00Z")


def test_snapshot_lookback_and_miss(monkeypatch):
    calls = []

    def fake_query(sql, params=()):
        calls.append(params)
        return []

    monkeypatch.setattr(bookviz, "_query", fake_query)
    out = bookviz.snapshot("NQU6", "2026-08-20T13:30:00Z")
    assert out["found"] is False
    assert len(calls) == 2
    assert calls[0][0] == "NQU6"
    # first lookback 5s, then 30s
    assert (calls[0][2] - calls[0][1]).total_seconds() == pytest.approx(5)
    assert (calls[1][2] - calls[1][1]).total_seconds() == pytest.approx(30)


def test_book_routes_unwired(client):
    """Live Chart no longer serves the MBP-10 ladder (ADR-148)."""
    meta = client.get("/api/book/meta")
    heat = client.get("/api/book/heatmap", params={
        "symbol": "NQ",
        "start": "2026-08-20T13:30:00Z",
        "end": "2026-08-20T13:35:00Z",
    })
    snap = client.get("/api/book/snapshot", params={
        "symbol": "NQ", "ts": "2026-08-20T13:30:00Z",
    })
    assert meta.status_code == 404
    assert heat.status_code == 404
    assert snap.status_code == 404


def test_aggregate_imbalance_mean_over_1s_bucket():
    # 100 ms frames; 1s epoch bucket at t=1000 covers [1000, 2000)
    t = [800, 900, 1000, 1100, 1900, 2000]
    bid = [10, 10, 20, 30, 40, 99]
    ask = [10, 10, 10, 10, 10, 99]
    s = bookviz.aggregate_imbalance(t, bid, ask, at_ms=1500, period_ms=1000)
    assert s["n"] == 3  # 1000, 1100, 1900
    assert s["start_ms"] == 1000
    assert s["end_ms"] == 2000
    assert s["bid_count"] == pytest.approx((20 + 30 + 40) / 3)
    assert s["ask_count"] == pytest.approx(10)
    assert s["imbalance_diff"] == pytest.approx(s["bid_count"] - s["ask_count"])
    assert s["imbalance_log"] == pytest.approx(math.log(s["bid_count"]) - math.log(s["ask_count"]))
    pt = bookviz.aggregate_imbalance(t, bid, ask, at_ms=1150, period_ms=0)
    assert pt["n"] == 1 and pt["bid_count"] == 30  # last t <= 1150 is 1100


def test_imbalance_stats_sums_and_log():
    s = bookviz.imbalance_stats([10, 5, None], [{"sz": 8}, {"sz": 2}])
    assert s["bid_count"] == 15
    assert s["ask_count"] == 10
    assert s["imbalance_diff"] == 5
    assert s["imbalance_log"] == pytest.approx(math.log(15) - math.log(10))
    empty = bookviz.imbalance_stats([0, 0], [4])
    assert empty["bid_count"] == 0
    assert empty["imbalance_log"] is None
    assert empty["imbalance_diff"] == -4


def test_spa_declares_border_tab():
    html = (REPO / "src/webui/static/index.html").read_text()
    js = (REPO / "src/webui/static/app.js").read_text()
    assert 'data-nav="chart"' in html
    assert "Live Chart" in html
    assert "viewLiveChart" in js
    assert "viewBorderVisualizer" in js
    assert "#\\/border" in js or "#/border" in js
    assert "bvImbalanceStats" not in js
    assert 'id="bv-imb"' not in js
    assert "MBP-10 ladder" not in js
    assert 'id="lc-hover-show"' in js
