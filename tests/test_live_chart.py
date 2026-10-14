"""Live Chart — TBBO candles + overlay catalog (ADR-087 / ADR-097)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.webui import datasources as ds
from src.webui import live_chart as lc
from src.webui.app import create_app
from src.webui.jobs import JobManager

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "SCOPING_ROOT", tmp_path / "scoping")
    monkeypatch.setattr(ds, "SEARCH_ROOT", tmp_path / "search")
    jm = JobManager(root=tmp_path / "jobs")
    with TestClient(create_app(jobs=jm, enable_live_worker=False), base_url="http://127.0.0.1") as c:
        yield c


def test_coerce_tf_aliases():
    assert lc.coerce_tf("1m") == "1min"
    assert lc.coerce_tf("15s") == "15s"
    assert lc.coerce_tf("1h") == "1hour"
    with pytest.raises(ValueError):
        lc.coerce_tf("3min")


def test_parse_print_min():
    assert lc.parse_print_min(None) == 10
    assert lc.parse_print_min("10") == 10
    assert lc.parse_print_min(25) == 25
    assert lc.parse_print_min("100000") == lc.PRINT_MIN_MAX
    with pytest.raises(ValueError, match=">= 1"):
        lc.parse_print_min(0)
    with pytest.raises(ValueError, match="integer"):
        lc.parse_print_min("nope")


def test_parse_overlay_keys():
    assert lc.parse_overlay_keys("of_cvd,vw_vwap") == ["of_cvd", "vw_vwap"]
    assert lc.parse_overlay_keys("") == []
    with pytest.raises(ValueError, match="unknown overlay"):
        lc.parse_overlay_keys("not_a_feature")
    with pytest.raises(ValueError, match="at most 8"):
        lc.parse_overlay_keys(",".join(o["key"] for o in lc.OVERLAYS))


def test_parse_emas_and_lookback():
    assert lc.parse_emas(None) == []
    assert lc.parse_emas("") == []
    specs = lc.parse_emas(
        '[{"id":"a","n":1,"lookback":9},{"id":"b","n":2,"lookback":21}]'
    )
    assert [s["lookback"] for s in specs] == [9, 21]
    assert [s["key"] for s in specs] == ["ema_1", "ema_2"]
    assert specs[0]["label"] == "EMA1 (9)"
    assert lc.parse_ema_lookback(None) == lc.EMA_LOOKBACK_DEFAULT
    with pytest.raises(ValueError, match="between"):
        lc.parse_ema_lookback(0)
    with pytest.raises(ValueError, match="at most"):
        lc.parse_emas("[" + ",".join('{"lookback":9}' for _ in range(9)) + "]")
    with pytest.raises(ValueError, match="duplicated"):
        lc.parse_emas('[{"n":1,"lookback":9},{"n":1,"lookback":21}]')


def test_parse_sar_params():
    d = lc.parse_sar_params(None)
    assert d == {
        "start": lc.SAR_START_DEFAULT,
        "increment": lc.SAR_INCREMENT_DEFAULT,
        "maximum": lc.SAR_MAXIMUM_DEFAULT,
    }
    got = lc.parse_sar_params('{"start":0.03,"increment":0.02,"maximum":0.2}')
    assert got["start"] == pytest.approx(0.03)
    assert got["increment"] == pytest.approx(0.02)
    assert got["maximum"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="between"):
        lc.parse_sar_params('{"start":0}')
    with pytest.raises(ValueError, match="JSON"):
        lc.parse_sar_params("{nope")


def test_parse_stochs():
    assert lc.parse_stochs(None) == []
    specs = lc.parse_stochs(
        '[{"id":"a","n":1,"k_period":9,"d_period":3},'
        '{"id":"b","n":2,"k_period":14,"d_period":3}]'
    )
    assert [s["k_period"] for s in specs] == [9, 14]
    assert [s["key"] for s in specs] == ["stoch_1", "stoch_2"]
    assert specs[0]["label"].startswith("Stoch1")
    with pytest.raises(ValueError, match="duplicated"):
        lc.parse_stochs(
            '[{"n":1,"k_period":9,"d_period":3},{"n":2,"k_period":9,"d_period":3}]'
        )
    with pytest.raises(ValueError, match="at most"):
        lc.parse_stochs(
            "[" + ",".join('{"k_period":9,"d_period":3,"n":%d}' % (i + 1) for i in range(9)) + "]"
        )


def test_series_stochs_overlay(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=40, freq="1min", tz="UTC")
    t = np.linspace(0, 4 * np.pi, 40)
    close = 100 + 5 * np.sin(t)
    ohlc = pd.DataFrame({
        "bucket": buckets,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: ohlc)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "expand_vp_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="so_sync_score_9_60",
        end=end.isoformat(),
        stochs='[{"id":"a","n":1,"k_period":9,"d_period":3}]',
    )
    assert len(out["stochs"]) == 1
    assert out["stochs"][0]["key"] == "stoch_1"
    assert "stoch_1" in out["overlays"]
    assert "so_sync_score_9_60" in out["overlays"]
    finite = [v for v in out["overlays"]["stoch_1"] if v is not None]
    assert finite
    assert min(finite) >= 0 and max(finite) <= 100


def test_vwap_bollinger_bands_two_sigma():
    """Volume-weighted σ around VWAP; upper/lower = VWAP ± 2σ (ADR-090 math)."""
    idx = pd.date_range("2026-08-25 14:00", periods=20, freq="1min", tz="UTC")
    px = np.concatenate([np.full(10, 100.0), np.full(10, 104.0)])
    sz = np.ones(20)
    bars = pd.DataFrame({
        "ts_event": idx,
        "pv": px * sz,
        "p2v": (px ** 2) * sz,
        "size": sz,
        "close": px,
    })
    out = lc._overlays_from_vwap_bars(
        bars, idx, ["vw_vwap", "vw_bb_upper", "vw_bb_lower"], vwap_origin="session",
    )
    mid = out["vw_vwap"][-1]
    upper = out["vw_bb_upper"][-1]
    lower = out["vw_bb_lower"][-1]
    assert mid is not None and upper is not None and lower is not None
    assert upper > mid > lower
    assert upper - mid == pytest.approx(mid - lower)
    # Reconstruct σ from E[p²]−VWAP² on the full window (one origin).
    vwap = float(np.sum(px * sz) / np.sum(sz))
    var = float(np.sum((px ** 2) * sz) / np.sum(sz) - vwap ** 2)
    sigma = float(np.sqrt(max(var, 0.0)))
    assert mid == pytest.approx(vwap)
    assert upper == pytest.approx(vwap + lc.VWAP_BB_STDEV * sigma)
    assert lower == pytest.approx(vwap - lc.VWAP_BB_STDEV * sigma)


def test_series_vwap_bands_flag(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=20, freq="1min", tz="UTC")
    close = np.concatenate([np.full(10, 100.0), np.full(10, 104.0)])
    ohlc = pd.DataFrame({
        "bucket": buckets,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    bars = pd.DataFrame({
        "ts_event": buckets,
        "pv": close,
        "p2v": close ** 2,
        "size": np.ones(20),
        "close": close,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: ohlc)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "load_vwap_bars", lambda *a, **k: bars)
    monkeypatch.setattr(lc, "expand_vp_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="vw_vwap",
        end=end.isoformat(), vwap_bands=True,
    )
    assert out["vwap_bands"] is True
    assert out["vwap_bb_stdev"] == lc.VWAP_BB_STDEV
    assert "vw_bb_upper" in out["overlays"] and "vw_bb_lower" in out["overlays"]
    assert out["overlays"]["vw_bb_upper"][-1] > out["overlays"]["vw_vwap"][-1]
    assert out["overlays"]["vw_bb_lower"][-1] < out["overlays"]["vw_vwap"][-1]


def test_series_sar_params(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=30, freq="1min", tz="UTC")
    close = np.linspace(100.0, 110.0, 30)
    df = pd.DataFrame({
        "bucket": buckets,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "expand_vp_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="ps_sar",
        end=end.isoformat(),
        sar='{"start":0.03,"increment":0.02,"maximum":0.2}',
    )
    assert out["sar"] == {"start": 0.03, "increment": 0.02, "maximum": 0.2}
    assert "ps_sar" in out["overlays"]
    assert any(v is not None for v in out["overlays"]["ps_sar"])


def test_ema_on_close_includes_current_bar():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    ema1 = lc.ema_on_close(close, 1)
    assert list(ema1) == list(close)
    ema3 = lc.ema_on_close(close, 3)
    assert pd.isna(ema3.iloc[0]) and pd.isna(ema3.iloc[1])
    assert np.isfinite(ema3.iloc[2])
    # Bumping the bar being evaluated must move the EMA (current bar is in the window).
    bumped = close.copy()
    bumped.iloc[-1] += 10.0
    assert lc.ema_on_close(bumped, 3).iloc[-1] > lc.ema_on_close(close, 3).iloc[-1]


def test_series_emas_overlay(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=10, freq="1min", tz="UTC")
    close = np.arange(100.0, 110.0)
    df = pd.DataFrame({
        "bucket": buckets,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "expand_vp_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="",
        end=end.isoformat(),
        emas='[{"id":"a","n":1,"lookback":3},{"id":"b","n":2,"lookback":5}]',
    )
    assert len(out["emas"]) == 2
    assert out["emas"][0]["key"] == "ema_1"
    assert out["emas"][1]["lookback"] == 5
    assert "ema_1" in out["overlays"] and "ema_2" in out["overlays"]
    assert out["overlays"]["ema_1"][0] is None
    assert out["overlays"]["ema_1"][1] is None
    assert out["overlays"]["ema_1"][2] is not None
    expected = lc.ema_on_close(pd.Series(close), 3)
    assert out["overlays"]["ema_1"][9] == pytest.approx(float(expected.iloc[9]))


def test_ema_warmup_uses_adjacent_prior_bars_not_rth_jump(monkeypatch):
    """08:30 EMA3 must see 08:28/08:29 even when Aggregate is off."""
    end = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    # Visible window is cash only; prior minutes exist only in the warmup load.
    vis = pd.date_range("2026-09-22 13:30", periods=4, freq="1min", tz="UTC")  # 08:30 CT
    prior = pd.date_range("2026-09-22 13:26", periods=4, freq="1min", tz="UTC")  # 08:26–08:29 CT
    def _frame(idx, close0):
        close = close0 + np.arange(len(idx), dtype="float64")
        return pd.DataFrame({
            "bucket": idx, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 10.0, "symbol": "NQU6",
        })
    visible = _frame(vis, 100.0)
    earlier = _frame(prior, 90.0)

    def fake_load(tf, start, end):
        # Warmup request ends at the visible start; display request starts there.
        if end <= vis[0].to_pydatetime():
            return earlier
        return visible

    monkeypatch.setattr(lc, "load_ohlc", fake_load)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "expand_vp_jobs", lambda *a, **k: [])
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="30m", overlays="",
        end=end.isoformat(), eth_aggregate=False,
        emas='[{"id":"a","n":1,"lookback":3}]',
    )
    cont = pd.concat([earlier, visible], ignore_index=True)
    expected = lc.ema_on_close(cont["close"], 3)
    # First visible bar (index 4 of the continuous series) is finite.
    assert out["overlays"]["ema_1"][0] == pytest.approx(float(expected.iloc[4]))
    assert out["n"] == 4


def test_window_for_session_and_no_tf_cap():
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    start, fin = lc.window_for("session", timeframe="1min", end=end)
    assert fin == end
    assert start < fin
    s2, e2 = lc.window_for("30m", timeframe="5s", end=end)
    assert e2 - s2 == timedelta(minutes=30)
    s3, e3 = lc.window_for("20d", timeframe="5s", end=end)
    assert e3 - s3 == timedelta(days=20)


def test_parse_lookback_custom_and_presets():
    assert lc.parse_lookback("session") == ("session", None)
    assert lc.parse_lookback("GLOBEX")[0] == "session"
    assert lc.parse_lookback("4h") == ("4h", timedelta(hours=4))
    tok, delta = lc.parse_lookback("45m")
    assert tok == "45m" and delta == timedelta(minutes=45)
    tok, delta = lc.parse_lookback("12 hours")
    assert tok == "12h" and delta == timedelta(hours=12)
    tok, delta = lc.parse_lookback("2w")
    assert tok == "2w" and delta == timedelta(weeks=2)
    tok, delta = lc.parse_lookback("1y")
    assert tok == "1y" and delta == timedelta(days=365)
    with pytest.raises(ValueError, match="duration"):
        lc.parse_lookback("nope")
    with pytest.raises(ValueError):
        lc.parse_lookback("0d")
    with pytest.raises(ValueError, match="5 years"):
        lc.parse_lookback("10y")


def test_window_for_wide_custom_and_tf_cap():
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    s, e = lc.window_for("10d", timeframe="1min", end=end)
    assert e - s == timedelta(days=10)
    s2, e2 = lc.window_for("90d", timeframe="1min", end=end)
    assert e2 - s2 == timedelta(days=90)
    s3, e3 = lc.window_for("2w", timeframe="5min", end=end)
    assert e3 - s3 == timedelta(weeks=2)
    s4, e4 = lc.window_for("1y", timeframe="1hour", end=end)
    assert e4 - s4 == timedelta(days=365)


def test_parse_lookback_date_range():
    tok, delta = lc.parse_lookback("2026-08-01..2026-08-15")
    assert tok == "2026-08-01..2026-08-15"
    start, stop = lc.range_window(date(2026, 8, 1), date(2026, 8, 15))
    assert delta == stop - start
    tok, _ = lc.parse_lookback("2026-08-01 to 2026-08-01")
    assert tok == "2026-08-01..2026-08-01"
    assert lc.parse_lookback_range("7d") is None
    with pytest.raises(ValueError, match="To must be on or after From"):
        lc.parse_lookback("2026-08-15..2026-08-01")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        lc.parse_lookback("2026-02-31..2026-03-01")
    with pytest.raises(ValueError, match="5 years"):
        lc.parse_lookback("2018-01-01..2026-01-01")


def test_window_for_date_range_ct_and_asof_clip():
    CT = ZoneInfo("America/Chicago")
    as_of = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    s, e = lc.window_for("2026-08-20..2026-08-22", timeframe="1min", end=as_of)
    assert s == datetime(2026, 8, 20, tzinfo=CT).astimezone(timezone.utc)
    assert e == datetime(2026, 8, 23, tzinfo=CT).astimezone(timezone.utc)
    # To = as-of's CT date → right edge is as-of, not midnight tonight.
    s2, e2 = lc.window_for("2026-08-20..2026-08-28", timeframe="1min", end=as_of)
    assert s2 == datetime(2026, 8, 20, tzinfo=CT).astimezone(timezone.utc)
    assert e2 == as_of
    s3, e3 = lc.window_for("2026-08-01..2026-08-22", timeframe="5s", end=as_of)
    assert e3 - s3 == (
        datetime(2026, 8, 23, tzinfo=CT).astimezone(timezone.utc)
        - datetime(2026, 8, 1, tzinfo=CT).astimezone(timezone.utc)
    )
    with pytest.raises(ValueError, match="empty window"):
        lc.window_for("2026-09-01..2026-09-05", timeframe="1min", end=as_of)


def test_lookback_catalog_has_wide_presets():
    lbs = list(lc._LOOKBACK)
    for k in ("8h", "12h", "2d", "7d", "10d", "30d", "60d", "90d", "1y"):
        assert k in lbs
    assert getattr(lc, "_TF_MAX_LOOKBACK", {}) == {}
    assert lc.MAX_BARS == 50_000


def test_overlay_catalog_keys_unique():
    keys = [o["key"] for o in lc.OVERLAYS]
    assert len(keys) == len(set(keys))
    assert "of_cvd" in keys and "vw_vwap" in keys
    assert "of_ofi_cum" in keys
    assert "mf_mfi" in keys and "ad_line" in keys
    assert "lt_prints" in keys
    by = {o["key"]: o for o in lc.OVERLAYS}
    assert by["of_delta"]["label"] == "Delta Volume"
    assert by["vw_vwap"]["group"] == "Price Action"
    assert by["vw_vwap_dist"]["group"] == "Price Action"
    assert by["cs_clv"]["group"] == "Price Action"
    assert by["ps_sar"]["group"] == "Models"
    assert all(o["group"] not in ("VWAP", "Candle", "SAR") for o in lc.OVERLAYS)


def test_lightweight_charts_vendor_is_v5():
    vendor = (REPO / "src/webui/static/vendor/lightweight-charts.standalone.production.js").read_text(errors="replace")[:500]
    assert "v5." in vendor
    js = (REPO / "src/webui/static/app.js").read_text()
    assert "lwAddSeries" in js
    assert "CandlestickSeries" in js
    assert "createSeriesMarkers" in js or "lwSetMarkers" in js


def test_overlay_osc_panes_have_own_pane_and_zero_line():
    by = {o["key"]: o for o in lc.OVERLAYS}
    assert by["of_cvd"]["pane"] == "osc"
    assert by["of_cvd"]["zero_line"] == "1"
    assert by["vw_vwap"]["pane"] == "price"
    assert by["ps_sar"]["pane"] == "price"
    assert by["lt_prints"]["pane"] == "canvas"
    for k in ("of_delta", "of_ofi", "of_ofi_cum", "ad_line", "cs_clv", "vw_vwap_dist"):
        assert by[k]["pane"] == "osc"
        assert by[k]["zero_line"] == "1"
    assert by["mf_mfi"]["pane"] == "osc"
    assert by["mf_mfi"].get("zero_line") != "1"
    assert by["so_sync_score_all"]["pane"] == "osc"
    assert by["so_sync_score_all"]["zero_line"] == "1"
    assert by["so_sync_score_9_60"]["pane"] == "osc"
    assert by["so_sync_score_9_60"]["zero_line"] == "1"
    assert "so_fast_d_9_3" not in by


def test_pack_prints_and_cap():
    idx = pd.date_range("2026-08-28 14:00", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "ts_event": idx,
        "symbol": "NQU6",
        "price": [23900.25, 23901.0, 23902.5],
        "size": [10, 25, 12],
        "side": ["B", "A", "N"],
    })
    rows, capped = lc.pack_prints(df)
    assert capped is False
    assert rows[0]["sz"] == 10 and rows[0]["side"] == "B"
    assert rows[1]["p"] == 23901.0 and rows[1]["side"] == "A"
    assert rows[2]["side"] == "N"
    assert rows[0]["ts"].endswith("Z") and "T" in rows[0]["ts"]
    big = pd.concat([df] * (lc.MAX_PRINTS // 2 + 1), ignore_index=True)
    _, capped2 = lc.pack_prints(big)
    assert capped2 is True


def test_load_large_prints_sql_filters_size(monkeypatch):
    captured = {}

    def fake_query(sql, params, **k):
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(lc, "_query", fake_query)
    monkeypatch.setattr(
        lc, "active_contracts_in_range",
        lambda *a, **k: [(date(2026, 8, 28), "NQU6")],
    )
    start = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    out = lc.load_large_prints(start, end, 10)
    assert "size >=" in captured["sql"]
    assert captured["params"][0] == start
    assert captured["params"][1] == end
    assert captured["params"][3] == 10
    assert out.empty


def test_prints_uses_chart_lookback_window(monkeypatch):
    captured = {}

    def fake_load(start, end, min_size, **k):
        captured["start"] = start
        captured["end"] = end
        captured["min_size"] = min_size
        captured["rth_only"] = k.get("rth_only")
        return pd.DataFrame(columns=["ts_event", "symbol", "price", "size", "side"])

    monkeypatch.setattr(lc, "load_large_prints", fake_load)
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    out = lc.prints(
        timeframe="1min", lookback="4h", end=end.isoformat(), min_size=12,
    )
    s, e = lc.window_for("4h", timeframe="1min", end=end)
    assert captured["start"] == s
    assert captured["end"] == e
    assert captured["min_size"] == 12
    assert captured["rth_only"] is False
    assert out["n"] == 0
    assert out["min_size"] == 12


def test_chart_prints_bad_min_size_is_400(client):
    r = client.get("/api/chart/prints", params={"min_size": "nope"})
    assert r.status_code == 400


def test_series_asof_is_profile_end(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    asof = datetime(2026, 8, 28, 14, 2, tzinfo=timezone.utc)
    captured = {}
    buckets = pd.date_range("2026-08-28 14:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bucket": buckets,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())

    def fake_expand(*a, **k):
        captured["chart_end"] = k.get("chart_end")
        captured["n_ohlc"] = 0 if k.get("ohlc") is None else len(k["ohlc"])
        return []

    monkeypatch.setattr(lc, "expand_vp_jobs", fake_expand)
    monkeypatch.setattr(lc, "profiles_payloads", lambda *a, **k: [])
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="",
        end=end.isoformat(), asof=asof.isoformat(),
        profiles='[{"kind":"current"}]',
    )
    assert captured["chart_end"] == asof
    assert captured["n_ohlc"] == 3
    assert out["n"] == 5
    assert out["asof"] is not None


def test_chart_series_bad_tf_is_400(client):
    r = client.get("/api/chart/series", params={"timeframe": "3min"})
    assert r.status_code == 400


def test_chart_series_bad_lookback_is_400(client):
    r = client.get("/api/chart/series", params={"lookback": "nope"})
    assert r.status_code == 400
    assert "lookback" in r.json()["detail"].lower()
    r2 = client.get("/api/chart/series", params={"lookback": "2026-08-15..2026-08-01"})
    assert r2.status_code == 400
    assert "from" in r2.json()["detail"].lower()


def test_spa_declares_live_chart_tab():
    html = (REPO / "src/webui/static/index.html").read_text()
    js = (REPO / "src/webui/static/app.js").read_text()
    css = (REPO / "src/webui/static/style.css").read_text()
    assert 'data-nav="chart"' in html
    assert "Live Chart" in html
    assert "viewLiveChart" in js
    assert "#\\/chart" in js or "#/chart" in js
    assert "#\\/border" in js or "#/border" in js
    assert 'id="lc-tf"' in js
    assert 'id="lc-lb"' in js
    assert 'id="lc-lb-prompt"' in js
    assert 'id="lc-lb-custom"' in js
    assert "Look back how far" in js
    assert "Custom…" in js
    assert "parseLookbackToken" in js
    assert "parseLookbackRange" in js
    assert "openLookbackPrompt" in js
    assert 'name="lc-lb-mode"' in js
    assert 'id="lc-lb-from"' in js
    assert 'id="lc-lb-to"' in js
    assert "Date range" in js
    assert "Chart date range" in js
    assert "lc-lb-modes" in css
    assert "lc-lb-range-row" in css
    assert "bars_capped" in js
    assert "showing newest" in js
    assert 'id="lc-ov"' in js
    assert 'id="lc-cvd-origin"' in js
    assert 'id="lc-ofi-origin"' in js
    assert 'id="lc-ad-origin"' in js
    assert "CVD_CUM origin" in js
    assert "lc-ov-gear" in js
    assert "Volume Profile" in js
    assert 'id="lc-vp"' in js
    assert 'id="lc-vp-axis"' in js
    assert "paintVpAxis" in js
    assert "fitVpAxisWidth" in js
    assert "--lc-vp-axis-width" in js
    assert "vpAxisAutoscale" in js
    assert "updateVpAxisPriceRange" in js
    assert "autoscaleInfoProvider: vpAxisAutoscale" in js
    assert "lc-vp-axis-on" in js
    assert "vpSessionMd" in js
    assert "vpAxisLabelText" in js
    assert "boundEdgeSession" in js
    assert "VPn LEVEL MM/DD price" in js
    assert "Math.min(220," in js
    assert "Axis labels" in js
    assert "lc-vp-add" in js
    assert 'id="lc-vp-refresh"' in js
    assert "Apply Volume Profile settings" in js
    assert "vpDraftSpecs" in js
    assert "vpEnabled" in js
    assert 'id="lc-ema"' in js
    assert 'id="lc-ema-add"' in js
    assert 'id="lc-ema-add-kind"' in js
    assert 'id="lc-ema-refresh"' in js
    assert "paintEmaCards" in js
    assert "syncEmaSeries" in js
    assert "emasForApi" in js
    assert "emaDraftSpecs" in js
    assert "emaEnabled" in js
    assert "Lookback bars" in js
    assert "current bar included" in js
    assert ".lc-ema-card" in css
    assert ".lc-ema-add" in css
    assert ".lc-ema-refresh" in css
    assert "Single VP" in js or "vpMark" in js
    assert "Current SVP" not in js
    assert "Custom range VP" not in js
    assert "Recurring" in js
    assert "Session timeframe" in js or ">Session<" in js or "Session" in js
    assert "full_session" in js
    assert "vpMark" in js
    assert "1 sess" in js
    assert "2 sess" in js
    assert "All sess" in js
    assert "sessionDayRightX" in js
    assert "vpForwardX" in js
    assert "sessionDayLeftX" in js
    assert "levelBlendExtendRange" in js
    assert "opposite chart edge" in js
    assert "addSessionEndDates" in js
    assert "lineVisible: false" in js
    assert "VP1" in js
    assert "Extend VAH" not in js
    assert 'option value="custom"' in js
    assert 'data-f="session_from"' in js
    assert 'data-f="session_to"' in js
    assert 'data-f="session_date"' in js
    assert "drawSessionDayLines" in js
    assert 'ctx.strokeStyle = "#1f2a36"' in js
    assert "ctx.lineWidth = 1" in js
    assert "ctx.setLineDash([4, 4])" in js
    assert "spacing) * 0.5 + leftPad" in js
    assert "globexSessionEndDate" in js
    assert "--line: #4d647c" in css
    assert ".lc-vp-custom-hm" in css
    assert ".lc-vp-axis" in css
    assert ".lc-vp-refresh" in css
    assert "var(--lc-vp-axis-width, 108px)" in css
    assert "lc-vp-axis-on" in css
    assert "left_frame" in js
    assert "right_frame" in js
    assert "overlay_side" in js
    assert 'data-f="extend"' in js
    assert 'data-f="hist_width"' in js
    assert 'step="any"' in js
    assert "syncVpDraftFromDom" in js
    assert 'dataset.f === "hist_width"' in js
    assert 'data-vp-width-step="1"' in js
    assert 'data-vp-width-step="-1"' in js
    assert 'data-f="chart_labels"' in js
    assert 'data-f="axis_labels"' in js
    assert "lc-vp-labels" in js
    assert ".lc-vp-labels" in css
    assert "defaultVpOverlaySide" in js
    assert "[...vpSpecs, ...vpDraftSpecs]" in js
    assert "vpDraftSpecs = assignVpNs(readVpSpecsFromDom());" in js
    assert "hist_width: blend ? 0.35 : .25" in js
    assert 'value="level_blend">Level Blend' in js
    assert "isLevelBlend" in js
    assert "lookback_sessions" in js
    assert "level_blend" in js
    assert "vpHistogramWidth" in js
    assert "applyVpDisplaySettings" in js
    assert "const span = Math.abs(b - a);" in js
    assert "const width = Math.max(1, Math.abs(x1 - x0));" in js
    assert "if (x1 > plotW)" not in js
    assert "chart_labels: false" in js
    assert "axis_labels: false" in js
    assert "lc-vp-canvas" in js
    assert "MBP-10 ladder" not in js
    assert "bvImbalanceStats" not in js
    assert 'id="lc-mbp"' not in js
    assert 'id="bv-imb"' not in js
    assert 'id="lc-hover-show"' in js
    assert 'g === "Stochastic"' in js
    assert "Stochastic Fast" not in js
    assert "function deltaHistData" in js
    assert 'color: v < 0 ? "#ff6b6b" : "#7ee0a3"' in js
    assert 'k === "of_delta" ? deltaHistData(p)' in js
    assert 'g === "Price Action"' in js
    assert 'lc-og-h">Chart formatting' in js
    assert ">Text size" in js
    assert ">In-chart text" in js
    assert 'id="lc-chart-label"' in js
    assert "function chartLabelSize" in js
    assert "lc_chart_label" in js
    assert 'getItem("lc_chart_label")' in js
    assert "raw != null && Number.isFinite(cl)" in js
    assert "Chart text" not in js
    assert 'lc-og-h">Models' in js
    assert 'lc-og-h">SAR' not in js
    assert 'lc-og-h">EMA' not in js
    assert 'id="lc-md-open"' in js
    assert "/api/chart/series" in js
    assert "cvd_origin" in js
    assert "ofi_origin" in js
    assert "profiles" in js
    assert "premarket" in js
    assert "of_ofi_cum" in js
    assert "OFI_CUM" in js
    assert 'id="lc-readout"' in js
    assert 'id="lc-hover"' in js
    assert "lc-hover-grip" in js
    assert "profileCoversUnix" in js
    assert "profilesCoveringUnix" in js
    assert "profiles.slice(0, 4)" not in js
    assert "${tag} VHi" in js
    assert "${tag} VLo" in js
    assert "${tag} VAH" in js
    assert "${tag} VAL" in js
    assert "${tag} POC" in js
    assert "nearestHoverCorner" in js
    assert "lc_hover_corner" in js
    assert 'id="lc-cvd-style"' in js
    assert "syncCvdSeries" in js
    assert "applyPaneLayout" in js
    assert "armPaneStretchSave" in js
    assert "lwAddSeries" in js
    assert "lwZeroLine" in js
    assert "ensureZeroInScale" in js
    assert "lwEnsurePane" in js
    assert "lwForcePane" in js
    assert "addPane" in js
    assert "moveToPane" in js
    assert "applyPaneHeights" in js
    assert "showPaneScale" in js
    assert 'priceScale("right"' in js
    assert "separatorColor" in js
    assert "lc_pane_stretch" in js
    assert "own pane below volume" in js or "Candle body" in js
    assert "CVD O" in js
    assert "CVD Δ" in js
    assert "CVD OHLC" not in js
    assert "CVD O/H/L/C" not in js
    assert 'title: wantLine ? "" : ttl' in js
    assert "?v=5.2.1" in js
    assert "drawSarStars" in js
    assert "drawStar" in js
    assert "mf_mfi" in js
    assert "ad_line" in js
    assert 'color: (p.c[i] || 0) >= (p.o[i] || 0) ? "#7ee0a3" : "#ff6b6b"' in js
    assert 'lineVisible: k !== "ps_sar"' in js
    assert "paintCursor" in js
    assert "cursorSnapshot" in js
    assert "createPriceLine" in js
    assert "scheduleDrawProfiles" in js
    assert "reloadQueued" in js
    assert "chart-fill" in js
    assert "chart-mode" in js
    assert "lc-eth-display" in js
    assert "lc-eth-shade" in js
    assert "lc-eth-agg" in js
    assert 'id="lc-replay"' in js
    assert "setReplayUnix" in js
    assert "withFrozenTimeScale" in js
    assert "asof" in js
    assert "lc-replay-slider" in js
    assert "lc-replay-axis" not in js
    assert "lc-text-size" in js
    assert "2 sess" in js
    assert "Tick histogram" in js
    assert "Grouped ticks" in js
    assert "groupTicks" in js
    assert "Group ticks" in js
    assert "Right display" in js
    assert "Left display" in js
    assert "Right opacity" in js
    assert "Left opacity" in js
    assert "LVN node rectangles" not in js
    assert "show_lvn" in js
    assert "hist_style" in js
    assert "extend_lvn" in js
    assert "LVN %" in js
    assert "LVN min grouped bars" in js
    assert 'data-f="lvn_min_groups"' in js
    assert "Extend LVN" in js
    assert "Adaptive KDE" in js
    assert 'data-f="kde_bandwidth"' in js
    assert 'data-f="kde_prominence"' in js
    assert 'data-f="kde_valley_depth"' in js
    assert 'data-f="kde_volume"' in js
    assert 'data-f="kde_separation"' in js
    assert 'data-f="kde_valley_width"' in js
    assert 'data-f="kde_max_zones"' in js
    assert "Use multi-section preset" in js
    assert "kde_preset" in js or "data-kde-preset=\"multi\"" in js
    assert "hvn_zones" in js
    assert "Plot variable bounds" in js
    assert 'data-f="plot_variable_bounds"' in js
    assert "variableSegs" in js
    assert 'variableSegs(item, "high", "#ff4d4d"' in js
    assert 'variableSegs(item, "poc", "#ffd166"' in js
    assert "hvn_peaks" in js
    assert "lvn_valleys" in js
    assert 'kind: "HVN"' in js
    assert 'kind: "LVN"' in js
    assert 'color: "#050505"' in js
    assert 'color: "#ffffff"' in js
    assert "structureBars" in js
    assert "coarse_bins" in js
    assert "lt_prints" in js
    assert "Large prints" in js
    assert "lc-print-min" in js
    assert "lc-print-opacity" in js
    assert "lc-print-display-value" in js
    assert "lc-print-extend-bars" in js
    assert "drawLargePrints" in js
    assert "/api/chart/prints" in js
    assert "printCache" in js
    assert "unixToBarFrac" in js
    assert "printBarGeometry" in js
    assert "bar.spacing * .45" in js
    assert "fmtTsUs" in js
    assert "hitPrint" in js
    assert "printHits" in js
    assert "Print price" in js
    assert "Print qty" in js
    assert "Print time" in js
    assert "chip(\"Print px\"" not in js
    assert "applyPaneHeights(true)" in js
    assert "Unspecified (N)" in js
    assert 'strokeStyle = "#000000"' in js
    assert "fmtCrosshairTime" in js
    assert "drawEthBands" in js
    assert "eth_aggregate" in js
    assert "nonRthIndexRuns" in js
    assert "isRthCtUnix" in js
    assert "globexOvernightWindows" in js
    assert "timeToX" in js
    assert "17:00:00" in js
    assert "08:30:00" in js
    assert "addCivilDays" in js
    assert "nextCtDate" not in js
    assert "36 * 3600" not in js
    html = (REPO / "src/webui/static/index.html").read_text()
    css = (REPO / "src/webui/static/style.css").read_text()
    assert "chart-fill" in css
    assert "chart-mode" in css
    assert "flex-wrap: nowrap" in css
    assert "contain: layout" in css


CT = ZoneInfo("America/Chicago")


def test_origin_start_session_bod_mkt():
    # 2026-08-28 18:00 UTC = 13:00 CDT
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    sess = lc.origin_start(end, "session")
    assert sess == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    bod = lc.origin_start(end, "bod")
    assert bod == datetime(2026, 8, 28, 0, 0, tzinfo=CT).astimezone(timezone.utc)
    mkt = lc.origin_start(end, "mkt")
    assert mkt == datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    # just before 17:00 CT still previous Globex open
    pre = datetime(2026, 8, 28, 16, 59, tzinfo=CT).astimezone(timezone.utc)
    assert lc.origin_start(pre, "session") == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    at = datetime(2026, 8, 28, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert lc.origin_start(at, "session") == at
    pre_mkt = datetime(2026, 8, 28, 8, 29, tzinfo=CT).astimezone(timezone.utc)
    assert lc.origin_start(pre_mkt, "mkt") == datetime(2026, 8, 27, 8, 30, tzinfo=CT).astimezone(timezone.utc)


def test_profile_window_uses_last_bar_not_wall_clock():
    # Saturday noon UTC — Globex session from Friday 17:00 is empty.
    end = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)  # Friday 15:00 CT
    ohlc = pd.DataFrame({"bucket": [last]})
    start, stop = lc.profile_window(ohlc, end, "session")
    assert start == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert stop == end
    empty_start, _ = lc.profile_window(pd.DataFrame(), end, "session")
    # no bars → wall clock Saturday → Friday 17:00 CT
    assert empty_start == datetime(2026, 8, 28, 17, 0, tzinfo=CT).astimezone(timezone.utc)


def test_coerce_origin_aliases():
    assert lc.coerce_origin("RTH") == "mkt"
    assert lc.coerce_origin("globex") == "session"
    assert lc.coerce_origin("midnight") == "bod"
    with pytest.raises(ValueError):
        lc.coerce_origin("asia")
    with pytest.raises(ValueError):
        lc.coerce_origin("premarket")
    assert lc.coerce_origin("overnight", allowed=lc.SVP_ORIGINS) == "premarket"


def test_premarket_svp_window_caps_at_830():
    # Friday 10:00 CDT — premarket is Thu 17:00 → Fri 08:30
    end = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)
    ohlc = pd.DataFrame({"bucket": [last]})
    start, stop = lc.profile_window(ohlc, end, "premarket")
    assert start == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert stop == datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    # Friday 07:00 CDT — still inside premarket, stop = now
    early_end = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    s2, e2 = lc.profile_window(pd.DataFrame({"bucket": [early_end]}), early_end, "premarket")
    assert s2 == start
    assert e2 == early_end
    # Friday 18:00 CDT — new Globex, stop = now (before Sat 08:30)
    late = datetime(2026, 8, 28, 23, 0, tzinfo=timezone.utc)
    s3, e3 = lc.profile_window(pd.DataFrame({"bucket": [late]}), late, "premarket")
    assert s3 == datetime(2026, 8, 28, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert e3 == late


def test_money_flow_index_all_up_is_100():
    n = 20
    close = np.arange(100.0, 100.0 + n)
    ohlc = pd.DataFrame({
        "bucket": pd.date_range("2026-08-28 14:00", periods=n, freq="1min", tz="UTC"),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 10.0,
    })
    mfi = lc.money_flow_index(ohlc, period=14)
    assert pd.isna(mfi.iloc[12])
    assert mfi.iloc[13] == 100.0
    assert mfi.iloc[-1] == 100.0


def test_ad_increment_is_clv_times_volume():
    ohlc = pd.DataFrame({
        "bucket": pd.date_range("2026-08-28 14:00", periods=3, freq="1min", tz="UTC"),
        "high": [102.0, 102.0, 101.0],
        "low": [100.0, 100.0, 99.0],
        "close": [101.5, 100.5, 100.5],
        "volume": [10.0, 10.0, 8.0],
    })
    inc = lc.clv_volume(ohlc)
    # CLV = (2c - h - l) / (h - l)
    assert abs(inc.iloc[0] - ((2 * 101.5 - 102 - 100) / 2.0) * 10.0) < 1e-9
    ad = lc.origin_cumsum(inc, "session")
    assert abs(ad.iloc[2] - inc.iloc[:3].sum()) < 1e-9


def test_volume_nodes_lvn_is_poc_relative_valley():
    """LVN = contiguous bins ≤ 30% of POC, not containing POC (no CME formula)."""
    px = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    vol = np.array([40.0, 50.0, 8.0, 5.0, 9.0, 48.0, 45.0])
    buy = np.array([20.0, 30.0, 4.0, 2.0, 3.0, 20.0, 22.0])
    sell = vol - buy
    nodes = lc.volume_nodes(px, vol, buy, sell, lvn_pct=0.30)
    lvn = [n for n in nodes if n["lvn"]]
    assert len(lvn) == 1
    assert lvn[0]["lo"] == 3.0
    assert lvn[0]["hi"] == 5.0
    assert lvn[0]["v"] == 22.0
    body = [n for n in nodes if not n["lvn"]]
    assert any(n["lo"] <= 2.0 <= n["hi"] for n in body)
    assert all(not (n["lo"] <= 2.0 <= n["hi"]) for n in lvn)


def test_grouped_lvn_min_groups_removes_single_bar_valleys():
    bins = [
        {"lo": 1.0, "hi": 2.0, "p": 1.5, "v": 50.0, "b": 25.0, "s": 25.0},
        {"lo": 2.0, "hi": 3.0, "p": 2.5, "v": 5.0, "b": 2.5, "s": 2.5},
        {"lo": 3.0, "hi": 4.0, "p": 3.5, "v": 50.0, "b": 25.0, "s": 25.0},
        {"lo": 4.0, "hi": 5.0, "p": 4.5, "v": 4.0, "b": 2.0, "s": 2.0},
        {"lo": 5.0, "hi": 6.0, "p": 5.5, "v": 4.0, "b": 2.0, "s": 2.0},
        {"lo": 6.0, "hi": 7.0, "p": 6.5, "v": 50.0, "b": 25.0, "s": 25.0},
    ]
    one = lc.lvn_nodes_from_bins(bins, lvn_pct=0.30, lvn_min_groups=1)
    two = lc.lvn_nodes_from_bins(bins, lvn_pct=0.30, lvn_min_groups=2)
    assert len([n for n in one if n["lvn"]]) == 2
    assert len([n for n in two if n["lvn"]]) == 1
    assert [n for n in two if n["lvn"]][0]["lo"] == 4.0


def test_grouped_tick_bins_preserve_volume_and_tick_alignment():
    px = np.linspace(100.0, 109.0, 37)
    vol = np.ones(len(px))
    vol[len(px) // 2] = 40.0
    buy = vol * 0.6
    sell = vol - buy
    bins = lc.grouped_tick_bins(px, vol, buy, sell, tick_size=0.25, group_ticks=4)
    assert len(bins) == 10
    widths = [b["hi"] - b["lo"] for b in bins]
    assert max(widths) - min(widths) < 1e-9
    assert abs(bins[0]["lo"] - 100.0) < 1e-9
    assert abs(bins[-1]["hi"] - 110.0) < 1e-9
    assert abs(sum(b["v"] for b in bins) - float(vol.sum())) < 1e-9
    nodes = lc.lvn_nodes_from_bins(bins, lvn_pct=0.30)
    assert any(n["lvn"] for n in nodes)
    assert any(not n["lvn"] for n in nodes)


def test_volume_nodes_poc_never_lvn():
    px = np.array([1.0, 2.0, 3.0])
    vol = np.array([1.0, 100.0, 1.0])
    nodes = lc.volume_nodes(px, vol, lvn_pct=0.30)
    assert any(n["lvn"] and n["lo"] == 1.0 for n in nodes)
    assert any(n["lvn"] and n["lo"] == 3.0 for n in nodes)
    poc_nodes = [n for n in nodes if n["lo"] <= 2.0 <= n["hi"]]
    assert poc_nodes and not poc_nodes[0]["lvn"]


def test_value_area_expands_from_poc():
    px = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vol = np.array([1.0, 10.0, 50.0, 10.0, 1.0])
    poc, va_h, va_l = lc.value_area(px, vol, va_pct=0.70)
    assert poc == 3.0
    assert va_l == 2.0
    assert va_h == 4.0
    poc2, h2, l2 = lc.value_area(px, np.array([1.0, 1.0, 100.0, 1.0, 1.0]), va_pct=0.70)
    assert poc2 == h2 == l2 == 3.0


def test_adaptive_kde_finds_multiple_hvn_zones():
    bins = [{"p": float(i), "v": float(v)} for i, v in enumerate(
        [1, 2, 10, 3, 1, 1, 1, 8, 2, 1, 1]
    )]
    zones = lc.adaptive_kde_hvn_zones(
        bins, tick_size=1.0, bandwidth=1.0, prominence=0.05,
        volume_fraction=0.05, min_separation=2, max_zones=5,
    )
    assert len(zones) == 2
    assert {z["peak"] for z in zones} == {2.0, 7.0}
    assert all(z["volume_fraction"] > 0 for z in zones)
    assert all("right_valley" in z and "left_valley" in z for z in zones)


def test_profile_payload_includes_kde_hvn_relationship():
    vbp = pd.DataFrame({
        "px": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "vol": [1.0, 2.0, 10.0, 3.0, 1.0, 1.0, 1.0, 8.0, 2.0, 1.0, 1.0],
        "buy": [0.5] * 11, "sell": [0.5] * 11,
    })
    payload = lc.profile_payload(
        vbp, start=datetime(2026, 9, 7, tzinfo=timezone.utc),
        end=datetime(2026, 9, 8, tzinfo=timezone.utc),
        tick_size=1.0, group_ticks=1, hvn_method="adaptive_kde",
        kde_bandwidth=1.0, kde_prominence=0.05, kde_volume=0.05,
        kde_valley_depth=0.05, kde_separation=2, kde_valley_width=2,
        kde_max_zones=5,
    )
    assert len(payload["hvn_zones"]) == 2
    assert {z["va_relation"] for z in payload["hvn_zones"]} <= {"inside", "intersects", "outside"}
    assert [p["label"] for p in payload["hvn_peaks"]] == ["HVN 1", "HVN 2"]
    assert [v["label"] for v in payload["lvn_valleys"]] == ["LVN 1"]


def test_cvd_ohlc_from_closes_and_fine_path():
    idx = pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC")
    cvd = pd.Series([10.0, 12.0, 9.0, 15.0], index=idx)
    ohlc = lc.cvd_ohlc_from_closes(cvd)
    assert list(ohlc["c"]) == [10.0, 12.0, 9.0, 15.0]
    assert ohlc["o"].iloc[0] == 10.0
    assert ohlc["o"].iloc[1] == 10.0
    assert ohlc["h"].iloc[1] == 12.0
    assert ohlc["l"].iloc[2] == 9.0
    buckets = pd.DatetimeIndex(["2026-08-28 14:00", "2026-08-28 14:02"], tz="UTC")
    agg = lc.cvd_ohlc_from_fine(cvd, buckets)
    assert agg["o"].iloc[0] == 10.0
    assert agg["h"].iloc[0] == 12.0
    assert agg["l"].iloc[0] == 10.0
    assert agg["c"].iloc[0] == 12.0
    assert agg["o"].iloc[1] == 9.0
    assert agg["h"].iloc[1] == 15.0
    assert agg["c"].iloc[1] == 15.0


def test_cvd_candle_body_equals_bar_delta_except_first_and_fine_path():
    """Close−open is the bar delta on 1m after the first print; 5m-from-1m body is not."""
    idx = pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC")
    delta = pd.Series([10.0, 2.0, -3.0, 6.0], index=idx)
    ohlc = lc.cvd_ohlc_from_closes(delta.cumsum())
    body = ohlc["c"] - ohlc["o"]
    assert float(body.iloc[0]) == 0.0
    assert [float(x) for x in body.iloc[1:]] == [2.0, -3.0, 6.0]
    fine_idx = pd.date_range("2026-08-28 14:00", periods=5, freq="1min", tz="UTC")
    fine_delta = pd.Series([10.0, 2.0, 3.0, 4.0, 5.0], index=fine_idx)
    buckets = pd.DatetimeIndex(["2026-08-28 14:00", "2026-08-28 14:05"], tz="UTC")
    agg = lc.cvd_ohlc_from_fine(fine_delta.cumsum(), buckets)
    bar_delta = float(fine_delta.iloc[:5].sum())
    body0 = float(agg["c"].iloc[0] - agg["o"].iloc[0])
    assert bar_delta == 24.0
    assert body0 == 14.0
    assert body0 != bar_delta


def test_cvd_resets_at_session_and_accepts_warmup():
    # 16:00 CT and 18:00 CT — two Globex sessions
    t0 = datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    t1 = datetime(2026, 8, 28, 18, 0, tzinfo=CT).astimezone(timezone.utc)
    bars = pd.DataFrame({
        "bucket": [t0, t1],
        "buy_volume": [5.0, 3.0],
        "sell_volume": [0.0, 0.0],
    })
    cvd = lc.cvd_on_bars(bars, "session")
    assert list(cvd.to_numpy()) == [5.0, 3.0]
    same = pd.DataFrame({
        "bucket": pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC"),
        "buy_volume": [2.0, 2.0, 2.0, 2.0],
        "sell_volume": [0.0, 0.0, 0.0, 0.0],
    })
    cvd2 = lc.cvd_on_bars(same, "session")
    assert list(cvd2.to_numpy()) == [2.0, 4.0, 6.0, 8.0]
    warm = pd.DataFrame({
        "bucket": pd.date_range("2026-08-28 13:50", periods=2, freq="1min", tz="UTC"),
        "buy_volume": [10.0, 10.0],
        "sell_volume": [0.0, 0.0],
    })
    cvd3 = lc.cvd_on_bars(same, "session", warmup=warm)
    assert list(cvd3.to_numpy()) == [22.0, 24.0, 26.0, 28.0]


def test_cvd_candle_opens_at_zero_on_mkt_origin():
    start = datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    idx = pd.date_range(start, periods=3, freq="1min", tz="UTC")
    cvd = pd.Series([12.0, 17.0, 14.0], index=idx)
    ohlc = lc.cvd_ohlc_from_closes(cvd, "mkt")
    assert list(ohlc["o"].to_numpy()) == [0.0, 12.0, 17.0]
    assert list(ohlc["c"].to_numpy()) == [12.0, 17.0, 14.0]

    later = pd.date_range(start - timedelta(minutes=1), periods=4, freq="1min", tz="UTC")
    fine = pd.Series([300.0, 12.0, 17.0, 14.0], index=later)
    buckets = pd.DatetimeIndex([later[0], start])
    agg = lc.cvd_ohlc_from_fine(fine, buckets, "mkt")
    assert agg.loc[start, "o"] == 0.0
    assert agg.loc[start, "l"] == 0.0


def test_ofi_cum_resets_at_same_origin_as_cvd():
    t0 = datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    t1 = datetime(2026, 8, 28, 18, 0, tzinfo=CT).astimezone(timezone.utc)
    ofi = pd.Series([5.0, 3.0], index=pd.DatetimeIndex([t0, t1]))
    cum = lc.origin_cumsum(ofi, "session")
    assert list(cum.to_numpy()) == [5.0, 3.0]
    same = pd.Series(
        [2.0, 2.0, 2.0, 2.0],
        index=pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC"),
    )
    assert list(lc.origin_cumsum(same, "session").to_numpy()) == [2.0, 4.0, 6.0, 8.0]
    warm = pd.Series(
        [10.0, 10.0],
        index=pd.date_range("2026-08-28 13:50", periods=2, freq="1min", tz="UTC"),
    )
    assert list(lc.origin_cumsum(same, "session", warm).to_numpy()) == [22.0, 24.0, 26.0, 28.0]


def test_bar_ofi_cont_formula():
    bars = pd.DataFrame({
        "bucket": pd.date_range("2026-08-28 14:00", periods=3, freq="1min", tz="UTC"),
        "bid_px": [100.0, 100.25, 100.25],
        "ask_px": [100.25, 100.50, 100.25],
        "bid_sz": [4.0, 6.0, 5.0],
        "ask_sz": [3.0, 2.0, 4.0],
    })
    ofi = lc._bar_ofi(bars)
    assert ofi.iloc[0] == 0.0  # no previous BBO
    # bar 1: bid up → +6, ask up → +3 (prev ask_sz)
    assert ofi.iloc[1] == 6.0 + 3.0
    # bar 2: bid unchanged → +5 - 6, ask down → -4
    assert ofi.iloc[2] == (5.0 - 6.0) + (-4.0)


def test_series_lookback_capped_flag(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: pd.DataFrame())
    out = lc.series(timeframe="5s", lookback="20d", end=end.isoformat())
    assert out["lookback"] == "20d"
    assert out["lookback_capped"] is False
    assert out["bars_capped"] is False
    assert out["max_bars"] == lc.MAX_BARS
    out2 = lc.series(timeframe="5s", lookback="30m", end=end.isoformat())
    assert out2["lookback"] == "30m"
    assert out2["lookback_capped"] is False
    out3 = lc.series(timeframe="1min", lookback="45 min", end=end.isoformat())
    assert out3["lookback"] == "45m"
    assert out3["lookback_capped"] is False
    out4 = lc.series(
        timeframe="1min", lookback="2026-08-20..2026-08-22", end=end.isoformat(),
    )
    assert out4["lookback"] == "2026-08-20..2026-08-22"
    assert out4["lookback_capped"] is False
    out5 = lc.series(
        timeframe="5s", lookback="2026-08-01..2026-08-22", end=end.isoformat(),
    )
    assert out5["lookback"] == "2026-08-01..2026-08-22"
    assert out5["lookback_capped"] is False
    assert out5["start"].startswith("2026-08-01")


def test_series_bars_capped_at_max_bars(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    n = lc.MAX_BARS + 25
    buckets = pd.date_range("2026-08-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bucket": buckets,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQZ6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df)
    out = lc.series(timeframe="1min", lookback="20d", end=end.isoformat())
    assert out["n"] == lc.MAX_BARS
    assert out["bars_capped"] is True
    assert out["lookback_capped"] is True
    assert out["t"][0] == int(buckets[-lc.MAX_BARS].timestamp())
    assert out["t"][-1] == int(buckets[-1].timestamp())


def test_series_ofi_cum_covers_lookback(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bucket": buckets,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "bid_px": [100.0, 100.25, 100.25, 100.50],
        "ask_px": [100.25, 100.50, 100.50, 100.75],
        "bid_sz": [4.0, 6.0, 6.0, 7.0],
        "ask_sz": [3.0, 2.0, 2.0, 1.0],
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ticks")))
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="of_ofi_cum",
        end=end.isoformat(), cvd_origin="bod",
    )
    assert out["n"] == 4
    cum = out["overlays"]["of_ofi_cum"]
    assert all(v is not None for v in cum)
    assert cum[0] == 0.0
    assert cum[-1] == cum[0] + sum(lc._bar_ofi(df).iloc[1:])


def test_profile_payload_includes_high_low():
    start = datetime(2026, 8, 28, 13, 30, tzinfo=timezone.utc)
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    vbp = pd.DataFrame({
        "px": [100.0, 101.0, 102.25],
        "vol": [1.0, 10.0, 2.0],
        "buy": [1.0, 6.0, 1.0],
        "sell": [0.0, 4.0, 1.0],
    })
    out = lc.profile_payload(vbp, start=start, end=end, origin="session")
    assert out["high"] == 102.25
    assert out["low"] == 100.0
    assert out["poc"] == 101.0
    assert out["va_high"] >= out["va_low"]


def test_profile_body_extremes_use_the_extreme_candles_bodies():
    start = datetime(2026, 8, 28, 13, 30, tzinfo=timezone.utc)
    ohlc = pd.DataFrame({
        "bucket": [start, start + timedelta(minutes=1), start + timedelta(minutes=2)],
        "open": [100.0, 101.0, 102.0], "high": [103.0, 106.0, 104.0],
        "low": [99.0, 100.0, 96.0], "close": [102.0, 103.0, 98.0],
    })
    hi_body, lo_body = lc.profile_body_extremes(ohlc, [(start, start + timedelta(minutes=3))])
    assert hi_body == 103.0
    assert lo_body == 98.0


def test_parse_vp_start_naive_is_chicago():
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    ts = lc.parse_vp_start("2026-08-22T09:30:00", end=end)
    assert ts == datetime(2026, 8, 22, 9, 30, tzinfo=CT).astimezone(timezone.utc)
    assert lc.parse_vp_start("", end=end) is None
    with pytest.raises(ValueError):
        lc.parse_vp_start("2026-08-29T00:00:00Z", end=end)


def test_parse_vp_end_date_only_is_exclusive_next_midnight():
    ts = lc.parse_vp_bound("2026-08-28", field="vp_end", date_is_end=True)
    assert ts == datetime(2026, 8, 29, 0, 0, tzinfo=CT).astimezone(timezone.utc)
    close = lc.parse_vp_bound("2026-08-28T16:00:00", field="vp_end")
    assert close == datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)


def test_series_cvd_covers_lookback_without_ticks(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bucket": buckets,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ticks")))
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="of_cvd",
        end=end.isoformat(), cvd_origin="bod",
    )
    assert out["n"] == 5
    assert out["cvd_origin"] == "bod"
    assert out["overlays"]["of_cvd"] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert all(v is not None for v in out["overlays"]["of_cvd"])
    assert out["cvd_ohlc"] is not None
    assert out["cvd_ohlc"]["c"] == out["overlays"]["of_cvd"]
    assert out["cvd_ohlc"]["o"][0] == out["cvd_ohlc"]["c"][0]
    assert out["cvd_ohlc"]["h"][-1] == max(out["cvd_ohlc"]["o"][-1], out["cvd_ohlc"]["c"][-1])


def test_vwap_origin_resets_at_selected_clock():
    ticks = pd.DataFrame({
        "ts_event": pd.to_datetime([
            "2026-08-28T13:29:30Z", "2026-08-28T13:30:30Z",
        ]),
        "price": [100.0, 102.0],
        "size": [10.0, 10.0],
    })
    buckets = pd.date_range("2026-08-28T13:29:00Z", periods=2, freq="1min")
    out = lc._overlays_from_ticks(ticks, buckets, "1min", ["vw_vwap"], vwap_origin="mkt")
    assert out["vw_vwap"] == [100.0, 102.0]


def test_vwap_aggregate_path_covers_multiple_origin_periods():
    bars = pd.DataFrame({
        "ts_event": pd.to_datetime([
            "2026-08-27T14:00:00Z", "2026-08-28T14:00:00Z",
        ]),
        "pv": [1000.0, 2400.0], "size": [10.0, 20.0],
        "close": [100.0, 120.0],
    })
    buckets = pd.DatetimeIndex(bars["ts_event"])
    out = lc._overlays_from_vwap_bars(bars, buckets, ["vw_vwap"], vwap_origin="mkt")
    assert out["vw_vwap"] == [100.0, 120.0]


def test_series_svp_anchors_to_last_bar(monkeypatch):
    end = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bucket": [last],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    captured = {}

    def fake_vbp(start, end_, **k):
        captured["start"] = start
        captured["end"] = end_
        return pd.DataFrame({"px": [23950.0], "vol": [100.0], "buy": [60.0], "sell": [40.0]})

    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_vbp", fake_vbp)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="5d", overlays="",
        end=end.isoformat(), svp_origin="session",
    )
    want = datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert captured["start"] == want
    assert out["svp"] is not None
    assert out["svp"]["poc"] == 23950.0
    assert out["svp"]["bins"]


def test_series_vp_clamps_to_explicit_end(monkeypatch):
    chart_end = datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bucket": [last],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    captured = {}

    def fake_vbp(start, end_, **k):
        captured["start"] = start
        captured["end"] = end_
        return pd.DataFrame({"px": [24000.0], "vol": [50.0], "buy": [30.0], "sell": [20.0]})

    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_vbp", fake_vbp)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="5d", overlays="",
        end=chart_end.isoformat(),
        vp_start="2026-08-27T17:00:00",
        vp_end="2026-08-28T16:00:00",
    )
    assert captured["start"] == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert captured["end"] == datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    assert out["vp"]["poc"] == 24000.0
    with pytest.raises(ValueError, match="after From"):
        lc.series(
            timeframe="1min", lookback="5d", overlays="",
            end=chart_end.isoformat(),
            vp_start="2026-08-28T16:00:00",
            vp_end="2026-08-28T09:00:00",
        )


def test_chart_series_bad_origin_is_400(client):
    r = client.get("/api/chart/series", params={"cvd_origin": "asia"})
    assert r.status_code == 400


def test_session_tf_windows():
    # Friday 15:00 CT — in cash and full session ending 16:00
    ref = datetime(2026, 8, 28, 15, 0, tzinfo=CT)
    s, e = lc.session_tf_window(ref, "full_session")
    assert s == datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert e == datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    cs, ce = lc.session_tf_window(ref, "cash")
    assert cs == datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    assert ce == datetime(2026, 8, 28, 15, 0, tzinfo=CT).astimezone(timezone.utc)
    ps, pe = lc.session_tf_window(ref, "premarket")
    assert ps == datetime(2026, 8, 28, 8, 0, tzinfo=CT).astimezone(timezone.utc)
    assert pe == datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    bs, be = lc.session_tf_window(ref, "bod")
    assert bs == datetime(2026, 8, 28, 0, 0, tzinfo=CT).astimezone(timezone.utc)
    assert be == datetime(2026, 8, 29, 0, 0, tzinfo=CT).astimezone(timezone.utc)
    # halt 16:30 CT still yesterday 17:00 → today 16:00
    halt = datetime(2026, 8, 28, 16, 30, tzinfo=CT)
    hs, he = lc.session_tf_window(halt, "full_session")
    assert hs == s and he == e
    # after 17:00 CT the next Globex session is open
    eve = datetime(2026, 8, 28, 17, 30, tzinfo=CT)
    ns, ne = lc.session_tf_window(eve, "full_session")
    assert ns == datetime(2026, 8, 28, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert ne == datetime(2026, 8, 29, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    # cash before 08:30 is previous day
    early = datetime(2026, 8, 28, 7, 0, tzinfo=CT)
    xs, xe = lc.session_tf_window(early, "cash")
    assert xs == datetime(2026, 8, 27, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    assert xe == datetime(2026, 8, 27, 15, 0, tzinfo=CT).astimezone(timezone.utc)


def test_custom_session_tf_windows():
    ref = datetime(2026, 8, 28, 15, 0, tzinfo=CT)
    s, e = lc.session_tf_window(ref, "custom", from_hm=(9, 0), to_hm=(11, 30))
    assert s == datetime(2026, 8, 28, 9, 0, tzinfo=CT).astimezone(timezone.utc)
    assert e == datetime(2026, 8, 28, 11, 30, tzinfo=CT).astimezone(timezone.utc)
    eve = datetime(2026, 8, 28, 18, 0, tzinfo=CT)
    ns, ne = lc.session_tf_window(eve, "custom", from_hm=(17, 0), to_hm=(16, 0))
    assert ns == datetime(2026, 8, 28, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    assert ne == datetime(2026, 8, 29, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    early = datetime(2026, 8, 28, 7, 0, tzinfo=CT)
    xs, xe = lc.session_tf_window(early, "custom", from_hm=(9, 0), to_hm=(11, 0))
    assert xs == datetime(2026, 8, 27, 9, 0, tzinfo=CT).astimezone(timezone.utc)
    assert xe == datetime(2026, 8, 27, 11, 0, tzinfo=CT).astimezone(timezone.utc)
    pinned = lc.session_tf_window_for_end_date(
        date(2026, 8, 20), "custom", from_hm=(8, 0), to_hm=(9, 30),
    )
    assert pinned[0] == datetime(2026, 8, 20, 8, 0, tzinfo=CT).astimezone(timezone.utc)
    assert pinned[1] == datetime(2026, 8, 20, 9, 30, tzinfo=CT).astimezone(timezone.utc)


def test_expand_single_vp_uses_session_date():
    end = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    spec = lc.parse_profiles(
        '[{"kind":"current","session_tf":"cash","session_date":"2026-08-20"}]'
    )
    jobs = lc.expand_vp_jobs(
        spec, ohlc=None, chart_start=end - timedelta(days=5), chart_end=end, now=end,
    )
    assert len(jobs) == 1
    assert jobs[0]["start"] == datetime(2026, 8, 20, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    assert jobs[0]["end"] == datetime(2026, 8, 20, 15, 0, tzinfo=CT).astimezone(timezone.utc)


def test_recurring_vps_keep_all_visible_dates_without_shared_limit():
    chart_start = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)
    chart_end = datetime(2026, 9, 7, 18, 0, tzinfo=timezone.utc)
    days = pd.bdate_range("2026-08-18", "2026-09-04")
    buckets = [
        datetime(day.year, day.month, day.day, 8, 35, tzinfo=CT).astimezone(timezone.utc)
        for day in days
    ]
    ohlc = pd.DataFrame({"bucket": buckets})
    specs = lc.parse_profiles(
        '[{"id":"vp1","n":1,"kind":"recurring","recur":"daily",'
        '"session_tf":"custom","session_from":"08:30","session_to":"08:45"},'
        '{"id":"vp2","n":2,"kind":"recurring","recur":"daily",'
        '"session_tf":"custom","session_from":"08:30","session_to":"08:45"}]'
    )
    jobs = lc.expand_vp_jobs(
        specs, ohlc=ohlc, chart_start=chart_start, chart_end=chart_end, now=chart_end,
    )
    assert len(jobs) == len(days) * 2
    assert {job["spec_id"] for job in jobs} == {"vp1", "vp2"}
    assert sum(job["spec_id"] == "vp1" for job in jobs) == len(days)
    assert sum(job["spec_id"] == "vp2" for job in jobs) == len(days)


def test_recurring_vp_limits_apply_per_profile(monkeypatch):
    chart_start = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)
    chart_end = datetime(2026, 9, 7, 18, 0, tzinfo=timezone.utc)
    days = pd.bdate_range("2026-08-18", "2026-09-04")
    ohlc = pd.DataFrame({"bucket": [
        datetime(day.year, day.month, day.day, 8, 35, tzinfo=CT).astimezone(timezone.utc)
        for day in days
    ]})
    specs = lc.parse_profiles(
        '[{"id":"vp1","kind":"recurring","recur":"daily","session_tf":"cash"},'
        '{"id":"vp2","kind":"recurring","recur":"daily","session_tf":"cash"}]'
    )
    monkeypatch.setattr(lc, "MAX_VP_EXPANDED", 3)
    jobs = lc.expand_vp_jobs(
        specs, ohlc=ohlc, chart_start=chart_start, chart_end=chart_end, now=chart_end,
    )
    assert len(jobs) == 6
    assert sum(job["spec_id"] == "vp1" for job in jobs) == 3
    assert sum(job["spec_id"] == "vp2" for job in jobs) == 3


def test_load_vbp_windows_keeps_all_requested_profile_windows(monkeypatch):
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    windows = [(start, start + timedelta(minutes=15)), (
        start + timedelta(days=30), start + timedelta(days=30, minutes=15),
    )]
    captured = {}

    def fake_query(sql, params, **kwargs):
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(lc, "_query", fake_query)
    monkeypatch.setattr(
        lc, "active_contracts_in_range", lambda *args: [(date(2026, 8, 1), "NQU6")],
    )
    lc.load_vbp_windows(windows)
    assert captured["params"][1] == start
    assert captured["params"][5] == windows[-1][1]


def test_load_vbp_windows_repeats_overlapping_tape_for_each_profile(monkeypatch):
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=15)
    captured = {}

    def fake_query(sql, params, **kwargs):
        captured["sql"] = sql
        return pd.DataFrame({
            "wid": [0, 1], "chi_day": [date(2026, 8, 1), date(2026, 8, 1)],
            "symbol": ["NQU6", "NQU6"], "px": [24000.0, 24000.0],
            "vol": [10.0, 10.0], "buy": [6.0, 6.0], "sell": [4.0, 4.0],
        })

    monkeypatch.setattr(lc, "_query", fake_query)
    monkeypatch.setattr(
        lc, "active_contracts_in_range", lambda *args: [(date(2026, 8, 1), "NQU6")],
    )
    frames = lc.load_vbp_windows([(start, end), (start, end)])
    assert "WITH vp_windows" in captured["sql"]
    assert "JOIN vp_windows" in captured["sql"]
    assert len(frames) == 2
    assert all(frame["vol"].sum() == 10.0 for frame in frames)


def test_iter_daily_cash_and_weekly_globex():
    start = datetime(2026, 8, 26, 12, 0, tzinfo=CT).astimezone(timezone.utc)  # Wed
    end = datetime(2026, 8, 28, 12, 0, tzinfo=CT).astimezone(timezone.utc)  # Fri
    days = lc.iter_daily_session_windows(start, end, "cash")
    assert len(days) == 3
    assert days[0][0] == datetime(2026, 8, 26, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    assert days[-1][1] == datetime(2026, 8, 28, 15, 0, tzinfo=CT).astimezone(timezone.utc)
    weeks = lc.iter_weekly_session_windows(start, end, "full_session")
    assert len(weeks) == 1
    ws, we, parts = weeks[0]
    assert ws == datetime(2026, 8, 23, 17, 0, tzinfo=CT).astimezone(timezone.utc)  # Sun
    assert we == datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)  # Fri
    assert parts == [(ws, we)]


def test_parse_profiles_json_and_caps():
    specs = lc.parse_profiles(
        '[{"kind":"current","session_tf":"cash","placement":"overlay",'
        '"overlay_side":"left","extend_lines":true}]'
    )
    assert specs[0]["kind"] == "current"
    assert specs[0]["session_tf"] == "cash"
    assert specs[0]["placement"] == "overlay"
    assert specs[0]["extend"] == 1
    assert specs[0]["extend_lines"] is True
    assert specs[0]["extend_va"] is True
    assert specs[0]["n"] == 1
    assert specs[0]["hist_width"] == 0.25
    assert specs[0]["chart_labels"] is False
    assert specs[0]["axis_labels"] is False
    configured_display = lc.parse_profiles(
        '[{"kind":"current","hist_width":2.0,"chart_labels":true,"axis_labels":true}]'
    )
    assert configured_display[0]["hist_width"] == 2.0
    assert configured_display[0]["chart_labels"] is True
    assert configured_display[0]["axis_labels"] is True
    with pytest.raises(ValueError, match="greater than zero"):
        lc.parse_profiles('[{"kind":"current","hist_width":0}]')
    va = lc.parse_profiles(
        '[{"kind":"current","session_tf":"cash","extend":2,"n":3}]'
    )
    assert va[0]["extend"] == 2
    assert va[0]["extend_va"] is True
    assert va[0]["n"] == 3
    allt = lc.parse_profiles('[{"kind":"current","extend":"all"}]')
    assert allt[0]["extend"] == 3
    off = lc.parse_profiles(
        '[{"kind":"current","extend_lines":false,"extend_va":false}]'
    )
    assert off[0]["extend"] == 0
    lvn = lc.parse_profiles(
        '[{"kind":"current","hist_style":"lvn_nodes","lvn_pct":30,"extend_lvn":true}]'
    )
    assert lvn[0]["hist_style"] == "binned"
    assert lvn[0]["show_lvn"] is False
    assert lvn[0]["lvn_pct"] == 0.30
    assert lvn[0]["extend_lvn"] is True
    grouped_lvn = lc.parse_profiles(
        '[{"kind":"current","show_lvn":true,"lvn_min_groups":2}]'
    )
    assert grouped_lvn[0]["lvn_min_groups"] == 2
    variable = lc.parse_profiles(
        '[{"kind":"current","plot_variable_bounds":true}]'
    )
    assert variable[0]["plot_variable_bounds"] is True
    binned = lc.parse_profiles(
        '[{"kind":"current","hist_style":"binned","group_ticks":16,"right_display":"buy","left_display":"delta","show_lvn":true}]'
    )
    assert binned[0]["hist_style"] == "binned"
    assert binned[0]["group_ticks"] == 16
    assert binned[0]["right_display"] == "buy"
    assert binned[0]["left_display"] == "delta"
    kde = lc.parse_profiles(
        '[{"kind":"current","hvn_method":"adaptive_kde","kde_bandwidth":6,'
        '"kde_prominence":15,"kde_volume":8,"kde_separation":3,"kde_max_zones":4}]'
    )
    assert kde[0]["hvn_method"] == "adaptive_kde"
    assert kde[0]["kde_bandwidth"] == 6.0
    assert kde[0]["kde_prominence"] == 0.15
    assert kde[0]["kde_valley_depth"] == 0.05
    assert kde[0]["kde_volume"] == 0.08
    assert kde[0]["kde_separation"] == 3
    assert kde[0]["kde_valley_width"] == 2
    assert kde[0]["kde_max_zones"] == 4
    assert binned[0]["right_opacity"] == 100.0
    assert binned[0]["left_opacity"] == 100.0
    opacity = lc.parse_profiles(
        '[{"kind":"current","right_opacity":35,"left_opacity":70}]'
    )
    assert opacity[0]["right_opacity"] == 35.0
    assert opacity[0]["left_opacity"] == 70.0
    with pytest.raises(ValueError, match="right_opacity"):
        lc.parse_profiles('[{"kind":"current","right_opacity":101}]')
    assert binned[0]["show_lvn"] is True
    with pytest.raises(ValueError, match="hist_style"):
        lc.parse_profiles('[{"kind":"current","hist_style":"heatmap"}]')
    custom = lc.parse_profiles(
        '{"kind":"custom","session_tf":"full_session","range_start":"2026-08-27T17:00:00",'
        '"range_end":"2026-08-28T16:00:00","placement":"right_frame"}'
    )
    assert custom[0]["kind"] == "current"
    assert custom[0]["session_tf"] == "custom"
    assert custom[0]["session_from"] == (17, 0)
    assert custom[0]["session_to"] == (16, 0)
    assert custom[0]["session_date"] == date(2026, 8, 28)
    dated = lc.parse_profiles(
        '{"kind":"current","session_tf":"cash","session_date":"2026-08-20"}'
    )
    assert dated[0]["kind"] == "current"
    assert dated[0]["session_date"] == date(2026, 8, 20)
    clocked = lc.parse_profiles(
        '{"kind":"recurring","session_tf":"custom","session_from":"09:00","session_to":"11:30"}'
    )
    assert clocked[0]["session_tf"] == "custom"
    assert clocked[0]["session_from"] == (9, 0)
    assert clocked[0]["session_to"] == (11, 30)
    with pytest.raises(ValueError, match="JSON"):
        lc.parse_profiles("not-json")
    with pytest.raises(ValueError, match="kind"):
        lc.parse_profiles('[{"kind":"nope"}]')
    too_many = [{"kind": "current"} for _ in range(9)]
    with pytest.raises(ValueError, match="at most 8"):
        lc.parse_profiles(__import__("json").dumps(too_many))


def test_parse_level_blend_profile():
    specs = lc.parse_profiles(
        '[{"kind":"level_blend","n":3,"placement":"left_frame","extend":3,'
        '"session_date":"2026-05-18","lookback_sessions":20,"hist_width":0.4}]'
    )
    assert specs[0]["kind"] == "level_blend"
    assert specs[0]["session_tf"] == "cash"
    assert specs[0]["session_date"] == date(2026, 5, 18)
    assert specs[0]["lookback_sessions"] == 20
    assert specs[0]["placement"] == "left_frame"
    assert specs[0]["extend"] == 3
    assert specs[0]["right_display"] == "none"
    assert specs[0]["left_display"] == "none"
    assert specs[0]["show_lvn"] is False
    assert specs[0]["hvn_method"] == "off"
    with pytest.raises(ValueError, match="lookback_sessions"):
        lc.parse_profiles('[{"kind":"level_blend","lookback_sessions":0}]')
    with pytest.raises(ValueError, match="lookback_sessions"):
        lc.parse_profiles('[{"kind":"level_blend","lookback_sessions":200}]')


def test_expand_level_blend_survives_recurring_limiter():
    """Level Blend is not recurring — the per-profile tape-span cap must not drop it."""
    buckets = pd.date_range("2026-05-18 13:30", periods=30, freq="1min", tz="UTC")
    ohlc = pd.DataFrame({
        "bucket": buckets,
        "open": 21000.0, "high": 21010.0, "low": 20990.0, "close": 21000.0,
        "volume": 1.0,
    })
    specs = lc.parse_profiles(
        '[{"id":"lb","n":1,"kind":"level_blend","placement":"right_frame",'
        '"session_date":"2026-05-18","extend":3}]'
    )
    jobs = lc.expand_vp_jobs(
        specs, ohlc=ohlc,
        chart_start=buckets[0].to_pydatetime(),
        chart_end=buckets[-1].to_pydatetime(),
        now=buckets[-1].to_pydatetime(),
    )
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "level_blend"
    assert jobs[0]["session_tf"] == "cash"


def test_level_blend_payload_from_session_ssot(monkeypatch):
    """Level Blend builds KDE bounds from session_levels without a VBP query."""
    import src.features.session_profile as sp

    rows = []
    for i, d in enumerate(pd.date_range("2026-04-01", periods=25, freq="B")):
        base = 20000.0 + i * 5.0
        rows.append({
            "session": pd.Timestamp(d.date()),
            "lo": base, "val": base + 10, "poc": base + 20,
            "vah": base + 30, "hi": base + 40,
            "open": base + 15, "close": base + 25,
            "volume": 1000.0, "delta": 10.0, "n_prints": 100,
            "va_width_pts": 20.0, "shape": 0.0, "r_value": 40.0,
        })
    levels = pd.DataFrame(rows)
    monkeypatch.setattr(sp, "load_levels", lambda root=None: levels)

    eval_day = date(2026, 5, 6)
    s, e = lc.session_tf_window_for_end_date(eval_day, "cash")
    # OHLC spans the newest prior session's mid so lookback stays at 20.
    prior = levels[levels["session"] < pd.Timestamp(eval_day)].iloc[-1]
    mid = 0.5 * (float(prior.lo) + float(prior.hi))
    buckets = pd.date_range(s, periods=20, freq="5min", tz="UTC")
    ohlc = pd.DataFrame({
        "bucket": buckets,
        "open": mid, "high": mid + 15, "low": mid - 15, "close": mid,
        "volume": 10.0,
    })
    job = {
        "id": "lb1", "n": 1, "kind": "level_blend", "session_tf": "cash",
        "placement": "overlay", "overlay_side": "left", "extend": 3,
        "session_date": eval_day, "lookback_sessions": 20,
        "start": s, "end": e, "win": [(s, e)],
        "hist_style": "volume", "group_ticks": 4,
        "right_display": "none", "left_display": "none",
        "right_opacity": 100, "left_opacity": 100, "hist_width": 0.35,
        "chart_labels": False, "axis_labels": False,
        "show_lvn": False, "extend_lvn": False, "hvn_method": "off",
        "lvn_pct": 0.3, "lvn_min_groups": 1,
    }
    payload = lc.level_blend_payload(job, ohlc=ohlc)
    assert payload["kind"] == "level_blend"
    assert payload["level_blend"] is True
    assert payload["bins"] == []
    assert payload["bounds"]
    assert payload["sessions_used"] >= 20
    assert payload["bounds"][0]["members"]
    member0 = payload["bounds"][0]["members"][0]
    assert "session" in member0
    assert member0["session"]  # YYYY-MM-DD of the closed cash session
    assert payload["bounds"][0]["lo_session"]
    assert payload["bounds"][0]["hi_session"]
    assert payload["high"] is not None and payload["low"] is not None
    assert payload["error"] is None


def test_chart_meta_includes_vp_catalog(client):
    r = client.get("/api/chart/meta")
    assert r.status_code == 200
    body = r.json()
    assert "full_session" in body["session_tfs"]
    assert "binned" in body["vp_hist_styles"]
    assert body["vp_hist_style_labels"]["binned"] == "Grouped ticks"
    assert body["vp_hist_style_labels"]["volume"] == "Tick histogram"
    assert "lvn_nodes" not in body["vp_hist_styles"]
    assert body["group_ticks"] == [1, 2, 4, 8, 16, 32, 64]
    assert body["group_ticks_default"] == 4
    assert body["vp_display_modes"] == ["total", "total_split", "total_delta_split", "buy", "sell", "delta", "none"]
    assert body["vp_display_mode_labels"]["total_delta_split"] == "Total Volume delta split"
    assert body["lvn_pct_default"] == 0.30
    assert body["hvn_methods"] == ["off", "adaptive_kde"]
    assert body["kde_defaults"]["bandwidth"] == 2.0
    assert body["lookback_custom"] is True
    assert "level_blend" in body["vp_kinds"]
    assert body["vp_kind_labels"]["level_blend"] == "Level Blend"
    assert body["level_blend_lookback_default"] == 20
    assert body["lookback_range"] is True
    assert body["vp_kind_labels"]["current"] == "Single"
    assert body["vp_kind_labels"]["recurring"] == "Recurring"
    assert body["vp_placement_labels"]["overlay"] == "Overlay"
    assert body["vp_extend"] == [0, 1, 2, 3]
    assert body["vp_extend_labels"]["1"] == "1 sess"
    assert body["vp_extend_labels"]["2"] == "2 sess"
    assert body["vp_extend_labels"]["3"] == "All sess"
    assert "custom" not in body["vp_kinds"]
    assert "custom" in body["session_tfs"]
    assert body["session_tf_labels"]["custom"] == "Custom"
    assert "overlay" in body["vp_placements"]
    cvd = next(o for o in body["overlays"] if o["key"] == "of_cvd")
    assert cvd["params"] == "cvd"
    assert cvd["pane"] == "osc"
    assert cvd["zero_line"] == "1"
    ofi = next(o for o in body["overlays"] if o["key"] == "of_ofi_cum")
    assert ofi["params"] == "ofi"


def test_series_ofi_origin_independent_of_cvd(monkeypatch):
    end = datetime(2026, 8, 28, 18, 0, tzinfo=timezone.utc)
    buckets = pd.date_range("2026-08-28 14:00", periods=4, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "bucket": buckets,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "bid_px": [100.0, 100.25, 100.25, 100.50],
        "ask_px": [100.25, 100.50, 100.50, 100.75],
        "bid_sz": [4.0, 6.0, 6.0, 7.0],
        "ask_sz": [3.0, 2.0, 2.0, 1.0],
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="of_cvd,of_ofi_cum",
        end=end.isoformat(), cvd_origin="session", ofi_origin="bod",
    )
    assert out["cvd_origin"] == "session"
    assert out["ofi_origin"] == "bod"
    assert out["overlays"]["of_cvd"]
    assert out["overlays"]["of_ofi_cum"]


def test_series_profiles_current_uses_session_tf(monkeypatch):
    end = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)  # 15:00 CT
    last = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bucket": [last],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": 6.0, "sell_volume": 4.0,
        "symbol": "NQU6",
    })
    captured = {}

    def fake_vbp(start, end_, **k):
        captured["start"] = start
        captured["end"] = end_
        return pd.DataFrame({"px": [23950.0], "vol": [100.0], "buy": [60.0], "sell": [40.0]})

    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_vbp", fake_vbp)
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    specs = '[{"id":"a","kind":"current","session_tf":"cash","placement":"overlay",' \
            '"overlay_side":"left","extend_lines":true}]'
    out = lc.series(
        timeframe="1min", lookback="5d", overlays="",
        end=end.isoformat(), profiles=specs,
    )
    assert captured["start"] == datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc)
    assert captured["end"] == datetime(2026, 8, 28, 15, 0, tzinfo=CT).astimezone(timezone.utc)
    assert out["profiles"]
    assert out["profiles"][0]["kind"] == "current"
    assert out["profiles"][0]["session_tf"] == "cash"
    assert out["profiles"][0]["placement"] == "overlay"
    assert out["profiles"][0]["poc"] == 23950.0
    assert out["svp"]["poc"] == 23950.0
    assert "nodes" in out["profiles"][0]
    assert out["profiles"][0]["hist_style"] == "volume"


def test_series_profiles_recurring_daily(monkeypatch):
    chart_end = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    last = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "bucket": [
            datetime(2026, 8, 27, 9, 0, tzinfo=CT).astimezone(timezone.utc),
            datetime(2026, 8, 28, 9, 0, tzinfo=CT).astimezone(timezone.utc),
            last,
        ],
        "open": [100.0, 100.0, 100.0], "high": [101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0], "close": [100.5, 100.5, 100.5],
        "volume": [10.0, 10.0, 10.0], "buy_volume": [6.0, 6.0, 6.0],
        "sell_volume": [4.0, 4.0, 4.0], "symbol": ["NQU6", "NQU6", "NQU6"],
    })
    windows = []

    def fake_windows(wins, **k):
        windows.extend(wins)
        return [
            pd.DataFrame({"px": [24000.0 + i], "vol": [10.0], "buy": [6.0], "sell": [4.0]})
            for i, _ in enumerate(wins)
        ]

    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_vbp_windows", fake_windows)
    monkeypatch.setattr(lc, "load_vbp", lambda *a, **k: pd.DataFrame(
        {"px": [1.0], "vol": [1.0], "buy": [1.0], "sell": [0.0]}
    ))
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    specs = '[{"id":"r","kind":"recurring","session_tf":"cash","placement":"overlay",' \
            '"overlay_side":"left","extend_lines":true,"recur":"daily"}]'
    out = lc.series(
        timeframe="1min", lookback="3d", overlays="",
        end=chart_end.isoformat(), profiles=specs,
    )
    assert len(out["profiles"]) >= 2
    assert all(p["kind"] == "recurring" for p in out["profiles"])
    assert all(p["placement"] == "overlay" for p in out["profiles"])
    assert windows
    last = out["profiles"][-1]
    t = int(last["start_unix"]) + 60
    covering = [p for p in out["profiles"]
                if int(p["start_unix"]) <= t < int(p["end_unix"])]
    assert covering == [last]
    first = out["profiles"][0]
    t0 = int(first["start_unix"]) + 60
    covering0 = [p for p in out["profiles"]
                 if int(p["start_unix"]) <= t0 < int(p["end_unix"])]
    assert covering0 == [first]


def test_chart_series_bad_profiles_is_400(client):
    r = client.get("/api/chart/series", params={"profiles": "nope"})
    assert r.status_code == 400


def test_globex_overnight_windows_are_5pm_to_830am():
    start = datetime(2026, 8, 20, 12, 0, tzinfo=CT).astimezone(timezone.utc)
    end = datetime(2026, 8, 21, 12, 0, tzinfo=CT).astimezone(timezone.utc)
    wins = lc.globex_overnight_windows(start, end)
    night = (
        datetime(2026, 8, 20, 17, 0, tzinfo=CT).astimezone(timezone.utc),
        datetime(2026, 8, 21, 8, 30, tzinfo=CT).astimezone(timezone.utc),
    )
    assert night in wins
    cash = datetime(2026, 8, 21, 10, 0, tzinfo=CT).astimezone(timezone.utc)
    assert all(not (a <= cash < b) for a, b in wins)


def test_non_rth_runs_keep_cash_clear_and_overnight_contiguous():
    """Shade must cover Globex night without painting the next cash session.

    The first Shade implementation walked civil dates with noon+36h, which
    skipped a calendar day: 8/19 08:30–15:00 CT was painted and
    8/20 17:00–8/21 08:30 CT was dropped. Runs are bar-contiguous instead.
    """
    t = datetime(2026, 8, 19, 7, 0, tzinfo=CT)
    end = datetime(2026, 8, 21, 10, 0, tzinfo=CT)
    times = []
    while t <= end:
        times.append(t.astimezone(timezone.utc))
        t += timedelta(minutes=30)
    idx = pd.DatetimeIndex(times)
    runs = lc.non_rth_runs(idx)

    def at(*hm, day=19):
        ts = datetime(2026, 8, day, hm[0], hm[1], tzinfo=CT).astimezone(timezone.utc)
        return int(idx.get_indexer([ts])[0])

    def covered(i):
        return any(a <= i < b for a, b in runs)

    assert not covered(at(8, 30, day=19))
    assert not covered(at(14, 0, day=19))
    assert covered(at(15, 0, day=19))
    assert covered(at(17, 0, day=20))
    assert covered(at(7, 0, day=21))
    assert not covered(at(8, 30, day=21))
    assert not covered(at(14, 0, day=21))
    night = [r for r in runs if r[0] <= at(17, 0, day=20) < r[1]]
    assert len(night) == 1
    a, b = night[0]
    assert a <= at(15, 0, day=20)
    assert b == at(8, 30, day=21)


def test_rth_bar_mask_cash_hours():
    idx = pd.DatetimeIndex([
        datetime(2026, 8, 28, 7, 0, tzinfo=CT),
        datetime(2026, 8, 28, 8, 29, tzinfo=CT),
        datetime(2026, 8, 28, 8, 30, tzinfo=CT),
        datetime(2026, 8, 28, 14, 59, tzinfo=CT),
        datetime(2026, 8, 28, 15, 0, tzinfo=CT),
        datetime(2026, 8, 28, 17, 0, tzinfo=CT),
    ]).tz_convert("UTC")
    mask = lc.rth_bar_mask(idx)
    assert list(mask) == [False, False, True, True, False, False]


def test_parse_bool_flags():
    assert lc.parse_bool(None, default=True) is True
    assert lc.parse_bool("0") is False
    assert lc.parse_bool("false") is False
    assert lc.parse_bool("1") is True
    assert lc.parse_bool("off", default=True) is False


def test_series_eth_aggregate_nulls_overnight_overlays(monkeypatch):
    end = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
    night = datetime(2026, 8, 28, 7, 0, tzinfo=CT).astimezone(timezone.utc)
    rth = datetime(2026, 8, 28, 10, 0, tzinfo=CT).astimezone(timezone.utc)
    df = pd.DataFrame({
        "bucket": [night, rth],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0, "buy_volume": [6.0, 8.0], "sell_volume": [4.0, 2.0],
        "symbol": "NQU6",
    })
    monkeypatch.setattr(lc, "load_ohlc", lambda *a, **k: df.copy())
    monkeypatch.setattr(lc, "load_ticks", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(lc, "_load_cagg", lambda *a, **k: pd.DataFrame())
    out = lc.series(
        timeframe="1min", lookback="4h", overlays="of_delta",
        end=end.isoformat(), eth_aggregate=False,
    )
    assert out["n"] == 2
    assert out["eth_aggregate"] is False
    assert out["overlays"]["of_delta"][0] is None
    assert out["overlays"]["of_delta"][1] == 6.0
    on = lc.series(
        timeframe="1min", lookback="4h", overlays="of_delta",
        end=end.isoformat(), eth_aggregate=True,
    )
    assert on["overlays"]["of_delta"][0] == 2.0
    assert on["overlays"]["of_delta"][1] == 6.0


def test_clip_wins_rth_intersects_cash():
    start = datetime(2026, 8, 27, 17, 0, tzinfo=CT).astimezone(timezone.utc)
    end = datetime(2026, 8, 28, 16, 0, tzinfo=CT).astimezone(timezone.utc)
    wins = lc._clip_wins_rth([(start, end)])
    assert wins == [(
        datetime(2026, 8, 28, 8, 30, tzinfo=CT).astimezone(timezone.utc),
        datetime(2026, 8, 28, 15, 0, tzinfo=CT).astimezone(timezone.utc),
    )]


def test_chart_meta_display_defaults(client):
    r = client.get("/api/chart/meta")
    assert r.status_code == 200
    d = r.json()["defaults"]
    assert d["eth_display"] is True
    assert d["eth_shade"] is False
    assert d["eth_aggregate"] is True
    assert d["text_size"] == 12


def test_live_chart_zoom_and_mbo_stats_chrome():
    js = (REPO / "src/webui/static/app.js").read_text()
    orders = (REPO / "src/webui/static/lc_orders.js").read_text()
    i_sel = js.index('id="lc-cursor-select"')
    i_in = js.index('id="lc-zoom-in"')
    i_out = js.index('id="lc-zoom-out"')
    i_note = js.index('id="lc-note-create"')
    assert i_sel < i_in < i_out < i_note
    assert 'id="lc-mbo-head"' in js
    assert 'if (cursorMode !== "select") clearNoteSelection();' in js
    assert "MBO stats" in orders
    assert "pulled in" in orders and "pulled away" in orders
    assert "Limit:" in orders and "day max" in orders
    assert 'data-act="legend"' in orders
    assert "Order Map" in orders
    assert "pulled near" not in orders
    assert "pulled far" not in orders
    assert "newest 48" not in orders
    assert "MBO through" not in orders
    assert 'addEventListener("dblclick", onDblClick, true)' in orders
    assert "/api/chart/mbo-order?" in orders
    assert "sorted by ts_event, sequence" in orders
    assert "closeDetail" in orders
    assert "params.symbol = sym" in orders
    assert "timeoutMs: 180000" in orders
    assert "busy — retrying…" in orders
    assert "loading order map…" in orders
    assert '"order map: " + err' in orders
    assert "isBusy" in orders
    assert 'chk("icebergs", "Icebergs"' in orders
    assert "icebergs: true" in orders
    assert "o.ice && !S.icebergs" in orders
    assert 'data-act="icebergs"' not in orders
    assert "lc-mbo-ice-btn" not in orders
    assert '"s"} hidden`' in orders
    assert "function labelPx" in orders
    assert "chartLabelSize" in orders
    assert "10px SF Mono" not in orders
    app_py = (REPO / "src/webui/app.py").read_text()
    assert '/api/chart/mbo-order' in app_py
    assert 'if ($("#lc-mbo-order") && !$("#lc-mbo-order").hidden) return;' in js
