"""External models: split SAR / EMA engines, book knobs, stacked consensus."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.common.cme_holidays import is_cme_holiday, is_trading_day
from src.external_models.catalog import (
    coerce_book_params,
    coerce_params,
    factory_book_params,
    factory_params,
    param_spec,
)
from src.external_models.pine_sar_ema import (
    _bracket_fill,
    consensus_signal,
    ema_signals,
    ema_steep_slope,
    psar_signals,
    simulate,
    tv_ema,
)
from src.features.psar import sar_on_closed
from src.external_models.chart import build_chart_payload
from src.external_models.runner import parse_source, parse_stack, run_backtest
from src.external_models.store import (
    delete_hardened,
    effective_book_params,
    effective_params,
    list_hardened,
    reset_book_defaults,
    reset_defaults,
    save_book_defaults,
    save_defaults,
    save_hardened,
)
from src.webui import datasources as ds
from src.webui.app import create_app
from src.webui.jobs import JobManager

H = {"X-VTech-Console": "1"}
CT = ZoneInfo("America/Chicago")


def _rth_1m(day="2026-06-02", start_ct="08:00", end_ct="15:00",
            close0=22000.0, step=0.5, range_pts=24.0):
    t0 = pd.Timestamp(f"{day} {start_ct}", tz=CT)
    t1 = pd.Timestamp(f"{day} {end_ct}", tz=CT)
    idx = pd.date_range(t0, t1, freq="1min", inclusive="left").tz_convert("UTC")
    n = len(idx)
    close = close0 + np.arange(n, dtype="float64") * step
    high = close + range_pts / 2.0
    low = close - range_pts / 2.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


def test_parse_source():
    assert parse_source("external:psar") == ("psar", None)
    assert parse_source("external:psar:hardened:tight-trail") == (
        "psar", "tight-trail")
    assert parse_source("external:psar_ema20") == ("psar_ema20", None)
    with pytest.raises(ValueError):
        parse_source("/opt/vtech/data/models/model_staging/foo")
    with pytest.raises(ValueError):
        parse_source("external:")
    with pytest.raises(ValueError):
        parse_source("external:stack:psar+ema20")


def test_parse_stack_compat_and_explicit():
    assert parse_stack("external:psar_ema20") == [
        {"model": "psar", "hardened": None, "params": None, "timeframe": None},
        {"model": "ema20", "hardened": None, "params": None, "timeframe": None},
    ]
    legs = parse_stack(None, [{"model": "psar"}, {"model": "ema20", "hardened": "steep"}])
    assert legs[0]["model"] == "psar"
    assert legs[1]["hardened"] == "steep"
    assert legs[0]["timeframe"] is None
    stacked = parse_stack("external:stack:psar+ema20:hardened:steep")
    assert stacked[0]["model"] == "psar" and stacked[1]["model"] == "ema20"
    assert stacked[1]["hardened"] == "steep"
    mixed = parse_stack(None, [
        {"model": "psar", "timeframe": "15min"},
        {"model": "zigzag", "timeframe": "5m"},
    ])
    assert mixed[0]["timeframe"] == "15min"
    assert mixed[1]["timeframe"] == "5min"
    src = parse_stack("external:stack:psar@15min+zigzag@5min")
    assert src[0]["timeframe"] == "15min" and src[1]["timeframe"] == "5min"
    hour_day = parse_stack("external:stack:psar@1h+ema20@1d")
    assert hour_day[0]["timeframe"] == "1hour" and hour_day[1]["timeframe"] == "1day"


def test_coerce_params_rejects_bad_af():
    with pytest.raises(ValueError, match="maximum"):
        coerce_params("psar", {"start": 0.2, "maximum": 0.1})
    with pytest.raises(ValueError, match="unknown"):
        coerce_params("psar", {"not_a_knob": 1})
    off = coerce_book_params({"profit_target_pts": 0, "stop_loss_pts": 0})
    assert off["profit_target_pts"] == 0.0
    assert off["stop_loss_pts"] == 0.0
    mixed = coerce_params("psar", {"start": 0.03, "min_atr": 1.0, "ema_length": 20})
    assert mixed["start"] == pytest.approx(0.03)
    assert "min_atr" not in mixed
    both = coerce_book_params({"long_only": True, "short_only": True})
    assert both["trade_long"] is True and both["trade_short"] is True
    lo = coerce_book_params({"long_only": True})
    assert lo["trade_long"] is True and lo["trade_short"] is False


def test_tv_ema_sma_seed():
    x = np.arange(10, dtype="float64")
    e = tv_ema(x, 3)
    assert np.isnan(e[0]) and np.isnan(e[1])
    assert e[2] == pytest.approx((0 + 1 + 2) / 3)
    alpha = 2.0 / 4.0
    assert e[3] == pytest.approx(alpha * 3 + (1 - alpha) * e[2])


def test_steep_slope_two_bars():
    ema = np.array([1.0, 2.0, 3.0, 2.5, 2.0], dtype="float64")
    up, dn = ema_steep_slope(ema, 2)
    assert bool(up[2]) is True
    assert bool(dn[4]) is True
    assert bool(up[3]) is False


def test_factory_split_and_book():
    p = factory_params("psar")
    assert "start" in p and "exit_at_sar" in p
    assert "min_atr" not in p
    assert p["exit_at_sar"] is False
    e = factory_params("ema20")
    assert e["ema_length"] == 20
    assert e["ema20_l_bars"] == 8
    book = factory_book_params()
    assert book["min_atr"] == pytest.approx(15.0)
    assert book["profit_target_pts"] == pytest.approx(50.0)
    spec_keys = {row["key"] for row in param_spec("psar")}
    assert spec_keys == {"start", "increment", "maximum", "exit_at_sar"}
    ema_keys = {row["key"] for row in param_spec("ema20")}
    assert ema_keys == {"ema_length", "ema20_l_bars", "ema20_s_bars"}
    z = factory_params("zigzag")
    assert z["percentage_reversal"] == pytest.approx(2.5)
    assert z["atr_length"] == 5
    zz_keys = {row["key"] for row in param_spec("zigzag")}
    assert "percentage_reversal" in zz_keys and "tick_size" in zz_keys
    t = factory_params("timed_psar")
    assert t["entry_mode"] == "clock"
    assert t["direction_tf"] == "1day"
    assert t["entry_time"] == "08:30"
    timed_keys = {row["key"] for row in param_spec("timed_psar")}
    assert timed_keys == {
        "entry_mode", "entry_time", "bars_from_open", "direction_tf",
        "start", "increment", "maximum",
    }
    from src.external_models.catalog import BOOK_PARAM_SPEC
    book_keys = {row["key"] for row in BOOK_PARAM_SPEC}
    assert "session_end" in book_keys
    assert "trade_start" in book_keys and "trade_end" in book_keys
    assert "trade_long" in book_keys
    label = next(r["label"] for r in BOOK_PARAM_SPEC if r["key"] == "session_end")
    assert label == "Session end (CT)"
    ts = next(r["label"] for r in BOOK_PARAM_SPEC if r["key"] == "trade_start")
    assert ts == "Trade start (CT)"
    atr = next(r for r in BOOK_PARAM_SPEC if r["key"] == "atr_period")
    assert atr["label"] == "ATR bars (wilders)"
    assert not atr.get("pine")
    pt = next(r for r in BOOK_PARAM_SPEC if r["key"] == "profit_target_pts")
    assert pt["type"] == "qty_unit" and pt["unit_key"] == "profit_target_unit"
    assert pt["group"] == "exits"
    assert "stop" in (pt.get("unit_choices") or [])
    assert book["trade_start"] == "08:30"
    assert book["min_atr_unit"] == "points"
    assert book["profit_target_unit"] == "points"
    assert coerce_params("psar", {"exit_at_sar": True})["exit_at_sar"] is True
    assert coerce_params("psar", {"exit_at_sar": "off"})["exit_at_sar"] is False


def test_bracket_zero_disables_each_side():
    assert _bracket_fill(+1, 100.0, 0.0, 0.0, o=100.0, h=200.0, l=0.0) is None
    tgt = _bracket_fill(+1, 100.0, 50.0, 0.0, o=100.0, h=160.0, l=80.0)
    assert tgt is not None and tgt[1] == "LG Profit Target"
    stp = _bracket_fill(+1, 100.0, 0.0, 10.0, o=100.0, h=160.0, l=80.0)
    assert stp is not None and stp[1] == "LG Stop Loss"


def test_bracket_sar_stop_fill_and_tighter_vs_oco():
    tagged = _bracket_fill(
        +1, 100.0, 0.0, 0.0, o=100.0, h=101.0, l=90.0, sar_stop=95.0,
    )
    assert tagged is not None
    assert tagged[0] == pytest.approx(95.0)
    assert tagged[1] == "sar_stop"
    gap = _bracket_fill(
        +1, 100.0, 0.0, 0.0, o=90.0, h=91.0, l=80.0, sar_stop=95.0,
    )
    assert gap is not None and gap[0] == pytest.approx(90.0)
    oco = _bracket_fill(
        +1, 100.0, 0.0, 2.0, o=100.0, h=101.0, l=80.0, sar_stop=95.0,
    )
    assert oco is not None
    assert oco[0] == pytest.approx(98.0)
    assert oco[1] == "LG Stop Loss"
    sar_first = _bracket_fill(
        +1, 100.0, 0.0, 10.0, o=100.0, h=101.0, l=80.0, sar_stop=95.0,
    )
    assert sar_first is not None
    assert sar_first[0] == pytest.approx(95.0)
    assert sar_first[1] == "sar_stop"
    short_hit = _bracket_fill(
        -1, 100.0, 0.0, 0.0, o=100.0, h=110.0, l=99.0, sar_stop=105.0,
    )
    assert short_hit is not None
    assert short_hit[0] == pytest.approx(105.0)
    assert short_hit[1] == "sar_stop"


def test_bracket_fill_stop_wins_when_both_tag():
    hit = _bracket_fill(+1, 100.0, 50.0, 10.0, o=100.0, h=160.0, l=80.0)
    assert hit is not None
    assert hit[1] == "LG Stop Loss"
    assert hit[0] == pytest.approx(90.0)
    hit_s = _bracket_fill(-1, 100.0, 50.0, 10.0, o=100.0, h=120.0, l=40.0)
    assert hit_s is not None
    assert hit_s[1] == "ST Stop Loss"


def test_simulate_psar_longs_and_flatten():
    ohlc = _rth_1m(step=1.0, range_pts=1.0)
    result = simulate(ohlc, {
        "min_atr": 0.25,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert result.n_bars == len(ohlc)
    assert not result.trades.empty
    assert (result.trades["side"] == "long").all()
    last = result.trades.iloc[-1]
    assert last["exit_reason"] == "flatten_1445"
    exit_local = pd.Timestamp(last["exit_ts"]).tz_convert(CT)
    assert exit_local.hour == 14 and exit_local.minute == 45


def test_simulate_profit_target_long():
    ohlc = _rth_1m(step=0.5, range_pts=1.0)
    result = simulate(ohlc, {
        "min_atr": 0.25,
        "profit_target_pts": 5.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert not result.trades.empty
    assert (result.trades["side"] == "long").all()
    assert (result.trades["exit_reason"] == "LG Profit Target").any()
    first = result.trades.iloc[0]
    assert first["exit_px"] - first["entry_px"] == pytest.approx(5.0, abs=0.26)


def test_simulate_stop_loss_long():
    up = _rth_1m(start_ct="08:00", end_ct="10:00", step=0.5, range_pts=1.0)
    t0 = pd.Timestamp("2026-06-02 10:00", tz=CT).tz_convert("UTC")
    last = float(up["close"].iloc[-1])
    crash = pd.DataFrame(
        {
            "open": [last],
            "high": [last + 0.25],
            "low": [last - 500.0],
            "close": [last - 400.0],
            "volume": [100.0],
        },
        index=pd.DatetimeIndex([t0]),
    )
    ohlc = pd.concat([up, crash])
    result = simulate(ohlc, {
        "min_atr": 0.25,
        "profit_target_pts": 0.0, "stop_loss_pts": 3.0,
    }, model_id="psar")
    assert not result.trades.empty
    assert (result.trades["exit_reason"] == "LG Stop Loss").any()


def _reversal_ohlc():
    """Rising RTH then a crash bar that tags a long SAR trail."""
    up = _rth_1m(start_ct="08:30", end_ct="10:00", step=2.0, range_pts=1.0)
    last = float(up["close"].iloc[-1])
    t0 = pd.Timestamp("2026-06-02 10:00", tz=CT).tz_convert("UTC")
    crash = pd.DataFrame(
        {
            "open": [last],
            "high": [last + 0.25],
            "low": [last - 80.0],
            "close": [last - 50.0],
            "volume": [100.0],
        },
        index=pd.DatetimeIndex([t0]),
    )
    rest = _rth_1m(
        start_ct="10:01", end_ct="15:00", close0=last - 50.0, step=-1.0, range_pts=1.0,
    )
    return pd.concat([up, crash, rest]), t0, last


def test_simulate_exit_at_sar_fills_trail_not_close():
    ohlc, crash_ts, last = _reversal_ohlc()
    common = {
        "min_atr": 0.25,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }
    at_close = simulate(ohlc, common, model_id="psar")
    at_sar = simulate(ohlc, {**common, "exit_at_sar": True}, model_id="psar")
    assert not at_close.trades.empty and not at_sar.trades.empty
    long_close = at_close.trades[at_close.trades["side"] == "long"].iloc[0]
    long_sar = at_sar.trades[at_sar.trades["side"] == "long"].iloc[0]
    assert pd.Timestamp(long_close["exit_ts"]) == crash_ts
    assert pd.Timestamp(long_sar["exit_ts"]) == crash_ts
    assert long_close["exit_px"] == pytest.approx(last - 50.0)
    assert long_sar["exit_reason"] == "sar_stop"
    sar = sar_on_closed(ohlc)
    crash_i = int(ohlc.index.get_loc(crash_ts))
    trail = float(sar["next_sar"].iloc[crash_i - 1])
    assert np.isfinite(trail)
    assert long_sar["exit_px"] == pytest.approx(trail)
    assert long_sar["exit_px"] != pytest.approx(long_close["exit_px"])


def test_simulate_long_only():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    result = simulate(ohlc, {
        "min_atr": 1.0, "trade_short": False,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert not result.trades.empty
    assert (result.trades["side"] == "long").all()


def test_simulate_short_only():
    ohlc = _rth_1m(step=-1.0, range_pts=30.0)
    result = simulate(ohlc, {
        "min_atr": 1.0, "trade_long": False,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert not result.trades.empty
    assert (result.trades["side"] == "short").all()


def test_simulate_signal_lock_blocks_same_signal():
    ohlc = _rth_1m(step=0.5, range_pts=1.0)
    common = {
        "min_atr": 0.25,
        "profit_target_pts": 5.0, "stop_loss_pts": 50.0,
    }
    unlocked = simulate(ohlc, common, model_id="psar")
    locked = simulate(ohlc, {**common, "signal_lock": True}, model_id="psar")
    assert not unlocked.trades.empty
    assert not locked.trades.empty
    assert len(locked.trades) < len(unlocked.trades)
    assert locked.skipped.get("signal_lock", 0) >= 1


def test_simulate_atr_gate_blocks():
    ohlc = _rth_1m(range_pts=1.0)
    result = simulate(ohlc, {"min_atr": 50.0}, model_id="psar")
    assert result.trades.empty


def test_consensus_and_opposing():
    idx = pd.date_range("2026-06-02", periods=4, freq="1min", tz="UTC")
    a = pd.DataFrame({"signal": np.array([1, 1, 1, -1], dtype="int8")}, index=idx)
    b = pd.DataFrame({"signal": np.array([1, 0, -1, -1], dtype="int8")}, index=idx)
    out = consensus_signal([a, b])
    assert list(out) == [1, 0, 0, -1]


def test_ema_signals_follow_slope():
    ohlc = _rth_1m(step=1.0, range_pts=2.0)
    frame = ema_signals(ohlc, {"ema_length": 5, "ema20_l_bars": 3, "ema20_s_bars": 2})
    assert (frame["signal"] >= 0).all()
    assert int(frame["signal"].iloc[-1]) == 1
    down = _rth_1m(step=-1.0, range_pts=2.0)
    dn = ema_signals(down, {"ema_length": 5, "ema20_l_bars": 3, "ema20_s_bars": 2})
    assert int(dn["signal"].iloc[-1]) == -1


def test_psar_signals_follow_trend():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    frame = psar_signals(ohlc, {})
    live = frame["signal"] != 0
    assert live.any()
    assert set(frame.loc[live, "signal"].unique()) <= {1, -1}


def test_stack_matching_enters_on_uptrend():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    result = simulate(ohlc, {
        "min_atr": 1.0, "ema_length": 5, "ema20_l_bars": 3, "ema20_s_bars": 2,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, stack=["psar", "ema20"])
    assert not result.trades.empty
    assert (result.trades["side"] == "long").all()
    assert result.trades.iloc[-1]["exit_reason"] == "flatten_1445"


def test_stack_opposing_exits_on_crash():
    ohlc, crash_ts, last = _reversal_ohlc()
    common = {
        "min_atr": 0.25, "ema_length": 20, "ema20_l_bars": 8, "ema20_s_bars": 2,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }
    stacked = simulate(ohlc, common, stack=["psar", "ema20"])
    at_sar = simulate(ohlc, {**common, "exit_at_sar": True}, stack=["psar", "ema20"])
    assert not stacked.trades.empty and not at_sar.trades.empty
    first = stacked.trades.iloc[0]
    first_sar = at_sar.trades.iloc[0]
    assert first["side"] == "long" and first_sar["side"] == "long"
    assert pd.Timestamp(first["exit_ts"]) == crash_ts
    assert first["exit_reason"] in {"stack_disagree", "stack_reverse", "sar_dn_close"}
    assert pd.Timestamp(first_sar["exit_ts"]) == crash_ts
    assert first_sar["exit_reason"] == "sar_stop"
    assert first_sar["exit_px"] != pytest.approx(last - 50.0)


def test_store_defaults_and_hardened(tmp_path, monkeypatch):
    monkeypatch.setattr("src.external_models.store.store_root", lambda: tmp_path / "ext")
    factory = factory_params("psar")
    assert effective_params("psar")["start"] == factory["start"]
    rec = save_defaults("psar", {**factory, "start": 0.03}, note="tweak")
    assert rec["params"]["start"] == pytest.approx(0.03)
    assert effective_params("psar")["start"] == pytest.approx(0.03)
    hard = save_hardened("psar", "tight-trail", {**factory, "maximum": 0.1})
    assert hard["name"] == "tight-trail"
    names = [h["name"] for h in list_hardened("psar")]
    assert "tight-trail" in names
    reset_defaults("psar")
    assert effective_params("psar")["start"] == factory["start"]
    delete_hardened("psar", "tight-trail")
    assert list_hardened("psar") == []
    book = save_book_defaults({**factory_book_params(), "min_atr": 12.5})
    assert book["params"]["min_atr"] == pytest.approx(12.5)
    assert effective_book_params()["min_atr"] == pytest.approx(12.5)
    reset_book_defaults()
    assert effective_book_params()["min_atr"] == pytest.approx(15.0)


def test_run_backtest_payload_from_ohlc():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    payload = run_backtest(
        "external:psar",
        start="2026-06-02", end="2026-06-02",
        params={
            "min_atr": 1.0,
            "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
        },
        ohlc=ohlc,
    )
    assert payload["config"]["source_kind"] == "external"
    assert payload["config"]["candle_table"] == "databento.tbbo_1min"
    assert payload["config"]["bar_seconds"] == 60
    assert payload["config"]["point_value"] == pytest.approx(20.0)
    assert payload["summary"]["n_trades"] >= 1
    assert payload["equity_curve"]
    assert payload["trades"][0]["side"] in ("long", "short")
    assert "window" in payload["config"]
    assert payload["config"]["stack"][0]["model"] == "psar"


def test_run_backtest_stack_payload():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    payload = run_backtest(
        stack=[{"model": "psar"}, {"model": "ema20"}],
        start="2026-06-02", end="2026-06-02",
        book={"min_atr": 1.0, "profit_target_pts": 0.0, "stop_loss_pts": 0.0},
        params={"ema_length": 5, "ema20_l_bars": 3, "ema20_s_bars": 2},
        ohlc=ohlc,
    )
    assert payload["config"]["source"].startswith("external:stack:")
    assert [x["model"] for x in payload["config"]["stack"]] == ["psar", "ema20"]
    assert payload["summary"]["n_trades"] >= 1


def test_chart_payload_session_and_sar():
    ohlc = _rth_1m(start_ct="08:00", end_ct="15:00", step=0.5, range_pts=4.0)
    full = build_chart_payload(
        source="external:psar",
        start="2026-06-02", end="2026-06-02",
        session_only=False, ohlc=ohlc,
    )
    sess = build_chart_payload(
        source="external:psar",
        start="2026-06-02", end="2026-06-02",
        session_only=True, ohlc=ohlc,
    )
    assert full["n"] == len(ohlc)
    assert sess["n"] < full["n"]
    assert sess["n"] > 50
    assert sess["table"] == "databento.tbbo_1min"
    assert len(sess["t"]) == sess["n"] == len(sess["sar"]) == len(sess["c"]) == len(sess["atr"])
    assert any(v is not None for v in sess["sar"])
    assert any(v is not None for v in sess["atr"])
    assert sess["atr_period"] >= 1
    assert sess["min_atr"] is None or sess["min_atr"] > 0
    assert sess["t"] == sorted(sess["t"])
    assert sess["holidays_omitted"] is True
    assert full["holidays_omitted"] is True


def _ct_date(unix: int) -> date:
    return datetime.fromtimestamp(int(unix), tz=timezone.utc).astimezone(CT).date()


def test_chart_payload_drops_cme_holidays():
    """Display never includes CME holiday Chicago dates; overlays still run on them."""
    assert is_cme_holiday(date(2026, 7, 3)) is True
    assert is_cme_holiday(date(2026, 7, 2)) is False
    assert is_trading_day(date(2026, 7, 2)) is True
    open_day = _rth_1m(day="2026-07-02", start_ct="08:00", end_ct="15:00")
    hol_day = _rth_1m(day="2026-07-03", start_ct="08:00", end_ct="12:00")
    ohlc = pd.concat([open_day, hol_day]).sort_index()
    sess = build_chart_payload(
        source="external:psar",
        start="2026-07-02", end="2026-07-03",
        session_only=True, ohlc=ohlc,
    )
    full = build_chart_payload(
        source="external:psar",
        start="2026-07-02", end="2026-07-03",
        session_only=False, ohlc=ohlc,
    )
    assert sess["n"] > 0
    assert full["n"] == len(open_day)
    assert { _ct_date(t) for t in sess["t"] } == {date(2026, 7, 2)}
    assert { _ct_date(t) for t in full["t"] } == {date(2026, 7, 2)}
    empty = build_chart_payload(
        source="external:psar",
        start="2026-07-03", end="2026-07-03",
        session_only=True, ohlc=hol_day,
    )
    assert empty["n"] == 0
    assert empty["t"] == []


def test_run_backtest_window_counts_trading_days_only():
    ohlc = pd.concat([
        _rth_1m(day="2026-07-01"),
        _rth_1m(day="2026-07-02"),
        _rth_1m(day="2026-07-03"),
    ]).sort_index()
    payload = run_backtest(
        source="external:psar",
        start="2026-07-01",
        end="2026-07-03",
        book={"min_atr": 0.0, "profit_target_pts": 0.0, "stop_loss_pts": 0.0},
        ohlc=ohlc,
    )
    # Thu 7/2 is a regular session; Fri 7/3 is the Independence early-close holiday.
    assert payload["config"]["window"]["n_sessions"] == 2
    for t in payload["trades"]:
        entry = pd.Timestamp(t["entry_ts"]).tz_convert(CT).date()
        assert entry != date(2026, 7, 3)


def test_chart_payload_atr_uses_book_period():
    ohlc = _rth_1m(start_ct="08:00", end_ct="15:00", step=0.5, range_pts=4.0)
    sess = build_chart_payload(
        source="external:psar",
        start="2026-06-02", end="2026-06-02",
        book={"atr_period": 5, "min_atr": 10.0},
        session_only=True, ohlc=ohlc,
    )
    assert sess["atr_period"] == 5
    assert sess["min_atr"] == pytest.approx(10.0)
    finite = [v for v in sess["atr"] if v is not None]
    assert len(finite) > 50
    assert min(finite) > 0


def test_chart_payload_ema_overlay():
    ohlc = _rth_1m(start_ct="08:00", end_ct="15:00", step=0.5, range_pts=4.0)
    sess = build_chart_payload(
        source="external:ema20",
        start="2026-06-02", end="2026-06-02",
        session_only=True, ohlc=ohlc,
    )
    assert sess["has_ema"] is True
    assert any(v is not None for v in sess["ema"])
    assert len(sess["atr"]) == sess["n"]
    assert any(v is not None for v in sess["atr"])


def _boom_span(*_a, **_k):
    raise AssertionError("catalog list/describe must not query tbbo_1min")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "SCOPING_ROOT", tmp_path / "scoping")
    monkeypatch.setattr(ds, "SEARCH_ROOT", tmp_path / "search")
    monkeypatch.setattr("src.external_models.store.store_root", lambda: tmp_path / "ext")
    monkeypatch.setattr("src.external_models.runner.candle_span", _boom_span)
    monkeypatch.setattr("src.external_models.candles.candle_span", _boom_span)
    jm = JobManager(root=tmp_path / "jobs")
    with TestClient(create_app(jobs=jm, enable_live_worker=False),
                    base_url="http://127.0.0.1") as c:
        yield c


def test_api_list_and_save_defaults(client):
    payload = client.get("/api/external-models").json()
    rows = payload["models"]
    ids = {r["id"] for r in rows}
    assert "psar" in ids and "ema20" in ids and "zigzag" in ids
    assert "timed_psar" in ids
    assert "psar_ema20" not in ids
    listed = next(r for r in rows if r["id"] == "psar")
    assert listed.get("span") in (None, {})
    assert listed["candle_table"] == "databento.tbbo_1min"
    assert listed["param_spec"]
    assert any(s.get("key") == "exit_at_sar" and s.get("type") == "bool"
               for s in listed["param_spec"])
    assert not any(s.get("key") == "min_atr" for s in listed["param_spec"])
    assert listed["defaults"]["exit_at_sar"] is False
    book = payload["book"]
    assert any(s.get("key") == "min_atr" for s in book["param_spec"])
    assert any(s.get("key") == "trade_start" and s.get("label") == "Trade start (CT)"
               for s in book["param_spec"])
    assert any(s.get("key") == "session_end" and s.get("label") == "Session end (CT)"
               for s in book["param_spec"])
    assert book["timeframes"] == ["1min", "5min", "15min", "1hour", "1day"]
    assert book["defaults"]["trade_end"] == "14:45"
    atr = next(s for s in book["param_spec"] if s.get("key") == "atr_period")
    assert atr["label"] == "ATR bars (wilders)"
    assert not atr.get("pine")
    assert book["defaults"]["min_atr"] == pytest.approx(15.0)
    card = client.get("/api/external-models/psar").json()
    assert card["kind"] == "pine_replica"
    assert card["engine"] == "psar"
    body = {**card["factory_params"], "start": 0.04}
    saved = client.put(
        "/api/external-models/psar/defaults",
        headers=H, json={"params": body, "note": "ui"},
    )
    assert saved.status_code == 200
    assert saved.json()["params"]["start"] == pytest.approx(0.04)
    again = client.get("/api/external-models/psar").json()
    assert again["defaults"]["start"] == pytest.approx(0.04)
    reset = client.post("/api/external-models/psar/defaults/reset", headers=H, json={})
    assert reset.status_code == 200
    assert reset.json()["params"]["start"] == pytest.approx(0.02)
    book_saved = client.put(
        "/api/external-models/book/defaults",
        headers=H, json={"params": {**book["factory_params"], "min_atr": 11.0}},
    )
    assert book_saved.status_code == 200
    assert book_saved.json()["params"]["min_atr"] == pytest.approx(11.0)
    book_again = client.get("/api/external-models/book").json()
    assert book_again["defaults"]["min_atr"] == pytest.approx(11.0)
    book_reset = client.post("/api/external-models/book/defaults/reset", headers=H, json={})
    assert book_reset.status_code == 200
    assert book_reset.json()["params"]["min_atr"] == pytest.approx(15.0)


def test_api_hardened_and_backtest_listing(client):
    card = client.get("/api/external-models/psar").json()
    r = client.post(
        "/api/external-models/psar/hardened",
        headers=H,
        json={"name": "tight-trail", "params": {**card["factory_params"], "maximum": 0.1}},
    )
    assert r.status_code == 201
    listed = client.get("/api/external-models/psar/hardened").json()
    assert any(x["name"] == "tight-trail" for x in listed)
    models = client.get("/api/backtest/models").json()
    assert not any(m.get("source_kind") == "external" for m in models)
    catalog = client.get("/api/external-models").json()["models"]
    psar = next(m for m in catalog if m["id"] == "psar")
    assert any(h["name"] == "tight-trail" for h in psar["hardened"])
    gone = client.delete(
        "/api/external-models/psar/hardened/tight-trail", headers=H)
    assert gone.status_code == 200


def test_api_backtest_run_external(client, monkeypatch):
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    monkeypatch.setattr(
        "src.external_models.runner.load_lead_ohlc",
        lambda *a, **k: ohlc,
    )
    r = client.post("/api/backtest/run", headers=H, json={
        "source": "external:psar",
        "start": "2026-06-02",
        "end": "2026-06-02",
        "params": {"min_atr": 1.0, "profit_target_pts": 0, "stop_loss_pts": 0},
    })
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["config"]["source_kind"] == "external"
    assert p["summary"]["n_trades"] >= 1
    assert p["trades"][0]["exit_reason"]


def test_api_backtest_run_stack(client, monkeypatch):
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    monkeypatch.setattr(
        "src.external_models.runner.load_lead_ohlc",
        lambda *a, **k: ohlc,
    )
    r = client.post("/api/backtest/run", headers=H, json={
        "stack": [{"model": "psar"}, {"model": "ema20"}],
        "book": {"min_atr": 1.0, "profit_target_pts": 0, "stop_loss_pts": 0},
        "params": {"ema_length": 5, "ema20_l_bars": 3, "ema20_s_bars": 2},
        "start": "2026-06-02",
        "end": "2026-06-02",
    })
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["config"]["source_kind"] == "external"
    assert [x["model"] for x in p["config"]["stack"]] == ["psar", "ema20"]
    assert p["summary"]["n_trades"] >= 1


def test_api_backtest_chart(client, monkeypatch):
    ohlc = _rth_1m(start_ct="08:00", end_ct="15:00")
    monkeypatch.setattr(
        "src.external_models.chart.load_lead_ohlc",
        lambda *a, **k: ohlc,
    )
    r = client.post("/api/backtest/chart", headers=H, json={
        "source": "external:psar",
        "start": "2026-06-02",
        "end": "2026-06-02",
        "session_only": True,
    })
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n"] == len(d["t"]) == len(d["sar"]) == len(d["o"]) == len(d["atr"])
    assert d["n"] > 50
    assert d["session_only"] is True
    assert d["holidays_omitted"] is True
    assert any(v is not None for v in d["sar"])
    assert any(v is not None for v in d["atr"])
    assert d["atr_period"] == 8


def test_api_csrf_blocks_save(client):
    r = client.put(
        "/api/external-models/psar/defaults",
        json={"params": factory_params("psar")},
    )
    assert r.status_code == 403


def test_query_uses_session_statement_timeout(monkeypatch):
    from src.external_models import candles as cmod

    calls: list[Any] = []

    class _Cur:
        description = [("x",)]

        def execute(self, sql, params=None):
            calls.append(("execute", sql))

        def fetchall(self):
            return [(1,)]

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _connect(*_a, **kwargs):
        calls.append(("connect", kwargs))
        return _Conn()

    monkeypatch.setattr(cmod.psycopg, "connect", _connect)
    monkeypatch.setattr(cmod, "get_dsn", lambda: "postgresql://x")
    out = cmod._query("SELECT 1", (), timeout_ms=8000)
    assert list(out["x"]) == [1]
    conn_kw = next(k for kind, k in calls if kind == "connect")
    assert conn_kw["connect_timeout"] == 5
    assert "statement_timeout=8000" in conn_kw["options"]
    assert not any("set_config" in str(sql) for kind, sql in calls if kind == "execute")


def test_candle_span_skip_scan_not_count(monkeypatch):
    from src.external_models import candles as cmod

    cmod.clear_span_cache()
    captured: dict[str, Any] = {}

    def fake_query(sql, params, timeout_ms=0):
        captured["sql"] = sql
        captured["params"] = params
        captured["timeout_ms"] = timeout_ms
        return pd.DataFrame([{
            "lo": pd.Timestamp("2025-04-01T00:00:00Z"),
            "hi": pd.Timestamp("2026-08-26T20:00:00Z"),
        }])

    monkeypatch.setattr(cmod, "_query", fake_query)
    monkeypatch.setattr(cmod, "lead_symbols", lambda *_a, **_k: ["NQM6", "NQU6"])
    span = cmod.candle_span("NQ")
    assert span["available"] is True
    assert span["start"] == "2025-03-31"
    assert span["end"] == "2026-08-26"
    assert span["n_rows"] is None
    assert "COUNT(" not in captured["sql"].upper()
    assert "ORDER BY bucket" in captured["sql"]
    assert "LIMIT 1" in captured["sql"]
    assert captured["params"] == (["NQM6", "NQU6"],)
    again = cmod.candle_span("NQ")
    assert again["available"] is True
    cmod.clear_span_cache()


def test_coerce_trade_window_and_units():
    p = coerce_book_params({
        "session_start": "08:30", "session_end": "14:45",
        "trade_start": "09:00", "trade_end": "11:00",
        "profit_target_pts": 2.0, "profit_target_unit": "atr",
        "stop_loss_pts": 1.0, "stop_loss_unit": "atr",
        "min_atr": 0.5, "min_atr_unit": "atr",
    })
    assert p["trade_start"] == "09:00" and p["trade_end"] == "11:00"
    assert p["profit_target_unit"] == "atr"
    assert p["min_atr_unit"] == "atr"
    stop_u = coerce_book_params({"profit_target_unit": "stop", "profit_target_pts": 2.0})
    assert stop_u["profit_target_unit"] == "stop"
    with pytest.raises(ValueError, match="profit_target_unit"):
        coerce_book_params({"profit_target_unit": "ticks"})
    with pytest.raises(ValueError, match="trade_end must be after trade_start"):
        coerce_book_params({"trade_start": "11:00", "trade_end": "09:00"})
    with pytest.raises(ValueError, match="trade_start must be at or after"):
        coerce_book_params({"trade_start": "08:00", "session_start": "08:30"})
    with pytest.raises(ValueError, match="trade_end must be at or before"):
        coerce_book_params({"trade_end": "15:00", "session_end": "14:45"})


def test_simulate_trade_window_blocks_new_entries_not_exits():
    ohlc = _rth_1m(step=1.0, range_pts=1.0)
    result = simulate(ohlc, {
        "min_atr": 0.25,
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
        "trade_start": "08:30", "trade_end": "10:00",
        "session_start": "08:30", "session_end": "14:45",
    }, model_id="psar")
    assert not result.trades.empty
    entries = pd.to_datetime(result.trades["entry_ts"], utc=True)
    if entries.dt.tz is None:
        entries = entries.dt.tz_localize("UTC")
    local = entries.dt.tz_convert(CT)
    assert (local.dt.hour * 60 + local.dt.minute < 10 * 60).all()
    last = result.trades.iloc[-1]
    exit_local = pd.Timestamp(last["exit_ts"]).tz_convert(CT)
    assert last["exit_reason"] == "flatten_1445"
    assert exit_local.hour == 14 and exit_local.minute == 45


def test_simulate_profit_target_atr_multiple():
    ohlc = _rth_1m(step=0.5, range_pts=1.0)
    result = simulate(ohlc, {
        "min_atr": 0.25,
        "atr_period": 8,
        "profit_target_pts": 2.0, "profit_target_unit": "atr",
        "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert not result.trades.empty
    assert (result.trades["exit_reason"] == "LG Profit Target").any()
    first = result.trades.iloc[0]
    assert first["exit_px"] - first["entry_px"] > 1.0


def test_simulate_min_atr_true_range_multiple_blocks():
    ohlc = _rth_1m(range_pts=1.0)
    blocked = simulate(ohlc, {
        "min_atr": 50.0, "min_atr_unit": "atr",
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert blocked.trades.empty
    open_ = simulate(ohlc, {
        "min_atr": 0.0, "min_atr_unit": "atr",
        "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
    }, model_id="psar")
    assert not open_.trades.empty


def test_align_slower_tf_is_causal():
    from src.external_models.timeframe import align_signal_frame, resample_ohlc

    ohlc = _rth_1m(step=1.0, range_pts=1.0)
    src = resample_ohlc(ohlc, "15min")
    assert len(src) < len(ohlc)
    frame = pd.DataFrame(
        {"signal": np.arange(len(src), dtype="int8"), "tagged": np.zeros(len(src), dtype=bool)},
        index=src.index,
    )
    aligned = align_signal_frame(frame, ohlc.index, "15min", "1min")
    t = src.index[4]
    avail = t + pd.Timedelta(minutes=14)
    before = t + pd.Timedelta(minutes=13)
    assert int(aligned.loc[avail, "signal"]) == int(frame.loc[t, "signal"])
    assert int(aligned.loc[before, "signal"]) != int(frame.loc[t, "signal"])
    assert bool(aligned.loc[avail, "signal_fresh"]) is True
    assert bool(aligned.loc[before, "signal_fresh"]) is False


def test_coerce_hour_and_day_aliases():
    from src.external_models.timeframe import (
        coerce_candle_tf, tf_seconds, warmup_days_for,
    )

    assert coerce_candle_tf("1h") == "1hour"
    assert coerce_candle_tf("60min") == "1hour"
    assert coerce_candle_tf("1d") == "1day"
    assert coerce_candle_tf("1day") == "1day"
    assert tf_seconds("1hour") == 3600
    assert tf_seconds("1day") == 86400
    assert warmup_days_for("1min") == 2
    assert warmup_days_for("1hour") == 10
    assert warmup_days_for("1min", "1day") == 40
    with pytest.raises(ValueError, match="candle timeframe"):
        coerce_candle_tf("30min")


def test_resample_hour_and_chicago_day():
    from src.external_models.timeframe import resample_ohlc

    ohlc = pd.concat([
        _rth_1m(day="2026-06-02"),
        _rth_1m(day="2026-06-03"),
    ]).sort_index()
    hourly = resample_ohlc(ohlc, "1hour")
    daily = resample_ohlc(ohlc, "1day")
    assert len(hourly) == 14  # 08:00–14:00 CT × 2 sessions
    assert len(daily) == 2
    day0 = daily.index[0].tz_convert(CT)
    assert day0.hour == 8 and day0.minute == 30
    assert str(day0.date()) == "2026-06-02"
    h0 = hourly.index[0].tz_convert(CT)
    assert h0.hour == 8 and h0.minute == 0


def test_align_hour_and_day_are_causal():
    from src.external_models.timeframe import align_signal_frame, resample_ohlc

    ohlc = pd.concat([
        _rth_1m(day="2026-06-02"),
        _rth_1m(day="2026-06-03"),
    ]).sort_index()
    hour = resample_ohlc(ohlc, "1hour")
    hframe = pd.DataFrame(
        {"signal": np.arange(len(hour), dtype="int8") + 3,
         "tagged": np.zeros(len(hour), dtype=bool)},
        index=hour.index,
    )
    h_aligned = align_signal_frame(hframe, ohlc.index, "1hour", "1min")
    t_h = hour.index[3]
    avail_h = t_h + pd.Timedelta(minutes=59)
    before_h = t_h + pd.Timedelta(minutes=58)
    assert int(h_aligned.loc[avail_h, "signal"]) == int(hframe.loc[t_h, "signal"])
    assert int(h_aligned.loc[before_h, "signal"]) != int(hframe.loc[t_h, "signal"])
    assert bool(h_aligned.loc[avail_h, "signal_fresh"]) is True
    assert bool(h_aligned.loc[before_h, "signal_fresh"]) is False

    daily = resample_ohlc(ohlc, "1day")
    dframe = pd.DataFrame(
        {"signal": np.arange(len(daily), dtype="int8") + 3,
         "tagged": np.zeros(len(daily), dtype=bool)},
        index=daily.index,
    )
    d_aligned = align_signal_frame(dframe, ohlc.index, "1day", "1min")
    t_d = daily.index[0]
    avail_d = t_d + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    before_d = avail_d - pd.Timedelta(minutes=1)
    assert int(d_aligned.loc[avail_d, "signal"]) == int(dframe.loc[t_d, "signal"])
    assert int(d_aligned.loc[before_d, "signal"]) != int(dframe.loc[t_d, "signal"])
    assert bool(d_aligned.loc[avail_d, "signal_fresh"]) is True
    assert bool(d_aligned.loc[before_d, "signal_fresh"]) is False


def test_run_backtest_hour_engine():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    payload = run_backtest(
        stack=[{"model": "psar", "timeframe": "1hour"}],
        start="2026-06-02", end="2026-06-02",
        book={
            "timeframe": "1min", "min_atr": 1.0,
            "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
        },
        ohlc=ohlc,
    )
    assert payload["config"]["stack"][0]["timeframe"] == "1hour"
    assert payload["config"]["bar_seconds"] == 60


def test_run_backtest_mixed_engine_timeframe():
    ohlc = _rth_1m(step=1.0, range_pts=30.0)
    payload = run_backtest(
        stack=[{"model": "psar", "timeframe": "15min"}],
        start="2026-06-02", end="2026-06-02",
        book={
            "timeframe": "1min", "min_atr": 1.0,
            "profit_target_pts": 0.0, "stop_loss_pts": 0.0,
        },
        ohlc=ohlc,
    )
    assert payload["config"]["timeframe"] == "1min"
    assert payload["config"]["stack"][0]["timeframe"] == "15min"
    assert payload["config"]["bar_seconds"] == 60


def _rth_days(days, *, start_ct="07:00", end_ct="15:00",
              close0=22000.0, step=0.5, range_pts=4.0):
    frames = []
    px = float(close0)
    for day in days:
        part = _rth_1m(
            day=day, start_ct=start_ct, end_ct=end_ct,
            close0=px, step=step, range_pts=range_pts,
        )
        frames.append(part)
        px = float(part["close"].iloc[-1]) + float(step)
    return pd.concat(frames).sort_index()


def _timed_params(**extra):
    p = {
        "min_atr": 0.0,
        "profit_target_pts": 0.0,
        "stop_loss_pts": 0.0,
        "entry_mode": "clock",
        "entry_time": "08:30",
        "direction_tf": "1hour",
        "bars_from_open": 0,
    }
    p.update(extra)
    return p


def test_timed_psar_clock_entry_follows_hour_sar():
    days = ["2026-06-01", "2026-06-02", "2026-06-03"]
    up = _rth_days(days, step=0.5)
    long_r = simulate(up, _timed_params(), model_id="timed_psar")
    assert not long_r.trades.empty
    last_day = long_r.trades[long_r.trades["session_date"] == "2026-06-03"]
    assert not last_day.empty
    assert (last_day["side"] == "long").all()
    entry = pd.Timestamp(last_day.iloc[0]["entry_ts"]).tz_convert(CT)
    assert entry.hour == 8 and entry.minute == 30

    dn = _rth_days(days, step=-0.5)
    short_r = simulate(dn, _timed_params(), model_id="timed_psar")
    last_short = short_r.trades[short_r.trades["session_date"] == "2026-06-03"]
    assert not last_short.empty
    assert (last_short["side"] == "short").all()


def test_timed_psar_bars_from_open_and_once_per_session():
    ohlc = _rth_days(["2026-06-02", "2026-06-03"], step=0.5, range_pts=2.0)
    result = simulate(ohlc, _timed_params(
        entry_mode="bars_from_open",
        bars_from_open=60,
        profit_target_pts=1.0,
        stop_loss_pts=50.0,
    ), model_id="timed_psar")
    assert not result.trades.empty
    day3 = result.trades[result.trades["session_date"] == "2026-06-03"]
    assert len(day3) == 1
    entry = pd.Timestamp(day3.iloc[0]["entry_ts"]).tz_convert(CT)
    assert entry.hour == 9 and entry.minute == 30
    assert result.skipped.get("once_per_session", 0) >= 0


def test_timed_psar_profit_target_multiple_of_stop():
    ohlc = _rth_days(["2026-06-02", "2026-06-03"], step=0.25, range_pts=1.0)
    result = simulate(ohlc, _timed_params(
        profit_target_pts=4.0, profit_target_unit="stop",
        stop_loss_pts=2.0, stop_loss_unit="points",
        entry_time="08:30",
    ), model_id="timed_psar")
    assert not result.trades.empty
    last = result.trades[result.trades["session_date"] == "2026-06-03"].iloc[0]
    if last["exit_reason"] == "LG Profit Target":
        assert last["exit_px"] - last["entry_px"] == pytest.approx(8.0, abs=0.3)


def test_timed_psar_chart_uses_direction_tf_sar():
    ohlc = _rth_days(["2026-06-02", "2026-06-03"], step=0.5)
    sess = build_chart_payload(
        source="external:timed_psar",
        start="2026-06-03", end="2026-06-03",
        params={"direction_tf": "1hour"},
        book={"min_atr": 0.0, "timeframe": "1min"},
        session_only=True, ohlc=ohlc,
    )
    assert sess["has_sar"] is True
    assert any(v is not None for v in sess["sar"])


def test_timed_psar_run_backtest_payload():
    ohlc = _rth_days(["2026-06-02", "2026-06-03"], step=0.5)
    payload = run_backtest(
        source="external:timed_psar",
        start="2026-06-03", end="2026-06-03",
        params={"direction_tf": "1hour", "entry_time": "09:00"},
        book={"min_atr": 0.0, "profit_target_pts": 0.0, "stop_loss_pts": 0.0},
        ohlc=ohlc,
    )
    assert payload["config"]["candidate_id"] == "timed_psar"
    assert "Timed PSAR" in payload["config"]["weights_note"]
    assert payload["config"]["params"]["direction_tf"] == "1hour"
    assert payload["summary"]["n_trades"] >= 1


def test_chart_payload_includes_trade_window():
    ohlc = _rth_1m(start_ct="08:00", end_ct="15:00", step=0.5, range_pts=4.0)
    sess = build_chart_payload(
        source="external:psar",
        start="2026-06-02", end="2026-06-02",
        book={"trade_start": "09:00", "trade_end": "11:00"},
        session_only=True, ohlc=ohlc,
    )
    assert sess["trade_start"] == "09:00"
    assert sess["trade_end"] == "11:00"
    assert sess["session_start"] == "08:30"
    assert sess["min_atr_unit"] == "points"
