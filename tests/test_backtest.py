"""Tests for the user-defined strategy backtest engine — Phase 23 entry path 3.

Covers the rule DSL, spec loading/validation, the trade state machine (target/stop/time/
eod exits, cooldown, conflicting signals), calendar-cadence entries + settle modes, KPIs,
the canonical-frame contract handoff, and an end-to-end dry-run. Uses synthetic in-memory
feature frames — no parquet, no GPU, no real training required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import parse_duration_bars, run_backtest
from src.backtest.weights_score import _resolve_signal_on_close
from src.backtest.rules import eval_condition, eval_rules, rule_features
from src.backtest.spec import StrategySpec, load_strategy_spec


# ── fixtures ────────────────────────────────────────────────────────────────

def _frame(closes, feats=None, *, start="2025-06-02 08:00", tz="America/Chicago",
           freq="5s"):
    """Build a feature frame indexed by a UTC ts with a _close column."""
    n = len(closes)
    idx_local = pd.date_range(start=start, periods=n, freq=freq, tz=tz)
    idx = idx_local.tz_convert("UTC")
    data = {"_close": np.asarray(closes, dtype="float64")}
    if feats:
        for k, v in feats.items():
            data[k] = np.asarray(v, dtype="float64")
    return pd.DataFrame(data, index=idx)


def _spec(**over):
    base = {
        "schema_version": "v1",
        "strategy_id": "t",
        "entry": {"long": {"all": [{"feature": "sig", "op": "gte", "value": 1}]}},
        "exit": {"target_points": 10, "stop_points": 5, "eod_flat": True},
        "backtest": {"point_value": 1.0},
    }
    base.update(over)
    return load_strategy_spec(base)


# ── rule DSL ──────────────────────────────────────────────────────────────────

def test_eval_condition_operators():
    df = _frame([1, 2, 3, 4], feats={"a": [0, 5, 10, -3]})
    assert eval_condition(df, {"feature": "a", "op": "gt", "value": 4}).tolist() == \
        [False, True, True, False]
    assert eval_condition(df, {"feature": "a", "op": "abs_gt", "value": 4}).tolist() == \
        [False, True, True, False]
    assert eval_condition(df, {"feature": "a", "op": "between", "value": [0, 5]}).tolist() == \
        [True, True, False, False]


def test_eval_condition_feature_rhs():
    df = _frame([1, 2, 3], feats={"a": [1, 5, 2], "b": [2, 4, 9]})
    assert eval_condition(df, {"feature": "a", "op": "gt", "feature_rhs": "b"}).tolist() == \
        [False, True, False]


def test_cross_above_fires_only_on_crossing_bar():
    df = _frame([1, 1, 1, 1], feats={"a": [-1, -1, 1, 2]})
    out = eval_condition(df, {"feature": "a", "op": "cross_above", "value": 0})
    assert out.tolist() == [False, False, True, False]


def test_cross_respects_day_boundary():
    # 3 bars day 1 (crossing at idx2) then day 2 starts low->high but must NOT cross at day open
    df = _frame([1] * 6, feats={"a": [-1, -1, 1, -1, 1, 2]})
    day_id = np.array([0, 0, 0, 1, 1, 1])
    out = eval_condition(df, {"feature": "a", "op": "cross_above", "value": 0}, day_id=day_id)
    # idx2: -1->1 within day0 = cross; idx3 is day1 first bar (no cross); idx4: -1->1 within day1
    assert out.tolist() == [False, False, True, False, True, False]


def test_eval_rules_all_any_nested():
    df = _frame([1, 1, 1], feats={"a": [1, 0, 1], "b": [0, 1, 1]})
    blk = {"all": [{"feature": "a", "op": "gte", "value": 1},
                   {"any": [{"feature": "b", "op": "gte", "value": 1}]}]}
    assert eval_rules(df, blk).tolist() == [False, False, True]


def test_rule_features_collects_all_columns():
    blk = {"all": [{"feature": "a", "op": "gt", "value": 1},
                   {"any": [{"feature": "b", "op": "gt", "feature_rhs": "c"}]}]}
    assert rule_features(blk) == {"a", "b", "c"}


# ── duration parsing ────────────────────────────────────────────────────────

def test_parse_duration_bars():
    assert parse_duration_bars("10m") == 120       # 600s / 5s
    assert parse_duration_bars("90s") == 18
    assert parse_duration_bars("1h") == 720
    assert parse_duration_bars(None) is None


# ── spec loading / validation ─────────────────────────────────────────────────

def test_load_example_spec():
    spec = load_strategy_spec("config/strategy.example.yaml")
    assert spec.strategy_id == "vwap_stretch_reversal"
    assert spec.mode == "rules"
    assert "vw_vwap_dist" in spec.referenced_features()
    assert "of_delta" in spec.referenced_features()


def test_spec_rejects_both_entry_and_schedule():
    with pytest.raises(ValueError):
        load_strategy_spec({
            "strategy_id": "x",
            "entry": {"long": {"all": [{"feature": "a", "op": "gt", "value": 1}]}},
            "schedule": {"frequency": "daily", "side": "long"},
        })


def test_spec_rejects_neither_entry_nor_schedule():
    with pytest.raises(ValueError):
        load_strategy_spec({"strategy_id": "x"})


def test_spec_objective_mismatch_raises():
    with pytest.raises(ValueError):
        load_strategy_spec(
            {"strategy_id": "x", "objective_id": "other",
             "entry": {"long": {"all": [{"feature": "a", "op": "gt", "value": 1}]}}},
            objective_id="nq-50pt")


def test_spec_bad_date_window_raises():
    with pytest.raises(ValueError):
        load_strategy_spec({
            "strategy_id": "x",
            "entry": {"long": {"all": [{"feature": "a", "op": "gt", "value": 1}]}},
            "backtest": {"start": "2026-05-01", "end": "2026-01-01"},
        })


# ── engine: exits ───────────────────────────────────────────────────────────

def test_target_exit_long():
    # enter at idx0 (close=100), target +10 hit at idx2 (close=110)
    df = _frame([100, 105, 110, 108], feats={"sig": [1, 0, 0, 0]})
    res = run_backtest(df, _spec())
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["side"] == "long"
    assert t["pnl_points"] == pytest.approx(10.0)


def test_stop_exit_long():
    df = _frame([100, 98, 94, 94], feats={"sig": [1, 0, 0, 0]})
    res = run_backtest(df, _spec())
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    # Fills at the bar close that first breached the stop (94 = -6pt), not the theoretical
    # -5 level — 5s bar closes, no intrabar modelling (conservative + honest).
    assert t["pnl_points"] == pytest.approx(-6.0)


def test_on_close_market_target_uses_triggering_candle_then_market_quote():
    # The long enters at the ask (100.25). The next 5s close reaches its 150.25
    # trigger, but the executable sell bid is 150.00. MKT must exit at that bid,
    # not at the exact target price and not defer until the bid reaches target.
    ns = 1_000_000_000
    result = _resolve_signal_on_close(
        is_long=True,
        signal_ts_ns=0,
        signal_close_px=100.0,
        bar_ts_ns=np.array([0, 5 * ns], dtype="int64"),
        bar_close_px=np.array([100.0, 150.5]),
        horizon_ns=10 * ns,
        target_pts=50.0,
        stop_pts=6.0,
        tbbo_ts=np.array([0, 5 * ns], dtype="int64"),
        tbbo_px=np.array([100.0, 150.25]),
        tbbo_bid=np.array([99.75, 150.0]),
        tbbo_ask=np.array([100.25, 150.5]),
        price_mode="mkt",
    )
    assert result["outcome"] == "target"
    assert result["entry_px"] == pytest.approx(100.25)
    assert result["exit_px"] == pytest.approx(150.0)
    assert result["pnl_points"] == pytest.approx(49.75)


def test_stop_wins_same_bar_conflict():
    # a bar that is both >= target and <= stop distance can't happen on one close, but
    # stop appearing at an EARLIER bar than target must win.
    df = _frame([100, 94, 111, 100], feats={"sig": [1, 0, 0, 0]})
    res = run_backtest(df, _spec())
    assert res.trades.iloc[0]["exit_reason"] == "stop"


def test_time_stop_exit():
    df = _frame([100] * 6, feats={"sig": [1, 0, 0, 0, 0, 0]})
    spec = _spec(exit={"stop_points": 50, "target_points": 50,
                       "max_holding": "10s", "eod_flat": False})
    res = run_backtest(df, spec)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "time"
    assert t["holding_bars"] == 2                  # 10s / 5s bars


def test_eod_flat_closes_at_session_end():
    df = _frame([100, 101, 102], feats={"sig": [1, 0, 0]})
    spec = _spec(exit={"target_points": 999, "stop_points": 999, "eod_flat": True})
    res = run_backtest(df, spec)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "eod"
    assert t["exit_idx"] == 2                       # last bar of the (only) session


def test_run_backtest_aliases_cs_close():
    """Native 15s/1m hive has cs_close, not the 5s _close passthrough."""
    n = 3
    idx_local = pd.date_range("2025-06-02 08:30", periods=n, freq="15s", tz="America/Chicago")
    df = pd.DataFrame(
        {"cs_close": [100.0, 101.0, 112.0], "sig": [1.0, 0.0, 0.0]},
        index=idx_local.tz_convert("UTC"),
    )
    spec = _spec(exit={"target_points": 10, "stop_points": 50, "eod_flat": True})
    res = run_backtest(df, spec)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "target"


def test_backtest_payload_labels_cs_close_without__close():
    """WebUI 500: engine aliased cs_close locally; confusion labels still read _close."""
    from src.backtest.runner import _backtest_payload

    n = 8
    idx_local = pd.date_range("2025-06-02 08:30", periods=n, freq="15s", tz="America/Chicago")
    df = pd.DataFrame(
        {"cs_close": np.linspace(100.0, 120.0, n), "sig": [1.0] + [0.0] * (n - 1)},
        index=idx_local.tz_convert("UTC"),
    )
    spec = _spec(
        objective_id="nq-50pt-econ-lg-st",
        exit={"target_points": 10, "stop_points": 50, "eod_flat": True},
    )
    res = run_backtest(df, spec)
    assert len(res.trades) == 1
    assert "_close" not in df.columns
    payload = _backtest_payload(res, spec, df, "2025-06-02", "2025-06-02", 1)
    assert payload["summary"]["n_trades"] == 1
    assert payload["confusion_matrix"]["labels"] == ["down", "flat", "up"]


def test_short_side_pnl():
    df = _frame([100, 95, 90], feats={"sig_s": [1, 0, 0]})
    spec = _spec(entry={"short": {"all": [{"feature": "sig_s", "op": "gte", "value": 1}]}},
                 exit={"target_points": 10, "stop_points": 5, "eod_flat": True})
    res = run_backtest(df, spec)
    t = res.trades.iloc[0]
    assert t["side"] == "short"
    assert t["exit_reason"] == "target"
    assert t["pnl_points"] == pytest.approx(10.0)


def test_conflicting_long_short_skipped():
    df = _frame([100, 110], feats={"a": [1, 0], "b": [1, 0]})
    spec = _spec(entry={"long":  {"all": [{"feature": "a", "op": "gte", "value": 1}]},
                        "short": {"all": [{"feature": "b", "op": "gte", "value": 1}]}})
    res = run_backtest(df, spec)
    assert res.skipped["conflicting_signal"] == 1
    assert len(res.trades) == 0


def test_cooldown_blocks_immediate_reentry():
    # signal fires every bar; target hits next bar; cooldown must gate re-entry.
    df = _frame([100, 110, 120, 130, 140, 150],
                feats={"sig": [1, 1, 1, 1, 1, 1]})
    spec = _spec(exit={"target_points": 10, "stop_points": 50, "eod_flat": False},
                 backtest={"cooldown_bars": 2, "point_value": 1.0})
    res = run_backtest(df, spec)
    # without cooldown every bar re-enters; with cooldown_bars=2 entries are spaced out.
    assert res.skipped["cooldown"] >= 1


def test_costs_and_slippage_applied():
    df = _frame([100, 110], feats={"sig": [1, 0]})
    spec = _spec(exit={"target_points": 10, "stop_points": 50, "eod_flat": True},
                 costs={"cost_per_trade": 1.0, "slippage_points": 0.5})
    res = run_backtest(df, spec)
    t = res.trades.iloc[0]
    # entry 100+0.5, exit 110-0.5, minus 1.0 cost => 10 - 1(slip) - 1(cost) = 8
    assert t["pnl_points"] == pytest.approx(8.0)


# ── engine: schedule (calendar cadence) ───────────────────────────────────────

def _multiday_frame(n_days=5, bars_per_day=4, base=100.0):
    closes, idxs = [], []
    day = pd.Timestamp("2025-06-02 08:00", tz="America/Chicago")
    for d in range(n_days):
        start = day + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            idxs.append((start + pd.Timedelta(seconds=5 * b)))
            closes.append(base + d + b * 0.1)
    idx = pd.DatetimeIndex(idxs).tz_convert("UTC")
    return pd.DataFrame({"_close": closes}, index=idx)


def test_schedule_daily_eod():
    df = _multiday_frame(n_days=3, bars_per_day=4)
    spec = load_strategy_spec({
        "strategy_id": "sched",
        "schedule": {"frequency": "daily", "side": "long", "settle": {"mode": "eod"}},
    })
    res = run_backtest(df, spec)
    assert len(res.trades) == 3                     # one per day
    assert (res.trades["exit_reason"] == "settle").all()


def test_schedule_hold_days_crosses_sessions():
    df = _multiday_frame(n_days=4, bars_per_day=3)
    spec = load_strategy_spec({
        "strategy_id": "sched",
        "schedule": {"frequency": "monthly", "day_of_month": "first", "side": "long",
                     "settle": {"mode": "hold_days", "days": 2}},
    })
    res = run_backtest(df, spec)
    assert len(res.trades) == 1                     # first trading day of the month
    t = res.trades.iloc[0]
    assert t["holding_bars"] > 3                    # held across >1 session


def test_schedule_weekly_picks_weekday():
    # 2025-06-02 is a Monday; 10 trading days => 2 Mondays
    df = _multiday_frame(n_days=10, bars_per_day=2)
    spec = load_strategy_spec({
        "strategy_id": "wk",
        "schedule": {"frequency": "weekly", "day_of_week": "MON", "side": "long",
                     "settle": {"mode": "eod"}},
    })
    res = run_backtest(df, spec)
    assert len(res.trades) == 2


# ── KPIs + canonical contract ─────────────────────────────────────────────────

def test_kpis_computed():
    df = _frame([100, 110, 100, 90, 100, 110],
                feats={"sig": [1, 0, 1, 0, 1, 0]})
    spec = _spec(exit={"target_points": 10, "stop_points": 10, "eod_flat": False},
                 backtest={"point_value": 20.0})
    res = run_backtest(df, spec)
    k = res.kpis
    assert k["n_trades"] >= 1
    assert 0.0 <= k["win_rate"] <= 1.0
    assert k["total_value"] == pytest.approx(k["total_points"] * 20.0)


def test_canonical_frame_shape():
    df = _frame([100, 110, 100], feats={"sig": [1, 0, 0]})
    res = run_backtest(df, _spec())
    assert list(res.canonical.columns) == ["ts", "direction", "confidence"]
    assert (res.canonical["confidence"] == 1.0).all()
    assert res.canonical["direction"].isin(["up", "down"]).all()


def test_empty_when_no_signal():
    df = _frame([100, 101, 102], feats={"sig": [0, 0, 0]})
    res = run_backtest(df, _spec())
    assert res.trades.empty
    assert res.kpis["n_trades"] == 0


def test_holdout_split_reported():
    df = _multiday_frame(n_days=6, bars_per_day=4)
    spec = load_strategy_spec({
        "strategy_id": "h",
        "schedule": {"frequency": "daily", "side": "long", "settle": {"mode": "eod"}},
        "backtest": {"holdout_start": "2025-06-05"},
    })
    res = run_backtest(df, spec, objective={"session": {"timezone": "America/Chicago"}})
    assert "in_sample" in res.kpis and "out_of_sample" in res.kpis
    assert (res.kpis["in_sample"]["n_trades"]
            + res.kpis["out_of_sample"]["n_trades"] == res.kpis["n_trades"])


def test_concat_etc_utc_and_utc_indexes_does_not_raise():
    """Live 8/14 parquet is tz=UTC; older days are Etc/UTC. Concat used to fail
    with 'Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True'."""
    from src.serving.predict import predict_canonical
    from src.training.data_pipeline import ensure_utc_index

    a = pd.DataFrame(
        {"_close": [100.0, 101.0], "f1": [1.0, 1.0], "sig": [0.0, 0.0]},
        index=pd.date_range("2026-08-13 13:00", periods=2, freq="5s", tz="Etc/UTC"),
    )
    b = pd.DataFrame(
        {"_close": [102.0, 103.0], "f1": [1.0, 1.0], "sig": [0.0, 0.0]},
        index=pd.date_range("2026-08-14 13:00", periods=2, freq="5s", tz="UTC"),
    )
    raw = pd.concat([a, b]).sort_index()
    assert not isinstance(raw.index, pd.DatetimeIndex)

    with pytest.raises((TypeError, ValueError), match="utc=True"):
        pd.DatetimeIndex(raw.index)

    fixed = pd.concat([ensure_utc_index(a), ensure_utc_index(b)]).sort_index()
    assert str(fixed.index.tz) == "UTC"
    ts = pd.to_datetime(fixed.index, utc=True)
    assert len(ts) == 4

    class _Boost:
        def predict_proba(self, X):
            n = len(X)
            p = np.zeros((n, 3), dtype="float64")
            p[:, 1] = 1.0
            return p

    bundle = {"feature_cols": ["f1"], "booster": _Boost(), "scaler": None, "dir_thr": 0.5}
    pred = predict_canonical(fixed, bundle)
    assert len(pred) == 4
    assert str(pd.DatetimeIndex(pred["ts"]).tz) == "UTC"

    spec = _spec()
    res = run_backtest(raw, spec)  # mixed object index must be accepted
    assert res.n_bars == 4


def test_cme_holiday_sessions_are_not_traded():
    """Black Friday 2025-11-28 is in cme_holidays; weekday parquet still exists."""
    from datetime import date

    from src.backtest.engine import drop_non_trading_sessions
    from src.common.cme_holidays import holiday_iso_dates, is_cme_holiday, is_trading_day

    assert is_trading_day(date(2025, 11, 28)) is False
    assert is_trading_day(date(2025, 12, 1)) is True
    assert "2025-11-28" in holiday_iso_dates()
    assert is_cme_holiday(date(2026, 7, 2)) is False
    assert is_trading_day(date(2026, 7, 2)) is True
    assert is_cme_holiday(date(2026, 7, 3)) is True
    assert is_trading_day(date(2026, 7, 3)) is False
    assert "2026-07-02" not in holiday_iso_dates()
    assert "2026-07-03" in holiday_iso_dates()

    hol = _frame(
        [100, 101, 102], feats={"sig": [1, 1, 1]},
        start="2025-11-28 08:30", tz="America/Chicago",
    )
    mon = _frame(
        [103, 104, 105], feats={"sig": [1, 1, 1]},
        start="2025-12-01 08:30", tz="America/Chicago",
    )
    df = pd.concat([hol, mon]).sort_index()
    assert len(drop_non_trading_sessions(df, "America/Chicago")) == 3

    res = run_backtest(df, _spec())
    assert res.n_bars == 3
    assert res.kpis["n_trades"] >= 1
    assert set(res.trades["session_date"]) == {"2025-12-01"}


def test_available_dates_skips_cme_holidays(tmp_path):
    from src.backtest.runner import _available_dates

    for day in ("2025-11-26", "2025-11-27", "2025-11-28", "2025-12-01"):
        (tmp_path / f"features_{day}.parquet").touch()
    assert _available_dates(tmp_path) == ["2025-11-26", "2025-12-01"]


def test_infer_bundle_feature_store_from_candidate_and_schema():
    from src.backtest.weights_score import infer_bundle_feature_store

    assert infer_bundle_feature_store({"candidate_id": "xgb_1m"}) == "1m"
    assert infer_bundle_feature_store({"candidate_id": "xgb_5m"}) == "5m"
    assert infer_bundle_feature_store({"candidate_id": "lstm_15s"}) == "15s"
    assert infer_bundle_feature_store({
        "candidate_id": "xgboost",
        "feature_cols": ["so_fast_d_9_3", "cs_close"],
    }) == "1m"
    assert infer_bundle_feature_store({
        "candidate_id": "xgboost",
        "feature_cols": ["svp_poc_dist", "cs_atr_5"],
    }) == "5m"
    assert infer_bundle_feature_store({
        "manifest": {"native_tf": "1m"},
        "candidate_id": "xgb_1m",
    }) == "1m"


def test_recanonicalize_dual_oof_falls_back_when_fire_too_hot():
    from src.backtest.runner import _recanonicalize_dual_oof

    ts = pd.date_range("2026-02-13 08:30", periods=4, freq="15s", tz="UTC")
    raw = pd.DataFrame({
        "ts": ts,
        "direction": ["flat"] * 4,
        "confidence": [0.35, 0.20, 0.31, 0.10],
        "p_long": [0.35, 0.05, 0.31, 0.02],
        "p_short": [0.10, 0.20, 0.11, 0.10],
    })
    obj = {"labels": {"encoding": "dual_binary", "fire": 0.40}}
    book = _recanonicalize_dual_oof(raw, obj, confidence_gate=True)
    assert list(book["direction"]) == ["flat"] * 4
    off = _recanonicalize_dual_oof(raw, obj, confidence_gate=False)
    assert off["direction"].tolist() == ["up", "down", "up", "down"]
    assert float(off.attrs.get("dual_binary_fire")) == 0.1
    at_ui = _recanonicalize_dual_oof(
        raw, obj, confidence_gate=True, tau=0.10)
    assert at_ui["direction"].tolist() == ["up", "down", "up", "down"]
    assert float(at_ui.attrs.get("dual_binary_fire")) == 0.1

    raw2 = raw.copy()
    raw2.loc[0, "p_long"] = 0.41
    raw2.loc[0, "confidence"] = 0.41
    sparse = _recanonicalize_dual_oof(raw2, obj, confidence_gate=False)
    assert sparse["direction"].tolist() == ["up", "down", "up", "down"]
    assert float(sparse.attrs.get("dual_binary_fire")) == 0.1


def test_last_fold_fire_follows_ui_tau_not_book_040():
    from src.backtest.weights_score import _with_dual_binary_fire

    obj = {"labels": {"encoding": "dual_binary", "fire": 0.40}}
    base = {"family": "attention_lstm"}
    baked = _with_dual_binary_fire(
        base, obj, confidence_gate=True, tau=0.40)
    assert baked["dual_binary_fire"] == pytest.approx(0.40)
    ui = _with_dual_binary_fire(
        base, obj, confidence_gate=True, tau=0.10)
    assert ui["dual_binary_fire"] == pytest.approx(0.10)
    off = _with_dual_binary_fire(
        base, obj, confidence_gate=False, tau=0.40)
    assert off["dual_binary_fire"] == pytest.approx(0.10)
    zero = _with_dual_binary_fire(
        base, obj, confidence_gate=True, tau=0.0)
    assert zero["dual_binary_fire"] == pytest.approx(0.10)


def test_infer_source_feature_store_from_oof_paths(tmp_path):
    from src.backtest.weights_score import infer_source_feature_store

    assert infer_source_feature_store(
        tmp_path / "oof_xgb_1m.parquet") == "1m"
    assert infer_source_feature_store(
        tmp_path / "oof_xgb_5m.parquet") == "5m"
    assert infer_source_feature_store(
        tmp_path / "oof_lstm_15s.parquet") == "15s"
    nested = tmp_path / "models" / "xgb_1m"
    nested.mkdir(parents=True)
    (nested / "oof.parquet").write_text("")
    assert infer_source_feature_store(nested / "oof.parquet") == "1m"
    assert infer_source_feature_store(
        tmp_path / "oof_tf_oof_meta__lstm_15s_xgb_1m_xgb_5m.parquet") == "15s"
