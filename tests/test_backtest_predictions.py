"""Tests for predictions mode (Tier B) — replaying a trained model's OOF frame through
the backtest engine, gated by confidence, with one position at a time.

Covers: spec one-of enforcement, the engine's confidence gate / sides filter /
one-position rule reusing the existing exit+ledger machinery, and the runner helpers
(source slug, objective-barrier exit defaults, OOF join, canonical frame). Synthetic
in-memory frames — no parquet, no training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _compute_kpis, run_backtest
from src.backtest.runner import (
    _apply_prediction_exit_defaults,
    _attach_predictions,
    _predictions_canonical,
    _slug_from_source,
)
from src.backtest.spec import load_strategy_spec


# ── fixtures ────────────────────────────────────────────────────────────────────

def _frame(closes, *, start="2025-06-02 08:00", tz="America/Chicago", freq="5s"):
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz=tz).tz_convert("UTC")
    return pd.DataFrame({"_close": np.asarray(closes, dtype="float64")}, index=idx)


def _with_preds(df, direction, confidence):
    df = df.copy()
    df["_pred_direction"] = list(direction)
    df["_pred_confidence"] = np.asarray(confidence, dtype="float64")
    return df


def _pred_spec(**over):
    base = {"schema_version": "v1", "strategy_id": "p",
            "predictions": {"source": "x.parquet", "tau": 0.8},
            "exit": {"target_points": 10, "stop_points": 5, "eod_flat": True}}
    base.update(over)
    return load_strategy_spec(base)


# ── spec validation ─────────────────────────────────────────────────────────────

def test_spec_predictions_parses():
    spec = load_strategy_spec({"strategy_id": "x",
                               "predictions": {"source": "y.parquet", "tau": 0.9}})
    assert spec.mode == "predictions"
    assert spec.predictions["tau"] == 0.9


def test_spec_rejects_two_modes():
    with pytest.raises(ValueError):
        load_strategy_spec({
            "strategy_id": "x",
            "entry": {"long": {"all": [{"feature": "a", "op": "gt", "value": 0}]}},
            "predictions": {"source": "y.parquet"},
        })


def test_spec_rejects_no_mode():
    with pytest.raises(ValueError):
        load_strategy_spec({"strategy_id": "x"})


# ── engine: gating, sides, one-position ─────────────────────────────────────────

def test_predictions_gate_enters_and_hits_target():
    df = _with_preds(_frame([100, 105, 110, 115, 120, 120, 120, 120]),
                     ["up", "flat", "flat", "flat", "flat", "flat", "flat", "flat"],
                     [0.90, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    res = run_backtest(df, _pred_spec())
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["side"] == "long"
    assert t["exit_reason"] == "target"
    assert t["pnl_points"] == pytest.approx(10.0)          # +10 target from 100


def test_predictions_confidence_gate_blocks_low_conf():
    df = _with_preds(_frame([100, 110, 120, 130]),
                     ["up", "up", "up", "up"], [0.5, 0.5, 0.5, 0.5])   # all below τ
    assert run_backtest(df, _pred_spec()).trades.empty


def test_predictions_confidence_gate_off_admits_low_conf():
    # 08:30 CT is inside the default execution window (08:30–14:45).
    df = _with_preds(_frame([100, 110, 120, 130, 140], start="2025-06-02 08:30"),
                     ["up", "flat", "flat", "flat", "flat"],
                     [0.10, 0.10, 0.10, 0.10, 0.10])
    spec = _pred_spec(predictions={
        "source": "x.parquet", "tau": 0.8, "confidence_gate": False,
    })
    res = run_backtest(df, spec)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["side"] == "long"


def test_spec_predictions_confidence_gate_parses():
    spec = load_strategy_spec({
        "strategy_id": "x",
        "predictions": {"source": "y.parquet", "tau": 0.05, "confidence_gate": False},
    })
    assert spec.predictions["tau"] == 0.05
    assert spec.predictions["confidence_gate"] is False


def test_predictions_sides_filter():
    df = _with_preds(_frame([100, 100, 100, 100, 100, 100]),
                     ["up", "down", "up", "down", "up", "down"], [0.9] * 6)
    spec = _pred_spec(predictions={"source": "x.parquet", "tau": 0.8, "sides": ["down"]})
    res = run_backtest(df, spec)
    assert not res.trades.empty
    assert (res.trades["side"] == "short").all()           # long predictions ignored


def test_predictions_one_position_at_a_time():
    # every bar screams "up" at high conf, but only one position may be open at once;
    # with no target/stop it holds to the session close → exactly one trade that day.
    df = _with_preds(_frame([100] * 8), ["up"] * 8, [0.9] * 8)
    res = run_backtest(df, _pred_spec(exit={"eod_flat": True}))
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "eod"
    assert res.skipped["in_position"] > 0                  # later signals were skipped


def test_predictions_concurrency_limit():
    # flat price → every position holds to the session close; with N slots exactly N of
    # the 8 high-conf signals enter (the rest are skipped while all slots are full).
    df = _with_preds(_frame([100] * 8), ["up"] * 8, [0.9] * 8)
    for n in (1, 2, 3):
        spec = _pred_spec(exit={"eod_flat": True}, backtest={"max_concurrent_positions": n})
        res = run_backtest(df, spec)
        assert len(res.trades) == n, f"expected {n} concurrent positions, got {len(res.trades)}"


def test_concurrency_frees_slots_on_exit():
    # rising price + a tight target: each position exits quickly, freeing its slot, so with
    # 2 slots more than 2 trades are taken across the window.
    df = _with_preds(_frame([100, 102, 104, 106, 108, 110, 112, 114]),
                     ["up"] * 8, [0.9] * 8)
    spec = _pred_spec(exit={"target_points": 2, "eod_flat": True},
                      backtest={"max_concurrent_positions": 2})
    res = run_backtest(df, spec)
    assert len(res.trades) > 2
    assert (res.trades["exit_reason"] == "target").any()


def test_predictions_requires_pred_columns():
    # engine must refuse a predictions spec when the runner hasn't joined the columns
    with pytest.raises(ValueError, match="_pred_confidence"):
        run_backtest(_frame([100, 101, 102]), _pred_spec())


# ── runner helpers ──────────────────────────────────────────────────────────────

def test_engine_ledger_drawdown_counts_leading_loss():
    # realised-ledger drawdown must agree with the label-implied definition: a leading
    # losing trade is drawdown (high-water mark seeded at 0), not hidden.
    ledger = pd.DataFrame({
        "pnl_points":   [-6.0, 50.0, 50.0],
        "session_date": ["2025-06-02"] * 3,
        "holding_bars": [1, 1, 1],
        "side":         ["long"] * 3,
        "exit_reason":  ["stop", "target", "target"],
        "mae_points":   [6.0, 0.0, 0.0],
        "mfe_points":   [0.0, 50.0, 50.0],
    })
    k = _compute_kpis(ledger, 1.0, n_bars=100)
    assert k["max_drawdown_points"] == pytest.approx(-6.0)


def test_attach_predictions_dedups_and_drops_nat():
    df = _frame([100, 101, 102, 103])
    pf = pd.DataFrame({
        "ts": [df.index[0], df.index[0], pd.NaT, df.index[2]],   # dup ts + a NaT row
        "direction": ["up", "down", "up", "up"],
        "confidence": [0.5, 0.9, 0.9, 0.95],
        "direction_true": ["up", "down", "flat", "up"],
    })
    out = _attach_predictions(df, pf)
    assert out["_pred_direction"].iloc[0] == "down"          # duplicate ts → last wins
    assert out["_pred_confidence"].iloc[0] == pytest.approx(0.9)
    assert out["_pred_direction"].iloc[2] == "up"            # NaT dropped, real row kept
    assert out["_pred_confidence"].iloc[1] == 0.0            # no prediction at bar 1


def test_attach_predictions_all_nat_raises():
    df = _frame([100, 101, 102])
    pf = pd.DataFrame({"ts": [pd.NaT] * 3, "direction": ["up"] * 3,
                       "confidence": [0.9] * 3, "direction_true": ["up"] * 3})
    with pytest.raises(ValueError, match="no usable timestamps"):
        _attach_predictions(df, pf)


def test_slug_from_source():
    assert _slug_from_source(
        "data/search/o/run/oof_xgb_tbm_full.parquet") == "pred_xgb_tbm_full"
    assert _slug_from_source("weird name!.parquet").startswith("pred_")


def test_apply_exit_defaults_from_objective_barriers():
    spec = load_strategy_spec({"strategy_id": "x", "predictions": {"source": "y.parquet"}})
    obj = {"labels": {"move_threshold": 50, "mae_tolerance": 6},
           "horizon": {"max_trade_horizon": "10m"}}
    _apply_prediction_exit_defaults(spec, obj)
    assert spec.exit["target_points"] == 50
    assert spec.exit["stop_points"] == 6
    assert spec.exit["max_holding"] == "10m"


def test_apply_exit_defaults_does_not_override_explicit():
    spec = load_strategy_spec({"strategy_id": "x", "predictions": {"source": "y.parquet"},
                               "exit": {"target_points": 30}})
    _apply_prediction_exit_defaults(spec, {"labels": {"move_threshold": 50, "mae_tolerance": 6}})
    assert spec.exit["target_points"] == 30                # explicit wins
    assert spec.exit["stop_points"] == 6                   # still filled


def test_attach_predictions_aligns_on_ts():
    df = _frame([100, 110, 120, 130])
    pf = pd.DataFrame({"ts": df.index,
                       "direction": ["up", "flat", "flat", "down"],
                       "confidence": [0.9, 0.1, 0.1, 0.85],
                       "direction_true": ["up", "flat", "flat", "up"]})
    out = _attach_predictions(df, pf)
    assert list(out["_pred_direction"]) == ["up", "flat", "flat", "down"]
    assert out["_pred_confidence"].iloc[0] == pytest.approx(0.9)
    assert "_pred_true" in out.columns


def test_predictions_canonical_schema_and_marks_entries():
    df = _with_preds(_frame([100, 105, 110, 115]),
                     ["up", "flat", "flat", "flat"], [0.9, 0.1, 0.1, 0.1])
    df["_pred_true"] = ["up", "flat", "flat", "up"]
    res = run_backtest(df, _pred_spec(exit={"target_points": 5, "stop_points": 5}))
    canon = _predictions_canonical(df, res)
    assert list(canon.columns) == ["ts", "direction", "confidence", "direction_true"]
    assert (canon["confidence"] == 1.0).all()              # deterministic post-gate marker
    assert (canon["direction"] == "up").any()              # the entry bar is marked long
