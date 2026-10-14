"""Tests for label-implied trade economics (Tier A PnL) in src.evaluation._kpi.

Covers the asymmetric barrier payoff (+move_threshold win / -mae_tolerance loss), the
always-on expectancy (no longer gated behind costs), and the new ``pnl_at_gate`` /
``pnl_at_tau_sweep`` blocks in the diagnostics report. Pure synthetic frames — no parquet,
no training.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation._kpi import (
    canonical_trade_kpi,
    compute_diagnostics,
    label_implied_pnl,
    payoff_params,
)

_OBJ = {"labels": {"move_threshold": 50, "mae_tolerance": 6}}


def _oof():
    """6-bar OOF: at τ≥0.8 the directional signals are rows 0,1,2,5 (row3 sub-gate,
    row4 flat). Of those: 0 & 2 correct, 1 & 5 wrong → 2 correct / 2 wrong."""
    return pd.DataFrame({
        "ts":             pd.date_range("2025-01-01 09:00", periods=6, freq="1min", tz="UTC"),
        "direction":      ["up", "up",   "down", "up",   "flat", "down"],
        "confidence":     [0.90, 0.90,   0.90,   0.40,   0.90,   0.95],
        "direction_true": ["up", "flat", "down", "down", "flat", "up"],
    })


# ── payoff params ───────────────────────────────────────────────────────────────

def test_payoff_params_reads_objective():
    assert payoff_params(
        {"labels": {"move_threshold": 50, "mae_tolerance": 6},
         "costs": {"cost_per_trade": 1.5}}) == (50.0, 6.0, 1.5)


def test_payoff_params_defaults():
    assert payoff_params({}) == (50.0, 5.0, 0.0)
    assert payoff_params({"costs": {"cost_per_trade": None}}) == (50.0, 5.0, 0.0)


# ── label_implied_pnl: the asymmetric payoff ────────────────────────────────────

def test_label_implied_pnl_asymmetry():
    df = pd.DataFrame({"direction": ["up", "up", "down"],
                       "direction_true": ["up", "down", "flat"]})   # 1 correct, 2 wrong
    r = label_implied_pnl(df, win_points=50, loss_points=6, cost=0)
    assert r["n_trades"] == 3
    assert r["total_points"] == pytest.approx(50 - 6 - 6)           # 38
    assert r["net_expectancy"] == pytest.approx(38 / 3)
    assert r["hit_rate"] == pytest.approx(1 / 3)
    # a wrong trade must cost the STOP, never the full move
    assert r["loss_points"] == 6 and r["win_points"] == 50


def test_label_implied_pnl_beats_symmetric_assumption():
    """The whole point of the fix: loss is capped at the stop, so a mostly-wrong book
    is far less negative than the old symmetric ±move assumption implied."""
    df = pd.DataFrame({"direction": ["up"] * 10,
                       "direction_true": ["up"] * 4 + ["flat"] * 6})  # 4 win, 6 stop
    r = label_implied_pnl(df, win_points=50, loss_points=5, cost=0)
    assert r["total_points"] == pytest.approx(4 * 50 - 6 * 5)         # +170, not 4*50-6*50=-100
    assert r["net_expectancy"] > 0


def test_label_implied_pnl_cost_applied_both_legs():
    df = pd.DataFrame({"direction": ["up", "down"], "direction_true": ["up", "up"]})
    r = label_implied_pnl(df, win_points=50, loss_points=5, cost=2)   # 48 + (-7)
    assert r["total_points"] == pytest.approx(41)


def test_label_implied_pnl_empty():
    r = label_implied_pnl(pd.DataFrame({"direction": [], "direction_true": []}),
                          win_points=50, loss_points=5)
    assert r["n_trades"] == 0
    assert r["net_expectancy"] is None
    assert r["max_drawdown"] == 0.0


def test_drawdown_counts_a_leading_loss():
    # equity starts at 0: a first losing trade (-6) is immediately -6 underwater.
    df = pd.DataFrame({"direction": ["up", "up", "up"],
                       "direction_true": ["flat", "up", "up"]})   # pnl = [-6, +50, +50]
    r = label_implied_pnl(df, win_points=50, loss_points=6, cost=0)
    assert r["max_drawdown"] == pytest.approx(-6.0)


def test_drawdown_peak_to_trough():
    # win then loss → equity peaks at +50 then dips to +44: a -6 drawdown from the peak
    df = pd.DataFrame({"direction": ["up", "up", "up"],
                       "direction_true": ["up", "flat", "up"]})   # pnl = [+50, -6, +50]
    r = label_implied_pnl(df, win_points=50, loss_points=6, cost=0)
    assert r["max_drawdown"] == pytest.approx(-6.0)


def test_drawdown_zero_when_only_wins():
    # a trade that wins outright never puts equity below its high-water mark
    df = pd.DataFrame({"direction": ["up", "up"], "direction_true": ["up", "up"]})
    r = label_implied_pnl(df, win_points=50, loss_points=6, cost=0)
    assert r["max_drawdown"] == 0.0


def test_drawdown_accumulates_consecutive_losses():
    # three straight stop-outs → cumulative current loss deepens to -18
    df = pd.DataFrame({"direction": ["up", "up", "up"],
                       "direction_true": ["flat", "down", "flat"]})  # pnl = [-6, -6, -6]
    r = label_implied_pnl(df, win_points=50, loss_points=6, cost=0)
    assert r["max_drawdown"] == pytest.approx(-18.0)


# ── canonical_trade_kpi: always-on pnl block ────────────────────────────────────

def test_canonical_trade_kpi_exposes_pnl_without_costs():
    score = canonical_trade_kpi(_oof(), _oof()[["direction_true"]], _OBJ, tau=0.8)
    assert "pnl" in score                       # present even though objective has no costs
    assert score["pnl"]["n_trades"] == 4
    assert score["net_expectancy"] == pytest.approx((50 + 50 - 6 - 6) / 4)  # 22.0
    assert score["hit_rate"] == pytest.approx(0.5)
    assert score["trades_at_raw"]["taken"] == 5
    assert score["pnl_raw"]["n_trades"] == 5


# ── compute_diagnostics: pnl_at_gate + pnl_at_tau_sweep ─────────────────────────

def test_compute_diagnostics_no_declared_tau_uses_argmax_book():
    oof = _oof()
    diag = compute_diagnostics(oof, oof[["direction_true"]], _OBJ)
    assert diag["trades_at_gate"]["gate_tau"] is None
    assert diag["trades_at_gate"]["taken"] == 5
    assert diag["pnl_at_raw"]["n_trades"] == 5


def test_compute_diagnostics_gate_tau_follows_objective_primary_tau():
    """n_signals KPI reads trades_at_gate.taken — it must use requirement τ."""
    oof = _oof()
    obj = {
        "labels": {"move_threshold": 50, "mae_tolerance": 6},
        "kpi": {"requirement": {"all": [
            {"metric": "precision_at_tau", "tau": 0.65, "min": 0.55, "min_signals": 30},
        ]}},
    }
    diag = compute_diagnostics(oof, oof[["direction_true"]], obj)
    assert diag["trades_at_gate"]["gate_tau"] == pytest.approx(0.65)
    # All 4 directional rows in _oof have conf >= 0.65 (the 0.40 up is dropped).
    assert diag["trades_at_gate"]["taken"] == 4
    # Raw book is ungated argmax: includes the 0.40 up (5 directional, 1 flat).
    assert diag["confusion_matrix_raw"]["operating_point"] == "argmax"
    assert diag["trades_at_raw"]["taken"] == 5
    assert diag["pnl_at_raw"]["n_trades"] == 5
    assert diag["pnl_at_raw"]["operating_point"] == "argmax"


def test_compute_diagnostics_pnl_blocks():
    oof = _oof()
    diag = compute_diagnostics(oof, oof[["direction_true"]], _OBJ, gate_tau=0.8)
    assert "pnl_at_gate" in diag and "pnl_at_tau_sweep" in diag

    # pnl_at_gate must agree with trades_at_gate counts (2 correct, 2 wrong)
    assert diag["trades_at_gate"]["taken"] == 4
    assert diag["trades_at_gate"]["correct"] == 2
    g = diag["pnl_at_gate"]
    assert g["n_trades"] == 4
    assert g["total_points"] == pytest.approx(50 + 50 - 6 - 6)       # 88
    assert g["gate_tau"] == 0.8

    # sweep is one row per τ with an expectancy; n_trades is monot<= as τ rises
    sweep = diag["pnl_at_tau_sweep"]
    taus = [r["tau"] for r in sweep]
    assert taus == sorted(taus)
    counts = [r["n_trades"] for r in sweep]
    assert all(a >= b for a, b in zip(counts, counts[1:]))           # non-increasing


def test_compute_diagnostics_pnl_uses_tbm_stop():
    """A stricter stop (bigger mae_tolerance) lowers expectancy on a book with losses."""
    oof = _oof()
    d5 = compute_diagnostics(oof, oof[["direction_true"]],
                             {"labels": {"move_threshold": 50, "mae_tolerance": 5}}, gate_tau=0.8)
    d20 = compute_diagnostics(oof, oof[["direction_true"]],
                              {"labels": {"move_threshold": 50, "mae_tolerance": 20}}, gate_tau=0.8)
    assert d20["pnl_at_gate"]["net_expectancy"] < d5["pnl_at_gate"]["net_expectancy"]
