"""Tests for CVD-versus-structure features (ADR-122)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.cvd_structure import COLUMNS, compute

CT_OFFSET_MIN = 360  # 08:30 CT == 14:30 UTC in January


def _ts(day: int, ct_minute: float):
    return datetime(2026, 1, day, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=ct_minute + CT_OFFSET_MIN
    )


def _ticks(rows):
    """rows: list of (ct_minute, price, size, side)."""
    return pd.DataFrame(
        [
            {"ts_event": _ts(6, m), "price": float(p), "size": float(s), "side": sd}
            for m, p, s, sd in rows
        ]
    )


def _grid(prices, start_min: float = 510.0, step_s: int = 5):
    idx = pd.date_range(_ts(6, start_min), periods=len(prices), freq=f"{step_s}s", tz="UTC")
    return pd.Series([float(p) for p in prices], index=idx)


def _steady(n=240, price=100.0, size=1.0, side="B", start=510.0):
    return [(start + i / 12.0, price, size, side) for i in range(n)]


# ── contract ─────────────────────────────────────────────────────────────

def test_returns_float32_frame_with_expected_columns():
    out = compute(_ticks(_steady()), _grid([100.0] * 240))
    assert list(out.columns) == list(COLUMNS)
    assert all(str(d) == "float32" for d in out.dtypes)


def test_empty_tbbo_yields_all_nan():
    out = compute(pd.DataFrame(), _grid([100.0] * 10))
    assert out.isna().all().all()


def test_pre_session_ticks_are_excluded():
    # All ticks at 07:00 CT, before the 08:30 session start.
    early = [(420.0 + i / 12.0, 100.0, 1.0, "B") for i in range(60)]
    out = compute(_ticks(early), _grid([100.0] * 60))
    assert out["of_cvd_at_level"].isna().all()


# ── at-level absorption ──────────────────────────────────────────────────

def test_cvd_at_level_accumulates_only_near_a_prior_level():
    levels = {"poc": 100.0}
    on = _ticks(_steady(price=100.0, side="A"))       # sitting on the POC
    off = _ticks(_steady(price=500.0, side="A"))      # far away
    out_on = compute(on, _grid([100.0] * 240), levels=levels)
    out_off = compute(off, _grid([500.0] * 240), levels=levels)
    assert out_on["of_cvd_at_level"].iloc[-1] < 0     # sellers absorbed at POC
    assert out_off["of_cvd_at_level"].iloc[-1] == pytest.approx(0.0)


def test_cvd_at_level_is_nan_without_levels():
    out = compute(_ticks(_steady()), _grid([100.0] * 240), levels=None)
    assert out["of_cvd_at_level"].isna().all()
    assert out["lt_cvd_at_level"].isna().all()


def test_large_trade_at_level_ignores_small_lots():
    levels = {"poc": 100.0}
    small = _ticks(_steady(price=100.0, size=1.0, side="B"))
    large = _ticks(_steady(price=100.0, size=50.0, side="B"))
    o_small = compute(small, _grid([100.0] * 240), levels=levels, large_trade_size=10)
    o_large = compute(large, _grid([100.0] * 240), levels=levels, large_trade_size=10)
    assert o_small["lt_cvd_at_level"].iloc[-1] == pytest.approx(0.0)
    assert o_large["lt_cvd_at_level"].iloc[-1] > 0
    # The unfiltered column still sees the small lots.
    assert o_small["of_cvd_at_level"].iloc[-1] > 0


def test_at_level_tolerance_is_respected():
    levels = {"poc": 100.0}
    just_inside = _ticks(_steady(price=100.75))   # 3 ticks away, tol = 4 ticks
    just_outside = _ticks(_steady(price=102.0))   # 8 ticks away
    a = compute(just_inside, _grid([100.75] * 240), levels=levels, level_tol_ticks=4.0)
    b = compute(just_outside, _grid([102.0] * 240), levels=levels, level_tol_ticks=4.0)
    assert a["of_cvd_at_level"].iloc[-1] > 0
    assert b["of_cvd_at_level"].iloc[-1] == pytest.approx(0.0)


# ── divergence slope ─────────────────────────────────────────────────────

def test_divergence_slope_is_positive_when_cvd_and_price_move_together():
    rows, prices = [], []
    for i in range(240):
        px = 100.0 + i * 0.25
        rows.append((510.0 + i / 12.0, px, 1.0, "B"))   # buying all the way up
        prices.append(px)
    out = compute(_ticks(rows), _grid(prices), slope_minutes=5)
    assert out["of_cvd_divergence_slope"].iloc[-1] > 0


def test_divergence_slope_is_negative_when_price_rises_on_selling():
    rows, prices = [], []
    for i in range(240):
        px = 100.0 + i * 0.25
        rows.append((510.0 + i / 12.0, px, 1.0, "A"))   # CVD falls while price rises
        prices.append(px)
    out = compute(_ticks(rows), _grid(prices), slope_minutes=5)
    assert out["of_cvd_divergence_slope"].iloc[-1] < 0


# ── session percentile ───────────────────────────────────────────────────

def test_session_pctile_ranks_today_against_prior_sessions():
    rows = _steady(n=240, side="B")                 # today's CVD climbs positive
    grid = _grid([100.0] * 240)
    minutes = sorted({510 + i // 12 for i in range(240)})
    hist = pd.DataFrame(
        {"d1": [-500.0] * len(minutes), "d2": [-400.0] * len(minutes)},
        index=pd.Index(minutes, name="ct_min"),
    )
    out = compute(_ticks(rows), grid, cvd_history=hist)
    assert out["of_cvd_session_pctile"].iloc[-1] == pytest.approx(1.0)


def test_session_pctile_is_nan_without_history():
    out = compute(_ticks(_steady()), _grid([100.0] * 240))
    assert out["of_cvd_session_pctile"].isna().all()


# ── VWAP-cross reversion ─────────────────────────────────────────────────

def test_vwap_reversion_is_positive_when_delta_confirms_the_cross():
    """Price rallies above VWAP, breaks back down through it, and sellers keep
    pressing — delta confirms the cross, so reversion is positive."""
    rows, prices = [], []
    for i in range(120):                    # rally: price above the rising VWAP
        px = 100.0 + i * 0.25
        rows.append((510.0 + i / 12.0, px, 10.0, "B"))
        prices.append(px)
    for i in range(120, 300):               # break down through VWAP, sellers press
        rows.append((510.0 + i / 12.0, 90.0, 10.0, "A"))
        prices.append(90.0)
    out = compute(_ticks(rows), _grid(prices))
    assert out["of_cvd_vwap_reversion"].iloc[-1] > 0


def test_vwap_reversion_stays_nan_when_price_only_departs_equality():
    """Leaving VWAP from exact equality at the open is not a cross."""
    rows, prices = [], []
    for i in range(120):
        rows.append((510.0 + i / 12.0, 120.0, 10.0, "B"))
        prices.append(120.0)
    for i in range(120, 300):
        rows.append((510.0 + i / 12.0, 80.0, 10.0, "A"))
        prices.append(80.0)
    out = compute(_ticks(rows), _grid(prices))
    assert out["of_cvd_vwap_reversion"].isna().all()


def test_vwap_reversion_is_nan_before_any_cross():
    out = compute(_ticks(_steady(n=240, price=100.0)), _grid([100.0] * 240))
    assert out["of_cvd_vwap_reversion"].isna().all()


# ── live-safety ──────────────────────────────────────────────────────────

def test_features_are_live_safe_under_truncation():
    rows, prices = [], []
    for i in range(300):
        px = 100.0 + (5.0 if i % 2 else -5.0)
        rows.append((510.0 + i / 12.0, px, 2.0, "B" if i % 3 else "A"))
        prices.append(px)
    tb = _ticks(rows)
    grid = _grid(prices)
    levels = {"poc": 100.0, "vah": 105.0, "val": 95.0}

    cut = len(grid) // 2
    grid_t = grid.iloc[:cut]
    tb_t = tb.loc[tb["ts_event"] <= grid_t.index[-1]]

    full = compute(tb, grid, levels=levels).iloc[:cut]
    trunc = compute(tb_t, grid_t, levels=levels)

    for c in ("of_cvd_at_level", "lt_cvd_at_level", "of_cvd_divergence_slope"):
        assert np.allclose(
            full[c].to_numpy(), trunc[c].to_numpy(), equal_nan=True
        ), f"{c} leaked future information"
