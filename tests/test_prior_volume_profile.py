"""Tests for the prior-session volume-profile structure module (ADR-122)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.prior_volume_profile import (
    SHAPE_DOUBLE,
    _naked_pocs,
    compute,
    session_levels,
)

CT_OFFSET_MIN = 360  # CT = UTC-6 in January (CST): 08:30 CT == 14:30 UTC


def _bar(day: int, minute: int, o, h, l, c, v):
    """``minute`` is minutes past CT midnight (510 == 08:30 CT)."""
    ts = datetime(2026, 1, day, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minute + CT_OFFSET_MIN
    )
    return {"bucket": ts, "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": float(v)}


def _session(day: int, prices, *, vol=100.0, start_min=510):
    """Build one 08:30 CT session; ``prices`` is a list of (o, h, l, c)."""
    rows = []
    for k, (o, h, l, c) in enumerate(prices):
        rows.append(_bar(day, start_min + k, o, h, l, c, vol))
    return rows


def _frame(rows):
    return pd.DataFrame(rows).set_index("bucket").sort_index()


def _flat_session(day: int, center: float, n: int = 60, spread: float = 5.0):
    return _session(day, [(center, center + spread, center - spread, center)] * n)


def _grid(day: int, n: int = 60, px: float = 100.0, start_min: int = 510):
    """5s price grid over the session for ``day``."""
    base = datetime(2026, 1, day, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=start_min + CT_OFFSET_MIN
    )
    idx = pd.date_range(base, periods=n * 12, freq="5s", tz="UTC")
    return pd.Series(np.full(len(idx), float(px)), index=idx)


# ── primitives ───────────────────────────────────────────────────────────

def test_session_levels_finds_poc_inside_value_area():
    bars = _frame(_flat_session(5, 100.0))
    lv = session_levels(bars)
    assert lv["val"] <= lv["poc"] <= lv["vah"]
    assert lv["low"] == pytest.approx(95.0)
    assert lv["high"] == pytest.approx(105.0)


def test_session_levels_empty_returns_nans():
    lv = session_levels(pd.DataFrame())
    assert all(np.isnan(v) for v in lv.values())


def test_double_distribution_is_detected():
    # Two well-separated volume clusters with a thin gap between them.
    rows = _session(5, [(100, 102, 98, 100)] * 40)
    rows += _session(5, [(140, 142, 138, 140)] * 40, start_min=550)
    rows += _session(5, [(120, 121, 119, 120)] * 2, start_min=590)
    lv = session_levels(_frame(rows))
    assert lv["shape"] == SHAPE_DOUBLE


def test_naked_poc_is_dropped_once_a_later_session_trades_through_it():
    ordered = [
        ("d1", {"poc": 100.0, "high": 105.0, "low": 95.0}),
        ("d2", {"poc": 200.0, "high": 205.0, "low": 195.0}),  # never revisits 100
        ("d3", {"poc": 201.0, "high": 206.0, "low": 196.0}),
    ]
    naked = _naked_pocs(ordered, upto_i=3)
    assert (100.0, 3) in naked          # untested, printed 3 sessions ago
    # d2's POC (200) sits inside d3's range, so it is already tested.
    assert all(p != 200.0 for p, _ in naked)

    # Now let d3 trade back down through 100 — that POC is no longer naked.
    ordered[2] = ("d3", {"poc": 201.0, "high": 206.0, "low": 90.0})
    naked = _naked_pocs(ordered, upto_i=3)
    assert all(p != 100.0 for p, _ in naked)


# ── compute() contract ───────────────────────────────────────────────────

def test_compute_returns_float32_pv_prefixed_frame_on_the_grid():
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 100.0))
    grid = _grid(6)
    out = compute(bars, grid)
    assert list(out.index) == list(grid.resample("5s").last().ffill().index)
    assert all(c.startswith("pv_") for c in out.columns)
    assert all(str(d) == "float32" for d in out.dtypes)


def test_compute_without_a_prior_session_yields_all_nan():
    bars = _frame(_flat_session(6, 100.0))
    out = compute(bars, _grid(6))
    assert out.isna().all().all()


def test_distances_are_atr_normalised_and_signed():
    # Prior session centred at 100 (range 10 → atr 10). Today's price 110.
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 110.0))
    out = compute(bars, _grid(6, px=110.0))
    # 110 is above the prior POC (~100) → positive, ~1 ATR away.
    assert out["pv_dist_poc"].iloc[-1] > 0
    assert out["pv_dist_poc"].iloc[-1] == pytest.approx(1.0, abs=0.25)


def test_price_scale_invariance_of_normalised_distances():
    """Same geometry at NQ-2011 and NQ-2026 price levels must agree."""
    lo = _frame(_flat_session(5, 3000.0, spread=150.0) + _flat_session(6, 3150.0, spread=150.0))
    hi = _frame(_flat_session(5, 25000.0, spread=1250.0) + _flat_session(6, 26250.0, spread=1250.0))
    a = compute(lo, _grid(6, px=3150.0))["pv_dist_poc"].iloc[-1]
    b = compute(hi, _grid(6, px=26250.0))["pv_dist_poc"].iloc[-1]
    assert a == pytest.approx(b, abs=0.05)


def test_zone_ordinal_spans_below_val_to_above_vah():
    bars_below = _frame(_flat_session(5, 100.0) + _flat_session(6, 50.0))
    bars_above = _frame(_flat_session(5, 100.0) + _flat_session(6, 150.0))
    assert compute(bars_below, _grid(6, px=50.0))["pv_zone"].iloc[-1] == 0.0
    assert compute(bars_above, _grid(6, px=150.0))["pv_zone"].iloc[-1] == 4.0


def test_open_type_classifies_gap_above_prior_value_area():
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 150.0))
    out = compute(bars, _grid(6, px=150.0))
    assert out["pv_open_vs_prior_va"].iloc[-1] == 2.0   # opened above VAH
    assert out["pv_gap_atr"].iloc[-1] > 0


def test_gap_is_negative_when_opening_below_prior_close():
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 60.0))
    out = compute(bars, _grid(6, px=60.0))
    assert out["pv_open_vs_prior_va"].iloc[-1] == 0.0
    assert out["pv_gap_atr"].iloc[-1] < 0


def test_va_overlap_is_high_when_today_repeats_yesterday():
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 100.0))
    out = compute(bars, _grid(6, px=100.0))
    assert out["pv_va_overlap_pct"].iloc[-1] > 0.7


def test_va_overlap_is_zero_when_today_trades_entirely_away():
    bars = _frame(_flat_session(5, 100.0) + _flat_session(6, 400.0))
    out = compute(bars, _grid(6, px=400.0))
    assert out["pv_va_overlap_pct"].iloc[-1] == pytest.approx(0.0)


def test_poc_touch_counters_accumulate_only_on_touch():
    # Today oscillates through the prior POC (100) repeatedly.
    prior = _flat_session(5, 100.0)
    today = _session(6, [(100, 105, 95, 100), (120, 122, 118, 120),
                         (100, 104, 96, 100), (120, 122, 118, 120)] * 10)
    out = compute(_frame(prior + today), _grid(6, n=40, px=100.0))
    assert out["pv_poc_test_count_today"].iloc[-1] > 1
    assert out["pv_bars_since_poc_touch"].notna().any()


def test_poc_test_count_stays_zero_when_price_never_reaches_the_level():
    prior = _flat_session(5, 100.0)
    today = _session(6, [(500, 502, 498, 500)] * 40)
    out = compute(_frame(prior + today), _grid(6, n=40, px=500.0))
    assert out["pv_poc_test_count_today"].iloc[-1] == 0.0
    assert out["pv_bars_since_poc_touch"].isna().all()


def test_naked_poc_distances_are_signed_by_side():
    # d5 POC ~100 left untested; d6 trades at 200; today (d7) sits at 200.
    bars = _frame(
        _flat_session(5, 100.0) + _flat_session(6, 200.0) + _flat_session(7, 200.0)
    )
    out = compute(bars, _grid(7, px=200.0))
    assert out["pv_naked_poc_below_dist"].iloc[-1] > 0     # 100 is below 200
    assert out["pv_naked_poc_below_age"].iloc[-1] >= 1


def test_va_acceptance_run_length_grows_inside_the_prior_value_area():
    prior = _flat_session(5, 100.0)
    today = _session(6, [(100, 101, 99, 100)] * 30)   # every close inside prior VA
    out = compute(_frame(prior + today), _grid(6, n=30, px=100.0))
    assert out["pv_va_acceptance"].iloc[-1] > 5


def test_val_rejection_counts_a_probe_below_that_closes_back_inside():
    prior = _flat_session(5, 100.0)
    lv = session_levels(_frame(prior))
    val = lv["val"]
    # Bars that wick below VAL but close above it.
    today = _session(6, [(100, 101, val - 10, 100)] * 20)
    out = compute(_frame(prior + today), _grid(6, n=20, px=100.0))
    assert out["pv_val_rejection"].iloc[-1] > 0


# ── live-safety ──────────────────────────────────────────────────────────

def test_compute_is_live_safe_truncating_the_session_does_not_change_earlier_rows():
    """Values at time t must not depend on bars after t."""
    prior = _flat_session(5, 100.0)
    today = _session(6, [(100, 105, 95, 100), (130, 135, 125, 130)] * 20)
    bars_full = _frame(prior + today)
    grid_full = _grid(6, n=40, px=100.0)

    cut = len(grid_full) // 2
    grid_trunc = grid_full.iloc[:cut]
    keep = bars_full.index <= grid_trunc.index[-1]
    bars_trunc = bars_full.loc[keep]

    full = compute(bars_full, grid_full).iloc[:cut]
    trunc = compute(bars_trunc, grid_trunc)

    causal = [
        "pv_dist_poc", "pv_zone", "pv_bars_since_poc_touch",
        "pv_poc_test_count_today", "pv_va_acceptance",
        "pv_val_rejection", "pv_vah_rejection",
    ]
    for c in causal:
        a = full[c].to_numpy()
        b = trunc[c].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{c} leaked future information"
