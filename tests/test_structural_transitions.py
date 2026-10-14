"""Tests for the ``st_`` structural transition features.

The state machine is the risky part: a break, a failed break and a
retrace-and-continue differ only in what price did *after* crossing an edge, and
all three look identical to a signed distance. Each is constructed by hand here
so the distinction is asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cash_session_profile import SessionLevels
from src.features.structural_transitions import (
    COLUMNS, DIRECTIONAL, TransitionConfig, build_structural_transitions,
)

PRIOR = SessionLevels(session=pd.Timestamp("2026-06-01"), lo=80.0, val=90.0,
                      poc=100.0, vah=110.0, hi=120.0)
R = 10.0


def bars(prices, *, step_s=1, volume=10.0, start="2026-06-02 13:30:00"):
    """1s OHLCV frame from a close path, matching ``tape_to_path``'s shape."""
    idx = pd.date_range(start, periods=len(prices), freq=f"{step_s}s", tz="UTC")
    p = np.asarray(prices, dtype="float64")
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p,
                         "volume": np.full(len(p), volume)}, index=idx)


def build(prices, **kw):
    return build_structural_transitions(bars(prices), PRIOR, r_value=R, **kw)


# ── guards ───────────────────────────────────────────────────────────────────

def test_no_prior_session_is_all_nan():
    out = build_structural_transitions(bars([100.0] * 5), None, r_value=R)
    assert list(out.columns) == list(COLUMNS)
    assert out.isna().all().all()


def test_non_positive_r_is_all_nan():
    out = build_structural_transitions(bars([100.0] * 5), PRIOR, r_value=0.0)
    assert out.isna().all().all()


def test_directional_columns_are_real_columns():
    assert set(DIRECTIONAL) <= set(COLUMNS)


def test_zone_codes_are_symmetric_about_the_poc():
    # A short mirrors a long by negating the frame, which only works if the
    # ordinal has no 0-offset bias.
    out = build([75.0, 85.0, 95.0, 100.0, 105.0, 115.0, 125.0])
    assert list(out["st_zone"]) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


# ── break state machine ──────────────────────────────────────────────────────

def test_break_up_sets_direction_and_signed_extension():
    out = build([105.0, 115.0, 120.0])
    assert out["st_break_age_s"].iloc[0] == -1.0        # no cross yet
    assert out["st_break_level"].iloc[1] == 1.0
    assert out["st_break_extension_r"].iloc[1] == pytest.approx((115 - 110) / R)
    assert out["st_break_age_s"].iloc[2] == 1.0


def test_break_down_is_the_mirror():
    out = build([95.0, 85.0])
    assert out["st_break_level"].iloc[1] == -1.0
    assert out["st_break_extension_r"].iloc[1] == pytest.approx((85 - 90) / R)


def test_drift_to_the_same_distance_is_not_a_break():
    # Starts already outside and never crosses, so no break event exists —
    # this is the case a signed distance cannot separate.
    out = build([115.0, 116.0, 117.0])
    assert (out["st_break_age_s"] == -1.0).all()
    assert (out["st_break_level"] == 0.0).all()


def test_failed_break_sets_reclaim_not_bounce():
    out = build([105.0, 115.0, 98.0])
    assert out["st_reclaim_flag"].iloc[2] == 1.0
    assert out["st_bounce_confirm"].iloc[2] == 0.0
    assert out["st_reclaim_age_s"].iloc[2] == 0.0


def test_a_quickly_undone_break_is_flagged_as_a_rejection():
    out = build([105.0, 115.0, 98.0])
    assert out["st_reject_flag"].iloc[2] == 1.0


def test_a_slowly_undone_break_is_a_reclaim_but_not_a_rejection():
    # The return through the edge lands outside reject_window_s, so it reads as
    # a slow drift back rather than the level rejecting the break.
    idx = pd.DatetimeIndex(["2026-06-02T13:30:00Z", "2026-06-02T13:30:01Z",
                            "2026-06-02T14:30:01Z"])
    p = np.array([105.0, 115.0, 98.0])
    frame = pd.DataFrame({"open": p, "high": p, "low": p, "close": p,
                          "volume": [10.0] * 3}, index=idx)
    out = build_structural_transitions(frame, PRIOR, r_value=R)
    assert out["st_reclaim_flag"].iloc[2] == 1.0
    assert out["st_reject_flag"].iloc[2] == 0.0


def test_retrace_to_the_edge_then_continue_confirms_a_bounce():
    # 111 retraces to within bounce_tol of the broken 110 edge without crossing
    # back, then 118 exceeds the pre-retrace extreme: acceptance, not failure.
    out = build([105.0, 115.0, 111.0, 118.0])
    assert out["st_reclaim_flag"].iloc[3] == 0.0
    assert out["st_bounce_confirm"].iloc[3] == 1.0


def test_bounce_accept_requires_flow_in_the_break_direction():
    prices = [105.0, 115.0, 111.0, 118.0]
    idx = bars(prices).index
    with_flow = pd.Series([0.0, 0.0, 0.0, 12.0], index=idx)
    against = pd.Series([0.0, 0.0, 0.0, -12.0], index=idx)
    assert build(prices, accept=with_flow)["st_bounce_accept"].iloc[3] == 1.0
    assert build(prices, accept=against)["st_bounce_accept"].iloc[3] == 0.0


def test_bounce_accept_is_nan_when_no_flow_series_is_supplied():
    out = build([105.0, 115.0, 111.0, 118.0])
    assert np.isnan(out["st_bounce_accept"].iloc[3])


def test_pullback_and_max_extension_track_the_excursion():
    out = build([105.0, 120.0, 112.0])
    assert out["st_break_max_extension_r"].iloc[2] == pytest.approx(10.0 / R)
    assert out["st_break_pullback_r"].iloc[2] == pytest.approx(8.0 / R)


def test_a_second_break_resets_the_machine():
    out = build([105.0, 115.0, 98.0, 85.0])
    assert out["st_break_level"].iloc[3] == -1.0
    assert out["st_reclaim_flag"].iloc[3] == 0.0
    assert out["st_reject_flag"].iloc[3] == 0.0


# ── value-area entry ─────────────────────────────────────────────────────────

def test_entering_value_from_below_is_signed_positive():
    out = build([85.0, 95.0])
    assert out["st_va_entry_age_s"].iloc[1] == 0.0
    assert out["st_va_entry_depth_r"].iloc[1] == pytest.approx((95 - 90) / R)


def test_entering_value_from_above_is_signed_negative():
    out = build([115.0, 105.0])
    assert out["st_va_entry_depth_r"].iloc[1] == pytest.approx((105 - 110) / R)


def test_time_inside_value_accumulates_only_while_inside():
    out = build([95.0, 95.0, 125.0, 125.0])
    assert out["st_va_time_inside_s"].iloc[1] == 1.0
    assert out["st_va_time_inside_s"].iloc[3] == 1.0


# ── clock correctness ────────────────────────────────────────────────────────

def test_ages_are_wall_clock_not_row_counts():
    # tape_to_path omits seconds with no print, so row distance != elapsed time.
    idx = pd.DatetimeIndex(["2026-06-02T13:30:00Z", "2026-06-02T13:30:01Z",
                            "2026-06-02T13:35:01Z"])
    p = np.array([105.0, 115.0, 116.0])
    frame = pd.DataFrame({"open": p, "high": p, "low": p, "close": p,
                          "volume": [10.0] * 3}, index=idx)
    out = build_structural_transitions(frame, PRIOR, r_value=R)
    assert out["st_break_age_s"].iloc[2] == 300.0


# ── VWAP ─────────────────────────────────────────────────────────────────────

def test_monotone_rise_never_crosses_vwap():
    out = build([100.0, 101.0, 102.0, 103.0])
    assert out["st_vwap_cross_count"].iloc[-1] == 0.0
    assert out["st_vwap_cross_age_s"].iloc[-1] == -1.0


def test_round_trip_through_vwap_registers_a_cross():
    out = build([100.0, 90.0, 90.0, 120.0])
    assert out["st_vwap_cross_count"].iloc[-1] >= 1.0
    assert out["st_vwap_cross_side"].iloc[-1] == 1.0


def test_two_quick_crosses_flag_a_failed_cross():
    out = build([100.0, 90.0, 90.0, 120.0, 80.0])
    assert out["st_vwap_reclaim_flag"].iloc[-1] == 1.0


# ── level-tagged flow ────────────────────────────────────────────────────────

def test_touch_tags_the_level_and_samples_flow_at_that_bar():
    prices = [130.0, 90.5, 130.0]
    idx = bars(prices).index
    acc = pd.Series([0.0, 7.0, 0.0], index=idx)
    rej = pd.Series([0.0, -3.0, 0.0], index=idx)
    out = build(prices, accept=acc, reject=rej)
    assert out["st_level_touch_kind"].iloc[1] == -2.0          # VAL
    assert out["st_level_touch_accept"].iloc[2] == pytest.approx(7.0)
    assert out["st_level_touch_reject"].iloc[2] == pytest.approx(-3.0)
    assert out["st_level_touch_age_s"].iloc[2] == 1.0


def test_untouched_levels_leave_the_touch_age_at_the_sentinel():
    out = build([130.0, 131.0])
    assert (out["st_level_touch_age_s"] == -1.0).all()


def test_touch_tolerance_is_configurable():
    cfg = TransitionConfig(touch_tol_pts=0.1)
    out = build_structural_transitions(bars([90.5]), PRIOR, r_value=R, cfg=cfg)
    assert out["st_level_touch_age_s"].iloc[0] == -1.0


# ── prior range anchors ──────────────────────────────────────────────────────

def test_prior_high_and_low_distances_are_signed():
    out = build([130.0])
    assert out["st_dist_prior_hi_r"].iloc[0] == pytest.approx((130 - 120) / R)
    assert out["st_dist_prior_lo_r"].iloc[0] == pytest.approx((130 - 80) / R)
