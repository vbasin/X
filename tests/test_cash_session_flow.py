"""Tests for cash-session VP levels/bounds and order-flow state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cash_session_profile import (
    LEVEL_KINDS, LEVEL_WEIGHTS, Bound, LevelConfig, SessionLevels, _nearest_peak,
    bound_features, bound_interactions, build_bounds, cash_session_slice,
    entry_levels, levels_for_price, levels_for_range, next_bounds, session_levels,
)
from src.features.flow_state import (
    BODY, LOWER_WICK, UPPER_WICK, FlowConfig, build_flow_state, candle_location,
    classify_large_prints, rolling_impact_residual,
)


def _session(n: int = 23400, start: str = "2026-05-11 13:30:00", seed: int = 3):
    """A 1s session grid starting 08:30 CT (13:30 UTC in CDT)."""
    rng = np.random.default_rng(seed)
    close = 20000 + np.cumsum(rng.normal(0, 0.8, n))
    half = np.abs(rng.normal(0, 0.4, n))
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + half, "low": close - half,
                         "close": close, "volume": rng.integers(1, 25, n)}, index=idx)


# ── cash session window ──────────────────────────────────────────────────────

def test_cash_slice_is_0830_to_1500_chicago():
    day = pd.DataFrame({"close": 1.0},
                       index=pd.date_range("2026-05-11 11:00:00", periods=10 * 3600,
                                           freq="1s", tz="UTC"))
    ct = cash_session_slice(day).index.tz_convert("America/Chicago")
    assert ct.min().strftime("%H:%M:%S") == "08:30:00"
    assert ct.max().strftime("%H:%M:%S") == "14:59:59"


# ── session levels ───────────────────────────────────────────────────────────

def test_session_levels_are_ordered():
    sl = session_levels(_session(), pd.Timestamp("2026-05-11"), LevelConfig())
    assert sl is not None
    assert sl.lo <= sl.val <= sl.poc <= sl.vah <= sl.hi


def test_session_levels_none_outside_cash_hours():
    night = _session(n=3600, start="2026-05-11 04:00:00")
    assert session_levels(night, pd.Timestamp("2026-05-11"), LevelConfig()) is None


# ── clustering into bounds ───────────────────────────────────────────────────

def _levels(prices, session="2026-05-01"):
    return SessionLevels(pd.Timestamp(session), *prices)


def test_nearby_levels_merge_into_one_bound():
    """Levels within a bandwidth of each other raise a single peak."""
    cfg = LevelConfig(sigma_pts=3.0)
    bounds = build_bounds([_levels([100.0, 101.0, 102.0, 103.0, 104.0])], cfg)
    assert len(bounds) == 1
    assert bounds[0].n_levels == 5
    assert bounds[0].lo < 102.0 < bounds[0].hi


def test_distant_levels_stay_separate():
    cfg = LevelConfig(sigma_pts=3.0)
    bounds = build_bounds([_levels([100.0, 101.0, 102.0, 200.0, 201.0])], cfg)
    assert len(bounds) == 2
    assert sorted(b.n_levels for b in bounds) == [2, 3]


def test_sigma_sets_resolution_not_r():
    """The same levels resolve to more bounds as the bandwidth narrows."""
    hist = [_levels([100.0, 104.0, 108.0, 112.0, 116.0])]
    assert len(build_bounds(hist, LevelConfig(sigma_pts=1.0))) == 5
    assert len(build_bounds(hist, LevelConfig(sigma_pts=6.0))) == 1


def test_bound_edges_are_real_member_levels():
    """Variant B invariant: edges are min/max of assigned levels, plus buffer."""
    cfg = LevelConfig(sigma_pts=3.0, buffer_pts=1.0)
    b = build_bounds([_levels([100.0, 101.5, 103.0, 104.0, 107.0])], cfg)[0]
    assert b.lo == pytest.approx(100.0 - cfg.buffer_pts)
    assert b.hi == pytest.approx(107.0 + cfg.buffer_pts)


def test_isolated_level_spans_only_the_buffer():
    cfg = LevelConfig(sigma_pts=2.0, buffer_pts=1.5)
    b = build_bounds([_levels([100.0, 500.0, 900.0, 1300.0, 1700.0])], cfg)[0]
    assert b.hi - b.lo == pytest.approx(2 * cfg.buffer_pts)


def test_buffer_is_clipped_so_bounds_stay_disjoint():
    """A price must belong to at most one zone, or strength is ambiguous."""
    cfg = LevelConfig(sigma_pts=0.5, buffer_pts=10.0)
    bounds = sorted(build_bounds([_levels([100.0, 104.0, 108.0, 112.0, 116.0])], cfg),
                    key=lambda b: b.lo)
    assert len(bounds) > 1
    assert all(a.hi <= b.lo for a, b in zip(bounds, bounds[1:]))


def test_no_bound_can_span_a_wide_price_range():
    """The 435-pt single-linkage artefact: uniformly spaced levels never fuse."""
    hist = [_levels([100.0 + 13 * i + 65 * s for i in range(5)]) for s in range(20)]
    bounds = build_bounds(hist, LevelConfig(sigma_pts=1.0))
    assert max(b.hi - b.lo for b in bounds) < 10.0


def test_grouping_does_not_chain_through_intermediates():
    """Single linkage fused these; nearest-peak assignment cannot.

    Levels are spaced one bandwidth apart across 200 points, so every adjacent
    pair is mergeable under a gap rule and the whole run chains into one zone.
    """
    step = 4.0
    hist = [_levels([100.0 + step * (5 * s + i) for i in range(5)]) for s in range(10)]
    cfg = LevelConfig(sigma_pts=step / 2, lookback_sessions=10)
    bounds = build_bounds(hist, cfg)
    assert len(bounds) > 1
    assert max(b.hi - b.lo for b in bounds) < 10 * step


def test_every_level_is_assigned_to_exactly_one_bound():
    hist = [_levels([100.0 + 7 * i + 31 * s for i in range(5)]) for s in range(6)]
    bounds = build_bounds(hist, LevelConfig(sigma_pts=1.5, lookback_sessions=6))
    assert sum(b.n_levels for b in bounds) == 30


def test_level_joins_its_nearest_peak():
    peak_px = np.array([100.0, 110.0, 200.0])
    px = np.array([99.0, 104.0, 106.0, 150.0, 160.0, 500.0])
    assert list(_nearest_peak(px, peak_px)) == [0, 0, 1, 1, 2, 2]


def test_nearest_peak_breaks_ties_downward():
    assert list(_nearest_peak(np.array([105.0]), np.array([100.0, 110.0]))) == [0]


def test_value_area_edges_outweigh_extremes():
    cfg = LevelConfig(sigma_pts=1.0, recency_halflife=1e9)
    bounds = build_bounds([_levels([10.0, 50.0, 90.0, 130.0, 170.0])], cfg)
    by_kind = {next(iter(b.kinds)): b.strength for b in bounds}
    assert by_kind["vah"] == pytest.approx(LEVEL_WEIGHTS["vah"])
    assert by_kind["vah"] > by_kind["poc"] > by_kind["hi"]


def test_strength_is_the_integrated_basin_mass():
    cfg = LevelConfig(sigma_pts=1.0, recency_halflife=1e9)
    b = build_bounds([_levels([10.0, 500.0, 900.0, 1300.0, 1700.0])], cfg)
    total = sum(x.strength for x in b)
    assert total == pytest.approx(sum(LEVEL_WEIGHTS[k] for k in LEVEL_KINDS))


def test_recency_decay_favours_newer_sessions():
    cfg = LevelConfig(sigma_pts=1.0, recency_halflife=1.0)
    hist = [_levels([1.0, 2.0, 3.0, 4.0, 100.0]),      # newest has hi at 100
            _levels([1.0, 2.0, 3.0, 4.0, 200.0])]
    b = {round(x.mid): x.strength for x in build_bounds(hist, cfg)}
    assert b[100] > b[200]


def test_more_sessions_agreeing_gives_more_strength():
    cfg = LevelConfig(sigma_pts=1.0, recency_halflife=1e9)
    one = build_bounds([_levels([100.0] * 5)], cfg)[0]
    three = build_bounds([_levels([100.0] * 5)] * 3, cfg)[0]
    assert three.strength > one.strength
    assert three.n_levels == 15


def test_negligible_peaks_are_dropped():
    cfg = LevelConfig(sigma_pts=1.0, recency_halflife=1.0, min_peak_mass=0.5)
    # An age-9 `hi` carries 0.40 * 0.5**9 — far below the floor.
    hist = [_levels([100.0] * 5) for _ in range(9)] + [_levels([500.0] * 5)]
    assert all(b.mid < 200.0 for b in build_bounds(hist, cfg))


# ── lookback extension ───────────────────────────────────────────────────────

def test_lookback_extends_when_price_is_above_everything():
    cfg = LevelConfig(lookback_sessions=2, extend_step=2, sigma_pts=1.0,
                      max_lookback_sessions=10)
    hist = [_levels([100.0, 101.0, 102.0, 103.0, 104.0]) for _ in range(2)]
    hist += [_levels([500.0, 501.0, 502.0, 503.0, 504.0]) for _ in range(4)]
    bounds, used = levels_for_price(hist, price=300.0, cfg=cfg)
    assert used > cfg.lookback_sessions
    assert max(b.hi for b in bounds) > 300.0


def test_lookback_does_not_extend_when_price_is_bracketed():
    cfg = LevelConfig(lookback_sessions=2, extend_step=2)
    hist = [_levels([100.0, 110.0, 120.0, 130.0, 140.0]) for _ in range(6)]
    _, used = levels_for_price(hist, price=125.0, cfg=cfg)
    assert used == 2


def test_lookback_stops_at_max():
    cfg = LevelConfig(lookback_sessions=2, extend_step=2, max_lookback_sessions=4)
    hist = [_levels([100.0, 101.0, 102.0, 103.0, 104.0]) for _ in range(20)]
    _, used = levels_for_price(hist, price=10_000.0, cfg=cfg)
    assert used <= 4


def test_build_bounds_exposes_member_levels():
    cfg = LevelConfig(sigma_pts=3.0, buffer_pts=1.0)
    hist = [_levels([100.0, 101.0, 102.0, 103.0, 104.0], session="2026-05-01"),
            _levels([100.5, 101.5, 102.5, 103.5, 104.5], session="2026-05-02")]
    bounds = build_bounds(hist, cfg)
    assert bounds
    members = bounds[0].members
    assert members
    assert all(kind in LEVEL_KINDS for kind, _, _ in members)
    assert all(isinstance(price, float) for _, price, _ in members)
    assert all(isinstance(sess, pd.Timestamp) for _, _, sess in members)
    lo_sess = min(members, key=lambda m: (m[1], m[2]))[2]
    hi_sess = max(members, key=lambda m: (m[1], m[2]))[2]
    assert lo_sess == pd.Timestamp("2026-05-01")
    assert hi_sess == pd.Timestamp("2026-05-02")


def test_levels_for_range_extends_when_day_hi_is_outside():
    cfg = LevelConfig(lookback_sessions=2, extend_step=2, sigma_pts=1.0,
                      max_lookback_sessions=10)
    hist = [_levels([100.0, 101.0, 102.0, 103.0, 104.0]) for _ in range(2)]
    hist += [_levels([500.0, 501.0, 502.0, 503.0, 504.0]) for _ in range(4)]
    bounds, used = levels_for_range(hist, lo=101.0, hi=300.0, cfg=cfg)
    assert used > cfg.lookback_sessions
    span_lo = min(b.lo for b in bounds)
    span_hi = max(b.hi for b in bounds)
    assert span_lo <= 101.0
    assert span_hi >= 300.0


def test_levels_for_range_matches_levels_for_price_on_a_point():
    cfg = LevelConfig(lookback_sessions=3, extend_step=2, sigma_pts=2.0)
    hist = [_levels([100.0 + i, 110.0 + i, 120.0 + i, 130.0 + i, 140.0 + i])
            for i in range(8)]
    a, ua = levels_for_price(hist, price=125.0, cfg=cfg)
    b, ub = levels_for_range(hist, lo=125.0, hi=125.0, cfg=cfg)
    assert ua == ub
    assert [(x.lo, x.hi, x.n_levels) for x in a] == [(x.lo, x.hi, x.n_levels) for x in b]


def test_next_bounds_brackets_price():
    bounds = [Bound(90, 95, 1, 1), Bound(100, 105, 1, 1), Bound(120, 125, 1, 1)]
    inside, above, below = next_bounds(bounds, 110.0)
    assert inside is None and above.lo == 120 and below.hi == 105
    inside, _, _ = next_bounds(bounds, 102.0)
    assert inside is not None and inside.lo == 100


# ── interaction counters ─────────────────────────────────────────────────────

def test_break_counted_when_price_traverses_a_zone():
    b = [Bound(99.0, 101.0, 1.0, 1)]
    prices = np.array([95.0, 100.0, 105.0])          # below -> inside -> above
    out = bound_interactions(prices, b)
    assert out["bi_tests_total"].iloc[-1] == 1
    assert out["bi_breaks_total"].iloc[-1] == 1
    assert out["bi_rejects_total"].iloc[-1] == 0


def test_rejection_counted_when_price_turns_back():
    b = [Bound(99.0, 101.0, 1.0, 1)]
    prices = np.array([95.0, 100.0, 95.0])           # below -> inside -> below
    out = bound_interactions(prices, b)
    assert out["bi_rejects_total"].iloc[-1] == 1
    assert out["bi_breaks_total"].iloc[-1] == 0


def test_counters_are_cumulative_and_causal():
    b = [Bound(99.0, 101.0, 1.0, 1)]
    prices = np.array([95.0, 100.0, 95.0, 100.0, 105.0])
    out = bound_interactions(prices, b)
    assert list(out["bi_tests_total"]) == [0, 1, 1, 2, 2]
    assert out["bi_rejects_total"].iloc[2] == 1       # known only after the turn
    assert out["bi_breaks_total"].iloc[-1] == 1


def test_bound_features_report_channel_position():
    bounds = [Bound(90, 95, 2.0, 3), Bound(120, 125, 1.0, 1)]
    f = bound_features(bounds, price=107.5, r_value=10.0)
    assert f["lv_inside_bound"] == 0.0
    assert f["lv_dist_above_r"] == pytest.approx((120 - 107.5) / 10)
    assert f["lv_dist_below_r"] == pytest.approx((107.5 - 95) / 10)
    assert f["lv_pos_in_channel"] == pytest.approx(0.5)


# ── flow: candle location ─────────────────────────────────────────────────

def test_candle_location_splits_body_and_wicks():
    o = np.array([10.0] * 3)
    c = np.array([12.0] * 3)
    h = np.array([13.0] * 3)
    l = np.array([9.0] * 3)
    loc = candle_location(np.array([11.0, 12.5, 9.5]), o, h, l, c)
    assert list(loc) == [BODY, UPPER_WICK, LOWER_WICK]


# ── flow: the large-print taxonomy ───────────────────────────────────────────

def _bar(o, h, l, c, start="2026-05-11 13:30:00"):
    """Two consecutive start-labelled 1s bars: the signal bar at *start* (its
    OHLC is what a print inside [start, start+1) is classified against), plus
    a throwaway next bar -- a print is only visible once its own bar
    completes, so it needs somewhere causal to land.
    """
    t0 = pd.Timestamp(start, tz="UTC")
    idx = pd.DatetimeIndex([t0, t0 + pd.Timedelta(seconds=1)])
    return pd.DataFrame({"open": [o, c], "high": [h, c], "low": [l, c],
                         "close": [c, c], "volume": [100.0, 100.0]}, index=idx)


def _print(price, side, size=50, ts="2026-05-11 13:30:00.500000"):
    return pd.DataFrame({"price": [price], "size": [size], "side": [side]},
                        index=pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")]))


@pytest.mark.parametrize("price,side,expect,want_sign", [
    (107.0, "B", "accept_continue", 1),   # buy in the body of an up candle
    (101.0, "A", "absorb_continue", 1),   # sell in the bottom half of an up candle
    (115.0, "B", "absorb_reversal", -1),  # buy stalling in the upper wick
    (115.0, "A", "reject_reversal", -1),  # sell rejected at the upper wick
])
def test_up_candle_taxonomy(price, side, expect, want_sign):
    bars = _bar(100.0, 120.0, 99.0, 110.0)           # up candle, body 100-110, mid 105
    tax = classify_large_prints(_print(price, side), bars, FlowConfig())
    assert not tax.empty
    row = tax.iloc[0]
    assert row[expect] != 0.0, f"{expect} should fire"
    assert np.sign(row[expect]) == want_sign
    others = ({"accept_continue", "absorb_continue", "absorb_reversal",
              "reject_reversal"} - {expect})
    assert all(row[o] == 0.0 for o in others)


def test_continuation_scaled_by_body_over_full_range():
    bars = _bar(100.0, 120.0, 99.0, 110.0)   # body_range=10, full_range=21
    tax = classify_large_prints(_print(107.0, "B", size=40), bars, FlowConfig())
    norm_size = 40.0 / FlowConfig().large_order_size
    expected = norm_size * (10.0 / 21.0)
    assert tax.iloc[0]["accept_continue"] == pytest.approx(expected)


def test_reversal_scaled_by_wick_over_body_and_capped():
    bars = _bar(100.0, 120.0, 99.0, 110.0)   # upper_wick_range=10, body_range=10
    tax = classify_large_prints(_print(115.0, "B", size=20), bars, FlowConfig())
    norm_size = 20.0 / FlowConfig().large_order_size
    assert tax.iloc[0]["absorb_reversal"] == pytest.approx(-norm_size * 1.0)

    # A near-zero body must not blow the ratio up unbounded.
    doji = _bar(105.0, 120.0, 99.0, 105.01)
    cfg = FlowConfig(max_reversal_scale=10.0)
    tax2 = classify_large_prints(_print(115.0, "B", size=20), doji, cfg)
    assert abs(tax2.iloc[0]["absorb_reversal"]) == pytest.approx(norm_size * cfg.max_reversal_scale)


def test_zero_range_bar_yields_no_continuation_signal():
    bars = _bar(100.0, 100.0, 100.0, 100.0)   # full_range == 0
    tax = classify_large_prints(_print(100.0, "B"), bars, FlowConfig())
    assert tax.iloc[0]["accept_continue"] == 0.0


def test_taxonomy_is_signed_bullish_positive():
    up = _bar(100.0, 120.0, 99.0, 110.0)
    assert classify_large_prints(_print(107.0, "B"), up, FlowConfig()).iloc[0]["accept_continue"] > 0


def test_down_candle_mirrors_the_up_candle_taxonomy():
    bars = _bar(110.0, 120.0, 99.0, 100.0)           # down candle, body 100-110
    tax = classify_large_prints(_print(105.0, "A"), bars, FlowConfig())
    # Trend acceptance in a down candle is bearish, so the signed column is negative.
    assert tax.iloc[0]["accept_continue"] < 0


def test_small_prints_are_ignored():
    bars = _bar(100.0, 120.0, 99.0, 110.0)
    cfg = FlowConfig(large_order_size=20)
    assert classify_large_prints(_print(105.0, "B", size=5), bars, cfg).empty


# ── flow: absorption residual ────────────────────────────────────────────────

def test_absorption_residual_positive_when_price_does_not_follow_flow():
    n = 2000
    idx = pd.date_range("2026-05-11 13:30:00", periods=n, freq="1s", tz="UTC")
    rng = np.random.default_rng(0)
    ofi = pd.Series(rng.normal(0, 10, n), index=idx)
    dprice = ofi * 0.5                                # a clean beta of 0.5
    dprice.iloc[-1] = 0.0                             # last bar: flow, but no move
    ofi.iloc[-1] = 100.0
    r = rolling_impact_residual(dprice, ofi, window=500, min_obs=100)
    assert r.iloc[-1] > 0                             # absorbed


def test_absorption_residual_is_causal():
    n = 800
    idx = pd.date_range("2026-05-11 13:30:00", periods=n, freq="1s", tz="UTC")
    rng = np.random.default_rng(1)
    ofi = pd.Series(rng.normal(0, 5, n), index=idx)
    dprice = ofi * 0.3
    full = rolling_impact_residual(dprice, ofi, 300, 60)
    cut = rolling_impact_residual(dprice.iloc[:600], ofi.iloc[:600], 300, 60)
    pd.testing.assert_series_equal(full.iloc[:600], cut, check_freq=False)


# ── flow: assembled state ────────────────────────────────────────────────────

def test_build_flow_state_emits_prefixed_columns():
    bars = _session(n=3600)
    bars["delta"] = np.random.default_rng(0).normal(0, 5, len(bars))
    bars["ofi"] = bars["delta"]
    fs = build_flow_state(bars, None, FlowConfig(), r_value=40.0)
    assert len(fs) == len(bars)
    assert all(c.startswith("fs_") for c in fs.columns)
    for key in ("fs_vwap_velocity_r", "fs_vwap_accel_r",
                "fs_ema6_dist_r", "fs_absorption_1m", "fs_impact_efficiency_5m",
                "fs_flow_score", "fs_flow_score_5bar", "fs_last_reversal_level",
                "fs_vwap_velocity_5m_r", "fs_vwap_velocity_15m_r",
                "fs_ema20_slope_r", "fs_ema60_slope_r",
                "fs_ema20_vwap_diff_r", "fs_ema60_vwap_diff_r",
                "fs_ema6_slope_vwap_diff_r", "fs_ema20_slope_vwap_diff_r",
                "fs_ema60_slope_vwap_diff_r", "fs_ema6_slope_vwap_diff_accel_r",
                "fs_ema20_slope_vwap_diff_accel_r", "fs_ema60_slope_vwap_diff_accel_r"):
        assert key in fs.columns


def test_ema6_warmup_uses_prior_minutes():
    """Cash path starting at 08:30 must see 08:29… closes, not a cold EMA."""
    bars = _session(n=180, start="2026-05-11 13:30:00+00:00")  # 08:30 CT
    cold = build_flow_state(bars, None, FlowConfig(), r_value=40.0)
    prior_idx = pd.date_range("2026-05-11 13:24:00+00:00", periods=6, freq="1min")
    # Right-labelled 1min closes: 13:25 … 13:30 are the minutes before/at open.
    warm = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], index=prior_idx)
    hot = build_flow_state(bars, None, FlowConfig(), r_value=40.0, warmup_close_1m=warm)
    assert np.isfinite(hot["fs_ema6_dist_r"].iloc[0])
    assert hot["fs_ema6_dist_r"].iloc[0] != cold["fs_ema6_dist_r"].iloc[0]


def test_flow_state_without_tbbo_zeroes_the_taxonomy():
    bars = _session(n=1200)
    fs = build_flow_state(bars, None, FlowConfig(), r_value=40.0)
    assert (fs["fs_accept_continue"].fillna(0) == 0).all()
    assert (fs["fs_flow_score"].fillna(0) == 0).all()
    assert (fs["fs_flow_score_5bar"].fillna(0) == 0).all()
    assert fs["fs_last_reversal_level"].isna().all()


def test_ema_slope_vwap_diff_equals_component_slopes():
    bars = _session(n=600)
    fs = build_flow_state(bars, None, FlowConfig(), r_value=40.0)
    for tag in ("6", "20", "60"):
        expected = fs[f"fs_ema{tag}_slope_r"] - fs["fs_vwap_velocity_r"]
        pd.testing.assert_series_equal(fs[f"fs_ema{tag}_slope_vwap_diff_r"], expected,
                                       check_names=False)


def test_ema_slope_vwap_diff_accel_is_the_diffs_own_slope():
    bars = _session(n=600)
    cfg = FlowConfig()
    fs = build_flow_state(bars, None, cfg, r_value=40.0)
    for tag in ("6", "20", "60"):
        expected = fs[f"fs_ema{tag}_slope_vwap_diff_r"].diff(cfg.fast_window)
        pd.testing.assert_series_equal(fs[f"fs_ema{tag}_slope_vwap_diff_accel_r"], expected,
                                       check_names=False)


def test_vwap_velocity_5m_and_15m_use_their_own_windows():
    bars = _session(n=1200)
    r_value = 40.0
    cfg = FlowConfig()
    fs = build_flow_state(bars, None, cfg, r_value=r_value)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    cum_pv = np.cumsum(tp * bars["volume"])
    cum_v = np.maximum(np.cumsum(bars["volume"]), 1e-9)
    vwap = pd.Series((cum_pv / cum_v).to_numpy(), index=bars.index)
    expected_5m = vwap.diff(cfg.vwap_vel_window_5m) / r_value
    expected_15m = vwap.diff(cfg.vwap_vel_window_15m) / r_value
    pd.testing.assert_series_equal(fs["fs_vwap_velocity_5m_r"], expected_5m, check_names=False)
    pd.testing.assert_series_equal(fs["fs_vwap_velocity_15m_r"], expected_15m, check_names=False)


def test_flow_score_is_sum_of_the_four_categories():
    bars = _session(n=200)
    print_ts = (bars.index[50] - pd.Timedelta(seconds=1)).tz_localize(None)
    trades = _print(107.0, "B", size=40, ts=print_ts)
    fs = build_flow_state(bars, trades, FlowConfig(), r_value=40.0)
    total = (fs["fs_accept_continue"] + fs["fs_absorb_continue"]
            + fs["fs_absorb_reversal"] + fs["fs_reject_reversal"])
    pd.testing.assert_series_equal(fs["fs_flow_score"], total, check_names=False)


def test_last_reversal_level_expires_after_five_minutes():
    bars = _session(n=1200)
    cfg = FlowConfig()
    # Force a deterministic up candle so the reversal (upper-wick) side is known.
    bars.loc[bars.index[100], ["open", "high", "low", "close"]] = [100.0, 120.0, 99.0, 110.0]
    # A print inside bar 100's own window [t100, t100+1s) is classified
    # against bar 100's OHLC but only becomes visible once bar 100 completes,
    # i.e. at bar 101's row.
    print_ts = (bars.index[100] + pd.Timedelta(milliseconds=500)).tz_localize(None)
    trades = _print(115.0, "B", size=40, ts=print_ts)  # buy in the upper wick
    fs = build_flow_state(bars, trades, cfg, r_value=40.0)
    vis = bars.index[101]
    at_print = fs.index.get_indexer([vis])[0]
    just_before_expiry = fs.index.get_indexer(
        [vis + pd.Timedelta(seconds=cfg.slow_window - 1)])[0]
    after_expiry = fs.index.get_indexer(
        [vis + pd.Timedelta(seconds=cfg.slow_window + 1)])[0]
    assert fs["fs_last_reversal_level"].iloc[at_print] == pytest.approx(115.0)
    assert fs["fs_last_reversal_level"].iloc[just_before_expiry] == pytest.approx(115.0)
    assert np.isnan(fs["fs_last_reversal_level"].iloc[after_expiry])


def test_bound_geometry_has_no_r_dependence():
    """R sets how far a target is worth aiming at, not what counts as a level."""
    assert not hasattr(LevelConfig, "scaled")
    assert not any(f.endswith("_r") for f in LevelConfig.__dataclass_fields__)


def test_default_resolution_is_tick_based():
    d = LevelConfig()
    assert d.sigma_pts == 5.0 and d.grid_step == 0.25      # grid = 1 MNQ tick
    assert d.buffer_pts == 1.0


def test_bounds_are_price_scale_invariant():
    """Identical structure at 3,000 and 30,000 yields identical widths."""
    cfg = LevelConfig(sigma_pts=1.0)
    lo = build_bounds([_levels([3000.0 + 20 * i for i in range(5)])], cfg)
    hi = build_bounds([_levels([30000.0 + 20 * i for i in range(5)])], cfg)
    assert len(lo) == len(hi) == 5
    assert np.allclose(sorted(b.hi - b.lo for b in lo),
                       sorted(b.hi - b.lo for b in hi))


def test_wider_clustering_yields_fewer_bounds():
    hist = [_levels([100.0, 108.0, 116.0, 124.0, 132.0])]
    tight = build_bounds(hist, LevelConfig(sigma_pts=1.0))
    wide = build_bounds(hist, LevelConfig(sigma_pts=12.0))
    assert len(tight) > len(wide) == 1


# ── disjointness ─────────────────────────────────────────────────────────

def test_bounds_never_overlap():
    """A price must belong to at most one zone, or strength is ambiguous."""
    hist = [_levels([100.0 + 3 * i + 11 * s for i in range(5)]) for s in range(5)]
    bounds = sorted(build_bounds(hist, LevelConfig(sigma_pts=1.5)),
                    key=lambda b: b.lo)
    assert all(a.hi <= b.lo for a, b in zip(bounds, bounds[1:]))


def test_every_level_lands_in_exactly_one_basin():
    hist = [_levels([100.0 + 3 * i + 11 * s for i in range(5)]) for s in range(5)]
    bounds = build_bounds(hist, LevelConfig(sigma_pts=1.5, min_peak_mass=0.0))
    assert sum(b.n_levels for b in bounds) == 25


# ── resting entry levels ────────────────────────────────────────────────

def test_entry_rests_at_the_far_edge_of_the_containing_bound():
    bounds = [Bound(90.0, 110.0, 1.0, 3)]
    lng, sht = entry_levels(bounds, 100.0, LevelConfig())
    assert lng == pytest.approx(110.0)
    assert sht == pytest.approx(90.0)


def test_entry_uses_a_confirmation_offset_between_bounds():
    bounds = [Bound(50.0, 60.0, 1.0, 1), Bound(150.0, 160.0, 1.0, 1)]
    cfg = LevelConfig(entry_offset_pts=2.0)
    lng, sht = entry_levels(bounds, 100.0, cfg)
    assert lng == pytest.approx(102.0) and sht == pytest.approx(98.0)


def test_entry_never_rests_behind_the_decision_price():
    # Price sitting exactly on an edge must not produce an instant fill.
    bounds = [Bound(90.0, 100.0, 1.0, 2)]
    lng, sht = entry_levels(bounds, 100.0, LevelConfig())
    assert lng > 100.0 and sht < 100.0


def test_flow_state_has_no_infinities():
    bars = _session(n=1200)
    bars["delta"] = 0.0                               # zero flow -> division risk
    bars["ofi"] = 0.0
    fs = build_flow_state(bars, None, FlowConfig(), r_value=40.0)
    assert not np.isinf(fs.to_numpy(dtype="float64")).any()
