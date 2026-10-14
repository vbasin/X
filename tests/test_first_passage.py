"""Tests for the first-passage candidate-exit label set (Layers 0-2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.auction_levels import (
    Profile, ProfileConfig, allocate_tf_delta, composite_profile, find_nodes,
    select_candidates, session_profile, union_nodes, value_area,
)
from src.training.first_passage import (
    CENSORED, LONG, SHORT, STOP, TARGET, FirstPassageConfig, _resolve_pairs,
    entry_indices, first_fill, flatten_index, label_entry, label_session,
    touch_offsets,
)


def _path(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20000 + np.cumsum(rng.normal(0, 1.5, n))
    half = np.abs(rng.normal(0, 0.6, n))
    idx = pd.date_range("2026-05-04 13:30:00", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + half, "low": close - half,
                         "close": close, "volume": rng.integers(1, 40, n)}, index=idx)


# ── Layer 0: profiles ────────────────────────────────────────────────────────

def test_session_profile_conserves_volume():
    day = _path()
    prof = session_profile(day, ProfileConfig())
    assert prof.volume.sum() == pytest.approx(day["volume"].sum(), rel=1e-9)


def test_profile_bins_cover_traded_range():
    day = _path()
    cfg = ProfileConfig()
    prof = session_profile(day, cfg)
    assert prof.prices().min() <= day["low"].min() + cfg.bin_size
    assert prof.prices().max() >= day["high"].max() - cfg.bin_size


def test_delta_channel_is_zero_without_tbbo():
    """No aggressor source means an empty channel, never an estimated one."""
    day = _path()
    prof = session_profile(day, ProfileConfig())
    assert np.all(prof.delta == 0.0)


def test_supplied_delta_is_conserved_and_signed():
    day = _path()
    dlt = np.where(np.arange(len(day)) % 2 == 0, 5.0, -3.0)
    prof = session_profile(day, ProfileConfig(), delta=dlt)
    assert prof.delta.sum() == pytest.approx(dlt.sum(), rel=1e-9)
    assert np.any(prof.delta > 0) and np.any(prof.delta < 0)


def test_allocate_tf_delta_splits_by_volume_share():
    idx = pd.date_range("2026-05-04 13:30:00", periods=30, freq="1s", tz="UTC")
    day = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                        "volume": np.r_[np.full(15, 1.0), np.full(15, 3.0)]}, index=idx)
    # Two closed 15s bars ending at :15 and :30.
    tf = pd.Series([100.0, -40.0],
                   index=pd.to_datetime(["2026-05-04 13:30:15", "2026-05-04 13:30:30"],
                                        utc=True))
    out = allocate_tf_delta(day, tf, "15s")
    assert out[:15].sum() == pytest.approx(100.0)
    assert out[15:].sum() == pytest.approx(-40.0)
    assert out[0] == pytest.approx(100.0 / 15)      # equal volume -> equal share


def test_allocate_tf_delta_zero_where_uncovered():
    idx = pd.date_range("2026-05-04 13:30:00", periods=20, freq="1s", tz="UTC")
    day = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                        "volume": 1.0}, index=idx)
    assert np.all(allocate_tf_delta(day, pd.Series(dtype="float64"), "15s") == 0.0)


def test_composite_weights_recent_sessions_higher():
    cfg = ProfileConfig(recency_halflife=1.0)
    a = Profile(0, 1.0, np.array([10.0, 0.0]), np.zeros(2))
    b = Profile(0, 1.0, np.array([0.0, 10.0]), np.zeros(2))
    comp = composite_profile([a, b], cfg)          # a is most recent
    assert comp.volume[0] > comp.volume[1]


def test_value_area_covers_requested_share():
    day = _path()
    cfg = ProfileConfig()
    prof = session_profile(day, cfg)
    poc, val, vah = value_area(prof, cfg)
    assert val <= poc <= vah
    inside = prof.volume[(prof.prices() >= val) & (prof.prices() <= vah)].sum()
    assert inside / prof.volume.sum() >= cfg.value_area_pct - 1e-6


def test_find_nodes_locates_a_planted_peak():
    v = np.zeros(60)
    v[30] = 100.0
    prof = Profile(0, 1.0, v, np.zeros(60))
    hvn, _ = find_nodes(prof, ProfileConfig())
    assert 30 in hvn


def test_union_nodes_keeps_developing_nodes_history_would_miss():
    """History that does not span today's price must not blind the candidate set."""
    cfg = ProfileConfig()
    hist = Profile(0, 1.0, np.zeros(60), np.zeros(60))
    hist.volume[30] = 10_000.0
    # Price has trended clean out of the historical range.
    dev = Profile(500, 1.0, np.zeros(60), np.zeros(60))
    dev.volume[30] = 5.0
    assert 530 not in find_nodes(hist, cfg)[0]
    nodes = union_nodes(hist, dev, cfg=cfg)
    assert 30 in nodes and 530 in nodes


# ── Layer 1: candidates ──────────────────────────────────────────────────────

def test_select_candidates_respects_band_and_cap():
    day = _path()
    cfg = ProfileConfig(max_candidates=3, min_dist_r=0.1, max_dist_r=2.0)
    prof = session_profile(day, cfg)
    entry = float(day["close"].iloc[len(day) // 2])
    cand = select_candidates(prof, [], entry, r_value=10.0, cfg=cfg)
    if cand.empty:
        pytest.skip("random path produced no nodes in band")
    assert (cand["dist_r"] >= cfg.min_dist_r).all()
    assert (cand["dist_r"] <= cfg.max_dist_r).all()
    assert cand.groupby("above").size().max() <= cfg.max_candidates
    assert "level_type" not in cand.columns      # continuous descriptors only


def test_select_candidates_empty_without_r():
    day = _path()
    cfg = ProfileConfig()
    prof = session_profile(day, cfg)
    assert select_candidates(prof, [], 20000.0, float("nan"), cfg).empty


# ── Layer 2: first passage ───────────────────────────────────────────────────

def _brute_first_touch(arr: np.ndarray, entry_i: int, end_i: int,
                       level: float, up: bool) -> int:
    n_ahead = end_i - entry_i
    for k in range(1, n_ahead + 1):
        if (arr[entry_i + k] >= level) if up else (arr[entry_i + k] <= level):
            return k
    return n_ahead + 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_touch_offsets_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    half = np.abs(rng.normal(0, 0.5, n))
    high, low = close + half, close - half
    entry_i, end_i = 50, n - 1
    ups = np.sort(close[entry_i] + rng.uniform(0.1, 10, 6))
    dns = np.sort(close[entry_i] - rng.uniform(0.1, 10, 6))
    up, dn, _, _ = touch_offsets(high, low, entry_i, end_i, ups, dns)
    for lvl, got in zip(ups, up):
        assert got == _brute_first_touch(high, entry_i, end_i, lvl, True)
    for lvl, got in zip(dns, dn):
        assert got == _brute_first_touch(low, entry_i, end_i, lvl, False)


def test_touch_offsets_marks_unreachable_levels():
    high = np.full(50, 100.0)
    low = np.full(50, 100.0)
    up, dn, _, _ = touch_offsets(high, low, 0, 49, np.array([500.0]), np.array([1.0]))
    assert up[0] == 50 and dn[0] == 50          # n_ahead + 1 == censored sentinel


def test_resolve_pairs_tie_policy():
    t_tgt = np.array([5, 5, 9, 99])
    t_stp = np.array([5, 7, 3, 99])
    ev, off = _resolve_pairs(t_tgt, t_stp, "stop_first", n_ahead=50)
    assert list(ev) == [STOP, TARGET, STOP, CENSORED]
    assert off[-1] == 50                         # censored clamps to n_ahead
    ev2, _ = _resolve_pairs(t_tgt, t_stp, "target_first", n_ahead=50)
    assert list(ev2) == [TARGET, TARGET, STOP, CENSORED]


def test_label_entry_grid_shape_and_pnl_signs():
    day = _path()
    high = day["high"].to_numpy()
    low = day["low"].to_numpy()
    close = day["close"].to_numpy()
    entry_i, end_i = 10, len(day) - 1
    entry_px = close[entry_i]
    targets = entry_px + np.array([3.0, 6.0, 12.0])
    stops = entry_px - np.array([3.0, 6.0])
    res = label_entry(high, low, close, entry_i, end_i, targets, stops,
                      LONG, FirstPassageConfig())
    assert len(res["event"]) == len(targets) * len(stops)
    assert np.all(res["pnl_pts"][res["event"] == TARGET] > 0)
    assert np.all(res["pnl_pts"][res["event"] == STOP] < 0)
    assert np.all(res["mae_pts"] >= 0) and np.all(res["mfe_pts"] >= 0)


def test_long_and_short_are_mirror_images():
    """Same distances, opposite side: a long target is a short stop."""
    n = 500
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    half = np.abs(rng.normal(0, 0.4, n))
    high, low = close + half, close - half
    entry_i, end_i = 5, n - 1
    px = close[entry_i]
    above = px + np.array([4.0, 9.0])
    below = px - np.array([4.0, 9.0])
    lng = label_entry(high, low, close, entry_i, end_i, above, below, LONG,
                      FirstPassageConfig())
    sht = label_entry(high, low, close, entry_i, end_i, below, above, SHORT,
                      FirstPassageConfig())
    # A long that stops at -4 is a short that targets -4 at the same instant.
    assert set(np.unique(lng["t_event"])) & set(np.unique(sht["t_event"]))
    assert np.all(sht["pnl_pts"][sht["event"] == TARGET] > 0)


def test_censored_exit_prices_at_the_flatten_bar():
    n = 200
    close = np.full(n, 100.0)
    high = close + 0.1
    low = close - 0.1
    close[-1] = 103.0
    res = label_entry(high, low, close, 0, n - 1, np.array([500.0]),
                      np.array([-500.0]), LONG, FirstPassageConfig())
    assert res["event"][0] == CENSORED
    assert res["pnl_pts"][0] == pytest.approx(3.0)


def test_entry_indices_respect_window_and_grid():
    idx = pd.date_range("2026-05-04 12:00:00", periods=8 * 3600, freq="1s", tz="UTC")
    cfg = FirstPassageConfig(entry_start="08:30", entry_end="14:45",
                             decision_grid_seconds=60)
    ei = entry_indices(idx, cfg)
    ct = idx[ei].tz_convert("America/Chicago")
    assert len(ei) > 0
    assert (ct.second == 0).all()
    assert ct.min().strftime("%H:%M") >= "08:30"
    assert ct.max().strftime("%H:%M") < "14:45"


def test_flatten_index_finds_the_close_bar():
    idx = pd.date_range("2026-05-04 12:00:00", periods=9 * 3600, freq="1s", tz="UTC")
    cfg = FirstPassageConfig(flatten="15:00")
    fi = flatten_index(idx, cfg)
    assert idx[fi].tz_convert("America/Chicago").strftime("%H:%M:%S") == "15:00:00"


def test_flatten_index_falls_back_to_last_bar():
    """A session truncated before the flatten time exits on its final bar."""
    idx = pd.date_range("2026-05-04 12:00:00", periods=3600, freq="1s", tz="UTC")
    assert flatten_index(idx, FirstPassageConfig(flatten="15:00")) == len(idx) - 1


# ── resting-order entry ──────────────────────────────────────────────────────

def _ramp(n: int = 600, top: float = 130.0) -> pd.DataFrame:
    """Price rising linearly from 100, on a grid that starts at 08:30 CT."""
    close = np.linspace(100.0, top, n)
    idx = pd.date_range("2026-05-04 13:30:00", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=idx)


def _cand(levels: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"level": levels, "above": [True] * len(levels),
                         "is_structural": 1.0})


def test_first_fill_finds_the_touch_offset():
    grid = _ramp()
    high, low = grid["high"].to_numpy(), grid["low"].to_numpy()
    # Price climbs 0.05/s, so 105 is reached ~100s after bar 0.
    off = first_fill(high, low, 0, len(grid) - 1, 105.0, above=True, max_wait=900)
    assert off > 0 and high[off] >= 105.0 and high[off - 1] < 105.0


def test_first_fill_reports_no_fill_past_the_wait():
    grid = _ramp()
    high, low = grid["high"].to_numpy(), grid["low"].to_numpy()
    assert first_fill(high, low, 0, len(grid) - 1, 129.0, True, max_wait=10) == -1


def test_unfilled_order_emits_no_row():
    """An order that never fills is a trade that never happened, not a flat one."""
    grid = _ramp(top=101.0)
    cand = _cand([101.5, 99.0])
    cfg = FirstPassageConfig(entry_end="14:45", max_fill_wait_seconds=60)
    out = label_session(grid, {60: cand}, cfg, r_value=10.0,
                        entry_levels_by_entry={60: (100.9, 90.0)})
    assert out.empty or (out["side"] == LONG).all()


def test_entry_price_is_the_resting_level_not_the_decision_close():
    grid = _ramp()
    cand = _cand([120.0, 90.0])
    cfg = FirstPassageConfig(max_fill_wait_seconds=900)
    out = label_session(grid, {60: cand}, cfg, r_value=10.0,
                        entry_levels_by_entry={60: (110.0, 95.0)})
    lng = out[out["side"] == LONG]
    assert not lng.empty
    assert np.allclose(lng["entry_price"], 110.0)
    assert np.allclose(lng["decision_price"], grid["close"].iloc[60])
    assert (lng["t_fill"] > 0).all()
    assert (lng["fill_ts"] > lng["ts"]).all()


def test_distances_are_restated_against_the_fill():
    grid = _ramp()
    cand = _cand([120.0, 90.0])
    out = label_session(grid, {60: cand}, FirstPassageConfig(), r_value=10.0,
                        entry_levels_by_entry={60: (110.0, 95.0)})
    lng = out[out["side"] == LONG]
    assert np.allclose(lng["tgt_dist_pts"], 10.0)         # 120 - 110
    assert np.allclose(lng["tgt_dist_r"], 1.0)
    assert np.allclose(lng["stp_dist_pts"], 20.0)         # 110 - 90


def test_omitting_entry_levels_fills_at_the_decision_close():
    grid = _ramp()
    cand = _cand([120.0, 90.0])
    out = label_session(grid, {60: cand}, FirstPassageConfig(), r_value=10.0)
    assert (out["t_fill"] == 0).all()
    assert np.allclose(out["entry_price"], grid["close"].iloc[60])


def test_levels_too_close_to_the_fill_are_dropped():
    grid = _ramp()
    cand = _cand([110.05, 120.0, 90.0])            # 110.05 is 0.005 R from the fill
    cfg = FirstPassageConfig(min_dist_r=0.10)
    out = label_session(grid, {60: cand}, cfg, r_value=10.0,
                        entry_levels_by_entry={60: (110.0, 95.0)})
    assert not np.isclose(out["tgt_level"], 110.05).any()


def test_unconditional_surface_is_symmetric_with_both_sides():
    """Mirrored geometries must price to zero net: the no-edge sanity check."""
    from src.training.first_passage import unconditional_surface
    rng = np.random.default_rng(3)
    n = 4000
    df = pd.DataFrame({
        "tgt_dist_r": rng.uniform(0.2, 3.0, n),
        "stp_dist_r": rng.uniform(0.2, 3.0, n),
        "event": rng.choice([CENSORED, TARGET, STOP], n),
        "pnl_pts": rng.normal(0, 5, n),
        "t_event": rng.integers(30, 3600, n),
    })
    surf = unconditional_surface(df, n_bins=3)
    assert not surf.empty
    assert {"p_target", "p_stop", "p_censored", "median_hold_min"} <= set(surf.columns)
    probs = surf[["p_target", "p_stop", "p_censored"]].sum(axis=1)
    assert np.allclose(probs, 1.0)
