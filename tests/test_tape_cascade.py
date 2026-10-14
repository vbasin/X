"""Tests for the tape substrate, move labels, stop rule and uniqueness weighting.

The invariants pinned here are the ones that, if broken, would produce results that
look fine and are wrong: a label that peeks forward, an excursion measured after the
event it is supposed to gate, a uniqueness weight that fails to discount overlap.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.uniqueness import (
    concurrency, effective_n, label_uniqueness, uniqueness, weighted_tstat,
)
from src.features.tape import (
    first_cross, print_ofi, rmq, sparse_tables, tape_delta_1s, tape_flow_bars,
    tape_levels, tape_profile, tape_to_path,
)
from src.training.auction_levels import ProfileConfig
from src.training.move_label import LONG, SHORT, MoveConfig, label_moves
from src.training.stop_rule import (
    POLICIES, StopConfig, build_stop_bars, resting_stop, resting_stop_vec,
    stop_candidates, stop_candidates_vec,
)


def make_tape(prices, sizes=None, sides=None, start="2026-06-25 13:30:00",
              step_ms=1000) -> pd.DataFrame:
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq=f"{step_ms}ms", tz="UTC")
    return pd.DataFrame({
        "price": np.asarray(prices, dtype="float64"),
        "size": np.asarray(sizes if sizes is not None else np.ones(n), dtype="int32"),
        "side": np.asarray(sides if sides is not None else np.ones(n), dtype="int8"),
        "bid_px": np.asarray(prices, dtype="float64") - 0.25,
        "ask_px": np.asarray(prices, dtype="float64") + 0.25,
        "bid_sz": np.full(n, 10, dtype="int32"),
        "ask_sz": np.full(n, 10, dtype="int32"),
    }, index=pd.Index(idx, name="ts"))


# ── tape profile: exact, not smeared ──────────────────────────────────────────

def test_profile_puts_volume_at_the_traded_price():
    """Each print's size lands on its own bin — no spreading across a bar range."""
    tape = make_tape([100.0, 100.0, 105.0], sizes=[3, 4, 5])
    prof = tape_profile(tape, ProfileConfig(bin_size=1.0))
    assert prof.value_at(100.0) == pytest.approx(7.0)
    assert prof.value_at(105.0) == pytest.approx(5.0)
    assert prof.value_at(102.0) == pytest.approx(0.0)


def test_profile_delta_is_signed_by_aggressor():
    tape = make_tape([100.0, 100.0], sizes=[3, 4], sides=[1, -1])
    prof = tape_profile(tape, ProfileConfig(bin_size=1.0))
    assert prof.value_at(100.0, "volume") == pytest.approx(7.0)
    assert prof.value_at(100.0, "delta") == pytest.approx(-1.0)


def test_poc_is_the_most_traded_price():
    tape = make_tape([100.0] * 3 + [105.0] * 9 + [110.0] * 2)
    lv = tape_levels(tape, pd.Timestamp("2026-06-25"), ProfileConfig(bin_size=1.0))
    assert lv.poc == pytest.approx(105.0)
    assert lv.lo == pytest.approx(100.0)
    assert lv.hi == pytest.approx(110.0)


def test_tape_levels_use_nq_tick_value_area():
    tape = make_tape([100.0, 100.25, 100.50], sizes=[50, 100, 50])
    lv = tape_levels(tape, pd.Timestamp("2026-06-25"), ProfileConfig())
    assert (lv.poc, lv.vah, lv.val) == pytest.approx((100.25, 100.50, 100.00))


def test_delta_1s_totals_match_the_prints():
    tape = make_tape([100.0] * 4, sizes=[1, 2, 3, 4], sides=[1, 1, -1, -1],
                     step_ms=100)
    assert tape_delta_1s(tape).sum() == pytest.approx(1 + 2 - 3 - 4)


def test_tape_to_path_preserves_volume():
    tape = make_tape(np.arange(100.0, 110.0), sizes=np.arange(1, 11))
    path = tape_to_path(tape)
    assert path["volume"].sum() == tape["size"].sum()
    assert path["high"].max() == tape["price"].max()
    assert path["low"].min() == tape["price"].min()


# ── OFI is a book measure, not a restatement of delta ─────────────────────────

def test_ofi_is_not_identical_to_delta():
    """If OFI collapsed onto delta the feature would be redundant, not additive."""
    n = 40
    rng = np.random.default_rng(0)
    tape = make_tape(30000 + np.cumsum(rng.normal(0, 0.25, n)))
    tape["bid_sz"] = rng.integers(1, 50, n).astype("int32")
    tape["ask_sz"] = rng.integers(1, 50, n).astype("int32")
    ofi = print_ofi(tape)
    signed = tape["size"].to_numpy() * tape["side"].to_numpy()
    assert np.isnan(ofi[0]), "first print has no previous quote"
    assert not np.allclose(ofi[1:], signed[1:])


def test_flow_bars_carry_the_hive_column_names():
    tape = make_tape(np.arange(100.0, 200.0), step_ms=1000)
    fb = tape_flow_bars(tape, "15s")
    for c in ("of_delta", "of_ofi", "of_cvd", "of_ofi_cum", "cs_close"):
        assert c in fb.columns
    assert fb["of_cvd"].is_monotonic_increasing  # all buys in this fixture


# ── range queries ─────────────────────────────────────────────────────────────

def test_rmq_matches_brute_force():
    rng = np.random.default_rng(7)
    px = rng.normal(30000, 20, 500)
    hi, lo = sparse_tables(px)
    l = rng.integers(0, 400, 50)
    r = l + rng.integers(1, 90, 50)
    got_hi = rmq(hi, l, r, True)
    got_lo = rmq(lo, l, r, False)
    for k in range(len(l)):
        assert got_hi[k] == pytest.approx(px[l[k]:r[k] + 1].max(), abs=1e-3)
        assert got_lo[k] == pytest.approx(px[l[k]:r[k] + 1].min(), abs=1e-3)


def test_first_cross_finds_the_first_index_not_any_index():
    px = np.array([100.0, 101.0, 105.0, 99.0, 106.0])
    hi, _ = sparse_tables(px)
    got = first_cross(hi, np.array([0]), np.array([4]), np.array([105.0]), True)
    assert got[0] == 2


def test_first_cross_returns_minus_one_when_never_crossed():
    px = np.array([100.0, 101.0, 102.0])
    hi, _ = sparse_tables(px)
    got = first_cross(hi, np.array([0]), np.array([2]), np.array([200.0]), True)
    assert got[0] == -1


# ── move labels ───────────────────────────────────────────────────────────────

def _move_cfg(**kw):
    base = dict(entry_start="08:30", entry_end="08:31", flatten="15:00",
                decision_grid_seconds=60, horizon_seconds=600,
                move_r=0.25, adverse_r=0.10)
    base.update(kw)
    return MoveConfig(**base)


def test_clean_run_up_is_a_long_move_not_a_short_one():
    # 13:30 UTC == 08:30 CT. Price rises monotonically: clean long.
    tape = make_tape(np.arange(30000.0, 30100.0, 1.0))
    lab = label_moves(tape, _move_cfg(), r_value=40.0)
    first = lab[lab["ts"] == lab["ts"].min()]
    assert first.loc[first["side"] == LONG, "y_move"].iloc[0] == 1
    assert first.loc[first["side"] == SHORT, "y_move"].iloc[0] == 0


def test_adverse_excursion_before_the_move_disqualifies_it():
    # Dips 8 points (> 0.10 * 40) before running up: reached, but not clean.
    px = np.concatenate([np.arange(30000.0, 29992.0, -1.0),
                         np.arange(29992.0, 30100.0, 1.0)])
    lab = label_moves(make_tape(px), _move_cfg(), r_value=40.0)
    first = lab[(lab["ts"] == lab["ts"].min()) & (lab["side"] == LONG)]
    assert first["reached"].iloc[0] == 1
    assert first["y_move"].iloc[0] == 0


def test_adverse_excursion_after_the_move_does_not_disqualify_it():
    """A run that clears the threshold and then retraces is still a clean move."""
    px = np.concatenate([np.arange(30000.0, 30100.0, 1.0),
                         np.arange(30100.0, 29900.0, -1.0)])
    lab = label_moves(make_tape(px), _move_cfg(), r_value=40.0)
    first = lab[(lab["ts"] == lab["ts"].min()) & (lab["side"] == LONG)]
    assert first["y_move"].iloc[0] == 1


def test_t_move_is_when_the_threshold_was_first_cleared():
    tape = make_tape(np.arange(30000.0, 30100.0, 1.0))  # +1 pt per second
    lab = label_moves(tape, _move_cfg(), r_value=40.0)  # threshold = 10 pts
    row = lab[(lab["ts"] == lab["ts"].min()) & (lab["side"] == LONG)]
    assert row["t_move"].iloc[0] == 10


def test_label_carries_no_bracket_columns():
    """Stage 1 must not see target or stop geometry — that is the conflation."""
    lab = label_moves(make_tape(np.arange(30000.0, 30100.0)), _move_cfg(), 40.0)
    assert not [c for c in lab.columns if c.startswith(("tgt_", "stp_"))]
    assert "pnl_pts" not in lab.columns


def test_horizon_is_clipped_at_the_flatten():
    """A decision near the close gets the horizon it would really have."""
    cfg = _move_cfg(entry_start="14:58", entry_end="14:59", flatten="15:00",
                    horizon_seconds=3600)
    # 19:58 UTC == 14:58 CT (CDT).
    tape = make_tape(np.arange(30000.0, 30300.0),
                     start="2026-06-25 19:58:00")
    lab = label_moves(tape, cfg, r_value=40.0)
    assert not lab.empty
    # Nothing may be measured past 15:00 CT = 20:00 UTC.
    assert (lab["t_move"] <= 120).all()


# ── stop rule ─────────────────────────────────────────────────────────────────

def test_fixed_policy_places_the_stop_at_the_configured_distance():
    cfg = StopConfig(policy="fixed", fixed_dist_r=0.5)
    assert resting_stop({}, LONG, 100.0, 40.0, cfg) == pytest.approx(80.0)
    assert resting_stop({}, SHORT, 100.0, 40.0, cfg) == pytest.approx(120.0)


def test_stop_is_always_on_the_losing_side_of_the_entry():
    cfg = StopConfig(policy="wick")
    # A "floor" above the entry is not a floor.
    assert np.isnan(resting_stop({"wick": 110.0}, LONG, 100.0, 40.0, cfg))
    assert np.isnan(resting_stop({"wick": 90.0}, SHORT, 100.0, 40.0, cfg))


def test_stop_outside_the_distance_guards_is_rejected():
    cfg = StopConfig(policy="wick", min_dist_r=0.10, max_dist_r=1.0, buffer_pts=1.0)
    assert np.isnan(resting_stop({"wick": 99.9}, LONG, 100.0, 40.0, cfg))   # too tight
    assert np.isnan(resting_stop({"wick": 50.0}, LONG, 100.0, 40.0, cfg))   # too wide
    assert np.isfinite(resting_stop({"wick": 80.0}, LONG, 100.0, 40.0, cfg))


def test_tightest_and_safest_pick_opposite_ends():
    cands = {"wick": 90.0, "absorption": 85.0, "zone": 80.0}
    tight = resting_stop(cands, LONG, 100.0, 40.0, StopConfig(policy="tightest"))
    safe = resting_stop(cands, LONG, 100.0, 40.0, StopConfig(policy="safest"))
    assert tight == pytest.approx(89.0)   # highest floor, minus buffer
    assert safe == pytest.approx(79.0)    # lowest floor, minus buffer


def test_bid_wall_is_a_sell_print_in_the_lower_wick_of_an_up_candle():
    # Up candle: opens 100, closes 104. A large sell at 98 is in the lower wick.
    tape = make_tape([100.0, 98.0, 104.0], sizes=[1, 50, 1], sides=[1, -1, 1],
                     step_ms=1000)
    sb = build_stop_bars(tape, StopConfig(large_order_size=20))
    assert np.isfinite(sb.bid_wall).any()
    assert np.nanmin(sb.bid_wall) == pytest.approx(98.0)


def test_small_prints_do_not_make_walls():
    tape = make_tape([100.0, 98.0, 104.0], sizes=[1, 5, 1], sides=[1, -1, 1])
    sb = build_stop_bars(tape, StopConfig(large_order_size=20))
    assert not np.isfinite(sb.bid_wall).any()


def test_stop_candidates_only_read_closed_candles():
    """A candle that has not closed at the decision cannot inform the stop."""
    tape = make_tape(np.arange(100.0, 400.0), step_ms=1000)
    sb = build_stop_bars(tape, StopConfig())
    ts = tape.index[90]                      # inside the second minute
    c = stop_candidates(sb, ts, LONG, None, StopConfig())
    # Only the first minute has closed; its low is the first price.
    assert c["wick"] == pytest.approx(100.0)


@pytest.mark.parametrize("side", [LONG, SHORT])
@pytest.mark.parametrize("policy", POLICIES)
def test_vectorised_stop_matches_the_scalar_path(side, policy):
    """The batched path exists for speed; it must not change a single answer."""
    prices = 100.0 + np.sin(np.arange(600) / 7.0) * 6.0
    sizes = np.where(np.arange(600) % 37 == 0, 50, 2)
    sides = np.where(np.arange(600) % 5 < 2, -1, 1)
    tape = make_tape(prices, sizes=sizes, sides=sides, step_ms=1000)
    cfg = replace(StopConfig(large_order_size=20), policy=policy)
    sb = build_stop_bars(tape, cfg)

    ts = tape.index[120::20]
    entry = tape["price"].to_numpy()[120::20]
    zone = np.full(ts.size, 95.0)
    r = 10.0

    vec_c = stop_candidates_vec(sb, ts.values, side, zone, cfg)
    vec = resting_stop_vec(vec_c, side, entry, r, cfg)
    for i in range(ts.size):
        sc = stop_candidates(sb, ts[i], side, float(zone[i]), cfg)
        scalar = resting_stop(sc, side, float(entry[i]), r, cfg)
        assert np.isnan(scalar) == np.isnan(vec[i])
        if np.isfinite(scalar):
            assert scalar == pytest.approx(vec[i])


# ── uniqueness ────────────────────────────────────────────────────────────────

def test_disjoint_labels_are_fully_unique():
    u = uniqueness(np.array([0, 10, 20]), np.array([9, 19, 29]))
    assert np.allclose(u, 1.0)


def test_identical_windows_split_their_uniqueness():
    u = uniqueness(np.zeros(10, dtype=int), np.full(10, 9))
    assert np.allclose(u, 0.1)
    assert effective_n(u) == pytest.approx(1.0)


def test_concurrency_counts_live_windows():
    c = concurrency(np.array([0, 2]), np.array([3, 5]))
    assert c[0] == 1 and c[2] == 2 and c[4] == 1


def test_overlap_discounts_the_t_statistic_by_sqrt_of_overlap():
    x = np.random.default_rng(0).normal(0.1, 1.0, 1000)
    t_raw = weighted_tstat(x)
    t_lap = weighted_tstat(x, np.full(1000, 0.1))
    assert t_lap == pytest.approx(t_raw / np.sqrt(10), rel=0.02)


def test_label_uniqueness_groups_by_session():
    """Labels in different sessions cannot overlap, however close their offsets."""
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2026-06-25 13:30:00", "2026-06-26 13:30:00"], utc=True),
        "t_move": [600, 600],
        "session": pd.to_datetime(["2026-06-25", "2026-06-26"]),
    })
    assert np.allclose(label_uniqueness(df), 1.0)
