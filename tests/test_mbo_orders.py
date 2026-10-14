"""Tests for src/webui/mbo_orders.py — synthetic events only, no database."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.webui.mbo_orders import build_segments, classify, find_chains, seed_from_snapshot

T0 = pd.Timestamp("2026-09-24 13:30:00", tz="UTC")
T_END = T0 + pd.Timedelta(hours=1)


def _ts(sec: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(seconds=sec)


def _ev(rows: list[tuple]) -> pd.DataFrame:
    """rows: (sec, seq, order_id, action, side, size, price[, snap])"""
    df = pd.DataFrame(
        [(_ts(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7] if len(r) > 7 else False)
         for r in rows],
        columns=["ts_recv", "sequence", "order_id", "action", "side", "size", "price", "snap"])
    return df


def _candles(lows: list[float], highs: list[float]) -> pd.DataFrame:
    n = len(lows)
    t = pd.date_range(T0, periods=n, freq="min", tz="UTC")
    c = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
    return pd.DataFrame({"t": t, "o": c, "h": highs, "l": lows, "c": c, "v": 1})


def _seg(rows: list[tuple]) -> pd.DataFrame:
    return build_segments(_ev(rows), T_END)


def test_partial_fill_then_cancel_is_cancelled():
    seg = _seg([(0, 1, 7, "A", "B", 40, 100.0), (5, 2, 7, "F", "B", 10, 100.0),
                (5, 3, 7, "M", "B", 30, 100.0), (9, 4, 7, "C", "B", 30, 100.0)])
    assert len(seg) == 1
    r = seg.iloc[0]
    assert r["end"] == "cancelled" and r["filled"] == 10 and r["size0"] == 40


def test_full_fill_ends_filled():
    seg = _seg([(0, 1, 7, "A", "B", 25, 100.0), (4, 2, 7, "F", "B", 25, 100.0),
                (4, 3, 7, "C", "B", 25, 100.0)])
    assert seg.iloc[0]["end"] == "filled"


def test_filled_does_not_depend_on_record_order_within_a_tie():
    """CME can emit F and C under one (ts_recv, sequence); either order must give 'filled'."""
    fc = _seg([(0, 1, 7, "A", "B", 25, 100.0), (4, 2, 7, "F", "B", 25, 100.0),
               (4, 2, 7, "C", "B", 25, 100.0)])
    cf = _seg([(0, 1, 7, "A", "B", 25, 100.0), (4, 2, 7, "C", "B", 25, 100.0),
               (4, 2, 7, "F", "B", 25, 100.0)])
    assert fc.iloc[0]["end"] == cf.iloc[0]["end"] == "filled"


def test_price_modify_splits_segment_and_last_one_rests():
    seg = _seg([(0, 1, 7, "A", "A", 30, 100.0), (10, 2, 7, "M", "A", 30, 101.0)])
    assert list(seg["end"]) == ["moved", "resting"]
    assert list(seg["price"]) == [100.0, 101.0]
    assert seg.iloc[1]["t_end"] == T_END


def test_snapshot_relisting_of_a_live_order_is_ignored_but_snapshot_only_is_pre_existing():
    seg = _seg([(0, 1, 7, "A", "B", 30, 100.0), (60, 2, 7, "A", "B", 30, 100.0, True),
                (90, 3, 7, "C", "B", 30, 100.0),
                (61, 4, 8, "A", "B", 50, 99.0, True), (95, 5, 8, "C", "B", 50, 99.0)])
    live, pre = seg[seg["order_id"] == 7].iloc[0], seg[seg["order_id"] == 8].iloc[0]
    assert len(seg) == 2 and not live["pre"] and pre["pre"]


def test_orders_placed_before_the_window_are_ignored():
    """Without a resting-book snapshot seed (ADR-147), this is the known, documented gap: an
    order touched but never Added inside the window is invisible. seed_from_snapshot() below is
    the fix -- see the next test for the same scenario with a seed applied."""
    seg = _seg([(5, 2, 9, "F", "B", 3, 100.0), (6, 3, 9, "C", "B", 3, 100.0)])
    assert seg.empty


def test_snapshot_seed_recovers_an_order_placed_before_the_window():
    """ADR-147: the exact scenario the prior test documents as a gap, fixed by widening the
    query to the resting-book checkpoint and seeding the pre-existing order from it."""
    snap = pd.DataFrame({"order_id": [9], "side": ["B"], "price": [100.0], "size": [40],
                         "t_start": [T0 - pd.Timedelta(minutes=30)]})
    seed = seed_from_snapshot(snap)
    assert list(seed["action"]) == ["A"] and bool(seed["snap"].iloc[0])
    ev = pd.concat([seed, _ev([(5, 2, 9, "F", "B", 3, 100.0), (6, 3, 9, "C", "B", 3, 100.0)])],
                   ignore_index=True)
    seg = build_segments(ev, T_END)
    assert len(seg) == 1
    r = seg.iloc[0]
    assert r["pre"] and r["size0"] == 40 and r["filled"] == 3 and r["end"] == "cancelled"
    assert r["t_start"] == T0 - pd.Timedelta(minutes=30)   # true origin preserved, not clamped


def test_seed_from_snapshot_of_empty_frame_is_a_harmless_empty_frame():
    seed = seed_from_snapshot(pd.DataFrame(columns=["order_id", "side", "price", "size", "t_start"]))
    assert seed.empty and list(seed.columns) == ["ts_recv", "sequence", "order_id", "action",
                                                  "side", "size", "price", "snap"]


def test_classify_outcomes():
    seg = _seg([
        (0, 1, 1, "A", "B", 30, 100.0), (30, 2, 1, "F", "B", 30, 100.0), (30, 3, 1, "C", "B", 30, 100.0),
        (0, 4, 2, "A", "B", 30, 100.0), (150, 5, 2, "C", "B", 30, 100.0),         # touched
        (0, 6, 3, "A", "B", 30, 95.0), (150, 7, 3, "C", "B", 30, 95.0),           # near, untouched
        (0, 8, 4, "A", "B", 30, 60.0), (150, 9, 4, "C", "B", 30, 60.0),           # far
        (0, 10, 5, "A", "A", 30, 130.0),                                          # resting
    ])
    # price path: minute 0 trades 100..104, minutes 1-2 trade 101..104 (bid @100 touched in min 0)
    candles = _candles([100.0, 97.0, 97.0], [104.0, 104.0, 104.0])
    out = classify(seg, candles, approach=2.0).set_index("order_id")["outcome"].to_dict()
    assert out[1] == "FILLED"
    assert out[2] == "PULLED_AFTER_TOUCH"
    assert out[4] == "PULLED_FAR"
    assert out[5] == "RESTING"


def _chain_events(price: float, side: str, n: int, gap: float, size: int = 10,
                  start: float = 0.0, oid0: int = 100) -> list[tuple]:
    """n orders at one price, each fully filled after `gap` s then the next placed 0.5 s later."""
    rows, t, seq = [], start, oid0 * 10
    for i in range(n):
        oid = oid0 + i
        rows += [(t, seq, oid, "A", side, size, price),
                 (t + gap, seq + 1, oid, "F", side, size, price),
                 (t + gap, seq + 2, oid, "C", side, size, price)]
        t += gap + 0.5
        seq += 3
    return rows


def _chains(rows, candles=None, **kw):
    seg = _seg(rows)
    candles = candles if candles is not None else _candles([90.0] * 3, [110.0] * 3)
    return find_chains(seg, candles, dt_s=2.0, n_links=3, break_pts=1.0, **kw)


def test_chain_detected_when_same_price_re_add_follows_each_fill():
    ch = _chains(_chain_events(100.0, "B", 5, gap=8.0))
    assert len(ch) == 1
    r = ch.iloc[0]
    assert r["side"] == "B" and r["orders"] == 5 and r["absorbed"] == 50 and r["clip"] == 10


def test_persistence_floor_drops_sub_second_bursts():
    burst = _chain_events(100.0, "A", 5, gap=0.1)
    assert len(_chains(burst)) == 1                                # raw chain exists ...
    assert _chains(burst, min_secs=10, min_abs=50).empty           # ... but is a burst


def test_no_chain_when_re_add_is_at_a_different_price_or_too_late():
    rows = _chain_events(100.0, "B", 2, gap=5.0)
    rows += [(20.0, 900, 200, "A", "B", 10, 101.0)]                # different price
    assert _chains(rows).empty
    late = _chain_events(100.0, "B", 4, gap=5.0)
    late = [(r[0] + (10 * (r[2] - 100)), *r[1:]) for r in late]    # 10 s pauses between orders
    assert _chains(late).empty


def test_no_chain_when_re_add_size_is_not_similar():
    rows = [(0, 1, 1, "A", "B", 10, 100.0), (5, 2, 1, "F", "B", 10, 100.0), (5, 3, 1, "C", "B", 10, 100.0),
            (5.5, 4, 2, "A", "B", 200, 100.0)]                     # 20x the size: not a refill
    assert _chains(rows).empty


def test_chain_outcome_held_vs_broke():
    rows = _chain_events(100.0, "B", 5, gap=8.0)
    held = _candles([100.0, 101.0, 101.0, 101.0], [103.0, 104.0, 104.0, 104.0])
    broke = _candles([100.0, 97.0, 96.0, 96.0], [103.0, 104.0, 104.0, 104.0])
    assert _chains(rows, held).iloc[0]["after"] == "HELD"
    assert _chains(rows, broke).iloc[0]["after"] == "BROKE"
    assert not np.isnan(_chains(rows, held).iloc[0]["mfe5m"])


# --- exchange-side removal: the "R" indicator (ADR-149) ---------------------------------------

def test_order_omitted_from_a_snapshot_boundary_is_cleared_not_left_resting():
    """The reported bug's display half: a Day order CME purged at a session close sends no Cancel,
    so without the boundary check it draws as resting to the edge of the window."""
    ev = _ev([(0, 1, 7, "A", "A", 99, 31079.75)])
    boundary = _ts(1800)
    seg = build_segments(ev, T_END, boundaries=[boundary])
    assert len(seg) == 1
    assert seg.iloc[0]["end"] == "cleared"
    assert seg.iloc[0]["t_end"] == boundary


def test_order_relisted_at_the_boundary_keeps_resting():
    ev = _ev([(0, 1, 7, "A", "A", 99, 31079.75),
              (1800, 2, 7, "A", "A", 99, 31079.75, True)])   # snapshot re-listing
    seg = build_segments(ev, T_END, boundaries=[_ts(1800)])
    assert seg.iloc[-1]["end"] == "resting"
    assert seg.iloc[-1]["t_end"] == T_END


def test_real_exchange_clear_closes_an_open_order():
    ev = _ev([(0, 1, 7, "A", "B", 40, 30500.0)])
    seg = build_segments(ev, T_END, clears=[_ts(600)])
    assert seg.iloc[0]["end"] == "cleared" and seg.iloc[0]["t_end"] == _ts(600)


def test_a_cancel_before_the_boundary_still_reads_as_cancelled():
    """Trader-initiated exit must not be relabelled as an exchange clear."""
    ev = _ev([(0, 1, 7, "A", "B", 40, 30500.0), (300, 2, 7, "C", "B", 40, 30500.0)])
    seg = build_segments(ev, T_END, clears=[_ts(600)], boundaries=[_ts(1800)])
    assert seg.iloc[0]["end"] == "cancelled"


def test_cleared_segment_classifies_as_CLEARED_not_as_a_pull():
    ev = _ev([(0, 1, 7, "A", "A", 99, 105.0)])
    seg = build_segments(ev, T_END, boundaries=[_ts(1800)])
    out = classify(seg, _candles([100.0, 101.0], [102.0, 103.0]), approach=10.0)
    assert out.iloc[0]["outcome"] == "CLEARED"


def test_no_clears_or_boundaries_means_behaviour_is_unchanged():
    ev = _ev([(0, 1, 7, "A", "B", 40, 30500.0)])
    assert build_segments(ev, T_END).iloc[0]["end"] == "resting"


def test_a_moved_order_reports_fills_at_the_level_they_happened_not_everywhere():
    """The bug behind 'iceberg/filled marks outside the price range': an order that re-prices gets a
    fresh queue position, so its fills belong to the level they occurred at. Attributing the order's
    whole total to every level it ever visited painted activity at prices that never traded."""
    ev = _ev([
        (0,   1, 7, "A", "A", 5, 30857.50),      # placed high, nothing trades here
        (60,  2, 7, "M", "A", 5, 30800.00),      # re-priced down to where the market is
        (70,  3, 7, "F", "A", 5, 30800.00),      # filled entirely at the LOWER level
        (70,  4, 7, "C", "A", 5, 30800.00),
    ])
    seg = build_segments(ev, T_END).sort_values("t_start").reset_index(drop=True)
    high, low = seg.iloc[0], seg.iloc[1]
    assert high["price"] == 30857.50 and high["end"] == "moved"
    assert high["filled"] == 0, "nothing traded at 30857.50, so it must not claim fills"
    assert low["price"] == 30800.00 and low["end"] == "filled" and low["filled"] == 5


def test_a_single_price_order_is_unaffected_by_per_segment_accounting():
    ev = _ev([(0, 1, 7, "A", "B", 40, 30500.0), (30, 2, 7, "F", "B", 40, 30500.0),
              (30, 3, 7, "C", "B", 40, 30500.0)])
    seg = build_segments(ev, T_END)
    assert len(seg) == 1 and seg.iloc[0]["end"] == "filled" and seg.iloc[0]["filled"] == 40


def test_returning_to_a_price_keeps_each_visit_accounted_separately():
    """The last false-positive class: an order that leaves a level and comes back gets a separate
    segment per visit, and reserve must be judged per visit -- otherwise the quiet visit inherits
    the busy one's absorbed total and shows a badge with zero fills."""
    ev = _ev([
        (0,   1, 7, "A", "A", 5, 30570.75),     # visit 1 at the level -- nothing trades
        (10,  2, 7, "M", "A", 5, 30600.00),     # away
        (20,  3, 7, "M", "A", 5, 30570.75),     # back to the same level
        (30,  4, 7, "F", "A", 22, 30570.75),    # absorbs 22 while showing 5 -> reserve, HERE
        (30,  5, 7, "C", "A", 5, 30570.75),
    ])
    seg = build_segments(ev, T_END).sort_values("t_start").reset_index(drop=True)
    first, last = seg.iloc[0], seg.iloc[-1]
    assert first["price"] == 30570.75 and first["filled"] == 0 and first["disp"] == 5
    assert last["price"] == 30570.75 and last["filled"] == 22 and last["disp"] == 5


def test_display_restored_after_a_fill_is_reserve_even_when_absorbed_stays_small():
    """The class the 'absorbed > shown' test alone misses: a 50-lot display filled 10 and put back
    to 50 must have held at least 60, yet 10 < 50 so the overfill test never fires."""
    ev = _ev([(0, 1, 7, "A", "B", 50, 30500.0),
              (10, 2, 7, "F", "B", 10, 30500.0),
              (10, 3, 7, "M", "B", 50, 30500.0)])     # display restored -> reserve
    seg = build_segments(ev, T_END)
    r = seg.iloc[-1]
    # conservation: gave up 10 and still showed 50 => held >= 60, but only ever displayed 50
    assert r["filled"] == 10 and r["disp"] == 50 and r["left"] == 50
    assert r["filled"] + r["left"] > r["disp"]


def test_an_ordinary_partial_fill_is_not_reserve():
    """Same F-then-M shape, but the Modify REDUCES to the remainder. Direction is the discriminator."""
    ev = _ev([(0, 1, 7, "A", "B", 50, 30500.0),
              (10, 2, 7, "F", "B", 10, 30500.0),
              (10, 3, 7, "M", "B", 40, 30500.0)])     # 50 - 10 = 40, a plain decrement
    seg = build_segments(ev, T_END)
    r = seg.iloc[-1]
    assert r["filled"] + r["left"] == r["disp"], "50 shown, 10 filled, 40 left: nothing hidden"


def test_a_trader_resizing_their_own_order_upward_is_not_reserve():
    """Regression for the exact order a user caught on the chart (6882428412977): A 10 -> M 20 ->
    M 15 -> M 10 -> M 5 -> C, zero fills, badged as an iceberg that had absorbed nothing. A raise
    only means reserve when it follows a FILL."""
    ev = _ev([(0, 1, 7, "A", "B", 10, 30600.0),
              (550, 2, 7, "M", "B", 20, 30600.0),     # trader raises their own size
              (2350, 3, 7, "M", "B", 15, 30600.0),
              (2830, 4, 7, "M", "B", 10, 30600.0),
              (5000, 5, 7, "C", "B", 5, 30600.0)])
    seg = build_segments(ev, T_END)
    r = seg.iloc[-1]
    assert r["filled"] == 0, "never traded, so nothing can be proven hidden"
    assert not (r["filled"] + r["left"] > r["disp"])
    assert seg.iloc[-1]["filled"] == 0


def test_a_raise_that_does_follow_a_fill_is_still_reserve():
    ev = _ev([(0, 1, 7, "A", "B", 50, 30500.0),
              (10, 2, 7, "F", "B", 10, 30500.0),
              (10, 3, 7, "M", "B", 50, 30500.0)])
    seg = build_segments(ev, T_END)
    r = seg.iloc[-1]
    assert r["filled"] + r["left"] > r["disp"]


def test_aggressive_fills_at_other_prices_do_not_inflate_the_resting_level():
    """A marketable limit can sweep (taking liquidity at higher prices) and then rest the
    remainder. Those aggressive fills carry their own price and must not be booked to the resting
    level -- real case 6882409490573, which read 68 absorbed at 30561.50 when only 51 executed
    there."""
    ev = _ev([
        (0,  1, 7, "A", "B", 1, 30561.50),
        (5,  2, 7, "F", "B", 15, 30561.50),     # genuine absorption at the resting level
        (10, 3, 7, "F", "B", 4, 30569.50),      # aggressive fill, different price
        (10, 4, 7, "F", "B", 8, 30570.00),      # aggressive fill, different price
        (11, 5, 7, "M", "B", 1, 30570.00),      # only now does it re-price
    ])
    seg = build_segments(ev, T_END).sort_values("t_start").reset_index(drop=True)
    at_resting = seg.iloc[0]
    assert at_resting["price"] == 30561.50
    assert at_resting["filled"] == 15, "must exclude the 4 and 8 taken at other prices"
