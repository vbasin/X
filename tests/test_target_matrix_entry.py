"""Tests for Stage 3 row construction: resting entry and the context pack.

Resting entry changes the row *population*, not just its values: a decision whose
order never fills must emit no row at all, because a trade that did not happen is
not a flat one. That and the restatement of geometry against the fill are the two
things most likely to be got wrong, so both are pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.cash_session_profile import LevelConfig, SessionLevels
from src.training.move_label import LONG

SPEC = importlib.util.spec_from_file_location(
    "build_target_matrix",
    Path(__file__).resolve().parents[1] / "scripts" / "features" / "build_target_matrix.py")
btm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(btm)

R = 30.0
SESSION = pd.Timestamp("2026-06-02").date()
# Five widely separated levels repeated every session, so the KDE resolves one
# bound per level and the geometry below is exact rather than approximate.
HISTORY = [SessionLevels(session=pd.Timestamp("2026-05-01") + pd.Timedelta(days=i),
                         lo=100.0, val=130.0, poc=160.0, vah=190.0, hi=220.0)
           for i in range(20)][::-1]


def make_tape(prices, start="2026-06-02 14:00:00"):
    idx = pd.date_range(start, periods=len(prices), freq="1s", tz="UTC")
    return pd.DataFrame({"price": np.asarray(prices, dtype="float64"),
                         "size": np.ones(len(prices)),
                         "side": np.ones(len(prices), dtype="int8")}, index=idx)


def make_gated(ts, price=145.0, side=LONG, **ctx):
    return pd.DataFrame([{"ts": ts, "side": side, "price": price,
                          "p_move": 0.9, **ctx}])


def build(tape, gated, *, resting_entry=False, ctx_cols=(), max_fill_wait=900):
    return btm.build_session(tape, gated, SESSION, HISTORY, R, LevelConfig(),
                             0.5, 0.25, 5, 1800, list(ctx_cols),
                             resting_entry, max_fill_wait)


# ── baseline: fill at the decision price ─────────────────────────────────────

def test_without_resting_entry_the_fill_is_the_decision_price():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]))
    assert not out.empty
    assert (out["price"] == 145.0).all()
    assert (out["decision_price"] == 145.0).all()
    assert (out["t_fill"] == 0).all()


# ── resting entry ────────────────────────────────────────────────────────────

def test_resting_entry_between_bounds_uses_the_confirmation_offset():
    # 145 sits between the 131 and 159 bounds, so the order rests one point away
    # and only fills once price continues into it.
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]), resting_entry=True)
    assert not out.empty
    assert (out["price"] == 146.0).all()
    assert (out["t_fill"] == 1).all()


def test_unfilled_resting_orders_emit_no_row():
    tape = make_tape(np.full(60, 145.0))
    out = build(tape, make_gated(tape.index[0]), resting_entry=True)
    assert out.empty


def test_fill_wait_is_enforced():
    # Price only reaches the resting 146 after the order has been cancelled.
    prices = np.concatenate([np.full(30, 145.0), np.arange(145.0, 166.0)])
    tape = make_tape(prices)
    out = build(tape, make_gated(tape.index[0]), resting_entry=True,
                max_fill_wait=10)
    assert out.empty


def test_geometry_is_restated_against_the_fill():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]), resting_entry=True)
    near = out[out["tgt_rank"] == 0].iloc[0]
    assert near["tgt_level"] == pytest.approx(159.0)
    # Distance measured from the 146 fill, not the 145 decision.
    assert near["tgt_dist_r"] == pytest.approx((159.0 - 146.0) / R)
    assert near["stp_dist_r"] == pytest.approx(0.5 * (159.0 - 146.0) / R)


def test_t_reach_is_measured_from_the_fill_not_the_decision():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]), resting_entry=True)
    near = out[out["tgt_rank"] == 0].iloc[0]
    assert near["y_reach"] == 1
    # 159 is printed at index 14; the fill was at index 1.
    assert near["t_reach"] == 13


def test_stop_cap_binds_on_distant_targets():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]), resting_entry=True)
    far = out[out["tgt_rank"] == out["tgt_rank"].max()].iloc[0]
    assert far["stp_dist_r"] == pytest.approx(0.25)


# ── context pack ─────────────────────────────────────────────────────────────

def test_context_columns_ride_through_onto_every_target_row():
    tape = make_tape(np.arange(145.0, 166.0))
    gated = make_gated(tape.index[0], st_zone=2.0, fs_vwap_velocity_r=0.4)
    out = build(tape, gated, ctx_cols=("st_zone", "fs_vwap_velocity_r"))
    assert (out["st_zone"] == 2.0).all()
    assert out["fs_vwap_velocity_r"].to_numpy() == pytest.approx(0.4)
    assert len(out) > 1


def test_absent_context_columns_are_skipped_not_fabricated():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]), ctx_cols=("st_zone",))
    assert "st_zone" not in out.columns


def test_p_move_still_rides_along_for_the_joint_gate():
    tape = make_tape(np.arange(145.0, 166.0))
    out = build(tape, make_gated(tape.index[0]))
    assert out["p_move"].to_numpy() == pytest.approx(0.9)
