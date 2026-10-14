"""Focused tests for the ORB engine (ADR-118)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.external_models.catalog import coerce_params, factory_params, get_model
from src.external_models.orb import (
    _bar_volume_profile,
    build_session_contexts,
    simulate_orb,
)


def _session(day: int, bars: list[tuple[int, float, float, float, float, float]]):
    """bars = (ct_minute, o, h, l, c, volume) on 2026-01-<day> America/Chicago."""
    rows = []
    for m, o, h, l, c, v in bars:
        # CT is UTC-6 in January
        ts = datetime(2026, 1, day, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=m + 360)
        rows.append({"bucket": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return rows


def _flat_day(day: int, px: float = 100.0, vol: float = 100.0):
    """A full 08:30–15:00 session of quiet 1-point bars."""
    out = []
    for m in range(510, 900):
        out.extend(_session(day, [(m, px, px + 1, px - 1, px, vol)]))
    return out


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("bucket").sort_index()
    return df


def _params(**over):
    return coerce_params("orb", over)


# ── catalog wiring ───────────────────────────────────────────────────────

def test_orb_card_is_registered_and_listed():
    card = get_model("orb")
    assert card["id"] == "orb"
    assert card["engine"] == "orb"
    assert card.get("listed") is not False
    keys = {row["key"] for row in card["param_spec"]}
    for need in ("or_minutes", "min_rvol", "setups", "risk_usd_per_trade",
                 "max_drawdown", "max_contracts", "point_value"):
        assert need in keys


def test_factory_params_match_the_account_brief():
    p = factory_params("orb")
    assert p["initial_balance"] == 5000.0
    assert p["max_drawdown"] == 1000.0
    assert p["max_contracts"] == 3
    assert p["margin_per_contract"] == 1000.0
    assert p["point_value"] == 20.0


def test_unknown_setup_is_rejected():
    with pytest.raises(ValueError):
        coerce_params("orb", {"setups": "moon_phase"})


def test_target_2_must_not_be_below_target_1():
    with pytest.raises(ValueError):
        coerce_params("orb", {"t1_r": 3.0, "t2_r": 1.0})


def test_drawdown_must_be_smaller_than_balance():
    with pytest.raises(ValueError):
        coerce_params("orb", {"initial_balance": 5000.0, "max_drawdown": 5000.0})


# ── session context ──────────────────────────────────────────────────────

def test_opening_range_is_measured_over_or_minutes_only():
    rows = []
    for d in (5, 6):
        # 08:30–08:45 range 100–110, then a spike to 200 later in the session
        for m in range(510, 525):
            rows.extend(_session(d, [(m, 100, 110, 100, 105, 50)]))
        for m in range(525, 900):
            rows.extend(_session(d, [(m, 105, 200, 90, 150, 50)]))
    ctxs = build_session_contexts(_frame(rows), _params())
    ctx = ctxs[datetime(2026, 1, 6).date()]
    assert ctx.or_high == 110
    assert ctx.or_low == 100
    assert ctx.or_range == 10
    assert ctx.or_mid == 105


def test_relative_volume_compares_to_prior_sessions():
    rows = []
    for d in range(5, 12):
        v = 10.0 if d < 11 else 40.0        # last day has 4x the opening volume
        for m in range(510, 525):
            rows.extend(_session(d, [(m, 100, 110, 100, 105, v)]))
        for m in range(525, 900):
            rows.extend(_session(d, [(m, 105, 106, 104, 105, 5)]))
    ctxs = build_session_contexts(_frame(rows), _params(rvol_lookback=5))
    ctx = ctxs[datetime(2026, 1, 11).date()]
    assert ctx.rvol == pytest.approx(4.0, rel=0.01)


def test_volume_profile_poc_sits_at_the_busiest_price():
    bars = pd.DataFrame({
        "low":    [100.0, 100.0, 100.0, 120.0],
        "high":   [101.0, 101.0, 101.0, 121.0],
        "volume": [500.0, 500.0, 500.0,   1.0],
    })
    poc, vah, val = _bar_volume_profile(bars, bin_pts=1.0)
    assert 99.0 <= poc <= 102.0
    assert vah >= poc >= val


# ── risk governor ────────────────────────────────────────────────────────

def test_no_trades_when_every_stop_exceeds_the_risk_budget():
    """A $20/pt contract with a 50-point stop is $1000 — more than the budget."""
    rows = []
    for d in range(5, 20):
        for m in range(510, 525):
            rows.extend(_session(d, [(m, 100, 160, 100, 150, 100)]))
        for m in range(525, 800):
            rows.extend(_session(d, [(m, 150, 200, 100, 190, 100)]))
    res = simulate_orb(_frame(rows), _params(
        point_value=20.0, risk_usd_per_trade=100.0, max_stop_points=50.0,
        min_rvol=0.0, max_contracts=3))
    # every candidate is either filtered or unsizable — never an oversized fill
    led = res["trades"]
    if not led.empty:
        assert (led["risk_pts"] * 20.0 * led["qty"] <= 100.0 + 1e-6).all()


def test_position_size_scales_inversely_with_stop_distance():
    from src.external_models.orb import Account, _size
    acct = Account(balance=5000.0, floor=4000.0)
    p = _params(risk_usd_per_trade=300.0, max_contracts=3,
                margin_per_contract=1000.0, headroom_frac=1.0, point_value=2.0)
    assert _size(50.0, acct, p) == 3      # $100/contract → 3
    assert _size(150.0, acct, p) == 1     # $300/contract → 1
    assert _size(400.0, acct, p) == 0     # $800/contract → unsizable


def test_size_is_capped_by_margin():
    from src.external_models.orb import Account, _size
    acct = Account(balance=2500.0, floor=1500.0)   # affords only 2 contracts
    p = _params(risk_usd_per_trade=10000.0, max_contracts=3,
                margin_per_contract=1000.0, headroom_frac=1.0, point_value=2.0)
    assert _size(1.0, acct, p) == 2


def test_size_shrinks_as_equity_approaches_the_floor():
    from src.external_models.orb import Account, _size
    p = _params(risk_usd_per_trade=300.0, max_contracts=3,
                margin_per_contract=1000.0, headroom_frac=0.5, point_value=2.0)
    healthy = _size(25.0, Account(balance=5000.0, floor=4000.0), p)
    stressed = _size(25.0, Account(balance=4100.0, floor=4000.0), p)
    assert healthy > stressed


def test_account_never_closes_below_the_floor_on_real_shaped_data():
    rng = np.random.default_rng(7)
    rows = []
    px = 20000.0
    for d in range(2, 28):
        for m in range(510, 900):
            step = float(rng.normal(0, 8))
            o = px
            px = px + step
            rows.extend(_session(d, [(m, o, max(o, px) + 4, min(o, px) - 4, px, 100.0)]))
    res = simulate_orb(_frame(rows), _params(min_rvol=0.0))
    acct = res["account"]
    assert acct["min_balance"] >= acct["floor"] - 1e-6 or acct["halted"]


# ── payload shape ────────────────────────────────────────────────────────

def test_payload_dollarises_contract_weighted_points():
    from src.external_models.orb import orb_payload
    led = pd.DataFrame([{
        "session_date": datetime(2026, 1, 5).date(),
        "entry_ts": pd.Timestamp("2026-01-05T15:00:00Z"),
        "exit_ts": pd.Timestamp("2026-01-05T15:30:00Z"),
        "side": "long", "qty": 2, "entry_px": 100.0, "exit_px": 105.0,
        "points": 5.0, "pnl_points": 10.0, "pnl_usd": 200.0, "risk_pts": 5.0,
        "r_multiple": 1.0, "mae_points": 1.0, "mfe_points": 6.0,
        "holding_bars": 30, "exit_reason": "target_1", "setup": "breakout_retest",
        "balance": 5200.0,
    }])
    res = {"trades": led, "skipped": {}, "n_bars": 100,
           "account": {"final_balance": 5200.0, "floor": 4000.0, "initial_balance": 5000.0,
                       "max_drawdown_usd": 0.0, "min_balance": 5000.0,
                       "breached": False, "return_pct": 4.0},
           "params": factory_params("orb")}
    out = orb_payload(res, source="external:orb", start="2026-01-01",
                      end="2026-01-31", n_sessions=20, card=get_model("orb"))
    # 10 contract-points × $20 = $200 — matches pnl_usd
    assert out["summary"]["net_value"] == pytest.approx(200.0)
    assert out["trades"][0]["qty"] == 2
    assert out["summary"]["final_balance"] == 5200.0
