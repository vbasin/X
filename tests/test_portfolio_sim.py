"""Tests for the path-realised portfolio simulator (src/backtest/portfolio_sim.py).

Synthetic 1-second paths are built so each account rule can be checked in isolation:
barrier fills, intrabar ambiguity (the close-only bias), the hold cap, forced flatten,
the concurrent-position cap, margin capacity, the cooldown and the session loss halt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio_sim import (
    SimConfig,
    compare_tie_policies,
    prepare_path,
    required_scaling,
    simulate,
    sweep_tau,
)

_DAY = "2026-03-02"          # a Monday
_OPEN_UTC = f"{_DAY} 14:30:00+00:00"   # 08:30 America/Chicago (CST, UTC-6)


def _path(closes, *, highs=None, lows=None, start=_OPEN_UTC) -> pd.DataFrame:
    """1s OHLC path from a close series; high/low default to the close (no excursion)."""
    n = len(closes)
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="1s", tz="UTC")
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {"high": c if highs is None else np.asarray(highs, dtype="float64"),
         "low": c if lows is None else np.asarray(lows, dtype="float64"),
         "close": c},
        index=idx,
    )


def _signals(offsets, direction="up", confidence=0.9, start=_OPEN_UTC) -> pd.DataFrame:
    base = pd.Timestamp(start)
    return pd.DataFrame({
        "ts": [base + pd.Timedelta(seconds=int(o)) for o in offsets],
        "direction": [direction] * len(offsets),
        "confidence": [confidence] * len(offsets),
    })


def _cfg(**kw) -> SimConfig:
    """Small, frictionless-by-default geometry so arithmetic is checkable by hand."""
    base = dict(
        target_points=10.0, stop_points=5.0, max_hold_seconds=60,
        trade_start="08:30", trade_end="10:00", flatten_at="10:05",
        point_value_usd=20.0, tick_size=0.25,
        commission_per_side_usd=0.0, entry_slippage_ticks=0.0,
        stop_slippage_ticks=0.0, target_slippage_ticks=0.0,
        starting_equity_usd=100_000.0, margin_per_contract_usd=1_000.0,
        contracts_per_trade=1, max_concurrent_positions=5,
        daily_loss_limit_frac=0.5, cooldown_seconds=0, tau=0.5,
    )
    base.update(kw)
    return SimConfig(**base)


# ── path preparation ──────────────────────────────────────────────────────────

def test_prepare_path_requires_close():
    with pytest.raises(ValueError, match="close"):
        prepare_path(pd.DataFrame({"high": [1.0]},
                                  index=pd.date_range("2026-01-01", periods=1, tz="UTC")))


def test_prepare_path_synthesises_high_low_from_close():
    out = prepare_path(_path([100.0, 101.0])[["close"]])
    assert (out["high"] == out["close"]).all() and (out["low"] == out["close"]).all()


def test_session_grid_fills_missing_seconds_without_excursion():
    p = _path([100.0] * 5)
    p = p.drop(p.index[2])                        # a second with no trade
    r = simulate(_signals([0]), p, _cfg(max_hold_seconds=3))
    assert len(r.trades) == 1
    # gap second contributes no MAE/MFE — price never moved
    assert r.trades.iloc[0]["mae_points"] == pytest.approx(0.0)


# ── barrier fills ─────────────────────────────────────────────────────────────

def test_long_target_fill_is_exact_and_priced_from_path():
    closes = [100.0] * 10 + [110.0] + [100.0] * 10
    r = simulate(_signals([0]), _path(closes), _cfg())
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["gross_points"] == pytest.approx(10.0)
    assert t["net_usd"] == pytest.approx(200.0)      # 10 pts x $20
    assert t["hold_seconds"] == 10


def test_short_target_uses_the_low():
    closes = [100.0] * 5 + [90.0] + [100.0] * 5
    r = simulate(_signals([0], direction="down"), _path(closes), _cfg())
    t = r.trades.iloc[0]
    assert t["side"] == "down" and t["exit_reason"] == "target"
    assert t["gross_points"] == pytest.approx(10.0)


def test_stop_fill_includes_adverse_slippage():
    closes = [100.0] * 5 + [95.0] + [100.0] * 5
    r = simulate(_signals([0]), _path(closes), _cfg(stop_slippage_ticks=2.0))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    # stop at 95, filled 2 ticks (0.50) through
    assert t["exit_px"] == pytest.approx(94.5)
    assert t["gross_points"] == pytest.approx(-5.5)


def test_entry_slippage_is_adverse_on_both_sides():
    flat = _path([100.0] * 5)
    long_px = simulate(_signals([0]), flat, _cfg(entry_slippage_ticks=1.0)).trades.iloc[0]["entry_px"]
    short_px = simulate(_signals([0], direction="down"), flat,
                        _cfg(entry_slippage_ticks=1.0)).trades.iloc[0]["entry_px"]
    assert long_px == pytest.approx(100.25)     # buy up
    assert short_px == pytest.approx(99.75)     # sell down


def test_commission_is_charged_round_trip():
    closes = [100.0] * 5 + [110.0] + [100.0] * 5
    r = simulate(_signals([0]), _path(closes), _cfg(commission_per_side_usd=2.50))
    t = r.trades.iloc[0]
    assert t["net_usd"] == pytest.approx(200.0 - 5.0)


def test_hold_cap_exits_on_time_at_the_close():
    r = simulate(_signals([0]), _path([100.0] + [101.0] * 30), _cfg(max_hold_seconds=10))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "time" and t["hold_seconds"] == 10
    assert t["gross_points"] == pytest.approx(1.0)


def test_mae_and_mfe_track_intrabar_extremes():
    closes = [100.0] * 6
    highs = [100.0, 100.0, 103.0, 100.0, 100.0, 100.0]
    lows = [100.0, 100.0, 100.0, 98.0, 100.0, 100.0]
    r = simulate(_signals([0]), _path(closes, highs=highs, lows=lows),
                 _cfg(max_hold_seconds=5))
    t = r.trades.iloc[0]
    assert t["mfe_points"] == pytest.approx(3.0)
    assert t["mae_points"] == pytest.approx(2.0)


# ── the close-only bias (the reason this module exists) ───────────────────────

def _straddle_path():
    """One bar whose range spans both barriers; its close is above the entry.

    A close-series evaluator sees only the +10 close and books a clean winner. The 1s
    high/low show the −5 stop was tagged inside the same second.
    """
    closes = [100.0, 110.0, 110.0]
    highs = [100.0, 110.0, 110.0]
    lows = [100.0, 94.0, 110.0]
    return _path(closes, highs=highs, lows=lows)


def test_stop_first_is_conservative_on_a_straddling_bar():
    r = simulate(_signals([0]), _straddle_path(), _cfg(max_hold_seconds=5))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert bool(t["ambiguous_bar"]) is True


def test_target_first_is_the_optimistic_bound():
    r = simulate(_signals([0]), _straddle_path(), _cfg(max_hold_seconds=5,
                                                       tie_policy="target_first"))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert bool(t["ambiguous_bar"]) is True


def test_close_only_reproduces_the_legacy_blind_spot():
    r = simulate(_signals([0]), _straddle_path(), _cfg(max_hold_seconds=5,
                                                       tie_policy="close_only"))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "target"          # the intrabar stop is invisible
    assert bool(t["ambiguous_bar"]) is False     # and unmeasurable by construction


def test_compare_tie_policies_quantifies_the_gap():
    df = compare_tie_policies(_signals([0]), _straddle_path(), _cfg(max_hold_seconds=5))
    by = df.set_index("tie_policy")
    assert by.loc["close_only", "net_usd"] > by.loc["stop_first", "net_usd"]
    assert by.loc["close_only", "ambiguous_exit_frac"] == pytest.approx(0.0)
    assert by.loc["stop_first", "ambiguous_exit_frac"] == pytest.approx(1.0)


# ── account and risk constraints ─────────────────────────────────────────────

def test_concurrent_position_cap_rejects_the_overflow():
    r = simulate(_signals(range(8)), _path([100.0] * 40),
                 _cfg(max_hold_seconds=30, max_concurrent_positions=3))
    assert len(r.trades) == 3
    assert r.daily.iloc[0]["reject_capacity"] == 5


def test_margin_capacity_binds_before_the_position_cap():
    # $2,500 equity at $1,000/contract funds 2 positions, not the 5 the cap allows.
    r = simulate(_signals(range(5)), _path([100.0] * 40),
                 _cfg(max_hold_seconds=30, max_concurrent_positions=5,
                      starting_equity_usd=2_500.0))
    assert len(r.trades) == 2
    assert r.daily.iloc[0]["reject_margin"] == 3


def test_cooldown_spaces_entries():
    r = simulate(_signals([0, 5, 30, 35, 60]), _path([100.0] * 120),
                 _cfg(max_hold_seconds=10, cooldown_seconds=30))
    assert len(r.trades) == 3                      # t=0, 30, 60
    assert r.daily.iloc[0]["reject_cooldown"] == 2


def test_opposing_sides_blocked_by_default():
    sig = pd.concat([_signals([0], direction="up"),
                     _signals([1], direction="down")], ignore_index=True)
    blocked = simulate(sig, _path([100.0] * 40), _cfg(max_hold_seconds=30))
    assert len(blocked.trades) == 1
    assert blocked.daily.iloc[0]["reject_opposing"] == 1
    hedged = simulate(sig, _path([100.0] * 40),
                      _cfg(max_hold_seconds=30, allow_opposing_positions=True))
    assert len(hedged.trades) == 2


def test_tau_gate_drops_low_confidence_signals():
    sig = pd.concat([_signals([0], confidence=0.60),
                     _signals([20], confidence=0.90)], ignore_index=True)
    r = simulate(sig, _path([100.0] * 60), _cfg(max_hold_seconds=10, tau=0.80))
    assert len(r.trades) == 1
    assert r.trades.iloc[0]["entry_ts"] == pd.Timestamp(_OPEN_UTC) + pd.Timedelta(seconds=20)


def test_trade_window_excludes_signals_outside_it():
    # 08:30 open; window 08:30-08:31 admits t=0 but not t=120 (08:32).
    r = simulate(_signals([0, 120]), _path([100.0] * 400),
                 _cfg(max_hold_seconds=10, trade_end="08:31", flatten_at="08:40"))
    assert len(r.trades) == 1


def test_forced_flatten_at_session_cutoff():
    # 4-minute path, flatten at 08:33 → a 08:32 entry cannot run its 10-minute hold.
    r = simulate(_signals([90]), _path([100.0] * 300),
                 _cfg(max_hold_seconds=600, trade_end="08:33", flatten_at="08:33"))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "flatten"
    assert t["exit_ts"].tz_convert("America/Chicago").strftime("%H:%M") == "08:33"


def test_daily_loss_halt_flattens_and_blocks_reentry():
    # One long into a 60-point collapse: $5,000 account, 50% limit = $2,500 = 125 pts.
    closes = [100.0] * 3 + [40.0] + [100.0] * 60
    r = simulate(_signals([0, 30, 50]), _path(closes),
                 _cfg(target_points=200.0, stop_points=200.0, max_hold_seconds=600,
                      starting_equity_usd=5_000.0, contracts_per_trade=5,
                      daily_loss_limit_frac=0.5))
    day = r.daily.iloc[0]
    assert bool(day["halted"]) is True
    assert r.trades.iloc[0]["exit_reason"] == "halt"
    assert day["reject_halted"] >= 1               # later signals refused
    assert len(r.trades) == 1


def test_equity_compounds_across_sessions():
    d1 = _path([100.0] * 5 + [110.0] + [100.0] * 5)
    d2 = _path([100.0] * 5 + [110.0] + [100.0] * 5, start="2026-03-03 14:30:00+00:00")
    sig = pd.concat([_signals([0]),
                     _signals([0], start="2026-03-03 14:30:00+00:00")], ignore_index=True)
    r = simulate(sig, pd.concat([d1, d2]), _cfg(starting_equity_usd=10_000.0))
    assert len(r.trades) == 2
    assert r.daily.iloc[1]["equity_open_usd"] == pytest.approx(10_200.0)
    assert r.kpis["final_equity_usd"] == pytest.approx(10_400.0)


# ── KPIs, frontier and scaling ────────────────────────────────────────────────

def test_kpis_report_frequency_not_just_economics():
    d1 = _path([100.0] * 5 + [110.0] + [100.0] * 5)
    d2 = _path([100.0] * 11, start="2026-03-03 14:30:00+00:00")   # no signal
    r = simulate(_signals([0]), pd.concat([d1, d2]), _cfg())
    k = r.kpis
    assert k["n_sessions"] == 2
    assert k["flat_session_frac"] == pytest.approx(0.5)
    assert k["trades_per_session"] == pytest.approx(0.5)
    assert k["hit_rate"] == pytest.approx(1.0)
    assert "n_trades" in r.summary()


def test_drawdown_counts_a_leading_loss():
    closes = [100.0] * 3 + [95.0] + [100.0] * 20
    r = simulate(_signals([0]), _path(closes), _cfg())
    assert r.kpis["max_drawdown_usd"] == pytest.approx(-100.0)   # 5 pts x $20


def test_sweep_tau_is_monotone_in_trade_count():
    sig = pd.DataFrame({
        "ts": [pd.Timestamp(_OPEN_UTC) + pd.Timedelta(seconds=s) for s in (0, 20, 40)],
        "direction": ["up"] * 3,
        "confidence": [0.55, 0.75, 0.95],
    })
    df = sweep_tau(sig, _path([100.0] * 120), _cfg(max_hold_seconds=10),
                   taus=(0.5, 0.7, 0.9))
    assert list(df["n_trades"]) == [3, 2, 1]
    assert df["flat_session_frac"].tolist() == [0.0, 0.0, 0.0]


def test_required_scaling_reports_contracts_and_equity():
    kpis = {"median_month_net_usd": 1_000.0, "max_drawdown_usd": -500.0}
    out = required_scaling(kpis, 50_000.0, _cfg(contracts_per_trade=1))
    assert out["feasible"] is True
    assert out["required_contracts"] == 50
    assert out["required_margin_usd"] == pytest.approx(50_000.0)
    assert out["scaled_max_drawdown_usd"] == pytest.approx(25_000.0)
    assert out["required_equity_usd"] == pytest.approx(75_000.0)


def test_required_scaling_refuses_a_negative_edge():
    out = required_scaling({"median_month_net_usd": -10.0}, 50_000.0, _cfg())
    assert out["feasible"] is False


def test_no_signals_yields_empty_but_well_formed_result():
    r = simulate(_signals([], direction="up"), _path([100.0] * 10), _cfg())
    assert r.trades.empty
    assert r.kpis["n_trades"] == 0 and r.kpis["net_usd"] == 0.0


def test_config_validation_rejects_bad_geometry():
    with pytest.raises(ValueError, match="tie_policy"):
        SimConfig(tie_policy="nope")
    with pytest.raises(ValueError, match="flatten_at"):
        SimConfig(trade_end="11:00", flatten_at="10:00")
    with pytest.raises(ValueError, match="positive"):
        SimConfig(stop_points=0.0)


# ── the economic boundary (step 4) ────────────────────────────────────────────
#
# The explicit comparison that licenses treating src/evaluation/
# economic.py as the authority for pricing, commission, multiplier, PnL and the
# ledger. If these ever disagree, the canonical layer must NOT be swapped in
# silently -- that is the whole point of pinning the equality.

def test_canonical_ledger_agrees_with_the_simulator_blotter():
    """Per trade and in aggregate, under non-zero commission and >1 contract."""
    from src.backtest.portfolio_sim import trade_ledger

    closes = [100.0] * 5 + [110.0] * 5 + [100.0] * 5 + [90.0] * 5
    cfg = _cfg(commission_per_side_usd=2.50, contracts_per_trade=3,
               cooldown_seconds=0, max_hold_seconds=60)
    res = simulate(_signals([0, 10]), _path(closes), cfg)
    assert len(res.trades) >= 1, "need at least one filled trade to compare"

    ledger = trade_ledger(res)
    assert ledger.n_trades == len(res.trades)
    for row, priced in zip(res.trades.to_dict("records"), ledger.trades):
        assert priced.entry_price == pytest.approx(row["entry_px"])
        assert priced.exit_price == pytest.approx(row["exit_px"])
        assert priced.net_dollars == pytest.approx(row["net_usd"]), (
            "canonical net dollars must equal the simulator's own net_usd")
    assert ledger.net_dollars == pytest.approx(float(res.trades["net_usd"].sum()))
    assert ledger.total_commission == pytest.approx(
        cfg.round_trip_commission_usd * len(res.trades))


def test_canonical_ledger_does_not_double_count_slippage():
    """The blotter's prices already include entry/stop/target slippage, so the
    ledger must price them as filled legs and add none of its own."""
    from src.backtest.portfolio_sim import trade_ledger

    closes = [100.0] * 5 + [110.0] * 10
    cfg = _cfg(entry_slippage_ticks=4.0, target_slippage_ticks=2.0,
               commission_per_side_usd=0.0)
    res = simulate(_signals([0]), _path(closes), cfg)
    ledger = trade_ledger(res)
    row = res.trades.iloc[0]
    priced = ledger.trades[0]
    assert priced.entry_price == pytest.approx(row["entry_px"])
    assert priced.exit_price == pytest.approx(row["exit_px"])
    # Zero commission and no extra slippage: dollars are exactly the realised
    # price change times the multiplier.
    assert priced.gross_dollars == pytest.approx(
        (row["exit_px"] - row["entry_px"]) * cfg.point_value_usd
        * cfg.contracts_per_trade)
    assert priced.net_dollars == pytest.approx(priced.gross_dollars)


def test_cost_sensitivity_is_the_canonical_layers_job():
    """Repricing the same fills under several cost assumptions goes through
    economic.evaluate_cost_sensitivity -- the simulator does not reimplement it,
    and it does not re-run the path (which fills happened does not depend on
    what they cost)."""
    from src.backtest.portfolio_sim import cost_sensitivity
    from src.evaluation.economic import EconomicAssumptions

    closes = [100.0] * 5 + [110.0] * 10
    res = simulate(_signals([0]), _path(closes), _cfg(commission_per_side_usd=0.0))
    scenarios = [
        EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                            label="free"),
        EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=5.0,
                            label="retail"),
    ]
    out = cost_sensitivity(res, scenarios)
    assert set(out) == {"free", "retail"}
    assert out["free"].net_dollars - out["retail"].net_dollars == pytest.approx(
        5.0 * out["free"].n_trades)
    # The fills are identical across scenarios; only the cost differs.
    assert ([t.entry_price for t in out["free"].trades]
            == [t.entry_price for t in out["retail"].trades])


def test_simulator_keeps_only_one_cost_model():
    """SimConfig's cost fields exist, but nothing in this module may compute a
    commission or a multiplier except through the canonical bridge. The guard is
    that the two agree (above) plus: no second commission formula appears."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "backtest" / "portfolio_sim.py"
    text = source.read_text()
    # round_trip_commission_usd is the ONE derived commission in this module.
    assert text.count("commission_per_side_usd *") <= 1
    assert text.count("2.0 * self.commission_per_side_usd") == 2  # property + bridge


def test_the_simulator_hands_over_real_order_types():
    """A stop exit stays a STOP, a market entry stays MARKET, and a resting
    target is a LIMIT only when no fill-through is assumed."""
    from src.backtest.portfolio_sim import trade_ledger

    # target hit, no target slippage -> resting LIMIT
    hit = simulate(_signals([0]), _path([100.0] * 5 + [110.0] * 10), _cfg())
    row = trade_ledger(hit).trades[0]
    assert row.entry_order_type == "MARKET" and row.exit_order_type == "LIMIT"
    assert row.exit_role == "EXIT_TARGET"

    # target hit, but the fill model fills THROUGH the level -> MARKET, because
    # a limit order cannot do that.
    through = simulate(_signals([0]), _path([100.0] * 5 + [110.0] * 10),
                       _cfg(target_slippage_ticks=2.0))
    row = trade_ledger(through).trades[0]
    assert row.exit_order_type == "MARKET" and row.exit_role == "EXIT_TARGET"

    # stop hit -> STOP, carrying its trigger
    stopped = simulate(_signals([0]), _path([100.0] * 5 + [90.0] * 10), _cfg())
    row = trade_ledger(stopped).trades[0]
    assert row.exit_order_type == "STOP" and row.exit_role == "EXIT_STOP"

    # hold-limit flatten -> MARKET with the time role
    timed = simulate(_signals([0]), _path([100.0] * 20), _cfg(max_hold_seconds=3))
    if len(timed.trades):
        row = trade_ledger(timed).trades[0]
        assert row.exit_order_type == "MARKET" and row.exit_role == "EXIT_TIME"


def test_every_simulator_fill_is_attributed_to_the_path_simulator():
    from src.backtest.portfolio_sim import trade_ledger
    from src.evaluation.economic import FILL_SOURCE_PATH_SIM

    res = simulate(_signals([0]), _path([100.0] * 5 + [110.0] * 10), _cfg())
    for row in trade_ledger(res).trades:
        assert row.entry_fill_source == FILL_SOURCE_PATH_SIM
        assert row.exit_fill_source == FILL_SOURCE_PATH_SIM
        # Already executed upstream, so the canonical layer adds no slippage.
        assert row.entry_slippage_points == 0.0
        assert row.exit_slippage_points == 0.0


def test_execution_sensitivity_moves_the_fills_and_cost_sensitivity_does_not():
    """The two concepts, on a real simulated book."""
    from src.backtest.portfolio_sim import cost_sensitivity, execution_sensitivity
    from src.evaluation.economic import EconomicAssumptions

    signals = _signals([0])
    path = _path([100.0] * 5 + [110.0] * 10)
    cfg = _cfg(commission_per_side_usd=0.0)

    slippy = [
        EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                            entry_slippage_points=0.0, label="clean"),
        EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                            entry_slippage_points=1.0, label="one_point"),
    ]
    moved = execution_sensitivity(signals, path, cfg, slippy)
    assert moved["clean"].net_dollars > moved["one_point"].net_dollars, (
        "a worse entry fill must reduce PnL -- if these are equal the slippage "
        "assumption is not reaching the fill logic")

    # Fixed-fill sensitivity refuses the same question, by name.
    res = simulate(signals, path, cfg)
    with pytest.raises(ValueError, match="evaluate_execution_sensitivity"):
        cost_sensitivity(res, slippy)
