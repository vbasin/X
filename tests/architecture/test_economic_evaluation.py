"""Tests for src.evaluation.economic -- the execution contract.

Pure, DB-free. Numbers are chosen so every expected value is hand-checkable.
"""
import math
from pathlib import Path

import pytest

from src.evaluation.economic import (
    EconomicAssumptions,
    Leg,
    OrderType,
    Quote,
    Trade,
    compute_trade_ledger,
    evaluate_cost_sensitivity,
    execution_price,
    load_assumptions,
    zero_cost,
)

COSTS_YAML = Path(__file__).resolve().parents[2] / "config" / "execution_costs.yaml"

Q_IN = Quote(bid=100.00, ask=100.25)    # mid 100.125
Q_OUT = Quote(bid=110.00, ask=110.25)   # mid 110.125


def mkt(q, ts=None):
    return Leg(OrderType.MARKET, quote=q, ts=ts)


def nq(**kw):
    base = dict(contract_multiplier=20.0, commission_per_trade=0.0)
    base.update(kw)
    return EconomicAssumptions(**base)


# --- 1. MARKET uses executable bid/ask, never midpoint -----------------------

@pytest.mark.parametrize("direction,is_entry,expected", [
    (+1, True, 100.25),   # long entry  -> ask
    (-1, True, 100.00),   # short entry -> bid
    (+1, False, 100.00),  # long exit   -> bid
    (-1, False, 100.25),  # short exit  -> ask
])
def test_market_fills_at_executable_side(direction, is_entry, expected):
    p = execution_price(mkt(Q_IN), direction=direction, is_entry=is_entry, slippage_points=0.0)
    assert p == expected
    assert p != 100.125  # never the midpoint


def test_round_trip_pays_the_spread_without_any_slippage_term():
    """Spread is captured by bid/ask pricing, not by slippage."""
    flat = Quote(bid=100.0, ask=100.25)
    led = compute_trade_ledger([Trade("t", 1, mkt(flat), mkt(flat))], nq())
    assert led.trades[0].gross_points == -0.25
    assert led.assumptions.slippage_points_per_side == 0.0


# --- 2. zero slippage leaves the executable price unchanged ------------------

def test_zero_slippage_leaves_bid_ask_unchanged():
    assert execution_price(mkt(Q_IN), direction=1, is_entry=True, slippage_points=0.0) == Q_IN.ask
    assert execution_price(mkt(Q_IN), direction=1, is_entry=False, slippage_points=0.0) == Q_IN.bid


# --- 3. nonzero MARKET slippage moves execution adversely --------------------

def test_market_slippage_is_adverse_on_every_side():
    s = 0.5
    assert execution_price(mkt(Q_IN), direction=1, is_entry=True, slippage_points=s) == 100.75
    assert execution_price(mkt(Q_IN), direction=1, is_entry=False, slippage_points=s) == 99.50
    assert execution_price(mkt(Q_IN), direction=-1, is_entry=True, slippage_points=s) == 99.50
    assert execution_price(mkt(Q_IN), direction=-1, is_entry=False, slippage_points=s) == 100.75


def test_slippage_alters_price_before_gross_pnl():
    t = [Trade("t", 1, mkt(Q_IN), mkt(Q_OUT))]
    base = compute_trade_ledger(t, nq()).trades[0]
    slip = compute_trade_ledger(t, nq(slippage_points_per_side=0.5)).trades[0]
    assert (slip.entry_price, slip.exit_price) == (100.75, 109.50)
    assert slip.gross_points == base.gross_points - 1.0


# --- 4. LIMIT receives no slippage -------------------------------------------

def test_limit_fills_at_limit_and_ignores_slippage():
    leg = Leg(OrderType.LIMIT, limit_price=100.0)
    assert execution_price(leg, direction=1, is_entry=True, slippage_points=5.0) == 100.0


def test_limit_improvement_only_when_explicitly_resolved_upstream():
    """A better price is used only when the execution source reports it."""
    from src.evaluation.economic import FILL_SOURCE_REPORTED

    better = Leg(OrderType.LIMIT, limit_price=100.0, resolved_fill_price=99.75,
                 fill_source=FILL_SOURCE_REPORTED)
    assert execution_price(better, direction=1, is_entry=True, slippage_points=0) == 99.75
    plain = Leg(OrderType.LIMIT, limit_price=100.0)
    assert execution_price(plain, direction=1, is_entry=True, slippage_points=5.0) == 100.0


def test_limit_cannot_fill_through_its_limit():
    from src.evaluation.economic import FILL_SOURCE_REPORTED

    worse = Leg(OrderType.LIMIT, limit_price=100.0, resolved_fill_price=100.25,
                fill_source=FILL_SOURCE_REPORTED)
    with pytest.raises(ValueError, match="cannot fill through"):
        execution_price(worse, direction=1, is_entry=True, slippage_points=0)


# --- 5. STOP triggers, then follows MARKET semantics -------------------------

def test_stop_price_is_a_trigger_not_the_fill():
    gap = Quote(bid=94.00, ask=94.25)  # gapped through a 95.00 stop
    stop = Leg(OrderType.STOP, stop_price=95.0, quote=gap)
    # long exit on a stop sells at the BID at execution time, not at 95.00
    assert execution_price(stop, direction=1, is_entry=False, slippage_points=0) == 94.00


def test_stop_takes_market_slippage():
    stop = Leg(OrderType.STOP, stop_price=95.0, quote=Quote(94.0, 94.25))
    m = mkt(Quote(94.0, 94.25))
    for s in (0.0, 0.5):
        assert execution_price(stop, direction=1, is_entry=False, slippage_points=s) == \
               execution_price(m, direction=1, is_entry=False, slippage_points=s)


def test_legs_reject_missing_required_prices():
    with pytest.raises(ValueError):
        Leg(OrderType.MARKET)
    with pytest.raises(ValueError):
        Leg(OrderType.STOP, quote=Q_IN)          # no trigger
    with pytest.raises(ValueError):
        Leg(OrderType.LIMIT)
    with pytest.raises(ValueError):
        # A resolved fill is legal on ANY order type now, but must be
        # attributed: an unsourced fill price is not auditable.
        Leg(OrderType.MARKET, quote=Q_IN, resolved_fill_price=1.0)


# --- 6. commission applies to every filled order type ------------------------

@pytest.mark.parametrize("entry,exit_", [
    (mkt(Q_IN), mkt(Q_OUT)),
    (Leg(OrderType.LIMIT, limit_price=100.0), Leg(OrderType.LIMIT, limit_price=110.0)),
    (mkt(Q_IN), Leg(OrderType.STOP, stop_price=110.0, quote=Q_OUT)),
])
def test_commission_applies_to_all_filled_order_types(entry, exit_):
    r = compute_trade_ledger([Trade("t", 1, entry, exit_)], nq(commission_per_trade=3.36)).trades[0]
    assert r.commission == 3.36
    assert r.net_dollars == pytest.approx(r.gross_dollars - 3.36)


# --- 7. quantity scales PnL and commission -----------------------------------

def test_quantity_scales_gross_and_commission_linearly():
    a = nq(commission_per_trade=3.36)
    one = compute_trade_ledger([Trade("t", 1, mkt(Q_IN), mkt(Q_OUT), quantity=1)], a).trades[0]
    three = compute_trade_ledger([Trade("t", 1, mkt(Q_IN), mkt(Q_OUT), quantity=3)], a).trades[0]
    assert three.gross_dollars == pytest.approx(3 * one.gross_dollars)
    assert three.commission == pytest.approx(3 * 3.36)
    # long: bought 100.25, sold 110.00 -> 9.75 pts x $20 x 3
    assert three.gross_dollars == pytest.approx(9.75 * 20.0 * 3)


def test_quantity_must_be_positive():
    with pytest.raises(ValueError):
        Trade("t", 1, mkt(Q_IN), mkt(Q_OUT), quantity=0)


# --- 8. NQ $3.36 and MNQ-equivalent $1.03 are independent --------------------

def test_yaml_commissions_are_independent_assumptions():
    nq_a = load_assumptions(COSTS_YAML, "NQ")
    mnq_a = load_assumptions(COSTS_YAML, "MNQ_equivalent")
    assert nq_a.commission_per_trade == 3.36
    assert mnq_a.commission_per_trade == 1.03
    assert mnq_a.commission_per_trade != pytest.approx(0.1 * nq_a.commission_per_trade)


def test_only_the_multiplier_is_derived_for_mnq_equivalent():
    assert load_assumptions(COSTS_YAML, "NQ").contract_multiplier == 20.0
    assert load_assumptions(COSTS_YAML, "MNQ_equivalent").contract_multiplier == pytest.approx(2.0)


def test_commission_is_configurable_not_hardcoded(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("profiles:\n  NQ: {contract_multiplier: 20.0, commission_per_trade: 9.99}\n")
    assert load_assumptions(y, "NQ").commission_per_trade == 9.99


# --- 9. Track C MNQ-equivalent still prices off NQ data ----------------------

def test_mnq_equivalent_uses_nq_underlying_prices():
    """Same NQ quotes, same execution prices, 0.1x the dollars, own commission."""
    nq_a = load_assumptions(COSTS_YAML, "NQ")
    mnq_a = load_assumptions(COSTS_YAML, "MNQ_equivalent")
    assert mnq_a.underlying == "NQ" and nq_a.underlying == "NQ"

    t = [Trade("t", 1, mkt(Q_IN), mkt(Q_OUT))]
    r_nq = compute_trade_ledger(t, nq_a).trades[0]
    r_mnq = compute_trade_ledger(t, mnq_a).trades[0]
    assert (r_mnq.entry_price, r_mnq.exit_price) == (r_nq.entry_price, r_nq.exit_price)
    assert r_mnq.gross_dollars == pytest.approx(0.1 * r_nq.gross_dollars)
    assert r_mnq.net_dollars == pytest.approx(r_mnq.gross_dollars - 1.03)


# --- retained guards ----------------------------------------------------------

def test_zero_cost_requires_explicit_multiplier():
    with pytest.raises(TypeError):
        zero_cost()  # type: ignore[call-arg]
    assert zero_cost(contract_multiplier=2.0).commission_per_trade == 0.0


def test_empty_trades_returns_nan_stats_not_error():
    led = compute_trade_ledger([], nq())
    assert led.n_trades == 0 and math.isnan(led.hit_rate)


def test_ledger_exposes_no_significance_verdict():
    led = compute_trade_ledger([Trade("t", 1, mkt(Q_IN), mkt(Q_OUT))], nq())
    assert not hasattr(led, "is_significant")
    assert hasattr(led, "t_stat_iid")


def test_cost_sensitivity_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_cost_sensitivity([], [nq(label="x"), nq(label="x")])


def test_holding_period_is_recorded_on_the_result():
    t = Trade("t", 1, mkt(Q_IN, ts="2026-01-01T09:30:00Z"), mkt(Q_OUT, ts="2026-01-01T10:00:00Z"))
    r = compute_trade_ledger([t], nq()).trades[0]
    assert (r.entry_ts, r.exit_ts) == ("2026-01-01T09:30:00Z", "2026-01-01T10:00:00Z")


# ──────────────────────────────────────────────────────────────────────────────
# The boundary with src/backtest/portfolio_sim.py (step 6).
#
# ADR-136's founding audit found four mutually inconsistent economic
# implementations. The fix is not "add a fifth": the canonical layer defines
# what a point is worth, what a round trip costs and how a leg slips, and the
# simulator keeps trade sequencing, path resolution and account state. These
# tests pin that split so the two cannot drift apart again.
# ──────────────────────────────────────────────────────────────────────────────

def test_sim_config_projects_losslessly_into_canonical_assumptions():
    from src.backtest.portfolio_sim import SimConfig

    cfg = SimConfig()
    a = cfg.as_assumptions()
    assert a.contract_multiplier == cfg.point_value_usd
    assert a.commission_per_trade == 2.0 * cfg.commission_per_side_usd
    # Per-leg slippage survives the trip: a resting target and a swept stop are
    # not the same assumption, and the canonical object can now say so.
    assert a.slippage_for("entry") == cfg.entry_slip
    assert a.slippage_for("stop") == cfg.stop_slip
    assert a.slippage_for("target") == cfg.target_slip
    assert a.slippage_for("stop") != a.slippage_for("target")

    back = SimConfig.from_assumptions(a, tick_size=cfg.tick_size)
    for field_name in ("point_value_usd", "commission_per_side_usd", "tick_size",
                       "entry_slippage_ticks", "stop_slippage_ticks",
                       "target_slippage_ticks"):
        assert getattr(back, field_name) == getattr(cfg, field_name), field_name


def test_assumptions_without_per_leg_slippage_keep_the_single_number_model():
    """The per-leg fields are a refinement, not a new requirement."""
    a = EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=3.36,
                            slippage_points_per_side=0.5)
    assert a.slippage_for("entry") == 0.5
    assert a.slippage_for("stop") == 0.5
    assert a.slippage_for("target") == 0.5


def test_sim_defaults_are_not_silently_replaced_by_the_cost_profile():
    """config/execution_costs.yaml's NQ profile is $3.36 all-in; the simulator
    assumes $5.00. Adopting the cheaper number by default would restate every
    historical KPI, so it must take an explicit call."""
    from pathlib import Path

    from src.backtest.portfolio_sim import SimConfig

    assert SimConfig().round_trip_commission_usd == 5.0
    nq = load_assumptions(Path("config/execution_costs.yaml"), "NQ")
    explicit = SimConfig.from_assumptions(nq)
    assert explicit.round_trip_commission_usd == 3.36
    assert SimConfig().round_trip_commission_usd == 5.0


def test_both_layers_net_the_same_dollars_for_one_trade():
    """One long round trip, priced by the canonical ledger under the
    simulator's own assumptions, must net what the simulator's arithmetic nets.
    This is the equality that makes one of the two implementations redundant."""
    from src.backtest.portfolio_sim import SimConfig

    cfg = SimConfig(contracts_per_trade=1)
    a = cfg.as_assumptions()

    entry_px, exit_px = 20_000.0, 20_050.0
    trade = Trade(
        trade_id="t1", direction=1, quantity=1.0,
        entry=Leg(order_type=OrderType.LIMIT, limit_price=entry_px),
        exit=Leg(order_type=OrderType.LIMIT, limit_price=exit_px),
    )
    ledger = compute_trade_ledger([trade], a)

    gross_points = exit_px - entry_px
    sim_net = (gross_points * cfg.point_value_usd * cfg.contracts_per_trade
               - cfg.round_trip_commission_usd)
    assert ledger.net_dollars == pytest.approx(sim_net)
    assert ledger.total_commission == pytest.approx(cfg.round_trip_commission_usd)
    assert ledger.gross_dollars == pytest.approx(gross_points * cfg.point_value_usd)


def test_path_utilities_are_not_imported_privately_from_the_simulator():
    """prepare_path / session_grid are path-shape primitives, not economics.
    They were being imported as PRIVATE names from portfolio_sim by six
    modules; they now live in src/features/session_path.py."""
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py",
         "from src.backtest.portfolio_sim import", "src", "scripts"],
        cwd=repo, capture_output=True, text=True).stdout
    offenders = [line for line in out.splitlines()
                 if "_session_grid" in line or "_parse_hhmm" in line]
    assert offenders == [], f"private path helpers still imported from the sim: {offenders}"


# ── order intent vs resolved fill (the corrected contract) ────────────────────

def test_a_resolved_stop_is_still_a_stop_in_the_ledger():
    """The defect this replaced: an adapter rewrote resolved MARKET/STOP fills
    as LIMIT legs to stop slippage being applied twice, losing the order
    semantics. Intent and resolution are now independent."""
    from src.evaluation.economic import FILL_SOURCE_PATH_SIM, LegRole

    trade = Trade(
        trade_id="t", direction=1, quantity=1.0,
        entry=Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY,
                  resolved_fill_price=20_000.25, fill_source=FILL_SOURCE_PATH_SIM),
        exit=Leg(order_type=OrderType.STOP, role=LegRole.EXIT_STOP,
                 stop_price=19_994.0, resolved_fill_price=19_993.5,
                 fill_source=FILL_SOURCE_PATH_SIM),
    )
    a = EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                            slippage_points_per_side=5.0)
    row = compute_trade_ledger([trade], a).trades[0]
    assert row.entry_order_type == "MARKET"
    assert row.exit_order_type == "STOP"
    assert row.exit_role == "EXIT_STOP"
    assert row.entry_fill_source == FILL_SOURCE_PATH_SIM
    # Resolved fills are used verbatim and no slippage is added on top.
    assert row.entry_price == pytest.approx(20_000.25)
    assert row.exit_price == pytest.approx(19_993.5)
    assert row.entry_slippage_points == 0.0
    assert row.exit_slippage_points == 0.0


def test_a_resolved_fill_must_name_its_source():
    from src.evaluation.economic import LegRole

    with pytest.raises(ValueError, match="fill_source"):
        Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY,
            resolved_fill_price=100.0)


def test_per_leg_slippage_participates_when_the_leg_is_priced_here():
    """entry/stop/target_slippage_points were previously recorded and ignored:
    compute_trade_ledger only ever used the generic per-side number."""
    from src.evaluation.economic import LegRole

    quote = Quote(bid=99.0, ask=101.0)
    trade = Trade(
        trade_id="t", direction=1, quantity=1.0,
        entry=Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY, quote=quote),
        exit=Leg(order_type=OrderType.STOP, role=LegRole.EXIT_STOP,
                 stop_price=95.0, quote=quote),
    )
    a = EconomicAssumptions(
        contract_multiplier=1.0, commission_per_trade=0.0,
        slippage_points_per_side=0.0,
        entry_slippage_points=1.0, stop_slippage_points=3.0)
    row = compute_trade_ledger([trade], a).trades[0]
    assert row.entry_price == pytest.approx(101.0 + 1.0)   # buy the ask, +entry slip
    assert row.exit_price == pytest.approx(99.0 - 3.0)     # sell the bid, -stop slip
    assert row.entry_slippage_points == pytest.approx(1.0)
    assert row.exit_slippage_points == pytest.approx(3.0)


def test_a_time_exit_uses_the_generic_assumption_not_the_stops():
    from src.evaluation.economic import LegRole

    a = EconomicAssumptions(
        contract_multiplier=1.0, commission_per_trade=0.0,
        slippage_points_per_side=0.5, stop_slippage_points=9.0)
    assert a.slippage_for(LegRole.EXIT_TIME) == pytest.approx(0.5)
    assert a.slippage_for(LegRole.EXIT_STOP) == pytest.approx(9.0)


def test_a_limit_cannot_be_resolved_through_its_own_limit():
    from src.evaluation.economic import FILL_SOURCE_PATH_SIM, LegRole

    trade = Trade(
        trade_id="t", direction=1, quantity=1.0,
        entry=Leg(order_type=OrderType.LIMIT, role=LegRole.ENTRY, limit_price=100.0),
        exit=Leg(order_type=OrderType.LIMIT, role=LegRole.EXIT_TARGET,
                 limit_price=110.0, resolved_fill_price=109.0,   # worse for a seller
                 fill_source=FILL_SOURCE_PATH_SIM),
    )
    a = EconomicAssumptions(contract_multiplier=1.0, commission_per_trade=0.0)
    with pytest.raises(ValueError, match="cannot fill through its limit"):
        compute_trade_ledger([trade], a)


def test_fixed_fill_sensitivity_refuses_a_scenario_that_cannot_take_effect():
    """Varying slippage against already-resolved fills returns an identical
    ledger under a label claiming otherwise. That is refused by name."""
    from src.evaluation.economic import FILL_SOURCE_PATH_SIM, LegRole

    trade = Trade(
        trade_id="t", direction=1, quantity=1.0,
        entry=Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY,
                  resolved_fill_price=100.0, fill_source=FILL_SOURCE_PATH_SIM),
        exit=Leg(order_type=OrderType.MARKET, role=LegRole.EXIT_TIME,
                 resolved_fill_price=110.0, fill_source=FILL_SOURCE_PATH_SIM),
    )
    base = EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                              label="base")
    cheaper = EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=5.0,
                                  label="commission")
    slipped = EconomicAssumptions(contract_multiplier=20.0, commission_per_trade=0.0,
                                  slippage_points_per_side=1.0, label="slippage")
    # Commission-only variation is exactly what fixed-fill sensitivity is for.
    out = evaluate_cost_sensitivity([trade], [base, cheaper])
    assert out["base"].net_dollars - out["commission"].net_dollars == pytest.approx(5.0)
    with pytest.raises(ValueError, match="evaluate_execution_sensitivity"):
        evaluate_cost_sensitivity([trade], [base, slipped])
    # Explicit opt-out still works for a caller that wants the identical ledger.
    relaxed = evaluate_cost_sensitivity([trade], [base, slipped], strict=False)
    assert relaxed["slippage"].net_dollars == pytest.approx(out["base"].net_dollars)


def test_execution_sensitivity_lets_the_fills_move():
    """The other half: slippage changes which price an order gets, so the book
    is re-derived rather than repriced."""
    from src.evaluation.economic import (
        FILL_SOURCE_TBBO, LegRole, evaluate_execution_sensitivity,
    )

    def _resimulate(scenario):
        # A stand-in execution source: the entry fills worse as slippage rises.
        slip = scenario.slippage_for(LegRole.ENTRY)
        return [Trade(
            trade_id="t", direction=1, quantity=1.0,
            entry=Leg(order_type=OrderType.MARKET, role=LegRole.ENTRY,
                      resolved_fill_price=100.0 + slip,
                      fill_source=FILL_SOURCE_TBBO),
            exit=Leg(order_type=OrderType.MARKET, role=LegRole.EXIT_TIME,
                     resolved_fill_price=110.0, fill_source=FILL_SOURCE_TBBO),
        )]

    scenarios = [
        EconomicAssumptions(contract_multiplier=1.0, commission_per_trade=0.0,
                            slippage_points_per_side=0.0, label="none"),
        EconomicAssumptions(contract_multiplier=1.0, commission_per_trade=0.0,
                            slippage_points_per_side=2.0, label="two_points"),
    ]
    out = evaluate_execution_sensitivity(scenarios, resimulate=_resimulate)
    assert out["none"].net_dollars == pytest.approx(10.0)
    assert out["two_points"].net_dollars == pytest.approx(8.0)
    assert out["none"].trades[0].entry_fill_source == FILL_SOURCE_TBBO
