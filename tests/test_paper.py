"""Live paper engine — limit entry + OCO on trade prints (not bar-close)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.paper.engine import MarketTick, PaperEngine
from src.paper.policy import PaperPolicy, load_paper_policy


def _ts(sec: float = 0.0) -> datetime:
    return datetime(2026, 8, 4, 14, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=sec)


def _policy(**kw) -> PaperPolicy:
    base = dict(
        objective_id="nq-50pt-econ",
        target_points=50.0,
        stop_points=6.0,
        dir_threshold=0.5,
        cost_per_trade=1.0,
        slippage_points=0.0,
        max_hold_seconds=600.0,
        eod_flat=True,
        max_open=1,
        entry_style="last",
        entry_timeout_seconds=60.0,
        session_tz="America/Chicago",
        trading_start="00:00",  # don't gate pre-open in unit tests
        trading_end="23:59",  # don't eod in unit tests
        model_path=None,
        symbol_parent="NQ.FUT",
    )
    base.update(kw)
    return PaperPolicy(**base)


def _tick(px: float, sec: float = 0.0, bid: float | None = None, ask: float | None = None) -> MarketTick:
    return MarketTick(ts=_ts(sec), price=px, size=1, bid=bid if bid is not None else px - 0.25,
                      ask=ask if ask is not None else px + 0.25, symbol="NQH6")


def test_load_paper_policy_from_econ_objective():
    pol = load_paper_policy(objective_id="nq-50pt-econ")
    assert pol.target_points == 50.0
    assert pol.stop_points == 6.0
    assert pol.cost_per_trade == 0.0
    assert pol.trading_start == "08:30"
    assert pol.trading_end == "14:45"


def test_limit_entry_then_target_oco():
    eng = PaperEngine(_policy())
    eng.on_tick(_tick(100.0, 0))
    od = eng.on_signal("up", 0.9, ts=_ts(0))
    assert od is not None
    assert od.kind == "entry_limit"
    assert od.limit_px == 100.0
    # trade through the limit
    fills = eng.on_tick(_tick(100.0, 1))
    assert any(f.kind == "entry_limit" for f in fills)
    assert eng.position is not None
    assert eng.position.side == "long"
    assert abs(eng.position.entry_px - 100.0) < 1e-9
    # target +50
    fills = eng.on_tick(_tick(150.0, 10))
    assert any(f.reason == "target" for f in fills)
    assert eng.position is None
    assert len(eng.closed) == 1
    assert eng.closed[0].exit_reason == "target"
    assert eng.closed[0].pnl_points == pytest.approx(49.0)  # 50 - 1 cost


def test_stop_beats_via_trade_print():
    eng = PaperEngine(_policy())
    eng.on_tick(_tick(100.0, 0))
    eng.on_signal("up", 0.9, ts=_ts(0))
    eng.on_tick(_tick(99.5, 1))  # fill entry (trade below limit still fills buy limit)
    assert eng.position is not None
    entry = eng.position.entry_px  # min(99.5, 100) = 99.5
    fills = eng.on_tick(_tick(entry - 6.5, 5))  # through stop
    assert any(f.reason == "stop" for f in fills)
    assert eng.closed[-1].exit_reason == "stop"
    assert eng.closed[-1].pnl_points < 0


def test_short_bracket():
    eng = PaperEngine(_policy())
    eng.on_tick(_tick(100.0, 0))
    eng.on_signal("down", 0.9, ts=_ts(0))
    eng.on_tick(_tick(100.5, 1))  # fill short limit (trade >= limit)
    assert eng.position is not None and eng.position.side == "short"
    eng.on_tick(_tick(50.0, 20))
    assert eng.closed[-1].exit_reason == "target"
    # entry max(100.5,100)=100.5; exit min(50, target); ~50 pts - cost
    assert eng.closed[-1].pnl_points == pytest.approx(49.5)


def test_entry_timeout():
    eng = PaperEngine(_policy(entry_timeout_seconds=5.0))
    eng.on_tick(_tick(100.0, 0))
    eng.on_signal("up", 0.9, ts=_ts(0))
    # price never comes back to 100 — tick far above, no fill for buy limit at 100
    eng.on_tick(_tick(101.0, 1))
    assert eng.position is None
    assert any(o.kind == "entry_limit" for o in eng.working)
    eng.on_tick(_tick(101.0, 6))  # past timeout
    assert not any(o.kind == "entry_limit" and o.status == "working" for o in eng.working)
    assert eng.skipped["entry_expired"] >= 1


def test_no_double_entry_while_pending_or_open():
    eng = PaperEngine(_policy())
    eng.on_tick(_tick(100.0, 0))
    eng.on_signal("up", 0.9, ts=_ts(0))
    eng.on_signal("up", 0.95, ts=_ts(1))
    assert eng.skipped["pending_entry"] >= 1
    eng.on_tick(_tick(100.0, 2))
    eng.on_signal("up", 0.99, ts=_ts(3))
    assert eng.skipped["in_position"] >= 1


def test_oco_cancel_sibling():
    eng = PaperEngine(_policy())
    eng.on_tick(_tick(100.0, 0))
    eng.on_signal("up", 0.9, ts=_ts(0))
    eng.on_tick(_tick(100.0, 1))
    assert sum(1 for o in eng.working if o.kind in ("target_limit", "stop")) == 2
    eng.on_tick(_tick(150.0, 5))
    assert eng.position is None
    assert not any(o.status == "working" and o.kind in ("target_limit", "stop") for o in eng.working)


def test_join_entry_style_uses_bid():
    eng = PaperEngine(_policy(entry_style="join"))
    eng.on_tick(_tick(100.0, 0, bid=99.75, ask=100.25))
    od = eng.on_signal("up", 0.9, ts=_ts(0))
    assert od is not None
    assert od.limit_px == pytest.approx(99.75)
