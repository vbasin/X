"""Risk-aware VM ↔ NinjaTrader execution bridge."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.execution.config import ExecutionConfig
from src.execution.gateway import ExecutionGateway
from src.execution.ledger import broker_blotter
from src.execution.protocol import SCHEMA_VERSION, atomic_write_json, make_event_id, read_json
from src.execution.simulator import acknowledge_commands


def _config(root: Path, **updates) -> ExecutionConfig:
    live = root / "live"
    base = ExecutionConfig(
        enabled=True,
        mode="SIM",
        account="Sim101",
        objectives=("nq-50pt-econ",),
        ninja_instrument="NQ 09-26",
        quantity=1,
        entry_order_type="MARKET",
        max_positions=1,
        max_orders_per_session=10,
        max_daily_loss_usd=500.0,
        signal_ttl_s=45.0,
        heartbeat_stale_s=30.0,
        command_ttl_s=20.0,
        poll_s=0.1,
        bridge_dir=root / "execution" / "bridge",
        state_dir=root / "execution" / "state",
        signal_path=live / "signals.jsonl",
        heartbeat_path=live / "heartbeat.json",
    )
    return replace(base, **updates)


def _signal(now: datetime, **updates) -> dict:
    base = {
        "ts": now.isoformat(),
        "dir": "up",
        "conf": 0.9,
        "entry_px": 22000.0,
        "objective_id": "nq-50pt-econ",
        "candidate_id": "winner",
        "model": "model.pkl",
        "tau": 0.8,
        "tgt_pts": 50.0,
        "stop_pts": 6.0,
        "horizon_s": 600,
        "instrument": "NQ",
        "point_value": 20.0,
        "session_date": now.date().isoformat(),
    }
    base.update(updates)
    return base


def _write_input(cfg: ExecutionConfig, now: datetime, signals: list[dict]) -> None:
    cfg.signal_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.signal_path.write_text("".join(json.dumps(row) + "\n" for row in signals))
    atomic_write_json(cfg.heartbeat_path, {"heartbeat_at": now.isoformat()})


@pytest.fixture(autouse=True)
def _execution_settings(monkeypatch):
    monkeypatch.setattr("src.webui.regmodel.effective_settings", lambda *args, **kwargs: {
        "execution": "OCO", "tau": 0.8, "max_positions": 1,
        "trading_start": "08:00", "trading_end": "11:00",
    })


def test_gateway_emits_one_atomic_idempotent_command(tmp_path):
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path)
    _write_input(cfg, now, [_signal(now)])

    gateway = ExecutionGateway(cfg)
    assert gateway.run_once(now=now) == {"seen": 1, "emitted": 1, "rejected": 0, "duplicate": 0}
    files = list(cfg.outbox_dir.glob("*.json"))
    assert len(files) == 1
    assert not list(cfg.outbox_dir.glob("*.tmp"))
    command = read_json(files[0])
    assert command["schema_version"] == "v1"
    assert command["mode"] == "SIM" and command["account"] == "Sim101"
    assert command["instrument"] == "NQ 09-26"
    assert command["action"] == "BUY" and command["native_oco_required"] is True
    assert command["target_points"] == 50.0 and command["stop_points"] == 6.0

    assert gateway.run_once(now=now) == {"seen": 0, "emitted": 0, "rejected": 0, "duplicate": 0}
    restarted = ExecutionGateway(cfg)
    assert restarted.run_once(now=now) == {"seen": 0, "emitted": 0, "rejected": 0, "duplicate": 0}


@pytest.mark.parametrize(("config_updates", "signal_updates", "reason"), [
    ({"enabled": False}, {}, "gateway_disabled"),
    ({}, {"ts": "2026-08-06T14:00:00+00:00"}, "stale_signal"),
    ({}, {"conf": 0.5}, "confidence_below_effective_tau"),
    ({}, {"objective_id": "other"}, "objective_not_allowed"),
])
def test_gateway_fails_closed(tmp_path, config_updates, signal_updates, reason):
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path, **config_updates)
    _write_input(cfg, now, [_signal(now, **signal_updates)])
    result = ExecutionGateway(cfg).run_once(now=now)
    assert result["rejected"] == 1 and result["emitted"] == 0
    decision = json.loads(cfg.decision_path.read_text().splitlines()[-1])
    assert decision["reason"] == reason


def test_gateway_rejects_fresh_after_hours_signal(tmp_path):
    now = datetime(2026, 8, 6, 22, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path)
    _write_input(cfg, now, [_signal(now)])
    result = ExecutionGateway(cfg).run_once(now=now)
    assert result["rejected"] == 1 and result["emitted"] == 0
    decision = json.loads(cfg.decision_path.read_text().splitlines()[-1])
    assert decision["reason"] == "outside_execution_session"


def test_kill_switch_blocks_new_command(tmp_path):
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path)
    _write_input(cfg, now, [_signal(now)])
    cfg.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.kill_switch_path.touch()
    result = ExecutionGateway(cfg).run_once(now=now)
    assert result["rejected"] == 1 and not list(cfg.outbox_dir.glob("*.json"))


def _event(command: dict, event_type: str, event_at: datetime, **fields) -> dict:
    discriminator = str(fields.get("role") or "") + str(fields.get("fill_price") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": make_event_id(command["command_id"], event_type, event_at.isoformat(), discriminator),
        "event_type": event_type,
        "event_at": event_at.isoformat(),
        "command_id": command["command_id"],
        "signal_id": command["signal_id"],
        "objective_id": command["objective_id"],
        "session_date": command["session_date"],
        "account": command["account"],
        "instrument": command["instrument"],
        "action": command["action"],
        **fields,
    }


def test_broker_blotter_uses_only_acknowledged_fills(tmp_path):
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path)
    _write_input(cfg, now, [_signal(now)])
    ExecutionGateway(cfg).run_once(now=now)
    command = read_json(next(cfg.outbox_dir.glob("*.json")))
    events = [
        _event(command, "FILLED", now + timedelta(seconds=1), role="ENTRY", fill_qty=1, fill_price=22000.25),
        _event(command, "FILLED", now + timedelta(seconds=10), role="TARGET", fill_qty=1, fill_price=22050.25),
    ]
    for event in events:
        atomic_write_json(cfg.inbox_dir / f"{event['event_id']}.json", event)

    result = broker_blotter(
        "nq-50pt-econ", session_date=now.date().isoformat(), bridge_dir=cfg.bridge_dir)
    assert result["stats"] == {
        "n_orders": 1, "n_open": 0, "n_closed": 1, "n_rejected": 0, "net_pnl_usd": 1000.0,
    }
    row = result["rows"][0]
    assert row["state"] == "CLOSED" and row["exit_reason"] == "target"
    assert row["pnl_points"] == 50.0 and row["pnl_usd"] == 1000.0


def test_protocol_simulator_returns_ack_without_fill(tmp_path):
    now = datetime(2026, 8, 6, 15, 0, tzinfo=timezone.utc)
    cfg = _config(tmp_path)
    _write_input(cfg, now, [_signal(now)])
    ExecutionGateway(cfg).run_once(now=now)
    assert acknowledge_commands(cfg.bridge_dir) == 1
    assert acknowledge_commands(cfg.bridge_dir) == 0
    result = broker_blotter(
        "nq-50pt-econ", session_date=now.date().isoformat(), bridge_dir=cfg.bridge_dir)
    assert result["stats"]["n_orders"] == 1
    assert result["rows"][0]["state"] == "ORDER_ACCEPTED"
    assert result["rows"][0]["entry_px"] is None