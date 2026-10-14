"""Signals → risk checks → atomic NinjaTrader command outbox."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.execution.config import ExecutionConfig, load_execution_config
from src.execution.ledger import broker_blotter
from src.execution.protocol import (
    SCHEMA_VERSION,
    append_jsonl,
    atomic_write_json,
    make_command_id,
    make_signal_id,
    parse_utc,
    read_json,
    utc_now,
)

logger = logging.getLogger("vtech.execution.gateway")


class ExecutionGateway:
    """Single authoritative producer of executable broker commands."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.config.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.config.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        try:
            state = read_json(self.config.state_path)
        except (OSError, ValueError):
            state = {}
        return {
            "signal_inode": state.get("signal_inode"),
            "signal_offset": int(state.get("signal_offset") or 0),
            "processed_signal_ids": list(state.get("processed_signal_ids") or [])[-10000:],
            "updated_at": state.get("updated_at"),
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = utc_now().isoformat()
        self.state["processed_signal_ids"] = self.state["processed_signal_ids"][-10000:]
        atomic_write_json(self.config.state_path, self.state)

    def _read_new_signals(self) -> list[dict[str, Any]]:
        path = self.config.signal_path
        if not path.exists():
            return []
        stat = path.stat()
        inode = int(stat.st_ino)
        offset = int(self.state["signal_offset"])
        if self.state["signal_inode"] != inode or offset > stat.st_size:
            offset = 0
        rows: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(start)
                    break
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except json.JSONDecodeError:
                    self._decision(None, "REJECTED", "malformed_signal_json")
            self.state["signal_offset"] = handle.tell()
        self.state["signal_inode"] = inode
        return rows

    def _decision(
        self,
        signal: dict[str, Any] | None,
        decision: str,
        reason: str,
        *,
        command_id: str | None = None,
    ) -> None:
        append_jsonl(self.config.decision_path, {
            "decided_at": utc_now().isoformat(),
            "decision": decision,
            "reason": reason,
            "signal_id": make_signal_id(signal) if signal else None,
            "command_id": command_id,
            "objective_id": signal.get("objective_id") if signal else None,
            "signal_ts": signal.get("ts") if signal else None,
        })

    def _heartbeat_age_s(self, now: datetime) -> float | None:
        try:
            heartbeat = read_json(self.config.heartbeat_path)
            return (now - parse_utc(heartbeat["heartbeat_at"])).total_seconds()
        except (OSError, ValueError, KeyError):
            return None

    def _pending_count(self, objective_id: str, session_date: str) -> int:
        blotter = broker_blotter(
            objective_id,
            session_date=session_date,
            bridge_dir=self.config.bridge_dir,
        )
        known = {str(row["command_id"]): row["state"] for row in blotter["rows"]}
        count = 0
        for path in self.config.outbox_dir.glob("*.json"):
            try:
                command = read_json(path)
            except (OSError, ValueError):
                continue
            if command.get("objective_id") != objective_id or str(command.get("session_date")) != session_date:
                continue
            if known.get(str(command.get("command_id"))) not in {"REJECTED", "CLOSED"}:
                count += 1
        return count

    def _session_command_count(self, objective_id: str, session_date: str) -> int:
        count = 0
        for path in self.config.outbox_dir.glob("*.json"):
            try:
                command = read_json(path)
            except (OSError, ValueError):
                continue
            if command.get("objective_id") == objective_id and str(command.get("session_date")) == session_date:
                count += 1
        return count

    def _risk_reason(self, signal: dict[str, Any], now: datetime) -> str | None:
        cfg = self.config
        if not cfg.enabled:
            return "gateway_disabled"
        if cfg.kill_switch_path.exists():
            return "kill_switch_active"
        objective_id = str(signal.get("objective_id") or "")
        if objective_id not in cfg.objectives:
            return "objective_not_allowed"
        if signal.get("dir") not in {"up", "down"}:
            return "non_directional_signal"
        try:
            signal_at = parse_utc(signal.get("ts"))
        except ValueError:
            return "invalid_signal_timestamp"
        age = (now - signal_at).total_seconds()
        if age < -5 or age > cfg.signal_ttl_s:
            return "stale_signal"
        heartbeat_age = self._heartbeat_age_s(now)
        if heartbeat_age is None or heartbeat_age < -5 or heartbeat_age > cfg.heartbeat_stale_s:
            return "stale_or_missing_vm_heartbeat"
        if not signal.get("session_date"):
            return "missing_session_date"
        try:
            from src.webui.live import objective_trade_params
            from src.webui.regmodel import effective_settings
            trade = objective_trade_params(objective_id)
            effective = effective_settings(objective_id, trade=trade)
        except Exception as exc:  # noqa: BLE001
            logger.warning("execution_settings_unavailable oid=%s err=%s", objective_id, exc)
            return "execution_settings_unavailable"
        if str(effective.get("execution")) != "OCO":
            return "execution_style_not_oco"
        try:
            signal_local = signal_at.astimezone(ZoneInfo(str(trade["timezone"])))
            start_h, start_m = (int(v) for v in str(effective["trading_start"]).split(":")[:2])
            end_h, end_m = (int(v) for v in str(effective["trading_end"]).split(":")[:2])
            signal_minute = signal_local.hour * 60 + signal_local.minute
            if (
                signal_local.date().isoformat() != str(signal["session_date"])
                or signal_local.weekday() >= 5
                or not (start_h * 60 + start_m <= signal_minute <= end_h * 60 + end_m)
            ):
                return "outside_execution_session"
            from src.common.cme_holidays import is_trading_day
            if not is_trading_day(signal_local.date()):
                return "outside_execution_session"
        except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
            return "invalid_execution_session"
        confidence = float(signal.get("conf") or 0.0)
        threshold = max(float(signal.get("tau") or 0.0), float(effective.get("tau") or 0.0))
        if confidence < threshold:
            return "confidence_below_effective_tau"
        session_date = str(signal["session_date"])
        blotter = broker_blotter(
            objective_id,
            session_date=session_date,
            point_value=float(signal.get("point_value") or 20.0),
            bridge_dir=cfg.bridge_dir,
        )
        if float(blotter["stats"]["net_pnl_usd"]) <= -cfg.max_daily_loss_usd:
            return "daily_loss_limit_reached"
        if self._session_command_count(objective_id, session_date) >= cfg.max_orders_per_session:
            return "session_order_limit_reached"
        max_positions = min(cfg.max_positions, int(effective.get("max_positions") or 1))
        if self._pending_count(objective_id, session_date) >= max_positions:
            return "position_or_order_limit_reached"
        return None

    def _command(self, signal: dict[str, Any], now: datetime) -> dict[str, Any]:
        signal_id = make_signal_id(signal)
        command_id = make_command_id(signal_id)
        entry_type = self.config.entry_order_type
        return {
            "schema_version": SCHEMA_VERSION,
            "command_id": command_id,
            "signal_id": signal_id,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.config.command_ttl_s)).isoformat(),
            "mode": self.config.mode,
            "account": self.config.account,
            "objective_id": signal["objective_id"],
            "session_date": signal["session_date"],
            "instrument": self.config.ninja_instrument,
            "source_instrument": signal.get("instrument") or "NQ",
            "action": "BUY" if signal["dir"] == "up" else "SELL",
            "quantity": self.config.quantity,
            "entry_order_type": entry_type,
            "limit_price": float(signal["entry_px"]) if entry_type == "LIMIT" else None,
            "time_in_force": "DAY",
            "execution_style": "OCO",
            "target_points": float(signal["tgt_pts"]),
            "stop_points": float(signal["stop_pts"]),
            "max_holding_seconds": int(signal.get("horizon_s") or 600),
            "point_value": float(signal.get("point_value") or 20.0),
            "confidence": float(signal["conf"]),
            "signal_at": signal["ts"],
            "candidate_id": signal.get("candidate_id"),
            "model": signal.get("model"),
            "native_oco_required": True,
        }

    def run_once(self, *, now: datetime | None = None) -> dict[str, int]:
        current = (now or utc_now()).astimezone(timezone.utc)
        processed_order = [str(v) for v in self.state["processed_signal_ids"]]
        processed = set(processed_order)
        counts = {"seen": 0, "emitted": 0, "rejected": 0, "duplicate": 0}
        for signal in self._read_new_signals():
            counts["seen"] += 1
            signal_id = make_signal_id(signal)
            if signal_id in processed:
                counts["duplicate"] += 1
                continue
            processed.add(signal_id)
            processed_order.append(signal_id)
            reason = self._risk_reason(signal, current)
            if reason:
                counts["rejected"] += 1
                self._decision(signal, "REJECTED", reason)
                continue
            try:
                command = self._command(signal, current)
            except (KeyError, TypeError, ValueError):
                counts["rejected"] += 1
                self._decision(signal, "REJECTED", "invalid_signal_fields")
                continue
            path = self.config.outbox_dir / f"{command['command_id']}.json"
            if not path.exists():
                atomic_write_json(path, command)
                counts["emitted"] += 1
            self._decision(signal, "EMITTED", "risk_checks_passed", command_id=command["command_id"])
        self.state["processed_signal_ids"] = processed_order[-10000:]
        self._save_state()
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="VTech risk-aware NinjaTrader execution gateway")
    parser.add_argument("--config", default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    gateway = ExecutionGateway(load_execution_config(args.config))
    if args.once:
        print(json.dumps(gateway.run_once(), sort_keys=True))
        return
    while True:
        try:
            result = gateway.run_once()
            if result["seen"]:
                logger.info("execution_gateway_cycle %s", result)
        except Exception:  # noqa: BLE001
            logger.exception("execution_gateway_cycle_failed")
        time.sleep(gateway.config.poll_s)


if __name__ == "__main__":
    main()