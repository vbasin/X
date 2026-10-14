"""Replay immutable NinjaTrader events into an authoritative broker blotter."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.data_paths import execution_dir
from src.execution.protocol import read_json, validate_broker_event


def read_broker_events(bridge_dir: Path | None = None) -> list[dict[str, Any]]:
    inbox = (bridge_dir or execution_dir() / "bridge") / "inbox"
    if not inbox.exists():
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(inbox.glob("*.json")):
        try:
            event = validate_broker_event(read_json(path))
        except (OSError, ValueError):
            continue
        event_id = str(event["event_id"])
        if event_id not in seen:
            seen.add(event_id)
            out.append(event)
    return sorted(out, key=lambda event: str(event["event_at"]))


def _commands(bridge_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((bridge_dir / "outbox").glob("*.json")):
        try:
            row = read_json(path)
            out[str(row["command_id"])] = row
        except (OSError, ValueError, KeyError):
            continue
    return out


def broker_blotter(
    objective_id: str,
    *,
    session_date: str | None = None,
    point_value: float = 20.0,
    bridge_dir: Path | None = None,
) -> dict[str, Any]:
    """Derive orders, fills, positions, and P/L only from broker-returned events."""
    bridge = bridge_dir or execution_dir() / "bridge"
    commands = _commands(bridge)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in read_broker_events(bridge):
        command = commands.get(str(event["command_id"]), {})
        oid = event.get("objective_id") or command.get("objective_id")
        day = event.get("session_date") or command.get("session_date")
        if oid != objective_id or (session_date is not None and str(day) != str(session_date)):
            continue
        grouped[str(event["command_id"])].append(event)

    rows: list[dict[str, Any]] = []
    net_usd = 0.0
    for command_id, events in grouped.items():
        command = commands.get(command_id, {})
        entry = [e for e in events if e.get("role") == "ENTRY" and e.get("event_type") in {"PARTIAL_FILL", "FILLED"}]
        exits = [e for e in events if e.get("role") in {"TARGET", "STOP", "FLATTEN"} and e.get("event_type") in {"PARTIAL_FILL", "FILLED"}]

        def weighted(fills: list[dict[str, Any]]) -> tuple[int, float | None]:
            qty = sum(max(0, int(e.get("fill_qty") or 0)) for e in fills)
            if not qty:
                return 0, None
            notional = sum(float(e["fill_price"]) * int(e.get("fill_qty") or 0) for e in fills)
            return qty, notional / qty

        entry_qty, entry_px = weighted(entry)
        exit_qty, exit_px = weighted(exits)
        closed = entry_qty > 0 and exit_qty >= entry_qty and exit_px is not None
        action = str(command.get("action") or events[0].get("action") or "")
        sign = 1.0 if action == "BUY" else -1.0
        pnl_points = (exit_px - entry_px) * sign * entry_qty if closed and entry_px is not None else None
        pnl_usd = pnl_points * float(command.get("point_value") or point_value) if pnl_points is not None else None
        if pnl_usd is not None:
            net_usd += pnl_usd
        rejected = next((e for e in reversed(events) if e["event_type"] == "ORDER_REJECTED"), None)
        last = events[-1]
        state = (
            "RISK_ERROR" if rejected and entry_qty
            else "REJECTED" if rejected
            else "CLOSED" if closed
            else "OPEN" if entry_qty
            else str(last["event_type"])
        )
        rows.append({
            "command_id": command_id,
            "signal_id": command.get("signal_id"),
            "objective_id": command.get("objective_id") or last.get("objective_id"),
            "session_date": command.get("session_date") or last.get("session_date"),
            "instrument": command.get("instrument") or last.get("instrument"),
            "account": command.get("account") or last.get("account"),
            "mode": command.get("mode"),
            "side": "long" if action == "BUY" else "short" if action == "SELL" else None,
            "quantity": int(command.get("quantity") or entry_qty or 0),
            "state": state,
            "entry_ts": entry[0]["event_at"] if entry else None,
            "entry_px": entry_px,
            "exit_ts": exits[-1]["event_at"] if exits else None,
            "exit_px": exit_px,
            "exit_reason": exits[-1].get("role", "").lower() if exits else None,
            "pnl_points": pnl_points,
            "pnl_usd": pnl_usd,
            "last_event_at": last["event_at"],
            "last_event_type": last["event_type"],
            "reason": rejected.get("reason") if rejected else None,
        })
    rows.sort(key=lambda row: str(row["last_event_at"]), reverse=True)
    latest_heartbeat = next((e for e in reversed(read_broker_events(bridge)) if e["event_type"] == "BRIDGE_HEARTBEAT"), None)
    return {
        "rows": rows,
        "stats": {
            "n_orders": len(rows),
            "n_open": sum(row["state"] == "OPEN" for row in rows),
            "n_closed": sum(row["state"] == "CLOSED" for row in rows),
            "n_rejected": sum(row["state"] in {"REJECTED", "RISK_ERROR"} for row in rows),
            "net_pnl_usd": round(net_usd, 2),
        },
        "meta": {
            "source": "NinjaTrader inbox events",
            "session_date": session_date,
            "bridge_heartbeat_at": latest_heartbeat.get("event_at") if latest_heartbeat else None,
            "note": None if rows else "No NinjaTrader execution events recorded for this session day.",
        },
    }