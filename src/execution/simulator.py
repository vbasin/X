"""Offline NinjaTrader protocol simulator for deployment smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.execution.protocol import (
    SCHEMA_VERSION,
    atomic_write_json,
    make_event_id,
    read_json,
    utc_now,
)


def acknowledge_commands(bridge_dir: Path, *, reject: bool = False) -> int:
    """Write deterministic broker acknowledgements without submitting any order."""
    outbox, inbox = bridge_dir / "outbox", bridge_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    count = 0
    acknowledged: set[tuple[str, str]] = set()
    for event_path in inbox.glob("*.json"):
        try:
            event = read_json(event_path)
            acknowledged.add((str(event.get("command_id")), str(event.get("event_type"))))
        except (OSError, ValueError):
            continue
    for path in sorted(outbox.glob("*.json")):
        command = read_json(path)
        event_type = "ORDER_REJECTED" if reject else "ORDER_ACCEPTED"
        if (str(command["command_id"]), event_type) in acknowledged:
            continue
        event_at = utc_now().isoformat()
        event_id = make_event_id(str(command["command_id"]), event_type, event_at, "simulator")
        target = inbox / f"{event_id}.json"
        if target.exists():
            continue
        atomic_write_json(target, {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "event_at": event_at,
            "command_id": command["command_id"],
            "signal_id": command.get("signal_id"),
            "objective_id": command.get("objective_id"),
            "session_date": command.get("session_date"),
            "account": command.get("account"),
            "instrument": command.get("instrument"),
            "action": command.get("action"),
            "reason": "protocol_simulator" if reject else None,
        })
        acknowledged.add((str(command["command_id"]), event_type))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Acknowledge execution commands without a broker")
    parser.add_argument("bridge_dir", type=Path)
    parser.add_argument("--reject", action="store_true")
    args = parser.parse_args()
    print(f"events_written={acknowledge_commands(args.bridge_dir, reject=args.reject)}")


if __name__ == "__main__":
    main()