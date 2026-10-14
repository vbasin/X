"""Versioned file-bridge protocol shared by the gateway and broker adapter."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v1"
TERMINAL_EVENT_TYPES = {"ORDER_REJECTED", "ENTRY_CANCELLED", "TRADE_CLOSED"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: Any) -> datetime:
    if not value:
        raise ValueError("timestamp is required")
    out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def make_signal_id(signal: dict[str, Any]) -> str:
    """Stable ID for one immutable model fire, including legacy rows without an ID."""
    existing = signal.get("signal_id")
    if existing:
        return str(existing)
    identity = "|".join(str(signal.get(k) or "") for k in (
        "objective_id", "session_date", "ts", "dir", "candidate_id", "model",
    ))
    return "sig_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def make_command_id(signal_id: str) -> str:
    return "cmd_" + hashlib.sha256(f"{SCHEMA_VERSION}|{signal_id}".encode()).hexdigest()[:24]


def make_event_id(command_id: str, event_type: str, event_at: str, discriminator: str = "") -> str:
    raw = f"{SCHEMA_VERSION}|{command_id}|{event_type}|{event_at}|{discriminator}"
    return "evt_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Publish one complete JSON message; readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def validate_broker_event(event: dict[str, Any]) -> dict[str, Any]:
    required = ("event_id", "event_type", "event_at", "command_id")
    missing = [key for key in required if not event.get(key)]
    if missing:
        raise ValueError(f"broker event missing fields: {missing}")
    if event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported broker event schema: {event.get('schema_version')!r}")
    parse_utc(event["event_at"])
    return event