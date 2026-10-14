"""Execution-gateway configuration with safe, fail-closed defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.common.data_paths import data_root, execution_dir, live_dir

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO / "config" / "runtime" / "execution_gateway.yaml"


@dataclass(frozen=True)
class ExecutionConfig:
    enabled: bool
    mode: str
    account: str
    objectives: tuple[str, ...]
    ninja_instrument: str
    quantity: int
    entry_order_type: str
    max_positions: int
    max_orders_per_session: int
    max_daily_loss_usd: float
    signal_ttl_s: float
    heartbeat_stale_s: float
    command_ttl_s: float
    poll_s: float
    bridge_dir: Path
    state_dir: Path
    signal_path: Path
    heartbeat_path: Path

    @property
    def outbox_dir(self) -> Path:
        return self.bridge_dir / "outbox"

    @property
    def inbox_dir(self) -> Path:
        return self.bridge_dir / "inbox"

    @property
    def kill_switch_path(self) -> Path:
        return self.bridge_dir / "KILL_SWITCH"

    @property
    def state_path(self) -> Path:
        return self.state_dir / "gateway_state.json"

    @property
    def decision_path(self) -> Path:
        return self.state_dir / "gateway_decisions.jsonl"


def _path(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else data_root() / path


def load_execution_config(path: str | Path | None = None) -> ExecutionConfig:
    config_path = Path(path or os.environ.get("VTECH_EXECUTION_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("execution config must be a mapping")

    bridge = _path(
        os.environ.get("VTECH_EXECUTION_BRIDGE_DIR") or raw.get("bridge_dir"),
        execution_dir() / "bridge",
    )
    state = _path(raw.get("state_dir"), execution_dir() / "state")
    live = live_dir()
    cfg = ExecutionConfig(
        enabled=bool(raw.get("enabled", False)),
        mode=str(raw.get("mode", "SIM")).upper(),
        account=str(raw.get("account", "Sim101")),
        objectives=tuple(str(v) for v in raw.get("objectives", [])),
        ninja_instrument=str(raw.get("ninja_instrument", "")).strip(),
        quantity=int(raw.get("quantity", 1)),
        entry_order_type=str(raw.get("entry_order_type", "MARKET")).upper(),
        max_positions=int(raw.get("max_positions", 1)),
        max_orders_per_session=int(raw.get("max_orders_per_session", 10)),
        max_daily_loss_usd=float(raw.get("max_daily_loss_usd", 500.0)),
        signal_ttl_s=float(raw.get("signal_ttl_s", 45.0)),
        heartbeat_stale_s=float(raw.get("heartbeat_stale_s", 30.0)),
        command_ttl_s=float(raw.get("command_ttl_s", 20.0)),
        poll_s=float(raw.get("poll_s", 0.25)),
        bridge_dir=bridge,
        state_dir=state,
        signal_path=_path(raw.get("signal_path"), live / "signals.jsonl"),
        heartbeat_path=_path(raw.get("heartbeat_path"), live / "heartbeat.json"),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: ExecutionConfig) -> None:
    if cfg.mode not in {"SIM", "LIVE"}:
        raise ValueError("mode must be SIM or LIVE")
    if cfg.entry_order_type not in {"MARKET", "LIMIT"}:
        raise ValueError("entry_order_type must be MARKET or LIMIT")
    if cfg.quantity < 1 or cfg.max_positions != 1:
        raise ValueError("execution bridge v1 requires quantity >= 1 and max_positions = 1")
    if cfg.max_orders_per_session < 1 or cfg.max_daily_loss_usd <= 0:
        raise ValueError("session order and loss limits must be positive")
    if min(cfg.signal_ttl_s, cfg.heartbeat_stale_s, cfg.command_ttl_s, cfg.poll_s) <= 0:
        raise ValueError("TTL and polling values must be positive")
    if cfg.enabled and not cfg.objectives:
        raise ValueError("enabled gateway requires at least one objective allowlist entry")
    if cfg.enabled and not cfg.ninja_instrument:
        raise ValueError("enabled gateway requires exact ninja_instrument, e.g. 'NQ 09-26'")
    if cfg.mode == "LIVE" and cfg.account.lower().startswith("sim"):
        raise ValueError("LIVE mode cannot target a simulation account")