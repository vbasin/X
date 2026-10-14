"""Persist paper ledger under ``ops/paper/<objective_id>/``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.data_paths import paper_dir
from src.paper.engine import PaperEngine

logger = logging.getLogger("vtech.paper.ledger")


def paper_root(objective_id: str) -> Path:
    p = paper_dir() / objective_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_engine(engine: PaperEngine, *, root: Path | None = None) -> Path:
    """Write summary, trades.jsonl, events tail, open state."""
    oid = engine.policy.objective_id
    base = root or paper_root(oid)
    base.mkdir(parents=True, exist_ok=True)

    ledger = engine.to_ledger()
    (base / "summary.json").write_text(json.dumps(ledger["summary"], indent=2, default=str))

    trades_path = base / "trades.jsonl"
    with trades_path.open("w") as f:
        for t in ledger["trades"]:
            f.write(json.dumps(t, default=str) + "\n")

    (base / "events.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in ledger["events"]) + ("\n" if ledger["events"] else "")
    )
    (base / "open_state.json").write_text(json.dumps({
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "position": ledger["summary"].get("position"),
        "working": ledger["working"],
        "last_tick": ledger["summary"].get("last_tick"),
    }, indent=2, default=str))

    logger.info("paper_ledger_saved path=%s n_closed=%s", base, ledger["summary"]["n_closed"])
    return base


def append_trade_line(objective_id: str, trade: dict[str, Any]) -> None:
    """Append one closed trade (live stream continuous mode)."""
    base = paper_root(objective_id)
    with (base / "trades.jsonl").open("a") as f:
        f.write(json.dumps(trade, default=str) + "\n")
