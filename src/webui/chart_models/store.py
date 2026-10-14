"""Persistence for chart-scoped user models (source JSON + defaults).

Storage layout::

    ops/webui/chart_models/
      <id>.json          — the full spec {id, name, notes, rules, updated_at}

The frontend also keeps an autosave draft in localStorage; only explicit
Save writes here.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.data_paths import webui_dir

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def store_root() -> Path:
    return webui_dir() / "chart_models"


def _path(mid: str) -> Path:
    if not _ID_RE.match(mid or ""):
        raise ValueError("model id must be lowercase alphanumeric / _ / -, 1–64 chars")
    return store_root() / f"{mid}.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_models() -> list[dict[str, Any]]:
    root = store_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict) or not rec.get("id"):
            continue
        out.append({
            "id": rec.get("id"),
            "name": rec.get("name") or rec.get("id"),
            "updated_at": rec.get("updated_at"),
            "n_rules": len(rec.get("rules") or []),
        })
    return sorted(out, key=lambda r: str(r.get("updated_at") or ""), reverse=True)


def load_model(mid: str) -> dict[str, Any]:
    p = _path(mid)
    if not p.is_file():
        raise FileNotFoundError(f"chart model not found: {mid}")
    rec = json.loads(p.read_text())
    if not isinstance(rec, dict):
        raise ValueError("chart-model file is not an object")
    return rec


def save_model(spec: dict[str, Any]) -> dict[str, Any]:
    mid = str(spec.get("id") or "").strip()
    if not _ID_RE.match(mid):
        raise ValueError("model id must be lowercase alphanumeric / _ / -, 1–64 chars")
    p = _path(mid)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": mid,
        "name": spec.get("name") or mid,
        "notes": spec.get("notes") or "",
        "rules": spec.get("rules") or [],
        "updated_at": _now(),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    tmp.replace(p)
    return rec


def delete_model(mid: str) -> None:
    p = _path(mid)
    if p.is_file():
        p.unlink()
