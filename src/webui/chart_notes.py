"""Persistence for Live Chart scenario notes (range + overlay snapshot).

Storage layout::

    ops/webui/chart_notes/
      <id>.json   — {id, name, text, created_at, symbol, instrument, range, chart}

Notes are created from the Live Chart select-cursor flow: the operator drags
a box (time and the pane's value scale), clicks Create Note, adds a short
text, and the console stores every visible overlay / parameter alongside
the selection. ``range.price_lo`` / ``price_hi`` / ``pane`` are optional.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.common.data_paths import webui_dir

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
_MAX_TEXT = 4000
_MAX_NAME = 240


def store_root() -> Path:
    return webui_dir() / "chart_notes"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(nid: str) -> Path:
    if not _ID_RE.match(nid or ""):
        raise ValueError("note id must be alphanumeric / _ / -, 1–80 chars")
    return store_root() / f"{nid}.json"


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def list_notes() -> list[dict[str, Any]]:
    root = store_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict) or not rec.get("id"):
            continue
        rng = rec.get("range") if isinstance(rec.get("range"), dict) else {}
        out.append({
            "id": rec.get("id"),
            "name": rec.get("name") or rec.get("id"),
            "text": rec.get("text") or "",
            "created_at": rec.get("created_at"),
            "symbol": rec.get("symbol"),
            "instrument": rec.get("instrument"),
            "range_start": rng.get("start"),
            "range_end": rng.get("end"),
            "range_label": rng.get("label"),
        })
    return sorted(out, key=lambda r: str(r.get("created_at") or ""), reverse=True)


def load_note(nid: str) -> dict[str, Any]:
    p = _path(nid)
    if not p.is_file():
        raise FileNotFoundError(f"chart note not found: {nid}")
    rec = json.loads(p.read_text())
    if not isinstance(rec, dict):
        raise ValueError("chart-note file is not an object")
    return rec


def _finite(value: Any, label: str) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(n):
        raise ValueError(f"{label} must be finite")
    return n


def _range_record(rng: dict[str, Any]) -> dict[str, Any]:
    """Time window plus the optional price box the Select tool drew."""
    out: dict[str, Any] = {
        "start": str(rng.get("start")),
        "end": str(rng.get("end")),
        "start_ct": rng.get("start_ct"),
        "end_ct": rng.get("end_ct"),
        "label": rng.get("label"),
    }
    lo, hi = rng.get("price_lo"), rng.get("price_hi")
    if lo is None and hi is None and rng.get("pane") is None:
        return out
    if lo is None or hi is None:
        raise ValueError("range.price_lo and range.price_hi are both required for a box")
    price_lo, price_hi = _finite(lo, "range.price_lo"), _finite(hi, "range.price_hi")
    if price_hi < price_lo:
        raise ValueError("range.price_hi must be ≥ range.price_lo")
    out["price_lo"] = price_lo
    out["price_hi"] = price_hi
    pane = rng.get("pane")
    if pane is None:
        pane = 0
    try:
        pane_i = int(pane)
    except (TypeError, ValueError) as exc:
        raise ValueError("range.pane must be an integer") from exc
    if pane_i < 0:
        raise ValueError("range.pane must be ≥ 0")
    out["pane"] = pane_i
    return out


def save_note(spec: dict[str, Any]) -> dict[str, Any]:
    nid = str(spec.get("id") or "").strip() or _new_id()
    if not _ID_RE.match(nid):
        raise ValueError("note id must be alphanumeric / _ / -, 1–80 chars")
    name = str(spec.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")
    if len(name) > _MAX_NAME:
        raise ValueError(f"name must be ≤ {_MAX_NAME} chars")
    text = str(spec.get("text") or "").strip()
    if len(text) > _MAX_TEXT:
        raise ValueError(f"text must be ≤ {_MAX_TEXT} chars")
    rng = spec.get("range")
    if not isinstance(rng, dict) or not rng.get("start") or not rng.get("end"):
        raise ValueError("range.start and range.end are required")
    chart = spec.get("chart")
    if chart is not None and not isinstance(chart, dict):
        raise ValueError("chart must be an object")
    rec = {
        "id": nid,
        "name": name,
        "text": text,
        "created_at": str(spec.get("created_at") or _now()),
        "symbol": str(spec.get("symbol") or "").strip() or None,
        "instrument": str(spec.get("instrument") or "/NQ").strip() or "/NQ",
        "range": _range_record(rng),
        "chart": chart or {},
    }
    root = store_root()
    root.mkdir(parents=True, exist_ok=True)
    p = _path(nid)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    tmp.replace(p)
    return rec


def delete_note(nid: str) -> None:
    p = _path(nid)
    if p.is_file():
        p.unlink()
