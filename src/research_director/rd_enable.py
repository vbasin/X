"""Per-objective Research Director enable flags.

Only objectives with a registry ``LATEST`` *and* RD enabled are cycled by the
timer. Flags live under ``ops/webui/rd_objectives.json`` (operator-editable via
WebUI)::

    {"enabled": {"nq-50pt-econ": true, "nq-50pt-tbm": false}}

Default when a key is missing: **enabled** if registry LATEST exists, else off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.data_paths import model_registry_dir, webui_dir

logger = logging.getLogger("vtech.rd_enable")

_STATE_NAME = "rd_objectives.json"


def _state_path() -> Path:
    return webui_dir() / _STATE_NAME


def load_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"enabled": {}}
    try:
        d = json.loads(p.read_text()) or {}
        if not isinstance(d.get("enabled"), dict):
            d["enabled"] = {}
        return d
    except Exception:
        return {"enabled": {}}


def save_state(state: dict[str, Any]) -> Path:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)
    return p


def set_rd_enabled(objective_id: str, enabled: bool) -> dict[str, Any]:
    st = load_state()
    st.setdefault("enabled", {})[str(objective_id)] = bool(enabled)
    save_state(st)
    logger.info("rd_enable objective=%s enabled=%s", objective_id, enabled)
    return {"objective_id": objective_id, "rd_enabled": bool(enabled)}


def is_rd_enabled(objective_id: str, *, has_latest: bool | None = None) -> bool:
    """Return whether RD should cycle this objective.

    Explicit false in state wins. Explicit true wins. Missing key → enabled
    iff a registry LATEST exists for the objective.
    """
    st = load_state()
    en = st.get("enabled") or {}
    if objective_id in en:
        return bool(en[objective_id])
    if has_latest is not None:
        return bool(has_latest)
    from src.serving.promote import resolve_latest_bundle
    return resolve_latest_bundle(objective_id) is not None


def list_rd_targets() -> list[dict[str, Any]]:
    """Objectives to run in one timer fire: registry LATEST + RD enabled."""
    from src.serving.promote import load_latest, resolve_latest_bundle

    root = model_registry_dir()
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        oid = d.name
        if oid.startswith(".") or oid.startswith("program="):
            continue  # skip legacy program= trees at top if any
        latest = load_latest(oid)
        bundle = resolve_latest_bundle(oid)
        has = bundle is not None
        enabled = is_rd_enabled(oid, has_latest=has)
        if has and enabled:
            out.append({
                "objective_id": oid,
                "version": (latest or {}).get("version"),
                "path": str(bundle) if bundle else None,
                "rd_enabled": True,
                "has_registry_latest": True,
            })
    return out


def status_for_objective(objective_id: str) -> dict[str, Any]:
    from src.serving.promote import load_latest, resolve_latest_bundle
    bundle = resolve_latest_bundle(objective_id)
    has = bundle is not None
    latest = load_latest(objective_id)
    return {
        "objective_id": objective_id,
        "has_registry_latest": has,
        "registry_version": (latest or {}).get("version") if has else None,
        "registry_path": str(bundle) if bundle else None,
        "rd_enabled": is_rd_enabled(objective_id, has_latest=has),
    }
