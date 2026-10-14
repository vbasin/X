"""Operator-saved defaults and named hardened snapshots for external models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.data_paths import webui_dir
from src.external_models.catalog import (
    LEGACY_COMBINED_ID,
    check_hardened_name,
    check_model_id,
    coerce_book_params,
    coerce_params,
    factory_book_params,
    factory_params,
    get_model,
    partition_params,
)

_STORE_REL = "external_models"


def store_root() -> Path:
    return webui_dir() / _STORE_REL


def model_dir(model_id: str) -> Path:
    check_model_id(model_id)
    return store_root() / model_id


def defaults_path(model_id: str) -> Path:
    return model_dir(model_id) / "defaults.json"


def book_dir() -> Path:
    """Shared book overlays. Leading underscore is not a catalog id."""
    return store_root() / "_book"


def book_defaults_path() -> Path:
    return book_dir() / "defaults.json"


def hardened_dir(model_id: str) -> Path:
    return model_dir(model_id) / "hardened"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_defaults(model_id: str) -> dict[str, Any] | None:
    path = defaults_path(model_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def effective_params(model_id: str) -> dict[str, Any]:
    maybe_migrate_legacy_psar_ema20()
    saved = load_defaults(model_id)
    overlay = (saved or {}).get("params") if saved else None
    return coerce_params(model_id, overlay if isinstance(overlay, dict) else None)


def save_defaults(
    model_id: str,
    params: dict[str, Any],
    *,
    note: str | None = None,
) -> dict[str, Any]:
    check_model_id(model_id)
    get_model(model_id)  # unknown id → FileNotFoundError
    coerced = coerce_params(model_id, params)
    rec = {
        "model_id": model_id,
        "saved_at": _now(),
        "note": (note or "").strip() or None,
        "params": coerced,
    }
    _atomic_write(defaults_path(model_id), rec)
    return rec


def reset_defaults(model_id: str) -> dict[str, Any]:
    path = defaults_path(model_id)
    if path.is_file():
        path.unlink()
    return {
        "model_id": model_id,
        "saved_at": None,
        "note": None,
        "params": factory_params(model_id),
        "reset": True,
    }


def list_hardened(model_id: str) -> list[dict[str, Any]]:
    d = hardened_dir(model_id)
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(d.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        name = rec.get("name") or path.stem
        out.append({
            "name": name,
            "saved_at": rec.get("saved_at"),
            "note": rec.get("note"),
            "params": rec.get("params") or {},
            "source": str(path),
        })
    out.sort(key=lambda r: str(r.get("saved_at") or ""), reverse=True)
    return out


def load_hardened(model_id: str, name: str) -> dict[str, Any]:
    check_hardened_name(name)
    path = hardened_dir(model_id) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"hardened snapshot not found: {model_id}/{name}")
    rec = json.loads(path.read_text())
    if not isinstance(rec, dict):
        raise ValueError("hardened snapshot is not an object")
    rec["params"] = coerce_params(model_id, rec.get("params") if isinstance(rec.get("params"), dict) else None)
    rec["name"] = name
    rec["source"] = str(path)
    return rec


def save_hardened(
    model_id: str,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    note: str | None = None,
    set_as_defaults: bool = False,
) -> dict[str, Any]:
    check_model_id(model_id)
    check_hardened_name(name)
    get_model(model_id)
    coerced = coerce_params(model_id, params)
    rec = {
        "model_id": model_id,
        "name": name,
        "saved_at": _now(),
        "note": (note or "").strip() or None,
        "params": coerced,
    }
    _atomic_write(hardened_dir(model_id) / f"{name}.json", rec)
    if set_as_defaults:
        save_defaults(model_id, coerced, note=f"from hardened:{name}")
        rec["set_as_defaults"] = True
    rec["source"] = str(hardened_dir(model_id) / f"{name}.json")
    return rec


def delete_hardened(model_id: str, name: str) -> None:
    check_hardened_name(name)
    path = hardened_dir(model_id) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"hardened snapshot not found: {model_id}/{name}")
    path.unlink()


def load_book_defaults() -> dict[str, Any] | None:
    path = book_defaults_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def effective_book_params() -> dict[str, Any]:
    maybe_migrate_legacy_psar_ema20()
    saved = load_book_defaults()
    overlay = (saved or {}).get("params") if saved else None
    return coerce_book_params(overlay if isinstance(overlay, dict) else None)


def save_book_defaults(
    params: dict[str, Any],
    *,
    note: str | None = None,
) -> dict[str, Any]:
    coerced = coerce_book_params(params)
    rec = {
        "model_id": "_book",
        "saved_at": _now(),
        "note": (note or "").strip() or None,
        "params": coerced,
    }
    _atomic_write(book_defaults_path(), rec)
    return rec


def reset_book_defaults() -> dict[str, Any]:
    path = book_defaults_path()
    if path.is_file():
        path.unlink()
    return {
        "model_id": "_book",
        "saved_at": None,
        "note": None,
        "params": factory_book_params(),
        "reset": True,
    }


def maybe_migrate_legacy_psar_ema20() -> bool:
    """Split a pre-ADR-062 ``psar_ema20`` overlay into psar / ema20 / book.

    Runs once: if the combined defaults file exists and none of the new
    overlays do, write the partitioned copies. The old file is left in
    place (never deleted).
    """
    legacy_path = store_root() / LEGACY_COMBINED_ID / "defaults.json"
    if not legacy_path.is_file():
        return False
    if (
        defaults_path("psar").is_file()
        or defaults_path("ema20").is_file()
        or book_defaults_path().is_file()
    ):
        return False
    rec = load_defaults(LEGACY_COMBINED_ID)
    if not rec or not isinstance(rec.get("params"), dict):
        return False
    per_model, book = partition_params(rec["params"])
    note = "migrated from psar_ema20"
    if per_model.get("psar"):
        save_defaults("psar", per_model["psar"], note=note)
    if per_model.get("ema20"):
        save_defaults("ema20", per_model["ema20"], note=note)
    if book:
        save_book_defaults(book, note=note)
    return True
