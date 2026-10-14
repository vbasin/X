"""WebUI Custom Test — compile a board into a manual spec + topology overlay.

The Trade Objective stays the SSOT for labels / KPI / session (ADR-018). This
module only decides *which* catalog families to fit, *which* feature units they
see, *which* hyperparameters they use (catalog defaults, per-run overrides, and
opt-in HPO — ADR-040), whether a meta gate sits on top, and (ADR-042) the
expanding calendar walk-forward window. Ablation flags expand per model: each
flagged item is included and excluded, so *k* flags on one slot yield ``2**k``
candidates (multiplicative).

Feature builder (compose a new column from existing names) is intentionally
not implemented — parked for a later increment.
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.common.config_paths import schema_path as _schema_path
from src.common.data_paths import webui_dir
from src.models.catalog import get_catalog, get_entry
from src.models.feature_views import FEATURE_VIEWS
from src.validation.walk_forward import default_walk_forward, overlay_from_board

logger = logging.getLogger("vtech.search.custom_test")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_MD = _REPO_ROOT / "docs" / "reference" / "FEATURE_CATALOG.md"
_SCHEMA_PATH = _schema_path("custom_test.schema.json")

# Latest stack (nq-50pt-xgb-tcn) — form defaults when an objective has no topology.
DEFAULT_META: dict[str, Any] = {
    "enabled": False,
    "mode": "meta_stack",
    "family": "logistic_regression",
    "scheme": "walk_forward_OOF",
    "feature": "linear_weights",
    "approve_threshold": 0.55,
    "n_folds": 5,
    "embargo_bars": 120,
    "context_view": None,
}

VIEW_INFO: dict[str, str] = {
    "regime_core": "Vol / HTF / daily context / VWAP / candle structure (~40 dims).",
    "regime_min": "Four isolated descriptors: VWAP location, slope, realized vol, vol ratio.",
    "regime_min_iv": "regime_min plus ATM IV / skew / butterfly.",
    "sequence_baseline": "Trader sequence set: VWAP, HTF, daily, candles, realized vol, time.",
    "microstructure": "Order-flow / book / location / CVD / arrival / VPIN / profile.",
    "momentum_edge": "Sparse L1 momentum (ADR-027): flow, book, VWAP, vol, HTF, time.",
    "tbm_topgroups": "Causal champion prefixes (ADR-046): cs_ / rv_ / vp_ / iv_.",
    "candle_footprint": "Closed 1m/5m wick/body volume + 1m POC/EMA (vf_, ADR-056).",
}

MAX_ABLATION_FLAGS = 6          # 2^6 = 64 combos per slot
MAX_CANDIDATES = 64             # hard cap across all slots
DEFAULT_TUNE_TRIALS = 20        # used when any slot has hpo and tune_trials is 0
MAX_TUNE_TRIALS = 200
_ITEM_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SAVED_VIEW_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SAVED_VIEWS_REL = "custom_feature_views"

_GROUP_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*`([a-z]{2,4}_)`\s*\|\s*\[([^\]]+)\][^|]*\|\s*(\d+)",
    re.MULTILINE,
)
_FEATURE_ROW = re.compile(
    r"^\|\s*`([a-z]{2,4}_[a-z0-9_]+)`\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE,
)


# ── Feature bank (views + prefix groups + catalog columns) ───────────────────

def parse_feature_catalog(text: str | None = None, *, store: str | None = None) -> dict[str, Any]:
    """Parse a feature catalog markdown into prefix groups and columns."""
    if text is None:
        from src.features.stores import catalog_path
        p = catalog_path(store or "5s")
        raw = p.read_text() if p.is_file() else (
            _CATALOG_MD.read_text() if _CATALOG_MD.is_file() else "")
    else:
        raw = text
    groups: list[dict[str, Any]] = []
    seen_prefix: set[str] = set()
    for m in _GROUP_ROW.finditer(raw):
        prefix, name, count = m.group(1), m.group(2), int(m.group(3))
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        groups.append({"prefix": prefix, "name": name, "count": count})

    features: list[dict[str, Any]] = []
    seen_feat: set[str] = set()
    for m in _FEATURE_ROW.finditer(raw):
        col, desc = m.group(1), m.group(2).strip()
        if col in seen_feat or col.startswith("_"):
            continue
        seen_feat.add(col)
        prefix = next((g["prefix"] for g in groups if col.startswith(g["prefix"])),
                      col[:3] if len(col) >= 3 and col[2] == "_" else "")
        features.append({"name": col, "description": desc, "prefix": prefix})

    by_prefix: dict[str, list[str]] = {}
    for f in features:
        by_prefix.setdefault(f["prefix"], []).append(f["name"])
    for g in groups:
        g["features"] = by_prefix.get(g["prefix"], [])
        if not g["count"]:
            g["count"] = len(g["features"])
    return {"groups": groups, "features": features}


# ── Operator-saved Custom Test views (column snapshots) ──────────────────────

def saved_views_dir() -> Path:
    """``ops/webui/custom_feature_views/`` — named column lists, not catalog prefixes."""
    return webui_dir() / _SAVED_VIEWS_REL


def check_saved_view_name(name: str) -> str:
    n = str(name or "").strip().lower().replace(" ", "_")
    if not _SAVED_VIEW_NAME_RE.match(n):
        raise ValueError(
            "view name must start with a letter and use lowercase letters, "
            "digits, '_' or '-' (max 64)"
        )
    return n


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _columns_for_prefixes(prefixes: list[str], feat_names: list[str]) -> list[str]:
    return [c for c in feat_names if any(c.startswith(p) for p in prefixes)]


def list_saved_views() -> list[dict[str, Any]]:
    root = saved_views_dir()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        try:
            name = check_saved_view_name(str(rec.get("name") or path.stem))
        except ValueError:
            continue
        if name in FEATURE_VIEWS:
            continue
        cols: list[str] = []
        seen: set[str] = set()
        for raw in rec.get("features") or []:
            col = str(raw).strip()
            if not col or col in seen:
                continue
            seen.add(col)
            cols.append(col)
        out.append({
            "name": name,
            "source": "saved",
            "prefixes": [],
            "features": cols,
            "n_prefixes": 0,
            "n_features": len(cols),
            "description": str(rec.get("description") or "").strip(),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        })
    return out


def save_custom_view(
    name: str,
    features: list[str],
    *,
    description: str = "",
    overwrite: bool = False,
    catalog_features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Persist a named column snapshot. Catalog ``FEATURE_VIEWS`` names are reserved."""
    name = check_saved_view_name(name)
    if name in FEATURE_VIEWS:
        raise ValueError(f"'{name}' is a catalog view — pick another name")
    known = {
        f["name"] for f in (
            catalog_features if catalog_features is not None
            else parse_feature_catalog()["features"]
        )
    }
    cols: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for raw in features or []:
        col = str(raw).strip()
        if not col or col in seen:
            continue
        if col not in known:
            unknown.append(col)
            continue
        seen.add(col)
        cols.append(col)
    if unknown:
        raise ValueError(f"unknown feature(s): {unknown[:8]}")
    if not cols:
        raise ValueError("save as view needs at least one catalog feature")
    path = saved_views_dir() / f"{name}.json"
    prev: dict[str, Any] = {}
    if path.is_file():
        if not overwrite:
            raise FileExistsError(
                f"saved view '{name}' already exists — set overwrite to replace"
            )
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            prev = loaded
    now = _now_utc()
    rec = {
        "name": name,
        "source": "saved",
        "features": cols,
        "n_features": len(cols),
        "description": str(description or prev.get("description") or "").strip()[:240],
        "created_at": prev.get("created_at") or now,
        "updated_at": now,
    }
    _atomic_json(path, rec)
    return rec


def delete_custom_view(name: str) -> None:
    name = check_saved_view_name(name)
    if name in FEATURE_VIEWS:
        raise ValueError("cannot delete a catalog view")
    path = saved_views_dir() / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"saved view '{name}' not found")
    path.unlink()


def list_views(
    *,
    catalog_features: list[dict[str, Any]] | None = None,
    saved: list[dict[str, Any]] | None = None,
    store: str | None = None,
) -> list[dict[str, Any]]:
    feat_names = [
        f["name"] for f in (
            catalog_features if catalog_features is not None
            else parse_feature_catalog()["features"]
        )
    ]
    out: list[dict[str, Any]] = []
    for rec in (saved if saved is not None else list_saved_views()):
        out.append({
            "name": rec["name"],
            "source": "saved",
            "prefixes": [],
            "features": list(rec.get("features") or []),
            "n_prefixes": 0,
            "n_features": int(rec.get("n_features") or len(rec.get("features") or [])),
            "description": str(rec.get("description") or ""),
            "created_at": rec.get("created_at"),
            "updated_at": rec.get("updated_at"),
        })
    from src.features.stores import STORE_VIEW_INFO, views_for_store

    view_map = views_for_store(store or "5s")
    for name, prefixes in view_map.items():
        cols = _columns_for_prefixes(list(prefixes), feat_names)
        if not cols and store not in (None, "5s"):
            continue
        out.append({
            "name": name,
            "source": "catalog",
            "prefixes": list(prefixes),
            "features": cols,
            "n_prefixes": len(prefixes),
            "n_features": len(cols),
            "description": STORE_VIEW_INFO.get(name) or VIEW_INFO.get(name, ""),
        })
    return out


def expand_items_to_features(
    items: list[dict[str, Any]],
    *,
    bank: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten view/group chips to per-column feature items (first-seen order).

    Catalog views expand by prefix against the current catalog. Saved views
    expand to their stored column list. Ablation on a view/group is dropped
    (flags stay per-column only).
    """
    bank = bank or feature_bank()
    known = {f["name"] for f in bank.get("features") or []}
    views_index = {v["name"]: v for v in bank.get("views") or []}
    groups_index = {g["prefix"]: g for g in bank.get("groups") or []}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in items or []:
        kind, key = it.get("kind"), str(it.get("key") or "")
        cols: list[str] = []
        ablation = False
        if kind == "feature":
            cols = [key]
            ablation = bool(it.get("ablation"))
        elif kind == "group":
            g = groups_index.get(key) or {}
            cols = list(g.get("features") or [])
            if not cols:
                cols = [f["name"] for f in bank.get("features") or []
                        if str(f.get("name") or "").startswith(key)]
        elif kind == "view":
            v = views_index.get(key) or {}
            cols = list(v.get("features") or [])
            if not cols:
                cols = _columns_for_prefixes(list(v.get("prefixes") or []),
                                             [f["name"] for f in bank.get("features") or []])
        for col in cols:
            if not col or col in seen:
                continue
            if known and col not in known:
                continue
            seen.add(col)
            out.append({
                "kind": "feature",
                "key": col,
                "label": col,
                "ablation": ablation,
            })
    return out


def list_models() -> list[dict[str, Any]]:
    return [{
        "name": e.name,
        "display_name": e.display_name,
        "family_group": e.family_group,
        "compute_class": e.compute_class,
        "cost_class": e.cost_class,
        "needs_sequence": e.needs_sequence,
        "needs_labels": e.needs_labels,
        "feature_view": e.feature_view,
        "notes": e.notes,
        # n_features is the assigned column count — not an operator knob.
        "params_default": {k: v for k, v in dict(e.params_default or {}).items()
                           if k != "n_features"},
        "search_space": dict(e.search_space or {}),
        "hpo_available": bool(e.search_space),
        "params_help": dict(e.params_help or {}),
        "search_max_rows": e.search_max_rows,
    } for e in get_catalog()]


def feature_bank(store: str | None = None) -> dict[str, Any]:
    from src.features.stores import list_feature_stores, require_store

    sid = require_store(store or "5s")
    parsed = parse_feature_catalog(store=sid)
    saved = list_saved_views()
    known = {f["name"] for f in parsed["features"]}
    saved_ok = []
    for rec in saved:
        cols = [c for c in (rec.get("features") or []) if c in known]
        if cols:
            saved_ok.append({**rec, "features": cols, "n_features": len(cols)})
    return {
        "store": sid,
        "stores": list_feature_stores(),
        "models": list_models(),
        "views": list_views(
            catalog_features=parsed["features"], saved=saved_ok, store=sid,
        ),
        "groups": parsed["groups"],
        "features": parsed["features"],
        "feature_builder": {
            "available": False,
            "status": "parked",
            "note": (
                "Feature builder is tabled. Compose-from-existing-names (and stacking "
                "a new feature into a later feature) is not in this increment. "
                "Request new catalog features to have them added."
            ),
        },
    }


# ── Default board from an objective ──────────────────────────────────────────

def default_board(objective: dict[str, Any]) -> dict[str, Any]:
    """Pre-fill models / views / meta from the objective's search block.

    ``nq-50pt-xgb-tcn`` (latest stack) yields xgboost+``regime_core``,
    tcn+``sequence_baseline``, and an enabled ``meta_stack`` logistic gate.
    """
    from src.features.stores import default_head, feature_store_id, views_for_store

    oid = str(objective.get("objective_id") or "")
    search = objective.get("search") or {}
    views = search.get("feature_views") or {}
    default_view = search.get("default_feature_view")
    topo = search.get("topology") or {}
    mode = str(topo.get("mode") or "none")
    store = feature_store_id(objective)
    head = default_head(objective)
    store_views = views_for_store(store)
    bases: list[str] = []
    for b in (topo.get("bases") or []):
        if isinstance(b, dict):
            fam = str(b.get("family") or "").strip()
        else:
            fam = str(b).strip()
        if fam and fam not in bases:
            bases.append(fam)
    eligible = [str(f) for f in (objective.get("eligible_model_families") or [])
                if str(f).strip()]

    if mode in ("meta_stack", "output_meta", "tf_oof_meta") and bases:
        families = bases
    elif eligible:
        families = eligible
    else:
        families = []

    known = {e.name for e in get_catalog()}
    models: list[dict[str, Any]] = []
    for fam in families:
        if fam not in known:
            continue
        view = views.get(fam) or default_view or get_entry(fam).feature_view
        items: list[dict[str, Any]] = []
        if view and (store == "5s" or str(view) in store_views):
            items.append({
                "kind": "view", "key": str(view),
                "label": str(view), "ablation": False,
            })
        models.append({"id": fam, "family": fam, "items": items, "head": head})

    meta = {**DEFAULT_META}
    if mode in ("meta_stack", "output_meta", "primary_meta"):
        raw = topo.get("meta") or {}
        meta.update({
            "enabled": True,
            "mode": mode,
            "family": str(raw.get("family") or meta["family"]),
            "scheme": str(raw.get("scheme") or meta["scheme"]),
            "feature": str(raw.get("feature") or meta["feature"]),
            "approve_threshold": float(raw.get("approve_threshold", meta["approve_threshold"])),
            "n_folds": int(raw.get("n_folds", meta["n_folds"])),
            "embargo_bars": int(raw.get("embargo_bars", meta["embargo_bars"])),
            "context_view": raw.get("context_view"),
        })
    return {
        "schema_version": "v1",
        "objective_id": oid,
        "feature_store": store,
        "head": head,
        "models": models,
        "meta": meta,
        "walk_forward": default_walk_forward(objective),
        "dry_run": False,
    }


# ── Validation + compile ─────────────────────────────────────────────────────

@dataclass
class CompiledTest:
    board: dict[str, Any]
    manual_spec: dict[str, Any]
    topology: dict[str, Any]
    candidates: list[dict[str, Any]]
    n_combos: int
    warnings: list[str] = field(default_factory=list)
    tune_trials: int = 0
    walk_forward: dict[str, Any] | None = None


def _validate_board(board: dict[str, Any]) -> None:
    import jsonschema
    schema = json.loads(_SCHEMA_PATH.read_text())
    # Pydantic model_dump() emits null for omitted optional objects.
    if board.get("meta") is None:
        board.pop("meta", None)
    if board.get("walk_forward") is None:
        board.pop("walk_forward", None)
    if board.get("feature_store") is None:
        board.pop("feature_store", None)
    if board.get("head") is None:
        board.pop("head", None)
    try:
        jsonschema.Draft202012Validator(schema).validate(board)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"custom-test board invalid: {exc.message}") from exc


def _known_keys(bank: dict[str, Any] | None = None) -> tuple[set[str], set[str], set[str]]:
    bank = bank or feature_bank()
    views = {v["name"] for v in bank["views"]}
    groups = {g["prefix"] for g in bank["groups"]}
    feats = {f["name"] for f in bank["features"]}
    return views, groups, feats


def _check_item(item: dict[str, Any], views: set[str], groups: set[str],
                feats: set[str]) -> None:
    kind, key = item.get("kind"), item.get("key")
    if not kind or not key:
        raise ValueError("each feature item needs kind + key")
    if not _ITEM_KEY_RE.match(str(key)) and kind != "feature":
        # features can be long catalog names; still reject path-y junk
        if "/" in str(key) or ".." in str(key):
            raise ValueError(f"invalid feature key: {key!r}")
    if kind == "view" and key not in views:
        raise ValueError(f"unknown feature view '{key}'; known: {sorted(views)}")
    if kind == "group" and key not in groups:
        raise ValueError(f"unknown feature group '{key}'; known: {sorted(groups)}")
    if kind == "feature" and feats and key not in feats:
        raise ValueError(f"unknown feature '{key}'")


def _slug(item: dict[str, Any]) -> str:
    key = str(item["key"]).rstrip("_")
    if item["kind"] == "view":
        return key
    if item["kind"] == "group":
        return f"g_{key}"
    return key


def _prefixes_for_view(name: str, store: str | None = None) -> list[str]:
    from src.features.stores import views_for_store
    m = views_for_store(store or "5s")
    if name in m:
        return list(m[name])
    if name in FEATURE_VIEWS:
        return list(FEATURE_VIEWS[name])
    return []


def _coerce_param_value(key: str, value: Any, default: Any, space: dict[str, Any] | None) -> Any:
    """Coerce a board param to the catalog / search-space type."""
    if value is None:
        return None
    t = (space or {}).get("type")
    if t == "int":
        return int(value)
    if t == "float":
        return float(value)
    if t == "categorical":
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(default, list) and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed
    return value


def coerce_slot_params(family: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Type-coerce a slot's param overrides using catalog defaults + search_space."""
    if not raw:
        return {}
    entry = get_entry(family)
    defaults = entry.params_default or {}
    space = entry.search_space or {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        out[name] = _coerce_param_value(name, value, defaults.get(name), space.get(name))
    return out


def slot_search_space(slot: dict[str, Any]) -> dict[str, Any]:
    """Catalog search_space subset for a slot, or ``{}`` when HPO is off.

    ``hpo: true`` with empty/omitted ``hpo_params`` includes every catalog knob.
    Unknown ``hpo_params`` keys raise ``ValueError``.
    """
    family = slot["family"]
    catalog_ss = dict(get_entry(family).search_space or {})
    if not slot.get("hpo"):
        return {}
    if not catalog_ss:
        return {}
    selected = slot.get("hpo_params")
    if not selected:
        return catalog_ss
    if not isinstance(selected, list):
        sid = slot.get("id") or family
        raise ValueError(f"slot '{sid}' hpo_params must be a list of parameter names")
    names = [str(k) for k in selected]
    unknown = [k for k in names if k not in catalog_ss]
    if unknown:
        sid = slot.get("id") or family
        raise ValueError(
            f"slot '{sid}' hpo_params not in catalog search_space for '{family}': "
            f"{unknown}; known: {sorted(catalog_ss)}"
        )
    return {k: catalog_ss[k] for k in names}


def resolve_tune_trials(board: dict[str, Any], n_hpo_candidates: int) -> int:
    """Trials for ``--tune-trials``. 0 when no slot is HPO-enabled."""
    if n_hpo_candidates <= 0:
        return 0
    raw = board.get("tune_trials")
    n = int(raw or 0)
    if n <= 0:
        n = DEFAULT_TUNE_TRIALS
    if n > MAX_TUNE_TRIALS:
        raise ValueError(f"tune_trials {n} exceeds max {MAX_TUNE_TRIALS}")
    return n


def compile_feature_selection(
    items: list[dict[str, Any]],
    *,
    catalog_features: list[dict[str, Any]] | None = None,
    views_index: dict[str, dict[str, Any]] | None = None,
    store: str | None = None,
) -> dict[str, Any] | None:
    """Collapse a slot's (non-empty) items to a manual-spec ``features`` object."""
    if not items:
        return None
    if views_index is None:
        views_index = {v["name"]: v for v in list_views(
            catalog_features=catalog_features)}
    materialized: list[dict[str, Any]] = []
    for i in items:
        if i.get("kind") == "view":
            rec = views_index.get(i.get("key")) or {}
            if rec.get("source") == "saved":
                for col in rec.get("features") or []:
                    materialized.append({
                        **i, "kind": "feature", "key": col, "label": col,
                    })
                continue
        materialized.append(i)
    items = materialized
    views = [i for i in items if i["kind"] == "view"]
    groups = [i for i in items if i["kind"] == "group"]
    cols = [i for i in items if i["kind"] == "feature"]

    if views and not groups and not cols and len(views) == 1:
        return {"view": views[0]["key"]}

    if cols:
        names: list[str] = []
        feat_names = [f["name"] for f in (catalog_features or [])]
        for v in views:
            prefs = _prefixes_for_view(v["key"], store)
            if feat_names:
                names.extend(c for c in feat_names if any(c.startswith(p) for p in prefs))
            else:
                names.extend(prefs)
        for g in groups:
            if feat_names:
                names.extend(c for c in feat_names if c.startswith(g["key"]))
            else:
                names.append(g["key"])
        names.extend(c["key"] for c in cols)
        # preserve first-seen order
        return {"columns": list(dict.fromkeys(names))}

    prefixes: list[str] = []
    for v in views:
        prefixes.extend(_prefixes_for_view(v["key"], store))
    prefixes.extend(g["key"] for g in groups)
    return {"prefixes": list(dict.fromkeys(prefixes))}


def expand_slot(
    slot: dict[str, Any],
    *,
    catalog_features: list[dict[str, Any]] | None = None,
    views_index: dict[str, dict[str, Any]] | None = None,
    store: str | None = None,
) -> list[dict[str, Any]]:
    """Expand one model slot into candidates (``2**k`` when *k* items are flagged)."""
    family = slot["family"]
    get_entry(family)  # KeyError → unknown
    slot_id = slot.get("id") or family
    items = list(slot.get("items") or [])
    flagged = [i for i in items if i.get("ablation")]
    always = [i for i in items if not i.get("ablation")]
    if len(flagged) > MAX_ABLATION_FLAGS:
        raise ValueError(
            f"slot '{slot_id}' flags {len(flagged)} items for ablation "
            f"(max {MAX_ABLATION_FLAGS}; {1 << len(flagged)} combos)")

    n = 1 << len(flagged) if flagged else 1
    out: list[dict[str, Any]] = []
    for mask in range(n):
        chosen = list(always)
        dropped: list[str] = []
        kept_flags: list[str] = []
        for bit, item in enumerate(flagged):
            if mask & (1 << bit):
                chosen.append(item)
                kept_flags.append(_slug(item))
            else:
                dropped.append(_slug(item))
        # Emit the all-included combo first so meta_stack's by_family.setdefault
        # lands on the full variant (mask = all bits set is last; we sort later).
        if not flagged:
            cid = str(slot_id)
            tag = "full"
        elif mask == n - 1:
            cid = str(slot_id)
            tag = "full"
        else:
            off = "no_" + "+".join(dropped) if dropped else "base"
            cid = f"{slot_id}__{off}"
            tag = off
        sel = compile_feature_selection(
            chosen, catalog_features=catalog_features, views_index=views_index,
            store=store)
        cand: dict[str, Any] = {
            "id": cid,
            "family": family,
            "features": sel,
            "slot_id": slot_id,
            "ablation_tag": tag,
            "included": [_slug(i) for i in chosen],
            "excluded": dropped,
            "is_full": (not flagged) or mask == n - 1,
            "hpo": bool(slot.get("hpo")),
        }
        params = coerce_slot_params(family, slot.get("params"))
        head = str(slot.get("head") or "").strip()
        if head:
            params = {**params, "head": head}
        if params:
            cand["params"] = params
        ss = slot_search_space(slot)
        if ss:
            cand["search_space"] = ss
        if "max_rows" in slot:
            cand["max_rows"] = slot["max_rows"]
        out.append(cand)
    # Full variant first, then the omissions (stable for meta lookup).
    out.sort(key=lambda c: (0 if c["is_full"] else 1, c["id"]))
    return out


def compile_board(board: dict[str, Any], *, bank: dict[str, Any] | None = None) -> CompiledTest:
    """Validate a board and expand it to a manual spec + topology overlay."""
    board = deepcopy(board)
    _validate_board(board)
    from src.features.stores import default_head, require_store
    store = require_store(str(board.get("feature_store") or "5s"))
    board["feature_store"] = store
    bank = bank or feature_bank(store)
    views, groups, feats = _known_keys(bank)
    known_fam = {m["name"] for m in bank["models"]}
    warnings: list[str] = []

    models = board.get("models") or []
    if not models:
        raise ValueError("select at least one model")

    seen_ids: set[str] = set()
    all_cands: list[dict[str, Any]] = []
    views_index = {v["name"]: v for v in bank.get("views") or []}
    for slot in models:
        fam = slot.get("family")
        if fam not in known_fam:
            raise ValueError(f"unknown model family '{fam}'")
        sid = slot.get("id") or fam
        if sid in seen_ids:
            raise ValueError(f"duplicate model slot id '{sid}'")
        seen_ids.add(sid)
        slot["id"] = sid
        for item in slot.get("items") or []:
            _check_item(item, views, groups, feats)
        if slot.get("hpo") and not get_entry(fam).search_space:
            warnings.append(
                f"slot '{sid}' requested HPO but '{fam}' has no catalog search_space "
                "— using fixed params"
            )
        coerce_slot_params(fam, slot.get("params"))  # raise early on bad types
        slot_search_space(slot)
        if not slot.get("head"):
            try:
                from src.objectives.loader import load_objective
                slot["head"] = default_head(load_objective(str(board.get("objective_id") or "")))
            except Exception:
                slot["head"] = str(board.get("head") or "multiclass")
        all_cands.extend(expand_slot(
            slot,
            catalog_features=bank.get("features"),
            views_index=views_index,
            store=store,
        ))

    if len(all_cands) > MAX_CANDIDATES:
        raise ValueError(
            f"ablation expanded to {len(all_cands)} candidates "
            f"(max {MAX_CANDIDATES}) — unflag some items")

    # Unique candidate ids (two slots of the same family both emit '<family>' for full).
    used: set[str] = set()
    for c in all_cands:
        cid = c["id"]
        if cid in used:
            n = 2
            while f"{cid}_{n}" in used:
                n += 1
            c["id"] = f"{cid}_{n}"
        used.add(c["id"])

    spec_candidates = []
    for c in all_cands:
        row: dict[str, Any] = {"id": c["id"], "family": c["family"]}
        if c["features"]:
            row["features"] = c["features"]
        if c.get("params"):
            row["params"] = c["params"]
        if c.get("search_space"):
            row["search_space"] = c["search_space"]
        if "max_rows" in c and c["max_rows"] is not None:
            row["max_rows"] = int(c["max_rows"])
        spec_candidates.append(row)

    hpo_cands = [c for c in all_cands if c.get("search_space")]
    tune_trials = resolve_tune_trials(board, len(hpo_cands))
    board["tune_trials"] = tune_trials
    if hpo_cands:
        full_hpo = [c for c in hpo_cands if c.get("is_full")]
        if not full_hpo:
            full_hpo = hpo_cands
        warnings.append(
            "HPO: "
            + f"{tune_trials} Optuna trial(s) on "
            + ", ".join(
                f"{c['id']}[{', '.join(c['search_space'])}]" for c in full_hpo
            )
        )
        n_ab_hpo = sum(1 for c in hpo_cands if not c.get("is_full"))
        if n_ab_hpo:
            warnings.append(
                f"HPO also runs on {n_ab_hpo} ablation variant(s) — "
                f"each variant gets {tune_trials} trial(s)"
            )

    manual_spec = {
        "schema_version": "v1",
        "objective_id": board["objective_id"],
        "ensembles": False,
        "candidates": spec_candidates,
    }

    meta = {**DEFAULT_META, **(board.get("meta") or {})}
    enabled = bool(meta.get("enabled"))
    mode = str(meta.get("mode") or "meta_stack")
    if not enabled or mode in ("none", "off", ""):
        topology: dict[str, Any] = {"mode": "none", "feature_store": store}
    else:
        full_fams = []
        for slot in models:
            fam = slot["family"]
            if fam not in full_fams:
                full_fams.append(fam)
        meta_block = {
            "family": meta.get("family") or DEFAULT_META["family"],
            "scheme": meta.get("scheme") or DEFAULT_META["scheme"],
            "feature": meta.get("feature") or DEFAULT_META["feature"],
            "approve_threshold": float(meta.get("approve_threshold", 0.55)),
            "n_folds": int(meta.get("n_folds", 5)),
            "embargo_bars": int(meta.get("embargo_bars", 120)),
        }
        if meta.get("context_view"):
            meta_block["context_view"] = meta["context_view"]
        topology = {
            "mode": mode, "bases": full_fams, "meta": meta_block,
            "feature_store": store,
        }
        if mode in ("meta_stack", "output_meta") and len(full_fams) < 1:
            warnings.append("meta enabled but no bases — overlay will be skipped")
        n_full = sum(1 for c in all_cands if c.get("is_full"))
        n_ab = len(all_cands) - n_full
        if n_ab:
            warnings.append(
                "meta trains on the full (no-omission) variant of each selected "
                f"family; {n_ab} ablation variant(s) are scored independently")

    walk_overlay: dict[str, Any] | None = None
    if board.get("walk_forward") is not None:
        try:
            walk_overlay = overlay_from_board(board["walk_forward"])
        except ValueError as exc:
            raise ValueError(f"walk-forward: {exc}") from exc
        if walk_overlay:
            n_wf = int(walk_overlay.get("walk_forward_folds") or 0)
            warnings.append(
                f"walk-forward: {n_wf} expanding calendar fold(s) "
                f"{walk_overlay['date_range']['start']} → {walk_overlay['date_range']['end']}"
            )
            val = (board.get("walk_forward") or {}).get("val") or {}
            if val.get("enabled"):
                warnings.append(
                    "val is used by sequence/DL families (TCN/LSTM/…) for early stopping; "
                    "tree models (XGBoost, LightGBM, …) ignore it"
                )

    return CompiledTest(
        board=board,
        manual_spec=manual_spec,
        topology=topology,
        candidates=all_cands,
        n_combos=len(all_cands),
        warnings=warnings,
        tune_trials=tune_trials,
        walk_forward=walk_overlay,
    )


def preview_board(board: dict[str, Any]) -> dict[str, Any]:
    compiled = compile_board(board)
    return {
        "objective_id": compiled.board["objective_id"],
        "n_candidates": compiled.n_combos,
        "n_models": len(compiled.board["models"]),
        "meta": compiled.topology,
        "candidates": compiled.candidates,
        "manual_spec": compiled.manual_spec,
        "warnings": compiled.warnings,
        "tune_trials": compiled.tune_trials,
        "walk_forward": compiled.walk_forward,
    }


def persist_compiled(
    compiled: CompiledTest,
    dest_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    """Write board + compiled spec + topology / walk-forward overlays."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    board_p = dest_dir / f"{run_id}.board.yaml"
    spec_p = dest_dir / f"{run_id}.manual.yaml"
    topo_p = dest_dir / f"{run_id}.topology.yaml"
    board_p.write_text(yaml.safe_dump(compiled.board, sort_keys=False, allow_unicode=True))
    spec_p.write_text(yaml.safe_dump(compiled.manual_spec, sort_keys=False, allow_unicode=True))
    topo_p.write_text(yaml.safe_dump(compiled.topology, sort_keys=False, allow_unicode=True))
    out: dict[str, Path] = {"board": board_p, "manual_spec": spec_p, "topology": topo_p}
    if compiled.walk_forward:
        wf_p = dest_dir / f"{run_id}.walk_forward.yaml"
        wf_p.write_text(yaml.safe_dump(compiled.walk_forward, sort_keys=False, allow_unicode=True))
        out["walk_forward"] = wf_p
    return out
