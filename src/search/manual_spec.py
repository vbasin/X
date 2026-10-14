"""User-override candidate loader — Phase 23.

Turns a **manual candidates spec** (see ``config/manual_candidates.schema.json``) into a
list of :class:`~src.models.catalog.CatalogEntry` objects the search controller can run
directly, *bypassing* the automated Model-Scout / catalog candidate proposal.

The Trade Objective still owns labels, KPI, targets and the walk-forward/eval contract;
this spec only replaces the *which-model(s)* decision. Each user candidate becomes a
synthetic ``CatalogEntry`` derived from its family's catalog defaults with:

* ``name``            ← the user's ``id`` (unique candidate id; defaults to ``family``)
* ``plugin_family``   ← ``family`` (so several candidates can share one family)
* ``params_default``  ← catalog defaults **merged with** the user's ``params``
* ``feature_selection`` ← the user's ``features`` (view / prefixes / columns) or None
* ``search_max_rows`` ← the user's ``max_rows`` (default ``None`` = full matrix; manual
                        mode fits the real data, it does not sub-sample like the search)
* ``search_space``    ← the candidate's ``search_space`` when present (Custom Test
                        opt-in HPO, ADR-040); otherwise cleared (``{}``) so
                        ``--tune-trials`` cannot override params the user fixed
                        (ADR-019 default).

Usage::

    from src.search.manual_spec import load_manual_spec
    spec = load_manual_spec("config/manual_candidates.example.yaml")
    entries = spec.entries          # list[CatalogEntry] for run_search
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.models.catalog import CatalogEntry, get_catalog, get_entry

logger = logging.getLogger("vtech.search.manual_spec")

_REPO_ROOT = Path(__file__).resolve().parents[2]
from src.common.config_paths import schema_path as _schema_path
_SCHEMA_PATH = _schema_path("manual_candidates.schema.json")


@dataclass
class ManualSpec:
    """Parsed + validated user-override candidate set."""

    entries: list[CatalogEntry]
    objective_id: str | None = None
    ensembles: bool = False
    source: str = "<dict>"
    raw: dict[str, Any] = field(default_factory=dict)


def load_manual_spec(
    spec_or_path: str | Path | dict[str, Any],
    *,
    objective_id: str | None = None,
) -> ManualSpec:
    """Load, validate and materialise a manual candidates spec.

    Parameters
    ----------
    spec_or_path:
        A path to a YAML/JSON spec, or an already-parsed dict.
    objective_id:
        Optional. When given and the spec also names an ``objective_id``, the two must
        match — a guard against pairing a spec with the wrong objective.

    Returns
    -------
    ManualSpec
        With ``entries`` ready to feed the search controller.

    Raises
    ------
    FileNotFoundError, ValueError
        On a missing file, schema-invalid spec, unknown family, or duplicate id.
    """
    if isinstance(spec_or_path, dict):
        raw = spec_or_path
        source = "<dict>"
    else:
        from src.common.config_paths import resolve_user_path
        p = resolve_user_path(spec_or_path, kind="specs")
        if not p.exists():
            raise FileNotFoundError(f"Manual candidates spec not found: {spec_or_path}")
        raw = yaml.safe_load(p.read_text()) or {}
        source = str(p)

    _validate(raw, source)

    spec_obj_id = raw.get("objective_id")
    if objective_id and spec_obj_id and spec_obj_id != objective_id:
        raise ValueError(
            f"manual spec objective_id '{spec_obj_id}' != run objective '{objective_id}' "
            f"({source}) — refusing to run a spec against the wrong objective")

    entries = _build_entries(raw.get("candidates") or [], source)

    return ManualSpec(
        entries=entries,
        objective_id=spec_obj_id,
        ensembles=bool(raw.get("ensembles", False)),
        source=source,
        raw=raw,
    )


def _build_entries(candidates: list[dict[str, Any]], source: str) -> list[CatalogEntry]:
    known = {e.name for e in get_catalog()}
    entries: list[CatalogEntry] = []
    seen: set[str] = set()

    for i, cand in enumerate(candidates):
        family = cand["family"]
        if family not in known:
            raise ValueError(
                f"manual candidate #{i} references unknown family '{family}' ({source}); "
                f"known families: {sorted(known)}")

        cid = cand.get("id") or family
        if cid in seen:
            raise ValueError(
                f"duplicate manual candidate id '{cid}' ({source}); "
                "set a unique `id` when several candidates share a family")
        seen.add(cid)

        base = get_entry(family)
        params = {**base.params_default, **(cand.get("params") or {})}
        features = cand.get("features")            # dict | None (schema-validated shape)
        search_space = _candidate_search_space(cand.get("search_space"), cid)

        entries.append(dataclasses.replace(
            base,
            name=cid,
            # Alias catalog names (xgboost_ovr) keep the registered plugin
            # (xgboost). Using `family` here made family_key=xgboost_ovr and
            # the plugin loader skipped the candidate.
            plugin_family=base.plugin_family or family,
            params_default=params,
            feature_selection=features,
            feature_view=None,                     # feature_selection supersedes named view
            search_max_rows=cand.get("max_rows"),  # None (omitted/null) = full matrix
            search_space=search_space,
        ))

    logger.info("manual_spec_loaded n=%d families=%s source=%s",
                len(entries), [e.family_key for e in entries], source)
    return entries


def _candidate_search_space(raw: Any, cid: str) -> dict[str, Any]:
    """Keep an explicit Optuna space; omit / empty → no auto-tuning (ADR-019)."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"manual candidate '{cid}' search_space must be an object")
    out: dict[str, Any] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"manual candidate '{cid}' search_space '{name}' must be an object")
        t = spec.get("type")
        if t not in ("int", "float", "categorical"):
            raise ValueError(
                f"manual candidate '{cid}' search_space '{name}' needs type "
                f"int|float|categorical, got {t!r}")
        out[str(name)] = spec
    return out


def _validate(raw: dict[str, Any], source: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        logger.warning("jsonschema not installed — manual-spec schema validation skipped")
        return
    schema = json.loads(_SCHEMA_PATH.read_text())
    errs = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: list(e.path))
    if errs:
        msgs = "; ".join(f"{list(e.path)}: {e.message}" for e in errs)
        raise ValueError(f"Manual spec {source} failed schema validation: {msgs}")
