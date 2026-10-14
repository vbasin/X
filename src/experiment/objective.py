"""What an ``objective_id`` MEANS, read from the existing objective schema.

``config/objectives/objective.schema.json`` tags properties with
``x-objective-role``:

  identity  defines the trading problem; a material change normally needs a
            new objective_id (underlying, directions, horizon, eligible
            session, outcome/stop/path rule, economic_intent)
  default   an experiment default that may live in the objective YAML but
            does NOT define it (models, features, dates, folds, thresholds,
            search, costs/sizing, kpi.primary -- a RANKING metric, not the
            durable economic intent -- decision_grid_seconds -- sampling
            cadence, not the underlying problem -- ...)
  meta      documentation / bookkeeping (e.g. a free-text ``note``)
  mixed     container whose children carry their own roles; an untagged
            child of a ``mixed`` container defaults to ``default``

This module only READS those tags -- it is not a second classification.

Role resolution is RECURSIVE, not "one tag covers the whole subtree": an
``identity``-tagged object/array is walked field-by-field, and a nested
field explicitly tagged ``meta``/``default`` is excluded even though its
container is ``identity``. This is why ``traded products[].note`` (free-text
commentary on the traded instrument) does not enter the identity hash even
though ``traded products`` itself is identity -- only its untagged siblings
(``symbol``, ``security_type``, ``asset_class``, ``role``) do, by inheriting
the container's role.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "objectives" / "objective.schema.json"
)

_EXCLUDED = object()  # sentinel: this node contributes nothing to identity


def _schema() -> Mapping[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def field_roles() -> dict[str, str]:
    """Dotted path -> role, for top-level properties and one level into any
    ``mixed`` container. (Deeper nesting, e.g. inside an ``identity`` array
    of objects, is resolved by :func:`objective_identity` directly rather
    than flattened here -- a dotted path through an array is ambiguous.)
    """
    out: dict[str, str] = {}
    for key, spec in (_schema().get("properties") or {}).items():
        role = spec.get("x-objective-role")
        out[key] = role
        if role == "mixed":
            for sub, sspec in (spec.get("properties") or {}).items():
                out[f"{key}.{sub}"] = sspec.get("x-objective-role", "default")
    return out


def _role(schema_node: Any, inherited: str | None) -> str | None:
    """A node's own tag if present, else whatever its parent resolved to."""
    if isinstance(schema_node, Mapping) and "x-objective-role" in schema_node:
        return schema_node["x-objective-role"]
    return inherited


def _filter(value: Any, schema_node: Any, inherited_role: str | None) -> Any:
    """Recursively keep only identity-bearing content. Returns _EXCLUDED for
    a node that contributes nothing (meta/default, or an untyped mismatch).
    """
    role = _role(schema_node, inherited_role)
    node = schema_node if isinstance(schema_node, Mapping) else {}

    if role in (None, "meta", "default"):
        return _EXCLUDED

    if role == "mixed":
        # Untagged children of a MIXED container are experiment defaults by
        # convention (this is the pre-existing session/labels/kpi rule) --
        # only children explicitly tagged identity survive.
        if not isinstance(value, Mapping):
            return _EXCLUDED
        props = node.get("properties") or {}
        out = {k: f for k, v in value.items()
               if (f := _filter(v, props.get(k), "default")) is not _EXCLUDED}
        return out if out else _EXCLUDED

    if role == "identity":
        # Untagged children of an IDENTITY container INHERIT identity (the
        # opposite default from `mixed`): traded products[].symbol has no
        # tag of its own and is identity only because its container is.
        # Only an EXPLICIT meta/default tag (e.g. traded products[].note)
        # removes a field from an identity container.
        if isinstance(value, Mapping) and "properties" in node:
            props = node["properties"]
            return {k: f for k, v in value.items()
                    if (f := _filter(v, props.get(k), "identity")) is not _EXCLUDED}
        if isinstance(value, list) and isinstance(node.get("items"), Mapping):
            item_schema = node["items"]
            return [f for v in value
                    if (f := _filter(v, item_schema, "identity")) is not _EXCLUDED]
        return value  # scalar, or a container the schema doesn't describe further

    return _EXCLUDED


def objective_identity(objective: Mapping[str, Any]) -> dict[str, Any]:
    """The identity-bearing subset of an objective document."""
    schema_props = _schema().get("properties") or {}
    return {
        k: f for k, v in objective.items()
        if (f := _filter(v, schema_props.get(k), None)) is not _EXCLUDED
    }


def objective_identity_sha256(objective: Mapping[str, Any]) -> str:
    """Stable hash of what the objective MEANS -- unchanged by default or
    metadata edits (kpi.primary, decision_grid_seconds, a `note`, ...)."""
    blob = json.dumps(objective_identity(objective), sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
