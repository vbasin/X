"""Rule DSL: schema + top-level compile / validate.

The rule tree is a plain JSON document. A minimal valid model:

.. code-block:: json

    {
      "id": "vp_break_long",
      "name": "VP1 break long",
      "rules": [
        {
          "when": [{"time_between": ["08:46", "11:00"]}],
          "if": {"all": [
            {"eq": ["C.color", "green"]},
            {"gt": ["Co", "VP1.vah"]},
            {"gt": ["Ch", "VP1.vah"]}
          ]},
          "then": [
            {"entry_long": "Cc + 1"},
            {"set_target": "next_up_of(entry, VP1.vah, VP1.high, VP2.vah, VP2.high)"},
            {"set_stop": "Cl"}
          ]
        }
      ]
    }

Node kinds
----------

- ``when``: list of gates that must all hold (``time_between``,
  ``bars_from_open_ge``). Missing/empty ``when`` means "always".
- ``if``: a boolean node: ``{"all": [...]}``, ``{"any": [...]}``,
  ``{"not": <node>}``, or a leaf predicate object. Leaf predicates:

  - ``{"eq":  [expr, expr]}`` — equal
  - ``{"ne":  [expr, expr]}`` — not equal
  - ``{"gt":  [expr, expr]}``
  - ``{"lt":  [expr, expr]}``
  - ``{"ge":  [expr, expr]}``
  - ``{"le":  [expr, expr]}``
  - ``{"between": [expr, low, high]}``
  - ``{"cross_above": [a, b]}`` — a crosses above b this bar
  - ``{"cross_below": [a, b]}``

- ``then``: list of actions and/or nested ``{"if": ..., "then": ...}``:

  - ``{"entry_long":  <expr>}`` / ``{"entry_short": <expr>}`` —
    ``<expr>`` is the fill price at signal-bar close
  - ``{"set_target": <expr>}`` / ``{"set_stop": <expr>}`` —
    resolved at entry and latched for the life of the trade
  - ``{"exit": true}`` — flatten now (next bar open)
"""

from __future__ import annotations

import re
from typing import Any

from src.webui.chart_models.expr import compile_expr, ExprError

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

_LEAF_PREDICATES = {
    "eq", "ne", "gt", "lt", "ge", "le",
    "between", "cross_above", "cross_below",
}
_WHEN_GATES = {"time_between", "bars_from_open_ge", "bars_from_open_le"}
_ACTIONS = {"entry_long", "entry_short", "set_target", "set_stop", "exit"}


class SchemaError(ValueError):
    """Raised for a malformed rule tree."""


def _err(where: str, msg: str) -> SchemaError:
    return SchemaError(f"{where}: {msg}")


def _need_hhmm(v: Any, where: str) -> None:
    if not (isinstance(v, str) and _HHMM.match(v)):
        raise _err(where, f"expected HH:MM string, got {v!r}")


def _need_int(v: Any, where: str, *, ge: int | None = None) -> None:
    if not isinstance(v, int) or isinstance(v, bool):
        raise _err(where, f"expected integer, got {v!r}")
    if ge is not None and v < ge:
        raise _err(where, f"must be >= {ge}")


def _validate_expr(expr: Any, where: str, catalog: dict[str, Any] | None) -> None:
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return
    if not isinstance(expr, str):
        raise _err(where, f"expected expression string or number, got {type(expr).__name__}")
    try:
        compile_expr(expr, catalog=catalog)
    except ExprError as exc:
        raise _err(where, f"invalid expression: {exc}") from exc


def _validate_when(gates: Any, catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    if gates is None:
        return []
    if not isinstance(gates, list):
        raise _err("when", "must be a list")
    out: list[dict[str, Any]] = []
    for i, g in enumerate(gates):
        if not isinstance(g, dict) or len(g) != 1:
            raise _err(f"when[{i}]", "must be a single-key object")
        kind, val = next(iter(g.items()))
        if kind not in _WHEN_GATES:
            raise _err(f"when[{i}]", f"unknown gate {kind!r}; allowed: {sorted(_WHEN_GATES)}")
        if kind == "time_between":
            if not (isinstance(val, list) and len(val) == 2):
                raise _err(f"when[{i}].time_between", "expected [HH:MM, HH:MM]")
            _need_hhmm(val[0], f"when[{i}].time_between[0]")
            _need_hhmm(val[1], f"when[{i}].time_between[1]")
        else:
            _need_int(val, f"when[{i}].{kind}", ge=0)
        out.append(g)
    return out


def _validate_if(node: Any, where: str, catalog: dict[str, Any] | None) -> dict[str, Any]:
    if node is None:
        return {"all": []}
    if not isinstance(node, dict) or len(node) != 1:
        raise _err(where, "must be a single-key object")
    kind, val = next(iter(node.items()))
    if kind in ("all", "any"):
        if not isinstance(val, list):
            raise _err(f"{where}.{kind}", "must be a list of predicate nodes")
        for i, child in enumerate(val):
            _validate_if(child, f"{where}.{kind}[{i}]", catalog)
        return node
    if kind == "not":
        _validate_if(val, f"{where}.not", catalog)
        return node
    if kind in _LEAF_PREDICATES:
        if not isinstance(val, list):
            raise _err(f"{where}.{kind}", "must be an argument list")
        expected = 3 if kind == "between" else 2
        if len(val) != expected:
            raise _err(f"{where}.{kind}", f"expected {expected} arguments, got {len(val)}")
        for i, arg in enumerate(val):
            _validate_expr(arg, f"{where}.{kind}[{i}]", catalog)
        return node
    raise _err(where, f"unknown predicate {kind!r}")


def _validate_then(actions: Any, where: str, catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise _err(where, "must be a list of actions/nested if-then nodes")
    out: list[dict[str, Any]] = []
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            raise _err(f"{where}[{i}]", "must be an object")
        # nested rule (if+then) inside then — supports hierarchical trees
        if "if" in a or "then" in a:
            if "if" not in a or "then" not in a:
                raise _err(f"{where}[{i}]", "nested branch must have both 'if' and 'then'")
            a = {
                "if": _validate_if(a["if"], f"{where}[{i}].if", catalog),
                "then": _validate_then(a["then"], f"{where}[{i}].then", catalog),
            }
            out.append(a)
            continue
        if len(a) != 1:
            raise _err(f"{where}[{i}]", "must be a single-key action object")
        kind, val = next(iter(a.items()))
        if kind not in _ACTIONS:
            raise _err(f"{where}[{i}]", f"unknown action {kind!r}; allowed: {sorted(_ACTIONS)}")
        if kind == "exit":
            if val not in (True, 1, "true"):
                raise _err(f"{where}[{i}].exit", "expected true")
            out.append({"exit": True})
            continue
        # entry / set_target / set_stop — value is an expression that may
        # reference `entry` (only meaningful in set_target/set_stop).
        cat = catalog
        if kind in ("set_target", "set_stop") and catalog is not None:
            cat = {**catalog, "entry": {"kind": "scalar"}}
        _validate_expr(val, f"{where}[{i}].{kind}", cat)
        out.append({kind: val})
    return out


def _validate_rule(rule: Any, where: str, catalog: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(rule, dict):
        raise _err(where, "rule must be an object")
    when = _validate_when(rule.get("when"), catalog)
    if_node = _validate_if(rule.get("if"), f"{where}.if", catalog)
    then = _validate_then(rule.get("then"), f"{where}.then", catalog)
    # A rule must contain at least one entry action somewhere in its `then`
    # (nested or top-level), or be a pure-exit rule.
    if not _contains_entry_or_exit(then):
        raise _err(where, "rule must contain at least one entry_long/entry_short/exit action")
    return {"when": when, "if": if_node, "then": then}


def _contains_entry_or_exit(then: list[dict[str, Any]]) -> bool:
    for node in then:
        if "if" in node and "then" in node:
            if _contains_entry_or_exit(node["then"]):
                return True
        elif any(k in node for k in ("entry_long", "entry_short", "exit")):
            return True
    return False


def validate_model(
    model: dict[str, Any],
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a rule spec against the schema and a variable *catalog*.

    Passing ``catalog`` (built by :func:`.vars.build_var_catalog`) lets the
    expression parser reject references to variables that are not actually
    exposed by the current chart configuration.
    """
    if not isinstance(model, dict):
        raise SchemaError("model must be an object")
    mid = model.get("id")
    if mid is not None and not (isinstance(mid, str) and _ID_RE.match(mid)):
        raise SchemaError("id must be lowercase alphanumeric / _ / -, 1–64 chars")
    name = model.get("name") or mid or ""
    if not isinstance(name, str) or len(name) > 128:
        raise SchemaError("name must be a string (<=128 chars)")
    rules = model.get("rules")
    if not isinstance(rules, list) or not rules:
        raise SchemaError("rules must be a non-empty list")
    if len(rules) > 32:
        raise SchemaError("at most 32 rules per model")
    out_rules = [_validate_rule(r, f"rules[{i}]", catalog) for i, r in enumerate(rules)]
    return {
        "id": mid or "",
        "name": name.strip() or (mid or "untitled"),
        "notes": str(model.get("notes") or ""),
        "rules": out_rules,
    }


def compile_model(
    model: dict[str, Any],
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate + pre-compile all expressions inside the model.

    Returns a spec where every expression string is replaced by a compiled
    AST (see :mod:`.expr`). Any error is surfaced as :class:`SchemaError`.
    """
    spec = validate_model(model, catalog=catalog)
    spec["rules"] = [_compile_rule(r, catalog) for r in spec["rules"]]
    return spec


def _compile_rule(rule: dict[str, Any], catalog: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "when": rule["when"],
        "if": _compile_if(rule["if"], catalog),
        "then": _compile_then(rule["then"], catalog),
    }


def _compile_if(node: dict[str, Any], catalog: dict[str, Any] | None) -> dict[str, Any]:
    kind, val = next(iter(node.items()))
    if kind in ("all", "any"):
        return {kind: [_compile_if(child, catalog) for child in val]}
    if kind == "not":
        return {"not": _compile_if(val, catalog)}
    # leaf predicate: compile each argument if it is a string
    return {kind: [_compile_arg(arg, catalog) for arg in val]}


def _compile_arg(arg: Any, catalog: dict[str, Any] | None) -> Any:
    if isinstance(arg, str):
        return {"__expr__": compile_expr(arg, catalog=catalog)}
    return arg


def _compile_then(actions: list[dict[str, Any]], catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in actions:
        if "if" in a and "then" in a:
            out.append({
                "if": _compile_if(a["if"], catalog),
                "then": _compile_then(a["then"], catalog),
            })
            continue
        kind, val = next(iter(a.items()))
        if kind == "exit":
            out.append({"exit": True})
            continue
        cat = catalog
        if kind in ("set_target", "set_stop") and catalog is not None:
            cat = {**catalog, "entry": {"kind": "scalar"}}
        out.append({kind: {"__expr__": compile_expr(str(val), catalog=cat)}})
    return out
