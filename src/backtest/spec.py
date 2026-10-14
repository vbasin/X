"""Strategy-spec loader — validates a user-defined strategy YAML against its JSON schema.

The spec (see ``config/strategy_spec.schema.json`` + ``config/strategy.example.yaml``) is
the user interface for the rule-based strategy entry path: entry rules (feature values &
interactions) OR a calendar cadence (daily/weekly/monthly + settle), exits, the trade
date range, and cost knobs. The referenced Trade Objective still owns labels/KPI defaults
so strategies land on the same frozen evaluation contract as ML model candidates.

Usage::

    from src.backtest.spec import load_strategy_spec
    spec = load_strategy_spec("config/strategy.example.yaml")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.backtest.rules import rule_features, _OPS

logger = logging.getLogger("vtech.backtest.spec")

_REPO_ROOT = Path(__file__).resolve().parents[2]
from src.common.config_paths import schema_path as _schema_path
_SCHEMA_PATH = _schema_path("strategy_spec.schema.json")

_DEFAULT_EXIT = {"eod_flat": True}
_DEFAULT_BACKTEST = {"cooldown_bars": 0, "point_value": 1.0, "max_trades_per_day": 0}


@dataclass
class StrategySpec:
    """Parsed + validated strategy definition."""

    strategy_id: str
    objective_id: str | None
    mode: str                                   # "rules" | "schedule" | "predictions"
    entry: dict[str, Any] | None                # {"long": block, "short": block}
    schedule: dict[str, Any] | None
    predictions: dict[str, Any] | None          # {"source", "tau", "confidence_gate", "sides"}
    filters: dict[str, Any] | None
    exit: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_EXIT))
    backtest: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_BACKTEST))
    costs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    source: str = "<dict>"
    raw: dict[str, Any] = field(default_factory=dict)

    def referenced_features(self) -> set[str]:
        """All feature columns any rule in the spec touches (for availability checks)."""
        feats: set[str] = set()
        for side in ("long", "short"):
            feats |= rule_features((self.entry or {}).get(side))
            feats |= rule_features(((self.exit or {}).get("rules") or {}).get(side))
        feats |= rule_features(self.filters)
        return feats


def load_strategy_spec(
    spec_or_path: str | Path | dict[str, Any],
    *,
    objective_id: str | None = None,
) -> StrategySpec:
    """Load, schema-validate and semantically check a strategy spec.

    Raises ``FileNotFoundError`` / ``ValueError`` on a missing file, schema violation,
    objective mismatch, or an internally inconsistent spec (e.g. holdout outside the
    backtest window).
    """
    if isinstance(spec_or_path, dict):
        raw, source = spec_or_path, "<dict>"
    else:
        from src.common.config_paths import resolve_user_path
        p = resolve_user_path(spec_or_path, kind="specs")
        if not p.exists():
            raise FileNotFoundError(f"Strategy spec not found: {spec_or_path}")
        raw = yaml.safe_load(p.read_text()) or {}
        source = str(p)

    _coerce_dates(raw)
    _validate_schema(raw, source)

    spec_obj_id = raw.get("objective_id")
    if objective_id and spec_obj_id and spec_obj_id != objective_id:
        raise ValueError(
            f"strategy spec objective_id '{spec_obj_id}' != run objective '{objective_id}' "
            f"({source}) — refusing to backtest a strategy against the wrong objective")

    entry = raw.get("entry")
    schedule = raw.get("schedule")
    predictions = raw.get("predictions")
    active = [name for name, blk in
              (("rules", entry), ("schedule", schedule), ("predictions", predictions)) if blk]
    if len(active) != 1:
        raise ValueError(
            f"strategy spec must define exactly one of entry|schedule|predictions "
            f"(got {active or 'none'}) ({source})")
    mode = active[0]

    spec = StrategySpec(
        strategy_id=raw["strategy_id"],
        objective_id=spec_obj_id or objective_id,
        mode=mode,
        entry=entry,
        schedule=schedule,
        predictions=predictions,
        filters=raw.get("filters"),
        exit={**_DEFAULT_EXIT, **(raw.get("exit") or {})},
        backtest={**_DEFAULT_BACKTEST, **(raw.get("backtest") or {})},
        costs=dict(raw.get("costs") or {}),
        description=raw.get("description") or "",
        source=source,
        raw=raw,
    )
    _validate_semantics(spec)
    return spec


# ── validation ────────────────────────────────────────────────────────────────

def _coerce_dates(raw: dict[str, Any]) -> None:
    """YAML parses unquoted ``2025-06-02`` as a ``datetime.date``; the schema wants an ISO
    string. Coerce date/datetime values in the backtest block in place."""
    import datetime as _dt
    bt = raw.get("backtest")
    if not isinstance(bt, dict):
        return
    for k in ("start", "end", "holdout_start"):
        v = bt.get(k)
        if isinstance(v, (_dt.date, _dt.datetime)):
            bt[k] = v.isoformat()[:10]


def _validate_schema(raw: dict[str, Any], source: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        logger.warning("jsonschema not installed — strategy-spec schema validation skipped")
        return
    schema = json.loads(_SCHEMA_PATH.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda e: list(e.path))
    if errors:
        msgs = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                         for e in errors[:5])
        raise ValueError(f"strategy spec fails schema validation ({source}): {msgs}")


def _validate_semantics(spec: StrategySpec) -> None:
    """Checks the JSON schema cannot express."""
    src = spec.source

    def _walk(block: dict[str, Any] | None, where: str) -> None:
        if not block:
            return
        items = block.get("all") or block.get("any") or []
        for item in items:
            if "feature" in item:
                op = item.get("op")
                if op not in _OPS:
                    raise ValueError(f"unknown op '{op}' in {where} ({src})")
                if op == "between":
                    v = item.get("value")
                    if not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]):
                        raise ValueError(
                            f"'between' needs value: [low, high] with low<=high in {where} ({src})")
            else:
                _walk(item, where)

    for side in ("long", "short"):
        _walk((spec.entry or {}).get(side), f"entry.{side}")
        _walk(((spec.exit or {}).get("rules") or {}).get(side), f"exit.rules.{side}")
    _walk(spec.filters, "filters")

    bt = spec.backtest
    start, end, holdout = bt.get("start"), bt.get("end"), bt.get("holdout_start")
    for k, v in (("start", start), ("end", end), ("holdout_start", holdout)):
        if v is not None:
            try:
                import pandas as pd
                pd.Timestamp(str(v))
            except Exception as exc:
                raise ValueError(f"backtest.{k} is not a valid date: {v!r} ({src})") from exc
    if start and end and str(start) > str(end):
        raise ValueError(f"backtest.start {start} is after backtest.end {end} ({src})")
    if holdout and start and str(holdout) < str(start):
        raise ValueError(f"backtest.holdout_start {holdout} precedes start {start} ({src})")
    if holdout and end and str(holdout) > str(end):
        raise ValueError(f"backtest.holdout_start {holdout} is after end {end} ({src})")

    if spec.mode == "schedule":
        sched = spec.schedule or {}
        freq = sched.get("frequency")
        if freq == "weekly" and not sched.get("day_of_week"):
            raise ValueError(f"weekly schedule requires day_of_week ({src})")
        if freq == "monthly" and sched.get("day_of_month") is None:
            raise ValueError(f"monthly schedule requires day_of_month ({src})")
        settle = sched.get("settle") or {}
        if settle.get("mode") == "hold_bars" and not settle.get("bars"):
            raise ValueError(f"settle.mode=hold_bars requires settle.bars ({src})")
        if settle.get("mode") == "hold_days" and not settle.get("days"):
            raise ValueError(f"settle.mode=hold_days requires settle.days ({src})")
