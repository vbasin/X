"""Fail-loud validation for the CURRENT config architecture (docs/CONFIG_AUDIT.md).

This module validates `config.yaml` / `config/objectives/*.yaml` **as they
exist today** -- it does not define or require a new config format (see
`docs/CONFIG_AUDIT.md` section 6, "Current vs. Target"). It exists to catch,
mechanically, the two concrete classes of failure the audit found by hand:

  1. Unknown/unsupported top-level keys (nothing catches this today --
     `config.yaml`'s ``model hyperparameters``/``training`` block is dead
     for the live trainer and nothing warns about it).
  2. Known-dead or contradicted fields, checked by an explicit rule list
     derived from ``docs/CONFIG_AUDIT.md`` (e.g. ``costs.slippage_model``
     declared but never translated into what
     ``src/backtest/engine.py`` actually reads).

Read-only: loads YAML from disk with ``yaml.safe_load``, does not import or
touch any DB client, does not write anything. Does not change model science
-- it reports findings, it does not alter config values or executable
behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Known top-level keys for config.yaml, as consumed by *some* code path
# (see docs/CONFIG_AUDIT.md section 1 for which keys are live vs. dead).
KNOWN_CONFIG_YAML_KEYS = frozenset({
    "objective_id",
    "mlflow",
    "dataset parameters",
    "model hyperparameters",
    "training",
    "logging",
})

# Confirmed DEAD for the live Custom Test Lab trainer (src/search/custom_test.py
# -> src/models/catalog.py), per docs/CONFIG_AUDIT.md section 1: these are only
# consumed by the disconnected src/training/trainer.py.
DEAD_CONFIG_YAML_SECTIONS = frozenset({"model hyperparameters", "training"})

# Known top-level keys for an objective doc (config/objectives/*.yaml).
# Verified against the REAL config/objectives/nq-50pt-econ.yaml on disk
# (read directly, not inferred) during this restructuring's own fresh-agent
# test -- an earlier version of this set was built from secondhand notes and
# was missing more than half of these, producing false-positive "unknown
# key" warnings. Left as a cautionary note: always verify a schema against
# the real file, not a summary of it.
OBJECTIVE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "objectives" / "objective.schema.json"
)
"""The AUTHORITY for an objective document's shape.

This module previously carried a hand-written ``KNOWN_OBJECTIVE_KEYS``
allowlist. That was a parallel abstraction and it was already wrong: it
omitted ``custom_optimization`` and ``cost_matrix``, both of which the real
schema declares, so it emitted false "unknown key" warnings against valid
objectives. ``config/objectives/objective.schema.json`` is enforced at load
time by ``src/objectives/loader.py`` (consumed by ``src/paper/policy.py``,
``src/backtest/runner.py``, ``src/search/finalize.py``) and sets
``additionalProperties: false`` at the top level -- structural validation is
already solved and must not be re-implemented here.

What this module adds is what the schema deliberately cannot express: the
CONSUMER-DERIVED semantic checks below, where a document that is perfectly
valid structurally still produces a silently wrong result.
"""


@dataclass
class ConfigValidationResult:
    """Findings from validating one config file. Never raises on its own --
    callers decide what to do with ``errors`` vs ``warnings``."""

    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> str:
        lines = [f"Config validation: {self.path}"]
        if not self.errors and not self.warnings:
            lines.append("  no findings")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping, got {type(data).__name__}")
    return data


def validate_config_yaml(path: str | Path) -> ConfigValidationResult:
    """Validate a ``config.yaml``-shaped file.

    Checks (see module docstring):
      - required field ``objective_id`` present and non-empty (ERROR if not)
      - unknown top-level keys (WARNING -- unknown does not necessarily mean
        wrong, but it means nothing in this validator's knowledge vouches
        for it being consumed by anything)
      - known-dead sections present (WARNING, cites docs/CONFIG_AUDIT.md)
    """
    path = Path(path)
    result = ConfigValidationResult(path=str(path))
    data = _load_yaml(path)

    if not data.get("objective_id"):
        result.errors.append(
            "required field 'objective_id' is missing or empty"
        )

    unknown = set(data) - KNOWN_CONFIG_YAML_KEYS
    for key in sorted(unknown):
        result.warnings.append(
            f"unknown top-level key {key!r} -- not in the known-consumed set "
            f"{sorted(KNOWN_CONFIG_YAML_KEYS)}; verify it is actually read by "
            f"some code path before relying on it"
        )

    for section in sorted(DEAD_CONFIG_YAML_SECTIONS & set(data)):
        result.warnings.append(
            f"section {section!r} is present but confirmed DEAD for the live "
            f"Custom Test Lab trainer (src/search/custom_test.py never "
            f"imports src/training/trainer.py, which is the only consumer of "
            f"this section) -- see docs/CONFIG_AUDIT.md section 1"
        )

    return result


def _unknown_objective_keys(data: Mapping[str, Any]) -> set[str]:
    """Top-level keys the real JSON Schema does not declare.

    Read from ``objective.schema.json`` rather than from a local allowlist,
    so this can never drift out of sync with what ``src/objectives/loader.py``
    actually enforces. If the schema is unreadable, report nothing rather
    than guessing -- a fabricated "unknown key" warning is worse than silence.
    """
    try:
        schema = json.loads(OBJECTIVE_SCHEMA_PATH.read_text())
    except (OSError, ValueError):
        return set()
    if schema.get("additionalProperties") is not False:
        return set()
    return set(data) - set(schema.get("properties") or {})


def validate_objective_yaml(path: str | Path) -> ConfigValidationResult:
    """Validate a ``config/objectives/<id>.yaml``-shaped file.

    This does NOT re-implement structural validation -- see
    ``OBJECTIVE_SCHEMA_PATH``. It adds the consumer-derived semantic checks
    the schema cannot express:

      - unknown top-level keys, read from the schema itself (WARNING)
      - the confirmed ``slippage_model``/``slippage_points`` mismatch: if
        ``costs.slippage_model`` is set but ``costs.slippage_points`` is not,
        this is the exact condition under which
        ``src/backtest/engine.py`` silently evaluates at zero slippage
        (docs/CONFIG_AUDIT.md section 2) -- flagged as an ERROR, not a
        warning, because it produces a specific, previously-confirmed silent
        wrong result rather than just an unverified field.
    """
    path = Path(path)
    result = ConfigValidationResult(path=str(path))
    data = _load_yaml(path)

    for key in sorted(_unknown_objective_keys(data)):
        result.warnings.append(
            f"unknown top-level key {key!r} -- not declared in "
            f"config/objectives/objective.schema.json, which sets "
            f"additionalProperties: false. src/objectives/loader.py will "
            f"reject this document outright."
        )

    costs = data.get("costs") or {}
    if isinstance(costs, dict) and costs.get("slippage_model") and "slippage_points" not in costs:
        result.errors.append(
            "costs.slippage_model is set but costs.slippage_points is not -- "
            "src/backtest/engine.py only reads 'slippage_points' and will "
            "silently evaluate at zero slippage despite slippage_model being "
            "declared (confirmed bug, docs/CONFIG_AUDIT.md section 2)"
        )

    # Two walk-forward engines with incompatible config contracts
    # (docs/CONFIG_AUDIT.md, "eval.walk_forward_folds"):
    #
    #   expanding_equal_count  src/search/controller.py:2038
    #       reads eval.walk_forward_folds as an INPUT.
    #   expanding_calendar     src/validation/walk_forward.py:311-330
    #       never reads it; derives the schedule from eval.walk_forward.*
    #       and eval.date_range, reporting the count as an OUTPUT (:293).
    #
    # So declaring BOTH is a silent contradiction: the calendar schedule wins
    # and the declared fold count is never consulted. That is mechanically
    # detectable from the YAML alone, unlike the bare declaration (which is
    # perfectly valid on its own and must NOT be warned about -- two earlier
    # versions of this module got that wrong in opposite directions).
    eval_block = data.get("eval") or {}
    if isinstance(eval_block, dict):
        declared_folds = eval_block.get("walk_forward_folds")
        calendar_block = eval_block.get("walk_forward")
        if declared_folds is not None and isinstance(calendar_block, dict) and calendar_block:
            result.warnings.append(
                f"eval.walk_forward_folds={declared_folds!r} is declared alongside an "
                f"eval.walk_forward calendar block. These configure two different "
                f"engines: the calendar engine "
                f"(src/validation/walk_forward.py:311-330) derives its own "
                f"schedule and never reads walk_forward_folds, so the declared count "
                f"will be silently ignored when the calendar path is selected. Three "
                f"real nq-50pt-econ runs collapsed to a single fold -- i.e. no "
                f"walk-forward validation -- this way (docs/CONFIG_AUDIT.md)."
            )

    return result
