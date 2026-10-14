"""Tests for src.experiment.config_validation.

Includes read-only validation runs against the REAL config.yaml and the
real config/objectives/nq-50pt-econ.yaml in this repository (no mutation,
no DB) -- this is the "compare against existing file-based fixtures"
integration check recorded in docs/MIGRATION_MANIFEST.md: the tool's
output is checked to match what the manual audit in docs/CONFIG_AUDIT.md
already found.
"""
from pathlib import Path

import pytest
import yaml

from src.experiment.config_validation import (
    validate_config_yaml,
    validate_objective_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


def test_config_yaml_requires_objective_id(tmp_path):
    p = _write(tmp_path, "config.yaml", {"logging": {"level": "INFO"}})
    result = validate_config_yaml(p)
    assert not result.ok
    assert any("objective_id" in e for e in result.errors)


def test_config_yaml_flags_unknown_top_level_key(tmp_path):
    p = _write(tmp_path, "config.yaml", {"objective_id": "x", "totally_unknown_key": 1})
    result = validate_config_yaml(p)
    assert result.ok  # unknown keys are warnings, not errors
    assert any("totally_unknown_key" in w for w in result.warnings)


def test_config_yaml_flags_dead_sections(tmp_path):
    p = _write(tmp_path, "config.yaml", {
        "objective_id": "x",
        "model hyperparameters": {"attention_heads": 0},
        "training": {"class_weight_directional": 4.0},
    })
    result = validate_config_yaml(p)
    assert result.ok
    joined = " ".join(result.warnings)
    assert "model hyperparameters" in joined
    assert "training" in joined
    assert "DEAD" in joined


def test_config_yaml_clean_file_has_no_findings(tmp_path):
    p = _write(tmp_path, "config.yaml", {"objective_id": "nq-50pt-econ", "logging": {"level": "INFO"}})
    result = validate_config_yaml(p)
    assert result.ok
    assert not result.warnings


def test_objective_yaml_flags_slippage_model_without_slippage_points(tmp_path):
    p = _write(tmp_path, "obj.yaml", {
        "labels": {"scheme": "triple_barrier"},
        "costs": {"cost_per_trade": 0, "slippage_model": "fixed_1pt"},
    })
    result = validate_objective_yaml(p)
    assert not result.ok
    assert any("slippage_points" in e for e in result.errors)


def test_objective_yaml_ok_when_slippage_points_present(tmp_path):
    p = _write(tmp_path, "obj.yaml", {
        "costs": {"cost_per_trade": 0, "slippage_model": "fixed_1pt", "slippage_points": 1.0},
    })
    result = validate_objective_yaml(p)
    assert result.ok


def test_objective_yaml_ok_when_no_slippage_model_declared(tmp_path):
    p = _write(tmp_path, "obj.yaml", {"costs": {"cost_per_trade": 0}})
    result = validate_objective_yaml(p)
    assert result.ok


def test_bare_walk_forward_folds_declaration_is_not_flagged(tmp_path):
    """Regression guard against a twice-made error.

    Declaring eval.walk_forward_folds on its own is valid and IS honored by
    the equal-count engine (src/search/controller.py:2038). Two earlier
    versions of the validator got this wrong in opposite directions -- one
    warned that the field was always ignored, one asserted it was never
    ignored. Neither is true; only the combination below is a problem.
    """
    p = _write(tmp_path, "obj.yaml", {"eval": {"walk_forward_folds": 3}})
    result = validate_objective_yaml(p)
    assert result.ok
    assert not any("walk_forward_folds" in w for w in result.warnings)


def test_folds_declared_alongside_calendar_block_is_flagged(tmp_path):
    """The calendar engine derives its own schedule and ignores the declared
    count, so declaring both is a silent contradiction."""
    p = _write(tmp_path, "obj.yaml", {
        "eval": {
            "walk_forward_folds": 3,
            "walk_forward": {"train": {"n": 2, "unit": "months"},
                             "test": {"n": 2, "unit": "months"}},
        }
    })
    result = validate_objective_yaml(p)
    assert result.ok  # warning, not an error
    assert any("walk_forward_folds" in w and "calendar" in w for w in result.warnings)


def test_calendar_block_alone_is_not_flagged(tmp_path):
    p = _write(tmp_path, "obj.yaml", {
        "eval": {"walk_forward": {"train": {"n": 2, "unit": "months"}}}
    })
    result = validate_objective_yaml(p)
    assert not any("walk_forward_folds" in w for w in result.warnings)


def test_report_format_includes_path_and_findings(tmp_path):
    p = _write(tmp_path, "config.yaml", {})
    result = validate_config_yaml(p)
    report = result.report()
    assert str(p) in report
    assert "ERROR" in report


# --- Integration check against the real repository files (read-only) ---
# Confirms the automated tool reproduces the manual audit's findings in
# docs/CONFIG_AUDIT.md, without touching a database or mutating anything.

@pytest.mark.skipif(not (REPO_ROOT / "config.yaml").exists(), reason="real config.yaml not present")
def test_real_config_yaml_reproduces_known_dead_sections():
    result = validate_config_yaml(REPO_ROOT / "config.yaml")
    joined = " ".join(result.warnings)
    # docs/CONFIG_AUDIT.md section 1: both sections confirmed dead for the
    # live trainer as of this restructuring.
    assert "model hyperparameters" in joined
    assert "training" in joined


@pytest.mark.skipif(
    not (REPO_ROOT / "config" / "objectives" / "nq-50pt-econ.yaml").exists(),
    reason="real nq-50pt-econ.yaml not present",
)
def test_real_objective_yaml_reproduces_known_slippage_bug():
    result = validate_objective_yaml(REPO_ROOT / "config" / "objectives" / "nq-50pt-econ.yaml")
    # docs/CONFIG_AUDIT.md section 2: confirmed silent-zero-slippage bug.
    assert not result.ok
    assert any("slippage_points" in e for e in result.errors)


# --- structural validation is delegated, not re-implemented -------------------

def test_unknown_key_check_is_read_from_the_real_schema_not_a_local_allowlist():
    """Regression: a hand-written allowlist in this module omitted
    ``custom_optimization`` and ``cost_matrix`` and therefore emitted false
    "unknown key" warnings against valid objectives. The authority is
    config/objectives/objective.schema.json, enforced by
    src/objectives/loader.py."""
    from src.experiment.config_validation import _unknown_objective_keys

    assert _unknown_objective_keys(
        {"objective_id": "x", "custom_optimization": {}, "cost_matrix": {}}
    ) == set()
    assert _unknown_objective_keys({"objective_id": "x", "nonsense_key": 1}) == {
        "nonsense_key"
    }


def test_real_objective_yaml_has_no_unknown_top_level_keys():
    """The shipped objective must not trip our own structural check."""
    import yaml as _yaml
    from src.experiment.config_validation import _unknown_objective_keys

    data = _yaml.safe_load(
        (REPO_ROOT / "config" / "objectives" / "nq-50pt-econ-lg-st.yaml").read_text()
    )
    assert _unknown_objective_keys(data) == set()
