"""Experiment: durable, comparable identity for one research run.

Status: PARTIAL_IMPLEMENTATION -- ``manifest.py`` is the run identity for
Track B search runs and for objective runs. ``artifacts.py`` names the
files. ``analysis.py`` joins a model's declared inputs as-of the decision timestamp. ``config_validation.py``,
``comparison.py``, and ``schemas.py`` are real and tested. Search still
also writes its older ``run.json``; new objective runs do not.

Explicitly replaces version-suffixed filenames (``move_matrix_v3``, ``_v4``)
as the unit of scientific identity -- a filename tells you a build happened,
not what was controlled or what a result should be compared against. See
``docs/EXPERIMENTS.md`` for the full contract and rationale.
"""

from src.experiment.comparison import ComparisonReport, compare_runs
from src.experiment.config_validation import (
    ConfigValidationResult,
    validate_config_yaml,
    validate_objective_yaml,
)
from src.experiment.manifest import (
    CodeIdentity,
    DatasetRef,
    RunManifest,
    ValidationDesign,
)
from src.experiment.schemas import (
    CalibrationResult,
    FoldMetrics,
    SelectedPopulation,
)

__all__ = [
    "RunManifest",
    "CodeIdentity",
    "DatasetRef",
    "ValidationDesign",
    "compare_runs",
    "ComparisonReport",
    "validate_config_yaml",
    "validate_objective_yaml",
    "ConfigValidationResult",
    "FoldMetrics",
    "CalibrationResult",
    "SelectedPopulation",
]
