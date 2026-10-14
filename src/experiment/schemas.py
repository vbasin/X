"""Canonical schemas for the persisted-result artifact classes named in
docs/EXPERIMENTS.md section 2.1: aggregate metrics, fold metrics,
calibration, selected populations, and the out-of-fold prediction column
contract. Predictions/trades/economics already have their own homes
(TradeLedger in src/evaluation/economic.py covers trades +
economics + cost assumptions together, by design -- see that module's
docstring for why those three are one artifact, not three).

Artifacts in that table that deliberately have NO schema here, and why:
``manifest.json`` (RunManifest is its own module), ``report.md`` (prose),
``SHAP/*`` (src/evaluation/shap_explain.py already produces a good,
stable three-file artifact -- re-specifying it would be exactly the
recreate-what-exists failure this pass is correcting), and trained model
checkpoints (owned by the trainer that wrote them).

These are lightweight, validated dataclasses -- not a DataFrame schema
enforcement library. They exist so "fold metrics" (for example) means the
same shape everywhere a canonical run persists them, rather than each
script inventing its own dict shape (the exact problem docs/EXPERIMENTS.md
section 2 exists to prevent). None of this is wired into any existing
script yet -- see docs/MIGRATION_MANIFEST.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


OOF_PREDICTION_COLUMNS: tuple[str, ...] = (
    "ts",
    "direction",
    "confidence",
    "direction_true",
)
"""The out-of-fold prediction frame's column contract.

NOT invented here: this is the column set ``src/backtest/runner.py`` already
writes and ``src/backtest/archetypes.py:122`` already reads back
(``pd.read_parquet(oof_path, columns=[...])``), documented at
``src/backtest/runner.py:14`` as "comparable with model OOF frames". It is
named here so a canonical run has one stated contract to satisfy rather than
an undocumented convention that happens to hold.
"""


def validate_oof_columns(columns) -> None:
    """Raise if an OOF frame is missing a contract column.

    Extra columns are allowed -- real frames carry model-specific scores
    alongside the contract set, and rejecting them would break existing
    producers for no benefit.
    """
    missing = [c for c in OOF_PREDICTION_COLUMNS if c not in set(columns)]
    if missing:
        raise ValueError(
            f"OOF frame is missing required column(s) {missing}; the contract "
            f"is {list(OOF_PREDICTION_COLUMNS)} (src/backtest/runner.py:14)"
        )


@dataclass(frozen=True)
class RunMetrics:
    """A run's aggregate predictive result -- the ``metrics.json`` artifact.

    The aggregate counterpart to ``FoldMetrics``. Both are required when a
    run is fold-validated, because an aggregate alone hides the per-fold
    dispersion that both tracks' eval scripts already print (and which is
    the only way to see a run that is carried by one lucky fold).

    ``n_samples`` is mandatory: a metric without the population it was
    computed over cannot be compared against another run's.
    """

    n_samples: int
    metrics: Mapping[str, float]
    n_positive: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if not self.metrics:
            raise ValueError("metrics must record at least one value")
        if self.n_positive is not None and not (0 <= self.n_positive <= self.n_samples):
            raise ValueError("n_positive must be within [0, n_samples]")


@dataclass(frozen=True)
class FoldMetrics:
    """One walk-forward fold's result.

    Mirrors the per-fold window Track B
    (``src/validation/walk_forward.py``) and Track C
    (``eval_move_model.py`` / ``eval_target_model.py``) record. There is no
    ``src/search/walk_forward_calendar.py``.
    """

    fold_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError("n_train/n_test must be >= 0")
        if not self.metrics:
            raise ValueError("metrics must record at least one value")


@dataclass(frozen=True)
class CalibrationResult:
    """Reliability-diagram-style calibration summary for one score->outcome
    relationship. Deliberately minimal: bucket-level predicted vs. observed
    rate plus an overall Brier score, sufficient to answer "is this score
    usable as a probability" without prescribing a specific calibration
    method (isotonic vs. Platt) -- that choice belongs to whatever
    Calibrator implementation eventually produces this.
    """

    method: str  # e.g. "isotonic", "platt", "none" (uncalibrated, raw score)
    n_samples: int
    bucket_edges: tuple[float, ...]
    predicted_rate: tuple[float, ...]
    observed_rate: tuple[float, ...]
    brier_score: float

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if len(self.predicted_rate) != len(self.observed_rate):
            raise ValueError("predicted_rate and observed_rate must be the same length")
        if len(self.bucket_edges) != len(self.predicted_rate) + 1:
            raise ValueError(
                "bucket_edges must have exactly one more entry than "
                "predicted_rate/observed_rate (N buckets -> N+1 edges)"
            )
        if not (0.0 <= self.brier_score <= 1.0):
            raise ValueError("brier_score must be in [0, 1]")


@dataclass(frozen=True)
class SelectedPopulation:
    """Population statistics for a Decision Policy's selection/gate step.

    This is the artifact that, if persisted for both sides of a comparison,
    closes the population-shift gap described in docs/ARCHITECTURE.md
    section 1.7 and docs/EXPERIMENTS.md section 3 point 2 -- two runs'
    ``SelectedPopulation`` can be diffed to check ``overlap_row_ids`` before
    trusting a downstream metric comparison.
    """

    n_raw: int
    n_selected: int
    selection_rule: str  # human-readable description, e.g. "p_move >= tau_0.95"
    overlap_row_ids: frozenset[str] | None = field(default=None)
    """Row identifiers shared with a baseline population being compared
    against, if known. ``None`` means "not computed", not "zero overlap" --
    callers must not conflate the two."""

    def __post_init__(self) -> None:
        if self.n_raw < 0 or self.n_selected < 0:
            raise ValueError("n_raw/n_selected must be >= 0")
        if self.n_selected > self.n_raw:
            raise ValueError("n_selected cannot exceed n_raw")

    @property
    def selection_rate(self) -> float:
        return self.n_selected / self.n_raw if self.n_raw else float("nan")
