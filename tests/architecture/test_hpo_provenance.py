"""HPO provenance records the folds that were scored.

The study writer used to stamp ``last_range`` / ``n_folds=1`` and read
``y_test`` with a boolean ``or``, which raises on a NumPy array and ignores
every fold after the first. These tests pin the correction. They do not run
Optuna.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "search" / "hpo_model_search.py"
_spec = importlib.util.spec_from_file_location("hpo_model_search", _PATH)
hpo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(hpo)


def _fold(i: int, split: str, n_oof: int) -> dict:
    return {
        "fold": i,
        "y_oof": np.arange(n_oof),
        "y_test": np.arange(1),  # present and wrong; must not win over y_oof
        "fold_meta": {
            "fold": i,
            "split": split,
            "train": {"start": f"2025-0{i+1}-01", "end": f"2025-0{i+1}-28"},
            "oof": {"start": f"2025-0{i+2}-01", "end": f"2025-0{i+2}-28"},
        },
    }


def test_scored_rows_sums_y_oof_and_does_not_boolean_test_the_array():
    folds = [_fold(0, "expanding_equal_count", 4), _fold(1, "expanding_equal_count", 6)]
    assert hpo._scored_rows(folds) == 10
    with pytest.raises(ValueError):
        len(folds[0].get("y_oof") or [])


def test_scored_rows_falls_back_to_y_test_only_when_y_oof_is_absent():
    fold = {"y_test": np.zeros(3)}
    assert hpo._scored_rows([fold]) == 3
    assert hpo._scored_rows([{}]) == 0


def test_validation_records_every_scored_window_and_its_split():
    folds = [_fold(0, "expanding_equal_count", 2), _fold(1, "expanding_equal_count", 2)]
    validation = hpo._validation_from_folds(folds)
    assert validation["split"] == "expanding_equal_count"
    assert validation["n_folds"] == 2
    assert [w["fold"] for w in validation["fold_windows"]] == [0, 1]
    assert validation["fold_windows"][1]["oof"]["start"] == "2025-03-01"
    assert "last_range" not in validation["split"]


def test_windows_come_from_timestamps_when_a_fold_has_no_metadata():
    fold = {
        "split": "expanding_calendar",
        "y_oof": np.zeros(2),
        "ts_train": np.array(["2025-05-01T14:30:00", "2025-06-15T14:30:00"],
                             dtype="datetime64[ns]"),
        "ts_oof": np.array(["2025-07-01T14:30:00", "2025-07-20T14:30:00"],
                           dtype="datetime64[ns]"),
    }
    window = hpo._fold_window(fold, 0)
    assert window["split"] == "expanding_calendar"
    assert window["train"]["start"].startswith("2025-05-01")
    assert window["oof"]["end"].startswith("2025-07-20")
