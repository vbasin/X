"""One walk-forward implementation for both tracks (src/validation/).

Track B had src/search/walk_forward_calendar.py; each Track C evaluator had its
own inline _folds() month loop. Same job, different answers, no way to compare a
Track B result with a Track C one. These tests pin that there is now exactly one
implementation and that the same config yields the same fold boundaries whoever
calls it.

DB-free: pure date arithmetic.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.validation import (
    build_expanding_schedule,
    build_last_range_schedule,
    build_month_block_schedule,
    folds_from_objective,
)

REPO = Path(__file__).resolve().parents[2]
_SESSIONS = pd.date_range("2025-05-01", "2026-02-28", freq="D")

#: The row shape every schedule must emit, so a caller, manifest.validation and
#: fold_metrics.json cannot tell which schedule produced a fold.
_ROW_KEYS = {"fold", "split", "train", "oof", "val", "clipped"}


def test_every_schedule_emits_one_row_shape():
    schedules = [
        build_month_block_schedule(_SESSIONS, test_months=2, min_train_months=4),
        build_last_range_schedule(_SESSIONS),
        build_expanding_schedule(start="2025-05-01", stop="2026-02-28"),
    ]
    for rows in schedules:
        assert rows, "schedule produced no folds"
        for row in rows:
            assert _ROW_KEYS <= set(row), row
            assert set(row["train"]) >= {"start", "end"}
            assert set(row["oof"]) >= {"start", "end"}
            # Training never reaches into the test window.
            assert row["train"]["end"] < row["oof"]["start"], row


def test_the_same_config_gives_both_tracks_the_same_boundaries():
    """The point of the consolidation. A Track C evaluator (sessions + CLI
    overrides) and a Track B caller (the objective's own eval block) must land on
    identical folds when the config is identical."""
    objective = {"eval": {"split": "expanding_month_blocks",
                          "test_months": 2, "min_train_months": 4}}
    # Track B style: the objective decides.
    from_objective = folds_from_objective(objective, sessions=_SESSIONS)
    # Track C style: the runtime overrides an objective that names no schedule.
    from_overrides = folds_from_objective(
        {}, sessions=_SESSIONS,
        overrides={"test_months": 2, "min_train_months": 4})
    assert from_objective == from_overrides
    # ...and both equal the direct builder, so no caller has a private variant.
    assert from_objective == build_month_block_schedule(
        _SESSIONS, test_months=2, min_train_months=4)


def test_month_block_schedule_expands_and_advances_by_the_test_width():
    rows = build_month_block_schedule(_SESSIONS, test_months=2, min_train_months=4)
    assert [r["fold"] for r in rows] == list(range(len(rows)))
    first_train_start = rows[0]["train"]["start"]
    for i, row in enumerate(rows):
        # Expanding, not sliding: every fold trains from the first session.
        assert row["train"]["start"] == first_train_start
        assert row["split"] == "expanding_month_blocks"
        if i:
            # The test block advances by its own width, so windows do not overlap.
            assert row["oof"]["start"] > rows[i - 1]["oof"]["end"]


def test_too_few_months_yields_no_folds_rather_than_a_short_one():
    short = pd.date_range("2025-05-01", "2025-08-31", freq="D")
    assert build_month_block_schedule(short, test_months=2, min_train_months=4) == []
    assert build_last_range_schedule(pd.date_range("2025-05-01", "2025-05-20")) == []


def test_an_unknown_split_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown eval.split"):
        folds_from_objective({"eval": {"split": "whatever"}}, sessions=_SESSIONS)


def test_calendar_split_refuses_to_guess_a_window():
    with pytest.raises(ValueError, match="needs eval.start and eval.stop"):
        folds_from_objective({"eval": {"split": "expanding_calendar"}},
                             sessions=_SESSIONS)


def test_runtime_overrides_win_over_the_objective():
    """So an operator can change the schedule without editing the YAML, and the
    RESOLVED values are what the manifest records."""
    objective = {"eval": {"test_months": 2, "min_train_months": 4}}
    wider = folds_from_objective(objective, sessions=_SESSIONS,
                                overrides={"test_months": 3})
    assert all(r["split"] == "expanding_month_blocks" for r in wider)
    assert wider == build_month_block_schedule(
        _SESSIONS, test_months=3, min_train_months=4)


def test_no_evaluator_carries_its_own_fold_loop():
    """The regression this consolidation exists to prevent."""
    out = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E", r"^def _folds\(", "src", "scripts"],
        cwd=REPO, capture_output=True, text=True).stdout.strip()
    assert out == "", f"a private fold builder reappeared:\n{out}"
    # Both Track C evaluators must reach the shared entry point.
    for script in ("scripts/features/eval_move_model.py",
                   "scripts/features/eval_target_model.py"):
        text = (REPO / script).read_text()
        assert "folds_from_objective" in text, script
        assert "_fold_masks(df, fold)" in text, script
        assert "to_period(\"M\")" not in text, (
            f"{script} still derives its own month blocks")


def test_uniqueness_weighting_is_reexported_not_reimplemented():
    """Overlap discounting stays in src/evaluation/uniqueness.py; src/validation
    re-exports it so one import gives folds + weighting without a second copy."""
    import src.evaluation.uniqueness as canonical
    import src.validation as shared

    assert shared.label_uniqueness is canonical.label_uniqueness
    assert shared.effective_n is canonical.effective_n
