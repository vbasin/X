"""Validation: the shared walk-forward schedule.

Track C's ``eval_move_model.py`` and ``eval_target_model.py`` call
``folds_from_objective`` and apply each row with ``_fold_masks``. Track B
search calls ``apply_schedule_to_arrays`` when ``eval.walk_forward.mode`` is
``expanding_calendar``. Its other path is still the equal-count split in
``src/search/controller.py::_build_folds``.

It is NOT an interface layer. ``walk_forward.py`` is the real implementation.
The objective's ``eval`` block chooses the schedule for callers of
``folds_from_objective``.

Sample-uniqueness weighting for overlapping outcomes stays in
``src/evaluation/uniqueness.py``, where it already was. It is re-exported here
so a caller gets fold construction and overlap discounting from one import
without this package duplicating the implementation.
"""

from src.evaluation.uniqueness import (  # noqa: F401
    effective_n,
    label_uniqueness,
    weighted_tstat,
)
from src.validation.walk_forward import (  # noqa: F401
    MAX_FOLDS,
    apply_schedule_to_arrays,
    SPLITS,
    build_expanding_schedule,
    build_last_range_schedule,
    build_month_block_schedule,
    default_walk_forward,
    folds_from_objective,
    normalize_board_wf,
    overlay_from_board,
    probe_bar_seconds,
    resolve_purge_rows,
    schedule_from_eval,
)
