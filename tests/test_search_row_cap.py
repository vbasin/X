"""Train-fold row cap: single-base keeps it; sequence uses last-N contiguous bars."""

from __future__ import annotations

import numpy as np

from src.models.catalog import get_entry
from src.search.controller import (
    _effective_search_max_rows,
    _subsample_folds,
    _topology_n_bases,
)


def _fold(n: int = 20) -> dict:
    return {
        "X_train": np.arange(n * 3, dtype="float32").reshape(n, 3),
        "y_train": np.arange(n, dtype="int32"),
        "X_oof": np.arange(n * 3, dtype="float32").reshape(n, 3),
        "y_oof": np.arange(n, dtype="int32"),
    }


def test_topology_n_bases_meta_stack_and_none():
    assert _topology_n_bases({"search": {}}) == 0
    assert _topology_n_bases({
        "search": {"topology": {"mode": "meta_stack", "bases": ["attention_lstm"]}},
    }) == 1
    assert _topology_n_bases({
        "search": {"topology": {"mode": "output_meta", "bases": ["xgboost", "tcn"]}},
    }) == 2


def test_effective_cap_kept_for_single_base_dropped_for_multi():
    lstm = get_entry("attention_lstm")
    single = {"search": {"topology": {"mode": "meta_stack", "bases": ["attention_lstm"]}}}
    multi = {"search": {"topology": {"mode": "meta_stack", "bases": ["xgboost", "tcn"]}}}
    none = {"search": {}}
    assert _effective_search_max_rows(lstm, single) == 100_000
    assert _effective_search_max_rows(lstm, none) == 100_000
    assert _effective_search_max_rows(lstm, multi) is None


def test_subsample_random_preserves_oof_and_sorts():
    folds = [_fold(20)]
    out = _subsample_folds(folds, 8, seed=42, contiguous=False)
    assert len(out[0]["X_train"]) == 8
    assert len(out[0]["X_oof"]) == 20
    # sorted in time
    assert np.all(np.diff(out[0]["y_train"]) > 0)


def test_subsample_contiguous_is_last_n_bars():
    folds = [_fold(20)]
    out = _subsample_folds(folds, 8, contiguous=True)
    assert list(out[0]["y_train"]) == list(range(12, 20))
    assert len(out[0]["X_oof"]) == 20
