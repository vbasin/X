"""Track C persisted PnL uses the fill and the barrier, not the decision bar."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "features" / "eval_target_model.py"
_spec = importlib.util.spec_from_file_location("eval_target_model", _PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mod)


def test_entry_is_the_fill_and_exit_is_the_barrier():
    book = pd.DataFrame({
        "decision_price": [100.0, 100.0],
        "price": [100.25, 99.75],
        "y_reach": [1, 0],
        "tgt_level": [101.0, 102.0],
        "stop_px": [99.5, 99.0],
    })
    entry, exit_col = mod.execution_prices(book)
    assert entry == "price"
    assert exit_col == "_exit_px"
    assert book["_exit_px"].tolist() == [101.0, 99.0]
    assert "decision_price" not in (entry, exit_col)


def test_a_censored_row_is_still_charged_the_full_stop():
    book = pd.DataFrame({
        "price": [50.0],
        "y_reach": [0],
        "t_stop": [-1],
        "tgt_level": [51.0],
        "stop_px": [49.25],
    })
    mod.execution_prices(book)
    assert book["_exit_px"].iloc[0] == 49.25
