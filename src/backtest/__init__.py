"""User-defined strategy backtesting — Phase 23 experiment entry path 3.

Public API::

    from src.backtest import load_strategy_spec, run_backtest, run_strategy
"""

from src.backtest.spec import StrategySpec, load_strategy_spec
from src.backtest.engine import BacktestResult, run_backtest
from src.backtest.runner import run_strategy

__all__ = [
    "StrategySpec",
    "load_strategy_spec",
    "BacktestResult",
    "run_backtest",
    "run_strategy",
]
