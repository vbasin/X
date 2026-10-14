"""src.evaluation — frozen evaluation contract + trade-KPI metric suite.

Public API
----------
evaluate_model                 — existing 3-class detection evaluator (re-exported)
canonical_trade_kpi            — trade-economics KPI scoring on canonical predictions
compute_diagnostics            — full report: confusion matrices, per-class F1, tau sweep
EvalContract                   — frozen hold-out contract (hash-pinned, version-stamped)
ContractHashMismatch           — raised when data hash doesn't match the pinned hold-out
build_contract                 — factory: hash parquets → new EvalContract
pin_hold_out_from_parquet_dir  — convenience wrapper for date-range hold-outs
record_result                  — append a scored candidate to the leaderboard JSONL
load_leaderboard               — read all leaderboard entries for a contract
get_leaders                    — top-k candidates by primary precision
champion                       — current leaderboard leader
build_episodes                 — success/failure scenario rows from an OOF frame + context
record_episodes                — persist episodes to the outcome store
load_episodes                  — read recorded episodes
latest_outcome_reports         — newest mined outcome-pattern reports (for agent context)
mine_patterns                  — deterministic win/loss/missed pattern mining
analyze_and_write              — mine + persist an outcome report for a run
"""

from src.training.evaluate import evaluate_model           # noqa: F401
# economic.py is THE execution/cost authority for this repository (moved here
# from the retired src/canonical/evaluation/ tree, ADR-136 consolidation).
from src.evaluation.economic import (                      # noqa: F401
    EconomicAssumptions,
    Leg,
    OrderType,
    Quote,
    Trade,
    TradeLedger,
    TradeResult,
    assumptions_record,
    compute_trade_ledger,
    evaluate_cost_sensitivity,
    execution_price,
    load_assumptions,
    zero_cost,
)
from src.evaluation._kpi import canonical_trade_kpi, compute_diagnostics  # noqa: F401
from src.evaluation.contract import (                      # noqa: F401
    EvalContract,
    ContractHashMismatch,
    build_contract,
    pin_hold_out_from_parquet_dir,
)
from src.evaluation.leaderboard import (                   # noqa: F401
    record_result,
    load_leaderboard,
    get_leaders,
    champion,
)
from src.evaluation.episodes import (                      # noqa: F401
    build_episodes,
    record_episodes,
    load_episodes,
    latest_outcome_reports,
)
from src.evaluation.patterns import (                      # noqa: F401
    mine_patterns,
    analyze_run,
    analyze_and_write,
)
