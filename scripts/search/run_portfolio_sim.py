#!/usr/bin/env python3
"""Score an OOF prediction frame through the path-realised account simulator.

Unlike the label-implied KPI (:func:`src.evaluation._kpi.label_implied_pnl`) and the 5s
bar backtester, this prices every trade off the 1-second high/low path and runs it
through an account: margin, a concurrent-position cap, a session loss halt, commissions
and adverse stop slippage. Labels are never read.

Three things it is for:

1. **Bias measurement** — ``--compare-tie-policies`` re-prices identical signals under
   ``close_only`` (what the current leaderboard can see) and ``stop_first`` (honest
   floor). The gap is the optimism in every published KPI.
2. **Operating point** — ``--sweep-tau`` prints the frequency/economics frontier so the
   trade rate is chosen off a curve instead of forced by a quota.
3. **Capital sizing** — ``--target-usd-per-month`` reports the contracts and account the
   measured edge implies.

Examples::

    # honest re-score of a search run, ADR-047 open-session policy
    python scripts/search/run_portfolio_sim.py --oof <run_id> --objective nq-50pt-xgb-unique \\
        --tau 0.65 --cooldown-seconds 600 --trade-end 10:00

    # how much of the reported expectancy is close-only optimism?
    python scripts/search/run_portfolio_sim.py --oof <run_id> --compare-tie-policies

    # what account does $50k/month need at this edge?
    python scripts/search/run_portfolio_sim.py --oof <run_id> --target-usd-per-month 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtest.portfolio_sim import (  # noqa: E402
    SimConfig, compare_tie_policies, required_scaling, simulate, sweep_tau,
)
from src.common.data_paths import parquet_path1s_dir, search_dir  # noqa: E402

logger = logging.getLogger("vtech.scripts.run_portfolio_sim")


def load_path(start: str, end: str, path_dir: Path) -> pd.DataFrame:
    """Concatenate ``path_*.parquet`` files whose date falls in ``[start, end]``."""
    files = sorted(f for f in path_dir.glob("path_*.parquet")
                   if start <= f.stem.removeprefix("path_") <= end)
    if not files:
        raise FileNotFoundError(
            f"no path parquets in {path_dir} covering {start}..{end} — build them with "
            f"scripts/data/build_path1s.py --start {start} --end {end}")
    logger.info("loading %d path days from %s", len(files), path_dir)
    return pd.concat([pd.read_parquet(f) for f in files]).sort_index()


def resolve_oof(source: str, objective: str | None) -> Path:
    """Accept a parquet path, a run directory, or a bare run id under the objective."""
    p = Path(source)
    if p.is_file():
        return p
    for cand in ([p] if p.is_dir() else []) + (
            [search_dir() / objective / source] if objective else []):
        if cand.is_dir():
            hits = sorted(cand.glob("oof_*.parquet"))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise ValueError(f"{cand} holds {len(hits)} oof_*.parquet — name one")
    raise FileNotFoundError(f"cannot resolve OOF source {source!r}")


def _config_from_args(a: argparse.Namespace) -> SimConfig:
    return SimConfig(
        target_points=a.target_points, stop_points=a.stop_points,
        max_hold_seconds=a.max_hold_seconds,
        trade_start=a.trade_start, trade_end=a.trade_end, flatten_at=a.flatten_at,
        point_value_usd=a.point_value, tick_size=a.tick_size,
        commission_per_side_usd=a.commission, entry_slippage_ticks=a.entry_slippage_ticks,
        stop_slippage_ticks=a.stop_slippage_ticks,
        starting_equity_usd=a.equity, margin_per_contract_usd=a.margin,
        contracts_per_trade=a.contracts, max_concurrent_positions=a.max_positions,
        daily_loss_limit_frac=a.daily_loss_frac,
        allow_opposing_positions=a.allow_opposing,
        tau=a.tau, cooldown_seconds=a.cooldown_seconds, tie_policy=a.tie_policy,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True, help="oof_*.parquet, run dir, or run id")
    ap.add_argument("--objective", default=None, help="objective id (to resolve a run id)")
    ap.add_argument("--path-dir", default=None, help="default: parquet_path1s_dir()")
    ap.add_argument("--start", default=None, help="first session (default: OOF min ts)")
    ap.add_argument("--end", default=None, help="last session (default: OOF max ts)")

    g = ap.add_argument_group("trade geometry")
    g.add_argument("--target-points", type=float, default=50.0)
    g.add_argument("--stop-points", type=float, default=6.0)
    g.add_argument("--max-hold-seconds", type=int, default=600)
    g.add_argument("--trade-start", default="08:30")
    g.add_argument("--trade-end", default="10:00")
    g.add_argument("--flatten-at", default="14:45")

    g = ap.add_argument_group("contract economics")
    g.add_argument("--point-value", type=float, default=20.0)
    g.add_argument("--tick-size", type=float, default=0.25)
    g.add_argument("--commission", type=float, default=2.50, help="USD per side per contract")
    g.add_argument("--entry-slippage-ticks", type=float, default=1.0)
    g.add_argument("--stop-slippage-ticks", type=float, default=2.0)

    g = ap.add_argument_group("account & risk")
    g.add_argument("--equity", type=float, default=5_000.0)
    g.add_argument("--margin", type=float, default=1_000.0, help="USD per contract")
    g.add_argument("--contracts", type=int, default=1)
    g.add_argument("--max-positions", type=int, default=5)
    g.add_argument("--daily-loss-frac", type=float, default=0.5)
    g.add_argument("--allow-opposing", action="store_true")

    g = ap.add_argument_group("gating & fidelity")
    g.add_argument("--tau", type=float, default=0.65)
    g.add_argument("--cooldown-seconds", type=int, default=600)
    g.add_argument("--tie-policy", default="stop_first",
                   choices=("stop_first", "target_first", "close_only"))

    g = ap.add_argument_group("reports")
    g.add_argument("--sweep-tau", action="store_true")
    g.add_argument("--compare-tie-policies", action="store_true")
    g.add_argument("--target-usd-per-month", type=float, default=None)
    g.add_argument("--out-dir", default=None, help="write trades/daily/kpis here")

    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    oof_path = resolve_oof(a.oof, a.objective)
    signals = pd.read_parquet(oof_path)
    signals = signals[pd.to_datetime(signals["ts"], utc=True, errors="coerce").notna()]
    ts = pd.to_datetime(signals["ts"], utc=True)
    start = a.start or str(ts.min().tz_convert("America/Chicago").date())
    end = a.end or str(ts.max().tz_convert("America/Chicago").date())
    logger.info("OOF %s: %d signals, %s → %s", oof_path.name, len(signals), start, end)

    path = load_path(start, end, Path(a.path_dir) if a.path_dir else parquet_path1s_dir())
    cfg = _config_from_args(a)

    result = simulate(signals, path, cfg)
    print("\n── path-realised account simulation ──")
    print(result.summary())
    print(json.dumps(result.kpis, indent=2, default=str))

    if a.compare_tie_policies:
        print("\n── tie-policy sensitivity (close_only = what the leaderboard sees) ──")
        print(compare_tie_policies(signals, path, cfg).to_string(index=False))

    if a.sweep_tau:
        print("\n── τ frontier (pick the operating point here, not with a quota) ──")
        print(sweep_tau(signals, path, cfg).to_string(index=False))

    if a.target_usd_per_month:
        print(f"\n── scaling to ${a.target_usd_per_month:,.0f}/month ──")
        print(json.dumps(required_scaling(result.kpis, a.target_usd_per_month, cfg),
                         indent=2))

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.trades.to_parquet(out / "sim_trades.parquet", index=False)
        result.daily.to_parquet(out / "sim_daily.parquet", index=False)
        (out / "sim_kpis.json").write_text(json.dumps(
            {"kpis": result.kpis, "config": asdict(cfg), "oof": str(oof_path)},
            indent=2, default=str))
        logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
