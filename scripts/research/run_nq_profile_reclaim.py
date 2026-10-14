#!/usr/bin/env python3
"""Chronological TBBO replay of the experimental confluent-profile reclaim rule.

The script only reads local print tapes.  It creates an immutable artifact,
selects one policy on May 2025–March 2026 discovery, and opens April–June 2026
only after persisting the selected configuration.  It never reads the Jul–Sep
2026 reserve.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.nq_profile_reclaim_quotes import ExecutionConfig, replay_session
from src.common.data_paths import logical_path
from src.common.run_layout import new_run_ts, run_dir
from src.common.vm_config import get_vm_config
from src.evaluation.economic import load_assumptions
from src.experiment.manifest import CodeIdentity, DatasetRef, RunManifest, ValidationDesign
from src.features.nq_noise_area import eligible_sessions
from src.features.nq_profile_reclaim import ReclaimConfig, cash_minute_state, prior_bounds, reclaim_signals
from src.features.tape import list_tape_files, tape_levels
from src.training.auction_levels import ProfileConfig

OBJECTIVE = "nq-tbbo-profile-reclaim"
SEED = 20260924
DISCOVERY_END = "2026-03-31"
VALIDATION_START, VALIDATION_END = "2026-04-01", "2026-06-30"
SOURCE_FILES = [
    "scripts/research/run_nq_profile_reclaim.py",
    "src/features/nq_profile_reclaim.py",
    "src/backtest/nq_profile_reclaim_quotes.py",
    "src/features/tape.py",
    "src/features/session_profile.py",
    "src/features/cash_session_profile.py",
    "src/common/contract_calendar.py",
    "config/execution_costs.yaml",
    "docs/research/NQ_TBBO_PROFILE_RECLAIM_PROTOCOL.md",
]
POLICIES = [
    ReclaimConfig(min_bound_members=members, sweep_points=sweep, large_print_size=size)
    for members, sweep, size in itertools.product([2, 3], [1.0, 2.0], [20, 50])
]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cash_slice(tape: pd.DataFrame) -> pd.DataFrame:
    local = tape.index.tz_convert("America/Chicago")
    minute = local.hour * 60 + local.minute
    return tape.loc[(minute >= 510) & (minute < 900)]


def complete_cash_tape(tape: pd.DataFrame) -> bool:
    """Require prints at both ends of a normal 390-minute cash session."""
    cash = cash_slice(tape)
    if cash.empty:
        return False
    local = cash.index.tz_convert("America/Chicago")
    first, last = local[0], local[-1]
    return (first.hour, first.minute) <= (8, 31) and (last.hour, last.minute) >= (14, 59)


def source_inventory(files: list[Path]) -> list[dict[str, Any]]:
    inventory = []
    for n, path in enumerate(files, 1):
        tape = pd.read_parquet(path, columns=["price", "size", "bid_px", "ask_px"])
        local = tape.index.tz_convert("America/Chicago")
        cash = (local.hour * 60 + local.minute >= 510) & (local.hour * 60 + local.minute < 900)
        inventory.append(dict(session=path.stem[5:], path=logical_path(path), sha256=sha256(path),
                              rows=len(tape), cash_rows=int(cash.sum()),
                              first_ts=str(tape.index.min()), last_ts=str(tape.index.max())))
        if n % 100 == 0 or n == len(files):
            k = int(20 * n / len(files))
            print(f"[{'█' * k}{'░' * (20-k)}] hashed {n}/{len(files)} tape files", flush=True)
    return inventory


def build_levels(files: list[Path], eligible: set[str], root: Path) -> pd.DataFrame:
    """Build exact cash profile levels from each local tape, never cache trust."""
    rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for n, path in enumerate(files, 1):
        session = path.stem[5:]
        tape = pd.read_parquet(path)
        full = complete_cash_tape(tape)
        eligible_day = session in eligible
        cash = cash_slice(tape)
        levels = tape_levels(cash, pd.Timestamp(session), ProfileConfig()) if full and eligible_day else None
        quality.append(dict(session=session, eligible_calendar=eligible_day, full_cash_tape=full,
                            cash_prints=len(cash), usable=levels is not None))
        if levels is not None:
            rows.append(dict(session=pd.Timestamp(session), lo=levels.lo, val=levels.val, poc=levels.poc,
                             vah=levels.vah, hi=levels.hi))
        if n % 50 == 0 or n == len(files):
            k = int(20 * n / len(files))
            print(f"[{'█' * k}{'░' * (20-k)}] exact profiles {n}/{len(files)}", flush=True)
    pd.DataFrame(quality).to_parquet(root / "profile_input_quality.parquet", index=False)
    result = pd.DataFrame(rows).sort_values("session", kind="mergesort").reset_index(drop=True)
    result.to_parquet(root / "exact_prior_levels.parquet", index=False)
    return result


def _replay_day(task: tuple[str, pd.DataFrame, list[ReclaimConfig], ExecutionConfig]) -> dict[int, pd.DataFrame]:
    """Independent replay task; historical bounds are strictly date-filtered."""
    path_s, levels, policies, execution = task
    path = Path(path_s)
    session = path.stem[5:]
    tape = pd.read_parquet(path)
    result: dict[int, pd.DataFrame] = {}
    if not complete_cash_tape(tape):
        return result
    bounds = prior_bounds(levels, date.fromisoformat(session), policies[0])
    if not bounds:
        return result
    minute_by_size = {
        size: cash_minute_state(tape, large_print_size=size)
        for size in sorted({policy.large_print_size for policy in policies})
    }
    for i, policy in enumerate(policies):
        signals = reclaim_signals(minute_by_size[policy.large_print_size], bounds, policy)
        if signals.empty:
            continue
        ledger = replay_session(tape, signals, config=execution)
        if not ledger.empty:
            ledger["session"] = session
            ledger["policy_id"] = i
            result[i] = ledger
    return result


def replay_period(files: list[Path], levels: pd.DataFrame, policies: list[ReclaimConfig],
                  execution: ExecutionConfig, root: Path, label: str) -> dict[int, pd.DataFrame]:
    """Replay independent sessions across the configured worker capacity."""
    books: dict[int, list[pd.DataFrame]] = {i: [] for i in range(len(policies))}
    days = [p for p in files if p.stem[5:] in set(levels.session.dt.strftime("%Y-%m-%d"))]
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.0)
    tasks = [(str(path), levels, policies, execution) for path in days]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_replay_day, task) for task in tasks]
        for n, job in enumerate(as_completed(jobs), 1):
            for policy_id, ledger in job.result().items():
                books[policy_id].append(ledger)
            if n % 25 == 0 or n == len(days):
                k = int(20 * n / len(days))
                print(f"[{'█' * k}{'░' * (20-k)}] {label} replay {n}/{len(days)} sessions", flush=True)
    out: dict[int, pd.DataFrame] = {}
    for i, chunks in books.items():
        out[i] = (pd.concat(chunks, ignore_index=True).sort_values("entry_ts", kind="mergesort")
                  if chunks else pd.DataFrame(columns=["session", "policy_id"]))
    return out


def _daily_mark_dd(trades: pd.DataFrame, execution: ExecutionConfig) -> float:
    """Conservative day DD from each quote-mark path's recorded extrema.

    The full ordered quote audit follows any passed candidate.  Until then this
    preserves the possible within-trade high-to-low range rather than reporting
    a closed-trade-only drawdown as a risk limit.
    """
    if trades.empty:
        return 0.0
    maximum = 0.0
    for _, group in trades[trades.valid].groupby("session", sort=True):
        equity = peak = 0.0
        for row in group.sort_values("entry_ts").itertuples():
            min_mark = equity - row.mae_points * 20.0 - execution.slippage_points * 20.0 - execution.commission
            max_mark = equity + row.mfe_points * 20.0 - execution.slippage_points * 20.0 - execution.commission
            maximum = max(maximum, peak - min_mark, max_mark - min_mark)
            peak = max(peak, max_mark)
            equity += row.net_usd
            peak = max(peak, equity)
    return float(maximum)


def evaluate(trades: pd.DataFrame, sessions: list[str], execution: ExecutionConfig,
             *, bootstrap: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    valid = trades[trades.valid] if len(trades) and "valid" in trades else trades
    daily = pd.DataFrame(index=pd.Index(sessions, name="session"))
    if len(valid):
        daily["net_usd"] = valid.groupby("session").net_usd.sum()
    else:
        daily["net_usd"] = 0.0
    daily["net_usd"] = daily.net_usd.fillna(0.0)
    daily["equity_usd"] = daily.net_usd.cumsum()
    daily["drawdown_usd"] = daily.equity_usd - daily.equity_usd.cummax().clip(lower=0.0)
    pnl = valid.net_usd.to_numpy(float) if len(valid) else np.empty(0)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    result: dict[str, Any] = dict(
        n_sessions=len(sessions), n_trades=len(pnl), unresolved=int(len(trades) - len(valid)),
        net_usd=float(pnl.sum()), mean_daily_usd=float(daily.net_usd.mean()),
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else None,
        win_rate=float((pnl > 0).mean()) if len(pnl) else 0.0,
        loss_count=int(len(losses)), total_loss_usd=float(-losses.sum()),
        avg_trade_usd=float(pnl.mean()) if len(pnl) else 0.0,
        max_closed_daily_drawdown_usd=float(-daily.drawdown_usd.min()),
        conservative_quote_mark_daily_drawdown_usd=_daily_mark_dd(trades, execution),
        worst_trade_usd=float(pnl.min()) if len(pnl) else 0.0,
        worst_day_usd=float(daily.net_usd.min()),
    )
    if bootstrap:
        values = daily.net_usd.to_numpy(float)
        rng = np.random.default_rng(SEED)
        n = len(values)
        indices = (rng.integers(0, n, size=(5000, (n + 4) // 5, 1)) + np.arange(5)) % n
        means = values[indices.reshape(5000, -1)[:, :n]].mean(axis=1)
        result["daily_mean_95ci"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result, daily


def candidate_pass(metrics: dict[str, Any], stress: dict[str, Any]) -> bool:
    return (metrics["n_trades"] >= 20 and metrics["unresolved"] == 0
            and metrics["mean_daily_usd"] > 0 and stress["net_usd"] > 0
            and metrics["worst_trade_usd"] >= -500.0
            and metrics["conservative_quote_mark_daily_drawdown_usd"] <= 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=VALIDATION_END,
                        help="Last non-reserve session to make available (default %(default)s)")
    args = parser.parse_args()
    if args.end > VALIDATION_END:
        raise ValueError("this runner refuses post-June 2026 reserve data")
    started = time.monotonic()
    root = run_dir(OBJECTIVE, new_run_ts())
    root.mkdir(parents=True)
    print(f"RUN_DIR={root}", flush=True)
    source_hashes: dict[str, str] = {}
    for source in SOURCE_FILES:
        payload = Path(source).read_bytes()
        source_hashes[source] = hashlib.sha256(payload).hexdigest()
        target = root / "code" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    dump(root / "source_hashes.json", source_hashes)
    allowed = set(eligible_sessions("2025-05-01", args.end))
    files = [path for path in list_tape_files() if "2025-05-01" <= path.stem[5:] <= args.end
             and path.stem[5:] in allowed]
    if not files:
        raise ValueError("no eligible local NQ tapes")
    inventory = source_inventory(files)
    dump(root / "input_inventory.json", inventory)
    inventory_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    levels = build_levels(files, allowed, root)
    session_days = [p.stem[5:] for p in files]
    discovery_files = [p for p in files if p.stem[5:] <= DISCOVERY_END]
    validation_files = [p for p in files if VALIDATION_START <= p.stem[5:] <= VALIDATION_END]
    discovery_days = [p.stem[5:] for p in discovery_files]
    validation_days = [p.stem[5:] for p in validation_files]
    execution = ExecutionConfig(commission=load_assumptions("config/execution_costs.yaml", "NQ").commission_per_trade)
    protocol = dict(objective=OBJECTIVE, discovery=["2025-05-01", DISCOVERY_END],
                    validation=[VALIDATION_START, VALIDATION_END], reserve=["2026-07-01", "2026-09-22"],
                    policies=[asdict(p) for p in POLICIES], execution=asdict(execution),
                    source_contract="exact local NQ lead-out-right tape; same-contract prior profile only",
                    selection="max discovery mean daily dollars among safety-qualified candidates",
                    seed=SEED, holdout_read=False)
    dump(root / "protocol.json", protocol)
    discovery = replay_period(discovery_files, levels, POLICIES, execution, root, "discovery")
    stress_execution = ExecutionConfig(slippage_points=1.0)
    discovery_stress = replay_period(discovery_files, levels, POLICIES, stress_execution, root, "discovery 1pt stress")
    trial_rows: list[dict[str, Any]] = []
    for i, policy in enumerate(POLICIES):
        stats, daily = evaluate(discovery[i], discovery_days, execution, bootstrap=True)
        stress, _ = evaluate(discovery_stress[i], discovery_days, stress_execution)
        record = dict(policy_id=i, policy=asdict(policy), metrics=stats, stress_1pt=stress,
                      qualifies=candidate_pass(stats, stress))
        trial_rows.append(record)
        dest = root / "discovery" / f"policy_{i}"
        dest.mkdir(parents=True)
        discovery[i].to_parquet(dest / "trade_ledger.parquet", index=False)
        discovery_stress[i].to_parquet(dest / "stress_1pt_trade_ledger.parquet", index=False)
        daily.to_csv(dest / "daily.csv")
        dump(dest / "metrics.json", record)
    dump(root / "discovery_trials.json", trial_rows)
    candidates = [r for r in trial_rows if r["qualifies"]]
    selected = max(candidates or trial_rows, key=lambda row: row["metrics"]["mean_daily_usd"])
    frozen = dict(selected=selected, selection_passed=bool(candidates),
                  frozen_before_validation=True, selection_note=("safety-qualified" if candidates else "exploratory only; reserve locked"))
    dump(root / "frozen_policy.json", frozen)
    print("FROZEN", json.dumps(frozen), flush=True)
    # The one frozen policy is the only configuration replayed in historical validation.
    policy = POLICIES[selected["policy_id"]]
    summaries: dict[str, Any] = {}
    for name, execution_scenario in [("base", execution), ("stress_1pt", stress_execution)]:
        book = replay_period(validation_files, levels, [policy], execution_scenario, root, f"validation {name}")[0]
        stats, daily = evaluate(book, validation_days, execution_scenario, bootstrap=True)
        stats["yearly_net_usd"] = daily.groupby(daily.index.str[:4]).net_usd.sum().to_dict()
        dest = root / "validation" / name
        dest.mkdir(parents=True)
        book.to_parquet(dest / "trade_ledger.parquet", index=False)
        daily.to_csv(dest / "daily.csv")
        dump(dest / "metrics.json", stats)
        summaries[name] = stats
    base, stress = summaries["base"], summaries["stress_1pt"]
    gate = dict(
        discovery_selection_passed=bool(candidates), validation_resolved=base["unresolved"] == 0 and stress["unresolved"] == 0,
        validation_base_positive=base["net_usd"] > 0, validation_stress_positive=stress["net_usd"] > 0,
        validation_confidence=base["daily_mean_95ci"][0] > 0,
        target_1000_per_eligible_day=base["mean_daily_usd"] >= 1000.0,
        trade_loss_limit=base["worst_trade_usd"] >= -500.0,
        daily_drawdown_limit=base["conservative_quote_mark_daily_drawdown_usd"] <= 1000.0,
        holdout_read=False,
    )
    gate["passed"] = all(v for k, v in gate.items() if k not in {"holdout_read"})
    dump(root / "validation_gate.json", gate)
    manifest = RunManifest(
        run_id=f"{OBJECTIVE}/{root.name}", objective_id=OBJECTIVE,
        hypothesis="A confluent prior-profile excursion with outward large-print flow that reclaims the zone rotates to the next pre-open opposing node.",
        responsibility_versions={"features": source_hashes["src/features/nq_profile_reclaim.py"],
                                 "economic_evaluation": source_hashes["src/backtest/nq_profile_reclaim_quotes.py"]},
        code=CodeIdentity(git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), dirty=True),
        datasets=[DatasetRef(logical_id="NQ/lead-tape/TBBO/2025-05_to_2026-06", semantic_type="raw_source",
                             content_sha256=inventory_hash, location="sources/price_tape+parquet_tape", n_rows=sum(r["rows"] for r in inventory))],
        config_snapshot=protocol, config_snapshot_resolved=True,
        validation=ValidationDesign(split="chronological_discovery_then_frozen_validation", n_folds=1,
                                    purge_sessions=0, hold_out_dates=("2026-07-01", "2026-09-22")),
        seeds={"bootstrap": SEED}, execution_assumptions=asdict(execution), decision_policy=asdict(policy),
        calibration={"method": "none; directly specified policy"}, extra={"gate": gate, "validation": summaries},
    )
    manifest.write(root)
    print(json.dumps({"run": str(root), "frozen": frozen, "validation": summaries, "gate": gate}, indent=2), flush=True)
    print(f"✅ Done in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
