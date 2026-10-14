#!/usr/bin/env python3
"""Compare completed quote-model phases with a fixed 09:00–15:00 long baseline.

This diagnostic never selects policy parameters. Only sessions already listed
in the completed phase's daily ledger are read; holdout context is not traded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.data_paths import resolve_data_path
from src.common.vm_config import get_vm_config
from scripts.research.audit_nq_cadence_marks import (
    SLIPPAGE, COMMISSION, POINT_VALUE, digest, summarize_marks, aggregate_drawdown,
)
from scripts.research.run_nq_cadence_area import evaluate

SEED = 20260924


def benchmark_session(tape: pd.DataFrame, session: str) -> tuple[dict, dict | None]:
    """One long contract with clock-only decisions and source-executable fills."""
    if not isinstance(tape.index, pd.DatetimeIndex) or tape.index.tz is None:
        raise ValueError("Quote timestamps must be timezone-aware")
    if tape.index.hasnans or not tape.index.is_monotonic_increasing:
        raise ValueError("Quote timestamps must be ordered and finite")
    if not (tape.index.tz_convert("America/Chicago").strftime("%Y-%m-%d") == session).all():
        raise ValueError("Tape crosses named local session")
    bid = tape["bid_px" if "bid_px" in tape else "bid_px_00"].to_numpy(float)
    ask = tape["ask_px" if "ask_px" in tape else "ask_px_00"].to_numpy(float)
    keep = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    bid, ask = bid[keep], ask[keep]
    qt = tape.index.as_unit("ns").asi8[keep]
    decision = pd.Timestamp(f"{session} 09:00", tz="America/Chicago").tz_convert("UTC")
    close = pd.Timestamp(f"{session} 15:00", tz="America/Chicago").tz_convert("UTC")
    ready = decision + pd.Timedelta(milliseconds=250)
    exit_ready = close + pd.Timedelta(milliseconds=250)
    record = {"session": session, "entry_decision_ts": decision, "decision_ts": decision,
              "entry_ts": pd.NaT, "exit_ts": pd.NaT, "side": 1,
              "entry_price": np.nan, "exit_price": np.nan, "net_usd": np.nan,
              "gross_points": np.nan, "valid": False, "exit_reason": "no_entry_within_5s"}
    # This is forward execution-event resolution, never a feature as-of join.
    start = int(np.searchsorted(qt, ready.value, side="left"))
    if start >= len(qt) or qt[start] > (decision + pd.Timedelta(seconds=5)).value:
        return record, None
    entry = ask[start] + SLIPPAGE
    record.update(entry_ts=pd.Timestamp(qt[start], tz="UTC"), entry_price=entry,
                  exit_reason="missing_session_end")
    end = int(np.searchsorted(qt, exit_ready.value, side="left"))
    if end >= len(qt) or qt[end] > (close + pd.Timedelta(minutes=15)).value:
        return record, None
    exit_price = bid[end] - SLIPPAGE
    pnl = (exit_price-entry)*POINT_VALUE - COMMISSION
    mark = (bid[start:end+1]-SLIPPAGE-entry)*POINT_VALUE - COMMISSION
    stats = summarize_marks(mark)
    stats.update(net_usd=float(pnl), session=session,
                 entry_ts=pd.Timestamp(qt[start], tz="UTC").isoformat(),
                 exit_ts=pd.Timestamp(qt[end], tz="UTC").isoformat(),
                 quote_events=end-start+1,
                 max_quote_gap_seconds=float(np.diff(qt[start:end+1]).max()/1e9) if end > start else 0.)
    record.update(exit_ts=pd.Timestamp(qt[end], tz="UTC"), exit_price=exit_price,
                  gross_points=exit_price-entry, net_usd=pnl, valid=True,
                  exit_reason="fixed_session_close")
    return record, stats


def run_source(source: dict) -> tuple[dict, dict | None]:
    """Verify the frozen phase input identity before reading quote prices."""
    path = resolve_data_path(source["path"])
    if digest(path) != source["sha256"]:
        raise ValueError(f"Source hash mismatch: {source['session']}")
    tape = pd.read_parquet(path)
    if len(tape) != source["rows"]:
        raise ValueError(f"Source row mismatch: {source['session']}")
    return benchmark_session(tape, source["session"])


def select_phase_sources(inventory: list[dict], sessions: list[str], phase: str) -> list[dict]:
    """Exclude warm-up/context sessions using the evaluated daily population."""
    if len(sessions) != len(set(sessions)) or not sessions:
        raise ValueError("Daily population must be nonempty and unique")
    if phase not in {"compatibility", "holdout"}:
        raise ValueError("Unknown phase")
    if any((session >= "2026-07-01") != (phase == "holdout") for session in sessions):
        raise ValueError("Daily population crosses the reserved phase boundary")
    indexed = {row["session"]: row for row in inventory}
    if len(indexed) != len(inventory) or not set(sessions) <= set(indexed):
        raise ValueError("Missing or duplicate frozen source inventory")
    return [indexed[session] for session in sorted(sessions)]


def paired_bootstrap(model: np.ndarray, benchmark: np.ndarray) -> dict:
    """Paired circular five-session-block bootstrap of daily dollar difference."""
    model, benchmark = np.asarray(model, float), np.asarray(benchmark, float)
    if model.shape != benchmark.shape or model.ndim != 1 or not len(model):
        raise ValueError("Paired populations must be nonempty aligned vectors")
    difference = model-benchmark
    if not np.isfinite(difference).all():
        raise ValueError("Unresolved daily economics cannot enter comparison")
    n = len(difference)
    rng = np.random.default_rng(SEED)
    indices = (rng.integers(0, n, (5000, (n+4)//5, 1)) + np.arange(5)) % n
    means = difference[indices.reshape(5000, -1)[:, :n]].mean(axis=1)
    return {"model_minus_benchmark_mean_daily_usd": float(difference.mean()),
            "model_minus_benchmark_total_usd": float(difference.sum()),
            "paired_mean_daily_95ci": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_seed": SEED, "bootstrap_block_sessions": 5,
            "bootstrap_replicates": 5000, "paired_sessions": n}


def write_json(path: Path, value: dict | list) -> None:
    """Write strict JSON while preserving timestamps as explicit strings."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_dir", type=Path)
    args = parser.parse_args()
    phase = args.phase_dir.resolve()
    lock_path = phase / "evaluation_lock.json"
    lock = json.loads(lock_path.read_text())
    if phase.name not in {"compatibility", "holdout"} or lock["phase"] != phase.name:
        raise ValueError("Expected completed compatibility or holdout phase")
    if phase.name == "holdout" and not lock.get("reserved_read"):
        raise ValueError("Final reserve has not been consulted; benchmark cannot unlock it")
    execution = lock["execution"]
    base = [s for s in execution["scenarios"] if s["name"] == "base"]
    if (len(base) != 1 or base[0]["slippage_points_per_side"] != SLIPPAGE
            or base[0]["latency_ms"] != 250 or execution["commission_per_trade"] != COMMISSION
            or execution["point_value"] != POINT_VALUE):
        raise ValueError("Base costs differ from the fixed comparator")
    daily_path = phase / "base/daily.csv"
    model_daily = pd.read_csv(daily_path, dtype={"session": str}).set_index("session")
    model_ledger_path = phase / "base/trade_ledger.parquet"
    model_ledger = pd.read_parquet(model_ledger_path)
    if not set(model_ledger.session) <= set(model_daily.index):
        raise ValueError("Model trade sessions are outside its daily population")
    inventory = json.loads((phase / "input_inventory.json").read_text())
    sources = select_phase_sources(inventory, model_daily.index.tolist(), phase.name)
    destination = phase / "benchmarks"
    destination.mkdir(exist_ok=False)
    output = destination / "base"
    output.mkdir()
    write_json(destination / "input_inventory.json", sources)
    benchmark_lock = {"phase": phase.name, "reserved_read": lock.get("reserved_read", False),
                      "benchmark": "one_NQ_long_0900_to_1500_Chicago_clock_only",
                      "execution": execution, "model_evaluation_lock_sha256": digest(lock_path),
                      "model_daily_sha256": digest(daily_path),
                      "model_ledger_sha256": digest(model_ledger_path),
                      "code_sha256": digest(Path(__file__)),
                      "created_at": str(pd.Timestamp.now(tz="UTC")),
                      "population": "Only completed model-phase daily.csv sessions; no context trades"}
    write_json(destination / "evaluation_lock.json", benchmark_lock)
    started = time.monotonic()
    records, marks = [], []
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.)
    print(f"[{'░'*20}] fixed quote comparator {len(sources)} sessions; {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_source, source) for source in sources]
        for i, job in enumerate(as_completed(jobs), 1):
            trade, path_stats = job.result()
            records.append(trade)
            if path_stats is not None:
                marks.append(path_stats)
            if i % 25 == 0 or i == len(jobs):
                complete = int(20*i/len(jobs))
                print(f"[{'█'*complete}{'░'*(20-complete)}] fixed comparator {i}/{len(jobs)}", flush=True)
    trades = pd.DataFrame(records).sort_values("session")
    marks.sort(key=lambda row: row["session"])
    sessions = sorted(model_daily.index)
    metrics, daily = evaluate(trades, sessions, bootstrap=True)
    metrics.update(aggregate_drawdown(marks))
    metrics["maximum_intratrade_liquidation_drawdown_usd"] = max((r["observed_intratrade_drawdown_usd"] for r in marks), default=0.)
    metrics["quote_gaps_max_seconds"] = max((r["max_quote_gap_seconds"] for r in marks), default=0.)
    unresolved = int((~trades.valid).sum())
    model_unresolved = int((~model_ledger.valid.astype(bool)).sum())
    comparison = {"benchmark": benchmark_lock["benchmark"], "same_notional_contracts": 1,
                  "benchmark_unresolved": unresolved, "model_unresolved": model_unresolved,
                  "paired_comparison_complete": unresolved == 0 and model_unresolved == 0,
                  "benchmark_observed_liquidation_drawdown_usd": metrics["maximum_observed_liquidation_drawdown_usd"],
                  "model_closed_trade_drawdown_usd": evaluate(model_ledger, sessions)[0]["max_drawdown_usd"],
                  "limitations": ["One-contract exposure matched; holding time and risk are not equalized",
                                  "Trade-sampled BBO marks do not observe every market quote",
                                  "Benchmark is a fixed post-model diagnostic, not another selection trial"]}
    audit_path = phase / "audit/audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        comparison["model_observed_liquidation_drawdown_usd"] = audit.get("maximum_observed_liquidation_drawdown_usd")
        comparison["model_audit_sha256"] = digest(audit_path)
    if comparison["paired_comparison_complete"]:
        comparison.update(paired_bootstrap(model_daily.loc[sessions, "net_usd"].to_numpy(), daily.loc[sessions, "net_usd"].to_numpy()))
    else:
        daily.loc[trades.loc[~trades.valid, "session"], ["net_usd", "equity_usd", "drawdown_usd"]] = np.nan
        metrics["realized_totals_exclude_unresolved"] = True
    trades.to_parquet(output / "trade_ledger.parquet", index=False)
    daily.to_csv(output / "daily.csv")
    pd.DataFrame(marks).to_csv(output / "liquidation_marks.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(destination / "comparison.json", comparison)
    elapsed = int(time.monotonic()-started)
    print(f"✅ Done in {elapsed//60}m {elapsed%60}s: {destination}", flush=True)
    return 0 if comparison["paired_comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
