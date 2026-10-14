#!/usr/bin/env python3
"""Independently replay persisted NQ trades; append audit artifacts only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.common.vm_config import get_vm_config
from src.features.tape import load_tape, tape_path
from scripts.research.train_nq_selective import summarize


def digest(path: Path) -> str:
    """Hash file bytes without loading an entire file into memory."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit_session(task: tuple) -> tuple[list[dict], dict]:
    """Independent NumPy event replay and liquidation marks for one session."""
    session, records, expected_hash, config = task
    assert session < config["holdout"][0], "refusing holdout data"
    day = date.fromisoformat(session)
    raw = load_tape(day)
    if raw is None:
        raise ValueError(f"missing tape {session}")
    assert raw.index.is_monotonic_increasing
    bids = raw.bid_px.to_numpy(float)
    asks = raw.ask_px.to_numpy(float)
    good = np.isfinite(bids) & np.isfinite(asks) & (bids > 0) & (asks >= bids)
    times = raw.index.as_unit("ns").asi8[good]
    bids, asks = bids[good], asks[good]
    slip, fee = config["base_slippage_points"], config["commission"]
    delay = int(config["latency_ms"] * 1e6)
    targets = dict(config["templates"])
    output = []
    for r in records:
        decision = pd.Timestamp(r["ts"]).value
        horizon = decision + int(r["horizon"] * 60e9)
        entry = int(np.searchsorted(times, decision + delay))
        assert entry < len(times) and times[entry] <= decision + 5_000_000_000
        side = r["side"]
        entry_price = (asks[entry] + slip) if side == 1 else (bids[entry] - slip)
        liquidation = bids if side == 1 else asks
        last = int(np.searchsorted(times, horizon, side="right"))
        moves = side * (liquidation[entry:last] - entry_price)
        hit = np.flatnonzero((moves <= -r["stop_points"]) |
                             (moves >= r["stop_points"] * targets[r["horizon"]]))
        trigger = None
        if len(hit):
            trigger = entry + int(hit[0])
            reason = "stop" if moves[hit[0]] <= -r["stop_points"] else "target"
            exit_i = max(int(np.searchsorted(times, times[trigger] + delay)), trigger + 1)
        else:
            reason = "time"
            exit_i = max(int(np.searchsorted(times, horizon + delay)), entry + 1)
            assert exit_i < len(times) and times[exit_i] <= horizon + delay + 5_000_000_000
        assert exit_i < len(times)
        exit_price = liquidation[exit_i] - side * slip
        net = side * (exit_price-entry_price)*20-fee
        assert pd.Timestamp(r["entry_ts"]).value == times[entry]
        assert pd.Timestamp(r["exit_ts"]).value == times[exit_i]
        assert reason == r["exit_reason"]
        np.testing.assert_allclose([entry_price, exit_price, net],
                                   [r["entry_price"], r["exit_price"], r["net_usd"]],
                                   rtol=0, atol=1e-7)
        # Hypothetical immediate liquidation at each observed executable quote.
        # Entry already includes its slippage. Reserve exit slip + full fee.
        marks = side*(liquidation[entry:exit_i+1]-side*slip-entry_price)*20-fee
        path = np.r_[0.0, marks]
        internal_dd = float(np.max(np.maximum.accumulate(path)-path))
        gaps = np.diff(times[entry:exit_i+1])/1e9
        signed_gross = side*(exit_price-entry_price)
        output.append(dict(
            trade_id=r["audit_trade_id"], session=session, ts=str(r["ts"]),
            entry_ts=str(r["entry_ts"]), exit_ts=str(r["exit_ts"]), side=side,
            entry_price=entry_price, exit_price=exit_price, exit_reason=reason,
            stop_points=r["stop_points"], net_usd=net,
            mark_min_usd=float(path.min()), mark_peak_usd=float(path.max()),
            internal_mark_drawdown_usd=internal_dd,
            max_quote_gap_seconds=float(gaps.max()) if len(gaps) else 0.0,
            entry_wait_ms=(times[entry]-decision)/1e6,
            trigger_to_exit_ms=(times[exit_i]-times[trigger])/1e6 if trigger is not None else None,
            loss_beyond_stop_points=max(0.0,-signed_gross-r["stop_points"]) if reason=="stop" else 0.0,
            quote_marks=len(marks), exact_assertions_passed=True,
        ))
    actual_hash = digest(tape_path(day))
    return output, dict(session=session, expected_sha256=expected_hash,
                        actual_sha256=actual_hash, matches=actual_hash==expected_hash)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.run
    model = root/"pass_1"/"model_selective_ev"
    destination = model/"report"/"audit"
    if destination.exists():
        raise FileExistsError(f"preserving existing audit: {destination}")
    config = json.loads((root/"protocol.json").read_text())
    inventory = {r["session"]:r for r in json.loads((root/"pass_1"/"input_inventory.json").read_text())}
    trades = pd.read_parquet(model/"trade_ledger.parquet").sort_values("ts").reset_index(drop=True)
    trades["audit_trade_id"] = np.arange(len(trades))
    if len(trades)>1:
        assert (trades.ts.iloc[1:].reset_index(drop=True) > trades.exit_ts.iloc[:-1].reset_index(drop=True)).all()
    tasks = [(s,g.to_dict("records"),inventory[s]["sha256"],config) for s,g in trades.groupby("session")]
    rows, raw_hashes = [], []
    with ProcessPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.0)) as pool:
        futures = [pool.submit(audit_session,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            part, raw_hash = future.result()
            rows.extend(part)
            raw_hashes.append(raw_hash)
            if i%10==0 or i==len(tasks):
                n=int(20*i/max(len(tasks),1))
                print(f"[{'█'*n}{'░'*(20-n)}] Audit {i}/{len(tasks)} sessions",flush=True)
    realized = peak = drawdown = 0.0
    rows.sort(key=lambda r:r["trade_id"])
    for r in rows:
        drawdown=max(drawdown,peak-(realized+r["mark_min_usd"]),r["internal_mark_drawdown_usd"])
        peak=max(peak,realized+r["mark_peak_usd"])
        realized+=r["net_usd"]
        r["closed_equity_usd"]=realized
        r["running_mark_peak_usd"]=peak
        r["running_mark_drawdown_usd"]=drawdown
    daily=pd.read_csv(model/"report"/"daily_equity.csv",index_col="session")
    sessions=daily.index.astype(str).tolist()
    stats,_=summarize(trades,sessions,bootstrap=True)
    baseline_intervals={}
    for path in sorted(model.glob("baseline_*.parquet")):
        baseline=pd.read_parquet(path)
        if len(baseline):
            baseline=baseline.sort_values("ts")
        baseline_intervals[path.stem]=summarize(baseline,sessions,bootstrap=True)[0]
    hashes=json.loads((root/"source_hashes.json").read_text())
    source_checks={}
    for path,expected in hashes.items():
        snapshot=root/"code"/path
        source_checks[path]=dict(expected_sha256=expected,
            snapshot_sha256=digest(snapshot) if snapshot.exists() else None,
            working_sha256=digest(Path(path)) if Path(path).exists() else None)
        source_checks[path]["snapshot_matches"]=source_checks[path]["snapshot_sha256"]==expected
        source_checks[path]["working_matches"]=source_checks[path]["working_sha256"]==expected
    month=daily.assign(month=pd.to_datetime(daily.index).strftime("%Y-%m")).groupby("month").agg(
        net_usd=("net_usd","sum"),observed_sessions=("net_usd","size"))
    monthly_trades=trades.assign(month=pd.to_datetime(trades.ts).dt.strftime("%Y-%m")).groupby("month")
    month["n_trades"]=monthly_trades.size().reindex(month.index,fill_value=0)
    month["loss_count"]=monthly_trades.net_usd.apply(lambda s:int((s<0).sum())).reindex(month.index,fill_value=0)
    audit=dict(n_trades_verified=len(rows),n_quote_marks=sum(r["quote_marks"] for r in rows),
        net_usd=realized,full_observed_liquidation_mark_drawdown_usd=drawdown,
        prior_report_drawdown_proxy_usd=json.loads((model/"report"/"metrics.json").read_text()).get("conservative_intratrade_drawdown_usd"),
        max_quote_gap_seconds=max((r["max_quote_gap_seconds"] for r in rows),default=0),
        exact_execution_assertions_passed=True,raw_hashes=raw_hashes,source_checks=source_checks,
        recomputed_statistics=stats,baseline_block_bootstrap=baseline_intervals,
        largest_stop_overshoots=sorted(rows,key=lambda r:r["loss_beyond_stop_points"],reverse=True)[:5],
        marking_assumptions=dict(point_value=20,round_trip_fee=config["commission"],
            exit_slippage_points=config["base_slippage_points"],
            explanation="Observed executable liquidation quotes, full fee reserved immediately, exit slip reserved at every mark; no latency on hypothetical marks."),
        holdout_read=False,elapsed_seconds=time.monotonic()-started)
    destination.mkdir(parents=True)
    (destination/"audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    month.to_csv(destination/"monthly.csv")
    pd.DataFrame(rows).to_csv(destination/"trade_audit.csv",index=False)
    text=f"""# Independent NQ execution audit

Verified **{len(rows)} trades** against original tape events with independent NumPy
trigger/latency logic. All entry prices, entry timestamps, exit prices, exit
timestamps, exit reasons and net P&L assertions passed. No holdout data was read.

Net result: **${realized:,.2f}**. Full observed liquidation-mark peak-to-trough
drawdown: **${drawdown:,.2f}**, including favorable unrealized peaks before reversals.
The base report remains preserved. The old MAE-based figure did not capture all
unrealized peaks and was not a conservative bound on mark-to-market drawdown.

Marking reserves the full ${config['commission']:.2f} round-trip fee and
{config['base_slippage_points']} point exit slippage at every observed liquidation
quote. Entry slippage is already embedded in the entry price. These are observed
print-associated BBO marks, not a continuous quote feed or delayed liquidation
execution. Maximum observed within-trade quote gap: {audit['max_quote_gap_seconds']:.3f}s.

All selected-session input SHA256 matches: {all(r['matches'] for r in raw_hashes)}.
All source snapshot SHA256 matches: {all(r['snapshot_matches'] for r in source_checks.values())}.
Working-source differences are recorded separately; they do not modify historical evidence.

`audit.json` records exact assertion results, source hashes, selected stop overshoots,
and five-session block-bootstrap intervals for the original fixed baselines.
`monthly.csv` includes zero-trade observed months; `trade_audit.csv` contains per-trade
mark extrema and running drawdown. Baseline intervals are diagnostic comparisons,
not permission to select a winning baseline using this OOF period.
"""
    (destination/"audit.md").write_text(text)
    print(json.dumps({k:audit[k] for k in ("n_trades_verified","net_usd","full_observed_liquidation_mark_drawdown_usd","max_quote_gap_seconds")}),flush=True)
    print(f"✅ Done in {time.monotonic()-started:.1f}s: {destination}",flush=True)


if __name__=="__main__":
    main()
