#!/usr/bin/env python3
"""Independent fill and ordered liquidation-equity audit of completed quote replay.

Reads only an already completed compatibility/holdout phase. It neither chooses
a policy nor evaluates unopened reserve data. TBBO is trade-sampled: observed
mark drawdown cannot establish the maximum between unobserved quote events.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.data_paths import resolve_data_path
from src.common.vm_config import get_vm_config

SLIPPAGE = .25
COMMISSION = 3.36
POINT_VALUE = 20.


def digest(path: Path) -> str:
    """Return actual file-byte identity."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize_marks(marks: np.ndarray) -> dict:
    """Retain enough ordered-path statistics to compose exact observed drawdown."""
    marks = np.asarray(marks, dtype=float)
    if not len(marks) or not np.isfinite(marks).all():
        raise ValueError("marks must be finite and nonempty")
    with_flat = np.r_[0., marks]
    dd = np.maximum.accumulate(with_flat) - with_flat
    trough = int(np.argmax(dd))
    peak = int(np.argmax(with_flat[:trough + 1]))
    return {"min_liquidation_usd": float(marks.min()),
            "max_liquidation_usd": float(marks.max()),
            "observed_intratrade_drawdown_usd": float(dd[trough]),
            "intratrade_peak_position": peak - 1,
            "intratrade_trough_position": trough - 1,
            "terminal_liquidation_usd": float(marks[-1])}


def match_fill_event(times: np.ndarray, prices: np.ndarray, timestamp: int,
                     price: float, minimum_index: int = 0) -> tuple[int, int]:
    """Match recorded price inside an exact timestamp, preserving event order.

    The earliest eligible matching event is deterministic. Repeated matching
    prices make the event identity ambiguous; report their count rather than
    pretending the timestamp alone identifies a unique raw observation.
    """
    begin = max(minimum_index, int(np.searchsorted(times, timestamp, side="left")))
    end = int(np.searchsorted(times, timestamp, side="right"))
    matches = np.flatnonzero(np.isclose(prices[begin:end], price, atol=1e-7, rtol=0))
    if not len(matches):
        raise ValueError("Recorded fill has no exact-timestamp executable-price match")
    return begin + int(matches[0]), int(len(matches))


def audit_session(task: tuple[dict, list[dict]]) -> tuple[list[dict], dict]:
    """Verify one source hash and independently inspect its recorded fills."""
    source, trades = task
    path = resolve_data_path(source["path"])
    actual_hash = digest(path)
    if actual_hash != source["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    count = pq.ParquetFile(path).metadata.num_rows
    if count != source["rows"]:
        raise ValueError(f"Source row-count mismatch: {path}")
    inventory = {"session": source["session"], "path": str(path),
                 "sha256": actual_hash, "rows": count, "hash_verified": True}
    if not trades:
        return [], inventory
    tape = pd.read_parquet(path)
    if not isinstance(tape.index, pd.DatetimeIndex) or tape.index.tz is None:
        raise ValueError("Tape requires timezone-aware index")
    if tape.index.hasnans or not tape.index.is_monotonic_increasing:
        raise ValueError("Tape timestamp ordering invalid")
    bid = tape["bid_px" if "bid_px" in tape else "bid_px_00"].to_numpy(float)
    ask = tape["ask_px" if "ask_px" in tape else "ask_px_00"].to_numpy(float)
    valid = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    qt = tape.index.as_unit("ns").asi8[valid]
    bid, ask = bid[valid], ask[valid]
    results = []
    for trade in trades:
        entry_ts = pd.Timestamp(trade["entry_ts"])
        exit_ts = pd.Timestamp(trade["exit_ts"])
        decision_ts = pd.Timestamp(trade["entry_decision_ts"])
        side = int(trade["side"])
        if side not in (-1, 1) or not decision_ts < entry_ts <= exit_ts:
            raise ValueError("Trade direction/timestamp order invalid")
        if not .25 <= (entry_ts - decision_ts).total_seconds() <= 5.:
            raise ValueError("Base entry latency/window mismatch")
        # Execution-event matching is not a feature/context as-of join.
        entry_prices = ask + SLIPPAGE if side == 1 else bid - SLIPPAGE
        exit_prices = bid - SLIPPAGE if side == 1 else ask + SLIPPAGE
        begin, entry_matches = match_fill_event(qt, entry_prices, entry_ts.value,
                                                float(trade["entry_price"]))
        end, exit_matches = match_fill_event(qt, exit_prices, exit_ts.value,
                                             float(trade["exit_price"]), begin)
        expected_entry = ask[begin] + SLIPPAGE if side == 1 else bid[begin] - SLIPPAGE
        expected_exit = bid[end] - SLIPPAGE if side == 1 else ask[end] + SLIPPAGE
        if not np.isclose(trade["entry_price"], expected_entry, atol=1e-7, rtol=0):
            raise ValueError("Entry fill does not match exact-timestamp executable quote")
        if not np.isclose(trade["exit_price"], expected_exit, atol=1e-7, rtol=0):
            raise ValueError("Exit fill does not match exact-timestamp executable quote")
        mark = bid[begin:end + 1] if side == 1 else ask[begin:end + 1]
        # Entry friction is ALREADY in entry_price. Reserve only the remaining
        # exit slippage and one all-in commission at every open-position mark.
        liquidation = side * (mark - side * SLIPPAGE - expected_entry) * POINT_VALUE - COMMISSION
        stats = summarize_marks(liquidation)
        if not np.isclose(stats["terminal_liquidation_usd"], trade["net_usd"], atol=1e-7, rtol=0):
            raise ValueError("Independent terminal PnL differs from ledger")
        observed_times = qt[begin:end + 1]
        for label in ("peak", "trough"):
            pos = stats.pop(f"intratrade_{label}_position")
            stats[f"intratrade_{label}_ts"] = entry_ts.isoformat() if pos < 0 else pd.Timestamp(observed_times[pos], tz="UTC").isoformat()
        results.append({"trade_id": trade["trade_id"], "session": source["session"],
                        "entry_ts": entry_ts.isoformat(), "exit_ts": exit_ts.isoformat(),
                        "side": side, "entry_price": expected_entry, "exit_price": expected_exit,
                        "net_usd": float(trade["net_usd"]), "quote_events": len(mark),
                        "entry_matching_events": entry_matches, "exit_matching_events": exit_matches,
                        "ambiguous_fill_events": int(entry_matches > 1) + int(exit_matches > 1),
                        "maximum_observed_quote_gap_seconds": float(np.diff(observed_times).max()/1e9) if len(mark) > 1 else 0.,
                        "fills_verified": True, **stats})
    return results, inventory


def aggregate_drawdown(trades: list[dict]) -> dict:
    """Compose non-overlapping trade paths without losing cross-trade peaks."""
    equity = peak = drawdown = 0.
    for trade in trades:
        drawdown = max(drawdown, trade["observed_intratrade_drawdown_usd"],
                       peak - (equity + trade["min_liquidation_usd"]))
        peak = max(peak, equity + trade["max_liquidation_usd"])
        equity += trade["net_usd"]
        peak = max(peak, equity)
    return {"net_usd": equity, "maximum_observed_liquidation_drawdown_usd": drawdown,
            "full_observed_liquidation_mark_drawdown_usd": drawdown,
            "maximum_observed_liquidation_equity_usd": peak}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_dir", type=Path, help="completed run/compatibility or run/holdout")
    args = parser.parse_args()
    phase = args.phase_dir.resolve()
    if phase.name not in {"compatibility", "holdout"}:
        parser.error("Expected a completed compatibility or holdout directory")
    lock = json.loads((phase / "evaluation_lock.json").read_text())
    if phase.name == "holdout" and not lock.get("reserved_read"):
        parser.error("No existing consultation marker; reserve may not be opened")
    if lock.get("phase") != phase.name:
        raise ValueError("Evaluation lock phase does not match audit directory")
    execution = lock["execution"]
    base = [row for row in execution["scenarios"] if row["name"] == "base"]
    if (len(base) != 1 or base[0]["slippage_points_per_side"] != SLIPPAGE
            or base[0]["latency_ms"] != 250
            or execution["commission_per_trade"] != COMMISSION
            or execution["point_value"] != POINT_VALUE):
        raise ValueError("Frozen base execution assumptions differ from independent audit")
    ledger_path = phase / "base/trade_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    inventory_path = phase / "input_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    sessions = [r["session"] for r in inventory]
    if len(sessions) != len(set(sessions)):
        raise ValueError("Duplicate source session in inventory")
    if ledger.valid.isna().any() or not ledger.valid.isin([True, False]).all():
        raise ValueError("Ledger validity must be explicit boolean values")
    unresolved = ledger.loc[~ledger.valid.astype(bool)].copy()
    valid = ledger.loc[ledger.valid.astype(bool)].copy().sort_values("entry_ts").reset_index(drop=True)
    if valid.entry_ts.isna().any() or valid.exit_ts.isna().any():
        raise ValueError("Valid ledger contains missing timestamps")
    if (valid.entry_ts.iloc[1:].reset_index(drop=True) < valid.exit_ts.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("Overlapping positions in one-position ledger")
    if not set(valid.session) <= set(sessions):
        raise ValueError("Missing trade source in input inventory")
    valid["trade_id"] = np.arange(len(valid))
    grouped = {s: g.to_dict("records") for s, g in valid.groupby("session")}
    dest = phase / "audit"
    dest.mkdir(exist_ok=False)
    start = time.monotonic()
    records, checked, errors = [], [], []
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.)
    print(f"[{'░'*20}] verify source hashes and quote marks ({workers} workers)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(audit_session, (row, grouped.get(row["session"], []))): row["session"] for row in inventory}
        for i, job in enumerate(as_completed(jobs), 1):
            try:
                rows, source = job.result()
                records.extend(rows); checked.append(source)
            except Exception as exc:
                errors.append({"session": jobs[job], "error": str(exc)})
            if i % 25 == 0 or i == len(jobs):
                complete = int(20*i/len(jobs))
                print(f"[{'█'*complete}{'░'*(20-complete)}] audited {i}/{len(jobs)} sessions", flush=True)
    records.sort(key=lambda row: row["entry_ts"])
    verified = not errors and len(records) == len(valid)
    summary = {"audit_passed": verified and len(unresolved) == 0,
               "valid_trade_audit_passed": verified,
               "source_files_verified": len(checked), "ledger_trades": len(ledger),
               "valid_trades": len(valid), "unresolved_trades": len(ledger)-len(valid),
               "trades_audited": len(records), "errors": errors,
               "n_trades_verified": len(records),
               "max_quote_gap_seconds": max((row["maximum_observed_quote_gap_seconds"] for row in records), default=0.),
               "ambiguous_fill_events": sum(row["ambiguous_fill_events"] for row in records),
               "assumptions": {"slippage_per_side_points": SLIPPAGE, "commission_roundtrip_usd": COMMISSION,
                               "point_value_usd": POINT_VALUE, "quantity": 1},
               "ledger_sha256": digest(ledger_path), "input_inventory_sha256": digest(inventory_path),
               "audit_code_sha256": digest(Path(__file__)),
               "evaluation_lock_sha256": digest(phase / "evaluation_lock.json"),
               "evidence": "Independent raw-file hashes, exact-timestamp executable-price matches, ordered BBO liquidation marks, and cross-trade peak composition",
               "limitations": ["Trade-sampled pretrade BBO; no claim about unobserved between-event peaks or depth/liquidity",
                               "Liquidation equity reserves exit slippage and full roundtrip commission",
                               "No independent signal/policy verification; only source identity, fills, ordering and economics",
                               "Unresolved trades are disclosed and excluded from realized/MTM totals",
                               "Duplicate timestamp/price ambiguity uses the earliest eligible matching event; counts are disclosed"]}
    if verified:
        summary.update(aggregate_drawdown(records))
        summary["worst_observed_intratrade_drawdown_usd"] = max((r["observed_intratrade_drawdown_usd"] for r in records), default=0.)
    (dest / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (dest / "verified_sources.json").write_text(json.dumps(checked, indent=2) + "\n")
    pd.DataFrame(records).to_csv(dest / "trade_audit.csv", index=False)
    unresolved.to_csv(dest / "unresolved_trades.csv", index=False)
    monthly = []
    if records:
        for month, group in pd.DataFrame(records).groupby(lambda i: records[i]["session"][:7]):
            rows = group.to_dict("records")
            pnl = group.net_usd
            monthly.append({"month": month, "trades": len(group), "wins": int(pnl.gt(0).sum()),
                            "losses": int(pnl.lt(0).sum()), "total_loss_usd": float(-pnl[pnl < 0].sum()),
                            **aggregate_drawdown(rows)})
    pd.DataFrame(monthly).to_csv(dest / "monthly.csv", index=False)
    elapsed = int(time.monotonic()-start)
    print(f"✅ Done in {elapsed//60}m {elapsed%60}s: {dest}", flush=True)
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
