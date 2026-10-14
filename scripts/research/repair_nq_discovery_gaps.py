#!/usr/bin/env python3
"""Recover two known discovery-only NQ data gaps only at exactly zero vendor cost.

Writes a NEW preparation directory. Never modifies source caches or a database.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import databento as db
from dotenv import load_dotenv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.data_paths import parquet_path1s_dir
from src.common.vm_config import get_vm_config
from scripts.research.build_nq_long_bars import aggregate_session, build_one, sha256_file

WINDOWS = [("2020-02-28", "15:58", "21:15", 10204),
           ("2020-06-30", "14:10", "21:15", 14028)]


def recover(window: tuple, output: Path, replace_complete_session: bool = False,
            reuse_vendor_dir: Path | None = None) -> dict:
    """Cost-check immediately before requesting a fixed historical gap."""
    day, start, end, instrument_id = window
    if replace_complete_session:
        start = "13:00" if day == "2020-02-28" else "12:00"
        end = "21:15" if day == "2020-02-28" else "20:15"
    kwargs = dict(dataset="GLBX.MDP3", symbols=["NQ.c.0"], stype_in="continuous",
                  schema="ohlcv-1s", start=f"{day}T{start}:00Z", end=f"{day}T{end}:00Z")
    evidence_path = output / f"cost_{day}.json"
    raw_path = output / f"vendor_{day}.parquet"
    if reuse_vendor_dir is not None:
        evidence = json.loads((reuse_vendor_dir / f"cost_{day}.json").read_text())
        if evidence["request"] != kwargs or evidence["cost_usd"] != 0 or not evidence["download_performed"]:
            raise ValueError("Reused vendor file does not match the completed zero-cost request")
        cost = 0.
        shutil.copyfile(reuse_vendor_dir / f"vendor_{day}.parquet", raw_path)
        evidence["reused_local_source"] = str(reuse_vendor_dir / f"vendor_{day}.parquet")
        raw = pd.read_parquet(raw_path)
    else:
        client = db.Historical()
        cost = float(client.metadata.get_cost(**kwargs))
        evidence = {"request": kwargs, "cost_usd": cost,
                    "checked_at": datetime.now(timezone.utc).isoformat(), "download_performed": False}
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        if not np.isfinite(cost) or cost != 0.:
            raise RuntimeError(f"STOP: {day} vendor cost {cost}; explicit user approval required")
        raw = client.timeseries.get_range(**kwargs).to_df()
        evidence["download_performed"] = True
        evidence["downloaded_rows"] = len(raw)
        raw.to_parquet(raw_path, compression="zstd")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    if set(raw.instrument_id.astype(int)) != {instrument_id}:
        raise ValueError(f"Unexpected contract identity for {day}")
    fresh = raw[["open", "high", "low", "close", "volume"]].copy()
    if fresh.volume.max() > np.iinfo(np.int64).max:
        raise ValueError("Volume exceeds canonical signed int64 range")
    fresh["volume"] = fresh.volume.astype("int64")
    fresh.index = pd.to_datetime(fresh.index, utc=True).as_unit("ns")
    fresh.index.name = "ts"
    # The historical source cache's fixed window ends15:15 Chicago. Retain raw
    # vendor response in full, but respect that existing source-window contract.
    local = fresh.index.tz_convert("America/Chicago")
    minute = local.hour*60 + local.minute
    fresh = fresh.loc[(minute >= 420) & (minute < 915)]
    source = parquet_path1s_dir() / f"path_{day}.parquet"
    original = pd.read_parquet(source)
    overlap = original.index.intersection(fresh.index)
    if not len(overlap):
        raise ValueError(f"No overlap for independent consistency check: {day}")
    old_values, new_values = original.loc[overlap].to_numpy(float), fresh.loc[overlap].to_numpy(float)
    different = ~np.isclose(old_values, new_values, rtol=0, atol=1e-9)
    comparison = pd.DataFrame({"ts": overlap, "differing_fields": different.sum(axis=1)})
    comparison.to_parquet(output / f"overlap_comparison_{day}.parquet", index=False)
    if replace_complete_session:
        # A provider aggregation revision makes splicing heterogeneous seconds
        # unsafe. Whole-session replacement is explicit, preserved and audited.
        combined = fresh.sort_index()
    else:
        np.testing.assert_allclose(old_values, new_values, rtol=0, atol=1e-9)
        combined = pd.concat([original, fresh.loc[~fresh.index.isin(original.index)]]).sort_index()
    aggregate_session(combined, day)  # Validate OHLC, clock, order and date.
    merged_path = output / f"path_{day}.parquet"
    combined.to_parquet(merged_path, compression="zstd")
    return {"session": day, "instrument_id": instrument_id,
            "original_source": str(source), "original_sha256": sha256_file(source),
            "vendor_source": str(raw_path), "vendor_sha256": sha256_file(raw_path),
            "merged_source": str(merged_path), "merged_sha256": sha256_file(merged_path),
            "overlap_rows_compared": len(overlap), "overlap_rows_identical": int((~different.any(axis=1)).sum()),
            "overlap_rows_different": int(different.any(axis=1).sum()),
            "whole_session_replacement": replace_complete_session, "original_rows": len(original),
            "new_rows_added": len(combined)-len(original), "merged_rows": len(combined),
            "last_source_ts": combined.index.max().isoformat(), "cost_usd": cost}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace-complete-sessions", action="store_true",
                        help="fresh complete-session source, explicitly disclosing overlap revisions")
    parser.add_argument("--reuse-vendor-dir", type=Path,
                        help="reuse already fetched exact-window zero-cost vendor files without network")
    args = parser.parse_args()
    load_dotenv("/opt/vtech/.env")
    args.out.mkdir(parents=True, exist_ok=False)
    repairs = args.out / "repairs"
    repairs.mkdir()
    start = time.monotonic()
    print(f"[{'░'*20}] zero-cost checks and gap recovery", flush=True)
    recovered = []
    with ThreadPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.)) as pool:
        jobs = [pool.submit(recover, window, repairs, args.replace_complete_sessions,
                            args.reuse_vendor_dir) for window in WINDOWS]
        for job in as_completed(jobs):
            record = job.result(); recovered.append(record)
            print(f"[{'█'*10*len(recovered)}{'░'*(20-10*len(recovered))}] recovered {record['session']} +{record['new_rows_added']} rows", flush=True)
    bars_path = args.original / "bars_1m.parquet"
    bars = pd.read_parquet(bars_path)
    inventory = pd.read_parquet(args.original / "source_inventory.parquet")
    replacements, new_inventory = [], []
    for record in recovered:
        frame, inv = build_one(record["merged_source"])
        replacements.append(frame); new_inventory.append(inv)
        if inv["missing_cash_minutes"] != 0:
            raise ValueError(f"Repair still misses cash minutes: {record['session']}")
    days = [r["session"] for r in recovered]
    kept = bars.loc[~bars.session.isin(days)]
    merged = pd.concat([kept, *replacements]).sort_index()
    pd.testing.assert_frame_equal(kept, merged.loc[kept.index])
    new_bars_path = args.out / "bars_1m.parquet"
    merged.to_parquet(new_bars_path, compression="zstd")
    new_inventory = pd.concat([inventory.loc[~inventory.session.isin(days)], pd.DataFrame(new_inventory)]).sort_values("session")
    new_inventory_path = args.out / "source_inventory.parquet"
    new_inventory.to_parquet(new_inventory_path, compression="zstd", index=False)
    manifest = json.loads((args.original / "build_manifest.json").read_text())
    manifest.update({"created_at": datetime.now(timezone.utc).isoformat(),
                     "rows": len(merged), "bars_sha256": sha256_file(new_bars_path),
                     "source_inventory_sha256": sha256_file(new_inventory_path),
                     "repair_builder_sha256": sha256_file(Path(__file__)),
                     "parent_bars": str(bars_path), "parent_bars_sha256": sha256_file(bars_path),
                     "parent_build_manifest_sha256": sha256_file(args.original / "build_manifest.json"),
                     "repair_provenance": sorted(recovered, key=lambda r:r["session"]),
                     "unmodified_other_sessions_verified": True,
                     "provider_quality_caveat": "Vendor flags both recovered dates as degraded; full-session refresh avoids splicing revised aggregation with older cache",
                     "total_vendor_cost_usd": sum(r["cost_usd"] for r in recovered)})
    manifest["quality"] = {"sessions_with_incomplete_cash": int(new_inventory.missing_cash_minutes.gt(0).sum()),
                           "off_tick_price_count": int(new_inventory.source_off_tick_prices.sum()),
                           "maximum_source_gap_seconds": float(new_inventory.max_source_gap_seconds.max())}
    (args.out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    elapsed = int(time.monotonic()-start)
    print(f"✅ Done in {elapsed//60}m {elapsed%60}s: {new_bars_path}", flush=True)


if __name__ == "__main__":
    main()
