#!/usr/bin/env python3
"""Persist 5s 3-class TBM labels from 5s monthly parts ``_close`` (ADR-076).

    python scripts/features/build_labels_5s.py
    python scripts/features/build_labels_5s.py --objective nq-50pt-econ
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.vm_config import get_vm_config  # noqa: E402
from src.objectives.loader import load_objective  # noqa: E402
from src.training.label_store import (  # noqa: E402
    label_set_dir, labels_3class_from_close, tbm_contract_from_objective,
    write_contract, write_label_parts,
)
from src.training.data_pipeline import ensure_utc_index  # noqa: E402
from src.training.dual_binary_15s import label_params_from_objective  # noqa: E402

_BAR = 20


def _one_day(args: tuple) -> tuple[str, object]:
    day, close_idx, close_vals, high_vals, low_vals, params = args
    import pandas as pd
    close = pd.Series(close_vals, index=pd.to_datetime(close_idx, utc=True), name="_close")
    high = pd.Series(high_vals, index=close.index, name="cs_high")
    low = pd.Series(low_vals, index=close.index, name="cs_low")
    lab = labels_3class_from_close(close, target_high=high, target_low=low, **params)
    return day, lab.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default="nq-50pt-econ")
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--force", action="store_true", help="Rebuild even if dest exists")
    a = ap.parse_args()

    obj = load_objective(a.objective)
    params = label_params_from_objective(obj)
    params["src_step_s"] = 5
    contract = tbm_contract_from_objective(obj)
    contract["encoding"] = "3class"
    from src.features.stores import load_feature_frame

    out_dir = Path(a.out) if a.out else label_set_dir("5s", contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("labels_5s_part*.parquet")) and not a.force:
        print(f"label set already exists: {out_dir} (use --force to rebuild)")
        return 0
    write_contract(out_dir, contract)

    if a.src:
        from src.training.data_pipeline import DataPipeline
        df = DataPipeline(a.src).load_features(a.start or "2025-01-01", a.end or "2099-12-31")
        df = ensure_utc_index(df)
    else:
        df = load_feature_frame("5s", a.start, a.end)
    if "_close" not in df.columns:
        print("no _close in 5s features")
        return 1
    mode = params["target_observation"]
    if mode != "candle_close" and not {"cs_high", "cs_low"}.issubset(df.columns):
        print(f"target_observation={mode} requires cs_high and cs_low in 5s features")
        return 1
    chi = df.index.tz_convert("America/Chicago").strftime("%Y-%m-%d")
    jobs = []
    for day, part in df.groupby(chi):
        if a.start and day < a.start:
            continue
        if a.end and day > a.end:
            continue
        high = part["cs_high"] if "cs_high" in part else part["_close"]
        low = part["cs_low"] if "cs_low" in part else part["_close"]
        jobs.append((day, part.index.asi8, part["_close"].to_numpy(), high.to_numpy(),
                     low.to_numpy(), params))

    cfg = get_vm_config()
    workers = a.workers if a.workers is not None else cfg.workers_for(mem_per_worker_gb=1.0)
    print(cfg)
    print(f"⏳ 5s TBM labels: {len(jobs)} days, workers={workers}")
    if not jobs:
        print("nothing to do")
        return 0
    t0 = time.time()
    n_rows = done = 0
    frames = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_one_day, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            got = fut.result()
            done += 1
            _, frame = got
            frames.append(frame)
            n_rows += len(frame)
            pct = done / len(jobs)
            filled = int(_BAR * pct)
            bar = "█" * filled + "░" * (_BAR - filled)
            sys.stderr.write(f"\r  [{bar}] {pct:5.0%} — {got[0]}  {done}/{len(jobs)}")
            sys.stderr.flush()
    sys.stderr.write("\n")
    parts = write_label_parts(out_dir, frames, resolution="5s", force=a.force)
    print(f"✅ Done in {time.time()-t0:.0f}s — {n_rows:,} rows, parts={len(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
