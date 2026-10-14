#!/usr/bin/env python3
"""Build 15s dual-binary labels from 5s TBM ``_close`` labels.

    python scripts/features/build_dual_binary_15s.py --objective nq-50pt-econ-lg-st
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
from src.training.dual_binary_15s import (  # noqa: E402
    label_params_from_objective, labels_from_5s_close,
)
from src.training.label_store import (  # noqa: E402
    label_set_dir, tbm_contract_from_objective, write_contract, write_label_parts,
)

_BAR = 20


def _one_day(args: tuple) -> tuple[str, object]:
    day, close_idx, close_vals, high_vals, low_vals, params = args
    import pandas as pd
    close = pd.Series(close_vals, index=pd.to_datetime(close_idx, utc=True), name="_close")
    high = pd.Series(high_vals, index=close.index, name="cs_high")
    low = pd.Series(low_vals, index=close.index, name="cs_low")
    df = labels_from_5s_close(close, target_high=high, target_low=low, **params)
    return day, df.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default="nq-50pt-econ-lg-st")
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
    dr = (obj.get("eval") or {}).get("date_range") or {}
    # --force without --start/--end rebuilds every 5s hive day (stale April
    # files sat outside eval.date_range and kept timeout=600).
    if a.start or a.end or not a.force:
        start = a.start or dr.get("start")
        end = a.end or dr.get("end")
    else:
        start = a.start
        end = a.end
    from src.features.stores import load_feature_frame
    from src.training.data_pipeline import ensure_utc_index

    contract = tbm_contract_from_objective(obj)
    out_dir = Path(a.out) if a.out else label_set_dir("15s", contract)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("labels_15s_part*.parquet")) and not a.force:
        print(f"label set already exists: {out_dir} (use --force to rebuild)")
        return 0
    write_contract(out_dir, contract)

    if a.src:
        from src.training.data_pipeline import DataPipeline
        df = DataPipeline(a.src).load_features(start or "2025-01-01", end or "2099-12-31")
        df = ensure_utc_index(df)
    else:
        df = load_feature_frame("5s", start, end)
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
        if start and day < start:
            continue
        if end and day > end:
            continue
        high = part["cs_high"] if "cs_high" in part else part["_close"]
        low = part["cs_low"] if "cs_low" in part else part["_close"]
        jobs.append((day, part.index.asi8, part["_close"].to_numpy(), high.to_numpy(),
                     low.to_numpy(), params))

    cfg = get_vm_config()
    workers = a.workers if a.workers is not None else cfg.workers_for(mem_per_worker_gb=2.0)
    print(cfg)
    print(f"⏳ 15s dual-binary labels: {len(jobs)} days, workers={workers}")
    started = time.time()
    n_rows = n_long = n_short = done = 0
    frames = []
    if not jobs:
        print("nothing to do")
        return 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one_day, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            _, frame = rec
            frames.append(frame)
            n_rows += len(frame)
            n_long += int(frame["y_long"].sum())
            n_short += int(frame["y_short"].sum())
            done += 1
            frac = done / len(jobs)
            fill = int(frac * _BAR)
            el = time.time() - started
            sys.stderr.write(
                f"\r[{'█' * fill}{'░' * (_BAR - fill)}] {frac * 100:5.1f}% "
                f"— {done}/{len(jobs)} days, {el:.0f}s   ")
            sys.stderr.flush()
    sys.stderr.write("\n")
    parts = write_label_parts(out_dir, frames, resolution="15s", force=a.force)
    el = time.time() - started
    print(
        f"✅ Done in {el:.0f}s — {n_rows} rows, y_long=1:{n_long}, "
        f"y_short=1:{n_short}, parts={len(parts)} → {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
