#!/usr/bin/env python3
"""Rebuild one model's analysis table from its manifest and model.json.

The OOF file stays compact. This joins it back to the feature columns,
label source, and context named for that model. The join is as-of: a
feature or context value is attached only when it was knowable at or
before that row's decision timestamp. ``--materialize`` writes
``analysis_view.parquet`` next to the model for VS Code Data Wrangler.
That file is temporary and is not part of the run contract.

Example::

    python scripts/research/build_oof_analysis.py \
        --objective nq-Rscaled-mkt --run-ts 20260923T005724Z \
        --pass-n 1 --model-id 543eedbcf30b --materialize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiment.analysis import (  # noqa: E402
    build_oof_analysis,
    default_analysis_view_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--pass-n", type=int, required=True, dest="pass_n")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--materialize", nargs="?", const="default", default=None,
                    help="write analysis_view.parquet. Omit a path to use the model folder.")
    args = ap.parse_args()
    dest = None
    if args.materialize is not None:
        dest = (default_analysis_view_path(args.objective, args.run_ts, args.pass_n, args.model_id)
                if args.materialize == "default" else Path(args.materialize))
    frame = build_oof_analysis(
        args.objective, args.run_ts, args.pass_n, args.model_id, materialize=dest)
    print(f"rows {len(frame):,}  columns {len(frame.columns)}")
    if dest is not None:
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
