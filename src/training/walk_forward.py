"""Walk-forward validation for the detection-2 model.

Trains on an expanding/sliding window that advances one calendar month at a
time; evaluates on the next month. Produces a JSON summary of per-fold test
metrics so we can see how the model generalises over time instead of trusting
a single temporal split.

Run:
    VTECH_BBO_PRIMARY_ONLY=1 python -m src.training.walk_forward \
        --start=2025-10-01 --end=2025-12-31 \
        --train-months=1 --test-months=1 \
        --out=/opt/vtech/data/legacy/checkpoints/walk_forward_v008

Each fold runs the full training pipeline (~20–25 min on 4 vCPU for ~1 month
of data), so expect ~1.5–2 hours for 3 folds on Oct–Dec. Folds run sequentially
to avoid RAM contention.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.training.config import MLConfig
from src.training.trainer import Trainer

logger = logging.getLogger("vtech.training.walk_forward")


def month_starts(start: str, end: str) -> list[date]:
    """Month-start dates within [start, end], inclusive."""
    return [d.date() for d in pd.date_range(start, end, freq="MS")]


def build_folds(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    expanding: bool,
) -> list[dict]:
    """Return list of {train_start, train_end, test_start, test_end} folds."""
    starts = month_starts(start, end)
    folds = []
    origin = pd.to_datetime(start).date()
    for i in range(train_months, len(starts)):
        test_start = starts[i]
        test_end = (pd.to_datetime(test_start) + pd.offsets.MonthEnd(test_months)).date()
        if test_end > pd.to_datetime(end).date():
            break
        train_end = (pd.to_datetime(test_start) - pd.Timedelta(days=1)).date()
        train_start = origin if expanding else starts[i - train_months]
        folds.append({
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
        })
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--train-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--expanding", action="store_true",
                        help="Use expanding window (default: sliding)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    folds = build_folds(args.start, args.end, args.train_months, args.test_months, args.expanding)
    logger.info("folds", extra={"n": len(folds), "folds": folds})

    config = MLConfig.from_env()
    summary = {"config": {"train_months": args.train_months, "test_months": args.test_months,
                          "expanding": args.expanding, "folds": folds}, "results": []}

    for i, fold in enumerate(folds, 1):
        logger.info(f"fold_start_{i}", extra=fold)
        fold_dir = out / f"fold_{i:02d}_{fold['test_start']}"
        trainer = Trainer(config)
        # Use test_end+1 day because trainer uses [start, end) business-day range
        train_end_plus = (pd.to_datetime(fold["test_end"]) + pd.Timedelta(days=1)).date()
        try:
            metrics = trainer.run(
                start_date=fold["train_start"],
                end_date=str(train_end_plus),
                save_dir=str(fold_dir),
            )
        except Exception as e:
            logger.error(f"fold_failed_{i}", extra={"error": str(e)})
            summary["results"].append({**fold, "error": str(e)})
            continue
        summary["results"].append({
            **fold,
            "test_accuracy": metrics["test_accuracy"],
            "test_f1_macro": metrics["test_f1_macro"],
            "test_f1_per_class": metrics["test_f1_per_class"],
            "training_time_seconds": metrics["training_time_seconds"],
            "epochs": len(metrics["history"]["accuracy"]),
        })
        logger.info(f"fold_done_{i}", extra={
            "acc": metrics["test_accuracy"], "f1m": metrics["test_f1_macro"],
        })
        # persist after each fold
        with open(out / "walk_forward_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    logger.info("walk_forward_done", extra={"n_folds": len(summary["results"])})


if __name__ == "__main__":
    main()
