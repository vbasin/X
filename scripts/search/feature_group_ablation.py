#!/usr/bin/env python3
"""Feature-group ablation study.

Each trial enables EXACTLY ONE feature group (all others disabled) and trains
with the current v014 model/training settings from config.yaml.  The objective
is test_dir_f1 = (BIG_DOWN_F1 + BIG_UP_F1) / 2  (directional F1, ignoring NO_SIGNAL).

This answers: "Which individual feature group carries the most predictive signal
when trained in isolation?"

Study name : feature_group_ablation_v001  (separate from nq_momentum_v2)
Trials     : one per group — all 21 groups are systematically enqueued so the
             study is deterministic (no Bayesian sampling needed).

Usage:
    # Full run (all 21 groups) — default fold-3 window, 6 parallel workers
    python scripts/feature_group_ablation.py

    # Custom window or fewer workers
    python scripts/feature_group_ablation.py \\
        --train-start 2025-06-01 --train-end 2025-10-31 \\
        --n-jobs 3

    # Inspect results after run (or mid-run)
    python scripts/feature_group_ablation.py --show-best

    # Resume if interrupted (reruns enqueue missing groups automatically)
    python scripts/feature_group_ablation.py

Notes:
    All model / training / label / data hyperparameters are taken verbatim from
    config.yaml (the v014 baseline).  Only enabled_groups changes per trial.
    Wavelets and economic_calendar are currently disabled in config.yaml but are
    still tested here so every group gets a fair ablation result.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
sys.path.insert(0, "/opt/vtech")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.training.config import MLConfig
from src.training.trainer import Trainer

logger = logging.getLogger("vtech.ablation")

# ── All 21 feature groups in the canonical order from FeatureConfig ────────
ALL_GROUPS: list[str] = [
    "book_pressure",
    "order_flow",
    "trade_location",
    "same_side",
    "large_trade_cvd",
    "microstructure",
    "trade_arrival",
    "realized_vol",
    "sub_bar_dynamics",
    "volume_profile",
    "iv_surface",
    "dealer_gex",
    "daily_context",
    "wavelets",
    "candle_structure",
    "time_context",
    "vwap",
    "cross_asset",
    "economic_calendar",
    "hawkes_clustering",
    "higher_timeframe",
]


def objective(
    trial: optuna.Trial,
    train_start: str,
    train_end: str,
    n_threads_per_worker: int = 0,
    study_name: str = "feature_group_ablation",
) -> float:
    """Single trial: enable one feature group, train, return dir_f1 = (BD_F1 + BU_F1) / 2."""
    t0 = time.perf_counter()
    pid = os.getpid()
    group: str = trial.suggest_categorical("feature_group", ALL_GROUPS)
    logger.info("trial_start  group=%-22s  trial=%d  pid=%d", group, trial.number, pid)

    # Load the full v014 config from config.yaml (model, training, labels, data)
    config = MLConfig.from_env()

    # Override: disable all groups except the one under test
    config.features.enabled_groups = {g: (g == group) for g in ALL_GROUPS}

    # Thread pinning — applied whenever a non-zero value is passed
    # (works for both parallel workers and single-worker cap).
    if n_threads_per_worker > 0:
        config.training.intraop_threads = n_threads_per_worker

    # Skip per-epoch IG logging — no signal value for ablation, ~15s/epoch savings
    config.training.use_ig = False

    data_dir = os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data")
    from src.common.config import _load_yaml
    from pathlib import Path as _Path
    _cfg = _load_yaml(_Path("/opt/vtech/config.yaml"))
    program_id = _cfg.get("snapshot", {}).get("program_id", "nq-50pt-08-15-cst")
    save_dir = f"{data_dir}/legacy/research_runs/program={program_id}/run_id={study_name}/trial={group}"

    trainer = Trainer(config)
    try:
        metrics = trainer.run(train_start, train_end, save_dir=save_dir, data_dir=data_dir)
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("trial_failed group=%s error=%s", group, exc)
        raise optuna.TrialPruned()

    per_class = metrics.get("test_f1_per_class", {})
    bd_f1 = per_class.get("BIG_DOWN", 0.0)
    bu_f1 = per_class.get("BIG_UP", 0.0)
    dir_f1 = (bd_f1 + bu_f1) / 2.0
    elapsed_min = (time.perf_counter() - t0) / 60.0
    logger.info(
        "trial_done  group=%-22s  dir_f1=%.4f  bd_f1=%.4f  bu_f1=%.4f  pid=%d  elapsed_min=%.1f",
        group, dir_f1, bd_f1, bu_f1, pid, elapsed_min,
    )
    return dir_f1


def _already_queued_groups(study: optuna.Study) -> set[str]:
    """Return groups that already have a trial (queued, running, or completed)."""
    queued: set[str] = set()
    for t in study.trials:
        g = t.params.get("feature_group") or (
            t.system_attrs.get("fixed_params", {}).get("feature_group")
        )
        if g:
            queued.add(g)
    return queued


def show_best(study_name: str, storage: str) -> None:
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials if t.value is not None]
    print(f"\nStudy : {study_name}")
    print(f"Trials: {len(completed)} completed / {len(study.trials)} total\n")
    if not completed:
        print("No completed trials yet.")
        return

    ranked = sorted(completed, key=lambda t: t.value or 0.0, reverse=True)
    print(f"{'Rank':<5} {'Group':<25} {'Dir F1 (BD+BU)/2':>18}")
    print("-" * 50)
    for rank, t in enumerate(ranked, 1):
        g = t.params.get("feature_group", "?")
        print(f"{rank:<5} {g:<25} {t.value:>18.4f}")

    print(f"\nBest: {ranked[0].params.get('feature_group')}  dir_f1={ranked[0].value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature-group ablation: one group enabled per trial."
    )
    parser.add_argument("--show-best", action="store_true",
                        help="Print ranked results and exit (no training)")
    parser.add_argument("--storage", default="sqlite:////opt/vtech/data/optuna_ablation.db",
                        help="Optuna DB (separate from nq_momentum_v2 HPO DB)")
    parser.add_argument("--study-name", default="Features Test")
    parser.add_argument("--train-start", default="2025-06-01")
    parser.add_argument("--train-end", default="2025-10-31",
                        help="Trainer splits this 70/15/15 internally; Nov 2025 stays unseen.")
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="Parallel Optuna workers. Each trial loads ~16 GB RSS; "
                             "6 workers × 16 GB = 96 GB, safe on this VM.")
    parser.add_argument("--total-cpus", type=int, default=0,
                        help="Total CPUs (0 = auto-detect) for thread-pinning.")
    parser.add_argument("--intraop-threads", type=int, default=0,
                        help="Force intraop_threads to this value for every trial, "
                             "regardless of n_jobs. 0 = derive from total-cpus/n-jobs. "
                             "Useful when another process is already using cores.")
    parser.add_argument("--groups", default="",
                        help="Comma-separated groups to force-enqueue, e.g. 'volume_profile,vwap'. "
                             "Bypasses the already-queued check so failed/completed groups "
                             "can be re-run without changing the study name.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.show_best:
        show_best(args.study_name, args.storage)
        return

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    # Enqueue trials: either force-enqueue specific groups (--groups) or
    # auto-enqueue any group not yet seen in the study (resumable default).
    if args.groups:
        target_groups = [g.strip() for g in args.groups.split(",") if g.strip() in ALL_GROUPS]
        invalid = [g.strip() for g in args.groups.split(",") if g.strip() and g.strip() not in ALL_GROUPS]
        if invalid:
            logger.warning("Unknown groups ignored: %s", invalid)
        for g in target_groups:
            study.enqueue_trial({"feature_group": g})
        logger.info("Force-enqueued %d group(s): %s", len(target_groups), target_groups)
        already: set[str] = set()
        to_enqueue = target_groups
    else:
        # Default: skip groups already in the study so the script is safely
        # re-runnable / resumable after interruption.
        already = _already_queued_groups(study)
        to_enqueue = [g for g in ALL_GROUPS if g not in already]
        if to_enqueue:
            logger.info("Enqueueing %d group(s): %s", len(to_enqueue), to_enqueue)
            for g in to_enqueue:
                study.enqueue_trial({"feature_group": g})
        else:
            logger.info("All groups already enqueued — resuming remaining trials.")

    n_trials = len(to_enqueue) or sum(
        1 for t in study.trials
        if t.state in (optuna.trial.TrialState.WAITING, optuna.trial.TrialState.RUNNING)
    )

    print(f"\nStudy   : {args.study_name}")
    print(f"Window  : {args.train_start} → {args.train_end}")
    print(f"Groups  : {len(ALL_GROUPS)} total  ({len(already)} already enqueued, "
          f"{len(to_enqueue)} newly added)")
    print(f"Workers : {args.n_jobs}\n")

    total_cpus = args.total_cpus or multiprocessing.cpu_count()
    if args.intraop_threads > 0:
        threads_per_worker = args.intraop_threads
    elif args.n_jobs > 1:
        threads_per_worker = total_cpus // args.n_jobs
    else:
        threads_per_worker = 0
    if threads_per_worker > 0:
        print(f"Thread pinning: {threads_per_worker} intraop threads/worker\n")

    study.optimize(
        lambda trial: objective(
            trial, args.train_start, args.train_end,
            n_threads_per_worker=threads_per_worker,
            study_name=args.study_name,
        ),
        n_trials=n_trials if n_trials > 0 else len(ALL_GROUPS),
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    print()
    show_best(args.study_name, args.storage)


if __name__ == "__main__":
    main()
