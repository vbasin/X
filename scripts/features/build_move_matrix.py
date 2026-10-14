#!/usr/bin/env python3
"""Stage 1 of the cascade: the **move matrix**, built entirely from the print tape.

One row per ``(decision ts, side)``. No target, no stop, no bracket, no P&L. The
label is :func:`src.training.move_label.label_moves` — did price run ``move_r`` my
way before it went ``adverse_r`` against me — and the features are the ``es_`` and
``fs_`` families that describe the state of the auction at the decision.

Deliberately absent: every ``tgt_*`` and ``stp_*`` column. A model that can see how
far away the target is will learn that distant targets are worth more points, which
is the conflation this stage exists to remove. Whether a move is coming and how much
the move is worth are separate questions, answered by separate heads.

**Everything is derived from TBBO prints rather than 1-second OHLCV bars**, which
removes four approximations at once:

* volume-at-price is observed, not smeared uniformly across each bar's range, so
  the POC and value-area edges land where trade actually concentrated;
* aggressor delta is per-print, so :func:`allocate_tf_delta`'s redistribution of a
  15-second aggregate by volume share is gone;
* the large-print taxonomy reads the **lead contract** resolved per session, where
  ``large_prints.parquet`` matched ``symbol LIKE 'NQ%'`` and so mixed back months
  and calendar spreads into the same feature;
* excursions resolve at print granularity, so a move inside one second is visible.

Usage::

    python scripts/features/build_move_matrix.py --start 2025-06-01 --end 2026-06-30
    python scripts/features/build_move_matrix.py --start 2025-06-01 --end 2026-06-30 \
        --decision-grid 5 --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.session_path import (  # noqa: E402
    prepare_path,
    session_grid as _session_grid,
)
from src.common.vm_config import get_vm_config  # noqa: E402
from src.external_models.candles import load_lead_ohlc  # noqa: E402
from src.features.cash_session_profile import LevelConfig, SessionLevels  # noqa: E402
from src.features.flow_state import (  # noqa: E402
    DIRECTIONAL as FLOW_DIRECTIONAL, FlowConfig, build_flow_state,
)
from src.features.session_profile import (  # noqa: E402
    history_before, load_levels, prior_row, r_before,
)
from src.features.structural_transitions import (  # noqa: E402
    DIRECTIONAL as ST_DIRECTIONAL, build_structural_transitions,
)
from src.features.tape import list_tape_files, load_tape, tape_flow_bars, tape_to_path  # noqa: E402
from src.training.auction_levels import ProfileConfig  # noqa: E402
from src.training.entry_state import DIRECTIONAL, build_entry_state  # noqa: E402
from src.training.move_label import (  # noqa: E402
    SHORT, MoveConfig, cash_slice, label_moves,
)

logger = logging.getLogger("vtech.scripts.build_move_matrix")

# Measured peak RSS per shard is ~340 MB (one tape + its feature frames), so the
# 7 GB feature-engine budget behind workers_for()'s default would idle cores.
MEM_PER_WORKER_GB = 1.5


def _session_date(p: Path) -> date:
    return date.fromisoformat(p.stem.split("_", 1)[1])


def build_session(tape: pd.DataFrame, session: date, r_prior: float,
                  prior: SessionLevels | None, prior_close: float | None,
                  mcfg: MoveConfig, pcfg: ProfileConfig,
                  candles_1m: pd.DataFrame | None = None,
                  level_hist: list[SessionLevels] | None = None,
                  lcfg: LevelConfig | None = None) -> pd.DataFrame:
    """Labelled, featurised rows for one session.

    *candles_1m* is left-labelled 1-minute OHLC spanning (at least) this
    session, e.g. the ``databento.tbbo_1min`` cagg; it feeds
    ``es_atr_r``/``es_realized_vol`` and is sliced to this session's own span
    before being handed down. *level_hist* is the multi-session history
    (newest first) that :func:`build_entry_state` clusters into structural
    bounds -- the same historic volume-profile structure Stage 3 targets
    against, so Stage 1 sees it too rather than only the single prior
    session's levels.
    """
    lab = label_moves(tape, mcfg, r_prior)
    if lab.empty:
        return pd.DataFrame()

    day = tape_to_path(tape)
    grid = _session_grid(prepare_path(day))

    bars = pd.DatetimeIndex(np.sort(lab["ts"].unique()))
    entries = grid.index.get_indexer(bars)
    keep = entries >= 0
    if not keep.any():
        return pd.DataFrame()
    entries, bars = entries[keep], bars[keep]
    lab = lab[lab["ts"].isin(bars)]

    # EMA60 + a slope window, plus es_atr_r's 5 completed minutes. Prior
    # Globex minutes (08:29, 08:28, …) stay in the lookback (ADR-137).
    candle_warmup = pd.Timedelta(minutes=120)
    sess_candles = None
    warmup_close_1m = None
    if candles_1m is not None and not candles_1m.empty:
        lo_ts = day.index.min() - candle_warmup
        hi_ts = day.index.max()
        sess_candles = candles_1m[(candles_1m.index >= lo_ts) & (candles_1m.index <= hi_ts)]
        prior_1m = candles_1m[
            (candles_1m.index >= lo_ts) & (candles_1m.index < day.index.min())
        ]
        if not prior_1m.empty and "close" in prior_1m.columns:
            warmup_close_1m = prior_1m["close"]

    es = build_entry_state(tape, grid, entries, r_value=r_prior,
                           prior_poc=getattr(prior, "poc", None),
                           prior_vah=getattr(prior, "vah", None),
                           prior_val=getattr(prior, "val", None),
                           prior_hi=getattr(prior, "hi", None),
                           prior_lo=getattr(prior, "lo", None),
                           prior_close=prior_close,
                           hive=tape_flow_bars(tape), candles_1m=sess_candles,
                           level_hist=level_hist, lcfg=lcfg, pcfg=pcfg)
    merged = lab.merge(es, on="ts", how="inner")
    if merged.empty:
        return pd.DataFrame()

    cash = cash_slice(tape, "08:30", "15:00")
    if not cash.empty:
        fb = tape_to_path(cash)
        prints = cash.reset_index().rename(columns={"ts": "ts_event"})
        prints["side"] = np.where(prints["side"] > 0, "B", "A")
        prints = prints.set_index("ts_event")
        fs = build_flow_state(
            fb, prints, FlowConfig(), r_value=r_prior,
            warmup_close_1m=warmup_close_1m,
        )
        if not fs.empty:
            merged = merged.merge(fs.reset_index().rename(columns={"index": "ts"}),
                                  on="ts", how="left")
        # Structural transitions read the same 1s frame and tag their level
        # touches with the flow state measured at the touching bar.
        st = build_structural_transitions(
            fb, prior, r_value=r_prior, trades=cash,
            accept=fs["fs_accept_continue"] if "fs_accept_continue" in fs else None,
            reject=fs["fs_reject_reversal"] if "fs_reject_reversal" in fs else None)
        if not st.empty:
            merged = merged.merge(st.reset_index().rename(columns={"index": "ts"}),
                                  on="ts", how="left")

    # Features are built in the absolute frame; mirror the signed columns so a
    # short is a long in a rotated frame and one model serves both sides.
    short = merged["side"].to_numpy() == SHORT
    if short.any():
        cols = [c for c in (*DIRECTIONAL, *FLOW_DIRECTIONAL, *ST_DIRECTIONAL)
                if c in merged.columns]
        merged.loc[short, cols] = -merged.loc[short, cols]
    merged["session"] = pd.Timestamp(session)
    return merged


def run_shard(sel_paths: list[str], mcfg: MoveConfig, pcfg: ProfileConfig,
              r_lookback: int, out_dir: str, lcfg: LevelConfig,
              canonical_write: bool = False) -> tuple[int, int]:
    """Build every session in one shard.

    Prior-session levels and the R median come from the session-profile SSOT, so
    a shard carries no state across sessions and needs no warmup: any set of
    sessions can be handed to any worker. *canonical_write* switches the
    per-session filename from the legacy ``move_<date>.parquet`` to the
    canonical, family-agnostic ``part_<date>.parquet``
    (``label_store.outcome_part_name`` -- docs/STORAGE_CONTRACT.md §18); the
    directory this writes into is already the caller's concern (``out_dir``).
    """
    from src.training.label_store import outcome_part_name
    levels = load_levels()
    n_rows = n_sess = 0
    out = Path(out_dir)

    dates = [_session_date(Path(p)) for p in sel_paths]
    try:
        candles_1m = load_lead_ohlc(min(dates), max(dates), root="NQ",
                                    timeframe="1min", warmup_days=2)
    except Exception:
        logger.warning("databento.tbbo_1min unavailable for shard %s..%s; "
                       "es_atr_r/es_realized_vol will be NaN for this shard",
                       min(dates), max(dates), exc_info=True)
        candles_1m = None

    for p in sel_paths:
        d = _session_date(Path(p))
        r_prior = r_before(levels, d, r_lookback)
        prow = prior_row(levels, d)
        if not np.isfinite(r_prior) or prow is None:
            continue
        tape = load_tape(d)
        if tape is None or tape.empty:
            continue
        prior = SessionLevels(session=prow.session, lo=float(prow.lo),
                              val=float(prow.val), poc=float(prow.poc),
                              vah=float(prow.vah), hi=float(prow.hi))
        hist = history_before(levels, d, lcfg.max_lookback_sessions)
        m = build_session(tape, d, r_prior, prior, float(prow.close), mcfg, pcfg,
                          candles_1m=candles_1m, level_hist=hist, lcfg=lcfg)
        if not m.empty:
            name = outcome_part_name(d) if canonical_write else f"move_{d.isoformat()}.parquet"
            m.to_parquet(out / name, index=False, compression="zstd")
            n_rows += len(m)
            n_sess += 1
    return n_sess, n_rows


def _shard(sel: list[Path], n: int) -> list[list[Path]]:
    """Split *sel* into *n* contiguous shards of roughly equal size."""
    if n <= 1 or len(sel) <= n:
        return [sel]
    return [list(c) for c in np.array_split(np.array(sel, dtype=object), n) if len(c)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--decision-grid", type=int, default=MoveConfig.decision_grid_seconds)
    ap.add_argument("--horizon", type=int, default=MoveConfig.horizon_seconds)
    ap.add_argument("--move-r", type=float, default=MoveConfig.move_r)
    ap.add_argument("--adverse-r", type=float, default=MoveConfig.adverse_r)
    ap.add_argument("--r-lookback-sessions", type=int, default=5,
                    help="sessions in the R median; ATR-like, current conditions")
    ap.add_argument("--lookback-sessions", type=int, default=LevelConfig.lookback_sessions,
                    help="sessions clustered into the es_n_bounds/es_vs_bound_* "
                         "structural bounds (same structure Stage 3 targets against)")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: all cores the RAM budget allows")
    ap.add_argument("--out", default=None,
                    help="explicit output directory for the per-session files. "
                         "Omit this and pass --objective to write the run layout")
    ap.add_argument("--objective", default=None,
                    help="objective_id; writes the training matrix into "
                         "pass_N/model_<model_id>/")
    ap.add_argument("--run-ts", default=None,
                    help="run timestamp (default: now, UTC). Used with --objective")
    ap.add_argument("--pass-n", type=int, default=1, dest="pass_n",
                    help="pass number inside the run (default 1)")
    ap.add_argument("--model-id", default=None,
                    help="unique id for the model that trains on this matrix. "
                         "Omit to allocate one from the model type, hyperparameters, "
                         "feature view, and target. The model type is not an id.")
    ap.add_argument("--model-type", default="xgboost")
    ap.add_argument("--n-estimators", type=int, default=300,
                    help="recorded in the model id. Default matches eval_move_model.py")
    ap.add_argument("--max-depth", type=int, default=6,
                    help="recorded in the model id. Default matches eval_move_model.py")
    ap.add_argument("--learning-rate", type=float, default=0.05,
                    help="recorded in the model id. Default matches eval_move_model.py")
    ap.add_argument("--seed", type=int, default=42,
                    help="recorded in the model id")
    ap.add_argument("--no-order-flow", action="store_true",
                    help="feature view for the model id. The matrix still contains "
                         "those columns. Default matches eval_move_model.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--canonical-write", action="store_true",
                    help="write to the canonical outcome-dataset path "
                         "(sources/labels/directional_first_passage/<resolution>/"
                         "<contract_id>/, docs/STORAGE_CONTRACT.md §17) instead of "
                         "the legacy move_matrix_v4/ path; --out is ignored when set")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mcfg = MoveConfig(decision_grid_seconds=args.decision_grid,
                      horizon_seconds=args.horizon,
                      move_r=args.move_r, adverse_r=args.adverse_r)
    pcfg = ProfileConfig()
    lcfg = LevelConfig(lookback_sessions=args.lookback_sessions)

    files = list_tape_files()
    lo = date.fromisoformat(args.start)
    hi = date.fromisoformat(args.end)
    sel = [f for f in files if lo <= _session_date(f) <= hi]
    if not sel:
        logger.error("no tape sessions selected")
        return 1

    compact_to = None
    run_ts = None
    if args.objective and (args.out or args.canonical_write):
        logger.error("pass only one of --objective, --out, --canonical-write")
        return 1
    model_id = None
    hyper = None
    feature_spec = None
    if args.objective:
        from src.experiment.model_identity import (
            allocate_model_id, validate_model_id,
        )
        from src.common.run_layout import (
            model_dir, new_run_ts, pass_dir, population_cache_path,
        )
        from src.training.cascade_run_spec import move_feature_spec, xgb_hyperparameters
        run_ts = args.run_ts or new_run_ts()
        hyper = xgb_hyperparameters(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, seed=args.seed)
        feature_spec = move_feature_spec(drop_order_flow=args.no_order_flow)
        model_id = validate_model_id(
            args.model_id or allocate_model_id(
                args.objective, run_ts, args.pass_n, args.model_type,
                hyper, feature_spec=feature_spec, target="y_move"),
            args.model_type)
        scratch = pass_dir(args.objective, run_ts, args.pass_n) / "_sessions"
        scratch.mkdir(parents=True, exist_ok=True)
        out_dir = scratch
        # One population for every model on this pass. Feature subsets select
        # columns at fit time; they do not get another matrix.
        compact_to = population_cache_path(args.objective, run_ts, args.pass_n)
    elif args.canonical_write:
        from src.training.label_store import (
            move_outcome_contract, outcome_set_dir, write_build_sidecar,
            write_outcome_contract,
        )
        contract = move_outcome_contract(
            decision_grid_seconds=args.decision_grid, horizon_seconds=args.horizon,
            move_r=args.move_r, adverse_r=args.adverse_r,
            r_lookback_sessions=args.r_lookback_sessions)
        resolution = f"{args.decision_grid}s"
        out_dir = outcome_set_dir("directional_first_passage", resolution, contract)
        contract_id = write_outcome_contract(out_dir, contract)
        write_build_sidecar(out_dir, population_config={
            "start": args.start, "end": args.end,
            "lookback_sessions": args.lookback_sessions})
        logger.info("canonical-write: contract_id=%s -> %s", contract_id, out_dir)
    elif args.out:
        out_dir = Path(args.out)
    else:
        logger.error("pass --objective (run layout) or --out (explicit directory)")
        return 1
    workers = args.workers or get_vm_config().workers_for(
        mem_per_worker_gb=MEM_PER_WORKER_GB)
    shards = _shard(sel, workers)
    logger.info("move matrix: %d sessions, grid=%ds horizon=%ds "
                "move_r=%.3f adverse_r=%.3f, %d shards -> %s",
                len(sel), args.decision_grid, args.horizon,
                args.move_r, args.adverse_r, len(shards), out_dir)
    if args.dry_run:
        for i, sh in enumerate(shards):
            logger.info("  shard %d: %d sessions %s → %s", i, len(sh),
                        _session_date(sh[0]), _session_date(sh[-1]))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = n_sess = 0
    if len(shards) == 1:
        n_sess, n_rows = run_shard([str(p) for p in shards[0]], mcfg, pcfg,
                                   args.r_lookback_sessions, str(out_dir), lcfg,
                                   canonical_write=args.canonical_write)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(run_shard, [str(p) for p in sh], mcfg, pcfg,
                                args.r_lookback_sessions, str(out_dir), lcfg,
                                args.canonical_write): i
                    for i, sh in enumerate(shards)}
            for done, fut in enumerate(as_completed(futs), 1):
                s, r = fut.result()
                n_sess += s
                n_rows += r
                pct = int(20 * done / len(shards))
                logger.info("[%s%s] shard %d/%d done | %s sessions | %s rows",
                            "#" * pct, "." * (20 - pct), done, len(shards),
                            n_sess, f"{n_rows:,}")

    logger.info("done: %d sessions, %s rows -> %s", n_sess, f"{n_rows:,}", out_dir)
    if compact_to is not None:
        from src.common.data_paths import data_root, labels_dir
        from src.experiment.model_identity import write_model_json
        from src.common.run_layout import compact_parquets, merge_run_json, model_dir
        from src.training.cascade_run_spec import CASCADE_PASS
        from src.objectives.loader import load_objective
        from src.training.label_store import (
            MOVE_LABEL_COLUMNS, move_outcome_contract, outcome_contract_id,
            write_outcome_contract,
        )
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        parts = sorted(out_dir.glob("*.parquet"))
        n = compact_parquets(parts, compact_to)
        for p in parts:
            p.unlink()
        out_dir.rmdir()
        contract = move_outcome_contract(
            decision_grid_seconds=args.decision_grid, horizon_seconds=args.horizon,
            move_r=args.move_r, adverse_r=args.adverse_r,
            r_lookback_sessions=args.r_lookback_sessions)
        label_dir = labels_dir() / outcome_contract_id(contract)
        write_outcome_contract(label_dir, contract)
        have = set(pq.read_schema(compact_to).names)
        keep = [c for c in MOVE_LABEL_COLUMNS if c in have]
        if keep:
            labels_path = label_dir / "labels.parquet"
            pq.write_table(ds.dataset(str(compact_to), format="parquet").to_table(columns=keep),
                           labels_path, compression="zstd")
        rel_labels = str(label_dir.resolve().relative_to(data_root().resolve()))
        cid = outcome_contract_id(contract)
        rel_cache = f"pass_{args.pass_n}/training_matrix.parquet"
        inputs = {
            "labels": {
                "semantic_type": "outcomes",
                "kind": "labels",
                "contract_id": cid,
                "location": rel_labels,
            },
            "features": [{
                "semantic_type": "features",
                "kind": "features",
                "location": rel_cache,
                "availability": "timestamp",
                "time_column": "ts",
            }],
            "join_keys": ["ts", "session", "side"],
            "filters": [],
            "population_cache": rel_cache,
            "target": "y_move",
        }
        write_model_json(
            model_dir(args.objective, run_ts, args.pass_n, model_id),
            model_id=model_id, model_type=args.model_type,
            hyperparameters=hyper, label_contract_id=cid,
            feature_spec=feature_spec, inputs=inputs, target="y_move")
        described = CASCADE_PASS.get(args.pass_n, {})
        try:
            snapshot = load_objective(args.objective)
        except Exception:
            snapshot = {}
        snapshot = dict(snapshot)
        snapshot["resolved_overrides"] = {
            "decision_grid_seconds": args.decision_grid,
            "horizon_seconds": args.horizon,
            "move_r": args.move_r,
            "adverse_r": args.adverse_r,
            "r_lookback_sessions": args.r_lookback_sessions,
        }
        merge_run_json(args.objective, run_ts, {
            "pass": args.pass_n,
            "role": described.get("role", f"pass_{args.pass_n}"),
            "purpose": described.get("purpose", ""),
            "label_contract_id": cid,
            "population_cache": rel_cache,
            "inputs": inputs,
            "upstream": [
                {"kind": "price_tape", "path": "sources/price_tape/NQ/"},
                {"kind": "labels", "label_contract_id": cid, "location": rel_labels,
                 "semantic_type": "outcomes"},
            ],
            "models": [{
                "model_id": model_id,
                "model_type": args.model_type,
                "model_json": f"pass_{args.pass_n}/model_{model_id}/model.json",
            }],
        }, config_snapshot=snapshot, config_snapshot_resolved=bool(snapshot.get("objective_id")))
        logger.info("compacted %s rows -> %s", f"{n:,}", compact_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
