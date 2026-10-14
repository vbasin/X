#!/usr/bin/env python3
"""Stage 3 of the cascade: target selection *and* target-adaptive stop placement.

Runs **only** on decision bars that Stage 1 gated through, which is what makes
the whole design affordable.

Since ADR-130 (2026-09-15) Stage 2 has been folded in: the stop is no longer a
fixed 0.45 R distance, it is derived per-target from the target itself —

    stop_dist_r = min(stop_ratio * tgt_dist_r, max_stop_r)

Default ``stop_ratio = 0.5`` gives a 2:1 R:R at short distances (a 0.25 R target
gets a 0.125 R stop); ``max_stop_r = 0.25`` R caps risk on distant targets so
a 7 R target still only risks 0.25 R (28:1 R:R). Because the stop level now
depends on which target is being tested, every ``(ts, side, target)`` row scans
the tape with its **own** ``stop_px`` — which is precisely why the previous
separation of Stage 2 (per-bar stop) and Stage 3 (per-target target) no longer
fits the row grain.

The label is still ``y_reach`` — did price touch this target before that
target's own stop — and each row carries ``t_reach`` (seconds to the target,
``-1`` if never inside the horizon) and ``t_stop`` (seconds to the stop, ``-1``
if never inside the horizon). Row grain is still ``(ts, side, target_bound)``.

``tgt_dist_pts`` is deliberately excluded from the feature set while
``tgt_dist_r`` is kept: risk-normalised distance is legitimately informative,
raw point distance is the channel through which "further away is worth more"
leaks back in.

Usage::

    python scripts/features/build_target_matrix.py \\
        --scores data/sources/labels/move_scores.parquet \\
        --tau-q 0.95 --stop-ratio 0.5 --max-stop-r 0.25 \\
        --out data/sources/labels/target_matrix
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

from src.common.vm_config import get_vm_config  # noqa: E402
from src.features.cash_session_profile import (  # noqa: E402
    LevelConfig, entry_levels, levels_for_price, next_bounds,
)
from src.features.session_profile import (  # noqa: E402
    history_before, load_levels, r_before,
)
from src.features.tape import (  # noqa: E402
    first_cross, list_tape_files, load_tape, sparse_tables,
)
from src.training import cascade_policy as policy  # noqa: E402
from src.training.move_label import LONG, SHORT  # noqa: E402

logger = logging.getLogger("vtech.scripts.build_target_matrix")

# Stage 1 compresses the auction into "will a move happen"; Stage 3 needs it
# uncompressed to answer "how far will it go". Excluding p_move keeps the two
# *gates* independent — excluding the underlying features only left the target
# selector blind to trend, which is why it could never prefer a distant target.
CONTEXT: tuple[str, ...] = (
    "fs_vwap_dist_r", "fs_vwap_velocity_r", "fs_ema6_slope_r",
    "fs_vwap_velocity_5m_r", "fs_vwap_velocity_15m_r",
    "fs_ema20_slope_r", "fs_ema60_slope_r",
    "fs_vwap_ema_spread_r", "fs_ema20_vwap_diff_r", "fs_ema60_vwap_diff_r",
    "fs_ema6_slope_vwap_diff_r", "fs_ema20_slope_vwap_diff_r", "fs_ema60_slope_vwap_diff_r",
    "fs_ema6_slope_vwap_diff_accel_r", "fs_ema20_slope_vwap_diff_accel_r",
    "fs_ema60_slope_vwap_diff_accel_r",
    "es_ret_5m", "es_ret_15m", "es_cvd_slope_5m", "es_delta_1m", "es_delta_5m",
    "fs_flow_score", "fs_flow_score_5bar",
    "st_zone", "st_zone_dwell_s", "st_break_age_s", "st_break_level",
    "st_break_extension_r", "st_break_pullback_r", "st_reclaim_flag",
    "st_reject_flag", "st_bounce_confirm", "st_bounce_accept",
    "st_va_entry_age_s", "st_va_entry_depth_r",
    "st_vwap_cross_age_s", "st_vwap_cross_strength", "st_vwap_reclaim_flag",
    "st_level_touch_kind", "st_level_touch_accept",
    "st_dist_prior_hi_r", "st_dist_prior_lo_r",
)
CONTEXT_PREFIXES = ("es_", "fs_", "st_")

# One tape plus its sparse range tables per shard; measured well under 1 GB.
MEM_PER_WORKER_GB = 1.5


def _session_date(p: Path) -> date:
    return date.fromisoformat(p.stem.split("_", 1)[1])


def _targets(bounds, price: float, side: int, max_n: int):
    """Bounds ahead of price on the trade's side, nearest first."""
    if side == LONG:
        ahead = sorted((b for b in bounds if b.lo > price), key=lambda b: b.lo)
    else:
        ahead = sorted((b for b in bounds if b.hi < price), key=lambda b: -b.hi)
    return ahead[:max_n]


def build_session(tape: pd.DataFrame, gated: pd.DataFrame, session: date,
                  level_hist: list, r_prior: float,
                  lcfg: LevelConfig, stop_ratio: float, max_stop_r: float,
                  max_targets: int, horizon: int, ctx_cols: list[str],
                  resting_entry: bool, max_fill_wait: int) -> pd.DataFrame:
    """One row per (gated decision, candidate target bound).

    Each row's stop is derived from that row's target distance:
    ``stop_dist_r = min(stop_ratio * tgt_dist_r, max_stop_r)``. That means the
    same decision bar can have five different stop levels — one per candidate
    target — so the tape scan is vectorised over targets rather than shared
    across the row family.

    With *resting_entry* the order is placed at the far edge of the containing
    bound (ADR-126) and the row is **dropped entirely** when it never fills:
    a trade that did not happen is not a flat one. Geometry is then restated
    against the fill, while features stay measured at the decision bar.
    """
    if gated.empty:
        return pd.DataFrame()

    ts_ns = tape.index.values.astype("int64")
    px = tape["price"].to_numpy(dtype="float64")
    hi_tbl, lo_tbl = sparse_tables(px)
    horizon_ns = int(horizon) * 1_000_000_000
    wait_ns = int(max_fill_wait) * 1_000_000_000

    rows: list[dict] = []
    bounds = None
    for _, g in gated.iterrows():
        ts = g["ts"]
        side = int(g["side"])
        price = float(g["price"])
        if bounds is None or (bounds and (price > max(b.hi for b in bounds)
                                          or price < min(b.lo for b in bounds))):
            bounds, _ = levels_for_price(level_hist, price, lcfg)
        if not bounds:
            continue

        i0 = int(np.searchsorted(ts_ns, ts.value, side="left"))
        if i0 >= len(px):
            continue

        if resting_entry:
            long_px, short_px = entry_levels(bounds, price, lcfg)
            entry_px = float(long_px if side == LONG else short_px)
            i_wait = int(min(np.searchsorted(ts_ns, ts.value + wait_ns,
                                             side="right") - 1, len(px) - 1))
            if i_wait <= i0:
                continue
            i_fill = int(first_cross(hi_tbl if side == LONG else lo_tbl,
                                     np.array([i0]), np.array([i_wait]),
                                     np.array([entry_px]), side == LONG)[0])
            if i_fill < 0:
                continue
        else:
            entry_px, i_fill = price, i0

        fill_ns = int(ts_ns[i_fill])
        t_fill = int((fill_ns - ts.value) // 1_000_000_000)

        inside, _above, _below = next_bounds(bounds, entry_px)
        tgts = _targets(bounds, entry_px, side, max_targets)
        if not tgts:
            continue

        i1 = int(min(np.searchsorted(ts_ns, fill_ns + horizon_ns,
                                     side="right") - 1, len(px) - 1))
        if i1 <= i_fill:
            continue

        k = len(tgts)
        tgt_lvl = np.array(
            [float(b.lo if side == LONG else b.hi) for b in tgts],
            dtype="float64",
        )
        tgt_dist_r_arr = np.abs(tgt_lvl - entry_px) / max(r_prior, 1e-9)
        stp_dist_r_arr = np.minimum(stop_ratio * tgt_dist_r_arr, max_stop_r)
        stop_lvl = entry_px - side * stp_dist_r_arr * r_prior

        starts = np.full(k, i_fill, dtype="int64")
        ends = np.full(k, i1, dtype="int64")

        tgt_tbl = hi_tbl if side == LONG else lo_tbl
        stop_tbl = lo_tbl if side == LONG else hi_tbl
        i_tgt_arr = first_cross(tgt_tbl, starts, ends, tgt_lvl, side == LONG)
        i_stop_arr = first_cross(stop_tbl, starts, ends, stop_lvl, side == SHORT)

        ctx = {c: float(g[c]) for c in ctx_cols if c in g.index}
        for rank, b in enumerate(tgts):
            i_tgt = int(i_tgt_arr[rank])
            i_stop = int(i_stop_arr[rank])
            reached = i_tgt >= 0 and (i_stop < 0 or i_tgt <= i_stop)
            rows.append({
                "ts": ts, "side": side, "price": float(entry_px),
                "decision_price": price,
                "t_fill": t_fill,
                "stop_px": float(stop_lvl[rank]),
                "tgt_level": float(tgt_lvl[rank]),
                "y_reach": int(reached),
                "t_reach": int((ts_ns[i_tgt] - fill_ns) // 1_000_000_000)
                if i_tgt >= 0 else -1,
                "t_stop": int((ts_ns[i_stop] - fill_ns) // 1_000_000_000)
                if i_stop >= 0 else -1,
                "p_move": float(g["p_move"]),
                "tgt_rank": rank,
                "tgt_dist_r": float(tgt_dist_r_arr[rank]),
                "stp_dist_r": float(stp_dist_r_arr[rank]),
                "tgt_strength": float(b.strength),
                "tgt_n_levels": int(b.n_levels),
                "tgt_width_r": (b.hi - b.lo) / max(r_prior, 1e-9),
                "tgt_newest_age": int(b.newest_age),
                "tgt_frac_value_edges": (b.kinds.get("vah", 0)
                                         + b.kinds.get("val", 0))
                / max(b.n_levels, 1),
                "tgt_frac_poc": b.kinds.get("poc", 0) / max(b.n_levels, 1),
                "n_bounds": len(bounds),
                "inside_bound": int(inside is not None),
                "r_value": r_prior,
                "session": pd.Timestamp(session),
                **ctx,
            })
    return pd.DataFrame(rows)


def _context_columns(move_dir: Path, mode: str) -> list[str]:
    """Which pass-1 feature columns ride through onto this pass's rows."""
    if mode == "none":
        return []
    import pyarrow.parquet as pq
    if move_dir.is_file():
        sample = move_dir
    else:
        found = sorted(move_dir.glob("move_*.parquet")) or sorted(
            move_dir.glob("training_matrix.parquet"))
        if not found:
            raise SystemExit(f"no training matrix under {move_dir}")
        sample = found[-1]
    have = set(pq.read_schema(sample).names)
    if mode == "all":
        return sorted(c for c in have if c.startswith(CONTEXT_PREFIXES))
    return [c for c in CONTEXT if c in have]


def run_shard(days: list[str], gated: pd.DataFrame, move_dir: str, out_dir: str,
              ctx_cols: list[str], lcfg: LevelConfig, opts: dict,
              canonical_write: bool = False) -> tuple[int, int]:
    """Build every session in one shard.

    Level history and the R median come from the session-profile SSOT, so no
    warmup is carried across sessions and any set of dates can go to any worker.
    *canonical_write* switches the per-session filename from the legacy
    ``target_<date>.parquet`` to the canonical ``part_<date>.parquet``
    (``label_store.outcome_part_name`` -- docs/STORAGE_CONTRACT.md §18).
    """
    from src.training.label_store import outcome_part_name
    levels = load_levels()
    move = Path(move_dir)
    out = Path(out_dir)
    n_rows = n_sess = 0
    context_frame = None
    if ctx_cols and move.is_file():
        context_frame = pd.read_parquet(move, columns=["ts", "side", "session", *ctx_cols])
        context_frame["ts"] = pd.to_datetime(context_frame["ts"], utc=True)
        context_frame["session"] = pd.to_datetime(context_frame["session"])

    for iso in days:
        d = date.fromisoformat(iso)
        g = gated[gated["sess_date"] == d]
        if g.empty:
            continue
        r_prior = r_before(levels, d, opts["r_lookback"])
        hist = history_before(levels, d, lcfg.max_lookback_sessions)
        if not np.isfinite(r_prior) or len(hist) < opts["lookback_sessions"]:
            continue
        tape = load_tape(d)
        if tape is None or tape.empty:
            continue

        g = g.sort_values("ts", kind="mergesort")
        if ctx_cols:
            if context_frame is not None:
                mm = context_frame.loc[
                    context_frame["session"].dt.date == d, ["ts", "side", *ctx_cols]]
                if mm.empty:
                    continue
            else:
                mpath = move / f"move_{d.isoformat()}.parquet"
                if not mpath.exists():
                    continue
                mm = pd.read_parquet(mpath, columns=["ts", "side", *ctx_cols])
                mm["ts"] = pd.to_datetime(mm["ts"], utc=True)
            g = g.merge(mm, on=["ts", "side"], how="left")

        m = build_session(tape, g, d, hist, r_prior, lcfg,
                          opts["stop_ratio"], opts["max_stop_r"],
                          opts["max_targets"], opts["horizon"], ctx_cols,
                          opts["resting_entry"], opts["max_fill_wait"])
        if not m.empty:
            name = outcome_part_name(d) if canonical_write else f"target_{d.isoformat()}.parquet"
            m.to_parquet(out / name, index=False, compression="zstd")
            n_rows += len(m)
            n_sess += 1
    return n_sess, n_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=None,
                    help="previous pass OOF. Omit with --objective and --run-ts "
                         "to read that pass's oof_core_columns from the same run")
    ap.add_argument("--tau-q", type=float, default=0.95,
                    help="Stage-1 train quantile used as the gate")
    ap.add_argument("--max-targets", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=policy.HORIZON_SECONDS)
    ap.add_argument("--lookback-sessions", type=int, default=LevelConfig.lookback_sessions)
    ap.add_argument("--r-lookback-sessions", type=int, default=5)
    ap.add_argument("--stop-ratio", type=float, default=policy.STOP_RATIO,
                    help="stop_dist_r = min(stop_ratio * tgt_dist_r, max_stop_r)")
    ap.add_argument("--max-stop-r", type=float, default=policy.MAX_STOP_R,
                    help="hard cap on stop distance in R units")
    ap.add_argument("--context", choices=("curated", "all", "none"), default="curated",
                    help="Stage 1 feature columns carried onto the target rows")
    ap.add_argument("--move-dir", default=None,
                    help="move matrix dir supplying the context columns")
    ap.add_argument("--resting-entry", action="store_true",
                    help="rest at the far edge of the containing bound (ADR-126); "
                         "decisions that never fill emit no row")
    ap.add_argument("--max-fill-wait", type=int, default=900,
                    help="seconds a resting order stays live before cancelling")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: all cores the RAM budget allows")
    ap.add_argument("--out", default=None,
                    help="explicit output directory. Omit and pass --objective "
                         "to write the run layout")
    ap.add_argument("--objective", default=None,
                    help="objective_id; records this pass on the run manifest")
    ap.add_argument("--run-ts", default=None,
                    help="run timestamp (default: now, UTC). Used with --objective")
    ap.add_argument("--pass-n", type=int, default=2, dest="pass_n",
                    help="pass number inside the run (default 2)")
    ap.add_argument("--model-id", default=None,
                    help="unique id for the model that trains on this matrix. "
                         "Omit to allocate one from the model type, hyperparameters, "
                         "feature view, and target. The model type is not an id.")
    ap.add_argument("--model-type", default="xgboost")
    ap.add_argument("--n-estimators", type=int, default=300,
                    help="recorded in the model id. Default matches eval_target_model.py")
    ap.add_argument("--max-depth", type=int, default=5,
                    help="recorded in the model id. Default matches eval_target_model.py")
    ap.add_argument("--learning-rate", type=float, default=0.05,
                    help="recorded in the model id. Default matches eval_target_model.py")
    ap.add_argument("--seed", type=int, default=42,
                    help="recorded in the model id")
    ap.add_argument("--canonical-write", action="store_true",
                    help="write to the canonical outcome-dataset path "
                         "(sources/labels/candidate_level_reachability/<resolution>/"
                         "<contract_id>/, docs/STORAGE_CONTRACT.md §17) instead of "
                         "the legacy target_matrix/ path; --out is ignored when set")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.context != "none" and not args.move_dir and not (args.objective and args.run_ts):
        logger.error("--move-dir is required when --context != none: no implicit "
                     "historical dataset (e.g. move_matrix_v3) is selected for "
                     "you (docs/STORAGE_INVENTORY.md §4 STOR-2). Pass an explicit "
                     "--move-dir, --objective with --run-ts, or --context none.")
        return 1
    if args.objective and args.pass_n > 1 and not args.run_ts:
        logger.error("pass %d needs --run-ts of the run that already contains the previous pass",
                     args.pass_n)
        return 1
    if args.objective and not args.scores:
        if not args.run_ts:
            logger.error("pass --scores, or --objective with --run-ts to resolve the previous OOF")
            return 1
        from src.common.run_layout import previous_oof_core, read_previous_oof
        oof_path, oof_cols = previous_oof_core(args.objective, args.run_ts, args.pass_n)
        logger.info("previous OOF %s (%d core columns)", oof_path, len(oof_cols))
        scores = read_previous_oof(args.objective, args.run_ts, args.pass_n)
    elif args.scores:
        scores = pd.read_parquet(args.scores)
    else:
        logger.error("pass --scores, or --objective with --run-ts to resolve the previous OOF")
        return 1
    feature_source: Path | None = Path(args.move_dir) if args.move_dir else None
    if args.context != "none" and feature_source is None:
        if args.objective and args.run_ts:
            from src.common.run_layout import resolve_population_cache
            feature_source = resolve_population_cache(
                args.objective, args.run_ts, args.pass_n - 1)
            if feature_source is None:
                logger.error("pass %d needs the previous pass's population to join features",
                             args.pass_n)
                return 1
            logger.info("features join %s at fit time", feature_source)
        else:
            logger.error("--move-dir is required when --context != none")
            return 1
    scores["ts"] = pd.to_datetime(scores["ts"], utc=True)
    tau_col = f"tau_{args.tau_q}"
    if tau_col not in scores.columns:
        logger.error("scores lack %s; have %s", tau_col,
                     [c for c in scores.columns if c.startswith("tau_")])
        return 1
    gated_all = scores[scores["p_move"] >= scores[tau_col]].copy()
    gated_all["sess_date"] = gated_all["session"].dt.date
    logger.info("gate q=%.2f: %s of %s rows (%.3f%%)", args.tau_q,
                f"{len(gated_all):,}", f"{len(scores):,}",
                100 * len(gated_all) / max(len(scores), 1))

    lcfg = LevelConfig(lookback_sessions=args.lookback_sessions)
    if args.stop_ratio <= 0 or args.max_stop_r <= 0:
        logger.error("--stop-ratio and --max-stop-r must be positive")
        return 1

    feature_columns = (_context_columns(feature_source, args.context)
                       if feature_source is not None else [])
    # An objective run joins those columns when the model is fit. Writing them
    # into this pass's rows would copy one feature subset into the population.
    # --out keeps the historical matrix, which carries the columns.
    if args.objective:
        ctx_cols = []
        move_dir = None
    else:
        ctx_cols = feature_columns
        move_dir = feature_source

    contract_id = None
    compact_to = None
    run_ts = None
    if args.canonical_write and (args.objective or args.out):
        logger.error("pass only one of --canonical-write, --objective, --out")
        return 1
    if args.canonical_write:
        from src.training.label_store import (
            outcome_set_dir, target_outcome_contract, write_build_sidecar,
            write_outcome_contract,
        )
        contract = target_outcome_contract(
            horizon_seconds=args.horizon, max_targets=args.max_targets,
            stop_ratio=args.stop_ratio, max_stop_r=args.max_stop_r,
            resting_entry=args.resting_entry, max_fill_wait=args.max_fill_wait,
            lookback_sessions=args.lookback_sessions,
            r_lookback_sessions=args.r_lookback_sessions)
        resolution = f"{policy.DECISION_GRID_SECONDS}s"
        out_dir = outcome_set_dir("candidate_level_reachability", resolution, contract)
        contract_id = write_outcome_contract(out_dir, contract)
        # Upstream lineage: the move-score PREDICTIONS this target build gates
        # on are a separate semantic type from an outcome dataset (§19) -- a
        # model's claim about the future, not ground truth -- so they are
        # recorded as their own upstream entry, never folded into the move
        # dataset's own DatasetRef or hidden inside this build's own build_id.
        upstream = [{
            "logical_id": "move_scores", "semantic_type": "predictions",
            "location": str(args.scores),
        }]
        if move_dir is not None:
            upstream.append({
                "logical_id": "labels_directional_first_passage",
                "semantic_type": "outcomes", "location": str(move_dir),
            })
        write_build_sidecar(out_dir, population_config={
            "tau_q": args.tau_q, "context": args.context}, upstream=upstream)
        logger.info("canonical-write: contract_id=%s -> %s", contract_id, out_dir)
    elif args.objective and args.out:
        logger.error("pass only one of --objective and --out")
        return 1
    elif args.objective:
        from src.experiment.model_identity import (
            allocate_model_id, validate_model_id,
        )
        from src.common.run_layout import new_run_ts, pass_dir, population_cache_path
        from src.training.cascade_run_spec import target_feature_spec, xgb_hyperparameters
        run_ts = args.run_ts or new_run_ts()
        hyper = xgb_hyperparameters(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, seed=args.seed)
        feature_spec = target_feature_spec(context=args.context)
        model_id = validate_model_id(
            args.model_id or allocate_model_id(
                args.objective, run_ts, args.pass_n, args.model_type,
                hyper, feature_spec=feature_spec, target="y_reach"),
            args.model_type)
        out_dir = pass_dir(args.objective, run_ts, args.pass_n) / "_sessions"
        compact_to = population_cache_path(args.objective, run_ts, args.pass_n)
    elif args.out:
        out_dir = Path(args.out)
        compact_to = None
        run_ts = None
    else:
        logger.error("pass --objective (run layout) or --out (explicit directory)")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tape_days = {_session_date(f) for f in list_tape_files()}
    days = [d for d in sorted(gated_all["sess_date"].unique()) if d in tape_days]
    if not days:
        logger.error("no gated sessions have a tape")
        return 1

    opts = {
        "stop_ratio": float(args.stop_ratio), "max_stop_r": float(args.max_stop_r),
        "max_targets": args.max_targets, "horizon": args.horizon,
        "r_lookback": args.r_lookback_sessions,
        "lookback_sessions": args.lookback_sessions,
        "resting_entry": bool(args.resting_entry),
        "max_fill_wait": args.max_fill_wait,
    }
    workers = args.workers or get_vm_config().workers_for(
        mem_per_worker_gb=MEM_PER_WORKER_GB)
    chunks = [list(c) for c in np.array_split(np.array([d.isoformat() for d in days],
                                                       dtype=object),
                                              min(workers, len(days)))]
    chunks = [c for c in chunks if c]
    logger.info("target matrix: %d sessions | %d ctx cols (%s) | resting_entry=%s | "
                "%s | %d workers -> %s", len(days), len(ctx_cols), args.context,
                args.resting_entry, policy.describe(), workers, out_dir)

    move_dir_str = str(move_dir) if move_dir is not None else ""
    n_rows = n_sess = 0
    if len(chunks) == 1:
        n_sess, n_rows = run_shard(chunks[0], gated_all, move_dir_str, str(out_dir),
                                   ctx_cols, lcfg, opts, canonical_write=args.canonical_write)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for c in chunks:
                sub = gated_all[gated_all["sess_date"].isin(
                    {date.fromisoformat(x) for x in c})]
                futs[pool.submit(run_shard, c, sub, move_dir_str, str(out_dir),
                                 ctx_cols, lcfg, opts, args.canonical_write)] = c
            for done, fut in enumerate(as_completed(futs), 1):
                s, r = fut.result()
                n_sess += s
                n_rows += r
                pct = int(20 * done / len(chunks))
                logger.info("[%s%s] %d/%d chunks | %d sessions | %s rows",
                            "█" * pct, "░" * (20 - pct), done, len(chunks),
                            n_sess, f"{n_rows:,}")

    logger.info("done: %d sessions, %s rows -> %s", n_sess, f"{n_rows:,}", out_dir)
    if compact_to is not None:
        from src.common.data_paths import data_root, labels_dir
        from src.experiment.model_identity import write_model_json
        from src.common.run_layout import (
            compact_parquets, merge_run_json, model_dir, previous_model,
        )
        from src.training.cascade_run_spec import CASCADE_PASS
        from src.objectives.loader import load_objective
        from src.training.label_store import (
            TARGET_LABEL_COLUMNS, outcome_contract_id, target_outcome_contract,
            write_outcome_contract,
        )
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        parts = sorted(p for p in out_dir.glob("*.parquet"))
        n = compact_parquets(parts, compact_to)
        for p in parts:
            p.unlink()
        out_dir.rmdir()
        contract = target_outcome_contract(
            horizon_seconds=args.horizon, max_targets=args.max_targets,
            stop_ratio=args.stop_ratio, max_stop_r=args.max_stop_r,
            resting_entry=args.resting_entry, max_fill_wait=args.max_fill_wait,
            lookback_sessions=args.lookback_sessions,
            r_lookback_sessions=args.r_lookback_sessions)
        label_dir = labels_dir() / outcome_contract_id(contract)
        write_outcome_contract(label_dir, contract)
        have = set(pq.read_schema(compact_to).names)
        keep = [c for c in TARGET_LABEL_COLUMNS if c in have]
        if keep:
            pq.write_table(
                ds.dataset(str(compact_to), format="parquet").to_table(columns=keep),
                label_dir / "labels.parquet", compression="zstd")
        rel_labels = str(label_dir.resolve().relative_to(data_root().resolve()))
        cid = outcome_contract_id(contract)
        rel_cache = f"pass_{args.pass_n}/training_matrix.parquet"
        prev_n, prev_id = previous_model(args.objective, run_ts, args.pass_n)
        rel_oof = f"pass_{prev_n}/model_{prev_id}/oof_predictions.parquet"
        feature_location = None
        if feature_source is not None:
            try:
                feature_location = str(feature_source.resolve().relative_to(data_root().resolve()))
            except ValueError:
                feature_location = str(feature_source)
        inputs = {
            "labels": {
                "semantic_type": "outcomes",
                "kind": "labels",
                "contract_id": cid,
                "location": rel_labels,
            },
            "features": ([{
                "semantic_type": "features",
                "kind": "features",
                "location": feature_location,
                "columns": feature_columns,
                "availability": "timestamp",
                "time_column": "ts",
            }] if feature_location else []),
            "predictions": [{
                "semantic_type": "predictions",
                "kind": "oof",
                "location": rel_oof,
                "pass": prev_n,
                "model_id": prev_id,
                "availability": "timestamp",
                "time_column": "ts",
            }],
            "join_keys": ["ts", "side"],
            "filters": [{"op": "gte", "left": "p_move", "right": f"tau_{args.tau_q}"}],
            "population_cache": rel_cache,
            "target": "y_reach",
        }
        write_model_json(
            model_dir(args.objective, run_ts, args.pass_n, model_id),
            model_id=model_id, model_type=args.model_type,
            hyperparameters=hyper, label_contract_id=cid,
            feature_spec=feature_spec, feature_columns=feature_columns,
            inputs=inputs, target="y_reach")
        described = CASCADE_PASS.get(args.pass_n, {})
        try:
            snapshot = dict(load_objective(args.objective))
        except Exception:
            snapshot = {}
        merge_run_json(args.objective, run_ts, {
            "pass": args.pass_n,
            "role": described.get("role", f"pass_{args.pass_n}"),
            "purpose": described.get("purpose", ""),
            "label_contract_id": cid,
            "population_cache": rel_cache,
            "inputs": inputs,
            "upstream": [
                {
                    "kind": "oof",
                    "semantic_type": "predictions",
                    "pass": prev_n,
                    "model_id": prev_id,
                    "location": rel_oof,
                },
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
    if n_rows and compact_to is None:
        allr = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))],
                         ignore_index=True)
        print("\n=== Reach rate by target rank (nearest = 0) ===")
        print(allr.groupby("tgt_rank").agg(
            n=("y_reach", "size"), reach=("y_reach", "mean"),
            median_dist_r=("tgt_dist_r", "median")).round(4).to_string())
        print("\n=== Reach rate by whether price sat inside a bound ===")
        print(allr.groupby("inside_bound").agg(
            n=("y_reach", "size"), reach=("y_reach", "mean")).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
