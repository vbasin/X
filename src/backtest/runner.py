"""Backtest runner — CLI + orchestration for user-defined strategy evaluation.

The third experiment entry path (besides the automated model search and the manual model
spec): load a :class:`~src.backtest.spec.StrategySpec`, replay it over the feature
parquets, and score it on the **same frozen evaluation contract** as ML model candidates —
plus full trade economics from the ledger. Every run also records outcome **episodes**
(success/failure scenario metadata) so the Research Director's pattern analysis sees
strategy behaviour exactly like model behaviour.

Artifacts (``data/research/search/<objective_id>/strategy_<sid>_<ts>/``):

* ``trades.parquet``            — the trade ledger
* ``backtest_report.json``      — trade KPIs + signal-quality score + config echo
* ``oof_<sid>.parquet``         — full-window canonical frame (ts/direction/confidence/
  direction_true) — comparable with model OOF frames
* ``diagnostics_<sid>.json``    — the standard diagnostics report (tau sweep, confusion)

Usage::

    python -m src.backtest.runner --spec config/strategy.example.yaml
    python -m src.backtest.runner --spec ... --start 2025-10-01 --end 2026-02-28
    python -m src.backtest.runner --spec ... --dry-run     # validate only
"""

from __future__ import annotations

from src.common.data_paths import search_dir

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, ensure_price_close, run_backtest
from src.backtest.spec import StrategySpec, load_strategy_spec

logger = logging.getLogger("vtech.backtest.runner")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_LABEL_TO_DIRECTION = {0: "down", 1: "flat", 2: "up"}


def run_strategy(
    spec_or_path: str | Path | dict[str, Any],
    *,
    objective_id: str | None = None,
    parquet_dir: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backtest a user-defined strategy and score it on the objective's contract.

    Returns the report dict (also persisted as ``backtest_report.json``).
    """
    spec = load_strategy_spec(spec_or_path, objective_id=objective_id)
    obj_id = spec.objective_id or objective_id
    objective: dict[str, Any] | None = None
    if obj_id:
        from src.objectives.loader import load_objective
        objective = load_objective(obj_id)

    # ── Predictions mode: resolve the model OOF source + default exits to barriers ──
    # Resolve (and thus validate) the source path always; only read the full parquet on
    # a real run — a --dry-run stays cheap.
    pred_frame: pd.DataFrame | None = None
    if spec.mode == "predictions":
        oof_path = _resolve_oof_source((spec.predictions or {}).get("source"), obj_id)
        _apply_prediction_exit_defaults(spec, objective)
        if not dry_run:
            print(f"📈 Predictions mode — replaying {oof_path.name} "
                  f"(τ={float((spec.predictions or {}).get('tau', 0.8))}, "
                  f"sides={sorted((spec.predictions or {}).get('sides') or ['up', 'down'])})")
            pred_frame = pd.read_parquet(oof_path)

    # CLI date-range overrides win over the spec's backtest block.
    if start:
        spec.backtest["start"] = start
    if end:
        spec.backtest["end"] = end

    from src.features.stores import default_parquet_dir

    pdir = Path(parquet_dir) if parquet_dir else default_parquet_dir(objective)
    avail = _available_dates(pdir)
    if not avail:
        raise FileNotFoundError(f"No feature parquets found in {pdir}")
    lo = str(spec.backtest.get("start") or avail[0])
    hi = str(spec.backtest.get("end") or avail[-1])

    if dry_run:
        report = _dry_run_report(spec, pdir, avail, lo, hi)
        _print_dry_run(report)
        return report

    print(f"⏳ Loading features {lo} → {hi} from {pdir} …")
    t0 = time.time()
    from src.training.data_pipeline import DataPipeline
    df = DataPipeline(data_dir=str(pdir)).load_features(lo, hi)
    df = ensure_price_close(df)
    print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} cols in {time.time()-t0:.0f}s")

    if spec.mode == "predictions" and pred_frame is not None:
        df = _attach_predictions(df, pred_frame)

    missing = sorted(spec.referenced_features() - set(df.columns))
    if missing:
        raise ValueError(
            f"strategy references feature columns absent from the parquets: {missing} — "
            "check docs/reference/FEATURE_CATALOG.md for valid names")

    print(f"⏳ Running backtest '{spec.strategy_id}' ({spec.mode} mode)…")
    t0 = time.time()
    result = run_backtest(df, spec, objective=objective)
    print(f"✅ Backtest done in {time.time()-t0:.1f}s — {result.kpis['n_trades']} trades")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (search_dir() / (obj_id or "no-objective")
               / f"strategy_{spec.strategy_id}_{run_ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Signal-quality scoring on the frozen contract (same yardstick as models) ──
    oof, score, diagnostics = None, {}, None
    if objective is not None:
        oof = (_predictions_canonical(df, result)
               if spec.mode == "predictions" and "_pred_true" in df.columns
               else _full_window_canonical(df, result, objective))
        if oof is not None:
            from src.evaluation._kpi import canonical_trade_kpi, compute_diagnostics
            from src.evaluation.kpi_criteria import meets_requirement, meets_target, primary_tau
            gt = pd.DataFrame({"direction_true": oof["direction_true"]})
            score = canonical_trade_kpi(oof, gt, objective, tau=primary_tau(objective))
            diagnostics = compute_diagnostics(oof, gt, objective)
            oof.to_parquet(out_dir / f"oof_{spec.strategy_id}.parquet", index=False)
            (out_dir / f"diagnostics_{spec.strategy_id}.json").write_text(
                json.dumps(diagnostics, indent=2, default=str))
            req_ok, _ = meets_requirement(objective, diagnostics, score)
            tgt_ok, _ = meets_target(objective, diagnostics, score)
            score["meets_requirement"] = bool(req_ok)
            score["meets_target"] = bool(tgt_ok)

    if not result.trades.empty:
        result.trades.to_parquet(out_dir / "trades.parquet", index=False)

    report: dict[str, Any] = {
        "strategy_id":  spec.strategy_id,
        "objective_id": obj_id,
        "mode":         spec.mode,
        "run_ts":       run_ts,
        "window":       {"start": lo, "end": hi, "n_bars": result.n_bars},
        "trade_kpis":   result.kpis,
        "signal_score": _jsonable(score),
        "out_dir":      str(out_dir),
        "spec":         spec.raw,
    }
    (out_dir / "backtest_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ── Leaderboard row (best-effort; same contract as model candidates) ──────
    if objective is not None and score:
        try:
            from src.evaluation import EvalContract, record_result
            contract = EvalContract.load_latest(obj_id)
            if contract is not None:
                record_result(
                    objective_id=obj_id,
                    contract_version=contract.contract_version,
                    candidate_id=f"strategy_{spec.strategy_id}",
                    model_family=("model_backtest" if spec.mode == "predictions"
                                  else "user_strategy"),
                    score_result={
                        "primary_precision": score.get("precision_at_tau"),
                        "primary_dir_f1":    score.get("directional_f1"),
                        "primary_tau":       0.8,
                        "per_tau":           [_jsonable(score)],
                    },
                    spec_params={"strategy_id": spec.strategy_id, "mode": spec.mode},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("leaderboard_write_failed: %s", exc)

    # ── Outcome episodes + pattern report (feeds the Research Director) ───────
    if objective is not None and oof is not None:
        try:
            from src.evaluation.episodes import build_episodes, record_episodes
            from src.evaluation.patterns import analyze_and_write

            episodes = build_episodes(
                oof, features=df, objective=objective,
                trades=result.trades if not result.trades.empty else None,
            )
            if episodes is not None and not episodes.empty:
                record_episodes(obj_id, f"strategy_{spec.strategy_id}", run_ts, episodes,
                                kind="strategy")
                analyze_and_write(obj_id, f"strategy_{spec.strategy_id}", run_ts, episodes,
                                  meta={"kind": "strategy", "trade_kpis": result.kpis})
                print(f"📊 Episodes + outcome patterns recorded "
                      f"({len(episodes)} episodes)")
        except Exception as exc:  # noqa: BLE001 — analysis must never fail the backtest
            logger.warning("episode_record_failed: %s", exc)

    _print_summary(report, result)
    return report


# ── helpers ───────────────────────────────────────────────────────────────────

def _available_dates(pdir: Path) -> list[str]:
    from src.common.cme_holidays import is_trading_day
    from src.common.data_paths import list_feature_parquet_files

    files = list_feature_parquet_files(pdir)
    dates: list[str] = []
    seen: set[str] = set()
    for f in files:
        stem = f.stem.replace("features_", "")
        try:
            day = pd.Timestamp(stem).date()
        except Exception:
            day = None
        if day is not None:
            if is_trading_day(day):
                dates.append(day.isoformat())
            continue
        try:
            ts = pd.DatetimeIndex(pd.to_datetime(
                pd.read_parquet(f, columns=["bar_end_ts"])["bar_end_ts"], utc=True,
            ))
        except Exception:
            continue
        for d in ts.normalize().unique():
            day = pd.Timestamp(d).date()
            if is_trading_day(day):
                seen.add(day.isoformat())
    return dates if dates else sorted(seen)


# ── Predictions mode (model-OOF replay) ──────────────────────────────────────

def _oof_files(run_dir: Path) -> list[Path]:
    """OOF parquets in one run.

    A run-level ``oof_*.parquet`` wins when one is present, so a migrated run
    that also keeps ``pass_<n>/model_<id>/oof_predictions.parquet`` still
    resolves the file callers already used. The model-folder file is the
    fallback for a canonical run that has no run-level OOF.
    """
    root = [p for p in sorted(run_dir.glob("oof_*.parquet")) if p.is_file()]
    if root:
        return root
    return [p for p in sorted(run_dir.glob("pass_*/model_*/oof_predictions.parquet"))
            if p.is_file()]


def _resolve_oof_source(source: str | None, obj_id: str | None) -> Path:
    """Resolve ``predictions.source`` to one OOF parquet.

    Accepts a direct ``.parquet`` path, a run directory containing exactly one
    OOF (``oof_*.parquet`` or ``pass_<n>/model_<id>/oof_predictions.parquet``),
    or a run id under the roots in ``run_roots()`` (``research/runs/`` first).
    Raises with the candidate list when a directory holds several OOF frames.
    """
    if not source:
        raise ValueError("predictions mode requires predictions.source")
    p = Path(source)
    if p.suffix == ".parquet" and p.exists():
        return p
    from src.common.run_layout import run_roots
    roots = run_roots()
    dir_candidates: list[Path] = []
    if p.is_dir():
        dir_candidates.append(p)
    # research/runs/ first, then the historical roots -- one resolver, so a
    # migrated run is found in its canonical home.
    for root in roots:
        if obj_id:
            dir_candidates.append(root / obj_id / source)
        dir_candidates.append(root / source)
    if not p.is_absolute():
        dir_candidates.append(_REPO_ROOT / source)
    for d in dir_candidates:
        if d.is_dir():
            oofs = _oof_files(d)
            if len(oofs) == 1:
                return oofs[0]
            if len(oofs) > 1:
                names = "\n  ".join(str(o.relative_to(d)) for o in oofs)
                raise ValueError(
                    f"'{d}' holds multiple OOF frames — set predictions.source to one:"
                    f"\n  {names}")
    if p.exists():
        return p
    tried = ", ".join(str(r) for r in roots) or "research/runs"
    raise FileNotFoundError(
        f"could not resolve predictions.source '{source}' — expected an OOF parquet, "
        f"a run dir containing one, or a run id under {tried}")


def _apply_prediction_exit_defaults(
    spec: StrategySpec, objective: dict[str, Any] | None,
) -> None:
    """Fill exit target/stop/max_holding from the objective barriers when the spec
    leaves them unset — so "trade this model at its own barriers" needs no exit block.
    An explicit ``exit`` block in the spec always wins (setdefault).

    Also fills ``backtest.point_value`` from the objective's primary instrument
    (NQ → $20/pt) when the spec still has the engine default of 1.0.
    """
    if not objective:
        return
    lbl = objective.get("labels") or {}
    hor = objective.get("horizon") or {}
    spec.exit = {"eod_flat": True, **(spec.exit or {})}
    spec.exit.setdefault("target_points", float(lbl.get("move_threshold", 50.0)))
    spec.exit.setdefault("stop_points", float(lbl.get("mae_tolerance", 5.0)))
    if hor.get("max_trade_horizon"):
        spec.exit.setdefault("max_holding", str(hor["max_trade_horizon"]))
    # Instrument $ / point for PnL display (and total_value KPIs).
    from src.common.instruments import point_value_from_objective, primary_symbol
    pv = point_value_from_objective(objective, default=1.0)
    if pv != 1.0 and float(spec.backtest.get("point_value", 1.0)) == 1.0:
        spec.backtest["point_value"] = pv
    sym = primary_symbol(objective)
    if sym:
        spec.backtest.setdefault("instrument", sym)


def _attach_predictions(df: pd.DataFrame, pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Join a model OOF frame (ts, direction, confidence[, direction_true]) onto the
    feature frame as ``_pred_direction`` / ``_pred_confidence`` (+ ``_pred_true`` when
    present), aligned on the bar timestamp.

    Robust to malformed OOF frames: NaT timestamps are dropped and duplicate timestamps
    collapsed (last wins) — some older / ensemble OOFs repeat or omit ``ts``. Alignment is
    positional via ``get_indexer`` (a non-unique *feature* index is tolerated too). Raises
    ``ValueError`` with a clear message when the frame has no usable timestamps at all."""
    pf = pred_frame.copy()
    if "ts" in pf.columns:
        pf = pf.set_index("ts")
    pf.index = pd.to_datetime(pf.index, utc=True)
    pf = pf[pf.index.notna()]                             # drop NaT (unusable) timestamps
    if pf.index.has_duplicates:
        pf = pf[~pf.index.duplicated(keep="last")]
    if pf.empty:
        raise ValueError(
            "prediction frame has no usable timestamps (the ts column is empty / all NaT) "
            "— this model's OOF cannot be backtested; pick a run whose predictions carry a "
            "ts column")
    tzinfo = getattr(df.index, "tz", None)
    if tzinfo is not None:
        pf.index = (pf.index.tz_localize("UTC") if pf.index.tz is None
                    else pf.index.tz_convert(tzinfo))
    elif pf.index.tz is not None:
        pf.index = pf.index.tz_localize(None)

    df = df.copy()
    n = len(df)
    pos = pf.index.get_indexer(df.index)                 # -1 where a bar has no prediction
    valid = pos >= 0
    pred_dir = np.full(n, "flat", dtype=object)
    pred_conf = np.zeros(n, dtype="float64")
    dir_arr = pf["direction"].to_numpy()
    conf_arr = pd.to_numeric(pf["confidence"], errors="coerce").to_numpy()
    pred_dir[valid] = dir_arr[pos[valid]]
    pred_conf[valid] = np.nan_to_num(conf_arr[pos[valid]], nan=0.0)
    df["_pred_direction"] = pred_dir.astype(str)
    df["_pred_confidence"] = pred_conf
    if "direction_true" in pf.columns:
        true_arr = pf["direction_true"].to_numpy()
        pred_true = np.full(n, None, dtype=object)
        pred_true[valid] = true_arr[pos[valid]]
        df["_pred_true"] = pred_true
    if not valid.any():
        logger.warning("no OOF rows aligned to the feature frame — check ts/tz overlap "
                       "between the prediction source and the parquet window")
    return df


def _predictions_canonical(df: pd.DataFrame, result: BacktestResult) -> pd.DataFrame:
    """Canonical frame for predictions mode: ``direction_true`` from the model's own
    OOF labels (already joined as ``_pred_true``), ``direction`` = the traded side on
    entry bars else flat, ``confidence`` = 1.0 (the confidence gate already fired in the
    engine — the realised ledger is deterministic). Same schema/scoring as model OOF."""
    oof = pd.DataFrame({
        "ts":             df.index,
        "direction":      "flat",
        "confidence":     1.0,
        "direction_true": df["_pred_true"].to_numpy(),
    }).reset_index(drop=True)
    if not result.trades.empty:
        side_map = {"long": "up", "short": "down"}
        pos = df.index.get_indexer(pd.DatetimeIndex(result.trades["entry_ts"]))
        col = oof.columns.get_loc("direction")
        for j, p in enumerate(pos):
            if p >= 0:
                oof.iat[p, col] = side_map[result.trades.iloc[j]["side"]]
    return oof.dropna(subset=["direction_true"]).reset_index(drop=True)


def _slug_from_source(source: str) -> str:
    """A schema-valid strategy_id derived from an OOF source (for CLI --predictions)."""
    stem = Path(source).stem
    if stem.startswith("oof_"):
        stem = stem[len("oof_"):]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "model"
    return f"pred_{slug}"


# ── In-memory backtest (web-console quicklook — no artifacts) ─────────────────

def run_predictions_inmemory(
    *,
    source: str,
    tau: float = 0.8,
    confidence_gate: bool = True,
    max_positions: int = 1,
    start: str | None = None,
    end: str | None = None,
    objective_id: str | None = None,
    sides: list[str] | None = None,
    parquet_dir: str | Path | None = None,
    max_sessions: int = 400,
) -> dict[str, Any]:
    """Run a model-prediction backtest and return a UI-ready payload WITHOUT writing any
    artifacts (no run dir, no leaderboard/episode side effects) — the interactive
    counterpart to :func:`run_strategy` for the web console.

    Returns ``{config, summary, equity_curve, confusion_matrix, trades, skipped}``.
    Raises ``ValueError`` if the window spans more than *max_sessions* trading days (the
    console loads every 5s bar in range into memory).
    """
    objective = None
    if objective_id:
        from src.objectives.loader import load_objective
        objective = load_objective(objective_id)

    oof_path = _resolve_oof_source(source, objective_id)

    pred_block: dict[str, Any] = {
        "source": str(oof_path),
        "tau": float(tau),
        "confidence_gate": bool(confidence_gate),
    }
    if sides:
        pred_block["sides"] = list(sides)
    spec = load_strategy_spec({
        "schema_version": "v1",
        "strategy_id": _slug_from_source(str(oof_path)),
        **({"objective_id": objective_id} if objective_id else {}),
        "predictions": pred_block,
        "backtest": {
            "max_concurrent_positions": max(1, int(max_positions)),
            **({"start": start} if start else {}),
            **({"end": end} if end else {}),
        },
    }, objective_id=objective_id)
    _apply_prediction_exit_defaults(spec, objective)

    from src.features.stores import (
        default_parquet_dir, load_feature_frame, parquet_dir_for_store,
    )
    from src.backtest.weights_score import infer_source_feature_store
    from src.training.data_pipeline import DataPipeline

    store = infer_source_feature_store(oof_path)
    if parquet_dir is not None:
        pdir = Path(parquet_dir)
    elif store:
        pdir = parquet_dir_for_store(store)
    else:
        pdir = default_parquet_dir(objective)
    avail = _available_dates(pdir)
    if not avail:
        raise FileNotFoundError(f"No feature parquets found in {pdir}")
    lo = str(spec.backtest.get("start") or avail[0])
    hi = str(spec.backtest.get("end") or avail[-1])
    n_sessions = len([d for d in avail if lo <= d <= hi])
    if n_sessions == 0:
        raise ValueError(f"no feature parquets in {lo} → {hi}")
    if n_sessions > max_sessions:
        raise ValueError(
            f"date range spans {n_sessions} sessions (> {max_sessions}) — narrow the "
            f"window; the console loads every 5s feature bar in range into memory")

    if store:
        df = load_feature_frame(store, lo, hi)
        logger.info("oof_replay store=%s window=%s→%s rows=%s",
                    store, lo, hi, 0 if df is None else len(df))
    else:
        df = DataPipeline(data_dir=str(pdir)).load_features(lo, hi)
    if df is None or df.empty:
        raise ValueError(f"feature load returned 0 rows for {lo}→{hi} store={store}")
    df = ensure_price_close(df)
    pred_frame = pd.read_parquet(oof_path)
    pred_frame = _recanonicalize_dual_oof(
        pred_frame, objective, confidence_gate=bool(confidence_gate),
        tau=float(tau))
    df = _attach_predictions(df, pred_frame)
    result = run_backtest(df, spec, objective=objective)
    payload = _backtest_payload(result, spec, df, lo, hi, n_sessions)
    cfg = payload.setdefault("config", {})
    cfg["feature_store"] = store
    cfg["asof_oof"] = True
    n_aligned = 0
    if "_pred_direction" in df.columns:
        n_aligned = int(df["_pred_direction"].astype(str).isin(["up", "down"]).sum())
    cfg["n_pred_raw"] = n_aligned
    cfg["weights_note"] = (
        f"Walk-forward OOF replay on the {store or 'default'} hive (as-of fold "
        f"predictions, not last-fold weights). {n_aligned} directional OOF bars "
        f"aligned in {lo}→{hi}."
    )
    return payload


def _recanonicalize_dual_oof(
    pred_frame: pd.DataFrame,
    objective: dict[str, Any] | None,
    *,
    confidence_gate: bool,
    tau: float | None = None,
) -> pd.DataFrame:
    """Rebuild ``direction`` from ``p_long``/``p_short``.

    The OOF parquet bakes ``labels.fire`` (0.40 on lg-st). Session
    ``p_max`` on the Globex LSTM sat at 0.39, so that cut is empty in
    RTH and the UI only traded with the gate off. Event cut is the
    **UI τ** when the gate is on, else library fire 0.1 (ADR-104).
    """
    if pred_frame is None or pred_frame.empty:
        return pred_frame
    if "p_long" not in pred_frame.columns or "p_short" not in pred_frame.columns:
        return pred_frame
    from src.models.dual_binary import (
        DUAL_BINARY_FIRE, dual_binary_fire, dual_binary_to_canonical,
    )
    ts = pred_frame["ts"] if "ts" in pred_frame.columns else None
    if not confidence_gate:
        used = float(DUAL_BINARY_FIRE)
    elif tau is not None:
        used = float(tau)
    else:
        used = float(dual_binary_fire(objective))
    rec = dual_binary_to_canonical(
        pred_frame["p_long"], pred_frame["p_short"], ts=ts, fire=used)
    out = pred_frame.copy()
    out["direction"] = rec["direction"].to_numpy()
    out["confidence"] = rec["confidence"].to_numpy()
    out["p_max"] = np.fmax(
        pd.to_numeric(out["p_long"], errors="coerce"),
        pd.to_numeric(out["p_short"], errors="coerce"),
    )
    out.attrs["dual_binary_fire"] = used
    return out


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _backtest_payload(
    result: BacktestResult,
    spec: StrategySpec,
    df: pd.DataFrame,
    lo: str,
    hi: str,
    n_sessions: int,
    *,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the web-console payload (config + summary + equity curve + confusion
    matrix + trade blotter) from a :class:`BacktestResult`."""
    k = result.kpis
    ledger = result.trades
    ex = spec.exit or {}
    pcfg = spec.predictions or {}
    pv = float(spec.backtest.get("point_value", 1.0))
    instrument = spec.backtest.get("instrument")

    config = {
        "strategy_id": spec.strategy_id,
        "objective_id": spec.objective_id,
        "tau": float(pcfg.get("tau", 0.8)),
        "confidence_gate": bool(pcfg.get("confidence_gate", True)),
        "sides": sorted(pcfg.get("sides") or ["up", "down"]),
        "max_concurrent_positions": int(spec.backtest.get("max_concurrent_positions", 1)),
        "target_points": ex.get("target_points"),
        "stop_points": ex.get("stop_points"),
        "max_holding": ex.get("max_holding"),
        "point_value": pv,
        "instrument": instrument,
        "window": {"start": lo, "end": hi, "n_sessions": int(n_sessions),
                   "n_bars": int(result.n_bars)},
    }
    labels = ["down", "flat", "up"]

    if ledger is None or ledger.empty:
        empty_keys = ("gross_profit", "gross_loss", "net_pnl", "win_rate", "expectancy",
                      "profit_factor", "max_drawdown", "sharpe_daily", "avg_win", "avg_loss")
        return {"config": config,
                "summary": {"n_trades": 0, "n_wins": 0, "n_losses": 0, "point_value": pv,
                            "net_value": 0.0, "by_exit_reason": {},
                            **{key: None for key in empty_keys}},
                "equity_curve": [],
                "confusion_matrix": {"labels": labels,
                                     "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
                "trades": [], "skipped": result.skipped}

    pnl = ledger["pnl_points"].to_numpy(dtype="float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    cum = np.cumsum(pnl)
    entry_ts = pd.to_datetime(ledger["entry_ts"])
    exit_ts = pd.to_datetime(ledger["exit_ts"])

    summary = {
        "n_trades":         int(len(ledger)),
        "n_wins":           int((pnl > 0).sum()),
        "n_losses":         int((pnl <= 0).sum()),
        "gross_profit":     float(wins.sum()),
        "gross_loss":       float(losses.sum()),               # <= 0 (total loss)
        "net_pnl":          float(pnl.sum()),
        "net_value":        float(pnl.sum() * pv),
        "win_rate":         k.get("win_rate"),
        "expectancy":       k.get("expectancy_points"),
        "profit_factor":    k.get("profit_factor"),
        "max_drawdown":     k.get("max_drawdown_points"),
        "sharpe_daily":     k.get("sharpe_daily"),
        "sharpe_trade":     k.get("sharpe_trade"),
        "avg_win":          k.get("avg_win_points"),
        "avg_loss":         k.get("avg_loss_points"),
        "avg_holding_bars": k.get("avg_holding_bars"),
        "avg_mae":          k.get("avg_mae_points"),
        "avg_mfe":          k.get("avg_mfe_points"),
        "exposure_frac":    k.get("exposure_frac"),
        "n_sessions_traded": k.get("n_sessions_traded"),
        "by_exit_reason":   k.get("by_exit_reason", {}),
        "point_value":      pv,
    }

    # Equity curve: realised cumulative PnL at each exit, seeded at 0 on the first entry.
    equity_curve = [{"t": _iso(entry_ts.iloc[0]), "equity": 0.0}]
    equity_curve += [{"t": _iso(exit_ts.iloc[j]), "equity": float(cum[j])}
                     for j in range(len(ledger))]

    # Confusion matrix over ENTERED trades: rows = true move, cols = predicted side.
    # True labels come from:
    #   1) OOF ``direction_true`` joined as ``_pred_true`` (search archaeology), or
    #   2) objective barrier labels at each entry bar (Live-weights path — no OOF).
    lab_ix = {l: i for i, l in enumerate(labels)}
    cm = [[0] * 3 for _ in range(3)]
    pred_dir = ledger["side"].map({"long": "up", "short": "down"}).to_numpy()
    true_dir = _entry_true_directions(df, ledger, spec)
    if true_dir is not None:
        for t, p in zip(true_dir, pred_dir):
            if isinstance(t, str) and t in lab_ix and p in lab_ix:
                cm[lab_ix[t]][lab_ix[p]] += 1
    confusion = {"labels": labels, "rows_true_cols_pred": cm}

    entry_idx = ledger["entry_idx"].to_numpy(dtype="int64") \
        if "entry_idx" in ledger.columns else None
    n_total_feat: int | None = None
    n_present_by_entry: np.ndarray | None = None
    if feature_cols and entry_idx is not None:
        n_total_feat = len(feature_cols)
        feat = df.reindex(columns=feature_cols)
        n_present_all = (n_total_feat - feat.isna().sum(axis=1)).to_numpy(dtype="int64")
        n_present_by_entry = n_present_all[entry_idx]

    conf_by_entry: np.ndarray | None = None
    if entry_idx is not None and "_pred_confidence" in df.columns:
        conf_arr = pd.to_numeric(df["_pred_confidence"], errors="coerce").to_numpy(
            dtype="float64")
        conf_by_entry = conf_arr[entry_idx]

    trades_out = [{
        "entry_ts":     _iso(entry_ts.iloc[j]),
        "exit_ts":      _iso(exit_ts.iloc[j]),
        "side":         str(ledger.iloc[j]["side"]),
        "entry_px":     float(ledger.iloc[j]["entry_px"]),
        "exit_px":      float(ledger.iloc[j]["exit_px"]),
        "pnl_points":   float(pnl[j]),
        "cum_pnl":      float(cum[j]),
        "mae_points":   float(ledger.iloc[j]["mae_points"]),
        "mfe_points":   float(ledger.iloc[j]["mfe_points"]),
        "holding_bars": int(ledger.iloc[j]["holding_bars"]),
        "exit_reason":  str(ledger.iloc[j]["exit_reason"]),
        "n_present":    (int(n_present_by_entry[j])
                         if n_present_by_entry is not None else None),
        "n_total":      n_total_feat,
        "confidence":   (float(conf_by_entry[j])
                         if conf_by_entry is not None
                         and not np.isnan(conf_by_entry[j]) else None),
    } for j in range(len(ledger))]

    return {"config": config, "summary": summary, "equity_curve": equity_curve,
            "confusion_matrix": confusion, "trades": trades_out, "skipped": result.skipped}


def _entry_true_directions(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    spec: StrategySpec,
) -> np.ndarray | None:
    """True label (up/flat/down) at each trade's entry bar, aligned 1:1 with *ledger*.

    Prefer the model's OOF ``_pred_true`` when present. For Live-weights backtests the
    prediction frame has no ground-truth column, so recompute labels from the objective
    barrier scheme on the feature window (same contract as search KPI scoring).
    """
    if ledger is None or ledger.empty or "entry_idx" not in ledger.columns:
        return None
    idx = ledger["entry_idx"].to_numpy(dtype="int64")
    n = len(df)
    if (idx < 0).any() or (idx >= n).any():
        logger.warning("confusion_matrix: entry_idx out of range — skipping true labels")
        return None

    if "_pred_true" in df.columns:
        return df["_pred_true"].to_numpy()[idx]

    oid = spec.objective_id
    if not oid:
        return None
    try:
        from src.objectives.loader import load_objective
        objective = load_objective(oid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("confusion_matrix: cannot load objective %s: %s", oid, exc)
        return None
    labels = _objective_labels(df, objective)
    if labels is None:
        return None
    # _objective_labels returns a 1-d object array of 'up'/'flat'/'down'/nan
    arr = np.asarray(labels, dtype=object)
    if arr.shape[0] != n:
        logger.warning("confusion_matrix: label length %d != bars %d", arr.shape[0], n)
        return None
    return arr[idx]


def _full_window_canonical(
    df: pd.DataFrame,
    result: BacktestResult,
    objective: dict[str, Any],
) -> pd.DataFrame | None:
    """Canonical frame over EVERY bar: up/down on entry bars, flat elsewhere.

    Confidence is 1.0 everywhere (a rule strategy is deterministic: it either fires or
    it deliberately does not). This makes recall/missed-signal metrics real — of all true
    moves, how many did the strategy catch — and keeps precision@tau comparable with
    model candidates.
    """
    labels = _objective_labels(df, objective)
    if labels is None:
        return None
    oof = pd.DataFrame({
        "ts":         df.index,
        "direction":  "flat",
        "confidence": 1.0,
        "direction_true": labels,
    }).reset_index(drop=True)
    if not result.trades.empty:
        side_map = {"long": "up", "short": "down"}
        pos = df.index.get_indexer(pd.DatetimeIndex(result.trades["entry_ts"]))
        for j, p in enumerate(pos):
            if p >= 0:
                oof.iat[p, oof.columns.get_loc("direction")] = (
                    side_map[result.trades.iloc[j]["side"]])
    # Rows with NaN labels (forward-window overhang) are dropped from scoring.
    oof = oof.dropna(subset=["direction_true"]).reset_index(drop=True)
    return oof


def _objective_labels(df: pd.DataFrame, objective: dict[str, Any]) -> pd.Series | None:
    """Detection-2 labels for the loaded window, as 'up'/'flat'/'down' strings."""
    try:
        from src.training.config import LabelConfig
        from src.training.labels import generate_labels
        from src.backtest.engine import parse_duration_bars
    except ImportError:
        return None
    lbl = objective.get("labels") or {}
    hor = objective.get("horizon") or {}
    window = parse_duration_bars(hor.get("max_trade_horizon", "10m")) or 120
    cfg = LabelConfig(
        big_move_threshold=float(lbl.get("move_threshold", 50.0)),
        mae_tolerance=float(lbl.get("mae_tolerance", 5.0)),
        forward_windows=[window],
        scheme=str(lbl.get("scheme", "detection_2")),   # honor triple_barrier objectives
    )
    try:
        priced = ensure_price_close(df)
        labels_df = generate_labels(priced["_close"], cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("confusion_matrix: cannot label window: %s", exc)
        return None
    lab = labels_df["label"]
    # The final `window` bars of the data have no complete forward window — mark NaN.
    lab = lab.copy().astype("object")
    lab.iloc[-window:] = np.nan
    return lab.map(lambda v: _LABEL_TO_DIRECTION.get(int(v)) if pd.notna(v) else np.nan).values

def _dry_run_report(
    spec: StrategySpec, pdir: Path, avail: list[str], lo: str, hi: str,
) -> dict[str, Any]:
    """Validate the spec against real parquet columns without loading everything."""
    import pyarrow.parquet as pq
    sample = next(iter(sorted(pdir.glob("features_*.parquet"))), None)
    cols = set(pq.read_schema(sample).names) if sample else set()
    missing = sorted(spec.referenced_features() - cols) if cols else []
    n_days = len([d for d in avail if lo <= d <= hi])
    return {
        "strategy_id":      spec.strategy_id,
        "objective_id":     spec.objective_id,
        "mode":             spec.mode,
        "dry_run":          True,
        "window":           {"start": lo, "end": hi, "n_sessions": n_days},
        "referenced_features": sorted(spec.referenced_features()),
        "missing_features": missing,
        "valid":            not missing,
    }


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=str)) if d else {}


def _print_dry_run(r: dict[str, Any]) -> None:
    icon = "✅" if r["valid"] else "❌"
    print(f"\n{icon} DRY RUN — strategy '{r['strategy_id']}' ({r['mode']} mode)")
    print(f"   window: {r['window']['start']} → {r['window']['end']} "
          f"({r['window']['n_sessions']} sessions)")
    print(f"   features referenced: {len(r['referenced_features'])}")
    if r["missing_features"]:
        print(f"   ❌ MISSING from parquets: {r['missing_features']}")
    else:
        print("   all referenced features present in the parquets")


def _print_summary(report: dict[str, Any], result: BacktestResult) -> None:
    k = result.kpis
    s = report.get("signal_score") or {}
    print(f"\n{'─'*64}")
    print(f"  Strategy backtest  {report['strategy_id']}  [{report['run_ts']}]")
    print(f"  Window     : {report['window']['start']} → {report['window']['end']}"
          f"  ({k.get('n_sessions_traded', 0)} sessions traded)")
    print(f"  Trades     : {k['n_trades']}  win_rate={_fmt(k['win_rate'])}"
          f"  PF={_fmt(k['profit_factor'])}  expectancy={_fmt(k['expectancy_points'])}pt")
    print(f"  PnL        : {_fmt(k['total_points'])}pt  maxDD={_fmt(k['max_drawdown_points'])}pt"
          f"  sharpe(daily)={_fmt(k['sharpe_daily'])}")
    if k.get("by_exit_reason"):
        print(f"  Exits      : {k['by_exit_reason']}")
    if "in_sample" in k:
        i, o = k["in_sample"], k["out_of_sample"]
        print(f"  IS  ({i['n_trades']} trades): {_fmt(i['total_points'])}pt "
              f"wr={_fmt(i['win_rate'])}   |   OOS ({o['n_trades']} trades): "
              f"{_fmt(o['total_points'])}pt wr={_fmt(o['win_rate'])}")
    if s:
        req = "✅" if s.get("meets_requirement") else "✗"
        tgt = "🎯" if s.get("meets_target") else "✗"
        print(f"  Contract   : precision@τ={_fmt(s.get('precision_at_tau'))} "
              f"(n={s.get('n_signals')})  requirement {req}  target {tgt}")
    print(f"  Artifacts  : {report['out_dir']}")
    print(f"{'─'*64}\n")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.1f}"
    return str(v)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Backtest a rule/cadence strategy (--spec) OR a trained model's "
                    "predictions (--predictions) on the feature parquets")
    p.add_argument("--spec", default="", help="Path to a strategy spec YAML "
                   "(config/strategy_spec.schema.json)")
    p.add_argument("--predictions", default="", help="Backtest a model's OOF predictions "
                   "(oof_*.parquet path / run dir / run id) instead of a --spec")
    p.add_argument("--tau", type=float, default=None,
                   help="Confidence gate for --predictions mode (default 0.8)")
    p.add_argument("--strategy-id", default="",
                   help="strategy_id for --predictions mode (default: derived from source)")
    p.add_argument("--objective", default="", help="Objective id override/cross-check")
    p.add_argument("--parquet-dir", default="", help="Override parquet directory")
    p.add_argument("--start", default="", help="Override backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", default="", help="Override backtest end date (YYYY-MM-DD)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the spec + feature names only; no data load, no trades")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if bool(args.spec) == bool(args.predictions):
        p.error("provide exactly one of --spec or --predictions")

    spec_or_path: str | dict[str, Any]
    if args.predictions:
        pred_block: dict[str, Any] = {"source": args.predictions}
        if args.tau is not None:
            pred_block["tau"] = args.tau
        spec_or_path = {
            "schema_version": "v1",
            "strategy_id": args.strategy_id or _slug_from_source(args.predictions),
            "predictions": pred_block,
        }
        if args.objective:
            spec_or_path["objective_id"] = args.objective
    else:
        spec_or_path = args.spec

    run_strategy(
        spec_or_path,
        objective_id=args.objective or None,
        parquet_dir=args.parquet_dir or None,
        start=args.start or None,
        end=args.end or None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
