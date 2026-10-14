"""Outcome episode store — success/failure scenario metadata for pattern analysis.

Every evaluated candidate (ML model OOF, ensemble, or user-defined strategy backtest)
emits **episodes**: one row per *scenario* — a fired signal (win / loss) or a missed true
move — annotated with the **market context at that moment** (regime descriptors, time
structure) and, for strategies, the realised trade economics. This is the durable
reference to both successful and unsuccessful scenarios that lets the Research Director
and its two agents mine *patterns* explaining what works and what fails
(:mod:`src.evaluation.patterns`), instead of staring at aggregate metrics.

Outcome taxonomy (per episode):

* ``win``                  — fired, direction matched the true move
* ``wrong_direction``      — fired, a true move occurred the other way
* ``false_positive_flat``  — fired, no qualifying move happened (type-1)
* ``missed``               — a true move occurred but nothing fired (type-2)

Store layout (append-only)::

    data/evaluation/episodes/<objective_id>/
        <run_ts>__<source_id>.parquet          # the episodes
        <run_ts>__<source_id>__outcome_report.json   # written by patterns.analyze_and_write
        index.jsonl                            # one line per recorded artifact
"""

from __future__ import annotations

from src.common.data_paths import data_root, episodes_dir

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.evaluation.episodes")

# Curated context: regime descriptors + time structure — cheap, always meaningful.
DEFAULT_CONTEXT_PREFIXES = ("rv_", "ht_", "vw_", "dc_", "cs_", "tc_")

_MAX_MISSED_DEFAULT = 20_000     # bound the type-2 rows (deterministic head sample)


def _data_root() -> Path:
    return data_root()


def _episodes_dir(objective_id: str) -> Path:
    return episodes_dir() / objective_id


# ── Building ──────────────────────────────────────────────────────────────────

def build_episodes(
    oof: pd.DataFrame,
    *,
    features: pd.DataFrame | None = None,
    X: Any = None,
    ts: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    objective: dict[str, Any] | None = None,
    tau: float | None = None,
    context_cols: list[str] | None = None,
    trades: pd.DataFrame | None = None,
    max_missed: int = _MAX_MISSED_DEFAULT,
) -> pd.DataFrame | None:
    """Derive outcome episodes from a canonical OOF frame + market context.

    Parameters
    ----------
    oof:
        Canonical frame with ``ts / direction / confidence / direction_true``.
    features:
        Feature matrix indexed by ts (strategy path) — context is joined from it.
    X, ts, feature_cols:
        Alternative context source: fold-cache arrays (model path). ``X`` may be a
        memory-mapped array; only the selected context columns are materialised.
    objective:
        Supplies the firing threshold via ``kpi`` (primary tau) and the session
        timezone for derived time columns.
    tau:
        Explicit firing threshold override.
    context_cols:
        Explicit context column list; default = ``DEFAULT_CONTEXT_PREFIXES`` filter.
        Pass the model's own feature list to get per-feature win/loss separation for
        feature-refinement decisions.
    trades:
        Strategy trade ledger — merged by entry ts to attach pnl/exit_reason/MAE/MFE.
    """
    if oof is None or oof.empty or "direction_true" not in oof.columns:
        return None
    if tau is None:
        try:
            from src.evaluation.kpi_criteria import primary_tau
            tau = primary_tau(objective) if objective else None
        except Exception:
            tau = None

    f = oof.reset_index(drop=True).copy()
    directional = f["direction"].isin(["up", "down"])
    if tau is None:
        fired = directional
    else:
        fired = (f["confidence"] >= float(tau)) & directional
    true_move = f["direction_true"].isin(["up", "down"])

    outcome = pd.Series(pd.NA, index=f.index, dtype="object")
    outcome[fired & (f["direction"] == f["direction_true"])] = "win"
    outcome[fired & (f["direction_true"] == "flat")] = "false_positive_flat"
    outcome[fired & true_move & (f["direction"] != f["direction_true"])] = "wrong_direction"
    outcome[~fired & true_move] = "missed"

    keep = outcome.notna()
    ep = f.loc[keep, ["ts", "direction", "confidence", "direction_true"]].copy()
    ep["outcome"] = outcome[keep].values
    ep["fired"] = fired[keep].values

    # Bound the missed rows (they can dwarf the fired ones on long windows).
    n_missed_total = int((ep["outcome"] == "missed").sum())
    if n_missed_total > max_missed:
        missed_idx = ep.index[ep["outcome"] == "missed"]
        step = max(1, int(np.ceil(n_missed_total / max_missed)))
        keep_mask = (np.arange(n_missed_total) % step) == 0   # deterministic thinning
        ep = ep.drop(index=missed_idx[~keep_mask])
    ep.attrs["n_missed_total"] = n_missed_total

    ep["ts"] = pd.to_datetime(ep["ts"], utc=True)

    # ── Market-context join ───────────────────────────────────────────────────
    ctx = _context_frame(ep["ts"], features=features, X=X, ts_arr=ts,
                         feature_cols=feature_cols, context_cols=context_cols)
    if ctx is not None:
        ep = pd.concat([ep.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)

    # ── Derived time structure (exchange-local) ───────────────────────────────
    tz = ((objective or {}).get("session") or {}).get("timezone") or "America/Chicago"
    try:
        local = ep["ts"].dt.tz_convert(tz)
    except Exception:
        local = ep["ts"]
    ep["hour"] = local.dt.hour.astype("int16")
    ep["day_of_week"] = local.dt.dayofweek.astype("int16")

    # ── Strategy trade economics (by entry ts) ────────────────────────────────
    if trades is not None and not trades.empty and "entry_ts" in trades.columns:
        tcols = [c for c in ("pnl_points", "exit_reason", "holding_bars",
                             "mae_points", "mfe_points", "side") if c in trades.columns]
        tl = trades[["entry_ts", *tcols]].copy()
        tl["entry_ts"] = pd.to_datetime(tl["entry_ts"], utc=True)
        ep = ep.merge(tl.rename(columns={"entry_ts": "ts"}), on="ts", how="left")

    return ep.reset_index(drop=True)


def _context_frame(
    ep_ts: pd.Series,
    *,
    features: pd.DataFrame | None,
    X: Any,
    ts_arr: np.ndarray | None,
    feature_cols: list[str] | None,
    context_cols: list[str] | None,
) -> pd.DataFrame | None:
    """Extract context feature values at the episode timestamps."""
    if features is not None:
        cols = _select_context_cols(list(features.columns), context_cols)
        if not cols:
            return None
        fidx = pd.DatetimeIndex(pd.to_datetime(features.index, utc=True))
        pos = fidx.get_indexer(pd.DatetimeIndex(ep_ts))
        ok = pos >= 0
        out = pd.DataFrame(np.nan, index=range(len(ep_ts)), columns=cols, dtype="float64")
        if ok.any():
            vals = features.iloc[pos[ok]][cols].apply(pd.to_numeric, errors="coerce")
            out.iloc[np.flatnonzero(ok)] = vals.to_numpy(dtype="float64")
        return out

    if X is not None and ts_arr is not None and feature_cols:
        cols = _select_context_cols(list(feature_cols), context_cols)
        if not cols:
            return None
        col_idx = [feature_cols.index(c) for c in cols]
        base = pd.DatetimeIndex(pd.to_datetime(ts_arr, utc=True))
        pos = base.get_indexer(pd.DatetimeIndex(ep_ts))
        ok = pos >= 0
        out = pd.DataFrame(np.nan, index=range(len(ep_ts)), columns=cols, dtype="float64")
        if ok.any():
            rows = np.asarray(X)[pos[ok]][:, col_idx]       # memmap-friendly fancy index
            out.iloc[np.flatnonzero(ok)] = rows.astype("float64")
        return out
    return None


def _select_context_cols(available: list[str], context_cols: list[str] | None) -> list[str]:
    if context_cols:
        return [c for c in context_cols if c in available]
    return [c for c in available
            if any(c.startswith(p) for p in DEFAULT_CONTEXT_PREFIXES)]


# ── Persistence ───────────────────────────────────────────────────────────────

def record_episodes(
    objective_id: str,
    source_id: str,
    run_ts: str,
    episodes: pd.DataFrame,
    *,
    kind: str = "model",
) -> Path:
    """Persist an episodes frame + index entry. Returns the parquet path."""
    d = _episodes_dir(objective_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_ts}__{_safe(source_id)}.parquet"
    episodes.to_parquet(path, index=False)
    entry = {
        "run_ts": run_ts, "source_id": source_id, "kind": kind,
        "n_episodes": int(len(episodes)),
        "outcome_counts": episodes["outcome"].value_counts().to_dict(),
        "file": path.name,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (d / "index.jsonl").open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    logger.info("episodes_recorded objective=%s source=%s n=%d", objective_id,
                source_id, len(episodes))
    return path


def load_episodes(
    objective_id: str,
    *,
    source_id: str | None = None,
    run_ts: str | None = None,
) -> pd.DataFrame | None:
    """Load recorded episodes; latest run first when *run_ts* is not pinned."""
    d = _episodes_dir(objective_id)
    if not d.exists():
        return None
    pats = sorted(d.glob("*.parquet"), reverse=True)
    for p in pats:
        stem_run, _, stem_src = p.stem.partition("__")
        if run_ts and stem_run != run_ts:
            continue
        if source_id and stem_src != _safe(source_id):
            continue
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("episode_load_failed %s: %s", p, exc)
    return None


def latest_outcome_reports(
    objective_or_program_id: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Newest outcome-pattern reports for an objective (or a program that embeds one).

    The Research Director keys context by ``program_id`` (e.g. ``nq-50pt-08-15-cst``)
    while the episode store keys by ``objective_id`` (``nq-50pt``); when no exact
    directory exists, any objective directory whose id prefixes the program id is used.
    """
    root = _data_root() / "evaluation" / "episodes"
    d = root / objective_or_program_id
    if not d.exists() and root.exists():
        matches = [p for p in root.iterdir()
                   if p.is_dir() and objective_or_program_id.startswith(p.name)]
        d = max(matches, key=lambda p: p.name, default=d) if matches else d
    if not d.exists():
        return []
    reports: list[dict[str, Any]] = []
    for p in sorted(d.glob("*__outcome_report.json"), reverse=True)[:limit]:
        try:
            reports.append(json.loads(p.read_text()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("outcome_report_load_failed %s: %s", p, exc)
    return reports


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_.-") else "_" for c in s)
