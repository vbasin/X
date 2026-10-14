"""Binary up/down head on TBM-directional bars only (add-on to causal timing).

The 3-class tree is a long-only timer: 87% flats dominate softmax, fires are
~57/1 up/down. This module trains **P(TBM up | directional)** with flats
dropped, walk-forward, then combines with the open + 10-minute activity
filter (ADR-047). Live pin is not changed.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.data_paths import search_dir
from src.evaluation._kpi import label_implied_pnl
from src.search.event_meta import (
    DEFAULT_CACHE,
    date_mask,
    load_cache,
    score_trades,
    session_mask,
)
from src.search.uniqueness import apply_trade_cooldown, keep_mask
from src.validation.walk_forward import (
    DEFAULT_START,
    DEFAULT_STOP,
    build_expanding_schedule,
)

logger = logging.getLogger("vtech.search.side_head")

SIDE_FEATURES = (
    "cs_5m_clv", "cs_10m_clv", "cs_15m_clv", "cs_30m_clv", "cs_clv",
    "cs_body_direction", "cs_body_range_ratio",
    "vw_vwap_dist", "vw_vwap_slope_24", "vw_vwap_slope_120",
    "vp_va_high_dist", "vp_va_low_dist", "vp_poc_dist", "vp_in_value_area",
    "of_delta", "of_buy_ratio", "of_delta_zscore_12",
    "bp_imbalance", "tl_aggressive_pct",
    "ca_es_ret_lag1", "ca_es_ret_lag2", "ca_es_mom_60",
    "tc_minutes_into_session", "ht_daily_trend_10",
)


def all_feature_names(cols: list[str]) -> list[str]:
    return [c for c in cols if not str(c).startswith("_")]


# Named “side” hypotheses (vol, VWAP slope, stochastic, daily trend).
HYPOTHESIS_FEATURES = (
    "rv_realized_vol_60", "rv_vol_ratio",
    "vw_vwap_slope_24", "vw_vwap_slope_120", "vw_vwap_dist",
    "so_sync_score_9_60", "so_fast_d_9_3", "so_full_d_60_10",
    "ht_daily_trend_10", "ht_daily_trend_20",
    "cs_clv_ma_10", "cs_body_direction",
)
MARGIN_GRID = (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
ACTIVITY_TAU = 0.65


def _xgb(scale_pos: float):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
        scale_pos_weight=float(max(scale_pos, 0.1)),
    )


def _feat_mat(
    X: np.ndarray, cols: list[str], names: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    wanted = list(names) if names is not None else list(SIDE_FEATURES)
    keep = [n for n in wanted if n in cols]
    idx = [cols.index(n) for n in keep]
    return np.nan_to_num(np.asarray(X[:, idx], dtype=np.float32), nan=0.0), keep


def univariate_side_auc(
    X: np.ndarray, cols: list[str], y: np.ndarray, mask: np.ndarray,
) -> list[dict[str, Any]]:
    """P(TBM up | directional) AUC per column on *mask* (already directional)."""
    yb = _binary_y(y)
    m = np.asarray(mask) & _dir_mask(y)
    out = []
    if m.sum() < 30 or yb[m].min() == yb[m].max():
        return out
    yt = yb[m]
    for i, name in enumerate(cols):
        if str(name).startswith("_"):
            continue
        x = np.nan_to_num(np.asarray(X[:, i], dtype=np.float64)[m], nan=0.0)
        if np.nanstd(x) < 1e-15:
            continue
        try:
            auc = float(roc_auc_score(yt, x))
        except ValueError:
            continue
        out.append({"feature": name, "auc": auc, "auc_fade": 1.0 - auc, "n": int(m.sum())})
    out.sort(key=lambda r: max(r["auc"], r["auc_fade"]), reverse=True)
    return out


def _close_frame_for_ts(ts: np.ndarray) -> pd.DataFrame:
    """UTC ``ts`` / ``_close`` covering *ts* from the 5s monthly parts."""
    from src.features.stores import load_feature_frame

    idx = pd.to_datetime(ts, utc=True)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    start = pd.Timestamp(idx.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(idx.max()).strftime("%Y-%m-%d")
    try:
        df = load_feature_frame("5s", start, end)
    except FileNotFoundError:
        return pd.DataFrame(columns=["ts", "_close"])
    if df is None or df.empty or "_close" not in df.columns:
        return pd.DataFrame(columns=["ts", "_close"])
    di = df.index
    if not isinstance(di, pd.DatetimeIndex):
        return pd.DataFrame(columns=["ts", "_close"])
    if di.tz is None:
        di = di.tz_localize("UTC")
    else:
        di = di.tz_convert("UTC")
    return pd.DataFrame({"ts": di, "_close": df["_close"].to_numpy()})


def aligned_close(ts: np.ndarray) -> np.ndarray:
    """``_close`` from 5s monthly parts, aligned to *ts* (NaN if missing)."""
    idx = pd.to_datetime(ts, utc=True)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    px = np.full(len(ts), np.nan, dtype=np.float64)
    closes = _close_frame_for_ts(ts)
    if closes.empty:
        return px
    closes["ts"] = pd.to_datetime(closes["ts"], utc=True)
    left = pd.DataFrame({"ts": pd.DatetimeIndex(idx)})
    j = left.merge(closes.drop_duplicates("ts"), on="ts", how="left")
    return j["_close"].to_numpy(dtype=np.float64)


def causal_ema_cross_1m(ts: np.ndarray, px: np.ndarray, *, fast: int = 9, slow: int = 21) -> np.ndarray:
    """Sign(EMA_fast − EMA_slow) on 1-minute closes, stale-forward to 5s. Per Chicago day."""
    from src.search.event_meta import chicago_index
    ct = chicago_index(ts)
    days = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in ct])
    out = np.zeros(len(ts), dtype=np.float32)
    s = pd.Series(px, index=pd.to_datetime(ts, utc=True))
    for d in sorted(set(days.tolist())):
        m = days == d
        sl = s.iloc[np.flatnonzero(m)]
        if sl.notna().sum() < slow + 2:
            continue
        one = sl.resample("1min").last().dropna()
        if len(one) < slow + 2:
            continue
        ema_f = one.ewm(span=fast, adjust=False).mean()
        ema_s = one.ewm(span=slow, adjust=False).mean()
        cross = np.sign(ema_f - ema_s).replace(0, np.nan).ffill().fillna(0)
        # 1-min bar stamped T is visible at T+1m (left-labeled minute).
        cross.index = cross.index + pd.Timedelta(minutes=1)
        aligned = cross.reindex(sl.index, method="ffill")
        out[m] = np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0)
    return np.nan_to_num(out, nan=0.0)


def decile_side_table(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, name: str, n_bins: int = 5,
) -> dict[str, Any]:
    """P(TBM up|dir) and P(down|dir) by quantile of *x* on *mask*."""
    m = np.asarray(mask) & _dir_mask(y) & np.isfinite(x)
    rows = []
    if m.sum() < 50:
        return {"feature": name, "bins": rows, "auc": None}
    xv = np.asarray(x, dtype=np.float64)[m]
    yb = _binary_y(y[m])  # 1 = up
    try:
        auc = float(roc_auc_score(yb, xv))
    except ValueError:
        auc = None
    try:
        q = pd.qcut(xv, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return {"feature": name, "bins": rows, "auc": auc}
    for i in range(int(q.max()) + 1):
        b = q == i
        n = int(b.sum())
        n_up = int(yb[b].sum())
        n_dn = n - n_up
        rows.append({
            "bin": i, "n": n, "n_up": n_up, "n_down": n_dn,
            "p_up": n_up / max(n, 1), "p_down": n_dn / max(n, 1),
            "x_lo": float(np.min(xv[b])), "x_hi": float(np.max(xv[b])),
        })
    return {"feature": name, "auc": auc, "bins": rows}


def alt_tbm_y(
    ts: np.ndarray,
    *,
    move: float,
    mae: float,
    window: int = 120,
) -> np.ndarray:
    """Per-session-day TBM labels from 5s ``_close``, aligned to *ts*."""
    from src.search.event_meta import chicago_index
    from src.training.labels import _classify_triple_barrier

    idx = pd.to_datetime(ts, utc=True)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    ct = chicago_index(ts)
    days = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in ct])
    unique = sorted(set(days.tolist()))
    px = np.full(len(ts), np.nan, dtype=np.float64)
    closes = _close_frame_for_ts(ts)
    if not closes.empty:
        closes["ts"] = pd.to_datetime(closes["ts"], utc=True)
        left = pd.DataFrame({"ts": pd.DatetimeIndex(idx).tz_convert("UTC")})
        j = left.merge(closes.drop_duplicates("ts"), on="ts", how="left")
        px = j["_close"].to_numpy(dtype=np.float64)
    y = np.ones(len(ts), dtype=np.int32)
    for d in unique:
        m = days == d
        sl = px[m]
        finite = np.isfinite(sl)
        if finite.sum() < window + 2:
            continue
        s = pd.Series(sl).ffill().bfill().to_numpy(dtype=np.float64)
        y[m] = _classify_triple_barrier(s, int(window), float(move), float(mae))
    return y


def _dir_mask(y: np.ndarray) -> np.ndarray:
    return (y == 0) | (y == 2)


def _binary_y(y: np.ndarray) -> np.ndarray:
    """1 = TBM up, 0 = TBM down. Caller must have already dropped flats."""
    return (np.asarray(y) == 2).astype(np.int32)


def fit_side_fold(
    F: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    train: np.ndarray,
    oof: np.ndarray,
    *,
    uniq_bars: int = 120,
    session_only: bool = True,
) -> dict[str, Any]:
    """Fit one fold. *train*/*oof* are boolean masks on the full cache axis."""
    sess = session_mask(ts) if session_only else np.ones(len(y), dtype=bool)
    tr = train & _dir_mask(y) & sess
    if uniq_bars:
        ix = np.flatnonzero(tr)
        if len(ix):
            keep = np.zeros(len(y), dtype=bool)
            keep[ix] = keep_mask(ts[ix], uniq_bars)
            tr = keep
    # purge 10m before OOF
    if oof.any():
        oof0 = pd.to_datetime(ts[oof][0], utc=True)
        tr = tr & np.asarray(pd.to_datetime(ts, utc=True) <= (oof0 - pd.Timedelta(minutes=10)))

    n_tr = int(tr.sum())
    yb = _binary_y(y[tr])
    if n_tr < 40 or yb.min() == yb.max():
        return {"status": "skipped", "n_train": n_tr, "p_oof": None, "auc": None}
    n_up, n_dn = int(yb.sum()), int((yb == 0).sum())
    model = _xgb(n_dn / max(n_up, 1))
    model.fit(F[tr], yb)
    p_all = model.predict_proba(F)[:, 1]
    auc = None
    oof_dir = oof & _dir_mask(y) & sess
    if oof_dir.sum() >= 20 and _binary_y(y[oof_dir]).min() != _binary_y(y[oof_dir]).max():
        try:
            auc = float(roc_auc_score(_binary_y(y[oof_dir]), p_all[oof_dir]))
        except ValueError:
            auc = None
    return {
        "status": "ok",
        "n_train": n_tr,
        "n_train_up": n_up,
        "n_train_down": n_dn,
        "auc_oof_dir": auc,
        "p_oof": p_all,
        "model": model,
    }


def choose_margin(
    p_side: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    when: np.ndarray,
) -> float:
    """Pick |p-0.5| margin on TRAIN when-filter to max NE (min 12 trades)."""
    best_m, best_ne = 0.08, -1e18
    for m in MARGIN_GRID:
        side = np.zeros(len(y), dtype=np.int8)
        take = when & (np.abs(p_side - 0.5) >= m)
        side[take & (p_side >= 0.5)] = 1
        side[take & (p_side < 0.5)] = -1
        if not take.any():
            continue
        kept = np.zeros(len(y), dtype=bool)
        ix = np.flatnonzero(side != 0)
        kept[ix] = apply_trade_cooldown(ts[ix], 120)
        sc = score_trades(np.where(kept, side, 0).astype(np.int8), y, ts)
        n, ne = int(sc["n_trades"]), sc["net_expectancy"]
        if ne is not None and n >= 12 and float(ne) > best_ne:
            best_ne, best_m = float(ne), float(m)
    return best_m


def _when_from_timing(timing: pd.DataFrame, ts: np.ndarray, tau: float) -> np.ndarray:
    """Align 3-class OOF confidence onto cache timestamps; open session implied later."""
    tdf = timing.copy()
    tdf["ts"] = pd.to_datetime(tdf["ts"], utc=True)
    tdf = tdf.drop_duplicates("ts", keep="last")
    cache_ts = pd.to_datetime(ts, utc=True)
    merged = pd.Series(np.nan, index=pd.RangeIndex(len(cache_ts)))
    lookup = pd.Series(tdf["confidence"].to_numpy(), index=tdf["ts"])
    # reindex is slow if sets differ; use merge
    left = pd.DataFrame({"ts": cache_ts})
    j = left.merge(tdf[["ts", "confidence", "p_up", "p_down"]], on="ts", how="left")
    conf = j["confidence"].to_numpy(dtype=np.float64)
    pup = j["p_up"].to_numpy(dtype=np.float64) if "p_up" in j.columns else conf
    when = np.isfinite(conf) & (pup >= float(tau))
    return when, pup, j["p_down"].to_numpy(dtype=np.float64) if "p_down" in j.columns else np.zeros(len(ts))


def apply_side(
    p_side: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    when: np.ndarray,
    *,
    margin: float,
) -> dict[str, Any]:
    sess = session_mask(ts)
    take = sess & when & np.isfinite(p_side) & (np.abs(p_side - 0.5) >= float(margin))
    side = np.zeros(len(y), dtype=np.int8)
    side[take & (p_side >= 0.5)] = 1
    side[take & (p_side < 0.5)] = -1
    ix = np.flatnonzero(side != 0)
    kept = np.zeros(len(y), dtype=bool)
    if len(ix):
        kept[ix] = apply_trade_cooldown(ts[ix], 120)
    side_k = np.where(kept, side, 0).astype(np.int8)
    sc = score_trades(side_k, y, ts)
    fired = kept
    n_long = int((side_k == 1).sum())
    n_short = int((side_k == -1).sum())
    true = np.where(y == 2, "up", np.where(y == 0, "down", "flat"))
    pred = np.where(side_k == 1, "up", np.where(side_k == -1, "down", "flat"))
    d = pd.DataFrame({"direction": pred[fired], "direction_true": true[fired]})
    n_corr = int((d.direction == d.direction_true).sum()) if len(d) else 0
    n_flat = int((d.direction_true == "flat").sum()) if len(d) else 0
    n_wrong = int(((d.direction_true != "flat") & (d.direction != d.direction_true)).sum()) if len(d) else 0
    n_dir = n_corr + n_wrong
    sc.update({
        "n_long": n_long, "n_short": n_short,
        "n_correct": n_corr, "n_flat": n_flat, "n_wrong_side": n_wrong,
        "side_acc_dir": (n_corr / n_dir) if n_dir else None,
        "margin": float(margin),
    })
    return sc


def long_only_when(y: np.ndarray, ts: np.ndarray, when: np.ndarray) -> dict[str, Any]:
    sess = session_mask(ts)
    side = np.zeros(len(y), dtype=np.int8)
    side[sess & when] = 1
    ix = np.flatnonzero(side != 0)
    kept = np.zeros(len(y), dtype=bool)
    if len(ix):
        kept[ix] = apply_trade_cooldown(ts[ix], 120)
    return score_trades(np.where(kept, side, 0).astype(np.int8), y, ts)


def _auc(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float | None:
    m = np.asarray(mask) & _dir_mask(y) & np.isfinite(p)
    if m.sum() < 30:
        return None
    yb = _binary_y(y[m])
    if yb.min() == yb.max():
        return None
    try:
        return float(roc_auc_score(yb, p[m]))
    except ValueError:
        return None


def run_side_search(
    cache_dir: Path,
    timing_oof: Path | None,
    *,
    uniq_bars: int = 120,
    session_only: bool = True,
    tau: float = ACTIVITY_TAU,
    y_side: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    extra_features: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    bundle = load_cache(cache_dir)
    cols, X, y_econ, ts = bundle["cols"], bundle["X"], bundle["y"], bundle["ts"]
    y_fit = y_econ if y_side is None else np.asarray(y_side, dtype=np.int32)
    F, feat_names = _feat_mat(X, cols, feature_names)
    if extra_features:
        mats = []
        for k, v in extra_features.items():
            mats.append(np.nan_to_num(np.asarray(v, dtype=np.float32).reshape(-1, 1), nan=0.0))
            feat_names.append(k)
        F = np.hstack([F, *mats]) if mats else F
    timing = pd.read_parquet(timing_oof) if timing_oof else None
    when_all = None
    if timing is not None:
        when_all, pup, pdn = _when_from_timing(timing, ts, tau)
    else:
        when_all = np.zeros(len(y_econ), dtype=bool)
        pup = np.zeros(len(y_econ))

    schedule = build_expanding_schedule(start=DEFAULT_START, stop=DEFAULT_STOP)
    p_side = np.full(len(y_econ), np.nan, dtype=np.float64)
    folds = []
    for spec in schedule:
        tr = date_mask(ts, spec["train"]["start"], spec["train"]["end"])
        oof = date_mask(ts, spec["oof"]["start"], spec["oof"]["end"])
        rec = fit_side_fold(F, y_fit, ts, tr, oof, uniq_bars=uniq_bars, session_only=session_only)
        rec["auc_oof_econ"] = None
        logger.info("side_head fold %s status=%s n_train=%s auc_fit=%s",
                    spec["fold"], rec.get("status"), rec.get("n_train"), rec.get("auc_oof_dir"))
        if rec.get("status") == "ok" and rec.get("p_oof") is not None:
            p_side[oof] = rec["p_oof"][oof]
            rec["auc_oof_econ"] = _auc(y_econ, rec["p_oof"], oof & session_mask(ts))
            # margin from this fold's train when-filter, scored on *economic* 50/6 labels
            tr_when = tr & when_all if when_all.any() else tr & session_mask(ts)
            rec["margin"] = choose_margin(rec["p_oof"], y_econ, ts, tr_when)
        rec.pop("model", None)
        rec.pop("p_oof", None)
        rec["fold"] = spec["fold"]
        rec["train"] = spec["train"]
        rec["oof"] = spec["oof"]
        folds.append(rec)

    oof_any = np.isfinite(p_side)
    y = y_econ
    # default margin: median of fold margins
    margins = [f["margin"] for f in folds if f.get("status") == "ok" and f.get("margin") is not None]
    default_m = float(np.median(margins)) if margins else 0.08

    # Per-fold apply stored margin on that OOF slice, then pool.
    pooled_side = np.zeros(len(y), dtype=np.int8)
    for spec, rec in zip(schedule, folds):
        if rec.get("status") != "ok":
            continue
        oof = date_mask(ts, spec["oof"]["start"], spec["oof"]["end"])
        m = float(rec.get("margin") or default_m)
        take = oof & session_mask(ts) & when_all & np.isfinite(p_side) & (np.abs(p_side - 0.5) >= m)
        pooled_side[take & (p_side >= 0.5)] = 1
        pooled_side[take & (p_side < 0.5)] = -1
    ix = np.flatnonzero(pooled_side != 0)
    kept = np.zeros(len(y), dtype=bool)
    if len(ix):
        kept[ix] = apply_trade_cooldown(ts[ix], 120)
    combined = score_trades(np.where(kept, pooled_side, 0).astype(np.int8), y, ts)
    combined.update({
        "n_long": int((np.where(kept, pooled_side, 0) == 1).sum()),
        "n_short": int((np.where(kept, pooled_side, 0) == -1).sum()),
    })
    # breakdown against *economic* 50/6 labels
    pred = np.where(np.where(kept, pooled_side, 0) == 1, "up",
                    np.where(np.where(kept, pooled_side, 0) == -1, "down", "flat"))
    true = np.where(y == 2, "up", np.where(y == 0, "down", "flat"))
    fired = kept
    n_corr = int(((pred == true) & fired).sum())
    n_flat = int(((true == "flat") & fired).sum())
    n_wrong = int(((true != "flat") & (pred != true) & fired).sum())
    n_dir = n_corr + n_wrong
    combined["n_correct"] = n_corr
    combined["n_flat"] = n_flat
    combined["n_wrong_side"] = n_wrong
    combined["side_acc_dir"] = (n_corr / n_dir) if n_dir else None

    lo = long_only_when(y, ts, when_all)
    no_timing = apply_side(p_side, y, ts, np.ones(len(y), dtype=bool), margin=default_m)

    sess = session_mask(ts)
    auc_fit = _auc(y_fit, p_side, oof_any & sess)
    auc_econ = _auc(y_econ, p_side, oof_any & sess)

    frame = pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True),
        "p_side": p_side,
        "direction": pred,
        "direction_true": true,
        "when": when_all,
    })
    frame = frame.loc[oof_any].copy()

    return {
        "cache": str(cache_dir),
        "timing_oof": str(timing_oof) if timing_oof else None,
        "feat_names": feat_names,
        "n_features": len(feat_names),
        "folds": folds,
        "pooled_auc_dir_open": auc_econ,
        "pooled_auc_fit_label": auc_fit,
        "default_margin": default_m,
        "long_only_timing": lo,
        "combined_timing_plus_side": combined,
        "side_only_open": no_timing,
        "oof": frame,
        "n_oof_scored": int(oof_any.sum()),
        "fit_label_mix": {str(k): int(v) for k, v in zip(*np.unique(y_fit, return_counts=True))},
    }


def persist(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = result.get("oof")
    if isinstance(oof, pd.DataFrame) and len(oof):
        oof.to_parquet(out_dir / "oof_side_head.parquet", index=False)
    slim = {k: v for k, v in result.items() if k != "oof"}
    (out_dir / "side_head_result.json").write_text(json.dumps(slim, indent=2, default=str))
    return out_dir


def _cli() -> None:
    p = argparse.ArgumentParser(description="Binary TBM side head + timing combine")
    p.add_argument("--cache-key", default=DEFAULT_CACHE)
    p.add_argument("--timing-oof", default=str(
        search_dir() / "nq-50pt-xgb-unique" / "20260821T110058Z" / "oof_xgb_causal_topgroups.parquet"))
    p.add_argument("--tau", type=float, default=ACTIVITY_TAU)
    p.add_argument("--move", type=float, default=0.0,
                   help="If set with --mae, train side on this TBM (e.g. 10) not 50/6")
    p.add_argument("--mae", type=float, default=0.0)
    p.add_argument("--all-features", action="store_true")
    p.add_argument("--hypotheses", action="store_true",
                   help="Vol, VWAP slope, stochastic, daily trend + causal 9/21 EMA cross only")
    p.add_argument("--univariate", action="store_true")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cache = search_dir() / "_fold_cache" / args.cache_key
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) if args.out_dir else search_dir() / "nq-50pt-xgb-unique" / run_ts
    bundle = load_cache(cache)
    feat_names = all_feature_names(bundle["cols"]) if args.all_features else None
    extra = None
    deciles = []
    if args.hypotheses:
        feat_names = list(HYPOTHESIS_FEATURES)
        print("⏳ Hypothesis side features + causal 1m 9/21 EMA cross…")
        px = aligned_close(bundle["ts"])
        ema = causal_ema_cross_1m(bundle["ts"], px)
        extra = {"ema_cross_9_21": ema}
        from src.search.event_meta import session_mask as _sess
        mask = _sess(bundle["ts"])
        hyp_map = {
            "rv_realized_vol_60": bundle["X"][:, bundle["cols"].index("rv_realized_vol_60")],
            "vw_vwap_slope_120": bundle["X"][:, bundle["cols"].index("vw_vwap_slope_120")],
            "vw_vwap_dist": bundle["X"][:, bundle["cols"].index("vw_vwap_dist")],
            "so_sync_score_9_60": bundle["X"][:, bundle["cols"].index("so_sync_score_9_60")]
            if "so_sync_score_9_60" in bundle["cols"] else np.zeros(len(bundle["y"])),
            "ema_cross_9_21": ema,
            "ht_daily_trend_10": bundle["X"][:, bundle["cols"].index("ht_daily_trend_10")]
            if "ht_daily_trend_10" in bundle["cols"] else np.zeros(len(bundle["y"])),
        }
        print("  quintiles P(up|dir) vs P(down|dir) on 50/6, open:")
        for name, vec in hyp_map.items():
            tab = decile_side_table(np.asarray(vec, dtype=np.float64), bundle["y"], mask, name=name)
            deciles.append(tab)
            print(f"  {name} auc={tab.get('auc')}")
            for b in tab.get("bins") or []:
                print(f"    q{b['bin']} n={b['n']:6d}  P(up)={b['p_up']:.3f}  P(down)={b['p_down']:.3f}  "
                      f"x=[{b['x_lo']:.4g},{b['x_hi']:.4g}]")
    y_side = None
    if args.move and args.mae:
        print(f"⏳ Building TBM {args.move}/{args.mae} labels from parquet _close…")
        y_side = alt_tbm_y(bundle["ts"], move=args.move, mae=args.mae)
        mix = {int(k): int(v) for k, v in zip(*np.unique(y_side, return_counts=True))}
        print(f"  alt label mix {mix} dir_rate={((y_side==0)|(y_side==2)).mean():.3f}")
    uni = None
    if args.univariate:
        print("⏳ Univariate side AUC (open + directional, 50/6)…")
        from src.search.event_meta import session_mask as _sess
        mask = _sess(bundle["ts"]) & _dir_mask(bundle["y"])
        uni = univariate_side_auc(bundle["X"], bundle["cols"], bundle["y"], mask)
        print("  top |AUC-0.5|:")
        for row in uni[:15]:
            print(f"    {row['feature']:28s} auc={row['auc']:.3f} fade={row['auc_fade']:.3f}")
    print("⏳ Side-head walk-forward…")
    result = run_side_search(
        cache, Path(args.timing_oof) if args.timing_oof else None, tau=args.tau,
        y_side=y_side, feature_names=feat_names, extra_features=extra,
    )
    if uni:
        result["univariate_open_dir_50_6"] = uni[:40]
    if deciles:
        result["hypothesis_deciles"] = deciles
    persist(result, out)
    print(f"✅ wrote {out}")
    print(f"  AUC fit-label (open dir OOF) {result.get('pooled_auc_fit_label')}")
    print(f"  AUC 50/6 transfer (open dir OOF) {result.get('pooled_auc_dir_open')}")
    print(f"  n_features {result.get('n_features')} fit_mix {result.get('fit_label_mix')}")
    for k in ("long_only_timing", "combined_timing_plus_side", "side_only_open"):
        sc = result[k]
        print(f"  {k}: n={sc.get('n_trades')} HR={sc.get('hit_rate')} NE={sc.get('net_expectancy')} "
              f"MDD={sc.get('max_drawdown')} L/S={sc.get('n_long')}/{sc.get('n_short')} "
              f"side_acc={sc.get('side_acc_dir')}")
    print(f"  folds {[f.get('status') for f in result['folds']]} "
          f"auc_fit {[f.get('auc_oof_dir') for f in result['folds']]} "
          f"auc_50/6 {[f.get('auc_oof_econ') for f in result['folds']]}")


if __name__ == "__main__":
    _cli()
