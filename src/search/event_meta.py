"""Event-sampled primary rule + binary XGB veto (causal TBM 50/6).

5-second 3-class / OVR models learn activity, not side (ADR-046). This path
does not predict a side from scratch:

* Decision clock = closed 5-minute bar (when ``cs_5m_clv`` updates).
* Session     = 08:30–10:00 America/Chicago.
* Primary     = sign of causal ``cs_5m_clv`` if |CLV| ≥ threshold.
* Regime      = train-only realized-vol percentile (no test leak).
* Meta        = binary XGB P(TBM win | primary side) on those events.
* Inference   = 10-minute cooldown on approved trades.

Labels and payoff stay TBM +50 / −6 / 10m, cost 1. Live pin is untouched.
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
from src.evaluation._kpi import label_implied_pnl, max_drawdown
from src.search.uniqueness import apply_trade_cooldown, keep_mask
from src.validation.walk_forward import (
    DEFAULT_START,
    DEFAULT_STOP,
    build_expanding_schedule,
)

logger = logging.getLogger("vtech.search.event_meta")

WIN_PTS = 50.0
LOSS_PTS = 6.0
COST = 1.0
SESSION_START = (8, 30)
SESSION_END = (10, 0)
CLV_COL = "cs_5m_clv"
RV_COL = "rv_realized_vol_60"
DEFAULT_CACHE = "aabc60faea3863330077"
META_FEATURES = (
    "cs_5m_clv", "cs_10m_clv", "cs_15m_clv", "cs_clv",
    "rv_realized_vol_60", "rv_realized_vol_120", "rv_vol_ratio", "rv_vol_of_vol_120",
    "cs_range_ma_12", "cs_vol_regime",
    "vp_va_high_dist", "vp_poc_dist", "vp_in_value_area",
    "vw_vwap_dist", "vw_vwap_slope_120",
    "of_delta", "of_buy_ratio", "bp_imbalance", "tl_aggressive_pct",
    "tc_minutes_into_session",
)
TAU_GRID = tuple(round(x, 2) for x in np.linspace(0.15, 0.80, 14))


def chicago_index(ts: np.ndarray) -> pd.DatetimeIndex:
    idx = pd.to_datetime(ts, utc=True)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("America/Chicago")


def five_minute_close_mask(ts: np.ndarray) -> np.ndarray:
    """True on the 5s bar stamped at mm:ss = 00:00 of a 5-minute close."""
    ct = chicago_index(ts)
    return np.asarray((ct.minute % 5 == 0) & (ct.second == 0), dtype=bool)


def one_minute_close_mask(ts: np.ndarray) -> np.ndarray:
    """True on the 5s bar stamped at ss = 00 (1-minute close / ADR-041 visibility)."""
    ct = chicago_index(ts)
    return np.asarray(ct.second == 0, dtype=bool)


def event_clock_mask(ts: np.ndarray, clock: str = "5m") -> np.ndarray:
    if clock in ("1m", "1min", "1minute"):
        return one_minute_close_mask(ts)
    return five_minute_close_mask(ts)


def session_mask(ts: np.ndarray, start: tuple[int, int] = SESSION_START,
                 end: tuple[int, int] = SESSION_END) -> np.ndarray:
    ct = chicago_index(ts)
    minutes = ct.hour.to_numpy() * 60 + ct.minute.to_numpy()
    lo = start[0] * 60 + start[1]
    hi = end[0] * 60 + end[1]
    return (minutes >= lo) & (minutes < hi)


def date_mask(ts: np.ndarray, start: str, end: str) -> np.ndarray:
    ct = chicago_index(ts)
    lo = pd.Timestamp(str(start)[:10], tz="America/Chicago")
    hi = pd.Timestamp(str(end)[:10], tz="America/Chicago") + pd.Timedelta(days=1)
    return np.asarray((ct >= lo) & (ct < hi), dtype=bool)


def primary_side(clv: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """+1 long, −1 short, 0 skip."""
    out = np.zeros(len(clv), dtype=np.int8)
    x = np.asarray(clv, dtype=np.float64)
    out[x >= threshold] = 1
    out[x <= -threshold] = -1
    return out


def meta_win(side: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 iff acting on *side* would have won the TBM label."""
    y = np.asarray(y)
    s = np.asarray(side)
    return (((s == 1) & (y == 2)) | ((s == -1) & (y == 0))).astype(np.int8)


def cooldown_keep(ts: np.ndarray, bars: int = 120) -> np.ndarray:
    if len(ts) == 0:
        return np.zeros(0, dtype=bool)
    return keep_mask(ts, bars)


def _pnl_frame(side: np.ndarray, y: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    direction = np.where(side == 1, "up", np.where(side == -1, "down", "flat"))
    true = np.where(y == 2, "up", np.where(y == 0, "down", "flat"))
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True),
                         "direction": direction, "direction_true": true})


def score_trades(side: np.ndarray, y: np.ndarray, ts: np.ndarray) -> dict[str, Any]:
    fired = side != 0
    df = _pnl_frame(side[fired], y[fired], ts[fired])
    pnl = label_implied_pnl(df, win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST)
    pnl["n_long"] = int((side[fired] == 1).sum())
    pnl["n_short"] = int((side[fired] == -1).sum())
    return pnl


def choose_tau(p_win: np.ndarray, side: np.ndarray, y: np.ndarray, ts: np.ndarray,
               *, min_trades: int = 15) -> float:
    """Pick τ on TRAIN events: max NE among thresholds with enough cooldown trades."""
    best_tau, best_ne, best_n = 0.5, -1e18, 0
    for tau in TAU_GRID:
        take = p_win >= float(tau)
        if not take.any():
            continue
        kept = cooldown_keep(ts[take], 120)
        sc = score_trades(side[take][kept], y[take][kept], ts[take][kept])
        n = int(sc["n_trades"])
        ne = sc["net_expectancy"]
        if ne is None or n < min_trades:
            continue
        if ne > best_ne or (ne == best_ne and n > best_n):
            best_tau, best_ne, best_n = float(tau), float(ne), n
    return best_tau


def _xgb(scale_pos: float):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
        scale_pos_weight=float(scale_pos),
    )


def load_cache(cache_dir: Path) -> dict[str, Any]:
    meta = json.loads((cache_dir / "meta.json").read_text())
    cols = list(meta["feature_cols"])
    return {
        "cols": cols,
        "X": np.load(cache_dir / "X.npy", mmap_mode="r"),
        "y": np.load(cache_dir / "y.npy").astype(np.int32),
        "ts": np.load(cache_dir / "ts.npy"),
        "meta": meta,
    }


def _col_matrix(X: np.ndarray, cols: list[str], names: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    keep = [n for n in names if n in cols]
    idx = [cols.index(n) for n in keep]
    mat = np.nan_to_num(np.asarray(X[:, idx], dtype=np.float32), nan=0.0)
    return mat, keep


def build_events(
    bundle: dict[str, Any],
    *,
    clv_threshold: float = 0.4,
    rv_pct: float | None = 70.0,
    rv_cutoff: float | None = None,
    clock: str = "5m",
) -> dict[str, Any]:
    """Select session events. *rv_cutoff* from train overrides *rv_pct*."""
    cols, X, y, ts = bundle["cols"], bundle["X"], bundle["y"], bundle["ts"]
    if CLV_COL not in cols or RV_COL not in cols:
        raise ValueError(f"cache missing {CLV_COL} or {RV_COL}")
    clv = np.asarray(X[:, cols.index(CLV_COL)], dtype=np.float64)
    rv = np.asarray(X[:, cols.index(RV_COL)], dtype=np.float64)
    m = event_clock_mask(ts, clock) & session_mask(ts)
    side = primary_side(clv, clv_threshold)
    m = m & (side != 0)
    if rv_cutoff is None and rv_pct is not None:
        finite = rv[m & np.isfinite(rv)]
        rv_cutoff = float(np.percentile(finite, rv_pct)) if len(finite) else 0.0
    if rv_cutoff is not None:
        m = m & (rv >= float(rv_cutoff))
    ix = np.flatnonzero(m)
    feat, feat_names = _col_matrix(X, cols, META_FEATURES)
    extra = side[ix].astype(np.float32).reshape(-1, 1)
    F = np.hstack([feat[ix], extra])
    feat_names = feat_names + ["primary_side"]
    return {
        "ix": ix,
        "ts": ts[ix],
        "y": y[ix],
        "side": side[ix],
        "win": meta_win(side[ix], y[ix]),
        "F": F,
        "feat_names": feat_names,
        "rv_cutoff": rv_cutoff,
        "n_clock": int((event_clock_mask(ts, clock) & session_mask(ts)).sum()),
        "n_primary": int((event_clock_mask(ts, clock) & session_mask(ts) & (side != 0)).sum()),
        "clock": clock,
    }


def run_fold(
    bundle: dict[str, Any],
    fold: dict[str, Any],
    *,
    clv_threshold: float = 0.4,
    rv_pct: float = 70.0,
    clock: str = "5m",
) -> dict[str, Any]:
    ts = bundle["ts"]
    tr = date_mask(ts, fold["train"]["start"], fold["train"]["end"])
    oof = date_mask(ts, fold["oof"]["start"], fold["oof"]["end"])
    # Purge last 10 minutes of train (TBM horizon) so train labels don't overlap OOF.
    if oof.any():
        oof0 = pd.to_datetime(ts[oof][0], utc=True)
        tr_ts = pd.to_datetime(ts[tr], utc=True) if tr.any() else pd.DatetimeIndex([], tz="UTC")
        tr = tr.copy()
        tr[tr] = np.asarray(tr_ts <= (oof0 - pd.Timedelta(minutes=10)))

    # RV cutoff from train 5m-session bars (primary applied after).
    cols = bundle["cols"]
    rv_all = np.asarray(bundle["X"][:, cols.index(RV_COL)], dtype=np.float64)
    clock_m = event_clock_mask(ts, clock) & session_mask(ts) & tr
    finite = rv_all[clock_m & np.isfinite(rv_all)]
    use_rv = rv_pct is not None and float(rv_pct) > 0
    rv_cut = float(np.percentile(finite, rv_pct)) if (use_rv and len(finite)) else None

    ev_all = build_events(
        bundle, clv_threshold=clv_threshold, rv_pct=None,
        rv_cutoff=rv_cut, clock=clock,
    )
    in_tr = date_mask(ev_all["ts"], fold["train"]["start"], fold["train"]["end"])
    in_oof = date_mask(ev_all["ts"], fold["oof"]["start"], fold["oof"]["end"])
    if in_tr.any() and oof.any():
        oof0 = pd.to_datetime(ts[oof][0], utc=True)
        in_tr = in_tr & np.asarray(
            pd.to_datetime(ev_all["ts"], utc=True) <= (oof0 - pd.Timedelta(minutes=10))
        )

    Ftr, ytr = ev_all["F"][in_tr], ev_all["win"][in_tr]
    if ytr.sum() == 0 or (ytr == 0).sum() == 0 or len(ytr) < 30:
        return {"fold": fold["fold"], "status": "skipped",
                "reason": "too_few_train_events", "n_train": int(len(ytr))}
    scale = float((ytr == 0).sum() / max(int(ytr.sum()), 1))
    model = _xgb(scale)
    model.fit(Ftr, ytr)
    p_tr = model.predict_proba(Ftr)[:, 1]
    tau = choose_tau(p_tr, ev_all["side"][in_tr], ev_all["y"][in_tr], ev_all["ts"][in_tr])

    Foof = ev_all["F"][in_oof]
    p_oof = model.predict_proba(Foof)[:, 1] if in_oof.any() else np.zeros(0)
    side_oof = ev_all["side"][in_oof]
    y_oof = ev_all["y"][in_oof]
    ts_oof = ev_all["ts"][in_oof]

    primary_kept = cooldown_keep(ts_oof, 120) if len(ts_oof) else np.zeros(0, dtype=bool)
    primary_sc = score_trades(side_oof[primary_kept], y_oof[primary_kept], ts_oof[primary_kept]) if primary_kept.any() else score_trades(np.zeros(0, np.int8), np.zeros(0), np.array([], dtype="datetime64[ns]"))

    take = p_oof >= tau
    meta_side = np.where(take, side_oof, 0).astype(np.int8)
    if take.any():
        kept = np.zeros(len(take), dtype=bool)
        kept[np.flatnonzero(take)] = cooldown_keep(ts_oof[take], 120)
        meta_side = np.where(kept, side_oof, 0).astype(np.int8)
    meta_sc = score_trades(meta_side, y_oof, ts_oof)

    auc = None
    if in_oof.any() and y_oof.size and ev_all["win"][in_oof].min() != ev_all["win"][in_oof].max():
        try:
            auc = float(roc_auc_score(ev_all["win"][in_oof], p_oof))
        except ValueError:
            auc = None

    oof_frame = pd.DataFrame({
        "ts": pd.to_datetime(ts_oof, utc=True),
        "direction": np.where(meta_side == 1, "up", np.where(meta_side == -1, "down", "flat")),
        "confidence": p_oof if len(p_oof) else [],
        "direction_true": np.where(y_oof == 2, "up", np.where(y_oof == 0, "down", "flat")),
        "primary_side": side_oof,
        "win": ev_all["win"][in_oof],
    }) if in_oof.any() else pd.DataFrame()

    return {
        "fold": fold["fold"],
        "status": "ok",
        "train": fold["train"],
        "oof": fold["oof"],
        "rv_cutoff": rv_cut,
        "tau": tau,
        "n_train_events": int(in_tr.sum()),
        "n_oof_events": int(in_oof.sum()),
        "train_win_rate": float(ytr.mean()),
        "meta_auc_oof": auc,
        "primary": primary_sc,
        "meta": meta_sc,
        "oof_frame": oof_frame,
        "scale_pos_weight": scale,
    }


def run_search(
    cache_dir: Path,
    *,
    clv_threshold: float = 0.4,
    rv_pct: float = 70.0,
    clock: str = "5m",
    start: str = DEFAULT_START,
    stop: str = DEFAULT_STOP,
) -> dict[str, Any]:
    bundle = load_cache(cache_dir)
    schedule = build_expanding_schedule(start=start, stop=stop)
    folds = []
    frames = []
    for spec in schedule:
        logger.info("event_meta fold %s train=%s oof=%s", spec["fold"], spec["train"], spec["oof"])
        rec = run_fold(bundle, spec, clv_threshold=clv_threshold, rv_pct=rv_pct, clock=clock)
        fr = rec.pop("oof_frame", pd.DataFrame())
        folds.append(rec)
        if rec.get("status") == "ok" and len(fr):
            fr["fold"] = rec["fold"]
            frames.append(fr)

    oof = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(oof):
        oof = oof.sort_values("ts")
        fired = oof["direction"].isin(["up", "down"])
        pooled = label_implied_pnl(oof.loc[fired], win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST)
        prim = oof[oof["primary_side"] != 0]
        # primary pooled with cooldown on all primary events (already applied per fold)
        prim_fired = oof["primary_side"] != 0
        # Re-score primary as always-approve then cooldown on the pooled timeline.
        side_p = oof["primary_side"].to_numpy()
        kept = np.zeros(len(oof), dtype=bool)
        idx = np.flatnonzero(side_p != 0)
        if len(idx):
            kept[idx] = cooldown_keep(oof["ts"].to_numpy()[idx], 120)
        pooled_primary = score_trades(
            np.where(kept, side_p, 0).astype(np.int8),
            np.where(oof["direction_true"] == "up", 2,
                     np.where(oof["direction_true"] == "down", 0, 1)),
            oof["ts"].to_numpy(),
        )
    else:
        pooled = {"n_trades": 0, "net_expectancy": None, "hit_rate": None, "max_drawdown": 0.0}
        pooled_primary = pooled

    return {
        "cache": str(cache_dir),
        "clv_threshold": clv_threshold,
        "rv_pct": rv_pct,
        "session": {"start": "08:30", "end": "10:00", "tz": "America/Chicago"},
        "event_clock": clock,
        "cooldown_bars": 120,
        "folds": folds,
        "pooled_meta": pooled,
        "pooled_primary": pooled_primary,
        "oof": oof,
        "schedule": schedule,
        "n_cache": int(len(bundle["y"])),
    }


def score_canonical_policy(
    oof: pd.DataFrame,
    *,
    tau: float = 0.65,
    cooldown_bars: int = 120,
    session_start: tuple[int, int] = SESSION_START,
    session_end: tuple[int, int] = SESSION_END,
    walk_forward_tau: bool = True,
) -> dict[str, Any]:
    """Apply open-session + τ + 10-minute cooldown to a canonical OOF frame.

    This is inference uniqueness (the half of ADR-045 that train-only
    uniqueness did not implement). Optional walk-forward τ: pick on prior
    OOF folds, apply to the next.
    """
    df = oof.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts")
    ts_np = df["ts"].to_numpy()
    sess = session_mask(ts_np, session_start, session_end)
    dir_ok = df["direction"].isin(["up", "down"]).to_numpy()

    def _take(mask: np.ndarray) -> pd.DataFrame:
        g = df.loc[mask]
        if g.empty:
            return g
        kept = apply_trade_cooldown(g["ts"].to_numpy(), cooldown_bars)
        return g.loc[kept]

    fixed = _take(sess & dir_ok & (df["confidence"].to_numpy() >= float(tau)))
    fixed_sc = label_implied_pnl(fixed, win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST)

    wf_parts: list[pd.DataFrame] = []
    wf_folds: list[dict[str, Any]] = []
    if walk_forward_tau:
        schedule = build_expanding_schedule(start=DEFAULT_START, stop=DEFAULT_STOP)
        ct = chicago_index(ts_np)
        best_tau = float(tau)
        for spec in schedule:
            lo = pd.Timestamp(spec["oof"]["start"], tz="America/Chicago")
            hi = pd.Timestamp(spec["oof"]["end"], tz="America/Chicago") + pd.Timedelta(days=1)
            oof_m = np.asarray((ct >= lo) & (ct < hi))
            prev = np.asarray(ct < lo)
            cand_tau, cand_ne = best_tau, -1e18
            if prev.any():
                for ttry in TAU_GRID:
                    g = _take(prev & sess & dir_ok & (df["confidence"].to_numpy() >= ttry))
                    sc = label_implied_pnl(g, win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST)
                    n = int(sc["n_trades"])
                    ne = sc["net_expectancy"]
                    if ne is not None and n >= 8 and float(ne) > cand_ne:
                        cand_ne, cand_tau = float(ne), float(ttry)
            best_tau = cand_tau
            g = _take(oof_m & sess & dir_ok & (df["confidence"].to_numpy() >= best_tau))
            sc = label_implied_pnl(g, win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST)
            wf_folds.append({"fold": spec["fold"], "tau": best_tau, "oof": spec["oof"], **sc})
            if len(g):
                wf_parts.append(g)
        wf = pd.concat(wf_parts).sort_values("ts") if wf_parts else pd.DataFrame()
        wf_sc = label_implied_pnl(wf, win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST) if len(wf) else {
            "n_trades": 0, "net_expectancy": None, "hit_rate": None, "max_drawdown": 0.0,
        }
    else:
        wf_sc, wf_folds = {}, []

    raw = label_implied_pnl(
        df[dir_ok & (df["confidence"].to_numpy() >= float(tau))],
        win_points=WIN_PTS, loss_points=LOSS_PTS, cost=COST,
    )
    return {
        "tau_fixed": float(tau),
        "session": {"start": "08:30", "end": "10:00", "tz": "America/Chicago"},
        "cooldown_bars": cooldown_bars,
        "raw_overlapping": raw,
        "open_cooldown": fixed_sc,
        "walk_forward": wf_sc,
        "walk_forward_folds": wf_folds,
        "n_open_cooldown_trades": int(fixed_sc.get("n_trades") or 0),
    }


def persist(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    oof: pd.DataFrame = result.get("oof") if isinstance(result.get("oof"), pd.DataFrame) else pd.DataFrame()
    if len(oof):
        oof.to_parquet(out_dir / "oof_event_meta.parquet", index=False)
    slim = {k: v for k, v in result.items() if k != "oof"}
    (out_dir / "event_meta_result.json").write_text(json.dumps(slim, indent=2, default=str))
    return out_dir


def _cli() -> None:
    p = argparse.ArgumentParser(description="Event-sampled primary + XGB veto")
    p.add_argument("--cache-key", default=DEFAULT_CACHE)
    p.add_argument("--clv-threshold", type=float, default=0.4)
    p.add_argument("--rv-pct", type=float, default=70.0,
                   help="Train-only RV percentile gate; 0 disables")
    p.add_argument("--clock", default="5m", choices=("5m", "1m"))
    p.add_argument("--oof-parquet", default="",
                   help="Score an existing canonical OOF with open+cooldown policy")
    p.add_argument("--tau", type=float, default=0.65)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) if args.out_dir else search_dir() / "nq-50pt-xgb-unique" / run_ts
    if args.oof_parquet:
        oof = pd.read_parquet(args.oof_parquet)
        print(f"⏳ Policy score {args.oof_parquet} (open 08:30–10:00 + {args.tau} τ + 10m cooldown)…")
        policy = score_canonical_policy(oof, tau=args.tau)
        out.mkdir(parents=True, exist_ok=True)
        (out / "inference_policy.json").write_text(json.dumps(policy, indent=2, default=str))
        print(f"✅ wrote {out / 'inference_policy.json'}")
        print(f"  overlapping n={policy['raw_overlapping'].get('n_trades')} NE={policy['raw_overlapping'].get('net_expectancy')} HR={policy['raw_overlapping'].get('hit_rate')}")
        print(f"  open+cd     n={policy['open_cooldown'].get('n_trades')} NE={policy['open_cooldown'].get('net_expectancy')} HR={policy['open_cooldown'].get('hit_rate')} MDD={policy['open_cooldown'].get('max_drawdown')}")
        print(f"  WF tau      n={policy['walk_forward'].get('n_trades')} NE={policy['walk_forward'].get('net_expectancy')} HR={policy['walk_forward'].get('hit_rate')} MDD={policy['walk_forward'].get('max_drawdown')}")
        return
    cache = search_dir() / "_fold_cache" / args.cache_key
    print(f"⏳ Event-meta walk-forward on {cache.name}…")
    result = run_search(
        cache, clv_threshold=args.clv_threshold, rv_pct=args.rv_pct, clock=args.clock,
    )
    persist(result, out)
    meta = result["pooled_meta"]
    prim = result["pooled_primary"]
    print(f"✅ wrote {out}")
    print(f"  primary  n={prim.get('n_trades')} HR={prim.get('hit_rate')} NE={prim.get('net_expectancy')} MDD={prim.get('max_drawdown')}")
    print(f"  meta     n={meta.get('n_trades')} HR={meta.get('hit_rate')} NE={meta.get('net_expectancy')} MDD={meta.get('max_drawdown')}")
    print(f"  folds    {[f.get('status') for f in result['folds']]}")


if __name__ == "__main__":
    _cli()
