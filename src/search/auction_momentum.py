"""Auction-momentum strategy for TBM +50/−6 (independent of operator feature hints).

A +50 first-touch with a −6 stop is a *session impulse*, not a 5s tick. Direction
is taken from causal open-drive / session-extreme geometry; the 6-pt stop is an
entry filter (do not buy a pullback already larger than the stop). Long and short
are separate one-sided heads. Live pin is not changed.
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
from src.search.event_meta import (
    DEFAULT_CACHE,
    chicago_index,
    date_mask,
    load_cache,
    score_trades,
    session_mask,
)
from src.search.side_head import _xgb, aligned_close
from src.search.uniqueness import apply_trade_cooldown, keep_mask
from src.validation.walk_forward import (
    DEFAULT_START,
    DEFAULT_STOP,
    build_expanding_schedule,
)

logger = logging.getLogger("vtech.search.auction_momentum")

RTH = (8, 30)


def auction_features(ts: np.ndarray, px: np.ndarray) -> dict[str, np.ndarray]:
    """Causal per-Chicago-day session geometry from close.

    Running high/low/open use only prices at or before t (inclusive of the
    current closed 5s bar).
    """
    ct = chicago_index(ts)
    days = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in ct])
    minutes = ct.hour.to_numpy() * 60 + ct.minute.to_numpy()
    n = len(px)
    ret_open = np.full(n, np.nan)
    pb_high = np.full(n, np.nan)
    rally_low = np.full(n, np.nan)
    mins_since_high = np.full(n, np.nan)
    mins_since_low = np.full(n, np.nan)
    ret_1m = np.full(n, np.nan)
    ret_5m = np.full(n, np.nan)
    rth_m = 8 * 60 + 30
    for d in sorted(set(days.tolist())):
        m = days == d
        ix = np.flatnonzero(m)
        sl = px[ix].astype(np.float64)
        finite = np.isfinite(sl)
        if finite.sum() < 12:
            continue
        # RTH open = first finite close at/after 08:30 CT.
        loc_min = minutes[ix]
        rth_ix = np.flatnonzero(finite & (loc_min >= rth_m))
        if len(rth_ix) == 0:
            rth_ix = np.flatnonzero(finite)
        open_px = sl[rth_ix[0]]
        run_hi = np.maximum.accumulate(np.where(finite, sl, -np.inf))
        run_lo = np.minimum.accumulate(np.where(finite, sl, np.inf))
        run_hi[np.isneginf(run_hi)] = np.nan
        run_lo[np.isposinf(run_lo)] = np.nan
        ret_open[ix] = sl - open_px
        pb_high[ix] = run_hi - sl
        rally_low[ix] = sl - run_lo
        # bars since extreme (5s)
        last_hi = 0
        last_lo = 0
        msh = np.zeros(len(ix), dtype=np.float64)
        msl = np.zeros(len(ix), dtype=np.float64)
        for j in range(len(ix)):
            if finite[j] and sl[j] >= run_hi[j] - 1e-9:
                last_hi = j
            if finite[j] and sl[j] <= run_lo[j] + 1e-9:
                last_lo = j
            msh[j] = (j - last_hi) / 12.0
            msl[j] = (j - last_lo) / 12.0
        mins_since_high[ix] = msh
        mins_since_low[ix] = msl
        lag12 = np.roll(sl, 12)
        lag60 = np.roll(sl, 60)
        lag12[:12] = np.nan
        lag60[:60] = np.nan
        ret_1m[ix] = sl - lag12
        ret_5m[ix] = sl - lag60
    return {
        "ret_from_open": ret_open,
        "pullback_from_high": pb_high,
        "rally_from_low": rally_low,
        "mins_since_high": mins_since_high,
        "mins_since_low": mins_since_low,
        "ret_1m": ret_1m,
        "ret_5m": ret_5m,
        "minutes_ct": minutes.astype(np.float64),
    }


def _rule_score(side: np.ndarray, y: np.ndarray, ts: np.ndarray, name: str) -> dict[str, Any]:
    ix = np.flatnonzero(side != 0)
    kept = np.zeros(len(y), dtype=bool)
    if len(ix):
        kept[ix] = apply_trade_cooldown(ts[ix], 120)
    sl = np.where(kept, side, 0).astype(np.int8)
    sc = score_trades(sl, y, ts)
    sc["name"] = name
    sc["n_long"] = int((sl == 1).sum())
    sc["n_short"] = int((sl == -1).sum())
    return sc


def scan_rules(feat: dict[str, np.ndarray], y: np.ndarray, ts: np.ndarray,
               rv: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Handed auction setups — designed for +50/−6, not suggested by the operator."""
    sess = session_mask(ts)
    opn = feat["ret_from_open"]
    pb = feat["pullback_from_high"]
    ry = feat["rally_from_low"]
    r1 = feat["ret_1m"]
    r5 = feat["ret_5m"]
    msh = feat["mins_since_high"]
    msl = feat["mins_since_low"]
    mins = feat["minutes_ct"]
    early = (mins >= 8 * 60 + 30) & (mins < 9 * 60 + 45)
    rv_hi = np.ones(len(y), dtype=bool)
    if rv is not None and np.isfinite(rv).any():
        cut = np.nanpercentile(rv[sess & np.isfinite(rv)], 70)
        rv_hi = rv >= cut

    rules = []
    z = np.zeros(len(y), dtype=np.int8)

    def put(name, long_m, short_m=None):
        s = z.copy()
        lm = np.asarray(long_m, dtype=bool)
        s[np.nan_to_num(lm.astype(np.float64), nan=0).astype(bool)] = 1
        if short_m is not None:
            sm = np.nan_to_num(np.asarray(short_m, dtype=np.float64), nan=0).astype(bool)
            s[sm] = -1
            s[lm & sm] = 0
        rules.append(_rule_score(s, y, ts, name))

    none = np.zeros(len(y), dtype=bool)
    # Long: already +20 off the low, still within 6 of the high (stop-sized pullback).
    put("L_impulse_tight_pb",
        sess & early & (ry >= 20) & (pb <= 6) & rv_hi, none)
    put("L_open_drive",
        sess & early & (opn >= 20) & (r1 >= 0) & (pb <= 8), none)
    put("L_new_high",
        sess & early & (msh <= 0.25) & (opn >= 15) & rv_hi, none)
    put("S_open_drive", none,
        sess & early & (opn <= -20) & (ry <= 6) & (r1 <= 0))
    put("S_impulse_tight_bounce", none,
        sess & early & (pb >= 20) & (ry <= 6) & rv_hi)
    put("S_new_low", none,
        sess & early & (msl <= 0.25) & (opn <= -15) & rv_hi)
    put("LS_impulse",
        sess & early & (ry >= 20) & (pb <= 6) & rv_hi,
        sess & early & (pb >= 20) & (ry <= 6) & rv_hi)
    put("LS_5m_cont",
        sess & early & (r5 >= 12) & (r1 >= -2) & rv_hi,
        sess & early & (r5 <= -12) & (r1 <= 2) & rv_hi)
    return rules


def _fit_onesided(
    F: np.ndarray, y_bin: np.ndarray, ts: np.ndarray,
    train: np.ndarray, oof: np.ndarray,
) -> tuple[np.ndarray, Any]:
    """P(positive class) on all rows; train uniqueness 10m."""
    tr = train.copy()
    if oof.any():
        oof0 = pd.to_datetime(ts[oof][0], utc=True)
        tr = tr & np.asarray(pd.to_datetime(ts, utc=True) <= (oof0 - pd.Timedelta(minutes=10)))
    ix = np.flatnonzero(tr)
    if len(ix):
        keep = np.zeros(len(y_bin), dtype=bool)
        keep[ix] = keep_mask(ts[ix], 120)
        tr = keep
    n_pos = int(y_bin[tr].sum())
    n_neg = int(tr.sum() - n_pos)
    if n_pos < 25 or n_neg < 25:
        return np.full(len(y_bin), np.nan), None
    model = _xgb(n_neg / max(n_pos, 1))
    model.fit(F[tr], y_bin[tr])
    return model.predict_proba(F)[:, 1], model


def run_auction(
    cache_dir: Path,
    *,
    long_tau: float = 0.55,
    short_tau: float = 0.55,
) -> dict[str, Any]:
    bundle = load_cache(cache_dir)
    cols, X, y, ts = bundle["cols"], bundle["X"], bundle["y"].astype(np.int32), bundle["ts"]
    print("⏳ Loading _close + auction features…")
    px = aligned_close(ts)
    feat = auction_features(ts, px)
    rv = X[:, cols.index("rv_realized_vol_60")] if "rv_realized_vol_60" in cols else None
    cs5 = X[:, cols.index("cs_5m_clv")] if "cs_5m_clv" in cols else np.zeros(len(y))
    rng = X[:, cols.index("cs_range_ma_12")] if "cs_range_ma_12" in cols else np.zeros(len(y))

    print("⏳ Scanning auction rules (10m cooldown, 50/6 payoff)…")
    rules = scan_rules(feat, y, ts, rv=rv)

    # Dual one-sided heads on auction geometry + a little tape.
    parts = [np.nan_to_num(feat[k], nan=0.0).reshape(-1, 1) for k in (
        "ret_from_open", "pullback_from_high", "rally_from_low",
        "mins_since_high", "mins_since_low", "ret_1m", "ret_5m", "minutes_ct",
    )]
    parts.append(np.nan_to_num(cs5, nan=0.0).reshape(-1, 1))
    parts.append(np.nan_to_num(rng, nan=0.0).reshape(-1, 1))
    if rv is not None:
        parts.append(np.nan_to_num(rv, nan=0.0).reshape(-1, 1))
    F = np.hstack(parts).astype(np.float32)
    sess = session_mask(ts)
    y_up = (y == 2).astype(np.int32)
    y_dn = (y == 0).astype(np.int32)

    schedule = build_expanding_schedule(start=DEFAULT_START, stop=DEFAULT_STOP)
    p_up = np.full(len(y), np.nan)
    p_dn = np.full(len(y), np.nan)
    last_up = last_dn = None
    folds = []
    print("⏳ Walk-forward one-sided long/short heads…")
    for spec in schedule:
        tr = date_mask(ts, spec["train"]["start"], spec["train"]["end"]) & sess
        oof = date_mask(ts, spec["oof"]["start"], spec["oof"]["end"])
        pu, up_m = _fit_onesided(F, y_up, ts, tr, oof)
        pdn, dn_m = _fit_onesided(F, y_dn, ts, tr, oof)
        p_up[oof] = pu[oof]
        p_dn[oof] = pdn[oof]
        if up_m is not None:
            last_up = up_m
        if dn_m is not None:
            last_dn = dn_m
        oof_s = oof & sess
        auc_u = auc_d = None
        if oof_s.sum() > 50 and y_up[oof_s].min() != y_up[oof_s].max():
            try:
                auc_u = float(roc_auc_score(y_up[oof_s], pu[oof_s]))
                auc_d = float(roc_auc_score(y_dn[oof_s], pdn[oof_s]))
            except ValueError:
                pass
        folds.append({"fold": spec["fold"], "auc_up": auc_u, "auc_down": auc_d,
                      "train": spec["train"], "oof": spec["oof"]})
        logger.info("auction fold %s auc_up=%s auc_down=%s", spec["fold"], auc_u, auc_d)

    oof_any = np.isfinite(p_up) & np.isfinite(p_dn)
    # Fire long if p_up high and p_down not high; short symmetric. Grid on train-ish
    # pooled OOF is leaky — use fixed taus, report a small grid as diagnostics only.
    side = np.zeros(len(y), dtype=np.int8)
    long_m = oof_any & sess & (p_up >= long_tau) & (p_dn < short_tau)
    short_m = oof_any & sess & (p_dn >= short_tau) & (p_up < long_tau)
    both = long_m & short_m
    side[long_m] = 1
    side[short_m] = -1
    side[both] = 0
    model_sc = _rule_score(side, y, ts, f"dual_p{long_tau:.2f}")

    # Long-only / short-only from the same heads.
    lo = _rule_score(np.where(oof_any & sess & (p_up >= long_tau), 1, 0).astype(np.int8), y, ts, "head_long_only")
    so = _rule_score(np.where(oof_any & sess & (p_dn >= short_tau), -1, 0).astype(np.int8), y, ts, "head_short_only")

    tau_grid = []
    for lt in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        for st in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            s = np.zeros(len(y), dtype=np.int8)
            lm = oof_any & sess & (p_up >= lt) & (p_dn < st)
            sm = oof_any & sess & (p_dn >= st) & (p_up < lt)
            s[lm] = 1
            s[sm] = -1
            s[lm & sm] = 0
            rec = _rule_score(s, y, ts, f"dual_{lt:.2f}_{st:.2f}")
            rec["long_tau"] = lt
            rec["short_tau"] = st
            tau_grid.append(rec)

    return {
        "cache": str(cache_dir),
        "rules": rules,
        "folds": folds,
        "dual": model_sc,
        "head_long_only": lo,
        "head_short_only": so,
        "tau_grid": tau_grid,
        "auc_up_folds": [f.get("auc_up") for f in folds],
        "auc_down_folds": [f.get("auc_down") for f in folds],
        "feature_cols": [
            "ret_from_open", "pullback_from_high", "rally_from_low",
            "mins_since_high", "mins_since_low", "ret_1m", "ret_5m", "minutes_ct",
            "cs_5m_clv", "cs_range_ma_12", "rv_realized_vol_60",
        ],
        "_last_up": last_up,
        "_last_dn": last_dn,
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description="Auction-momentum +50/−6 strategy")
    p.add_argument("--cache-key", default=DEFAULT_CACHE)
    p.add_argument("--long-tau", type=float, default=0.55)
    p.add_argument("--short-tau", type=float, default=0.55)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cache = search_dir() / "_fold_cache" / args.cache_key
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) if args.out_dir else search_dir() / "nq-50pt-xgb-unique" / run_ts
    result = run_auction(cache, long_tau=args.long_tau, short_tau=args.short_tau)
    out.mkdir(parents=True, exist_ok=True)
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    last_up = result.pop("_last_up", None)
    last_dn = result.pop("_last_dn", None)
    feat_cols = result.get("feature_cols") or []
    if last_up is not None:
        last_up.save_model(str(models / "long_head.ubj"))
    if last_dn is not None:
        last_dn.save_model(str(models / "short_head.ubj"))
    rule = {
        "kind": "auction_short_rule",
        "objective": "nq-50pt-xgb-unique TBM +50/-6 cost 1",
        "session": "08:30-09:45 America/Chicago",
        "direction": "short",
        "when": [
            "pullback from today's session high >= 20 NQ points",
            "bounce from the low of that dump <= 6 NQ points",
            "realized vol >= 70th percentile of *train* session bars",
        ],
        "cooldown_bars": 120,
        "note": "This rule is the short book (ADR-051). The .ubj files are last-fold one-sided XGBs on the same auction features; they over-fire vs the rule.",
    }
    (models / "short_rule.json").write_text(json.dumps(rule, indent=2))
    (models / "feature_schema.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "n_features": len(feat_cols),
        "auction_from": "_close per Chicago day",
    }, indent=2))
    (out / "auction_momentum_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"✅ wrote {out / 'auction_momentum_result.json'}")
    print(f"  models: {models}")
    if last_dn is not None:
        print(f"    short_head.ubj  long_head.ubj  short_rule.json")
    print("\n  RULES")
    for r in result["rules"]:
        print(f"  {r['name']:28s} n={r.get('n_trades'):4d} HR={r.get('hit_rate')} "
              f"NE={r.get('net_expectancy')} MDD={r.get('max_drawdown')} "
              f"L/S={r.get('n_long')}/{r.get('n_short')}")
    print("\n  HEADS")
    for k in ("dual", "head_long_only", "head_short_only"):
        r = result[k]
        print(f"  {k:28s} n={r.get('n_trades'):4d} HR={r.get('hit_rate')} "
              f"NE={r.get('net_expectancy')} MDD={r.get('max_drawdown')} "
              f"L/S={r.get('n_long')}/{r.get('n_short')}")
    print(f"  auc_up   {result['auc_up_folds']}")
    print(f"  auc_down {result['auc_down_folds']}")
    print("\n  TAU GRID (OOF dual, 10m cd)")
    print(f"  {'lt':>5} {'st':>5} {'n':>5} {'HR':>7} {'NE':>8} {'MDD':>8} {'L':>5} {'S':>5}")
    for r in result.get("tau_grid") or []:
        print(f"  {r['long_tau']:5.2f} {r['short_tau']:5.2f} {r.get('n_trades'):5d} "
              f"{r.get('hit_rate') or float('nan'):7.3f} {r.get('net_expectancy') or float('nan'):8.2f} "
              f"{r.get('max_drawdown'):8.1f} {r.get('n_long'):5d} {r.get('n_short'):5d}")


if __name__ == "__main__":
    _cli()
