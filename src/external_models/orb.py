"""Opening Range Breakout engine for /NQ with account-aware risk governor (ADR-118).

Design notes
------------

The academic ORB literature (Zarattini & Aziz 2023/2025; Zarattini, Barbon &
Aziz 2024) establishes three things this engine takes seriously:

1. **An unfiltered ORB is weak.** The 2024 US-equity study found a plain
   5-minute ORB returned ~3.2 % annualised. The edge appeared only after
   filtering for *abnormal opening participation* (relative volume). We
   therefore gate every entry on ``min_rvol`` — the opening range's volume
   divided by its own trailing average.
2. **Payoff is asymmetric with a low hit rate.** The QQQ study reports a 24 %
   win rate carried by rare 10R winners, and 22–28 % max drawdown.
3. **(2) is incompatible with a hard drawdown floor.** A 24 %-win-rate
   process routinely strings 8–12 losses together. On a $5 000 account with a
   $1 000 loss barrier that is an account-ending sequence *even though the
   strategy has positive expectancy*.

So this engine deliberately **inverts** the academic payoff profile: it takes
partial profit early (``scale_out_pct`` at ``t1_r``) to lift the hit rate,
moves the remainder to breakeven, and lets only the residual runner chase the
larger target. That trades away tail upside for path safety, which is the
correct objective when a drawdown barrier — not volatility — is the binding
constraint.

Three setups, all evaluated on closed bars:

* **breakout** — continuation through the OR extreme (the classic ORB).
* **failed_break** — price pokes through the OR extreme, then closes back
  inside; fade back across the range. Professionally this is the highest
  expectancy ORB variant because it traps breakout entrants.
* **fade** — balance/inside-range day: reject an OR extreme from within the
  range and rotate toward the midpoint. This is what keeps the model
  productive on the ~40 % of sessions that never trend.

Risk is **contract-based, not percentage-based**: the stop distance in points
determines the size (1…``max_contracts``), so every trade risks approximately
``risk_usd_per_trade``. Trades whose structural stop is too wide to fit even
one contract inside the risk budget are skipped rather than downsized, which
is what keeps the loss distribution tight enough for the drawdown floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CT = ZoneInfo("America/Chicago")
DEFAULT_POINT_VALUE = 20.0
TICK = 0.25

SETUPS: tuple[str, ...] = ("breakout", "failed_break", "fade")


# ── session helpers ──────────────────────────────────────────────────────

def _hm_minutes(hhmm: str) -> int:
    h, m = str(hhmm).split(":")[:2]
    return int(h) * 60 + int(m)


def _ct_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Attach Chicago clock/session-date columns to a UTC-indexed OHLC frame."""
    df = ohlc.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    local = df.index.tz_convert(CT)
    df["_ct_min"] = local.hour * 60 + local.minute
    df["_date"] = local.date
    return df


def _round_tick(px: float) -> float:
    return round(px / TICK) * TICK


# ── per-session context ──────────────────────────────────────────────────

@dataclass
class SessionCtx:
    date: Any
    or_high: float
    or_low: float
    or_mid: float
    or_range: float
    or_volume: float
    rvol: float
    atr: float
    pd_high: float
    pd_low: float
    pd_vah: float
    pd_val: float
    pd_poc: float
    or_dir: int                 # +1 bullish OR candle, -1 bearish, 0 doji
    gap: float                  # RTH open − prior session close


def _bar_volume_profile(
    bars: pd.DataFrame, *, bin_pts: float = 1.0, va_pct: float = 0.70,
) -> tuple[float, float, float]:
    """POC / VAH / VAL from 1-minute bars (volume spread across each bar's range).

    Tape-accurate profiles need TBBO ticks; for a daily reference level a
    bar-distributed profile is within a tick or two of the tick-built one and
    costs three orders of magnitude less to compute.
    """
    if bars is None or bars.empty:
        return (float("nan"),) * 3
    lo = float(bars["low"].min())
    hi = float(bars["high"].max())
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return (float("nan"),) * 3
    n_bins = max(2, min(600, int(round((hi - lo) / bin_pts))))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = np.zeros(n_bins, dtype="float64")
    b_lo = bars["low"].to_numpy(dtype="float64")
    b_hi = bars["high"].to_numpy(dtype="float64")
    b_v = bars["volume"].to_numpy(dtype="float64")
    for l, h, v in zip(b_lo, b_hi, b_v):
        if not (math.isfinite(l) and math.isfinite(h) and math.isfinite(v)) or v <= 0:
            continue
        i0 = int(np.searchsorted(edges, l, side="right")) - 1
        i1 = int(np.searchsorted(edges, h, side="left"))
        i0 = max(0, min(n_bins - 1, i0))
        i1 = max(i0 + 1, min(n_bins, i1))
        hist[i0:i1] += v / float(i1 - i0)
    total = float(hist.sum())
    if total <= 0:
        return (float("nan"),) * 3
    poc_i = int(np.argmax(hist))
    poc = float(centers[poc_i])
    # expand around the POC until va_pct of volume is enclosed
    target = total * va_pct
    lo_i = hi_i = poc_i
    acc = float(hist[poc_i])
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1.0
        up = hist[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            acc += float(hist[hi_i])
        else:
            lo_i -= 1
            acc += float(hist[lo_i])
    return poc, float(centers[hi_i]), float(centers[lo_i])


def build_session_contexts(
    ohlc: pd.DataFrame, p: dict[str, Any],
) -> dict[Any, SessionCtx]:
    """One :class:`SessionCtx` per trading date. Uses only prior-session data
    for ATR / prior-day levels, and the just-closed OR for the day's own levels."""
    df = _ct_frame(ohlc)
    rth_open = _hm_minutes(p["session_start"])
    or_end = rth_open + int(p["or_minutes"])
    rth_close = _hm_minutes(p["session_end"])
    rvol_lb = int(p["rvol_lookback"])
    atr_lb = int(p["atr_days"])

    dates = sorted({d for d in df["_date"].unique()})
    rth = df[(df["_ct_min"] >= rth_open) & (df["_ct_min"] < rth_close)]
    or_bars_all = df[(df["_ct_min"] >= rth_open) & (df["_ct_min"] < or_end)]

    # per-date aggregates
    day_hi = rth.groupby("_date")["high"].max()
    day_lo = rth.groupby("_date")["low"].min()
    day_close = rth.groupby("_date")["close"].last()
    day_open = rth.groupby("_date")["open"].first()
    or_hi = or_bars_all.groupby("_date")["high"].max()
    or_lo = or_bars_all.groupby("_date")["low"].min()
    or_vol = or_bars_all.groupby("_date")["volume"].sum()
    or_open = or_bars_all.groupby("_date")["open"].first()
    or_close = or_bars_all.groupby("_date")["close"].last()

    # daily true range → Wilder-ish simple ATR over prior sessions
    tr_by_date: dict[Any, float] = {}
    prev_c: float | None = None
    for d in dates:
        h, l = day_hi.get(d), day_lo.get(d)
        if h is None or l is None or not math.isfinite(h) or not math.isfinite(l):
            prev_c = day_close.get(d, prev_c)
            continue
        tr = h - l
        if prev_c is not None and math.isfinite(prev_c):
            tr = max(tr, abs(h - prev_c), abs(l - prev_c))
        tr_by_date[d] = float(tr)
        prev_c = day_close.get(d, prev_c)

    out: dict[Any, SessionCtx] = {}
    for i, d in enumerate(dates):
        if d not in or_hi.index or d not in or_lo.index:
            continue
        oh, ol = float(or_hi[d]), float(or_lo[d])
        if not (math.isfinite(oh) and math.isfinite(ol)) or oh <= ol:
            continue
        prior = dates[:i]
        # ATR from prior sessions only
        prior_tr = [tr_by_date[x] for x in prior[-atr_lb:] if x in tr_by_date]
        atr = float(np.mean(prior_tr)) if len(prior_tr) >= max(3, atr_lb // 3) else float("nan")
        # relative volume of this OR vs its own trailing average
        prior_orv = [float(or_vol[x]) for x in prior[-rvol_lb:] if x in or_vol.index]
        base = float(np.mean(prior_orv)) if len(prior_orv) >= 3 else float("nan")
        v = float(or_vol.get(d, float("nan")))
        rvol = (v / base) if (math.isfinite(base) and base > 0 and math.isfinite(v)) else float("nan")
        # prior-day levels + volume profile
        pdh = pdl = poc = vah = val = float("nan")
        if prior:
            pdate = prior[-1]
            pdh = float(day_hi.get(pdate, float("nan")))
            pdl = float(day_lo.get(pdate, float("nan")))
            pbars = rth[rth["_date"] == pdate]
            poc, vah, val = _bar_volume_profile(pbars, va_pct=float(p["va_pct"]))
        o_o, o_c = float(or_open.get(d, np.nan)), float(or_close.get(d, np.nan))
        if math.isfinite(o_o) and math.isfinite(o_c):
            or_dir = 1 if o_c > o_o else (-1 if o_c < o_o else 0)
        else:
            or_dir = 0
        gap = float("nan")
        if prior:
            pc = day_close.get(prior[-1], float("nan"))
            do = day_open.get(d, float("nan"))
            if math.isfinite(pc) and math.isfinite(do):
                gap = float(do) - float(pc)
        out[d] = SessionCtx(
            date=d, or_high=oh, or_low=ol, or_mid=(oh + ol) / 2.0,
            or_range=oh - ol, or_volume=v, rvol=rvol, atr=atr,
            pd_high=pdh, pd_low=pdl, pd_vah=vah, pd_val=val, pd_poc=poc,
            or_dir=or_dir, gap=gap,
        )
    return out


# ── position + account state ─────────────────────────────────────────────

@dataclass
class Position:
    side: int
    qty: int
    qty_open: int
    entry_px: float
    entry_i: int
    entry_t: pd.Timestamp
    stop: float
    t1: float
    t2: float
    risk_pts: float
    setup: str
    date: Any
    scaled: bool = False
    be_moved: bool = False
    realized_pts: float = 0.0     # contract-weighted points already banked
    mae: float = 0.0
    mfe: float = 0.0
    peak: float = 0.0             # best price seen, for trailing


@dataclass
class Account:
    balance: float
    floor: float
    day_pnl_usd: float = 0.0
    day_trades: int = 0
    day: Any = None
    halted: bool = False


# ── the simulator ────────────────────────────────────────────────────────

def simulate_orb(ohlc: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Run the ORB book bar-by-bar. Returns ledger + account diagnostics."""
    p = dict(params)
    pv = float(p.get("point_value") or DEFAULT_POINT_VALUE)
    df = _ct_frame(ohlc)
    if df.empty:
        return {"trades": pd.DataFrame(), "skipped": {"empty": 1}, "n_bars": 0,
                "account": {}, "params": p}

    ctxs = build_session_contexts(ohlc, p)
    rth_open = _hm_minutes(p["session_start"])
    or_end_m = rth_open + int(p["or_minutes"])
    trade_start_m = or_end_m
    entry_start_m = max(or_end_m, _hm_minutes(p["entry_start"])) if p.get("entry_start") else or_end_m
    entry_end_m = _hm_minutes(p["trade_end"])
    flat_m = _hm_minutes(p["session_end"])

    o = df["open"].to_numpy("float64")
    h = df["high"].to_numpy("float64")
    l = df["low"].to_numpy("float64")
    c = df["close"].to_numpy("float64")
    ct_min = df["_ct_min"].to_numpy()
    dates = df["_date"].to_numpy()
    idx = df.index

    acct = Account(
        balance=float(p["initial_balance"]),
        floor=float(p["initial_balance"]) - float(p["max_drawdown"]),
    )
    pos: Position | None = None
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    equity_pts: list[dict[str, Any]] = []
    # per-session state used by the failed-break / retest setups
    sess: dict[Any, dict[str, Any]] = {}
    day_setups: dict[Any, set] = {}

    def _new_state() -> dict[str, Any]:
        return {"broke_up": False, "broke_dn": False,
                "poke_high": -math.inf, "poke_low": math.inf,
                "max_above": 0.0, "max_below": 0.0}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def close_out(i: int, px: float, reason: str, qty: int) -> None:
        nonlocal pos
        assert pos is not None
        pts = (px - pos.entry_px) * pos.side
        pos.realized_pts += pts * qty
        pos.qty_open -= qty
        if pos.qty_open > 0:
            return
        gross_pts = pos.realized_pts
        cost_usd = float(p["cost_per_trade"]) * pv * pos.qty
        usd = gross_pts * pv - cost_usd
        acct.balance += usd
        acct.day_pnl_usd += usd
        rows.append({
            "session_date": pos.date,
            "entry_ts": pos.entry_t,
            "exit_ts": idx[i],
            "side": "long" if pos.side > 0 else "short",
            "qty": pos.qty,
            "entry_px": pos.entry_px,
            "exit_px": px,
            "points": gross_pts / max(1, pos.qty),
            "pnl_points": gross_pts,           # contract-weighted → × $20 = USD
            "pnl_usd": usd,
            "risk_pts": pos.risk_pts,
            "r_multiple": (gross_pts / max(1, pos.qty)) / pos.risk_pts if pos.risk_pts > 0 else 0.0,
            "mae_points": pos.mae,
            "mfe_points": pos.mfe,
            "holding_bars": i - pos.entry_i,
            "exit_reason": reason,
            "setup": pos.setup,
            "balance": acct.balance,
        })
        equity_pts.append({"t": idx[i], "balance": acct.balance})
        if acct.balance <= acct.floor:
            acct.halted = True
        pos = None

    for i in range(len(df)):
        d = dates[i]
        m = int(ct_min[i])
        if acct.halted:
            break
        # new session bookkeeping
        if acct.day != d:
            acct.day = d
            acct.day_pnl_usd = 0.0
            acct.day_trades = 0
            sess[d] = _new_state()
            day_setups[d] = set()
        ctx = ctxs.get(d)
        st = sess.setdefault(d, _new_state())

        # ── manage an open position ──────────────────────────────────
        if pos is not None:
            if m >= flat_m:
                close_out(i, c[i], "session_flat", pos.qty_open)
            else:
                hit = _manage(pos, o[i], h[i], l[i], p)
                for px, reason, qty in hit:
                    if pos is None:
                        break
                    q = pos.qty_open if qty <= 0 else min(qty, pos.qty_open)
                    if reason == "t1_scale":
                        pos.realized_pts += (px - pos.entry_px) * pos.side * q
                        pos.qty_open -= q
                        pos.scaled = True
                        if bool(p["breakeven_after_t1"]):
                            be = pos.entry_px + pos.side * float(p["be_offset_pts"])
                            pos.stop = max(pos.stop, be) if pos.side > 0 else min(pos.stop, be)
                            pos.be_moved = True
                        continue
                    close_out(i, px, reason, q)
                if pos is not None:
                    _track(pos, h[i], l[i])

        # ── look for a new entry ─────────────────────────────────────
        if ctx is None:
            continue
        # OR-violation bookkeeping runs on every in-session bar, even while
        # a position is open, so the failed-break setup sees the full poke.
        if m >= trade_start_m and m < flat_m:
            if h[i] > ctx.or_high:
                st["broke_up"] = True
                st["poke_high"] = max(st["poke_high"], h[i])
                st["max_above"] = max(st["max_above"], h[i] - ctx.or_high)
            if l[i] < ctx.or_low:
                st["broke_dn"] = True
                st["poke_low"] = min(st["poke_low"], l[i])
                st["max_below"] = max(st["max_below"], ctx.or_low - l[i])
        if pos is not None:
            continue
        if not (entry_start_m <= m < entry_end_m):
            continue
        if acct.day_trades >= int(p["max_trades_per_day"]):
            continue
        if acct.day_pnl_usd <= -abs(float(p["daily_loss_limit"])):
            continue

        sig = _entry_signal(ctx, i, o, h, l, c, p, st, day_setups[d])
        if sig is None:
            continue
        side, entry_px, stop_px, setup = sig
        risk_pts = abs(entry_px - stop_px)
        if risk_pts <= 0:
            skip("zero_risk")
            continue
        if risk_pts > float(p["max_stop_points"]):
            skip("stop_too_wide")
            continue
        qty = _size(risk_pts, acct, p)
        if qty < 1:
            skip("no_size")
            continue
        t1 = entry_px + side * risk_pts * float(p["t1_r"])
        t2 = entry_px + side * risk_pts * float(p["t2_r"])
        pos = Position(
            side=side, qty=qty, qty_open=qty, entry_px=entry_px, entry_i=i,
            entry_t=idx[i], stop=stop_px, t1=t1, t2=t2, risk_pts=risk_pts,
            setup=setup, date=d, peak=entry_px,
        )
        acct.day_trades += 1
        day_setups[d].add(setup)

    if pos is not None:
        close_out(len(df) - 1, c[-1], "eod", pos.qty_open)

    ledger = pd.DataFrame(rows)
    return {
        "trades": ledger,
        "skipped": skipped,
        "n_bars": int(len(df)),
        "account": _account_summary(ledger, acct, p),
        "params": p,
    }


def _size(risk_pts: float, acct: Account, p: dict[str, Any]) -> int:
    """Contracts so total risk ≈ ``risk_usd_per_trade`` and margin/floor hold."""
    budget = float(p["risk_usd_per_trade"])
    # never risk more than what is left above the floor
    headroom = max(0.0, acct.balance - acct.floor)
    budget = min(budget, headroom * float(p["headroom_frac"]))
    per_contract = risk_pts * float(p["point_value"])
    if per_contract <= 0:
        return 0
    qty = int(math.floor(budget / per_contract))
    qty = max(0, min(int(p["max_contracts"]), qty))
    # margin ceiling
    max_by_margin = int(math.floor(acct.balance / float(p["margin_per_contract"])))
    return max(0, min(qty, max_by_margin))


def _track(pos: Position, hi: float, lo: float) -> None:
    if not (math.isfinite(hi) and math.isfinite(lo)):
        return
    if pos.side > 0:
        pos.mae = max(pos.mae, pos.entry_px - lo)
        pos.mfe = max(pos.mfe, hi - pos.entry_px)
        pos.peak = max(pos.peak, hi)
    else:
        pos.mae = max(pos.mae, hi - pos.entry_px)
        pos.mfe = max(pos.mfe, pos.entry_px - lo)
        pos.peak = min(pos.peak, lo)


def _manage(
    pos: Position, o: float, hi: float, lo: float, p: dict[str, Any],
) -> list[tuple[float, str, int]]:
    """Bar-level fills for an open position. Stop wins on a same-bar tie."""
    if not (math.isfinite(hi) and math.isfinite(lo)):
        return []
    out: list[tuple[float, str, int]] = []
    long = pos.side > 0
    hit_stop = (lo <= pos.stop) if long else (hi >= pos.stop)
    if hit_stop:
        gap = (o <= pos.stop) if long else (o >= pos.stop)
        px = o if gap else pos.stop
        reason = "breakeven" if pos.be_moved else ("stop_after_scale" if pos.scaled else "stop")
        return [(px, reason, 0)]
    # trailing stop on the runner once scaled
    if pos.scaled and float(p["trail_pts"]) > 0:
        trail = pos.peak - pos.side * float(p["trail_pts"])
        if long:
            pos.stop = max(pos.stop, trail)
        else:
            pos.stop = min(pos.stop, trail)
    if not pos.scaled:
        hit_t1 = (hi >= pos.t1) if long else (lo <= pos.t1)
        if hit_t1:
            n_scale = int(round(pos.qty * float(p["scale_out_pct"])))
            n_scale = max(0, min(pos.qty - 1, n_scale))
            gap = (o >= pos.t1) if long else (o <= pos.t1)
            px = o if gap else pos.t1
            if n_scale >= 1:
                out.append((px, "t1_scale", n_scale))
            else:
                # single contract → T1 is the whole trade
                return [(px, "target_1", 0)]
    hit_t2 = (hi >= pos.t2) if long else (lo <= pos.t2)
    if hit_t2:
        gap = (o >= pos.t2) if long else (o <= pos.t2)
        out.append((o if gap else pos.t2, "target_2", 0))
    return out


def _entry_signal(
    ctx: SessionCtx, i: int,
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
    p: dict[str, Any], st: dict[str, Any], used: set,
) -> tuple[int, float, float, str] | None:
    """Return ``(side, entry_px, stop_px, setup)`` for this closed bar, or None.

    Every setup here is chosen to put the stop **close to the entry**: on a
    $5k account with a $1k floor, /NQ at $20/point only affords single-digit
    point risk. Wide "opposite end of the range" stops — standard in the
    literature — cannot be sized to even one contract, so the model instead
    enters at the moment structure invalidates within a few points.
    """
    px, hi, lo, op = c[i], h[i], l[i], o[i]
    if not (math.isfinite(px) and math.isfinite(hi) and math.isfinite(lo)):
        return None
    atr = ctx.atr
    if not math.isfinite(atr) or atr <= 0:
        return None
    rr = ctx.or_range / atr
    if rr < float(p["min_or_atr"]) or rr > float(p["max_or_atr"]):
        return None
    if math.isfinite(ctx.rvol) and ctx.rvol < float(p["min_rvol"]):
        return None

    buf = float(p["break_buffer_ticks"]) * TICK
    slip = float(p["slippage_points"])
    enabled = set(str(p["setups"]).split(","))
    bar_rng = hi - lo
    # where the bar closed inside its own range — a rejection tell
    clv = ((px - lo) / bar_rng) if bar_rng > 0 else 0.5
    rej = float(p["rejection_clv"])
    min_poke = float(p["min_poke_atr"]) * atr
    band = ctx.or_range * float(p["retest_band_frac"])

    # ── A. failed breakout — fade the trapped side ───────────────────────
    if "failed_break" in enabled and "failed_break" not in used:
        poke_up = st["max_above"] >= min_poke
        if poke_up and px < ctx.or_high - buf and clv <= (1.0 - rej):
            entry = _round_tick(px - slip)
            stop = _round_tick(st["poke_high"] + buf)
            if TICK <= stop - entry <= float(p["max_stop_points"]):
                return -1, entry, stop, "failed_break"
        poke_dn = st["max_below"] >= min_poke
        if poke_dn and px > ctx.or_low + buf and clv >= rej:
            entry = _round_tick(px + slip)
            stop = _round_tick(st["poke_low"] - buf)
            if TICK <= entry - stop <= float(p["max_stop_points"]):
                return 1, entry, stop, "failed_break"

    # ── B. breakout retest — continuation with an invalidation-tight stop ─
    if "breakout_retest" in enabled and "breakout_retest" not in used:
        if st["broke_up"] and px > ctx.or_high and lo <= ctx.or_high + band and clv >= rej:
            entry = _round_tick(px + slip)
            stop = _round_tick(min(lo, ctx.or_high) - buf)
            if TICK <= entry - stop <= float(p["max_stop_points"]):
                return 1, entry, stop, "breakout_retest"
        if st["broke_dn"] and px < ctx.or_low and hi >= ctx.or_low - band and clv <= (1.0 - rej):
            entry = _round_tick(px - slip)
            stop = _round_tick(max(hi, ctx.or_low) + buf)
            if TICK <= stop - entry <= float(p["max_stop_points"]):
                return -1, entry, stop, "breakout_retest"

    # ── C. inside-range fade — balance-day rotation ──────────────────────
    if "fade" in enabled and "fade" not in used:
        inside = ctx.or_low <= px <= ctx.or_high
        fband = ctx.or_range * float(p["fade_band_frac"])
        if inside and rr <= float(p["fade_max_or_atr"]):
            if lo <= ctx.or_low + fband and clv >= rej:
                entry = _round_tick(px + slip)
                stop = _round_tick(min(lo, ctx.or_low) - buf)
                if TICK <= entry - stop <= float(p["max_stop_points"]):
                    return 1, entry, stop, "fade"
            if hi >= ctx.or_high - fband and clv <= (1.0 - rej):
                entry = _round_tick(px - slip)
                stop = _round_tick(max(hi, ctx.or_high) + buf)
                if TICK <= stop - entry <= float(p["max_stop_points"]):
                    return -1, entry, stop, "fade"
    return None


def _account_summary(ledger: pd.DataFrame, acct: Account, p: dict[str, Any]) -> dict[str, Any]:
    init = float(p["initial_balance"])
    if ledger is None or ledger.empty:
        return {
            "initial_balance": init, "final_balance": acct.balance,
            "floor": acct.floor, "halted": acct.halted,
            "min_balance": acct.balance, "max_balance": acct.balance,
            "max_drawdown_usd": 0.0, "peak_to_trough_usd": 0.0,
            "breached": False, "return_pct": 0.0,
        }
    bal = ledger["balance"].to_numpy("float64")
    curve = np.concatenate([[init], bal])
    hwm = np.maximum.accumulate(curve)
    dd = curve - hwm
    return {
        "initial_balance": init,
        "final_balance": float(curve[-1]),
        "floor": acct.floor,
        "halted": bool(acct.halted),
        "min_balance": float(curve.min()),
        "max_balance": float(curve.max()),
        # static drawdown vs the starting balance — the prop-style barrier
        "max_drawdown_usd": float(max(0.0, init - curve.min())),
        # classic peak-to-trough
        "peak_to_trough_usd": float(-dd.min()) if len(dd) else 0.0,
        "breached": bool(curve.min() <= acct.floor),
        "return_pct": float((curve[-1] - init) / init * 100.0),
    }


# ── Backtest-tab payload ─────────────────────────────────────────────────

def _iso(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def orb_payload(
    res: dict[str, Any], *, source: str, start: str | None, end: str | None,
    n_sessions: int, card: dict[str, Any],
) -> dict[str, Any]:
    """Shape the ORB result like the other external-model backtests.

    ``pnl_points`` is **contract-weighted** (points × qty) so the Backtest
    tab's ``pnl_points × point_value`` dollar conversion stays correct with
    variable position size. Per-contract points are in ``points``.
    """
    ledger: pd.DataFrame = res["trades"]
    p = res["params"]
    pv = float(p.get("point_value") or DEFAULT_POINT_VALUE)
    acct = res["account"]
    title = card.get("title") or "Opening Range Breakout"
    # Mirrors the shared external-model config contract so the Backtest tab's
    # header, chart tab and blotter read it without special-casing.
    config: dict[str, Any] = {
        "strategy_id": "orb",
        "objective_id": None,
        "tau": 0.0,
        "confidence_gate": False,
        "sides": ["up", "down"],
        "max_concurrent_positions": 1,
        "target_points": float(p["t2_r"]),
        "stop_points": float(p["max_stop_points"]),
        "max_holding": f"flatten {p['session_end']}",
        "point_value": pv,
        "instrument": "NQ",
        "window": {
            "start": start,
            "end": end,
            "n_sessions": int(n_sessions),
            "n_bars": int(res["n_bars"]),
        },
        "prediction_mode": "external",
        "fill_path": "orb_1m",
        "execution": "orb_candle",
        "source_kind": "external",
        "source": source,
        "run_label": title,
        "model_path": source,
        "candidate_id": "orb",
        "model_family": "external_orb",
        "kind": "external_orb",
        "title": title,
        "timeframe": "1min",
        "bar_seconds": 60,
        "candle_table": "databento.tbbo_1min",
        "params": p,
        "book": None,
        "stack": [{
            "model": "orb", "hardened": None, "params": p,
            "timeframe": "1min", "title": title,
        }],
        "hardened": None,
        "account": acct,
        "live_data": False,
        "tbbo_data": False,
        "oco": False,
        "price_mode": "close",
        "weights_note": (
            f"ORB {p['or_minutes']}min opening range · setups={p['setups']} · "
            f"rvol≥{p['min_rvol']} · risk ${p['risk_usd_per_trade']}/trade on "
            f"≤{p['max_contracts']} contracts @ ${pv:g}/pt · "
            f"start ${acct.get('initial_balance')} floor ${acct.get('floor')} · "
            f"final ${acct.get('final_balance', 0):,.0f} "
            f"(max DD ${acct.get('max_drawdown_usd', 0):,.0f}"
            f"{', FLOOR BREACHED' if acct.get('breached') else ''})"
        ),
    }
    labels = ["down", "flat", "up"]
    empty_keys = ("gross_profit", "gross_loss", "net_pnl", "win_rate", "expectancy",
                  "profit_factor", "max_drawdown", "sharpe_daily", "avg_win", "avg_loss")
    if ledger is None or ledger.empty:
        return {
            "config": config,
            "summary": {"n_trades": 0, "n_wins": 0, "n_losses": 0,
                        "point_value": pv, "net_value": 0.0,
                        "by_exit_reason": {}, "n_sessions_traded": 0,
                        **{k: None for k in empty_keys}},
            "equity_curve": [],
            "confusion_matrix": {"labels": labels,
                                 "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
            "trades": [], "skipped": dict(res["skipped"]), "archetypes": [],
        }
    pnl = ledger["pnl_points"].to_numpy("float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    cum = np.cumsum(pnl)
    hwm = np.maximum.accumulate(np.maximum(cum, 0.0))
    dd = cum - hwm
    daily = ledger.groupby("session_date")["pnl_points"].sum().to_numpy("float64")

    def _sharpe(x: np.ndarray) -> float | None:
        if len(x) < 2:
            return None
        s = float(x.std(ddof=1))
        return float(x.mean() / s) if s else None

    summary = {
        "n_trades": int(len(ledger)),
        "n_wins": int((pnl > 0).sum()),
        "n_losses": int((pnl <= 0).sum()),
        "gross_profit": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss": float(losses.sum()) if len(losses) else 0.0,
        "net_pnl": float(pnl.sum()),
        "net_value": float(pnl.sum() * pv),
        "win_rate": float((pnl > 0).mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if len(losses) and float(losses.sum()) != 0.0 else None),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sharpe_daily": _sharpe(daily),
        "sharpe_trade": _sharpe(pnl),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_holding_bars": float(ledger["holding_bars"].mean()),
        "avg_mae": float(ledger["mae_points"].mean()),
        "avg_mfe": float(ledger["mfe_points"].mean()),
        "avg_r": float(ledger["r_multiple"].mean()),
        "avg_qty": float(ledger["qty"].mean()),
        "exposure_frac": float(ledger["holding_bars"].sum() / max(1, res["n_bars"])),
        "n_sessions_traded": int(ledger["session_date"].nunique()),
        "by_exit_reason": {str(k): int(v)
                           for k, v in ledger["exit_reason"].value_counts().items()},
        "by_setup": {str(k): int(v) for k, v in ledger["setup"].value_counts().items()},
        "point_value": pv,
        # account-level figures — the constraint that actually matters here
        "final_balance": acct.get("final_balance"),
        "max_drawdown_usd": acct.get("max_drawdown_usd"),
        "min_balance": acct.get("min_balance"),
        "breached_floor": acct.get("breached"),
        "return_pct": acct.get("return_pct"),
    }
    entry_ts = pd.to_datetime(ledger["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(ledger["exit_ts"], utc=True)
    equity_curve = [{"t": _iso(entry_ts.iloc[0]), "equity": 0.0}]
    equity_curve += [{"t": _iso(exit_ts.iloc[j]), "equity": float(cum[j])}
                     for j in range(len(ledger))]
    lab_ix = {l: i for i, l in enumerate(labels)}
    cm = [[0] * 3 for _ in range(3)]
    for side, pts in zip(ledger["side"].to_numpy(), pnl):
        pred = "up" if side == "long" else "down"
        true = pred if pts > 0 else ("down" if pred == "up" else "up")
        cm[lab_ix[true]][lab_ix[pred]] += 1
    trades_out = []
    for j, row in ledger.reset_index(drop=True).iterrows():
        trades_out.append({
            "entry_ts": _iso(row["entry_ts"]),
            "exit_ts": _iso(row["exit_ts"]),
            "side": row["side"],
            "qty": int(row["qty"]),
            "entry_px": float(row["entry_px"]),
            "exit_px": float(row["exit_px"]),
            "points": float(row["points"]),
            "pnl_points": float(row["pnl_points"]),
            "pnl_usd": float(row["pnl_usd"]),
            "r_multiple": float(row["r_multiple"]),
            "risk_pts": float(row["risk_pts"]),
            "balance": float(row["balance"]),
            "cum_pnl": float(cum[j]),
            "mae_points": float(row["mae_points"]),
            "mfe_points": float(row["mfe_points"]),
            "holding_bars": int(row["holding_bars"]),
            "exit_reason": str(row["exit_reason"]),
            "setup": str(row["setup"]),
            "n_present": None, "n_total": None,
        })
    return {
        "config": config, "summary": summary, "equity_curve": equity_curve,
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm},
        "trades": trades_out, "skipped": dict(res["skipped"]), "archetypes": [],
    }
