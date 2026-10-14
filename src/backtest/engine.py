"""Backtest engine — bar-by-bar evaluation of user-defined strategies on 5s features.

Executes a :class:`~src.backtest.spec.StrategySpec` over the feature parquet matrix
(``_close`` price passthrough, 5-second bars, exchange sessions) and produces:

* a **trade ledger** (entry/exit ts+px, side, PnL net of costs, MAE/MFE, holding, reason),
* **trade-economics KPIs** (win rate, expectancy, profit factor, Sharpe, max drawdown,
  exposure, per-side + per-exit-reason breakdowns, optional in/out-of-sample split),
* a **canonical prediction frame** ``{ts, direction, confidence}`` — one row per entry —
  so a strategy is scored on the *same* frozen evaluation contract as ML model candidates.

Execution model (documented conservatisms):

* Fills at **bar close** of the signal bar, +/- ``slippage_points`` per side.
* Target/stop levels are measured from the entry **fill** price; exits are evaluated on
  subsequent bar closes (5s bars — no intrabar assumption). When a bar's close satisfies
  both stop and target, the **stop wins** (conservative).
* Up to ``max_concurrent_positions`` open at once (default 1); optional cooldown bars
  between trades; optional per-day cap.
* ``eod_flat`` force-closes at the last bar of each session (no overnight risk); schedule
  mode may hold across days (``hold_days`` / ``next_entry``) with mark-to-market on closes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.rules import eval_rules
from src.backtest.spec import StrategySpec
from src.common.cme_holidays import is_trading_day

logger = logging.getLogger("vtech.backtest.engine")

_BAR_SECONDS = 5
_DOW = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
_CLOSE_FALLBACKS = ("cs_close", "close")


def ensure_price_close(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the ``_close`` passthrough the path engine reads.

    5s hive/day files already have ``_close``. Native 15s / 1m stores (ADR-075 /
    ADR-085) expose the candle as ``cs_close``.
    """
    if df is None or df.empty:
        return df
    if "_close" in df.columns:
        return df
    for alt in _CLOSE_FALLBACKS:
        if alt in df.columns:
            out = df.copy(deep=False)
            out["_close"] = pd.to_numeric(out[alt], errors="coerce")
            return out
    raise ValueError("backtest requires the '_close' price passthrough column")

# Same-bar exit priority (lower = wins). Conservative: stop beats target.
_REASON_PRIORITY = {"stop": 0, "target": 1, "rule": 2}

# Cap-reason preference when two exit caps land on the same bar index (lower = reported).
# An intentional cap (eod / settle / time) always beats the "ran out of data" sentinel.
_CAP_REASON_PREF = {"eod": 0, "settle": 0, "time": 1, "end_of_data": 5}


@dataclass
class BacktestResult:
    strategy_id: str
    trades: pd.DataFrame                       # the ledger (may be empty)
    kpis: dict[str, Any]
    canonical: pd.DataFrame                    # {ts, direction, confidence} per entry
    daily_pnl: pd.Series                       # net points per session date
    n_bars: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def drop_non_trading_sessions(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Drop bars whose local session date is a weekend or CME holiday/early-close."""
    if df.empty:
        return df
    local_dates = pd.Series(df.index.tz_convert(tz).normalize().date, index=df.index)
    keep = {d for d in local_dates.unique() if is_trading_day(d)}
    if len(keep) == local_dates.nunique():
        return df
    return df.loc[local_dates.isin(keep)]


def parse_duration_bars(s: str | None) -> int | None:
    """'10m' / '90s' / '2h' -> number of 5s bars (ceil, >=1). None -> None."""
    if not s:
        return None
    s = str(s).strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}.get(s[-1])
    secs = int(s[:-1]) * mult if mult else int(s)
    return max(1, int(np.ceil(secs / _BAR_SECONDS)))


def run_backtest(
    df: pd.DataFrame,
    spec: StrategySpec,
    *,
    objective: dict[str, Any] | None = None,
    session_tz: str | None = None,
) -> BacktestResult:
    """Run *spec* over *df* (feature matrix indexed by ts, with ``_close``)."""
    df = ensure_price_close(df)
    if df.empty:
        return _empty_result(spec)

    df = df.sort_index()
    # Mixed Etc/UTC + UTC concat yields an object Index of tz-aware datetimes.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    bt = spec.backtest
    # ── Date-range selection (the user interface: spec dates, runner may override) ──
    tzinfo = getattr(df.index, "tz", None)
    if bt.get("start"):
        df = df[df.index >= pd.Timestamp(str(bt["start"]), tz=tzinfo)]
    if bt.get("end"):
        df = df[df.index < pd.Timestamp(str(bt["end"]), tz=tzinfo) + pd.Timedelta(days=1)]
    if df.empty:
        return _empty_result(spec)

    tz = session_tz or ((objective or {}).get("session") or {}).get("timezone") or "America/Chicago"
    df = drop_non_trading_sessions(df, tz)
    if df.empty:
        return _empty_result(spec)
    tzinfo = getattr(df.index, "tz", None)
    idx_local = df.index.tz_convert(tz) if tzinfo is not None else df.index

    close = df["_close"].to_numpy(dtype="float64")
    n = len(df)
    ts = df.index.to_numpy()

    # ── Session (day) structure ───────────────────────────────────────────────
    day_codes, day_dates = pd.factorize(idx_local.normalize())
    day_last = np.zeros(len(day_dates), dtype=np.int64)     # last row idx per day
    day_first = np.zeros(len(day_dates), dtype=np.int64)
    for d in range(len(day_dates)):
        w = np.flatnonzero(day_codes == d)
        day_first[d], day_last[d] = w[0], w[-1]

    # ── Costs ─────────────────────────────────────────────────────────────────
    obj_costs = (objective or {}).get("costs") or {}
    cost_rt = float(spec.costs.get("cost_per_trade", obj_costs.get("cost_per_trade", 0.0)))
    slip = float(spec.costs.get("slippage_points", obj_costs.get("slippage_points", 0.0)))

    # ── Exit config ───────────────────────────────────────────────────────────
    ex = spec.exit or {}
    target = ex.get("target_points")
    stop = ex.get("stop_points")
    max_hold = parse_duration_bars(ex.get("max_holding"))
    eod_flat = bool(ex.get("eod_flat", True))
    exit_rule_masks: dict[str, np.ndarray | None] = {"long": None, "short": None}
    for side in ("long", "short"):
        blk = (ex.get("rules") or {}).get(side)
        if blk:
            exit_rule_masks[side] = eval_rules(df, blk, day_id=day_codes).to_numpy(bool)

    # ── Entry candidates ──────────────────────────────────────────────────────
    skipped = {"conflicting_signal": 0, "in_position": 0, "cooldown": 0,
               "last_bar": 0, "day_cap": 0, "outside_trading_window": 0}

    # Trading-window mask (execution window from the objective session block):
    # signals outside [trading_start, trading_end) local time are blocked, and
    # eod force-close is anchored to trading_end when eod_flat is on. This is
    # independent of the training/data window used by the feature engine.
    session_cfg = (objective or {}).get("session") or {}
    trading_start_str = str(session_cfg.get("trading_start") or "08:30")
    trading_end_str = str(session_cfg.get("trading_end") or "14:45")
    try:
        _tsh, _tsm = (int(x) for x in trading_start_str.split(":")[:2])
        _teh, _tem = (int(x) for x in trading_end_str.split(":")[:2])
        _local_min = idx_local.hour * 60 + idx_local.minute
        _in_window = (_local_min >= _tsh * 60 + _tsm) & (_local_min < _teh * 60 + _tem)
        trading_window_mask = np.asarray(_in_window, dtype=bool)
    except Exception:  # noqa: BLE001 — malformed HH:MM: skip the filter, don't crash
        trading_window_mask = np.ones(n, dtype=bool)

    # Per-day trading-window end bar (last row idx with local time < trading_end),
    # used to cap positions to the objective's execution window.
    day_trading_last = np.copy(day_last)
    for d in range(len(day_dates)):
        w = np.flatnonzero((day_codes == d) & trading_window_mask)
        if w.size:
            day_trading_last[d] = int(w[-1])

    settle_mode, settle_bars, settle_days = None, None, None
    if spec.mode == "rules":
        filt = (eval_rules(df, spec.filters, day_id=day_codes).to_numpy(bool)
                if spec.filters else np.ones(n, dtype=bool))
        filt &= trading_window_mask
        m_long = (eval_rules(df, (spec.entry or {}).get("long"), day_id=day_codes).to_numpy(bool)
                  if (spec.entry or {}).get("long") else np.zeros(n, dtype=bool)) & filt
        m_short = (eval_rules(df, (spec.entry or {}).get("short"), day_id=day_codes).to_numpy(bool)
                   if (spec.entry or {}).get("short") else np.zeros(n, dtype=bool)) & filt
        both = m_long & m_short
        skipped["conflicting_signal"] = int(both.sum())
        m_long &= ~both
        m_short &= ~both
        cand_idx = np.flatnonzero(m_long | m_short)
        cand_side = np.where(m_long[cand_idx], 1, -1)       # +1 long, -1 short
    elif spec.mode == "predictions":
        # Model-prediction entries: gate the OOF frame by confidence, then trade the
        # predicted direction. The runner joins '_pred_direction'/'_pred_confidence'
        # onto df from the OOF parquet. Everything downstream (one-position-at-a-time,
        # cooldown, day-cap, stop/target/time exits, ledger, KPIs) is identical to
        # rule mode — only the entry-signal source differs.
        pcfg = spec.predictions or {}
        tau = float(pcfg.get("tau", 0.8))
        gate_on = bool(pcfg.get("confidence_gate", True))
        sides = set(pcfg.get("sides") or ("up", "down"))
        if "_pred_confidence" not in df.columns or "_pred_direction" not in df.columns:
            raise ValueError(
                "predictions mode requires '_pred_confidence' and '_pred_direction' "
                "columns on the frame (the runner joins them from the OOF parquet)")
        conf = df["_pred_confidence"].to_numpy(dtype="float64")
        pdir = df["_pred_direction"].to_numpy()
        filt = (eval_rules(df, spec.filters, day_id=day_codes).to_numpy(bool)
                if spec.filters else np.ones(n, dtype=bool))
        filt &= trading_window_mask
        gate = filt if not gate_on else ((conf >= tau) & filt)
        m_long = (gate & (pdir == "up")) if "up" in sides else np.zeros(n, dtype=bool)
        m_short = (gate & (pdir == "down")) if "down" in sides else np.zeros(n, dtype=bool)
        cand_idx = np.flatnonzero(m_long | m_short)
        cand_side = np.where(m_long[cand_idx], 1, -1)       # +1 long, -1 short
    else:
        cand_idx, cand_side, settle = _schedule_entries(
            spec.schedule or {}, idx_local, day_codes, day_first, day_last, df, spec.filters)
        settle_mode = settle.get("mode", "eod")
        settle_bars = settle.get("bars")
        settle_days = settle.get("days")

    # ── Sequential trade loop (vectorised exit scan per trade) ────────────────
    # Up to ``max_concurrent_positions`` may be open at once (default 1 = the classic
    # one-at-a-time book). A candidate is skipped as "in_position" only when all slots
    # are full at its bar; each position's exit is scanned independently of the others.
    cooldown = int(bt.get("cooldown_bars", 0))
    day_cap = int(bt.get("max_trades_per_day", 0))
    max_concurrent = max(1, int(bt.get("max_concurrent_positions", 1)))
    trades: list[dict[str, Any]] = []
    open_exits: list[int] = []                    # exit_idx of positions still open
    cooldown_until = -1
    trades_by_day: dict[int, int] = {}

    for k, i in enumerate(cand_idx):
        i = int(i)
        if not trading_window_mask[i]:
            skipped["outside_trading_window"] += 1
            continue
        if open_exits:                            # free slots whose position has closed
            open_exits = [e for e in open_exits if e >= i]
        if len(open_exits) >= max_concurrent:
            skipped["in_position"] += 1
            continue
        if i < cooldown_until:
            skipped["cooldown"] += 1
            continue
        d = int(day_codes[i])
        if day_cap and trades_by_day.get(d, 0) >= day_cap:
            skipped["day_cap"] += 1
            continue
        if i >= n - 1:
            skipped["last_bar"] += 1
            continue

        direction = int(cand_side[k])
        side = "long" if direction == 1 else "short"
        entry_px = close[i] + slip * direction              # pay slippage into the trade

        # Exit window cap: (last bar the position may live to, reason). Min idx wins.
        cap_candidates: list[tuple[int, str]] = [(n - 1, "end_of_data")]
        if spec.mode in ("rules", "predictions"):
            if eod_flat:
                # eod is the trading-window end (objective's trading_end), not the
                # last data bar — positions get flattened at the execution close.
                cap_candidates.append((int(day_trading_last[d]), "eod"))
        else:
            if settle_mode == "eod":
                cap_candidates.append((int(day_last[d]), "settle"))
            elif settle_mode == "hold_bars":
                cap_candidates.append((i + int(settle_bars), "settle"))
            elif settle_mode == "hold_days":
                dd = min(d + int(settle_days), len(day_dates) - 1)
                cap_candidates.append((int(day_last[dd]), "settle"))
            elif settle_mode == "next_entry":
                # Exit on the bar BEFORE the next scheduled entry so the re-entry bar
                # stays free (rebalance semantics: close, then re-open on schedule).
                nxt = int(cand_idx[k + 1]) if k + 1 < len(cand_idx) else n
                cap_candidates.append((max(i + 1, nxt - 1), "settle"))
        if max_hold is not None:
            cap_candidates.append((i + max_hold, "time"))
        cap, cap_reason = min(cap_candidates, key=lambda c: (c[0], _CAP_REASON_PREF.get(c[1], 9)))
        cap = min(cap, n - 1)
        if cap <= i:                                        # e.g. entry on a session's last bar
            skipped["last_bar"] += 1
            continue

        w = close[i + 1: cap + 1]                            # exit scan window
        exit_idx, reason = cap, cap_reason
        hits: list[tuple[int, str]] = []
        if stop is not None:
            adverse = (entry_px - w) * direction             # points against the position
            hs = np.flatnonzero(adverse >= stop)
            if hs.size:
                hits.append((i + 1 + int(hs[0]), "stop"))
        if target is not None:
            favorable = (w - entry_px) * direction
            ht = np.flatnonzero(favorable >= target)
            if ht.size:
                hits.append((i + 1 + int(ht[0]), "target"))
        erm = exit_rule_masks[side]
        if erm is not None:
            hr = np.flatnonzero(erm[i + 1: cap + 1])
            if hr.size:
                hits.append((i + 1 + int(hr[0]), "rule"))
        if hits:
            hits.sort(key=lambda h: (h[0], _REASON_PRIORITY.get(h[1], 9)))
            if hits[0][0] <= cap:
                exit_idx, reason = hits[0]

        exit_px = close[exit_idx] - slip * direction         # pay slippage out
        pnl = (exit_px - entry_px) * direction - cost_rt
        path = close[i + 1: exit_idx + 1]
        mae = float(np.max((entry_px - path) * direction, initial=0.0))
        mfe = float(np.max((path - entry_px) * direction, initial=0.0))

        trades.append({
            "entry_ts": ts[i], "exit_ts": ts[exit_idx], "side": side,
            "entry_px": float(entry_px), "exit_px": float(exit_px),
            "pnl_points": float(pnl), "mae_points": mae, "mfe_points": mfe,
            "holding_bars": int(exit_idx - i), "exit_reason": reason,
            "entry_idx": i, "exit_idx": int(exit_idx),
            "session_date": str(pd.Timestamp(day_dates[d]).date()),
        })
        trades_by_day[d] = trades_by_day.get(d, 0) + 1
        open_exits.append(exit_idx)
        if cooldown:                              # 0 = no cooldown (never blocks other slots)
            cooldown_until = max(cooldown_until, exit_idx + 1 + cooldown)

    ledger = pd.DataFrame(trades)
    point_value = float(bt.get("point_value", 1.0))
    kpis = _compute_kpis(ledger, point_value, n_bars=n)
    if bt.get("holdout_start") is not None and not ledger.empty:
        cut = pd.Timestamp(str(bt["holdout_start"]), tz=tzinfo)
        kpis["in_sample"] = _compute_kpis(ledger[ledger["entry_ts"] < cut], point_value, n_bars=n)
        kpis["out_of_sample"] = _compute_kpis(ledger[ledger["entry_ts"] >= cut], point_value, n_bars=n)
        kpis["holdout_start"] = str(bt["holdout_start"])
    kpis["skipped"] = skipped

    canonical = (pd.DataFrame({
        "ts": ledger["entry_ts"],
        "direction": ledger["side"].map({"long": "up", "short": "down"}),
        "confidence": 1.0,
    }) if not ledger.empty else pd.DataFrame(columns=["ts", "direction", "confidence"]))

    daily = (ledger.groupby("session_date")["pnl_points"].sum()
             if not ledger.empty else pd.Series(dtype="float64"))

    return BacktestResult(strategy_id=spec.strategy_id, trades=ledger, kpis=kpis,
                          canonical=canonical, daily_pnl=daily, n_bars=n, skipped=skipped)


# ── Schedule (calendar-cadence) entries ──────────────────────────────────────

def _schedule_entries(
    sched: dict[str, Any],
    idx_local: pd.DatetimeIndex,
    day_codes: np.ndarray,
    day_first: np.ndarray,
    day_last: np.ndarray,
    df: pd.DataFrame,
    filters: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve daily/weekly/monthly scheduled entry bars.

    Returns ``(entry_bar_indices, sides, settle_config)``. Missing scheduled days
    (holidays) roll forward to the next available trading day within the period.
    """
    freq = sched["frequency"]
    side = 1 if sched.get("side", "long") == "long" else -1
    settle = dict(sched.get("settle") or {"mode": "eod"})

    dates = pd.DatetimeIndex(pd.unique(idx_local.normalize()))
    chosen_days: list[int] = []                            # day ordinals (codes)

    if freq == "daily":
        chosen_days = list(range(len(dates)))
    elif freq == "weekly":
        want = _DOW[sched["day_of_week"]]
        iso = pd.Series(range(len(dates)), index=dates)
        for _, grp in iso.groupby([dates.isocalendar().year, dates.isocalendar().week]):
            days_in_week = grp.index
            on_or_after = [d for d in days_in_week if d.dayofweek >= want]
            pick = None
            exact = [d for d in days_in_week if d.dayofweek == want]
            if exact:
                pick = exact[0]
            elif on_or_after:
                pick = on_or_after[0]                       # holiday roll-forward
            if pick is not None:
                chosen_days.append(int(grp.loc[pick]))
    else:                                                    # monthly
        dom = sched.get("day_of_month", "first")
        ser = pd.Series(range(len(dates)), index=dates)
        for _, grp in ser.groupby([dates.year, dates.month]):
            days_in_month = grp.index
            if dom == "first":
                pick = days_in_month[0]
            elif dom == "last":
                pick = days_in_month[-1]
            else:
                onward = [d for d in days_in_month if d.day >= int(dom)]
                pick = onward[0] if onward else None
            if pick is not None:
                chosen_days.append(int(grp.loc[pick]))

    # Entry bar within each chosen day: first bar at/after entry_time (default: first bar)
    entry_time = sched.get("entry_time")
    filt = (eval_rules(df, filters, day_id=day_codes).to_numpy(bool)
            if filters else None)
    out_idx: list[int] = []
    for d in sorted(set(chosen_days)):
        lo, hi = int(day_first[d]), int(day_last[d])
        pos = lo
        if entry_time:
            hh, mm = (int(x) for x in entry_time.split(":"))
            times = idx_local[lo:hi + 1]
            mask = (times.hour > hh) | ((times.hour == hh) & (times.minute >= mm))
            w = np.flatnonzero(mask)
            if w.size == 0:
                continue                                     # entry_time after session end
            pos = lo + int(w[0])
        if filt is not None and not filt[pos]:
            continue                                         # filtered out that day
        out_idx.append(pos)

    arr = np.asarray(sorted(out_idx), dtype=np.int64)
    return arr, np.full(len(arr), side, dtype=np.int64), settle


# ── KPIs ──────────────────────────────────────────────────────────────────────

def _compute_kpis(ledger: pd.DataFrame, point_value: float, *, n_bars: int) -> dict[str, Any]:
    """Trade-economics KPI suite over a (possibly empty) ledger."""
    if ledger is None or ledger.empty:
        return {"n_trades": 0, "win_rate": None, "expectancy_points": None,
                "profit_factor": None, "total_points": 0.0, "total_value": 0.0,
                "max_drawdown_points": 0.0, "sharpe_trade": None, "sharpe_daily": None,
                "avg_win_points": None, "avg_loss_points": None,
                "avg_holding_bars": None, "exposure_frac": 0.0,
                "by_side": {}, "by_exit_reason": {}}

    pnl = ledger["pnl_points"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = pnl.cumsum()
    # High-water mark seeded at 0 (pre-trade equity): a leading losing trade counts as
    # drawdown — any dip below the running peak, including the starting 0, contributes.
    # Same definition as src.evaluation._kpi.max_drawdown.
    dd = equity - equity.cummax().clip(lower=0.0)
    daily = ledger.groupby("session_date")["pnl_points"].sum()

    def _sharpe(x: pd.Series) -> float | None:
        if len(x) < 2 or float(x.std(ddof=1)) == 0.0:
            return None
        return float(x.mean() / x.std(ddof=1))

    by_side = {
        s: {"n": int(len(g)), "win_rate": float((g["pnl_points"] > 0).mean()),
            "total_points": float(g["pnl_points"].sum())}
        for s, g in ledger.groupby("side")
    }
    return {
        "n_trades":             int(len(ledger)),
        "win_rate":             float((pnl > 0).mean()),
        "expectancy_points":    float(pnl.mean()),
        "profit_factor":        (float(wins.sum() / abs(losses.sum()))
                                 if len(losses) and float(losses.sum()) != 0.0 else None),
        "total_points":         float(pnl.sum()),
        "total_value":          float(pnl.sum() * point_value),
        "avg_win_points":       float(wins.mean()) if len(wins) else None,
        "avg_loss_points":      float(losses.mean()) if len(losses) else None,
        "max_drawdown_points":  float(dd.min()),
        "sharpe_trade":         _sharpe(pnl),
        "sharpe_daily":         _sharpe(daily),
        "avg_holding_bars":     float(ledger["holding_bars"].mean()),
        "avg_mae_points":       float(ledger["mae_points"].mean()),
        "avg_mfe_points":       float(ledger["mfe_points"].mean()),
        "exposure_frac":        float(ledger["holding_bars"].sum() / max(1, n_bars)),
        "n_sessions_traded":    int(ledger["session_date"].nunique()),
        "by_side":              by_side,
        "by_exit_reason":       {k: int(v) for k, v in
                                 ledger["exit_reason"].value_counts().items()},
    }


def _empty_result(spec: StrategySpec) -> BacktestResult:
    return BacktestResult(
        strategy_id=spec.strategy_id, trades=pd.DataFrame(),
        kpis=_compute_kpis(pd.DataFrame(), 1.0, n_bars=0),
        canonical=pd.DataFrame(columns=["ts", "direction", "confidence"]),
        daily_pnl=pd.Series(dtype="float64"), n_bars=0)
