"""Registered Model page backend (per-objective).

Serves the *Registered Model* sub-page of an objective:

* **status** — is the objective's registered model (registry LATEST, else
  newest staged bundle) *actively generating signals* right now (trading day
  + inside the session window + live tick feed)?  Includes a market-holiday
  notification for the current Chicago calendar day.
* **settings** — operator-tunable execution parameters persisted per
  objective under ``webui/regmodel/<objective_id>.json``: stop/target order
  type (MKT | LIMIT), execution style (OCO | ONCLOSE), confidence gate τ,
  max open positions, and a **today-only** session window override (stored
  with its session date so it silently resets the next trading day).
* **blotter** — the actual signals emitted by the live serving loop and
  appended to ``data/live/signals.jsonl`` for the current session day.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.common.data_paths import data_root, live_dir, webui_dir
from src.webui import live

logger = logging.getLogger("vtech.webui.regmodel")

_HHMM_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")

ORDER_TYPES = ("MKT", "LIMIT")
EXECUTIONS = ("OCO", "ONCLOSE")

_DEFAULTS: dict[str, Any] = {
    "order_type": "MKT",
    "execution": "OCO",
    "tau": None,              # None → objective primary τ
    "max_positions": 1,
    "session_override": None,  # {"date","trading_start","trading_end"} — today only
}

_LOCK = threading.Lock()
_BUNDLES: dict[str, dict[str, Any]] = {}          # model path → serving bundle
_TICKS: dict[str, dict[str, Any]] = {}            # day → tick arrays (incremental)
_BLOTTER: dict[str, dict[str, Any]] = {}          # objective → cached payload


# ── settings ──────────────────────────────────────────────────────────────────

def _settings_path(objective_id: str) -> Path:
    return webui_dir() / "regmodel" / f"{objective_id}.json"


def get_settings(objective_id: str) -> dict[str, Any]:
    """Persisted settings merged over defaults (raw — override not expired)."""
    out = dict(_DEFAULTS)
    p = _settings_path(objective_id)
    try:
        if p.exists():
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                out.update({k: d[k] for k in _DEFAULTS if k in d})
    except Exception as exc:  # noqa: BLE001
        logger.warning("regmodel_settings_read_failed oid=%s err=%s", objective_id, exc)
    return out


def save_settings(objective_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist settings; session times are stamped with today's
    session date so they auto-reset the next trading day."""
    cur = get_settings(objective_id)

    ot = updates.get("order_type")
    if ot is not None:
        ot = str(ot).upper()
        if ot not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {ORDER_TYPES}")
        cur["order_type"] = ot
    ex = updates.get("execution")
    if ex is not None:
        ex = str(ex).upper().replace("_", "").replace("-", "")
        if ex not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}")
        cur["execution"] = ex
    if "tau" in updates and updates["tau"] is not None:
        tau = float(updates["tau"])
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        cur["tau"] = round(tau, 4)
    if updates.get("max_positions") is not None:
        mp = int(updates["max_positions"])
        if not 1 <= mp <= 50:
            raise ValueError("max_positions must be in [1, 50]")
        cur["max_positions"] = mp

    ts_start = updates.get("trading_start")
    ts_end = updates.get("trading_end")
    if ts_start is not None or ts_end is not None:
        trade = live.objective_trade_params(objective_id)
        ctx = live.session_context(objective_id)
        start = str(ts_start or trade["trading_start"]).strip()
        end = str(ts_end or trade["trading_end"]).strip()
        for v in (start, end):
            if not _HHMM_RE.match(v):
                raise ValueError(f"session time must be HH:MM, got {v!r}")
        if start >= end:
            raise ValueError("trading_start must be before trading_end")
        if start == trade["trading_start"] and end == trade["trading_end"]:
            cur["session_override"] = None            # back to model defaults
        else:
            cur["session_override"] = {
                "date": str(ctx["session_date"]),
                "trading_start": start,
                "trading_end": end,
            }

    p = _settings_path(objective_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {**cur, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    tmp.replace(p)
    logger.info("regmodel_settings_saved oid=%s %s", objective_id, cur)
    return cur


def effective_settings(objective_id: str,
                       trade: dict[str, Any] | None = None,
                       ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """Settings with objective defaults filled in and stale overrides expired."""
    trade = trade or live.objective_trade_params(objective_id)
    ctx = ctx or live.session_context(objective_id)
    s = get_settings(objective_id)
    ov = s.get("session_override")
    override_active = bool(ov and str(ov.get("date")) == str(ctx["session_date"]))
    return {
        "order_type": s["order_type"],
        "execution": s["execution"],
        "tau": float(s["tau"]) if s.get("tau") is not None else float(trade["tau"]),
        "tau_is_default": s.get("tau") is None,
        "max_positions": int(s.get("max_positions") or 1),
        "trading_start": ov["trading_start"] if override_active else trade["trading_start"],
        "trading_end": ov["trading_end"] if override_active else trade["trading_end"],
        "session_override_active": override_active,
        "default_trading_start": trade["trading_start"],
        "default_trading_end": trade["trading_end"],
    }


# ── model resolution + status ─────────────────────────────────────────────────

def resolve_registered_model(objective_id: str) -> dict[str, Any] | None:
    """Registry LATEST for the objective, else newest registry/staging bundle."""
    try:
        from src.serving.promote import resolve_latest_bundle
        latest = resolve_latest_bundle(objective_id)
    except Exception:  # noqa: BLE001
        latest = None
    bundles = live.list_serving_bundles(objective_id)
    if latest is not None:
        latest = latest.resolve()
        for b in bundles:
            if Path(b["path"]).resolve() == latest:
                return {**b, "registered": "registry_latest"}
        return {"path": str(latest), "name": latest.name, "source": "registry",
                "objective_id": objective_id, "candidate_id": latest.name,
                "version": latest.name, "label": f"{latest.name} · registry LATEST",
                "registered": "registry_latest"}
    for src in ("registry", "staging"):
        for b in bundles:
            if b["source"] == src:
                return {**b, "registered": src}
    return None


def _hhmm_minutes(s: str) -> int:
    h, m = str(s).split(":")[:2]
    return int(h) * 60 + int(m)


def status(objective_id: str) -> dict[str, Any]:
    """Active indicator + settings + market-holiday notification."""
    trade = live.objective_trade_params(objective_id)
    ctx = live.session_context(objective_id)
    eff = effective_settings(objective_id, trade, ctx)
    model = resolve_registered_model(objective_id)
    health = live.streamer_health()

    now_local = datetime.fromisoformat(ctx["now_local"])
    today = now_local.date()
    try:
        from src.common.cme_holidays import is_trading_day
        trading_day = is_trading_day(today)
    except Exception:  # noqa: BLE001
        trading_day = today.weekday() < 5
    market_closed_reason = None
    if not trading_day:
        market_closed_reason = ("weekend" if today.weekday() >= 5
                                else "CME holiday")

    now_m = now_local.hour * 60 + now_local.minute
    in_session = (trading_day
                  and str(ctx["session_date"]) == str(today)
                  and _hhmm_minutes(eff["trading_start"]) <= now_m
                  <= _hhmm_minutes(eff["trading_end"]))
    feed_live = bool(health.get("live"))
    active = bool(model and trading_day and in_session and feed_live)

    return {
        "objective_id": objective_id,
        "model": model,
        "has_model": model is not None,
        "active": active,
        "active_checks": {
            "registered_model": model is not None,
            "trading_day": trading_day,
            "in_session": in_session,
            "feed_live": feed_live,
        },
        "market_closed_today": not trading_day,
        "market_closed_reason": market_closed_reason,
        "session": {
            "date": str(ctx["session_date"]),
            "timezone": ctx["timezone"],
            "trading_start": eff["trading_start"],
            "trading_end": eff["trading_end"],
            "now_local": ctx["now_local"],
        },
        "settings": get_settings(objective_id),
        "effective": eff,
        "trade": trade,
        "streamer": {"live": feed_live,
                     "tbbo_age_s": health.get("tbbo_age_s"),
                     "error": health.get("error")},
    }


# ── day inputs: features, bundle, ticks ───────────────────────────────────────

def _feature_day(objective_id: str) -> dict[str, Any]:
    """Session-day feature frame from disk (the live loop keeps it fresh)."""
    ctx = live.session_context(objective_id)
    day = ctx["session_date"]
    df, path = live._load_feature_frame(day)
    note = None
    if df.empty:
        d = day
        for _ in range(10):
            d = live._previous_trading_day(d)
            df, path = live._load_feature_frame(d)
            if not df.empty:
                day = d
                note = f"no features for session day — showing {day}"
                break
    return {"df": df, "path": str(path), "session_date": day, "context": ctx,
            "note": note}


def _bundle_for(path: str) -> dict[str, Any]:
    with _LOCK:
        b = _BUNDLES.get(path)
    if b is not None:
        return b
    from src.serving import load_serving_bundle
    b = load_serving_bundle(Path(path), validate=False)
    with _LOCK:
        _BUNDLES.clear()          # keep at most one loaded per process
        _BUNDLES[path] = b
    return b


def _load_ticks(day: str) -> dict[str, Any] | None:
    """Day TBBO prints as numpy arrays; incremental refresh on repeat calls."""
    import numpy as np
    import psycopg
    from src.common.db import get_dsn

    with _LOCK:
        cache = _TICKS.get(day)
    since_ns = int(cache["ts_ns"][-1]) if cache is not None and cache["n"] else None
    try:
        from src.common.contract_calendar import active_contract
        from datetime import date as _date
        symbol = active_contract("NQ", _date.fromisoformat(str(day)[:10]))
    except Exception:  # noqa: BLE001
        symbol = None

    sql = ("SELECT ts_event, price, bid_px, ask_px FROM databento.tbbo "
           "WHERE ts_event >= %s::date AND ts_event < (%s::date + interval '1 day') "
           "AND price IS NOT NULL AND price > 0")
    params: list[Any] = [str(day), str(day)]
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)
    if since_ns is not None:
        sql += " AND ts_event > %s"
        params.append(datetime.fromtimestamp(since_ns / 1e9, tz=timezone.utc))
    sql += " ORDER BY ts_event ASC"

    try:
        with psycopg.connect(get_dsn(), connect_timeout=4) as conn:
            with conn.cursor(name="regmodel_tbbo") as cur:
                cur.itersize = 10000
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("regmodel_ticks_failed day=%s err=%s", day, exc)
        return cache

    if rows:
        ts = np.array([r[0].replace(tzinfo=timezone.utc).timestamp() * 1e9
                       if r[0].tzinfo is None else r[0].timestamp() * 1e9
                       for r in rows], dtype="int64")
        px = np.array([float(r[1]) for r in rows], dtype="float64")
        bid = np.array([float(r[2]) if r[2] is not None else np.nan for r in rows])
        ask = np.array([float(r[3]) if r[3] is not None else np.nan for r in rows])
        mid = np.where(np.isfinite(bid) & np.isfinite(ask), (bid + ask) / 2.0, px)
        if cache is not None and cache["n"]:
            ts = np.concatenate([cache["ts_ns"], ts])
            px = np.concatenate([cache["px"], px])
            mid = np.concatenate([cache["mid"], mid])
            bid = np.concatenate([cache["bid"], bid])
            ask = np.concatenate([cache["ask"], ask])
        cache = {"ts_ns": ts, "px": px, "mid": mid, "bid": bid, "ask": ask,
                 "n": int(len(ts))}
        with _LOCK:
            _TICKS.clear()        # single-day cache
            _TICKS[day] = cache
    return cache


# ── blotter simulation ────────────────────────────────────────────────────────

def _session_bounds_ns(ctx: dict[str, Any], eff: dict[str, Any]) -> tuple[int, int]:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(ctx["timezone"])
    day = ctx["session_date"]
    if isinstance(day, str):
        from datetime import date as _date
        day = _date.fromisoformat(day)
    sh, sm = (int(x) for x in eff["trading_start"].split(":")[:2])
    eh, em = (int(x) for x in eff["trading_end"].split(":")[:2])
    start = datetime(day.year, day.month, day.day, sh, sm, tzinfo=tz)
    end = datetime(day.year, day.month, day.day, eh, em, tzinfo=tz)
    return int(start.timestamp() * 1e9), int(end.timestamp() * 1e9)


def _simulate(df, preds, n_present, *, ticks, eff: dict[str, Any],
              trade: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Signal rows for the day.

    Timeline per gated bar (bar labels are bar-start; the close exists at
    ``ts + 5s``), everything at exact ms resolution:

    * ``bar_ts``    — end of the 5s indicator bar.
    * ``signal_ts`` — bar close + ``signal_latency_ms`` (pipeline: live
      ingest → feature rebuild → scoring).  ``signal_px`` is the prevailing
      TBBO **mid** as of that instant.
    * ``entry_ts``  — signal + ``entry_latency_ms`` (order transmission).
      MKT entries fill at the prevailing **ask** (long) / **bid** (short);
      LIMIT entries rest at the signal mid and fill on first touch (or
      report ``entry_unfilled``).

    The profit target and MAE stop are anchored at the **indicator price**
    (the 5s close the model was scored on — the label definition); PnL is
    measured from the actual entry fill.  OCO exit brackets rest
    broker-side (no extra latency; MKT legs fill at the touch-tick bid/ask,
    LIMIT legs at the barrier price); ONCLOSE exit orders re-run the
    pipeline (triggering close + signal + entry latency).
    """
    import numpy as np

    BAR_NS = 5_000_000_000                            # 5s feature cadence

    tgt_pts = float(trade["move_threshold"])
    stop_pts = float(trade["mae_tolerance"])
    horizon_ns = int(trade["horizon_s"]) * 1_000_000_000
    tau = float(eff["tau"])
    max_pos = int(eff["max_positions"])
    order_type = eff["order_type"]
    execution = eff["execution"]
    lat_ns = max(0, int(eff.get("signal_latency_ms") or 0)) * 1_000_000
    ent_ns = max(0, int(eff.get("entry_latency_ms") or 0)) * 1_000_000
    n_total = int(trade.get("n_features") or 0)

    ts_ns = df.index.asi8
    close = df["_close"].astype("float64").ffill().to_numpy() if "_close" in df.columns \
        else np.full(len(df), np.nan)
    directions = preds["direction"].to_numpy()
    confidences = preds["confidence"].to_numpy(dtype="float64")
    start_ns, end_ns = _session_bounds_ns(ctx, eff)
    edge_ns = int(ts_ns[-1]) + BAR_NS if len(ts_ns) else 0    # last known close time

    t_ns = ticks["ts_ns"] if ticks is not None else None
    t_px = ticks["px"] if ticks is not None else None
    t_mid = ticks["mid"] if ticks is not None else None
    t_bid = ticks["bid"] if ticks is not None else None
    t_ask = ticks["ask"] if ticks is not None else None
    tick_edge_ns = int(t_ns[-1]) if t_ns is not None and len(t_ns) else 0

    def asof(ns: int) -> int | None:
        """Tick index prevailing at *ns* (last tick ≤ ns); None before the
        first tick or past the live feed edge."""
        if t_ns is None or not len(t_ns) or ns > tick_edge_ns:
            return None
        j = int(np.searchsorted(t_ns, ns, side="right")) - 1
        return j if j >= 0 else None

    def quote(j: int, side: float) -> float:
        """Marketable price at tick *j*: ask for a buy (side>0), bid for a
        sell; falls back to the mid when the book side is missing."""
        px = t_ask[j] if side > 0 else t_bid[j]
        return float(px) if np.isfinite(px) else float(t_mid[j])

    def _tick_exit(je: int, sign: float, tgt: float, stp: float,
                   deadline_ns: int) -> tuple[str, float, int] | None:
        """OCO first-touch on prints from index *je*. (reason, px, ts) or None."""
        jd = int(np.searchsorted(t_ns, min(deadline_ns, end_ns), side="right"))
        seg = t_px[je:jd]
        if len(seg):
            hit_t = np.flatnonzero(sign * (seg - tgt) >= 0)
            hit_s = np.flatnonzero(sign * (seg - stp) <= 0)
            ti = hit_t[0] if hit_t.size else np.inf
            si = hit_s[0] if hit_s.size else np.inf
            if ti <= si and ti != np.inf:
                k = je + int(ti)
                px = tgt if order_type == "LIMIT" else quote(k, -sign)
                return "target", float(px), int(t_ns[k])
            if si < ti:
                k = je + int(si)
                px = stp if order_type == "LIMIT" else quote(k, -sign)
                return "stop", float(px), int(t_ns[k])
        cut = min(deadline_ns, end_ns)
        if tick_edge_ns >= cut:                       # window complete, no touch
            j = asof(cut)
            if j is not None:                         # flatten at market
                reason = "timeout" if deadline_ns <= end_ns else "eod"
                return reason, quote(j, -sign), int(t_ns[j])
        return None

    def _close_exit(i: int, sign: float, tgt: float, stp: float,
                    deadline_ns: int) -> tuple[str, float, int] | None:
        """ONCLOSE: barriers assessed at 5s closes (known at ts+5s); the exit
        order re-runs the pipeline (close + signal + entry latency) and fills
        at the prevailing bid/ask."""
        cut = min(deadline_ns, end_ns)
        for k in range(i + 1, len(ts_ns)):
            close_ns = int(ts_ns[k]) + BAR_NS         # when this close is known
            if close_ns > cut:
                break
            c = close[k]
            if not np.isfinite(c):
                continue
            reason = None
            if sign * (c - tgt) >= 0:
                reason = "target"
            elif sign * (c - stp) <= 0:
                reason = "stop"
            if reason:
                sent_ns = close_ns + lat_ns + ent_ns  # pipeline + transmission
                if order_type == "LIMIT":
                    return reason, float(tgt if reason == "target" else stp), sent_ns
                j = asof(sent_ns)
                if j is not None:
                    return reason, quote(j, -sign), sent_ns
                return reason, float(c), sent_ns
        if edge_ns >= cut:                            # day/window complete
            k = min(int(np.searchsorted(ts_ns, cut - BAR_NS, side="right")) - 1,
                    len(ts_ns) - 1)
            if k > i and np.isfinite(close[k]):
                reason = "timeout" if deadline_ns <= end_ns else "eod"
                out_ns = int(ts_ns[k]) + BAR_NS
                j = asof(out_ns)
                px = quote(j, -sign) if j is not None else float(close[k])
                return reason, px, out_ns
        return None

    rows: list[dict[str, Any]] = []
    open_exits: list[int | None] = []                 # exit ts_ns of accepted fills

    def _iso(ns: int) -> str:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()

    for i in range(len(df)):
        d = str(directions[i])
        conf = float(confidences[i])
        if d not in ("up", "down") or conf < tau:
            continue
        bar_close_ns = int(ts_ns[i]) + BAR_NS         # bar close (label at ts+5s)
        signal_ns = bar_close_ns + lat_ns             # signal known after pipeline
        order_ns = signal_ns + ent_ns                 # entry order reaches market
        if not start_ns <= bar_close_ns <= end_ns:
            continue

        sign = 1.0 if d == "up" else -1.0
        n_open = sum(1 for e in open_exits if e is None or e > signal_ns)
        skipped = n_open >= max_pos

        ind_px = float(close[i]) if np.isfinite(close[i]) else None
        sj = asof(signal_ns)
        signal_px = float(t_mid[sj]) if sj is not None else ind_px
        row: dict[str, Any] = {
            "signal_ts": _iso(signal_ns),             # bar close + signal latency
            "bar_ts": _iso(bar_close_ns),             # end of the 5s indicator bar
            "indicator_px": round(ind_px, 2) if ind_px is not None else None,
            "signal_px": round(signal_px, 2) if signal_px is not None else None,
            "signal_px_source": "tbbo_mid" if sj is not None else "bar_close",
            "dir": d, "conf": round(conf, 3),
            "n_feat": int(n_present[i]), "n_feat_total": n_total,
            "entry_ts": None, "entry_px": None, "entry_px_source": None,
            "target_px": None, "stop_px": None,
            "exit_ts": None, "exit_px": None, "hold_s": None, "exit_reason": None,
            "pnl_pts": None, "pnl_usd": None,
            "order_type": order_type, "execution": execution,
            "pending": False, "skipped": skipped,
        }
        if skipped:
            row["exit_reason"] = "skipped_max_pos"
            rows.append(row)
            continue
        if ind_px is None or signal_px is None:
            row["exit_reason"] = "no_price"
            rows.append(row)
            continue

        # Barriers anchored at the indicator close (label definition), not the fill.
        tgt = ind_px + tgt_pts * sign
        stp = ind_px - stop_pts * sign
        row["target_px"] = round(tgt, 2)
        row["stop_px"] = round(stp, 2)
        deadline_ns = bar_close_ns + horizon_ns

        # ── entry fill ──
        entry_px: float | None = None
        entry_fill_ns: int | None = None
        je: int | None = None                          # first tick idx after entry
        if t_ns is None:                               # no tick data — bar fallback
            entry_px, entry_fill_ns = ind_px, order_ns
            row["entry_px_source"] = "bar_close"
        elif order_type == "LIMIT":
            limit_px = signal_px                       # rest at the signal mid
            j0 = int(np.searchsorted(t_ns, order_ns, side="left"))
            jd = int(np.searchsorted(t_ns, min(deadline_ns, end_ns), side="right"))
            seg = t_px[j0:jd]
            hit = np.flatnonzero(sign * (seg - limit_px) <= 0)
            if hit.size:
                k = j0 + int(hit[0])
                entry_px, entry_fill_ns, je = limit_px, int(t_ns[k]), k + 1
                row["entry_px_source"] = "limit_mid"
            elif tick_edge_ns >= min(deadline_ns, end_ns):
                row["exit_reason"] = "entry_unfilled"
                rows.append(row)
                continue
            else:                                      # live: still resting
                row["pending"] = True
                row["pending_entry"] = True
                open_exits.append(None)
                rows.append(row)
                continue
        else:                                          # MKT at prevailing quote
            ej = asof(order_ns)
            if ej is not None:
                entry_px, entry_fill_ns = quote(ej, sign), order_ns
                je = int(np.searchsorted(t_ns, order_ns, side="right"))
                row["entry_px_source"] = "ask" if sign > 0 else "bid"
            elif tick_edge_ns and order_ns > tick_edge_ns:   # live edge
                row["pending"] = True
                row["pending_entry"] = True
                open_exits.append(None)
                rows.append(row)
                continue
            else:                                      # before first tick of day
                entry_px, entry_fill_ns = ind_px, order_ns
                row["entry_px_source"] = "bar_close"

        row["entry_ts"] = _iso(entry_fill_ns)
        row["entry_px"] = round(entry_px, 2)

        if execution == "OCO" and je is not None:
            res = _tick_exit(je, sign, tgt, stp, deadline_ns)
        else:
            res = _close_exit(i, sign, tgt, stp, deadline_ns)

        if res is None:
            row["pending"] = True
            open_exits.append(None)
        else:
            reason, exit_px, exit_ns = res
            pnl = (exit_px - entry_px) * sign
            row.update(
                exit_reason=reason,
                exit_px=round(exit_px, 2),
                exit_ts=_iso(exit_ns),
                hold_s=round((exit_ns - entry_fill_ns) / 1e9, 1),
                pnl_pts=round(pnl, 2),
                pnl_usd=round(pnl * float(trade.get("point_value") or 20.0), 2),
            )
            open_exits.append(exit_ns)
        rows.append(row)

    rows.reverse()                                    # newest first
    return rows


def day_stats(rows: list[dict[str, Any]], point_value: float) -> dict[str, Any]:
    """Up/down counts, realized PnL, and intraday max drawdown (running-min ≤ 0)."""
    n_up = sum(1 for r in rows if r["dir"] == "up")
    n_down = sum(1 for r in rows if r["dir"] == "down")
    filled = [r for r in rows if isinstance(r.get("pnl_pts"), (int, float))]
    filled.sort(key=lambda r: r.get("exit_ts") or "")
    running = 0.0
    max_dd = 0.0
    n_win = n_stop = n_to = 0
    for r in filled:
        running += float(r["pnl_pts"])
        max_dd = min(max_dd, running)
        if r["exit_reason"] == "target":
            n_win += 1
        elif r["exit_reason"] == "stop":
            n_stop += 1
        elif r["exit_reason"] in ("timeout", "eod"):
            n_to += 1
    net = round(running, 2)
    return {
        "n_signals": len(rows),
        "n_up": n_up,
        "n_down": n_down,
        "n_filled": len(filled),
        "n_open": sum(1 for r in rows if r.get("pending")),
        "n_skipped": sum(1 for r in rows if r.get("skipped")),
        "n_win": n_win, "n_stop": n_stop, "n_timeout": n_to,
        "net_pnl_pts": net,
        "net_pnl_usd": round(net * point_value, 2),
        "max_drawdown_pts": round(max_dd, 2),
        "max_drawdown_usd": round(max_dd * point_value, 2),
        "point_value": point_value,
    }


# ── public signal ledger entrypoint ───────────────────────────────────────────

def _signal_rows(
    objective_id: str,
    session: dict[str, Any],
    effective: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return actual live-loop emissions for one objective and session day."""
    session_date = str(session["date"])
    path = live_dir() / "signals.jsonl"
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A concurrent append can leave its final line incomplete.
                    continue
                if (not isinstance(row, dict)
                        or row.get("objective_id") != objective_id
                        or str(row.get("session_date")) != session_date
                        or not row.get("ts")):
                    continue
                if not live._is_within_session_window(
                    str(row["ts"]),
                    session,
                    trading_start=effective["trading_start"],
                    trading_end=effective["trading_end"],
                ):
                    continue
                rows.append(row)
    except OSError as exc:
        logger.warning("regmodel_signal_ledger_read_failed path=%s err=%s", path, exc)
        return []
    return sorted(rows, key=lambda row: str(row["ts"]), reverse=True)


def _signal_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n_signals": len(rows),
        "n_up": sum(row.get("dir") == "up" for row in rows),
        "n_down": sum(row.get("dir") == "down" for row in rows),
    }


def _synthetic_replay(
    objective_id: str,
    session: dict[str, Any],
    effective: dict[str, Any],
    rows_desc: list[dict[str, Any]],
    trade: dict[str, Any],
) -> dict[str, Any]:
    """Replay real emitted signals through the 5s bar-close first-touch resolver
    at mid price (no OCO, no TBBO, no latency), capped by max_positions.

    Same target/stop/horizon barriers as the objective, applied every 5s at the
    feature ``_close``. Long P/L = exit - entry, short = entry - exit.
    Returns:
        {
          "trades": [{...blotter row per trade...}],
          "summary": {...KPI dict, backtest-style...},
          "meta":    {...replay context...},
        }
    """
    import math

    import numpy as np
    import pandas as pd

    from src.features.hive_5s import load_session_day

    empty: dict[str, Any] = {
        "trades": [],
        "summary": {
            "n_trades": 0, "n_wins": 0, "n_losses": 0, "win_rate": None,
            "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0, "net_value": 0.0,
            "expectancy": None, "profit_factor": None, "max_drawdown": 0.0,
            "avg_win": None, "avg_loss": None, "avg_holding_bars": None,
            "avg_holding_seconds": None,
            "by_exit_reason": {}, "n_open_at_close": 0,
        },
        "meta": {},
    }

    if not rows_desc:
        empty["meta"]["note"] = "No emitted signals for this session day."
        return empty

    session_date = str(session["date"])
    try:
        df = load_session_day(session_date)
        if "_close" in df.columns:
            df = df[["_close"]].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" in df.columns:
                df = df.set_index("ts")
            elif "bar_end_ts" in df.columns:
                df = df.set_index("bar_end_ts")
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("regmodel_replay_features_load_failed date=%s: %s",
                       session_date, exc)
        empty["meta"]["note"] = f"Failed to read 5s features: {exc}"
        return empty
    if df.empty:
        empty["meta"]["note"] = (
            f"5s features not yet materialized for {session_date} — "
            "replay will run once the monthly part lands.")
        return empty

    if df.empty or "_close" not in df.columns:
        empty["meta"]["note"] = "5s feature parquet is empty."
        return empty

    bar_ts_ns = df.index.asi8
    close_arr = df["_close"].astype("float64").to_numpy(copy=True)

    target_pts = float(trade.get("move_threshold") or 50.0)
    stop_pts = float(trade.get("mae_tolerance") or 6.0)
    horizon_s = int(trade.get("horizon_s") or 600)
    horizon_ns = int(horizon_s * 1_000_000_000)
    pv = float(trade.get("point_value") or 20.0)
    instrument = trade.get("instrument") or "NQ"
    max_positions = max(1, int(effective.get("max_positions") or 1))

    from src.backtest.weights_score import _resolve_signal_on_close

    empty_tbbo_ts = np.empty(0, dtype="int64")
    empty_tbbo_px = np.empty(0, dtype="float64")

    rows_asc = list(reversed(rows_desc))  # blotter cache is newest-first
    open_until_ns: list[int] = []
    trades: list[dict[str, Any]] = []
    skipped_slots = 0

    for r in rows_asc:
        direction = str(r.get("dir") or "").lower()
        if direction not in ("up", "down"):
            continue
        try:
            ts = pd.Timestamp(str(r["ts"]))
        except Exception:  # noqa: BLE001
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        signal_ts_ns = int(ts.value)

        open_until_ns = [ex for ex in open_until_ns if ex > signal_ts_ns]
        if len(open_until_ns) >= max_positions:
            skipped_slots += 1
            continue

        loc = int(np.searchsorted(bar_ts_ns, signal_ts_ns, side="right") - 1)
        if 0 <= loc < len(close_arr) and math.isfinite(float(close_arr[loc])):
            entry_close = float(close_arr[loc])
            entry_ts_ns = int(bar_ts_ns[loc])
        else:
            entry_close = r.get("entry_px")
            try:
                entry_close = float(entry_close)
            except (TypeError, ValueError):
                entry_close = float("nan")
            entry_ts_ns = signal_ts_ns
        if not math.isfinite(entry_close):
            continue

        res = _resolve_signal_on_close(
            is_long=(direction == "up"),
            signal_ts_ns=entry_ts_ns,
            signal_close_px=entry_close,
            bar_ts_ns=bar_ts_ns,
            bar_close_px=close_arr,
            horizon_ns=horizon_ns,
            target_pts=target_pts,
            stop_pts=stop_pts,
            tbbo_ts=empty_tbbo_ts,
            tbbo_px=empty_tbbo_px,
            tbbo_bid=empty_tbbo_px,
            tbbo_ask=empty_tbbo_px,
            price_mode="mid",
            entry_latency_ns=0,
            exit_latency_ns=0,
        )

        exit_ts_ns = int(res["exit_ts_ns"])
        fill_entry_ns = int(res.get("entry_ts_ns") or entry_ts_ns)
        open_until_ns.append(max(exit_ts_ns, fill_entry_ns + 1))
        hold_ns = int(res.get("holding_ns") or 0)
        hold_s = hold_ns / 1e9
        ep = float(res["entry_px"])
        xp = float(res["exit_px"])
        pnl = float(res["pnl_points"])

        trades.append({
            "signal_ts": ts.isoformat(),
            "entry_ts": pd.Timestamp(fill_entry_ns, unit="ns", tz="UTC").isoformat(),
            "exit_ts": pd.Timestamp(exit_ts_ns, unit="ns", tz="UTC").isoformat(),
            "side": "long" if direction == "up" else "short",
            "dir": direction,
            "confidence": (float(r.get("conf")) if r.get("conf") is not None else None),
            "entry_px": round(ep, 2) if math.isfinite(ep) else None,
            "exit_px": round(xp, 2) if math.isfinite(xp) else None,
            "pnl_points": pnl,
            "pnl_usd": pnl * pv,
            "exit_reason": res.get("outcome"),
            "holding_seconds": round(hold_s, 1),
            "holding_bars": max(1, int(round(hold_s / 5.0))) if hold_ns > 0 else 0,
            "n_present": r.get("n_present"),
            "n_total": r.get("n_total"),
            "candidate_id": r.get("candidate_id") or r.get("model"),
        })

    trades.sort(key=lambda t: str(t["entry_ts"]))

    if not trades:
        empty["meta"] = {
            "target_points": target_pts, "stop_points": stop_pts,
            "horizon_s": horizon_s, "max_positions": max_positions,
            "point_value": pv, "instrument": instrument,
            "price_mode": "mid", "resolver": "on_close_bar_first_touch",
            "n_signals_seen": len(rows_desc),
            "n_skipped_slots": skipped_slots,
            "note": "All signals blocked by max_positions cap or missing feature bars.",
        }
        return empty

    pnls = np.array([t["pnl_points"] for t in trades], dtype="float64")
    cum = np.cumsum(pnls)
    for j, t in enumerate(trades):
        t["cum_pnl"] = float(cum[j])
        t["cum_pnl_usd"] = float(cum[j] * pv)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    hwm = np.maximum.accumulate(cum)
    hwm = np.maximum(hwm, 0.0)
    dd = cum - hwm
    from collections import Counter as _Counter
    by_exit = _Counter(str(t["exit_reason"]) for t in trades)

    now_ns = int(pd.Timestamp.utcnow().value)
    n_open = sum(1 for ex in open_until_ns if ex > now_ns)

    summary = {
        "n_trades": len(trades),
        "n_wins": int((pnls > 0).sum()),
        "n_losses": int((pnls <= 0).sum()),
        "gross_profit": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss": float(losses.sum()) if len(losses) else 0.0,
        "net_pnl": float(pnls.sum()),
        "net_value": float(pnls.sum() * pv),
        "win_rate": float((pnls > 0).mean()),
        "expectancy": float(pnls.mean()),
        "profit_factor": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and float(losses.sum()) != 0.0 else None
        ),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_holding_bars": float(np.mean([t["holding_bars"] for t in trades])),
        "avg_holding_seconds": float(np.mean([t["holding_seconds"] for t in trades])),
        "by_exit_reason": dict(by_exit),
        "n_open_at_close": n_open,
        "point_value": pv,
    }

    meta = {
        "target_points": target_pts,
        "stop_points": stop_pts,
        "horizon_s": horizon_s,
        "max_positions": max_positions,
        "point_value": pv,
        "instrument": instrument,
        "price_mode": "mid",
        "resolver": "on_close_bar_first_touch",
        "n_signals_seen": len(rows_desc),
        "n_skipped_slots": skipped_slots,
        "n_open_at_end": n_open,
        "note": (
            f"Replay of {len(rows_desc)} emitted signals · 5s close first-touch on MID · "
            f"target +{target_pts:g} · stop −{stop_pts:g} · horizon {horizon_s}s · "
            f"max {max_positions} concurrent"
            + (f" · {skipped_slots} blocked by cap" if skipped_slots else "")
            + "."
        ),
    }
    return {"trades": trades, "summary": summary, "meta": meta}


def blotter(objective_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return immutable signals plus broker-derived execution state."""
    del force
    st = status(objective_id)
    session = st["session"]
    session_date = str(session["date"])
    rows = _signal_rows(objective_id, session, st["effective"])
    from src.execution.ledger import broker_blotter
    executions = broker_blotter(
        objective_id,
        session_date=session_date,
        point_value=float(st["trade"].get("point_value") or 20.0),
    )
    replay = _synthetic_replay(objective_id, session, st["effective"], rows, st["trade"])
    return {
        "rows": rows,
        "stats": _signal_stats(rows),
        "meta": {
            "source": "signals.jsonl",
            "session_date": session_date,
            "note": None if rows else "No emitted signals recorded for this session day.",
        },
        "executions": executions,
        "replay": replay,
        "status": st,
    }


def _build(objective_id: str, st: dict[str, Any]) -> dict[str, Any]:
    from src.serving import predict_canonical

    trade = st["trade"]
    eff = st["effective"]
    model = st["model"]
    feat = _feature_day(objective_id)
    df = feat["df"]
    notes = [n for n in (feat.get("note"),) if n]
    pv = float(trade.get("point_value") or 20.0)

    if df is None or df.empty:
        return {"rows": [], "stats": day_stats([], pv),
                "meta": {"note": "no feature data for the session day yet",
                         "session_date": st["session"]["date"]}}

    b = _bundle_for(model["path"])
    feat_cols = b["feature_cols"]
    trade = {**trade, "n_features": len(feat_cols)}
    preds = predict_canonical(df, b)
    n_present = (len(feat_cols)
                 - df.reindex(columns=list(feat_cols)).isna().sum(axis=1)).to_numpy()

    day = str(feat["session_date"])
    ticks = _load_ticks(day)
    if ticks is None or not ticks.get("n"):
        notes.append("no TBBO ticks for the day — entries/exits fall back to 5s closes")
        ticks = None

    ctx = dict(feat["context"])
    ctx["session_date"] = feat["session_date"]
    rows = _simulate(df, preds, n_present, ticks=ticks, eff=eff, trade=trade, ctx=ctx)
    return {
        "rows": rows,
        "stats": day_stats(rows, pv),
        "meta": {
            "objective_id": objective_id,
            "model": model.get("candidate_id") or model.get("name"),
            "model_path": model["path"],
            "model_source": model.get("registered") or model.get("source"),
            "session_date": day,
            "timezone": ctx.get("timezone"),
            "tau": eff["tau"],
            "max_positions": eff["max_positions"],
            "order_type": eff["order_type"],
            "execution": eff["execution"],
            "signal_latency_ms": eff["signal_latency_ms"],
            "entry_latency_ms": eff["entry_latency_ms"],
            "n_bars": int(len(df)),
            "n_features": len(feat_cols),
            "n_ticks": int(ticks["n"]) if ticks else 0,
            "tick_source": "tbbo_db" if ticks else "bar_close_fallback",
            "barriers": {"move_threshold": trade["move_threshold"],
                         "mae_tolerance": trade["mae_tolerance"],
                         "horizon_s": trade["horizon_s"]},
            "instrument": trade.get("instrument") or "NQ",
            "point_value": pv,
            "note": " · ".join(notes) if notes else None,
        },
    }


def invalidate_cache() -> None:
    with _LOCK:
        _BUNDLES.clear()
        _TICKS.clear()
        _BLOTTER.clear()
