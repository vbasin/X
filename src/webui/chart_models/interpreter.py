"""Bar-by-bar interpreter that runs a compiled rule tree against a chart payload.

Contract
--------

Inputs:

- ``payload`` — the object returned by
  :func:`src.webui.live_chart.series` (candles as parallel arrays; profile
  levels; overlay arrays; prints list).
- ``spec`` — a compiled model spec from
  :func:`src.webui.chart_models.schema.compile_model`.

Outputs (JSON-serialisable):

- ``markers`` — list of ``{time, type, side, price, text}`` for the chart
  ('entry_long' / 'entry_short' / 'exit_target' / 'exit_stop' /
  'exit_signal' / 'exit_session').
- ``trades`` — list of trade records (side, entry/exit ts+price, PnL in
  points and USD, MAE/MFE, exit reason, matching rule index).
- ``pnl`` — summary (n trades, gross/net, win rate, max drawdown, avg
  win/loss, profit factor).

Execution model
---------------

The interpreter loops over completed bars in time order. On each bar it
first checks *any open position* against latched stop/target using the
bar's high/low first-touch semantics; then, if no position is open, it
evaluates the rule tree at the bar close and — if a rule fires — opens a
position **at the same bar's close** (matching the existing external
models). Stop / target levels are resolved from their expressions at
that moment and latched for the life of the trade. Same-bar
stop+target → stop (conservative).

Trade sizing is fixed to 1 contract in v1; the point-value is 20 (NQ).
Point value can be overridden through ``sim_options``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.webui.chart_models.expr import evaluate, ExprError, _numval, _boolval


CT = ZoneInfo("America/Chicago")
DEFAULT_POINT_VALUE = 20.0
DEFAULT_SESSION_END = "15:00"


# ── bindings + helpers ───────────────────────────────────────────────────

@dataclass
class _Series:
    t: list[int]           # bar-open unix
    o: list[float | None]
    h: list[float | None]
    l: list[float | None]
    c: list[float | None]
    v: list[float | None]
    overlays: dict[str, list[float | None]]
    profiles: list[dict[str, Any]]
    prints: list[dict[str, Any]]     # sorted by t (unix)
    tf_seconds: int
    session_end_hm: str

    def size(self) -> int:
        return len(self.t)


def _hhmm(unix: int) -> str:
    d = datetime.fromtimestamp(unix, tz=timezone.utc).astimezone(CT)
    return f"{d.hour:02d}:{d.minute:02d}"


def _bars_from_open(series: _Series, i: int) -> int:
    """Count of completed bars since 08:30 CT of the same calendar day as bar i."""
    target = datetime.fromtimestamp(series.t[i], tz=timezone.utc).astimezone(CT)
    day_open = target.replace(hour=8, minute=30, second=0, microsecond=0)
    if target < day_open:
        return 0
    j = i
    n = 0
    while j >= 0:
        d = datetime.fromtimestamp(series.t[j], tz=timezone.utc).astimezone(CT)
        if d < day_open:
            break
        n += 1
        j -= 1
    return max(0, n - 1)


def _color(o: float | None, c: float | None) -> str | None:
    if o is None or c is None:
        return None
    if c > o: return "green"
    if c < o: return "red"
    return "doji"


def _flat_vp_level(profile: dict[str, Any], field: str) -> float | None:
    """Extract a level, mapping DSL names to payload names."""
    payload_key = {
        "high":     "high",
        "low":      "low",
        "poc":      "poc",
        "vah":      "va_high",
        "val":      "va_low",
        "hi_body":  "hi_body",
        "lo_body":  "lo_body",
    }[field]
    v = profile.get(payload_key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _base_bindings(series: _Series, i: int) -> dict[str, Any]:
    o, h, l, c = series.o[i], series.h[i], series.l[i], series.c[i]
    po = series.o[i - 1] if i > 0 else None
    ph = series.h[i - 1] if i > 0 else None
    pl = series.l[i - 1] if i > 0 else None
    pc = series.c[i - 1] if i > 0 else None
    body_hi = None if (o is None or c is None) else max(o, c)
    body_lo = None if (o is None or c is None) else min(o, c)
    b: dict[str, Any] = {
        "C.o": o, "C.h": h, "C.l": l, "C.c": c,
        "C.color": _color(o, c),
        "C.body_hi": body_hi, "C.body_lo": body_lo,
        "C.range": None if (h is None or l is None) else h - l,
        "C.body":  None if (o is None or c is None) else abs(c - o),
        "Co": o, "Ch": h, "Cl": l, "Cc": c,
        "Po": po, "Ph": ph, "Pl": pl, "Pc": pc,
        "time": _hhmm(series.t[i]),
        "tick": 0.25,
        "pi": math.pi,
        "bars_from_open": _bars_from_open(series, i),
    }
    # Static (session-wide) VP levels — same for every bar during the
    # profile's session-tf coverage; unavailable levels bind to None so
    # comparisons short-circuit false during warmup.
    for p in series.profiles:
        n = p.get("n")
        if n is None:
            continue
        for field in ("high", "low", "poc", "vah", "val", "hi_body", "lo_body"):
            b[f"VP{n}.{field}"] = _flat_vp_level(p, field)
    # Current-bar overlay values under DSL aliases.
    from src.webui.chart_models.vars import OVERLAY_LABELS
    for key, (alias, _help) in OVERLAY_LABELS.items():
        arr = series.overlays.get(key)
        if not arr or i >= len(arr):
            continue
        v = arr[i]
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            b[alias] = None
        else:
            try:
                b[alias] = float(v)
            except (TypeError, ValueError):
                b[alias] = None
    return b


def _make_helpers(series: _Series, i: int, atr_cache: dict[int, list[float | None]]) -> dict[str, Any]:
    def _atr(n_raw: Any) -> float | None:
        try:
            n = int(_numval(n_raw) or 0)
        except (TypeError, ValueError):
            return None
        if n < 1:
            return None
        if n not in atr_cache:
            atr_cache[n] = _wilder_atr(series, n)
        arr = atr_cache[n]
        v = arr[i] if i < len(arr) else None
        return v

    def _ema(n_raw: Any) -> float | None:
        try:
            n = int(_numval(n_raw) or 0)
        except (TypeError, ValueError):
            return None
        if n < 1:
            return None
        vals = series.c[:i + 1]
        return _tv_ema(vals, n)

    def _sma(n_raw: Any) -> float | None:
        try:
            n = int(_numval(n_raw) or 0)
        except (TypeError, ValueError):
            return None
        if n < 1:
            return None
        window = [x for x in series.c[max(0, i - n + 1): i + 1] if x is not None]
        if len(window) < n:
            return None
        return sum(window) / len(window)

    now = series.t[i] + series.tf_seconds  # end-of-bar unix
    def _lp_window(secs_raw: Any) -> list[dict[str, Any]]:
        try:
            secs = float(_numval(secs_raw) or 0)
        except (TypeError, ValueError):
            secs = 0.0
        if secs <= 0:
            return []
        cutoff = now - secs
        # prints are pre-sorted ascending; scan backward for efficiency
        out: list[dict[str, Any]] = []
        for p in reversed(series.prints):
            t = float(p.get("t") or 0)
            if t <= cutoff:
                break
            if t >= now:
                continue
            out.append(p)
        return out

    def _lp_net(secs: Any) -> float:
        w = _lp_window(secs)
        return float(sum(_signed_size(p) for p in w))

    def _lp_buy(secs: Any) -> float:
        w = _lp_window(secs)
        return float(sum(float(p.get("sz") or 0) for p in w if str(p.get("side")).upper() == "B"))

    def _lp_sell(secs: Any) -> float:
        w = _lp_window(secs)
        return float(sum(float(p.get("sz") or 0) for p in w if str(p.get("side")).upper() == "A"))

    def _lp_count(secs: Any) -> float:
        return float(len(_lp_window(secs)))

    def _lp_max_size(secs: Any) -> float:
        w = _lp_window(secs)
        if not w:
            return 0.0
        return float(max(float(p.get("sz") or 0) for p in w))

    return {
        "atr": _atr, "ema": _ema, "sma": _sma,
        "lp_net": _lp_net, "lp_buy": _lp_buy,
        "lp_sell": _lp_sell, "lp_count": _lp_count,
        "lp_max_size": _lp_max_size,
    }


def _signed_size(p: dict[str, Any]) -> float:
    sz = float(p.get("sz") or 0)
    side = str(p.get("side") or "").upper()
    if side == "B": return sz
    if side == "A": return -sz
    return 0.0


def _tv_ema(vals: list[float | None], length: int) -> float | None:
    clean = [v for v in vals if v is not None]
    if len(clean) < length:
        return None
    alpha = 2.0 / (length + 1.0)
    seed = sum(clean[:length]) / length
    ema = seed
    for v in clean[length:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _wilder_atr(series: _Series, n: int) -> list[float | None]:
    out: list[float | None] = [None] * series.size()
    tr_prev_close: float | None = None
    trs: list[float | None] = []
    for i in range(series.size()):
        h, l, c = series.h[i], series.l[i], series.c[i]
        if h is None or l is None:
            trs.append(None)
            tr_prev_close = c
            continue
        tr = h - l
        if tr_prev_close is not None:
            tr = max(tr, abs(h - tr_prev_close), abs(l - tr_prev_close))
        trs.append(tr)
        tr_prev_close = c
    # Seed at index n-1 with SMA of first n TRs; then Wilder recursion.
    if series.size() < n:
        return out
    window = trs[:n]
    if any(x is None for x in window):
        # find earliest full window
        return out
    atr = sum(window) / n
    out[n - 1] = atr
    for i in range(n, series.size()):
        tr = trs[i]
        if tr is None:
            out[i] = out[i - 1]
            continue
        atr = (out[i - 1] * (n - 1) + tr) / n
        out[i] = atr
    return out


# ── rule tree evaluation ────────────────────────────────────────────────

def _eval_arg(arg: Any, bindings: dict[str, Any], helpers: dict[str, Any]) -> Any:
    if isinstance(arg, dict) and "__expr__" in arg:
        return evaluate(arg["__expr__"], bindings, helpers)
    return arg


def _cmp(op: str, a: Any, b: Any) -> bool | None:
    na, nb = _numval(a), _numval(b)
    if op in ("eq", "ne"):
        if isinstance(a, str) or isinstance(b, str):
            eq = str(a) == str(b)
            return eq if op == "eq" else (not eq)
        if na is None or nb is None: return None
        return (na == nb) if op == "eq" else (na != nb)
    if na is None or nb is None: return None
    return {
        "gt": na > nb, "lt": na < nb,
        "ge": na >= nb, "le": na <= nb,
    }[op]


def _eval_predicate(
    node: dict[str, Any],
    bindings: dict[str, Any],
    helpers: dict[str, Any],
    prev_bindings: dict[str, Any] | None,
) -> bool | None:
    kind, val = next(iter(node.items()))
    if kind == "all":
        result: bool | None = True
        for child in val:
            r = _eval_predicate(child, bindings, helpers, prev_bindings)
            if r is False: return False
            if r is None: result = None
        return result
    if kind == "any":
        result: bool | None = False
        for child in val:
            r = _eval_predicate(child, bindings, helpers, prev_bindings)
            if r is True: return True
            if r is None: result = None
        return result
    if kind == "not":
        r = _eval_predicate(val, bindings, helpers, prev_bindings)
        return None if r is None else (not r)
    if kind in ("eq", "ne", "gt", "lt", "ge", "le"):
        a = _eval_arg(val[0], bindings, helpers)
        b = _eval_arg(val[1], bindings, helpers)
        return _cmp(kind, a, b)
    if kind == "between":
        x = _numval(_eval_arg(val[0], bindings, helpers))
        lo = _numval(_eval_arg(val[1], bindings, helpers))
        hi = _numval(_eval_arg(val[2], bindings, helpers))
        if x is None or lo is None or hi is None: return None
        return lo <= x <= hi
    if kind in ("cross_above", "cross_below"):
        if prev_bindings is None:
            return False
        a_now = _numval(_eval_arg(val[0], bindings, helpers))
        b_now = _numval(_eval_arg(val[1], bindings, helpers))
        a_prev = _numval(_eval_arg(val[0], prev_bindings, helpers))
        b_prev = _numval(_eval_arg(val[1], prev_bindings, helpers))
        if None in (a_now, b_now, a_prev, b_prev): return False
        if kind == "cross_above":
            return a_prev <= b_prev and a_now > b_now
        return a_prev >= b_prev and a_now < b_now
    raise ValueError(f"unknown predicate {kind!r}")


def _check_when(gates: list[dict[str, Any]], bindings: dict[str, Any]) -> bool:
    if not gates:
        return True
    tm = bindings.get("time")
    for g in gates:
        kind, val = next(iter(g.items()))
        if kind == "time_between":
            if not isinstance(tm, str):
                return False
            if not (val[0] <= tm < val[1]):
                return False
        elif kind == "bars_from_open_ge":
            if int(bindings.get("bars_from_open") or 0) < int(val):
                return False
        elif kind == "bars_from_open_le":
            if int(bindings.get("bars_from_open") or 0) > int(val):
                return False
    return True


# ── main simulator ──────────────────────────────────────────────────────

@dataclass
class _Open:
    side: int             # +1 long, -1 short
    entry_i: int
    entry_t: int
    entry_px: float
    stop: float | None
    target: float | None
    rule_index: int
    stop_expr: Any = None
    target_expr: Any = None
    mae: float = 0.0
    mfe: float = 0.0


def simulate(
    payload: dict[str, Any],
    spec: dict[str, Any],
    *,
    point_value: float = DEFAULT_POINT_VALUE,
    session_end: str | None = None,
) -> dict[str, Any]:
    series = _to_series(payload, session_end=session_end)
    if series.size() < 2:
        return _empty_result()
    prev_b: dict[str, Any] | None = None
    open_pos: _Open | None = None
    markers: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    atr_cache: dict[int, list[float | None]] = {}

    for i in range(series.size()):
        if series.c[i] is None:
            prev_b = None
            continue
        b = _base_bindings(series, i)
        helpers = _make_helpers(series, i, atr_cache)
        just_exited = False
        # 1) manage an open position on this bar first
        if open_pos is not None:
            # session flatten if crossing session end
            if _at_or_past_session_end(series, i):
                exit_px = float(series.c[i])
                trades.append(_close_trade(open_pos, series, i, exit_px, "session"))
                markers.append(_marker(series.t[i], "exit_session", open_pos.side, exit_px, "flat"))
                open_pos = None
                just_exited = True
            else:
                hit = _first_touch(open_pos, series, i)
                if hit is not None:
                    exit_px, reason = hit
                    trades.append(_close_trade(open_pos, series, i, exit_px, reason))
                    markers.append(_marker(series.t[i], f"exit_{reason}", open_pos.side, exit_px, reason))
                    open_pos = None
                    just_exited = True
                else:
                    _track_excursions(open_pos, series, i)
        # 2) attempt an entry on this bar's close (only if flat and not just exited)
        if open_pos is None and not just_exited:
            for r_index, rule in enumerate(spec["rules"]):
                if not _check_when(rule["when"], b):
                    continue
                ok = _eval_predicate(rule["if"], b, helpers, prev_b)
                if not ok:
                    continue
                action = _execute_then(
                    rule["then"], b, helpers, prev_b,
                    entry_bindings=b, entry_helpers=helpers,
                )
                if action is None:
                    continue
                side, entry_px, stop_lvl, target_lvl = action
                open_pos = _Open(
                    side=side, entry_i=i, entry_t=series.t[i],
                    entry_px=entry_px, stop=stop_lvl, target=target_lvl,
                    rule_index=r_index,
                )
                markers.append(_marker(
                    series.t[i],
                    "entry_long" if side > 0 else "entry_short",
                    side, entry_px, f"rule {r_index + 1}",
                ))
                break
        prev_b = b

    if open_pos is not None:
        last_i = series.size() - 1
        exit_px = float(series.c[last_i] or open_pos.entry_px)
        trades.append(_close_trade(open_pos, series, last_i, exit_px, "eod"))
        markers.append(_marker(series.t[last_i], "exit_session", open_pos.side, exit_px, "eod"))

    return {
        "markers": markers,
        "trades": trades,
        "pnl": _summary(trades, point_value),
        "n_bars": series.size(),
    }


# ── then-tree execution ─────────────────────────────────────────────────

def _execute_then(
    actions: list[dict[str, Any]],
    bindings: dict[str, Any],
    helpers: dict[str, Any],
    prev_bindings: dict[str, Any] | None,
    *,
    entry_bindings: dict[str, Any],
    entry_helpers: dict[str, Any],
) -> tuple[int, float, float | None, float | None] | None:
    """Walk the ``then`` tree in order and materialise the first entry action.

    Returns ``(side, entry_px, stop, target)`` or ``None`` if the tree does
    not culminate in an entry action for this bar.
    """
    side = 0
    entry_px: float | None = None
    stop_expr: Any = None
    target_expr: Any = None
    for a in actions:
        if "if" in a and "then" in a:
            r = _eval_predicate(a["if"], bindings, helpers, prev_bindings)
            if r is not True:
                continue
            nested = _execute_then(
                a["then"], bindings, helpers, prev_bindings,
                entry_bindings=entry_bindings, entry_helpers=entry_helpers,
            )
            if nested is not None:
                return nested
            continue
        if "entry_long" in a:
            side = 1
            entry_px = _numval(_eval_arg(a["entry_long"], bindings, helpers))
        elif "entry_short" in a:
            side = -1
            entry_px = _numval(_eval_arg(a["entry_short"], bindings, helpers))
        elif "set_stop" in a:
            stop_expr = a["set_stop"]
        elif "set_target" in a:
            target_expr = a["set_target"]
    if side == 0 or entry_px is None:
        return None
    # Resolve stop / target with `entry` bound to the fill price.
    entry_ctx = dict(entry_bindings)
    entry_ctx["entry"] = entry_px
    stop = _numval(_eval_arg(stop_expr, entry_ctx, entry_helpers)) if stop_expr is not None else None
    target = _numval(_eval_arg(target_expr, entry_ctx, entry_helpers)) if target_expr is not None else None
    return side, float(entry_px), stop, target


# ── trade lifecycle helpers ─────────────────────────────────────────────

def _first_touch(pos: _Open, series: _Series, i: int) -> tuple[float, str] | None:
    """Bar-level first-touch OCO. Same-bar stop+target → stop (conservative)."""
    o, h, l = series.o[i], series.h[i], series.l[i]
    if h is None or l is None:
        return None
    stop, target = pos.stop, pos.target
    if pos.side > 0:
        hit_stop = stop is not None and l <= stop
        hit_tgt  = target is not None and h >= target
        if hit_stop and hit_tgt:
            gap_stop = stop is not None and o is not None and o <= stop
            return (float(o) if gap_stop else float(stop), "stop")
        if hit_stop:
            gap = o is not None and o <= stop
            return (float(o) if gap else float(stop), "stop")
        if hit_tgt:
            gap = o is not None and o >= target
            return (float(o) if gap else float(target), "target")
        return None
    hit_stop = stop is not None and h >= stop
    hit_tgt  = target is not None and l <= target
    if hit_stop and hit_tgt:
        gap_stop = stop is not None and o is not None and o >= stop
        return (float(o) if gap_stop else float(stop), "stop")
    if hit_stop:
        gap = o is not None and o >= stop
        return (float(o) if gap else float(stop), "stop")
    if hit_tgt:
        gap = o is not None and o <= target
        return (float(o) if gap else float(target), "target")
    return None


def _track_excursions(pos: _Open, series: _Series, i: int) -> None:
    h, l = series.h[i], series.l[i]
    if h is None or l is None:
        return
    if pos.side > 0:
        pos.mae = max(pos.mae, pos.entry_px - l)
        pos.mfe = max(pos.mfe, h - pos.entry_px)
    else:
        pos.mae = max(pos.mae, h - pos.entry_px)
        pos.mfe = max(pos.mfe, pos.entry_px - l)


def _close_trade(pos: _Open, series: _Series, i: int, exit_px: float, reason: str) -> dict[str, Any]:
    pnl_pts = (exit_px - pos.entry_px) * pos.side
    dur_bars = i - pos.entry_i
    return {
        "side": "long" if pos.side > 0 else "short",
        "entry_ts": _iso(series.t[pos.entry_i]),
        "entry_price": float(pos.entry_px),
        "exit_ts": _iso(series.t[i]),
        "exit_price": float(exit_px),
        "pnl_points": float(pnl_pts),
        "mae_points": float(pos.mae),
        "mfe_points": float(pos.mfe),
        "duration_bars": int(dur_bars),
        "exit_reason": reason,
        "rule_index": pos.rule_index,
        "stop_level": None if pos.stop is None else float(pos.stop),
        "target_level": None if pos.target is None else float(pos.target),
    }


def _marker(t: int, kind: str, side: int, price: float, text: str) -> dict[str, Any]:
    return {
        "time": int(t),
        "type": kind,
        "side": "long" if side > 0 else "short",
        "price": float(price),
        "text": text,
    }


def _iso(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _at_or_past_session_end(series: _Series, i: int) -> bool:
    hm = _hhmm(series.t[i])
    return hm >= series.session_end_hm


def _empty_result() -> dict[str, Any]:
    return {"markers": [], "trades": [], "pnl": _summary([], DEFAULT_POINT_VALUE), "n_bars": 0}


# ── PnL summary ─────────────────────────────────────────────────────────

def _summary(trades: list[dict[str, Any]], point_value: float) -> dict[str, Any]:
    n = len(trades)
    if not n:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
            "gross_points": 0.0, "gross_usd": 0.0,
            "avg_win_points": 0.0, "avg_loss_points": 0.0,
            "profit_factor": 0.0, "max_dd_points": 0.0,
            "point_value_usd": float(point_value),
            "equity_points": [],
        }
    pnl = [t["pnl_points"] for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x <= 0]
    gross = float(sum(pnl))
    equity: list[float] = []
    e = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in pnl:
        e += x
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)
        equity.append(e)
    win_rate = len(wins) / n
    avg_w = (sum(wins) / len(wins)) if wins else 0.0
    avg_l = (sum(losses) / len(losses)) if losses else 0.0
    gross_loss = sum(-x for x in losses)
    pf = (sum(wins) / gross_loss) if gross_loss > 0 else (float("inf") if wins else 0.0)
    return {
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(win_rate, 4),
        "gross_points": round(gross, 4),
        "gross_usd": round(gross * point_value, 2),
        "avg_win_points": round(avg_w, 4),
        "avg_loss_points": round(avg_l, 4),
        "profit_factor": None if math.isinf(pf) else round(pf, 4),
        "max_dd_points": round(max_dd, 4),
        "point_value_usd": float(point_value),
        "equity_points": [round(x, 4) for x in equity],
    }


# ── payload adapter ─────────────────────────────────────────────────────

_TF_SECONDS = {
    "5s": 5, "15s": 15, "1min": 60, "5min": 300, "15min": 900,
    "1hour": 3600, "1day": 86400,
}


def _to_series(payload: dict[str, Any], *, session_end: str | None) -> _Series:
    candles = payload.get("candles") or {}
    tf = str(payload.get("timeframe") or "1min")
    tf_sec = _TF_SECONDS.get(tf, 60)
    return _Series(
        t=list(candles.get("t") or []),
        o=list(candles.get("o") or []),
        h=list(candles.get("h") or []),
        l=list(candles.get("l") or []),
        c=list(candles.get("c") or []),
        v=list(candles.get("v") or []),
        overlays={k: list(v) for k, v in (payload.get("overlays") or {}).items()},
        profiles=list(payload.get("profiles") or []),
        prints=sorted(list(payload.get("prints") or []), key=lambda p: float(p.get("t") or 0)),
        tf_seconds=tf_sec,
        session_end_hm=str(session_end or DEFAULT_SESSION_END),
    )
