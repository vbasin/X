"""Run an external model (or a stack) and emit the Backtest-tab payload shape."""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.common.cme_holidays import trading_days_in_range
from src.common.instruments import point_value_usd
from src.external_models.candles import CANDLE_TABLE, candle_span, load_lead_ohlc
from src.external_models.catalog import (
    check_hardened_name,
    check_model_id,
    coerce_book_params,
    coerce_params,
    compat_stack_for,
    factory_book_params,
    factory_params,
    get_model,
)
from src.external_models.pine_sar_ema import (
    build_signals,
    normalize_ohlc,
    simulate_book,
)
from src.external_models.timeframe import (
    CANDLE_TIMEFRAMES,
    align_signal_frame,
    coerce_candle_tf,
    resample_ohlc,
    retag_sar_on_book,
    tf_seconds,
    warmup_days_for,
)
from src.external_models.store import (
    effective_book_params,
    effective_params,
    list_hardened,
    load_book_defaults,
    load_defaults,
    load_hardened,
    maybe_migrate_legacy_psar_ema20,
)

logger = logging.getLogger("vtech.external_models.runner")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_PREFIX = "external:"
STACK_PREFIX = "external:stack:"


def parse_source(source: str) -> tuple[str, str | None]:
    """``external:<id>`` or ``external:<id>:hardened:<name>`` → (model_id, hardened).

    Stack sources (``external:stack:...``) raise ``ValueError`` — use
    :func:`parse_stack`.
    """
    raw = (source or "").strip()
    if not raw.startswith(SOURCE_PREFIX):
        raise ValueError("not an external-model source")
    if raw.startswith(STACK_PREFIX):
        raise ValueError("stacked source — use parse_stack")
    rest = raw[len(SOURCE_PREFIX):]
    if not rest:
        raise ValueError("external source is missing a model id")
    parts = rest.split(":")
    model_id, _tf = _split_model_tf(parts[0])
    hardened = None
    if len(parts) == 1:
        hardened = None
    elif len(parts) >= 3 and parts[1] == "hardened":
        hardened = ":".join(parts[2:])
    elif len(parts) == 2:
        hardened = parts[1]
    else:
        raise ValueError(f"unrecognised external source: {source}")
    return model_id, hardened or None


def is_external_source(source: str | None) -> bool:
    return bool(source) and str(source).startswith(SOURCE_PREFIX)


def _split_model_tf(token: str) -> tuple[str, str | None]:
    raw = str(token or "").strip()
    if "@" in raw:
        mid, tf = raw.split("@", 1)
        return check_model_id(mid), coerce_candle_tf(tf)
    return check_model_id(raw), None


def _leg_from_mapping(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("stack leg must be an object")
    model_token = str(raw.get("model") or raw.get("id") or "")
    model_id, tf_from_id = _split_model_tf(model_token)
    hardened = raw.get("hardened") or None
    if hardened:
        hardened = check_hardened_name(str(hardened))
    params = raw.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError("stack leg params must be an object")
    timeframe = raw.get("timeframe")
    if timeframe:
        timeframe = coerce_candle_tf(timeframe)
    else:
        timeframe = tf_from_id
    return {
        "model": model_id,
        "hardened": hardened,
        "params": params,
        "timeframe": timeframe,
    }


def parse_stack(
    source: str | None = None,
    stack: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalise a Backtest stack to ``[{model, hardened, params}, ...]``."""
    if stack:
        legs = [_leg_from_mapping(item) for item in stack]
        if not legs:
            raise ValueError("stack is empty")
        return legs
    raw = (source or "").strip()
    if raw.startswith(STACK_PREFIX):
        rest = raw[len(STACK_PREFIX):]
        if not rest:
            raise ValueError("stacked source is missing model ids")
        legs = []
        for part in rest.split("+"):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            model_id, timeframe = _split_model_tf(bits[0])
            hardened = None
            if len(bits) >= 3 and bits[1] == "hardened":
                hardened = check_hardened_name(":".join(bits[2:]))
            elif len(bits) == 2:
                hardened = check_hardened_name(bits[1])
            legs.append({
                "model": model_id, "hardened": hardened,
                "params": None, "timeframe": timeframe,
            })
        if not legs:
            raise ValueError("stacked source is missing model ids")
        return legs
    model_id, hardened = parse_source(raw)
    rest = raw[len(SOURCE_PREFIX):]
    head = rest.split(":")[0]
    _, tf_from_src = _split_model_tf(head)
    compat = compat_stack_for(model_id)
    if compat:
        return [
            {"model": mid, "hardened": None, "params": None, "timeframe": tf_from_src}
            for mid in compat
        ]
    return [{
        "model": model_id, "hardened": hardened,
        "params": None, "timeframe": tf_from_src,
    }]


def _leg_token(leg: dict[str, Any]) -> str:
    bit = str(leg["model"])
    tf = leg.get("timeframe")
    if tf:
        bit += f"@{coerce_candle_tf(tf)}"
    if leg.get("hardened"):
        bit += f":hardened:{leg['hardened']}"
    return bit


def format_stack_source(legs: list[dict[str, Any]]) -> str:
    if not legs:
        raise ValueError("stack is empty")
    if len(legs) == 1:
        return SOURCE_PREFIX + _leg_token(legs[0])
    return STACK_PREFIX + "+".join(_leg_token(leg) for leg in legs)


def resolve_model_params(
    model_id: str,
    *,
    hardened: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if params:
        return coerce_params(model_id, params)
    if hardened:
        rec = load_hardened(model_id, hardened)
        return coerce_params(model_id, rec.get("params"))
    return effective_params(model_id)


def resolve_params(
    model_id: str,
    *,
    hardened: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy helper: model knobs plus current book defaults (flat)."""
    model = resolve_model_params(model_id, hardened=hardened, params=params)
    book = coerce_book_params(params) if params else effective_book_params()
    return {**book, **model}


def resolve_book_params(book: dict[str, Any] | None = None,
                        params: dict[str, Any] | None = None) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    if params:
        overlay.update(params)
    if book:
        overlay.update(book)
    if overlay:
        return coerce_book_params(overlay)
    return effective_book_params()


def _iso(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


def _stack_title(legs: list[dict[str, Any]], cards: dict[str, dict[str, Any]]) -> str:
    bits = []
    for leg in legs:
        card = cards.get(leg["model"]) or {}
        name = card.get("title") or leg["model"]
        tf = leg.get("timeframe")
        if tf:
            name = f"{name} {tf}"
        if leg.get("hardened"):
            name = f"{name} / {leg['hardened']}"
        bits.append(str(name))
    return " + ".join(bits)


def _qty_note(label: str, value: Any, unit: Any) -> str:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        v = 0.0
    if v <= 0:
        return f"{label} off"
    if str(unit or "points") == "atr":
        return f"{label} {v:g}×ATR"
    if str(unit) == "stop":
        return f"{label} {v:g}×stop"
    return f"{label} {v:g}pt"


def _weights_note(title: str, legs: list[dict[str, Any]], book: dict[str, Any],
                  resolved_legs: list[dict[str, Any]]) -> str:
    parts = [
        f"External model {title}: 1m (or resampled) candles from {CANDLE_TABLE}.",
        "Not a 50/6 ML bundle.",
        f"Session {book.get('session_start')}–{book.get('session_end')} {book.get('timezone')}",
        f"(entries {book.get('trade_start')}–{book.get('trade_end')}).",
        f"ATR({book.get('atr_period')}) "
        f"{_qty_note('min', book.get('min_atr'), book.get('min_atr_unit'))};",
        "exit "
        f"{_qty_note('PT', book.get('profit_target_pts'), book.get('profit_target_unit'))}"
        "/"
        f"{_qty_note('SL', book.get('stop_loss_pts'), book.get('stop_loss_unit'))}.",
    ]
    if book.get("signal_lock"):
        parts.append("Signal lock on.")
    if book.get("trade_long") is False:
        parts.append("Longs off.")
    if book.get("trade_short") is False:
        parts.append("Shorts off.")
    if len(legs) > 1:
        parts.append("Stacked: matching sides enter; opposing or flat exits.")
    for rec in resolved_legs:
        mid = rec["model"]
        p = rec["params"]
        tf_note = rec.get("timeframe")
        tf_bit = f" {tf_note}" if tf_note else ""
        if mid == "psar":
            extra = (
                f"SAR{tf_bit} start={p.get('start')} inc={p.get('increment')} "
                f"max={p.get('maximum')}"
                f"{'; exit at SAR' if p.get('exit_at_sar') else ''}."
            )
        elif mid == "ema20":
            extra = (
                f"EMA{tf_bit} {p.get('ema_length')} L={p.get('ema20_l_bars')} "
                f"S={p.get('ema20_s_bars')}."
            )
        elif mid == "zigzag":
            extra = (
                f"ZigZag{tf_bit} %={p.get('percentage_reversal')} "
                f"abs={p.get('absolute_reversal')} "
                f"ATR({p.get('atr_length')})×{p.get('atr_reversal')} "
                f"ticks={p.get('tick_reversal')}."
            )
        elif mid == "timed_psar":
            mode = p.get("entry_mode") or "clock"
            when = (
                f"bars_from_open={p.get('bars_from_open')}"
                if mode == "bars_from_open"
                else f"clock {p.get('entry_time')}"
            )
            extra = (
                f"Timed PSAR{tf_bit} {when} "
                f"dir={p.get('direction_tf')} "
                f"AF {p.get('start')}/{p.get('increment')}/{p.get('maximum')}."
            )
        else:
            extra = f"{mid}{tf_bit}."
        parts.append(extra)
    return " ".join(parts)


def _payload(
    *,
    legs: list[dict[str, Any]],
    source: str,
    book: dict[str, Any],
    resolved_legs: list[dict[str, Any]],
    result,
    start: str | None,
    end: str | None,
    n_sessions: int,
    cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ledger = result.trades
    tf = str(book.get("timeframe") or "1min")
    try:
        bar_seconds = tf_seconds(tf)
    except ValueError:
        bar_seconds = 60 if tf.startswith("15") else (300 if tf.startswith("5") else 60)
    pv = point_value_usd("NQ", default=20.0) * int(book.get("qty") or 1)
    title = _stack_title(legs, cards)
    lo = start or ""
    hi = end or ""
    if ledger is not None and not ledger.empty:
        lo = lo or str(pd.Timestamp(ledger["entry_ts"].iloc[0]).date())
        hi = hi or str(pd.Timestamp(ledger["exit_ts"].iloc[-1]).date())
    primary = legs[0]["model"]
    flat_params = dict(book)
    for rec in resolved_legs:
        flat_params.update(rec["params"])

    config: dict[str, Any] = {
        "strategy_id": "+".join(leg["model"] for leg in legs),
        "objective_id": None,
        "tau": 0.0,
        "sides": ["up", "down"],
        "max_concurrent_positions": 1,
        "target_points": book.get("profit_target_pts"),
        "stop_points": book.get("stop_loss_pts"),
        "max_holding": f"flatten {book.get('session_end')}",
        "point_value": float(pv),
        "instrument": "NQ",
        "window": {
            "start": lo, "end": hi, "n_sessions": int(n_sessions),
            "n_bars": int(result.n_bars),
        },
        "prediction_mode": "external",
        "fill_path": "pine_1m",
        "execution": "pine_candle",
        "source_kind": "external",
        "source": source,
        "run_label": title,
        "model_path": source,
        "candidate_id": primary,
        "model_family": "external_pine",
        "timeframe": tf,
        "bar_seconds": bar_seconds,
        "candle_table": CANDLE_TABLE,
        "params": flat_params,
        "book": book,
        "stack": [
            {
                "model": rec["model"],
                "hardened": rec.get("hardened"),
                "params": rec["params"],
                "timeframe": rec.get("timeframe") or tf,
                "title": (cards.get(rec["model"]) or {}).get("title") or rec["model"],
            }
            for rec in resolved_legs
        ],
        "hardened": legs[0].get("hardened") if len(legs) == 1 else None,
        "live_data": False,
        "tbbo_data": False,
        "oco": False,
        "price_mode": "close",
        "weights_note": _weights_note(title, legs, book, resolved_legs),
    }

    labels = ["down", "flat", "up"]
    empty_keys = ("gross_profit", "gross_loss", "net_pnl", "win_rate", "expectancy",
                  "profit_factor", "max_drawdown", "sharpe_daily", "avg_win", "avg_loss")
    if ledger is None or ledger.empty:
        return {
            "config": config,
            "summary": {
                "n_trades": 0, "n_wins": 0, "n_losses": 0, "point_value": float(pv),
                "net_value": 0.0, "by_exit_reason": {}, "n_sessions_traded": 0,
                **{k: None for k in empty_keys},
            },
            "equity_curve": [],
            "confusion_matrix": {"labels": labels, "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
            "trades": [],
            "skipped": dict(result.skipped),
            "archetypes": [],
        }

    pnl = ledger["pnl_points"].to_numpy(dtype="float64")
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    cum = np.cumsum(pnl)
    equity = cum
    hwm = np.maximum.accumulate(np.maximum(equity, 0.0))
    dd = equity - hwm
    by_day = ledger.groupby("session_date")["pnl_points"].sum()
    daily = by_day.to_numpy(dtype="float64")

    def _sharpe(x: np.ndarray) -> float | None:
        if len(x) < 2:
            return None
        s = float(x.std(ddof=1))
        if s == 0.0:
            return None
        return float(x.mean() / s)

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
        "profit_factor": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and float(losses.sum()) != 0.0 else None
        ),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sharpe_daily": _sharpe(daily),
        "sharpe_trade": _sharpe(pnl),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_holding_bars": float(ledger["holding_bars"].mean()),
        "avg_mae": float(ledger["mae_points"].mean()),
        "avg_mfe": float(ledger["mfe_points"].mean()),
        "exposure_frac": float(ledger["holding_bars"].sum() / max(1, result.n_bars)),
        "n_sessions_traded": int(ledger["session_date"].nunique()),
        "by_exit_reason": {str(k): int(v) for k, v in ledger["exit_reason"].value_counts().items()},
        "point_value": float(pv),
    }

    entry_ts = pd.to_datetime(ledger["entry_ts"], utc=True)
    exit_ts = pd.to_datetime(ledger["exit_ts"], utc=True)
    equity_curve = [{"t": _iso(entry_ts.iloc[0]), "equity": 0.0}]
    equity_curve += [
        {"t": _iso(exit_ts.iloc[j]), "equity": float(cum[j])}
        for j in range(len(ledger))
    ]

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
            "entry_px": float(row["entry_px"]),
            "exit_px": float(row["exit_px"]),
            "pnl_points": float(row["pnl_points"]),
            "cum_pnl": float(cum[j]),
            "mae_points": float(row["mae_points"]),
            "mfe_points": float(row["mfe_points"]),
            "holding_bars": int(row["holding_bars"]),
            "exit_reason": str(row["exit_reason"]),
            "n_present": None,
            "n_total": None,
        })

    return {
        "config": config,
        "summary": summary,
        "equity_curve": equity_curve,
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm},
        "trades": trades_out,
        "skipped": dict(result.skipped),
        "archetypes": [],
    }


def run_backtest(
    source: str | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    params: dict[str, Any] | None = None,
    book: dict[str, Any] | None = None,
    stack: list[dict[str, Any]] | None = None,
    ohlc: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Simulate ``source`` / ``stack`` on PostgreSQL candles (or provided *ohlc*)."""
    legs = parse_stack(source, stack)
    book_p = resolve_book_params(book, params)
    book_tf = coerce_candle_tf(book_p.get("timeframe") or "1min")
    source_s = format_stack_source(legs)
    resolved_legs: list[dict[str, Any]] = []
    cards: dict[str, dict[str, Any]] = {}
    for leg in legs:
        mid = leg["model"]
        cards[mid] = get_model(mid)
        if isinstance(leg.get("params"), dict):
            model_p = resolve_model_params(
                mid, hardened=leg.get("hardened"), params=leg["params"])
        elif params:
            model_p = resolve_model_params(
                mid, hardened=leg.get("hardened"), params=params)
        else:
            model_p = resolve_model_params(mid, hardened=leg.get("hardened"))
        resolved_legs.append({
            "model": mid,
            "hardened": leg.get("hardened"),
            "params": model_p,
            "timeframe": coerce_candle_tf(leg.get("timeframe") or book_tf),
        })
    for d in (start, end):
        if d and not _DATE_RE.match(str(d)):
            raise ValueError("dates must be YYYY-MM-DD")
    root = str((next(iter(cards.values())) or {}).get("symbol_root") or "NQ")
    finest: pd.DataFrame
    if ohlc is None:
        if not start or not end:
            span = candle_span(root)
            if not span.get("available"):
                raise ValueError(
                    span.get("error") or "databento.tbbo_1min is unavailable")
            start = start or span["start"]
            end = end or span["end"]
        extra_tf = [
            rec["params"].get("direction_tf")
            for rec in resolved_legs
            if rec["model"] == "timed_psar"
        ]
        warmup = warmup_days_for(
            book_tf,
            *[rec["timeframe"] for rec in resolved_legs],
            *extra_tf,
        )
        logger.info("external_model_load %s %s→%s book_tf=%s warmup=%sd",
                    source_s, start, end, book_tf, warmup)
        try:
            finest = load_lead_ohlc(
                start, end, root=root, timeframe="1min",
                warmup_days=warmup,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"failed to load {CANDLE_TABLE}: {exc}") from exc
        if finest.empty:
            raise ValueError(
                f"no {CANDLE_TABLE} rows for lead NQ on {start}→{end}")
    else:
        finest = normalize_ohlc(ohlc)
    # ORB owns its own account-aware simulator (ADR-118) — it does not feed
    # the shared book, so it short-circuits the stack/consensus path.
    if len(resolved_legs) == 1 and resolved_legs[0]["model"] == "orb":
        from src.external_models.orb import simulate_orb, orb_payload
        orb_p = resolved_legs[0]["params"]
        res = simulate_orb(finest, orb_p)
        n_sessions = 0
        if start and end:
            n_sessions = len(trading_days_in_range(
                pd.Timestamp(start).date(), pd.Timestamp(end).date() + timedelta(days=1)))
        return orb_payload(
            res, source=source_s, start=start, end=end,
            n_sessions=n_sessions, card=cards["orb"],
        )
    # VWAP crossover is likewise standalone (owns its stops, targets, and
    # chop guards) and cannot be stacked with other engines.
    if len(resolved_legs) == 1 and resolved_legs[0]["model"] == "vwap_cross":
        from src.external_models.vwap_cross import simulate_vwap_cross, vwap_cross_payload
        vp = resolved_legs[0]["params"]
        # 15s cadence gives accurate stop / target / session_flat fills — see ADR notes in vwap_cross.py.
        finest_15s = None
        if ohlc is None and start and end:
            try:
                finest_15s = load_lead_ohlc(
                    start, end, root=root, timeframe="15s",
                    source_table="databento.tbbo_15s",
                    warmup_days=warmup_days_for(book_tf, "15s"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vwap_cross 15s load failed (%s); falling back to 1min", exc)
                finest_15s = None
        engine_input = finest_15s if (finest_15s is not None and not finest_15s.empty) else finest
        res = simulate_vwap_cross(engine_input, vp)
        n_sessions = 0
        if start and end:
            n_sessions = len(trading_days_in_range(
                pd.Timestamp(start).date(), pd.Timestamp(end).date() + timedelta(days=1)))
        return vwap_cross_payload(
            res, source=source_s, start=start, end=end,
            n_sessions=n_sessions, card=cards["vwap_cross"],
        )
    book_ohlc = resample_ohlc(finest, book_tf) if book_tf != "1min" else finest
    frames = []
    for rec in resolved_legs:
        leg_tf = rec["timeframe"]
        if rec["model"] == "timed_psar":
            frame = build_signals(
                rec["model"], finest, rec["params"],
                book_ohlc=book_ohlc, book=book_p,
            )
            if len(frame) != len(book_ohlc) or not frame.index.equals(book_ohlc.index):
                frame = align_signal_frame(frame, book_ohlc.index, book_tf, book_tf)
            frames.append(frame)
            continue
        src = finest if leg_tf == "1min" else resample_ohlc(finest, leg_tf)
        frame = build_signals(rec["model"], src, rec["params"])
        if leg_tf != book_tf:
            frame = align_signal_frame(frame, book_ohlc.index, leg_tf, book_tf)
            if rec["model"] == "psar":
                frame = retag_sar_on_book(frame, book_ohlc)
        elif len(frame) != len(book_ohlc) or not frame.index.equals(book_ohlc.index):
            frame = align_signal_frame(frame, book_ohlc.index, leg_tf, book_tf)
        frames.append(frame)
    result = simulate_book(
        book_ohlc, frames, book_p,
        model_ids=[rec["model"] for rec in resolved_legs],
        window_start=start, window_end=end,
    )
    n_sessions = 0
    if start and end:
        d0 = pd.Timestamp(start).date()
        d1 = pd.Timestamp(end).date()
        n_sessions = len(trading_days_in_range(d0, d1 + timedelta(days=1)))
    elif result.trades is not None and not result.trades.empty:
        n_sessions = int(result.trades["session_date"].nunique())
    return _payload(
        legs=legs, source=source_s, book=book_p, resolved_legs=resolved_legs,
        result=result, start=start, end=end, n_sessions=n_sessions, cards=cards,
    )


def _ui_card(card: dict[str, Any]) -> dict[str, Any]:
    """Factory YAML + operator overlays. No PostgreSQL on the catalog path."""
    mid = card["id"]
    saved = load_defaults(mid)
    return {
        **card,
        "defaults": effective_params(mid),
        "defaults_saved_at": (saved or {}).get("saved_at"),
        "defaults_note": (saved or {}).get("note"),
        "hardened": list_hardened(mid),
        "source": f"{SOURCE_PREFIX}{mid}",
        "candle_table": CANDLE_TABLE,
    }


def describe_book() -> dict[str, Any]:
    from src.external_models.catalog import BOOK_PARAM_GROUPS, BOOK_PARAM_SPEC

    maybe_migrate_legacy_psar_ema20()
    saved = load_book_defaults()
    return {
        "param_spec": list(BOOK_PARAM_SPEC),
        "factory_params": factory_book_params(),
        "defaults": effective_book_params(),
        "saved_at": (saved or {}).get("saved_at"),
        "note": (saved or {}).get("note"),
        "groups": [list(g) for g in BOOK_PARAM_GROUPS],
        "timeframes": list(CANDLE_TIMEFRAMES),
    }


def describe_model(model_id: str) -> dict[str, Any]:
    """Full card for the External Models tab (factory + saved defaults)."""
    maybe_migrate_legacy_psar_ema20()
    return _ui_card(get_model(model_id))


def list_for_ui() -> dict[str, Any]:
    from src.external_models.catalog import list_models

    maybe_migrate_legacy_psar_ema20()
    models = [_ui_card(card) for card in list_models()]
    return {"models": models, "book": describe_book()}


def list_backtest_entries() -> list[dict[str, Any]]:
    """Legacy flat dropdown rows (defaults + hardened). Unused by the split UI."""
    from src.external_models.catalog import list_models

    maybe_migrate_legacy_psar_ema20()
    out: list[dict[str, Any]] = []
    for card in list_models():
        mid = card["id"]
        params = {**effective_book_params(), **effective_params(mid)}
        out.append({
            "source_kind": "external",
            "source": f"{SOURCE_PREFIX}{mid}",
            "model": mid,
            "objective_id": "",
            "run_ts": "",
            "label": f"EXT · {card.get('title') or mid}",
            "run_label": card.get("title") or mid,
            "span_start": None,
            "span_end": None,
            "n_rows": None,
            "backtestable": True,
            "detail": (
                f"Pine replica on {CANDLE_TABLE}. "
                "Candle range is resolved when you run."
            ),
            "params": params,
            "timeframe": params.get("timeframe"),
            "candle_table": CANDLE_TABLE,
        })
        for rec in list_hardened(mid):
            name = rec["name"]
            out.append({
                "source_kind": "external",
                "source": f"{SOURCE_PREFIX}{mid}:hardened:{name}",
                "model": mid,
                "objective_id": "",
                "run_ts": rec.get("saved_at") or "",
                "label": f"EXT · {card.get('title') or mid} / {name}",
                "run_label": f"{card.get('title') or mid} / {name}",
                "span_start": None,
                "span_end": None,
                "n_rows": None,
                "backtestable": True,
                "detail": f"Hardened snapshot saved {rec.get('saved_at') or '?'}.",
                "params": rec.get("params") or factory_params(mid),
                "timeframe": (rec.get("params") or {}).get("timeframe"),
                "candle_table": CANDLE_TABLE,
                "hardened": name,
            })
    return out
