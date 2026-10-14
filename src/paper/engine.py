"""Paper execution engine: limit entry + OCO bracket on trade prints.

Fill rules (trade-through on last sale):

* **Buy limit** *L* fills when ``trade_px <= L`` → fill at ``min(trade, L)``
* **Sell limit** *L* fills when ``trade_px >= L`` → fill at ``max(trade, L)``
* **Sell stop** *S* triggers when ``trade_px <= S`` → fill at trade (marketable)
* **Buy stop** *S* triggers when ``trade_px >= S`` → fill at trade

On entry fill, an OCO pair is armed (target limit + stop). The first fill
cancels the sibling. This is **not** bar-close path-backtest logic.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from src.paper.policy import PaperPolicy

logger = logging.getLogger("vtech.paper.engine")

Side = Literal["long", "short"]
OrderKind = Literal["entry_limit", "target_limit", "stop"]


@dataclass
class MarketTick:
    """One market print (+ optional top of book) from Databento TBBO / replay."""

    ts: datetime
    price: float
    size: int = 0
    bid: float | None = None
    ask: float | None = None
    symbol: str = ""
    instrument_id: int | None = None


@dataclass
class WorkingOrder:
    order_id: str
    kind: OrderKind
    side: Side  # position side this order serves (long/short)
    limit_px: float  # for stop: trigger price
    qty: int = 1
    status: str = "working"  # working | filled | cancelled


@dataclass
class Position:
    trade_id: str
    side: Side
    entry_ts: datetime
    entry_px: float
    qty: int
    confidence: float
    signal_direction: str
    target_order_id: str | None = None
    stop_order_id: str | None = None


@dataclass
class PaperTrade:
    trade_id: str
    side: Side
    entry_ts: str
    exit_ts: str
    entry_px: float
    exit_px: float
    pnl_points: float
    holding_seconds: float
    exit_reason: str
    confidence: float
    objective_id: str
    model_path: str | None = None
    entry_style: str | None = None


@dataclass
class FillEvent:
    order_id: str
    kind: OrderKind
    side: Side
    fill_px: float
    ts: datetime
    trade_id: str | None = None
    reason: str = ""


@dataclass
class PaperEngine:
    """Stateful paper broker: signals → limit entry → OCO exits on ticks."""

    policy: PaperPolicy
    working: list[WorkingOrder] = field(default_factory=list)
    position: Position | None = None
    closed: list[PaperTrade] = field(default_factory=list)
    last_tick: MarketTick | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=lambda: {
        "flat_signal": 0,
        "low_conf": 0,
        "in_position": 0,
        "pending_entry": 0,
        "eod_no_entry": 0,
        "pre_open_no_entry": 0,
        "no_quote": 0,
        "entry_expired": 0,
    })

    # ── public API ────────────────────────────────────────────────

    def on_tick(self, tick: MarketTick) -> list[FillEvent]:
        """Advance market; try entry + OCO fills against this trade print."""
        if tick.price is None or tick.price <= 0:
            return []
        self.last_tick = tick
        fills: list[FillEvent] = []

        # expire stale entry limits
        fills.extend(self._expire_entries(tick))

        # try working orders (entry first, then OCO)
        still: list[WorkingOrder] = []
        for od in self.working:
            if od.status != "working":
                continue
            fe = self._try_fill(od, tick)
            if fe is None:
                still.append(od)
                continue
            fills.append(fe)
            od.status = "filled"
            self._on_fill(fe, tick)
        self.working = [o for o in still if o.status == "working"]

        # time / eod force-flat if in position
        if self.position is not None:
            reason = self._force_exit_reason(tick)
            if reason:
                fills.extend(self._force_close(tick, reason))

        return fills

    def on_signal(
        self,
        direction: str,
        confidence: float,
        *,
        ts: datetime | None = None,
    ) -> WorkingOrder | None:
        """Model fire: place a resting limit entry if flat and no pending entry."""
        ts = ts or (self.last_tick.ts if self.last_tick else datetime.now(timezone.utc))
        d = (direction or "flat").lower()
        if d not in ("up", "down"):
            self.skipped["flat_signal"] += 1
            return None
        if confidence < self.policy.dir_threshold:
            self.skipped["low_conf"] += 1
            return None
        if self._before_trading_start(ts):
            self.skipped["pre_open_no_entry"] += 1
            return None
        if self._past_trading_end(ts):
            self.skipped["eod_no_entry"] += 1
            return None
        if self.position is not None:
            self.skipped["in_position"] += 1
            return None
        if any(o.kind == "entry_limit" and o.status == "working" for o in self.working):
            self.skipped["pending_entry"] += 1
            return None

        limit_px = self._entry_limit_px(side="long" if d == "up" else "short")
        if limit_px is None:
            self.skipped["no_quote"] += 1
            return None

        side: Side = "long" if d == "up" else "short"
        oid = uuid.uuid4().hex[:12]
        od = WorkingOrder(
            order_id=oid, kind="entry_limit", side=side, limit_px=float(limit_px),
        )
        self.working.append(od)
        self.events.append({
            "type": "place_entry",
            "ts": _iso(ts),
            "side": side,
            "limit_px": float(limit_px),
            "confidence": float(confidence),
            "order_id": oid,
            "entry_style": self.policy.entry_style,
        })
        logger.info(
            "paper_place_entry id=%s %s limit=%.2f conf=%.3f style=%s",
            oid, side, limit_px, confidence, self.policy.entry_style,
        )
        # stash signal meta on the order via events; store conf on a side channel
        self._pending_meta[oid] = {
            "confidence": float(confidence),
            "signal_direction": d,
            "placed_ts": ts,
        }
        return od

    def summary(self) -> dict[str, Any]:
        pnls = [t.pnl_points for t in self.closed]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        return {
            "objective_id": self.policy.objective_id,
            "n_closed": n,
            "n_open": 1 if self.position else 0,
            "n_working": sum(1 for o in self.working if o.status == "working"),
            "net_pnl_points": float(sum(pnls)) if pnls else 0.0,
            "expectancy": float(sum(pnls) / n) if n else 0.0,
            "hit_rate": float(wins / n) if n else 0.0,
            "skipped": dict(self.skipped),
            "barriers": {
                "target_points": self.policy.target_points,
                "stop_points": self.policy.stop_points,
                "dir_threshold": self.policy.dir_threshold,
                "entry_style": self.policy.entry_style,
            },
            "position": asdict(self.position) if self.position else None,
            "last_tick": {
                "ts": _iso(self.last_tick.ts),
                "price": self.last_tick.price,
                "bid": self.last_tick.bid,
                "ask": self.last_tick.ask,
                "symbol": self.last_tick.symbol,
            } if self.last_tick else None,
        }

    def to_ledger(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "trades": [asdict(t) for t in self.closed],
            "events": list(self.events[-500:]),
            "working": [asdict(o) for o in self.working if o.status == "working"],
        }

    # ── internals ─────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self._pending_meta: dict[str, dict[str, Any]] = {}

    def _entry_limit_px(self, side: Side) -> float | None:
        t = self.last_tick
        if t is None:
            return None
        style = self.policy.entry_style
        if style == "join":
            if side == "long":
                return t.bid if t.bid is not None else t.price
            return t.ask if t.ask is not None else t.price
        if style == "cross":
            if side == "long":
                return t.ask if t.ask is not None else t.price
            return t.bid if t.bid is not None else t.price
        # last
        return float(t.price)

    def _try_fill(self, od: WorkingOrder, tick: MarketTick) -> FillEvent | None:
        px = float(tick.price)
        if od.kind == "entry_limit":
            if od.side == "long" and px <= od.limit_px:
                fill = min(px, od.limit_px)
                return FillEvent(od.order_id, od.kind, od.side, fill, tick.ts, reason="entry")
            if od.side == "short" and px >= od.limit_px:
                fill = max(px, od.limit_px)
                return FillEvent(od.order_id, od.kind, od.side, fill, tick.ts, reason="entry")
            return None
        if od.kind == "target_limit":
            # long target = sell limit; short target = buy limit
            if od.side == "long" and px >= od.limit_px:
                return FillEvent(
                    od.order_id, od.kind, od.side, max(px, od.limit_px), tick.ts, reason="target")
            if od.side == "short" and px <= od.limit_px:
                return FillEvent(
                    od.order_id, od.kind, od.side, min(px, od.limit_px), tick.ts, reason="target")
            return None
        if od.kind == "stop":
            # long stop = sell stop; short stop = buy stop
            if od.side == "long" and px <= od.limit_px:
                return FillEvent(od.order_id, od.kind, od.side, px, tick.ts, reason="stop")
            if od.side == "short" and px >= od.limit_px:
                return FillEvent(od.order_id, od.kind, od.side, px, tick.ts, reason="stop")
            return None
        return None

    def _on_fill(self, fe: FillEvent, tick: MarketTick) -> None:
        if fe.kind == "entry_limit":
            meta = self._pending_meta.pop(fe.order_id, {})
            slip = float(self.policy.slippage_points)
            sign = 1.0 if fe.side == "long" else -1.0
            entry_px = float(fe.fill_px) + slip * sign
            tid = uuid.uuid4().hex[:12]
            fe.trade_id = tid
            # arm OCO
            tgt = entry_px + self.policy.target_points * sign
            stp = entry_px - self.policy.stop_points * sign
            t_oid = uuid.uuid4().hex[:12]
            s_oid = uuid.uuid4().hex[:12]
            self.working.append(WorkingOrder(
                order_id=t_oid, kind="target_limit", side=fe.side, limit_px=float(tgt)))
            self.working.append(WorkingOrder(
                order_id=s_oid, kind="stop", side=fe.side, limit_px=float(stp)))
            self.position = Position(
                trade_id=tid, side=fe.side, entry_ts=fe.ts, entry_px=entry_px,
                qty=1, confidence=float(meta.get("confidence", 0.0)),
                signal_direction=str(meta.get("signal_direction", "")),
                target_order_id=t_oid, stop_order_id=s_oid,
            )
            self.events.append({
                "type": "fill_entry",
                "ts": _iso(fe.ts),
                "trade_id": tid,
                "side": fe.side,
                "fill_px": entry_px,
                "target_px": float(tgt),
                "stop_px": float(stp),
            })
            logger.info(
                "paper_fill_entry id=%s %s @ %.2f tgt=%.2f stop=%.2f",
                tid, fe.side, entry_px, tgt, stp,
            )
            return

        # exit fill (target or stop) — cancel sibling OCO
        if self.position is None:
            return
        pos = self.position
        self._cancel_oco(except_id=fe.order_id)
        slip = float(self.policy.slippage_points)
        sign = 1.0 if pos.side == "long" else -1.0
        exit_px = float(fe.fill_px) - slip * sign
        pnl = (exit_px - pos.entry_px) * sign - float(self.policy.cost_per_trade)
        hold = max(0.0, (fe.ts - pos.entry_ts).total_seconds())
        tr = PaperTrade(
            trade_id=pos.trade_id,
            side=pos.side,
            entry_ts=_iso(pos.entry_ts),
            exit_ts=_iso(fe.ts),
            entry_px=pos.entry_px,
            exit_px=exit_px,
            pnl_points=float(pnl),
            holding_seconds=hold,
            exit_reason=fe.reason or fe.kind,
            confidence=pos.confidence,
            objective_id=self.policy.objective_id,
            model_path=self.policy.model_path,
            entry_style=self.policy.entry_style,
        )
        self.closed.append(tr)
        self.position = None
        fe.trade_id = tr.trade_id
        self.events.append({
            "type": "fill_exit",
            "ts": _iso(fe.ts),
            "trade_id": tr.trade_id,
            "reason": tr.exit_reason,
            "exit_px": exit_px,
            "pnl": pnl,
        })
        logger.info(
            "paper_fill_exit id=%s reason=%s pnl=%.2f",
            tr.trade_id, tr.exit_reason, pnl,
        )

    def _cancel_oco(self, except_id: str | None = None) -> None:
        for od in self.working:
            if od.status != "working":
                continue
            if od.kind in ("target_limit", "stop") and od.order_id != except_id:
                od.status = "cancelled"
                self.events.append({
                    "type": "cancel_oco",
                    "order_id": od.order_id,
                    "kind": od.kind,
                })
        self.working = [o for o in self.working if o.status == "working"]

    def _expire_entries(self, tick: MarketTick) -> list[FillEvent]:
        out: list[FillEvent] = []
        keep: list[WorkingOrder] = []
        for od in self.working:
            if od.kind != "entry_limit" or od.status != "working":
                keep.append(od)
                continue
            meta = self._pending_meta.get(od.order_id) or {}
            placed = meta.get("placed_ts")
            if placed is not None:
                age = (tick.ts - placed).total_seconds()
                if age >= self.policy.entry_timeout_seconds:
                    od.status = "cancelled"
                    self.skipped["entry_expired"] += 1
                    self._pending_meta.pop(od.order_id, None)
                    self.events.append({
                        "type": "entry_expired",
                        "ts": _iso(tick.ts),
                        "order_id": od.order_id,
                        "limit_px": od.limit_px,
                    })
                    logger.info("paper_entry_expired id=%s limit=%.2f", od.order_id, od.limit_px)
                    continue
            keep.append(od)
        self.working = keep
        return out

    def _force_exit_reason(self, tick: MarketTick) -> str | None:
        if self.position is None:
            return None
        if self.policy.max_hold_seconds is not None:
            age = (tick.ts - self.position.entry_ts).total_seconds()
            if age >= self.policy.max_hold_seconds:
                return "time"
        if self.policy.eod_flat and self._past_trading_end(tick.ts):
            return "eod"
        return None

    def _force_close(self, tick: MarketTick, reason: str) -> list[FillEvent]:
        if self.position is None:
            return []
        pos = self.position
        self._cancel_oco()
        # cancel any leftover
        self.working = [o for o in self.working if o.status == "working"
                        and o.kind == "entry_limit"]
        slip = float(self.policy.slippage_points)
        sign = 1.0 if pos.side == "long" else -1.0
        exit_px = float(tick.price) - slip * sign
        pnl = (exit_px - pos.entry_px) * sign - float(self.policy.cost_per_trade)
        hold = max(0.0, (tick.ts - pos.entry_ts).total_seconds())
        tr = PaperTrade(
            trade_id=pos.trade_id, side=pos.side,
            entry_ts=_iso(pos.entry_ts), exit_ts=_iso(tick.ts),
            entry_px=pos.entry_px, exit_px=exit_px, pnl_points=float(pnl),
            holding_seconds=hold, exit_reason=reason,
            confidence=pos.confidence, objective_id=self.policy.objective_id,
            model_path=self.policy.model_path, entry_style=self.policy.entry_style,
        )
        self.closed.append(tr)
        self.position = None
        fe = FillEvent("force", "stop", pos.side, exit_px, tick.ts,
                       trade_id=tr.trade_id, reason=reason)
        self.events.append({
            "type": "force_exit", "ts": _iso(tick.ts), "reason": reason,
            "trade_id": tr.trade_id, "pnl": pnl,
        })
        logger.info("paper_force_exit id=%s reason=%s pnl=%.2f", tr.trade_id, reason, pnl)
        return [fe]

    def _past_trading_end(self, ts: datetime) -> bool:
        try:
            tz = ZoneInfo(self.policy.session_tz)
            local = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=timezone.utc).astimezone(tz)
            h, m = self.policy.trading_end.split(":")[:2]
            end_m = int(h) * 60 + int(m)
            return local.hour * 60 + local.minute >= end_m
        except Exception:
            return False

    def _before_trading_start(self, ts: datetime) -> bool:
        try:
            tz = ZoneInfo(self.policy.session_tz)
            local = ts.astimezone(tz) if ts.tzinfo else ts.replace(tzinfo=timezone.utc).astimezone(tz)
            h, m = self.policy.trading_start.split(":")[:2]
            start_m = int(h) * 60 + int(m)
            return local.hour * 60 + local.minute < start_m
        except Exception:
            return False


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()
