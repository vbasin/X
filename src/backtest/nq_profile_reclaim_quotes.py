"""Quote-event replay for the one-NQ confluent-profile reclaim policy.

Signals are emitted only after a right-labelled closed minute.  Entries, stops,
targets, and scheduled liquidations all fill at a later valid BBO event after a
fixed latency.  This is deliberately a conservative executable-price proxy;
TBBO does not prove queue priority, depth, or a broker fill.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    planned_risk_points: float = 20.0
    min_reward_r: float = 2.0
    latency_ms: int = 250
    entry_timeout_s: float = 5.0
    slippage_points: float = 0.25
    commission: float = 3.36
    max_entries: int = 2
    reentry_closed_loss_usd: float = -800.0
    max_quote_gap_s: float = 60.0

    def __post_init__(self) -> None:
        if self.planned_risk_points <= 0 or self.min_reward_r <= 0:
            raise ValueError("risk and reward multiples must be positive")
        if self.latency_ms < 0 or self.entry_timeout_s <= 0:
            raise ValueError("latency must be nonnegative and timeout positive")
        if self.slippage_points < 0 or self.commission < 0:
            raise ValueError("execution costs must be nonnegative")
        if self.max_entries < 1 or self.max_quote_gap_s <= 0:
            raise ValueError("max_entries and max_quote_gap_s must be positive")


_COLUMNS = [
    "decision_ts", "entry_ts", "exit_ts", "side", "entry_price", "exit_price",
    "stop_price", "target_price", "exit_reason", "gross_points", "net_usd",
    "planned_risk_usd", "mae_points", "mfe_points", "max_quote_gap_seconds", "valid",
]


def _valid_quotes(tape: pd.DataFrame) -> pd.DataFrame:
    required = {"bid_px", "ask_px", "bid_sz", "ask_sz"}
    if not required <= set(tape.columns):
        raise ValueError(f"tape missing BBO fields: {sorted(required-set(tape.columns))}")
    b = tape.bid_px.to_numpy(float)
    a = tape.ask_px.to_numpy(float)
    mask = np.isfinite(b) & np.isfinite(a) & (b > 0) & (a >= b)
    return tape.loc[mask, ["bid_px", "ask_px"]]


def _first_after(quotes: pd.DataFrame, when: pd.Timestamp) -> int | None:
    pos = int(quotes.index.searchsorted(when, side="left"))
    return pos if pos < len(quotes) else None


def _cash_end(ts: pd.Timestamp) -> pd.Timestamp:
    local = ts.tz_convert("America/Chicago")
    return (local.normalize() + pd.Timedelta(hours=15)).tz_convert("UTC")


def replay_session(tape: pd.DataFrame, signals: pd.DataFrame, *,
                   config: ExecutionConfig | None = None) -> pd.DataFrame:
    """Replay sorted profile-reclaim signals against one day's valid BBO events.

    Signals need ``decision_ts``, ``side`` and pre-open ``target_price``.  The
    20-point stop is set from the *actual* entry, so planned risk remains known
    only after a valid next quote exists.  Invalid entries/exits are persisted;
    no row is silently discarded.
    """
    config = config or ExecutionConfig()
    if not isinstance(tape.index, pd.DatetimeIndex) or tape.index.tz is None:
        raise ValueError("tape needs a timezone-aware DatetimeIndex")
    if tape.index.hasnans or not tape.index.is_monotonic_increasing:
        raise ValueError("tape timestamps must be sorted and finite")
    if signals.empty:
        return pd.DataFrame(columns=_COLUMNS)
    required = {"decision_ts", "side", "target_price"}
    if missing := required - set(signals.columns):
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    quotes = _valid_quotes(tape)
    if quotes.empty:
        raise ValueError("session has no valid BBO events")
    latency = pd.Timedelta(milliseconds=config.latency_ms)
    timeout = pd.Timedelta(seconds=config.entry_timeout_s)
    ordered = signals.sort_values("decision_ts", kind="mergesort")
    results: list[dict] = []
    closed_net = 0.0
    entries = 0
    occupied_until = pd.Timestamp.min.tz_localize("UTC")
    for signal in ordered.itertuples(index=False):
        decision = pd.Timestamp(signal.decision_ts)
        if decision.tz is None:
            raise ValueError("decision timestamps must be timezone-aware")
        side = int(signal.side)
        if side not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        if (entries >= config.max_entries or closed_net <= config.reentry_closed_loss_usd
                or decision < occupied_until):
            continue
        ready = decision + latency
        entry_i = _first_after(quotes, ready)
        if entry_i is None or quotes.index[entry_i] > decision + timeout:
            results.append(dict(decision_ts=decision, entry_ts=pd.NaT, exit_ts=pd.NaT,
                side=side, entry_price=np.nan, exit_price=np.nan, stop_price=np.nan,
                target_price=float(signal.target_price), exit_reason="no_entry_within_5s",
                gross_points=np.nan, net_usd=np.nan, planned_risk_usd=np.nan,
                mae_points=np.nan, mfe_points=np.nan, max_quote_gap_seconds=np.nan, valid=False))
            continue
        entry_ts = quotes.index[entry_i]
        entry = (float(quotes.ask_px.iloc[entry_i]) + config.slippage_points if side == 1
                 else float(quotes.bid_px.iloc[entry_i]) - config.slippage_points)
        target = float(signal.target_price)
        stop = entry - side * config.planned_risk_points
        reward = side * (target - entry)
        if not np.isfinite(target) or reward < config.min_reward_r * config.planned_risk_points:
            # The quote is observed before submitting the order.  Declining a
            # now-insufficient pre-open runway is not an entry or an unresolved
            # economic outcome; it must never inflate a trade count or hide a
            # missing exit.
            continue
        entries += 1
        end = _cash_end(entry_ts)
        after = quotes.iloc[entry_i + 1:]
        after = after.loc[after.index <= end]
        trigger_ts: pd.Timestamp | None = None
        reason = "session_close"
        for ts, quote in after.iterrows():
            mark = float(quote.bid_px) if side == 1 else float(quote.ask_px)
            if side * (mark - stop) <= 0:
                trigger_ts, reason = ts, "stop"
                break
            if side * (mark - target) >= 0:
                trigger_ts, reason = ts, "target"
                break
        if trigger_ts is None:
            trigger_ts = end
        exit_i = _first_after(quotes, trigger_ts + latency)
        if exit_i is None or quotes.index[exit_i] > end + pd.Timedelta(minutes=15):
            results.append(dict(decision_ts=decision, entry_ts=entry_ts, exit_ts=pd.NaT,
                side=side, entry_price=entry, exit_price=np.nan, stop_price=stop,
                target_price=target, exit_reason=f"missing_{reason}_exit", gross_points=np.nan,
                net_usd=np.nan, planned_risk_usd=np.nan, mae_points=np.nan, mfe_points=np.nan,
                max_quote_gap_seconds=np.nan, valid=False))
            occupied_until = end
            continue
        exit_ts = quotes.index[exit_i]
        exit_quote = quotes.iloc[exit_i]
        exit_price = (float(exit_quote.bid_px) - config.slippage_points if side == 1
                      else float(exit_quote.ask_px) + config.slippage_points)
        path = quotes.iloc[entry_i:exit_i + 1]
        marks = (path.bid_px.to_numpy(float) if side == 1 else path.ask_px.to_numpy(float))
        moves = side * (marks - entry)
        gross = side * (exit_price - entry)
        net = gross * 20.0 - config.commission
        gaps = np.diff(path.index.asi8) / 1e9
        row = dict(decision_ts=decision, entry_ts=entry_ts, exit_ts=exit_ts, side=side,
            entry_price=entry, exit_price=exit_price, stop_price=stop, target_price=target,
            exit_reason=reason, gross_points=gross, net_usd=net,
            planned_risk_usd=config.planned_risk_points * 20.0 + 2 * config.slippage_points * 20.0 + config.commission,
            mae_points=float(max(0.0, -moves.min())), mfe_points=float(max(0.0, moves.max())),
            max_quote_gap_seconds=float(gaps.max()) if len(gaps) else 0.0,
            valid=bool((gaps.max() if len(gaps) else 0.0) <= config.max_quote_gap_s))
        if not row["valid"]:
            row["exit_reason"] = "quote_gap_exceeds_limit"
            row["net_usd"] = np.nan
            row["gross_points"] = np.nan
            # An unknown interval cannot safely consume a second risk budget.
            closed_net = -np.inf
        else:
            closed_net += net
        results.append(row)
        occupied_until = exit_ts
    return pd.DataFrame(results, columns=_COLUMNS)
