"""Causal tape features for the experimental NQ profile-reclaim policy.

This module deliberately uses the exact tick profile from closed *prior* tape
sessions and right-labelled one-minute current-session bars.  It does not reuse
legacy left-labelled CVD or the mixed-contract large-print cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.features.cash_session_profile import Bound, LevelConfig, build_bounds
from src.common.contract_calendar import active_contract
from src.features.session_profile import history_before

_CT = "America/Chicago"


@dataclass(frozen=True)
class ReclaimConfig:
    """Fixed observable event conditions; all prices are NQ index points."""

    min_bound_members: int = 2
    sweep_points: float = 1.0
    large_print_size: int = 20
    min_reclaim_points: float = 2.0
    delta_fraction: float = 0.10
    lookback_sessions: int = 20
    min_target_points: float = 40.0

    def __post_init__(self) -> None:
        if self.min_bound_members < 1 or self.large_print_size < 1:
            raise ValueError("minimum members and print size must be positive")
        if self.sweep_points <= 0 or self.min_reclaim_points <= 0:
            raise ValueError("sweep and reclaim distances must be positive")
        if self.min_target_points <= 0:
            raise ValueError("min_target_points must be positive")
        if not 0 < self.delta_fraction < 1:
            raise ValueError("delta_fraction must be in (0, 1)")


def _sign(side: pd.Series) -> np.ndarray:
    """Normalise both tape conventions (numeric +/-1 and B/A) to signed flow."""
    raw = side.to_numpy()
    if np.issubdtype(raw.dtype, np.number):
        return np.sign(raw.astype("float64"))
    text = pd.Series(raw).astype(str).str.upper()
    return np.where(text.eq("B"), 1.0, np.where(text.eq("A"), -1.0, 0.0))


def cash_minute_state(tape: pd.DataFrame, *, large_print_size: int) -> pd.DataFrame:
    """Return fully observed, right-labelled cash-minute tape state.

    A row stamped ``T`` contains prints in ``[T-1 minute, T)`` and is therefore
    available only at ``T``.  Invalid/crossed BBOs are excluded from depth
    summaries but never removed from the actual price/volume bar.
    """
    if not isinstance(tape.index, pd.DatetimeIndex) or tape.index.tz is None:
        raise ValueError("tape requires a timezone-aware DatetimeIndex")
    required = {"price", "size", "side", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    missing = required - set(tape.columns)
    if missing:
        raise ValueError(f"tape missing columns: {sorted(missing)}")
    if not tape.index.is_monotonic_increasing:
        tape = tape.sort_index(kind="mergesort")
    local = tape.index.tz_convert(_CT)
    minute = local.hour * 60 + local.minute
    cash = tape.loc[(minute >= 8 * 60 + 30) & (minute < 15 * 60)].copy()
    if cash.empty:
        return pd.DataFrame()
    cash["_bucket"] = cash.index.floor("min")
    cash["_sign"] = _sign(cash["side"])
    cash["_signed"] = cash["size"].to_numpy(float) * cash["_sign"].to_numpy(float)
    big = cash["size"].to_numpy(float) >= float(large_print_size)
    cash["_big_buy"] = np.where(big & (cash["_sign"].to_numpy() > 0), cash["size"], 0.0)
    cash["_big_sell"] = np.where(big & (cash["_sign"].to_numpy() < 0), cash["size"], 0.0)
    valid_bbo = (
        np.isfinite(cash["bid_px"]) & np.isfinite(cash["ask_px"])
        & np.isfinite(cash["bid_sz"]) & np.isfinite(cash["ask_sz"])
        & (cash["bid_px"] > 0) & (cash["ask_px"] >= cash["bid_px"])
        & (cash["ask_sz"] >= 0) & (cash["bid_sz"] >= 0)
    )
    cash["_bid_depth"] = cash["bid_sz"].where(valid_bbo)
    cash["_ask_depth"] = cash["ask_sz"].where(valid_bbo)
    grouped = cash.groupby("_bucket", sort=True)
    out = pd.DataFrame({
        "open": grouped["price"].first(), "high": grouped["price"].max(),
        "low": grouped["price"].min(), "close": grouped["price"].last(),
        "volume": grouped["size"].sum().astype(float),
        "delta": grouped["_signed"].sum(),
        "big_buy": grouped["_big_buy"].sum(),
        "big_sell": grouped["_big_sell"].sum(),
        "bid_depth": grouped["_bid_depth"].mean(),
        "ask_depth": grouped["_ask_depth"].mean(),
        "last_print_ts": grouped.apply(lambda x: x.index[-1], include_groups=False),
    })
    out.index = out.index + pd.Timedelta(minutes=1)
    out.index.name = "bar_close_ts"
    numerator = (out["close"] * out["volume"]).cumsum()
    out["vwap"] = numerator / out["volume"].cumsum().replace(0.0, np.nan)
    return out


def prior_bounds(levels: pd.DataFrame, session: date, cfg: ReclaimConfig) -> list[Bound]:
    """Build today's bounds from strictly prior levels of today's lead contract.

    The tape contains unadjusted lead outright prices.  Filtering to the active
    contract avoids silently carrying an absolute profile level across a roll.
    This intentionally leaves a post-roll warm-up interval with too little
    history rather than synthesising a roll adjustment.
    """
    current_contract = active_contract("NQ", session)
    candidate = levels.copy()
    candidate["_contract"] = candidate["session"].map(
        lambda value: active_contract("NQ", pd.Timestamp(value).date())
    )
    candidate = candidate[candidate["_contract"].eq(current_contract)].drop(columns="_contract")
    history = history_before(candidate, session, cfg.lookback_sessions)
    if len(history) < cfg.lookback_sessions:
        return []
    return build_bounds(history, LevelConfig(lookback_sessions=cfg.lookback_sessions))


def _target(bounds: list[Bound], *, side: int, entry_reference: float,
            min_members: int) -> float | None:
    strong = [b for b in bounds if b.n_levels >= min_members]
    if side == -1:
        lower = [b.hi for b in strong if b.hi < entry_reference]
        return max(lower) if lower else None
    higher = [b.lo for b in strong if b.lo > entry_reference]
    return min(higher) if higher else None


def reclaim_signals(minutes: pd.DataFrame, bounds: list[Bound], cfg: ReclaimConfig) -> pd.DataFrame:
    """Find causal failed-auction signals from completed minute rows.

    Each bound can emit once in each direction per session.  A returned
    ``decision_ts`` is the close of the final required minute; execution must
    occur strictly later in an independently observed BBO event.
    """
    columns = ["decision_ts", "side", "bound_lo", "bound_hi", "bound_members",
               "bound_strength", "target_price", "sweep_high_low", "reclaim_points",
               "delta_fraction", "big_outward", "vwap"]
    if len(minutes) < 3 or not bounds:
        return pd.DataFrame(columns=columns)
    used: set[tuple[int, int]] = set()
    records: list[dict] = []
    for i in range(2, len(minutes)):
        window = minutes.iloc[i - 2:i + 1]
        if (window.index.to_series().diff().iloc[1:] != pd.Timedelta(minutes=1)).any():
            continue
        now = window.iloc[-1]
        vol = float(window.volume.sum())
        if not np.isfinite(now.vwap) or vol <= 0:
            continue
        delta = float(window.delta.sum())
        upper = float(window.high.max())
        lower = float(window.low.min())
        for j, bound in enumerate(bounds):
            if bound.n_levels < cfg.min_bound_members:
                continue
            # An upper excursion that re-enters the zone/below it: responsive
            # sellers only act after the completed reclaim is observable.
            if (j, -1) not in used and float(window.close.iloc[0]) <= bound.lo:
                reclaim = upper - float(now.close)
                outward = float(window.big_buy.sum())
                if (upper >= bound.hi + cfg.sweep_points and now.close < bound.hi
                        and now.close < now.vwap and reclaim >= cfg.min_reclaim_points
                        and delta >= cfg.delta_fraction * vol and outward >= cfg.large_print_size):
                    target = _target(bounds, side=-1, entry_reference=float(now.close),
                                     min_members=cfg.min_bound_members)
                    if target is not None and float(now.close) - target >= cfg.min_target_points:
                        records.append(dict(decision_ts=window.index[-1], side=-1,
                            bound_lo=bound.lo, bound_hi=bound.hi, bound_members=bound.n_levels,
                            bound_strength=bound.strength, target_price=target,
                            sweep_high_low=upper, reclaim_points=reclaim,
                            delta_fraction=delta / vol, big_outward=outward, vwap=float(now.vwap)))
                        used.add((j, -1))
            # Mirror lower excursion.
            if (j, 1) not in used and float(window.close.iloc[0]) >= bound.hi:
                reclaim = float(now.close) - lower
                outward = float(window.big_sell.sum())
                if (lower <= bound.lo - cfg.sweep_points and now.close > bound.lo
                        and now.close > now.vwap and reclaim >= cfg.min_reclaim_points
                        and delta <= -cfg.delta_fraction * vol and outward >= cfg.large_print_size):
                    target = _target(bounds, side=1, entry_reference=float(now.close),
                                     min_members=cfg.min_bound_members)
                    if target is not None and target - float(now.close) >= cfg.min_target_points:
                        records.append(dict(decision_ts=window.index[-1], side=1,
                            bound_lo=bound.lo, bound_hi=bound.hi, bound_members=bound.n_levels,
                            bound_strength=bound.strength, target_price=target,
                            sweep_high_low=lower, reclaim_points=reclaim,
                            delta_fraction=delta / vol, big_outward=outward, vwap=float(now.vwap)))
                        used.add((j, 1))
    return pd.DataFrame(records, columns=columns)
