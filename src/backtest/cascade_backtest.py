"""WebUI backtest adapter for the MNQ auction cascade OOF (``target_scores.parquet``).

The three-stage cascade (Stage 1 move detector → fixed 0.45 R stop → Stage 3 target
selector) writes one row per (decision bar, side, candidate target bound). This module
turns that OOF file into the same ``{config, summary, equity_curve, confusion_matrix,
trades, skipped}`` payload the existing backtest UI already renders — so the blotter,
PnL summary and chart trade markers work without any frontend changes to the rendering
path.

The evaluator's book from ``scripts/features/eval_target_model.py`` is applied here:

1. Stage 1 gate: literal ``p_move >= p_move_min``.
2. Stage 3 gate: literal ``p_reach >= p_reach_min``.
3. One candidate per decision bar: the surviving row with the highest ``p_reach``
   (``_best_per_bar``).
4. ``max_positions`` chronological book with a ≥1-second minimum busy interval and
   stable sort — the exact rule the evaluator uses (``_one_position``), generalised
   from 1 slot to *N* slots. A slot free set of size *N* means the earliest expiring
   position determines when the next candidate is admitted.

Trade outcomes follow the evaluator's pessimistic P&L accounting (``_expectancy_r``)
but with honest exit-time and exit-reason accounting when the label file carries
``t_stop`` (persisted since ADR-127 follow-up, 2026-09-15):

- ``y_reach == 1`` — target reached: exit at ``tgt_level`` after ``t_reach`` seconds,
  ``pnl_points = +tgt_dist_r * r_value``, ``exit_reason = "target"``.
- ``y_reach == 0`` and ``t_stop >= 0`` — stop pierced before the target: exit at
  ``stop_px`` after ``t_stop`` seconds, ``pnl_points = -stp_dist_r * r_value``,
  ``exit_reason = "stop"``.
- ``y_reach == 0`` and ``t_stop < 0`` (or column absent) — neither target nor stop hit
  inside the label horizon: exit at ``stop_px`` after the horizon,
  ``pnl_points = -stp_dist_r * r_value``, ``exit_reason = "timeout"``. Charging the
  full stop distance on a censored row is the evaluator's pessimistic convention
  (see ``eval_target_model.py``): treating timeouts as break-even is how a backtest
  flatters itself.

The same ``t_stop`` is used by ``_n_position_book`` so a stop-out that took three
seconds frees its slot after three seconds instead of the full horizon.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training import cascade_policy as policy

logger = logging.getLogger("vtech.backtest.cascade_backtest")

_BAR_SECONDS = 5
_HORIZON_SECONDS = 1800  # matches MoveConfig.horizon_seconds


def is_cascade_oof(path: str | Path) -> bool:
    """True when *path* is a parquet file with the cascade OOF schema."""
    p = Path(path)
    if not p.is_file() or p.suffix != ".parquet":
        return False
    try:
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(p).names)
    except Exception:
        return False
    return {"p_move", "p_reach", "tgt_dist_r", "stp_dist_r",
            "r_value", "y_reach", "ts", "side"}.issubset(cols)


def _best_per_bar(g: pd.DataFrame) -> pd.DataFrame:
    if g.empty:
        return g
    return g.loc[g.groupby("ts")["p_reach"].idxmax()]


def _hold_seconds(y_reach: np.ndarray, t_reach: np.ndarray,
                  t_stop: np.ndarray, horizon: int,
                  t_fill: np.ndarray | None = None) -> np.ndarray:
    """True seconds the slot is busy for each row.

    ``t_stop`` is -1 for rows where the stop was never pierced within the horizon
    (timeout) and also -1 for the whole column when the label file predates the
    ``t_stop`` addition — in both cases we fall back to the full horizon.

    ``t_reach``/``t_stop`` are measured from the fill, so a resting entry adds
    its wait: capital is committed from the decision bar (ADR-126).
    """
    out = np.where(y_reach == 1, np.maximum(t_reach, 1), horizon).astype("int64")
    stopped = (y_reach == 0) & (t_stop >= 0)
    out = np.where(stopped, np.maximum(t_stop, 1), out)
    if t_fill is not None:
        out = out + np.maximum(t_fill, 0).astype("int64")
    return np.maximum(out, 1)


def _fill_seconds(g: pd.DataFrame) -> np.ndarray | None:
    if "t_fill" not in g.columns:
        return None
    return g["t_fill"].to_numpy()


def _n_position_book(g: pd.DataFrame, horizon: int, max_positions: int
                     ) -> tuple[pd.DataFrame, int]:
    """Chronological *N*-slot book with stable ordering and ≥1-second busy floor.

    Returns the accepted rows and the count of candidates rejected because every slot
    was busy (surfaced as ``skipped.in_position`` for the UI, matching the existing
    predictions engine's contract).
    """
    if g.empty:
        return g, 0
    g = g.sort_values("ts", kind="mergesort")
    t_reach = g["t_reach"].to_numpy()
    y_reach = g["y_reach"].to_numpy()
    t_stop = (g["t_stop"].to_numpy() if "t_stop" in g.columns
              else np.full(len(g), -1, dtype="int64"))
    busy = _hold_seconds(y_reach, t_reach, t_stop, horizon, _fill_seconds(g))
    ts = g["ts"].values.astype("datetime64[ns]")

    slots = np.full(int(max(1, max_positions)),
                    np.datetime64("1970-01-01", "ns"), dtype="datetime64[ns]")
    keep = np.zeros(len(g), dtype=bool)
    skipped_slots_full = 0
    for i in range(len(g)):
        earliest = int(np.argmin(slots))
        if ts[i] >= slots[earliest]:
            keep[i] = True
            slots[earliest] = ts[i] + np.timedelta64(int(busy[i]), "s")
        else:
            skipped_slots_full += 1
    return g[keep], skipped_slots_full


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).isoformat()


def _apply_gates(df: pd.DataFrame,
                 *, p_reach_min: float, p_move_min: float, min_tgt_r: float,
                 confidence_gate: bool, mv_gate: bool,
                 sides: list[str] | None,
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply literal Stage 1/3 score floors, side filter and target-size gate."""
    notes: dict[str, Any] = {
        "p_reach_min": float(p_reach_min),
        "p_move_min": float(p_move_min),
        "min_tgt_r": float(min_tgt_r),
        "n_before": len(df),
    }

    if sides:
        side_map = {"up": 1, "long": 1, "down": -1, "short": -1}
        side_ints = {side_map[s] for s in sides if s in side_map}
        if side_ints:
            df = df[df["side"].isin(list(side_ints))]

    if confidence_gate:
        df = df[df["p_reach"] >= float(p_reach_min)]

    if mv_gate:
        df = df[df["p_move"] >= float(p_move_min)]

    if min_tgt_r > 0:
        df = df[df["tgt_dist_r"] >= float(min_tgt_r)]

    notes["n_after"] = len(df)
    return df, notes


def _kpis(trades: pd.DataFrame, point_value: float, horizon: int
          ) -> dict[str, Any]:
    if trades.empty:
        return {
            "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "net_pnl": 0.0, "net_value": 0.0,
            "win_rate": None, "expectancy": None, "profit_factor": None,
            "max_drawdown": None, "sharpe_daily": None, "sharpe_trade": None,
            "avg_win": None, "avg_loss": None,
            "avg_holding_bars": None, "avg_mae": None, "avg_mfe": None,
            "exposure_frac": None, "n_sessions_traded": 0,
            "by_exit_reason": {}, "point_value": float(point_value),
        }
    pnl = trades["pnl_points"].to_numpy(dtype="float64")
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    daily = trades.groupby("session_date")["pnl_points"].sum()
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    drawdown = float((peak - equity).max()) if len(equity) else 0.0

    hold = trades["holding_bars"].to_numpy(dtype="float64")
    exposure = float(hold.sum() * _BAR_SECONDS
                     / max(1, len(daily) * horizon)) if len(daily) else None
    by_exit = trades["exit_reason"].value_counts().to_dict()

    sharpe_daily = None
    if len(daily) > 1 and daily.std() > 0:
        sharpe_daily = float(daily.mean() / daily.std() * math.sqrt(252))
    sharpe_trade = None
    if len(pnl) > 1 and pnl.std() > 0:
        sharpe_trade = float(pnl.mean() / pnl.std() * math.sqrt(len(pnl)))

    return {
        "n_trades": int(len(trades)),
        "n_wins": int((pnl > 0).sum()),
        "n_losses": int((pnl <= 0).sum()),
        "gross_profit": float(wins.sum()),
        "gross_loss": float(losses.sum()),
        "net_pnl": float(pnl.sum()),
        "net_value": float(pnl.sum() * point_value),
        "win_rate": float((pnl > 0).mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": (float(wins.sum() / -losses.sum())
                          if losses.size and losses.sum() < 0 else None),
        "max_drawdown": -drawdown if drawdown > 0 else 0.0,
        "sharpe_daily": sharpe_daily,
        "sharpe_trade": sharpe_trade,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "avg_holding_bars": float(hold.mean()),
        "avg_mae": float(trades["mae_points"].mean()),
        "avg_mfe": float(trades["mfe_points"].mean()),
        "exposure_frac": exposure,
        "n_sessions_traded": int(len(daily)),
        "by_exit_reason": {str(k): int(v) for k, v in by_exit.items()},
        "point_value": float(point_value),
    }


def run_cascade_inmemory(
    *,
    source: str,
    p_reach_min: float = 0.75,
    p_move_min: float = 0.5,
    min_tgt_r: float = policy.MIN_TGT_R,
    tau: float = 0.75,
    mv: float = 0.5,
    confidence_gate: bool = True,
    mv_gate: bool = True,
    max_positions: int = 1,
    start: str | None = None,
    end: str | None = None,
    sides: list[str] | None = None,
    objective_id: str | None = None,
    horizon: int = _HORIZON_SECONDS,
    point_value: float = 2.0,           # MNQ-equivalent default sizing
    economic_contract_unit: str = "MNQ_equivalent",
) -> dict[str, Any]:
    """Backtest the cascade OOF at literal ``(p_reach_min, p_move_min)`` floors.

    ``point_value`` defaults to $2/pt (MNQ-equivalent, i.e. 0.1x the NQ $20/pt
    multiplier per config/execution_costs.yaml). The cascade is MNQ-equivalent-sized
    only today (ADR-127) — NQ-sized would be TEN times the P&L for the same points,
    not four (a previous version of this docstring understated the ratio).
    ``economic_contract_unit`` names the SIZING profile (config/execution_costs.yaml),
    not the underlying market: the underlying is always NQ (src.common.instruments,
    EconomicAssumptions.underlying). Renamed from ``instrument``, which wrongly
    implied the trade was in an actual MNQ contract on MNQ market data.

    ``min_tgt_r`` is a runtime filter on target distance in R units; the label
    build (ADR-130) keeps every candidate target so the threshold can be swept
    from the WebUI without rebuilding the parquet.
    """
    p = Path(source)
    if not is_cascade_oof(p):
        raise ValueError(f"not a cascade OOF (missing p_move/p_reach columns): {p}")

    p_reach_min = float(p_reach_min)
    p_move_min = float(p_move_min)
    min_tgt_r = float(min_tgt_r)
    if not 0.0 <= p_reach_min <= 1.0:
        raise ValueError("p_reach_min must be in [0, 1]")
    if not 0.0 <= p_move_min <= 1.0:
        raise ValueError("p_move_min must be in [0, 1]")
    if min_tgt_r < 0.0:
        raise ValueError("min_tgt_r must be >= 0")
    max_positions = int(max(1, max_positions))
    horizon = int(max(1, horizon))

    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["session"] = pd.to_datetime(df["session"])
    df = df.sort_values("ts").reset_index(drop=True)
    n_raw = len(df)

    if start:
        df = df[df["session"] >= pd.Timestamp(start)]
    if end:
        df = df[df["session"] <= pd.Timestamp(end)]

    sessions_total = int(df["session"].nunique())
    if df.empty:
        return _empty_payload(source, p_reach_min, p_move_min, min_tgt_r,
                      confidence_gate, mv_gate,
                              max_positions, start, end, sides,
                              point_value, economic_contract_unit,
                              n_raw=n_raw, note="window is empty")

    df_gated, notes = _apply_gates(df, p_reach_min=p_reach_min,
                                   p_move_min=p_move_min,
                                   min_tgt_r=min_tgt_r,
                                   confidence_gate=confidence_gate,
                                   mv_gate=mv_gate, sides=sides)
    n_pred_gated = len(df_gated)

    per_bar = _best_per_bar(df_gated)
    booked, skipped_slots_full = _n_position_book(per_bar, horizon, max_positions)

    # ── Build the trade ledger ────────────────────────────────────────────────
    trades = booked.copy()
    trades["side_str"] = np.where(trades["side"].to_numpy() == 1, "long", "short")
    y_reach = trades["y_reach"].to_numpy()
    t_reach = trades["t_reach"].to_numpy()
    t_stop = (trades["t_stop"].to_numpy() if "t_stop" in trades.columns
              else np.full(len(trades), -1, dtype="int64"))
    tgt_dist_r = trades["tgt_dist_r"].to_numpy(dtype="float64")
    stp_dist_r = trades["stp_dist_r"].to_numpy(dtype="float64")
    rv = trades["r_value"].to_numpy(dtype="float64")

    pnl_points = np.where(y_reach == 1, tgt_dist_r * rv, -stp_dist_r * rv)
    stopped = (y_reach == 0) & (t_stop >= 0)
    exit_reason = np.where(y_reach == 1, "target",
                           np.where(stopped, "stop", "timeout"))
    hold_seconds = _hold_seconds(y_reach, t_reach, t_stop, horizon,
                                 _fill_seconds(trades))
    holding_bars = np.maximum(1, np.ceil(hold_seconds / _BAR_SECONDS)).astype("int64")

    entry_px = trades["price"].to_numpy(dtype="float64")
    exit_px = np.where(y_reach == 1,
                       trades["tgt_level"].to_numpy(dtype="float64"),
                       trades["stop_px"].to_numpy(dtype="float64"))
    entry_ts = pd.DatetimeIndex(trades["ts"])
    exit_ts = entry_ts + pd.to_timedelta(hold_seconds, unit="s")

    mae_pts = np.where(y_reach == 1, 0.0, stp_dist_r * rv)
    mfe_pts = np.where(y_reach == 1, tgt_dist_r * rv, 0.0)

    ledger = pd.DataFrame({
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "side": trades["side_str"].to_numpy(),
        "entry_px": entry_px,
        "exit_px": exit_px,
        "pnl_points": pnl_points,
        "exit_reason": exit_reason,
        "holding_bars": holding_bars,
        "mae_points": mae_pts,
        "mfe_points": mfe_pts,
        "session_date": pd.to_datetime(trades["session"]).dt.date.astype(str).to_numpy(),
        "p_move": trades["p_move"].to_numpy(),
        "p_reach": trades["p_reach"].to_numpy(),
        "tgt_level": trades["tgt_level"].to_numpy(),
        "stop_px": trades["stop_px"].to_numpy(),
        "tgt_dist_r": tgt_dist_r,
        "stp_dist_r": stp_dist_r,
        "r_value": rv,
    })

    cum = ledger["pnl_points"].cumsum().to_numpy()
    kpi = _kpis(ledger, point_value, horizon)

    # ── Payload assembly (mirrors src/backtest/runner._backtest_payload shape) ─
    lo = str(df["session"].min().date()) if not df.empty else start
    hi = str(df["session"].max().date()) if not df.empty else end
    config = {
        "strategy_id": "mnq_auction_cascade",
        "objective_id": objective_id or "",
        "source": str(p),
        "source_kind": "cascade_oof",
        "run_label": "MNQ Auction Cascade",
        "tau": p_reach_min,
        "confidence_gate": bool(confidence_gate),
        "p_reach_min": p_reach_min,
        "mv": p_move_min,
        "mv_gate": bool(mv_gate),
        "p_move_min": p_move_min,
        "sides": sorted(sides) if sides else ["down", "up"],
        "max_concurrent_positions": max_positions,
        "min_tgt_r": min_tgt_r,
        "target_points": None,       # per-trade — no single value
        "stop_points": None,
        "max_holding": f"{horizon}s",
        "point_value": float(point_value),
        "economic_contract_unit": economic_contract_unit,
        "window": {"start": lo, "end": hi, "n_sessions": sessions_total,
                   "n_bars": n_raw},
        "n_pred_directional": n_raw,
        "n_pred_gated": int(n_pred_gated),
        "n_features": None,
        "dir_threshold": None,
        "weights_note": (
            f"Cascade OOF replay (ADR-130 target-adaptive stops) — Stage 1 gate: "
            f"p_move >= {p_move_min:g} (gate={'on' if mv_gate else 'off'}); "
            f"Stage 3 gate: p_reach >= {p_reach_min:g} "
            f"(gate={'on' if confidence_gate else 'off'}); "
            f"target-size gate: tgt_dist_r >= {min_tgt_r:g}. "
            f"Stop = min(0.5 * tgt_dist_r, 0.25) R per row (ADR-130). "
            f"Slots={max_positions}; horizon={horizon}s."
        ),
        "bar_seconds": _BAR_SECONDS,
    }

    if ledger.empty:
        return {"config": config,
                "summary": kpi,
                "equity_curve": [],
                "confusion_matrix": {"labels": ["down", "flat", "up"],
                                     "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
                "trades": [],
                "skipped": {"in_position": int(skipped_slots_full),
                            "conflicting_signal": 0, "cooldown": 0,
                            "day_cap": 0, "outside_trading_window": 0,
                            "last_bar": 0},
                "archetypes": []}

    equity_curve = [{"t": _iso(ledger["entry_ts"].iloc[0]), "equity": 0.0}]
    equity_curve += [{"t": _iso(ledger["exit_ts"].iloc[j]), "equity": float(cum[j])}
                     for j in range(len(ledger))]

    labels = ["down", "flat", "up"]
    lab_ix = {l: i for i, l in enumerate(labels)}
    cm = [[0] * 3 for _ in range(3)]
    pred_side = ledger["side"].map({"long": "up", "short": "down"}).to_numpy()
    true_side = np.where(y_reach == 1, pred_side, "flat")  # target = correct call
    for t, p_ in zip(true_side, pred_side):
        if t in lab_ix and p_ in lab_ix:
            cm[lab_ix[t]][lab_ix[p_]] += 1

    trades_out = [{
        "entry_ts": _iso(ledger["entry_ts"].iloc[j]),
        "exit_ts": _iso(ledger["exit_ts"].iloc[j]),
        "side": str(ledger["side"].iloc[j]),
        "entry_px": float(ledger["entry_px"].iloc[j]),
        "exit_px": float(ledger["exit_px"].iloc[j]),
        "pnl_points": float(ledger["pnl_points"].iloc[j]),
        "cum_pnl": float(cum[j]),
        "mae_points": float(ledger["mae_points"].iloc[j]),
        "mfe_points": float(ledger["mfe_points"].iloc[j]),
        "holding_bars": int(ledger["holding_bars"].iloc[j]),
        "exit_reason": str(ledger["exit_reason"].iloc[j]),
        "n_present": None,
        "n_total": None,
        "confidence": float(ledger["p_reach"].iloc[j]),
        "archetype": None,
        "archetype_label": None,
        "archetype_train_hit_rate": None,
    } for j in range(len(ledger))]

    return {"config": config,
            "summary": kpi,
            "equity_curve": equity_curve,
            "confusion_matrix": {"labels": labels,
                                 "rows_true_cols_pred": cm},
            "trades": trades_out,
            "skipped": {"in_position": int(skipped_slots_full),
                        "conflicting_signal": 0, "cooldown": 0,
                        "day_cap": 0, "outside_trading_window": 0,
                        "last_bar": 0},
            "archetypes": []}


def _empty_payload(source, p_reach_min, p_move_min, min_tgt_r,
                   confidence_gate, mv_gate,
                   max_positions, start, end, sides,
                   point_value, economic_contract_unit, *, n_raw, note):
    return {"config": {
                "strategy_id": "mnq_auction_cascade",
                "objective_id": "",
                "source": str(source),
                "source_kind": "cascade_oof",
                "run_label": "MNQ Auction Cascade",
                "tau": float(p_reach_min),
                "p_reach_min": float(p_reach_min),
                "confidence_gate": bool(confidence_gate),
                "mv": float(p_move_min), "p_move_min": float(p_move_min),
                "mv_gate": bool(mv_gate),
                "min_tgt_r": float(min_tgt_r),
                "sides": sorted(sides) if sides else ["down", "up"],
                "max_concurrent_positions": int(max_positions),
                "target_points": None, "stop_points": None,
                "max_holding": f"{_HORIZON_SECONDS}s",
                "point_value": float(point_value),
                "economic_contract_unit": economic_contract_unit,
                "window": {"start": start, "end": end, "n_sessions": 0,
                           "n_bars": int(n_raw)},
                "n_pred_directional": int(n_raw), "n_pred_gated": 0,
                "n_features": None, "dir_threshold": None,
                "weights_note": note, "bar_seconds": _BAR_SECONDS,
            },
            "summary": _kpis(pd.DataFrame(), point_value, _HORIZON_SECONDS),
            "equity_curve": [],
            "confusion_matrix": {"labels": ["down", "flat", "up"],
                                 "rows_true_cols_pred": [[0] * 3 for _ in range(3)]},
            "trades": [],
            "skipped": {"in_position": 0, "conflicting_signal": 0, "cooldown": 0,
                        "day_cap": 0, "outside_trading_window": 0, "last_bar": 0},
            "archetypes": []}
