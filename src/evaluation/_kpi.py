"""Trade-KPI scorer — internal module, imported by contract.py and __init__.py.

Kept separate to avoid a circular import between src.evaluation.__init__ and
src.evaluation.contract.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.evaluation.kpi")


# ── Label-implied trade economics (asymmetric barrier payoff) ─────────────────
# The barrier objective is asymmetric: a CORRECT directional call captures the full
# profit barrier (+move_threshold); a WRONG call is stopped out at the loss barrier
# (-mae_tolerance). This is the cheap, price-free PnL proxy that rides on every
# model's confusion matrix. It is deliberately CONSERVATIVE — a "flat" (time-barrier
# / no-touch) outcome is charged the full stop even though its realised excursion
# lies in (-mae_tolerance, +move_threshold). The path-realised backtester
# (``src.backtest`` predictions mode) prices those exactly from the forward close
# path, and will usually report HIGHER expectancy than this proxy.

def payoff_params(objective: dict[str, Any]) -> tuple[float, float, float]:
    """``(win_points, loss_points, cost_per_trade)`` from the trade objective.

    ``win`` = ``labels.move_threshold`` (profit barrier); ``loss`` =
    ``labels.mae_tolerance`` (stop barrier); ``cost`` = ``costs.cost_per_trade``
    (round-trip, defaults to 0 so PnL is always computable)."""
    labels = objective.get("labels") or {}
    costs = objective.get("costs") or {}
    win = float(labels.get("move_threshold", 50.0))
    loss = float(labels.get("mae_tolerance", 5.0))
    cost = float(costs.get("cost_per_trade") or 0.0)
    return win, loss, cost


def max_drawdown(pnl: np.ndarray) -> float:
    """Worst peak-to-trough decline of the cumulative-PnL curve, in points (<= 0).

    The high-water mark is seeded at 0 (pre-trade equity), so a *leading* losing trade
    counts as drawdown — any dip below the running peak, including the starting 0,
    contributes ("cumulative current loss"). A trade that wins outright never adds
    drawdown; a stopped-out trade lowers equity below its high-water mark. The engine's
    realised-ledger drawdown (``engine._compute_kpis``) uses the same definition."""
    if pnl is None or len(pnl) == 0:
        return 0.0
    equity = np.cumsum(pnl)
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)   # seed the peak at 0
    return float((equity - peak).min())


def label_implied_pnl(
    dir_signals: pd.DataFrame,
    *,
    win_points: float,
    loss_points: float,
    cost: float = 0.0,
) -> dict[str, Any]:
    """Asymmetric per-trade PnL summary over gated directional signals.

    First-touch trade model: each trade concludes at exactly one barrier — a correct
    call captures ``+win_points - cost`` (profit target hit), a wrong call is stopped
    at ``-loss_points - cost``. *dir_signals* carries ``direction`` and
    ``direction_true`` in time order (canonical OOF frames are ts-sorted), so the
    equity curve — and its drawdown — is chronological. Returns expectancy / total /
    hit-rate / trade-Sharpe / max-drawdown plus the payoff assumptions (self-describing).
    """
    n = 0 if dir_signals is None else len(dir_signals)
    base = {"n_trades": int(n), "win_points": float(win_points),
            "loss_points": float(loss_points), "cost_per_trade": float(cost)}
    if n == 0:
        return {**base, "net_expectancy": None, "total_points": 0.0,
                "hit_rate": None, "sharpe": None, "max_drawdown": 0.0}
    correct = (dir_signals["direction"].to_numpy() ==
               dir_signals["direction_true"].to_numpy())
    pnl = np.where(correct, win_points - cost, -loss_points - cost).astype("float64")
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    return {
        **base,
        "net_expectancy": float(pnl.mean()),
        "total_points":   float(pnl.sum()),
        "hit_rate":       float(correct.mean()),
        "sharpe":         (float(pnl.mean() / std) if std > 0 else None),
        "max_drawdown":   max_drawdown(pnl),
    }


def canonical_trade_kpi(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    objective: dict[str, Any],
    *,
    tau: float | None = None,
) -> dict[str, float]:
    """Score a canonical prediction frame against the trade objective KPIs.

    Parameters
    ----------
    predictions:
        Canonical frame (ts, direction "up"/"down"/"flat", confidence [0,1]).
    ground_truth:
        Frame with column ``direction_true``.
    objective:
        Parsed Trade Objective dict.
    tau:
        Confidence cut. ``None`` = no cut (every argmax up/down is a trade).

    Returns
    -------
    dict
        ``precision_at_tau``, ``directional_f1``, ``n_signals``, ``n_up``, ``n_down``,
        plus ``f1_up`` / ``f1_down`` (per-class F1), ``confusion_matrix`` (raw 3x3),
        and ``trades_at_tau`` (taken / correct / false_positive_flat / wrong_direction).
        Trade-economics keys added when objective supplies ``costs.cost_per_trade``.
    """
    from sklearn.metrics import f1_score, confusion_matrix

    preds = predictions.copy()
    preds["direction_true"] = ground_truth["direction_true"].values

    raw_dir = preds[preds["direction"].isin({"up", "down"})]
    win_pts, loss_pts, cost = payoff_params(objective)
    pnl_raw = label_implied_pnl(raw_dir, win_points=win_pts, loss_points=loss_pts,
                                cost=cost)
    from src.models.dual_binary import is_dual_binary_frame
    _op = "dual_binary" if is_dual_binary_frame(preds) else "argmax"
    pnl_raw["operating_point"] = _op

    if tau is None:
        signals = preds
        dir_signals = raw_dir
        n_signals = len(dir_signals)
    else:
        signals = preds[preds["confidence"] >= float(tau)].copy()
        n_signals = len(signals)
        dir_signals = signals[signals["direction"].isin({"up", "down"})]
        if n_signals == 0:
            return {
                "precision_at_tau": float("nan"), "directional_f1": 0.0,
                "n_signals": 0, "n_up": 0, "n_down": 0,
                "trades_at_raw": _argmax_trade_counts(raw_dir, operating_point=_op),
                "pnl_raw": pnl_raw,
            }
    if len(dir_signals):
        correct = (dir_signals["direction"] == dir_signals["direction_true"]).sum()
        precision_at_tau = float(correct / len(dir_signals))
    else:
        precision_at_tau = float("nan")

    all_dir = preds[preds["direction_true"].isin({"up", "down"})]
    if len(all_dir):
        dir_f1 = float(f1_score(
            all_dir["direction_true"], all_dir["direction"],
            labels=["up", "down"], average="macro", zero_division=0,
        ))
    else:
        dir_f1 = 0.0

    result: dict[str, Any] = {
        "precision_at_tau": precision_at_tau,
        "directional_f1":   dir_f1,
        "n_signals":        n_signals,
        "n_up":             int((dir_signals["direction"] == "up").sum()),
        "n_down":           int((dir_signals["direction"] == "down").sum()),
    }

    # ── Per-class F1 + confusion matrix (full frame) + type-1 trade breakdown ──
    # These make the type-1 (false-positive-on-flat) vs type-2 (missed-signal)
    # split explicit in every persisted score. f1_up/f1_down are standard
    # multiclass per-class F1 (they DO penalise false positives against flat),
    # unlike ``directional_f1`` which is restricted to true-move rows.
    _LABELS = ["down", "flat", "up"]
    _cm = confusion_matrix(preds["direction_true"], preds["direction"], labels=_LABELS)
    result["f1_up"] = float(
        f1_score(preds["direction_true"], preds["direction"],
                 labels=["up"], average="macro", zero_division=0))
    result["f1_down"] = float(
        f1_score(preds["direction_true"], preds["direction"],
                 labels=["down"], average="macro", zero_division=0))

    # Per-direction precision at the firing threshold (mirrors the trainer's
    # per-direction precision_at_tau entries: BIG_UP / BIG_DOWN precision).
    _up_sig   = dir_signals[dir_signals["direction"] == "up"]
    _down_sig = dir_signals[dir_signals["direction"] == "down"]
    result["precision_up"] = (
        float((_up_sig["direction"] == _up_sig["direction_true"]).mean())
        if len(_up_sig) else float("nan"))
    result["precision_down"] = (
        float((_down_sig["direction"] == _down_sig["direction_true"]).mean())
        if len(_down_sig) else float("nan"))

    result["confusion_matrix"] = {
        "labels": _LABELS, "rows_true_cols_pred": _cm.tolist(),
    }
    _taken = dir_signals
    trades_tau: dict[str, Any] = {
        "taken":               int(len(_taken)),
        "correct":             int((_taken["direction"] == _taken["direction_true"]).sum()),
        "false_positive_flat": int((_taken["direction_true"] == "flat").sum()),
        "wrong_direction":     int(((_taken["direction_true"] != "flat") &
                                    (_taken["direction"] != _taken["direction_true"])).sum()),
    }
    if tau is None:
        trades_tau["operating_point"] = _op
    else:
        trades_tau["tau"] = float(tau)
    result["trades_at_tau"] = trades_tau
    result["trades_at_raw"] = _argmax_trade_counts(raw_dir, operating_point=_op)

    # ── Label-implied trade economics (asymmetric barrier payoff) ──────────────
    # Correct call captures +move_threshold; wrong call stops at -mae_tolerance.
    # Always computed now (cost defaults to 0) so every scored candidate carries an
    # expectancy — not only objectives that pinned a cost_per_trade. The full block
    # (with payoff assumptions) is exposed under result["pnl"]; the headline scalars
    # stay top-level for the leaderboard / webui sort keys.
    pnl = label_implied_pnl(dir_signals, win_points=win_pts, loss_points=loss_pts,
                            cost=cost)
    result["pnl"]            = pnl
    result["net_expectancy"] = pnl["net_expectancy"]
    result["hit_rate"]       = pnl["hit_rate"]
    result["sharpe"]         = pnl["sharpe"]
    result["max_drawdown"]   = pnl["max_drawdown"]
    result["pnl_raw"] = pnl_raw

    return result


def _argmax_trade_counts(
    taken: pd.DataFrame,
    *,
    operating_point: str = "argmax",
) -> dict[str, Any]:
    """Directional-fire counts with no confidence cut (argmax / dual-binary book)."""
    if taken is None or len(taken) == 0:
        return {"operating_point": operating_point, "taken": 0, "correct": 0,
                "false_positive_flat": 0, "wrong_direction": 0}
    pred = taken["direction"]
    true = taken["direction_true"]
    return {
        "operating_point": operating_point,
        "taken": int(len(taken)),
        "correct": int((pred == true).sum()),
        "false_positive_flat": int((true == "flat").sum()),
        "wrong_direction": int(((true != "flat") & (pred != true)).sum()),
    }


# ── Full diagnostics report ───────────────────────────────────────────────────

_DEFAULT_TAU_SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_LOOKUP_TAU = object()


def compute_diagnostics(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    objective: dict[str, Any],
    *,
    taus: tuple[float, ...] = _DEFAULT_TAU_SWEEP,
    gate_tau: float | None | object = _LOOKUP_TAU,
) -> dict[str, Any]:
    """Full per-candidate diagnostics report from a canonical OOF frame.

    Mirrors the structure of the legacy ``training_result.json`` diagnostics so
    the search pipeline carries the same detail as the trainer:

    * ``classification_report`` — sklearn dict over [down, flat, up]
    * ``confusion_matrix_raw`` — 3x3 on raw argmax (no confidence cut)
    * ``trades_at_raw`` / ``pnl_at_raw`` — directional argmax book + label-implied
      economics (same population as the raw 3x3 off-flat cells)
    * ``confusion_matrix_gated`` — 3x3 on predictions with confidence >= ``gate_tau``
    * ``precision_at_tau_sweep`` — per-direction precision + n_signals across ``taus``
    * ``trades_at_gate`` — taken / correct / false_positive_flat / wrong_direction
    * ``pnl_at_gate`` — label-implied expectancy / total / hit-rate / drawdown at the
      gate (asymmetric +move_threshold / -mae_tolerance barrier payoff)
    * ``pnl_at_tau_sweep`` — that expectancy swept across ``taus`` (the
      expectancy-vs-confidence curve: which τ maximises PnL)

    ``gate_tau`` omitted → ``primary_tau(objective)`` (YAML-declared cut only;
    ``None`` if the YAML has no confidence cut). Pass ``gate_tau=None`` to
    force the argmax book (Custom Test). Never infers 0.8.

    Read-only: computed from the persisted OOF, so it can be regenerated for any
    past run without re-training. This is the cheap price-free proxy; the
    path-realised ledger comes from ``src.backtest`` predictions mode.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from src.evaluation.kpi_criteria import primary_tau

    if gate_tau is _LOOKUP_TAU:
        gate_tau = primary_tau(objective)
    cut: float | None = None if gate_tau is None else float(gate_tau)

    preds = predictions.copy()
    preds["direction_true"] = ground_truth["direction_true"].values
    labels = ["down", "flat", "up"]

    report = classification_report(
        preds["direction_true"], preds["direction"],
        labels=labels, zero_division=0, output_dict=True,
    )
    f1_per_class = {l: float(report[l]["f1-score"]) for l in labels}
    f1_macro = float(report["macro avg"]["f1-score"])
    cm_raw = confusion_matrix(preds["direction_true"], preds["direction"], labels=labels)

    gated = preds if cut is None else preds[preds["confidence"] >= cut]
    cm_gated = confusion_matrix(gated["direction_true"], gated["direction"], labels=labels)

    sweep: list[dict[str, Any]] = []
    for t in taus:
        g = preds[preds["confidence"] >= t]
        for d in ("up", "down"):
            sig = g[g["direction"] == d]
            n = int(len(sig))
            prec = float((sig["direction"] == sig["direction_true"]).mean()) if n else float("nan")
            sweep.append({"tau": float(t), "direction": d, "n_signals": n, "precision": prec})

    taken = gated[gated["direction"].isin({"up", "down"})]
    trades: dict[str, Any] = {
        "taken":               int(len(taken)),
        "correct":             int((taken["direction"] == taken["direction_true"]).sum()),
        "false_positive_flat": int((taken["direction_true"] == "flat").sum()),
        "wrong_direction":     int(((taken["direction_true"] != "flat") &
                                    (taken["direction"] != taken["direction_true"])).sum()),
    }
    from src.models.dual_binary import dual_binary_head_block, is_dual_binary_frame
    dual = is_dual_binary_frame(preds)
    _op = "dual_binary" if dual else "argmax"

    if cut is None:
        trades["operating_point"] = _op
        trades["gate_tau"] = None
    else:
        trades["gate_tau"] = float(cut)
    raw_taken = preds[preds["direction"].isin({"up", "down"})]
    trades_raw = _argmax_trade_counts(raw_taken, operating_point=_op)

    # ── Label-implied PnL: at the gate, and swept across τ (expectancy-vs-confidence).
    # Same asymmetric barrier payoff as canonical_trade_kpi — the price-free proxy that
    # answers "which confidence threshold maximises expectancy?" straight from the OOF.
    win_pts, loss_pts, cost = payoff_params(objective)
    pnl_at_gate = label_implied_pnl(taken, win_points=win_pts, loss_points=loss_pts,
                                    cost=cost)
    if cut is None:
        pnl_at_gate["operating_point"] = _op
        pnl_at_gate["gate_tau"] = None
    else:
        pnl_at_gate["gate_tau"] = float(cut)
    pnl_at_raw = label_implied_pnl(raw_taken, win_points=win_pts, loss_points=loss_pts,
                                   cost=cost)
    pnl_at_raw["operating_point"] = _op
    pnl_sweep: list[dict[str, Any]] = []
    for t in taus:
        taken_t = preds[(preds["confidence"] >= t)
                        & preds["direction"].isin({"up", "down"})]
        p = label_implied_pnl(taken_t, win_points=win_pts, loss_points=loss_pts, cost=cost)
        pnl_sweep.append({"tau": float(t), "n_trades": p["n_trades"],
                          "net_expectancy": p["net_expectancy"],
                          "total_points": p["total_points"], "hit_rate": p["hit_rate"]})

    out: dict[str, Any] = {
        "labels":                 labels,
        "n_rows":                 int(len(preds)),
        "truth_distribution":     preds["direction_true"].value_counts().to_dict(),
        "pred_distribution":      preds["direction"].value_counts().to_dict(),
        "f1_macro":               f1_macro,
        "f1_per_class":           f1_per_class,
        "classification_report":  report,
        "confusion_matrix_raw":   {"labels": labels, "operating_point": _op,
                                   "rows_true_cols_pred": cm_raw.tolist()},
        "confusion_matrix_gated": {"labels": labels,
                                   "gate_tau": None if cut is None else float(cut),
                                   "operating_point": _op if cut is None else "confidence",
                                   "rows_true_cols_pred": cm_gated.tolist()},
        "precision_at_tau_sweep": sweep,
        "trades_at_raw":          trades_raw,
        "pnl_at_raw":             pnl_at_raw,
        "trades_at_gate":         trades,
        "pnl_at_gate":            pnl_at_gate,
        "pnl_at_tau_sweep":       pnl_sweep,
    }
    if dual:
        block = dual_binary_head_block(preds)
        # Head metrics first so a dual-binary run is not mistaken for 3-class softmax.
        out = {**block, **{k: v for k, v in out.items() if k != "labels"}}
    return out
