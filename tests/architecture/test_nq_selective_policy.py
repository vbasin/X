"""Selection must depend on predictions and already-realized history only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.train_nq_selective import book, summarize


START = pd.Timestamp("2026-01-05T15:00:00Z")


def row(minute: int, *, side: int = 1, prediction: float = 0.5,
        net: float = 100.0, net_r: float = 0.5, duration: int = 1,
        valid: bool = True, day: int = 0) -> dict:
    ts = START + pd.Timedelta(days=day, minutes=minute)
    return dict(ts=ts, side=side, predicted_r=prediction,
                session=ts.date().isoformat(), exit_ts=ts+pd.Timedelta(minutes=duration),
                net_usd=net, net_r=net_r, valid=valid,
                invalid_reason="" if valid else "missing_time_exit_within_5s",
                mae_points=2.0)


def test_prediction_selects_side_before_future_pnl():
    scored = pd.DataFrame([
        row(0, side=1, prediction=0.6, net=-200, net_r=-1),
        row(0, side=-1, prediction=0.5, net=2000, net_r=10),
    ])
    trades, unresolved = book(scored, threshold=0.1)
    assert unresolved == []
    assert trades.side.tolist() == [1]
    assert trades.net_usd.tolist() == [-200]
    # Changing unselected future outcomes cannot change the selected side.
    scored.loc[scored.side.eq(-1), "net_usd"] = -999999
    changed, _ = book(scored, threshold=0.1)
    assert changed.side.tolist() == [1]


def test_no_fallback_to_valid_lower_score_when_selected_path_censored():
    scored = pd.DataFrame([
        row(0, prediction=0.6, valid=False),
        row(0, side=-1, prediction=0.5),
        row(5, prediction=0.9),
    ])
    trades, unresolved = book(scored, threshold=0.1)
    assert trades.empty
    assert len(unresolved) == 1
    assert unresolved[0]["reason"] == "missing_time_exit_within_5s"


def test_overlap_and_exact_exit_instant_are_blocked():
    scored = pd.DataFrame([row(0, duration=10), row(5), row(10), row(15)])
    trades, _ = book(scored, threshold=0.1)
    assert trades.ts.tolist() == [START, START+pd.Timedelta(minutes=15)]


def test_losses_only_affect_decisions_after_exit_and_reset_each_session():
    scored = pd.DataFrame([
        row(0, net=-100, net_r=-0.5, duration=10),
        row(5, net=-100, net_r=-0.5),  # blocked overlap; not another loss
        row(15, net=100, net_r=0.5),
        row(20, net=-100, net_r=-0.5),
        row(25, net=1000, net_r=5),   # daily second-loss stop
        row(0, day=1, net=100, net_r=0.5),
    ])
    trades, _ = book(scored, threshold=0.1)
    assert trades.ts.tolist() == [
        START, START+pd.Timedelta(minutes=15), START+pd.Timedelta(minutes=20),
        START+pd.Timedelta(days=1),
    ]


def test_risk_loss_stop_and_maximum_four_entries():
    stopped, _ = book(pd.DataFrame([
        row(0, net=-500, net_r=-2.5), row(5), row(10),
    ]), threshold=0.1)
    assert len(stopped) == 1
    capped, _ = book(pd.DataFrame([row(i*5) for i in range(8)]), threshold=0.1)
    assert len(capped) == 4


def test_future_prediction_threshold_and_nan_do_not_create_orders():
    scored = pd.DataFrame([row(0, prediction=0.01), row(5, prediction=np.nan),
                           row(10, prediction=0.1)])
    trades, _ = book(scored, threshold=0.1)
    assert trades.ts.tolist() == [START+pd.Timedelta(minutes=10)]


def test_summary_includes_zero_trade_sessions_and_initial_capital_peak():
    trades = pd.DataFrame([row(0, net=-100), row(5, net=50), row(0, day=2, net=100)])
    sessions = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    stats, daily = summarize(trades, sessions, bootstrap=True)
    assert daily.net_usd.tolist() == [-50, 0, 100, 0]
    assert daily.equity_usd.tolist() == [-50, -50, 50, 50]
    assert daily.drawdown_usd.tolist() == [-50, -50, 0, 0]
    assert stats["mean_daily_usd"] == 12.5
    assert stats["max_drawdown_usd"] == 100
    assert stats["loss_count"] == 1
    assert stats["gross_loss_usd"] == 100
    assert stats["profit_factor"] == 1.5
    assert len(stats["daily_mean_95ci"]) == 2


def test_cash_summary_is_zero_and_bootstrap_reproducible():
    empty = pd.DataFrame(columns=list(row(0)))
    stats, daily = summarize(empty, ["2026-01-05", "2026-01-06"], bootstrap=True)
    assert stats["n_trades"] == 0
    assert stats["net_usd"] == 0
    assert stats["daily_mean_95ci"] == [0, 0]
    assert (daily.net_usd == 0).all()
    assert (daily.drawdown_usd == 0).all()
    trades = pd.DataFrame([row(0)])
    one, _ = summarize(trades, ["2026-01-05", "2026-01-06"], bootstrap=True)
    two, _ = summarize(trades, ["2026-01-05", "2026-01-06"], bootstrap=True)
    assert one["daily_mean_95ci"] == two["daily_mean_95ci"]
