"""Event-sampled primary + XGB veto (causal 5-minute clock)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.search.event_meta import (
    choose_tau,
    cooldown_keep,
    event_clock_mask,
    five_minute_close_mask,
    meta_win,
    one_minute_close_mask,
    primary_side,
    session_mask,
)


def _ts(start: str, n: int, freq: str = "5s"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC").to_numpy()


def test_five_minute_close_is_ct_00():
    # 13:30 UTC = 08:30 CT in summer (CDT).
    ts = _ts("2025-07-01 13:30:00", 12 * 10)  # 10 minutes
    m = five_minute_close_mask(ts)
    ct = pd.to_datetime(ts, utc=True).tz_convert("America/Chicago")
    assert m.sum() == 2  # 08:30 and 08:35
    assert set(ct[m].strftime("%H:%M:%S")) == {"08:30:00", "08:35:00"}


def test_one_minute_close_every_minute():
    ts = _ts("2025-07-01 13:30:00", 12 * 3)
    m = one_minute_close_mask(ts)
    assert int(m.sum()) == 3
    assert event_clock_mask(ts, "1m").sum() == 3


def test_session_mask_0830_1000_ct():
    ts = _ts("2025-07-01 13:00:00", 12 * 180)  # 08:00–11:00 CT
    m = session_mask(ts)
    ct = pd.to_datetime(ts, utc=True).tz_convert("America/Chicago")
    assert ct[m].min().strftime("%H:%M") == "08:30"
    assert ct[m].max().hour < 10 or (ct[m].max().hour == 9)


def test_primary_side_threshold():
    clv = np.array([0.5, 0.2, -0.7, 0.0, np.nan])
    s = primary_side(clv, 0.4)
    assert list(s) == [1, 0, -1, 0, 0]


def test_meta_win_only_matching_tbm_side():
    side = np.array([1, 1, -1, -1, 1])
    y = np.array([2, 1, 0, 2, 0])  # win, flat, win, wrong, wrong
    assert list(meta_win(side, y)) == [1, 0, 1, 0, 0]


def test_cooldown_10min_keeps_first():
    ts = _ts("2025-07-01 13:30:00", 12 * 20)  # 20 min of 5s
    # two events 5 min apart
    ev = ts[[0, 12 * 5]]
    kept = cooldown_keep(ev, 120)
    assert list(kept) == [True, False]
    ev2 = ts[[0, 12 * 10]]
    assert list(cooldown_keep(ev2, 120)) == [True, True]


def test_score_canonical_policy_cooldown_reduces_n():
    from src.search.event_meta import score_canonical_policy
    n = 40
    ts = pd.date_range("2025-07-01 13:35:00", periods=n, freq="5s", tz="UTC")
    df = pd.DataFrame({
        "ts": ts,
        "direction": ["up"] * n,
        "confidence": [0.7] * n,
        "direction_true": ["up", "flat"] * (n // 2),
    })
    out = score_canonical_policy(df, tau=0.65, walk_forward_tau=False)
    assert out["raw_overlapping"]["n_trades"] == n
    assert out["open_cooldown"]["n_trades"] < n
    assert out["open_cooldown"]["n_trades"] >= 1


def test_choose_tau_uses_train_not_random():
    rng = np.random.default_rng(0)
    n = 80
    ts = _ts("2025-07-01 13:30:00", n, freq="10min")
    y = np.where(rng.random(n) < 0.4, 2, 1).astype(np.int32)
    side = np.ones(n, dtype=np.int8)
    # high p_win on actual wins
    p = np.where(y == 2, 0.8, 0.2).astype(float)
    tau = choose_tau(p, side, y, ts, min_trades=5)
    assert 0.15 <= tau <= 0.80
