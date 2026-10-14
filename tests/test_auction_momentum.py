"""Causal session-auction features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.search.auction_momentum import auction_features


def test_auction_features_trend_then_pullback():
    ts = pd.date_range("2025-07-01 13:30:00", periods=120, freq="5s", tz="UTC").to_numpy()
    px = np.linspace(20000, 20040, 80)
    px = np.concatenate([px, np.linspace(20040, 20034, 40)])  # 6-pt pullback
    f = auction_features(ts, px)
    assert f["rally_from_low"][79] > 35
    assert f["pullback_from_high"][-1] >= 5
    assert f["ret_from_open"][79] > 30
    assert f["mins_since_high"][-1] > 2
