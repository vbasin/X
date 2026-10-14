"""cs_* must ffill onto trailing 5s bars (live edge)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.live_runtime import tbbo_to_1m_candles
from src.features.engine import merge_feature_rows


def test_tbbo_to_1m_candles():
    idx = pd.date_range("2026-08-14 15:00:00", periods=120, freq="1s", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": idx,
        "price": np.linspace(100, 101, len(idx)),
        "size": 1.0,
        "symbol": "NQU6",
    })
    c = tbbo_to_1m_candles(tbbo)
    assert not c.empty
    assert list(c.columns) == ["open", "high", "low", "close", "volume"]
    assert c.index.tz is not None


def test_merge_then_cs_ffill_fills_trailing():
    idx = pd.date_range("2026-08-14 15:00:00", periods=5, freq="5s", tz="UTC")
    existing = pd.DataFrame({"cs_clv": [0.1, 0.2, np.nan, np.nan, np.nan], "_close": range(5)}, index=idx)
    # simulate ffill repair used by live_runtime
    existing["cs_clv"] = existing["cs_clv"].ffill()
    assert existing["cs_clv"].iloc[-1] == 0.2

    new = pd.DataFrame(
        {"cs_clv": [np.nan], "_close": [9]},
        index=pd.DatetimeIndex([idx[-1] + pd.Timedelta("5s")]),
    )
    out = merge_feature_rows(existing, new)
    out["cs_clv"] = out["cs_clv"].ffill()
    assert out["cs_clv"].iloc[-1] == 0.2
