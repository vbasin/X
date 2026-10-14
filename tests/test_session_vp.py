"""Session volume profile attach (ADR-099) — 5m store, Globex 17:00 CT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.session_vp import (
    SVP_COLS, attach_session_vp, globex_session_keys, value_area_contiguous,
    wants_session_vp,
)


def test_value_area_contiguous_from_poc():
    px = np.array([100.0, 100.25, 100.50])
    vol = np.array([10.0, 100.0, 10.0])
    poc, vah, val = value_area_contiguous(px, vol, va_pct=0.70)
    assert poc == 100.25
    assert val <= poc <= vah


def test_globex_session_keys_flip_at_1700_ct():
    idx = pd.DatetimeIndex(pd.to_datetime(
        ["2026-01-05 22:59:00+00:00", "2026-01-05 23:00:00+00:00"], utc=True,
    ))
    keys = globex_session_keys(idx)
    assert keys[0] == "2026-01-05"
    assert keys[1] == "2026-01-06"


def test_attach_session_vp_break_and_reclaim():
    idx = pd.date_range("2026-01-05 14:30", periods=8, freq="5min", tz="UTC")
    close = np.array([100.00, 100.00, 100.00, 101.00, 101.00, 100.00, 100.00, 99.00])
    vol = np.array([50, 50, 50, 5, 5, 50, 50, 5], dtype=float)
    df = pd.DataFrame({
        "cs_close": close, "cs_volume": vol,
        "cs_high": close, "cs_low": close,
    }, index=idx)
    out = attach_session_vp(df)
    for c in SVP_COLS:
        assert c in out.columns
    assert out["svp_broke_vah"].iloc[3] == 1.0
    assert out["svp_reclaim_vah"].iloc[5] == 1.0
    assert not np.isnan(out["svp_poc_dist"].iloc[2])


def test_wants_session_vp_only_5m():
    obj = {"search": {"topology": {"session_vp": True}}}
    assert wants_session_vp(obj, "5m")
    assert not wants_session_vp(obj, "15s")
    assert not wants_session_vp(obj, "1m")
    assert not wants_session_vp({"search": {"topology": {"mode": "tf_oof_meta"}}}, "5m")
