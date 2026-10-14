"""Tick-level OFI on the 5s grid (of_ofi / of_ofi_zscore_60)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.order_flow import OFI_COLS, _compute_ofi, compute


def _tbbo(ts, bid_px, bid_sz, ask_px, ask_sz, price=None, size=None, side=None):
    n = len(ts)
    return pd.DataFrame({
        "ts_event": pd.DatetimeIndex(ts, tz="UTC"),
        "bid_px": bid_px,
        "bid_sz": bid_sz,
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "price": price if price is not None else bid_px,
        "size": size if size is not None else np.ones(n),
        "side": side if side is not None else ["B"] * n,
    })


def test_ofi_cols_are_of_ofi_and_zscore_60():
    assert OFI_COLS == ["of_ofi", "of_ofi_zscore_60"]


def test_same_price_size_change_is_delta_bid_sz():
    # Cont 2014: both bid indicators true at an unchanged bid → +bid_sz − prev_bid_sz.
    ts = pd.date_range("2026-08-25 13:30:00", periods=2, freq="1s", tz="UTC")
    tbbo = _tbbo(ts, [100.0, 100.0], [10.0, 15.0], [100.25, 100.25], [10.0, 10.0])
    ofi = _compute_ofi(tbbo, "5s")
    assert ofi is not None
    np.testing.assert_allclose(ofi["of_ofi"].iloc[0], 5.0)


def test_bid_up_counts_new_bid_plus_prev_ask_when_ask_also_up():
    ts = pd.date_range("2026-08-25 13:30:00", periods=2, freq="1s", tz="UTC")
    tbbo = _tbbo(
        ts,
        bid_px=[100.00, 100.25],
        bid_sz=[10.0, 8.0],
        ask_px=[100.25, 100.50],
        ask_sz=[10.0, 12.0],
    )
    ofi = _compute_ofi(tbbo, "5s")
    assert ofi is not None
    # bid>=prev: +8; bid<=prev: 0; ask<=prev: 0; ask>=prev: +10 → 18
    np.testing.assert_allclose(ofi["of_ofi"].iloc[0], 18.0)


def test_bar_sums_ticks_in_the_5s_bucket():
    ts = [
        pd.Timestamp("2026-08-25 13:30:00", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:01", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:02", tz="UTC"),
    ]
    tbbo = _tbbo(
        ts,
        bid_px=[100.0, 100.0, 100.0],
        bid_sz=[10.0, 15.0, 12.0],
        ask_px=[100.25, 100.25, 100.25],
        ask_sz=[10.0, 10.0, 10.0],
    )
    ofi = _compute_ofi(tbbo, "5s")
    assert ofi is not None
    # tick1 Δbid=+5; tick2 Δbid=−3 → bar 2
    np.testing.assert_allclose(ofi["of_ofi"].iloc[0], 2.0)


def test_compute_emits_ofi_on_the_5s_index():
    ts = pd.date_range("2026-08-25 13:30:00", periods=3, freq="1s", tz="UTC")
    tbbo = _tbbo(
        ts,
        bid_px=[100.0, 100.0, 100.0],
        bid_sz=[10.0, 11.0, 9.0],
        ask_px=[100.25, 100.25, 100.25],
        ask_sz=[10.0, 10.0, 10.0],
        price=[100.25, 100.25, 100.0],
        size=[1, 2, 1],
        side=["B", "B", "A"],
    )
    out = compute(tbbo, timestep="5s")
    for c in OFI_COLS:
        assert c in out.columns
    np.testing.assert_allclose(out["of_ofi"].iloc[0], -1.0)  # +1 then −2


def test_missing_quote_size_returns_none():
    ts = pd.date_range("2026-08-25 13:30:00", periods=2, freq="1s", tz="UTC")
    tbbo = pd.DataFrame({
        "ts_event": ts,
        "price": [100.0, 100.0],
        "size": [1, 1],
        "side": ["B", "A"],
        "bid_px": [99.75, 99.75],
        "ask_px": [100.0, 100.0],
    })
    assert _compute_ofi(tbbo, "5s") is None
