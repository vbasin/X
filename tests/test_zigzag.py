"""Thinkorswim ZigZagHighLow external-model engine (ADR-063)."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.external_models.catalog import coerce_params, factory_params, param_spec
from src.external_models.chart import build_chart_payload
from src.external_models.pine_sar_ema import simulate
from src.external_models.zigzag import STATE_DN, STATE_UNDEF, STATE_UP, zzhl_on_closed, zzhl_signals

CT = ZoneInfo("America/Chicago")


def _ohlc(rows, start="2026-06-02 08:30"):
    t0 = pd.Timestamp(start, tz=CT)
    idx = pd.date_range(t0, periods=len(rows), freq="1min").tz_convert("UTC")
    high = np.array([r[0] for r in rows], dtype="float64")
    low = np.array([r[1] for r in rows], dtype="float64")
    close = np.array([r[2] for r in rows], dtype="float64")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 10.0},
        index=idx,
    )


def test_factory_and_coerce():
    p = factory_params("zigzag")
    assert p["percentage_reversal"] == pytest.approx(2.5)
    assert p["atr_length"] == 5
    assert p["atr_reversal"] == pytest.approx(1.0)
    assert p["tick_size"] == pytest.approx(0.25)
    keys = {row["key"] for row in param_spec("zigzag")}
    assert keys == {
        "percentage_reversal", "absolute_reversal", "atr_length",
        "atr_reversal", "tick_reversal", "tick_size",
    }
    with pytest.raises(ValueError, match="non-zero"):
        coerce_params("zigzag", {
            "percentage_reversal": 0, "absolute_reversal": 0,
            "atr_reversal": 0, "tick_reversal": 0,
        })
    ok = coerce_params("zigzag", {"percentage_reversal": 0, "absolute_reversal": 8})
    assert ok["absolute_reversal"] == pytest.approx(8.0)


def test_absolute_reversal_state_machine():
    # 10-pt absolute reverse; percentage and ATR off.
    rows = [
        (100.0, 99.0, 100.0),   # 0 undef  max 100 min 99
        (102.0, 100.0, 101.0),  # 1 up     max 102
        (105.0, 103.0, 104.0),  # 2 up     max 105
        (104.0, 96.0, 97.0),    # 3 up     96 > 105-10
        (98.0, 94.0, 95.0),     # 4 down   94 <= 95
        (95.0, 90.0, 91.0),     # 5 down   min 90
        (91.0, 89.0, 90.0),     # 6 down   min 89
        (99.0, 90.0, 98.0),     # 7 up     99 >= 89+10
    ]
    df = _ohlc(rows)
    out = zzhl_on_closed(
        df,
        percentage_reversal=0.0,
        absolute_reversal=10.0,
        atr_length=5,
        atr_reversal=0.0,
        tick_reversal=0,
        tick_size=0.25,
    )
    sig = out["signal"].to_numpy()
    assert list(sig) == [0, 1, 1, 1, -1, -1, -1, 1]
    assert int(out["state"].iloc[0]) == STATE_UNDEF
    assert int(out["state"].iloc[1]) == STATE_UP
    assert int(out["state"].iloc[4]) == STATE_DN
    assert int(out["state"].iloc[7]) == STATE_UP
    zz = out["zz"].to_numpy()
    assert zz[0] == pytest.approx(99.0)
    assert zz[2] == pytest.approx(105.0)
    assert zz[6] == pytest.approx(89.0)
    assert zz[7] == pytest.approx(99.0)
    assert out["max_price_h"].iloc[2] == pytest.approx(105.0)
    assert out["min_price_l"].iloc[6] == pytest.approx(89.0)
    assert out["reversal_pts"].iloc[2] == pytest.approx(10.0)


def test_tick_reversal_matches_absolute():
    rows = [
        (100.0, 99.0, 100.0),
        (102.0, 100.0, 101.0),
        (105.0, 103.0, 104.0),
        (98.0, 94.0, 95.0),
    ]
    df = _ohlc(rows)
    abs_ = zzhl_on_closed(
        df, percentage_reversal=0.0, absolute_reversal=10.0,
        atr_length=5, atr_reversal=0.0, tick_reversal=0, tick_size=0.25,
    )
    ticks = zzhl_on_closed(
        df, percentage_reversal=0.0, absolute_reversal=0.0,
        atr_length=5, atr_reversal=0.0, tick_reversal=40, tick_size=0.25,
    )
    np.testing.assert_array_equal(abs_["signal"].to_numpy(), ticks["signal"].to_numpy())
    np.testing.assert_allclose(abs_["zz"].to_numpy(), ticks["zz"].to_numpy(), equal_nan=True)


def test_percentage_reversal_distance():
    rows = [
        (100.0, 99.0, 100.0),
        (110.0, 108.0, 109.0),  # up, max 110
        (110.0, 100.0, 101.0),  # 10% of 110 = 11 → need L <= 99. Stay up (L=100)
        (110.0, 98.0, 99.0),    # 98 <= 99 → down
    ]
    df = _ohlc(rows)
    out = zzhl_on_closed(
        df, percentage_reversal=10.0, absolute_reversal=0.0,
        atr_length=5, atr_reversal=0.0, tick_reversal=0, tick_size=0.25,
    )
    assert list(out["signal"].to_numpy()) == [0, 1, 1, -1]
    assert out["reversal_pts"].iloc[1] == pytest.approx(11.0)


def test_atr_reversal_formula():
    # Constant 2-pt range so ATR(5) seeds at 2.0. close=1000.
    rows = [(1001.0, 999.0, 1000.0)] * 8
    df = _ohlc(rows)
    out = zzhl_on_closed(
        df, percentage_reversal=2.5, absolute_reversal=0.0,
        atr_length=5, atr_reversal=1.0, tick_reversal=0, tick_size=0.25,
    )
    # ATR valid from index 4. hl = 0.025 + ATR/close * 1.
    atr4 = float(out["atr"].iloc[4])
    assert atr4 == pytest.approx(2.0)
    hl = 0.025 + atr4 / 1000.0
    assert out["hl_pivot"].iloc[4] == pytest.approx(hl)
    assert out["reversal_pts"].iloc[4] == pytest.approx(1001.0 * hl)


def test_zzhl_signals_attrs():
    rows = [(100.0, 99.0, 100.0), (102.0, 100.0, 101.0)]
    frame = zzhl_signals(_ohlc(rows), {
        "percentage_reversal": 0.0, "absolute_reversal": 5.0,
        "atr_reversal": 0.0,
    })
    assert frame.attrs["engine"] == "zigzag"
    assert frame.attrs["model_id"] == "zigzag"
    assert "signal" in frame.columns
    assert int(frame["signal"].iloc[1]) == 1


def test_simulate_absolute_reverse_and_flatten():
    up = [(100.0 + i, 99.0 + i, 99.5 + i) for i in range(20)]
    crash = [(118.0, 80.0, 85.0)]  # >10 pts off the high → short
    rest = [(85.0 - i * 0.5, 84.0 - i * 0.5, 84.5 - i * 0.5) for i in range(30)]
    df = _ohlc(up + crash + rest, start="2026-06-02 08:30")
    result = simulate(df, {
        "percentage_reversal": 0.0,
        "absolute_reversal": 10.0,
        "atr_reversal": 0.0,
        "min_atr": 0.0,
        "profit_target_pts": 0.0,
        "stop_loss_pts": 0.0,
    }, model_id="zigzag")
    assert result.n_bars == len(df)
    assert not result.trades.empty
    sides = set(result.trades["side"])
    assert "long" in sides
    assert "short" in sides
    last = result.trades.iloc[-1]
    assert last["exit_reason"] in ("flatten_1445", "end_of_data", "zz_up_close", "zz_dn_close")


def test_chart_payload_zigzag_arrows():
    rows = [
        (100.0, 99.0, 100.0),
        (102.0, 100.0, 101.0),
        (105.0, 103.0, 104.0),
        (98.0, 94.0, 95.0),
    ]
    df = _ohlc(rows, start="2026-06-02 08:30")
    sess = build_chart_payload(
        source="external:zigzag",
        start="2026-06-02", end="2026-06-02",
        params={"percentage_reversal": 0.0, "absolute_reversal": 10.0, "atr_reversal": 0.0},
        session_only=True, ohlc=df,
    )
    assert sess["has_zz"] is True
    assert sess["has_sar"] is False
    assert sess["has_ema"] is False
    assert len(sess["zz_trend"]) == sess["n"]
    assert 1 in sess["zz_trend"]
    assert -1 in sess["zz_trend"]
    assert any(v is not None for v in sess["zz"])
    assert len(sess["atr"]) == sess["n"]
