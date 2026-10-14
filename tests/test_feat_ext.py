"""v5.5 extension features: YTD, ATR, IV rank, VIX, tl rolls, yday VP, stoch sync."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.bar_windows import bars, bars_m
from src.features import atr as atr_mod
from src.features import higher_timeframe
from src.features import stochastic
from src.features import trade_location
from src.features import volume_profile
from src.features import vix as vix_mod

_TL_COLS = (
    "tl_trade_location", "tl_aggressive_pct",
    "tl_loc_ma_30s", "tl_loc_ma_5m", "tl_loc_ma_30m",
    "tl_aggr_30s", "tl_aggr_5m", "tl_aggr_30m",
    "tl_loc_diff_30s_5m", "tl_loc_diff_5m_30m",
    "tl_loc_zscore_30s_5m", "tl_loc_zscore_5m_30m",
    "tl_aggr_diff_30s_5m", "tl_aggr_diff_5m_30m",
    "tl_aggr_zscore_30s_5m", "tl_aggr_zscore_5m_30m",
)

_TL_RETIRED_COLS = (
    "tl_loc_ma_1m", "tl_loc_ma_60m", "tl_aggr_1m", "tl_aggr_60m",
    "tl_loc_diff_30s", "tl_loc_diff_5m", "tl_loc_diff_30m",
    "tl_loc_zscore_30s", "tl_loc_zscore_5m", "tl_loc_zscore_30m",
    "tl_aggr_diff_30s", "tl_aggr_diff_5m", "tl_aggr_diff_30m",
    "tl_aggr_zscore_30s", "tl_aggr_zscore_5m", "tl_aggr_zscore_30m",
)


def _tl_tbbo(ts, price, bid=100.00, ask=100.25, size=None):
    n = len(ts)
    return pd.DataFrame({
        "ts_event": ts,
        "price": price,
        "size": np.ones(n) if size is None else size,
        "bid_px": np.full(n, bid),
        "ask_px": np.full(n, ask),
        "side": np.ones(n),
    })


def test_trade_location_rolling_windows():
    idx = pd.date_range("2026-06-02 14:00:00", periods=200, freq="1s", tz="UTC")
    tbbo = _tl_tbbo(idx, np.where(np.arange(200) % 2 == 0, 100.25, 100.00))
    out = trade_location.compute(tbbo, timestep="5s")
    assert list(out.columns) == list(_TL_COLS) or set(out.columns) == set(_TL_COLS)
    assert len(out.columns) == 16
    for col in _TL_RETIRED_COLS:
        assert col not in out.columns


def test_tl_rolls_are_bar_means_of_tl_trade_location():
    """30s/5m/30m MAs are rolling means of the 5s bar series, not tick windows."""
    assert bars(30) == 6
    assert bars_m(5) == 60
    assert bars_m(30) == 360

    idx = pd.date_range("2026-06-02 14:00:00", periods=80, freq="5s", tz="UTC")
    # Alternate ask/bid so the 5s bar value is ±1 and the 6-bar MA is not
    # identical to a print-weighted tick window of the same clock span.
    price = np.where(np.arange(80) % 2 == 0, 100.25, 100.00)
    out = trade_location.compute(_tl_tbbo(idx, price), timestep="5s")
    loc = out["tl_trade_location"]
    aggr = out["tl_aggressive_pct"]
    windows = (
        ("30s", bars(30)),
        ("5m", bars_m(5)),
        ("30m", bars_m(30)),
    )
    for suffix, win in windows:
        np.testing.assert_allclose(
            out[f"tl_loc_ma_{suffix}"].to_numpy(),
            loc.rolling(win, min_periods=1).mean().to_numpy(),
            rtol=1e-5, atol=1e-5, err_msg=f"tl_loc_ma_{suffix}",
        )
        np.testing.assert_allclose(
            out[f"tl_aggr_{suffix}"].to_numpy(),
            aggr.rolling(win, min_periods=1).mean().to_numpy(),
            rtol=1e-5, atol=1e-5, err_msg=f"tl_aggr_{suffix}",
        )


def test_tl_loc_ma_30s_equals_six_equal_weight_bars():
    """Uneven print density: bar MA ≠ print-weighted tick MA.

    Five 5s bars with 10 ask prints, then one 5s bar with a single bid print.
    Equal-weight 6-bar mean is 4/6; print-weighted 30s mean is 49/51.
    """
    t0 = pd.Timestamp("2026-06-02 14:00:00", tz="UTC")
    rows_ts = []
    rows_px = []
    for b in range(5):
        for k in range(10):
            rows_ts.append(t0 + pd.Timedelta(seconds=5 * b + 0.1 * k))
            rows_px.append(100.25)
    rows_ts.append(t0 + pd.Timedelta(seconds=25.1))
    rows_px.append(100.00)
    out = trade_location.compute(
        _tl_tbbo(pd.DatetimeIndex(rows_ts), np.array(rows_px)),
        timestep="5s",
    )
    last = out.iloc[-1]
    assert last["tl_trade_location"] == pytest.approx(-1.0)
    # mean(1,1,1,1,1,-1) = 4/6
    assert last["tl_loc_ma_30s"] == pytest.approx(4.0 / 6.0)
    tick_weighted = (50 * 1.0 + 1 * (-1.0)) / 51.0
    assert last["tl_loc_ma_30s"] != pytest.approx(tick_weighted)


def test_tl_trade_location_is_size_weighted():
    """One 5s bar: 10-lot at ask + 1-lot at bid → 9/11, not print-mean 0."""
    t0 = pd.Timestamp("2026-06-02 14:00:00", tz="UTC")
    ts = pd.DatetimeIndex([t0 + pd.Timedelta(milliseconds=100),
                           t0 + pd.Timedelta(milliseconds=200)])
    out = trade_location.compute(
        _tl_tbbo(ts, np.array([100.25, 100.00]), size=np.array([10.0, 1.0])),
        timestep="5s",
    )
    assert out["tl_trade_location"].iloc[0] == pytest.approx(9.0 / 11.0)
    assert out["tl_trade_location"].iloc[0] != pytest.approx(0.0)


def test_tl_aggressive_pct_is_size_weighted():
    """One 5s bar: 10-lot sweep + 1-lot at ask → 10/11, not print-mean 0.5."""
    t0 = pd.Timestamp("2026-06-02 14:00:00", tz="UTC")
    ts = pd.DatetimeIndex([t0 + pd.Timedelta(milliseconds=100),
                           t0 + pd.Timedelta(milliseconds=200)])
    # 100.50 > ask 100.25 → sweep; 100.25 == ask → not a sweep.
    out = trade_location.compute(
        _tl_tbbo(ts, np.array([100.50, 100.25]), size=np.array([10.0, 1.0])),
        timestep="5s",
    )
    assert out["tl_aggressive_pct"].iloc[0] == pytest.approx(10.0 / 11.0)
    assert out["tl_aggressive_pct"].iloc[0] != pytest.approx(0.5)


def test_tl_pair_zscore_and_short_long():
    """Pair stats only: short MA − long MA and that difference / std_long."""
    n = 400
    idx = pd.date_range("2026-06-02 14:00:00", periods=n, freq="5s", tz="UTC")
    price = np.where(np.arange(n) % 2 == 0, 100.25, 100.00)
    out = trade_location.compute(_tl_tbbo(idx, price), timestep="5s")
    wins = {"30s": bars(30), "5m": bars_m(5), "30m": bars_m(30)}
    families = (
        ("tl_trade_location", "tl_loc_ma", "tl_loc_diff", "tl_loc_zscore"),
        ("tl_aggressive_pct", "tl_aggr", "tl_aggr_diff", "tl_aggr_zscore"),
    )
    for src, ma_p, diff_p, z_p in families:
        series = out[src]
        ma = {s: series.rolling(w, min_periods=1).mean() for s, w in wins.items()}
        std = {s: series.rolling(w).std().replace(0, np.nan) for s, w in wins.items()}
        for short, long in (("30s", "5m"), ("5m", "30m")):
            pair = ma[short] - ma[long]
            np.testing.assert_allclose(
                out[f"{diff_p}_{short}_{long}"].to_numpy(),
                pair.to_numpy(),
                rtol=1e-5, atol=1e-5, err_msg=f"{diff_p}_{short}_{long}",
            )
            np.testing.assert_allclose(
                out[f"{z_p}_{short}_{long}"].to_numpy(),
                (pair / std[long]).to_numpy(),
                rtol=1e-5, atol=1e-5, equal_nan=True,
                err_msg=f"{z_p}_{short}_{long}",
            )
            assert f"{diff_p}_{short}" not in out.columns
            assert f"{z_p}_{short}" not in out.columns


def test_stoch_sync_score_80_down_and_20_up():
    n = 100
    idx = pd.date_range("2026-06-02 14:00", periods=n, freq="1min", tz="UTC")
    # Pin at the range high so %D sits at 100, then slam to the low.
    close = np.concatenate([np.full(70, 120.0), np.full(30, 100.0)])
    candles = pd.DataFrame(
        {"open": close, "high": 120.5, "low": 99.5, "close": close, "volume": 1.0},
        index=idx,
    )
    so = stochastic.compute(candles)
    assert list(so.columns) == list(stochastic.SO_COLS)
    assert "so_fast_d_9_3" in so.columns
    assert so["so_fast_d_9_3"].notna().any()
    assert float(so["so_sync_score_all"].min()) < 0.0
    assert "so_sync_score_9_60" in so.columns
    assert float(so["so_sync_score_all"].min()) >= -1.0
    assert float(so["so_sync_score_all"].max()) <= 1.0
    for retired in stochastic.SO_RETIRED_COLS:
        assert retired not in so.columns


def test_stoch_d_only_names_and_formulas():
    """%D-only columns; Fast is Wilder; Full(60) is EMA D=10 with slowing=1."""
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2026-06-02 13:00", periods=n, freq="1min", tz="UTC")
    close = 21000 + np.cumsum(rng.normal(0, 4, n))
    candles = pd.DataFrame({
        "open": close,
        "high": close + rng.uniform(0.5, 8, n),
        "low": close - rng.uniform(0.5, 8, n),
        "close": close,
        "volume": 1.0,
    }, index=idx)
    so = stochastic.compute(candles)

    assert list(so.columns) == [
        "so_fast_d_9_3", "so_fast_d_14_3", "so_fast_d_40_4",
        "so_full_d_60_10", "so_sync_score_all", "so_sync_score_9_60",
    ]
    k_cols = [c for c in so.columns if c.startswith("so_") and "_k_" in c]
    assert k_cols == []

    def raw_k(period):
        lowest = candles["low"].rolling(period, min_periods=period).min()
        highest = candles["high"].rolling(period, min_periods=period).max()
        denom = (highest - lowest).replace(0.0, np.nan)
        return (100.0 * (candles["close"] - lowest) / denom).clip(0.0, 100.0)

    want_fast9 = raw_k(9).ewm(com=2, adjust=False, min_periods=3).mean()
    np.testing.assert_allclose(
        so["so_fast_d_9_3"].to_numpy(), want_fast9.to_numpy(),
        rtol=1e-5, atol=1e-4, equal_nan=True,
    )
    # TOS Full slowing=1 → FullK = FastK; FullD = EMA(raw %K, 10)
    raw60 = raw_k(60)
    full_k = raw60.ewm(span=1, adjust=False, min_periods=1).mean()
    want_full = full_k.ewm(span=10, adjust=False, min_periods=10).mean()
    np.testing.assert_allclose(
        so["so_full_d_60_10"].to_numpy(), want_full.to_numpy(),
        rtol=1e-5, atol=1e-4, equal_nan=True,
    )
    wilder10 = raw60.ewm(com=9, adjust=False, min_periods=10).mean()
    both = so["so_full_d_60_10"].notna() & wilder10.notna()
    assert float(np.nanmax(np.abs(
        so["so_full_d_60_10"][both] - wilder10[both]
    ))) > 0.05


def _sync_decay(progress: float) -> float:
    p = min(max(progress, 0.0), 1.0)
    e_l = np.exp(-1.0)
    return float((np.exp(-1.0 * p) - e_l) / (1.0 - e_l))


def test_sync_score_exp_envelope_80_to_50():
    """Armed 80-down fades exponentially toward 50; ±1 when all members fire."""
    idx = pd.date_range("2026-06-02 14:00", periods=12, freq="1min", tz="UTC")
    # Cross 81→79, grind to 50, then through mid to 40.
    d = pd.Series(
        [90.0, 85.0, 81.0, 79.0, 70.0, 65.0, 55.0, 50.0, 40.0, 40.0, 40.0, 40.0],
        index=idx,
    )
    four = pd.DataFrame({c: d for c in ("a", "b", "c", "e")})
    s = stochastic._sync_score(four)
    assert float(s.min()) >= -1.0
    assert float(s.max()) <= 1.0

    cross = idx[3]  # 79
    assert s.loc[cross] == pytest.approx(-_sync_decay((80.0 - 79.0) / 30.0), abs=1e-5)
    at65 = idx[5]
    assert s.loc[at65] == pytest.approx(-_sync_decay((80.0 - 65.0) / 30.0), abs=1e-5)
    at50 = idx[7]
    assert s.loc[at50] == pytest.approx(0.0, abs=1e-5)
    assert s.loc[idx[8]] == pytest.approx(0.0, abs=1e-5)
    # Envelope still on 4 minutes after the cross (not a 3-bar pulse).
    assert s.loc[idx[5]] < -0.2


def test_sync_score_20_up_reaches_plus_one():
    idx = pd.date_range("2026-06-02 14:00", periods=6, freq="1min", tz="UTC")
    d = pd.Series([10.0, 15.0, 19.0, 21.0, 21.0, 21.0], index=idx)
    four = pd.DataFrame({c: d for c in ("a", "b", "c", "e")})
    s = stochastic._sync_score(four)
    cross = idx[3]
    assert s.loc[cross] == pytest.approx(_sync_decay((21.0 - 20.0) / 30.0), abs=1e-5)
    assert float(s.max()) <= 1.0
    # Exact print at 20 with four members → +1.
    d20 = pd.Series([10.0, 19.9, 20.0, 20.0], index=idx[:4])
    s20 = stochastic._sync_score(pd.DataFrame({c: d20 for c in ("a", "b", "c", "e")}))
    assert s20.iloc[2] == pytest.approx(1.0, abs=1e-5)


def test_sync_score_9_60_ignores_14_and_40():
    idx = pd.date_range("2026-06-02 14:00", periods=5, freq="1min", tz="UTC")
    # 9 and 60 fire 80-down; 14 and 40 fire 20-up — quartet cancels, pair is sell.
    down = pd.Series([90.0, 81.0, 79.0, 79.0, 79.0], index=idx)
    up = pd.Series([10.0, 19.0, 21.0, 21.0, 21.0], index=idx)
    candles_d = pd.DataFrame({
        "so_fast_d_9_3": down,
        "so_fast_d_14_3": up,
        "so_fast_d_40_4": up,
        "so_full_d_60_10": down,
    }, index=idx)
    quartet = stochastic._sync_score(candles_d)
    pair = stochastic._sync_score(candles_d[["so_fast_d_9_3", "so_full_d_60_10"]])
    assert pair.iloc[2] < -0.8
    assert abs(float(quartet.iloc[2])) < abs(float(pair.iloc[2]))
    assert float(pair.min()) >= -1.0
    assert float(pair.max()) <= 1.0


def test_sync_recross_and_two_bar_bounce_clear_envelope():
    idx = pd.date_range("2026-06-02 14:00", periods=8, freq="1min", tz="UTC")
    # 80-down then recross back above 80.
    recross = pd.Series([90.0, 81.0, 79.0, 81.0, 82.0, 82.0, 82.0, 82.0], index=idx)
    s = stochastic._sync_score(pd.DataFrame({"a": recross, "b": recross}))
    assert s.iloc[2] < -0.5
    assert s.iloc[3] == pytest.approx(0.0, abs=1e-5)

    # Two consecutive up bars after 80-down zero the armed vote (1-bar pause).
    bounce = pd.Series([90.0, 81.0, 79.0, 78.5, 77.0, 78.0, 79.5, 79.5], index=idx)
    s2 = stochastic._sync_score(pd.DataFrame({"a": bounce, "b": bounce}))
    assert s2.iloc[2] < -0.5
    assert s2.iloc[4] < 0.0  # still falling-ish / one pause not yet
    # bars 5–6: 77→78 and 78→79.5 are two contrary bars
    assert s2.iloc[6] == pytest.approx(0.0, abs=1e-5)


def test_atr_stale_forward_5m_10m():
    idx = pd.date_range("2026-06-02 14:00", periods=200, freq="1min", tz="UTC")
    rng = np.linspace(2.0, 8.0, 200)
    candles = pd.DataFrame({
        "open": 100.0, "close": 100.0 + rng / 4,
        "high": 100.0 + rng, "low": 100.0 - 0.2, "volume": 1.0,
    }, index=idx)
    grid = pd.date_range("2026-06-02 14:00", periods=200 * 12, freq="5s", tz="UTC")
    out = atr_mod.compute(candles, grid)
    assert "cs_atr_5m" in out.columns and "cs_atr_10m" in out.columns
    # After first 5m close, 60 consecutive 5s bars share a value.
    s = out["cs_atr_5m"].dropna()
    assert len(s) >= 60
    # Find a 60-bar hold
    vals = out["cs_atr_5m"].to_numpy()
    holds = []
    n = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1] or (pd.isna(vals[i]) and pd.isna(vals[i - 1])):
            n += 1
        else:
            if not pd.isna(vals[i - 1]):
                holds.append(n)
            n = 1
    assert any(h == 60 for h in holds[:-1] or holds)


def test_ytd_high_low_dist():
    prior = pd.DataFrame({
        "date": [date(2026, 1, 2), date(2026, 3, 1)],
        "high": [21000.0, 22000.0],
        "low": [18000.0, 19000.0],
    })
    idx = pd.date_range("2026-06-02 14:00", periods=12, freq="5s", tz="UTC")
    candles = pd.DataFrame(
        {"high": 22100.0, "low": 20000.0, "close": 22100.0, "open": 22000.0, "volume": 1},
        index=pd.date_range("2026-06-02 14:00", periods=2, freq="1min", tz="UTC"),
    )
    px = pd.Series(22100.0, index=idx)
    out = higher_timeframe.compute_ytd(prior, candles, px, idx)
    assert (out["ht_ytd_high_dist"].iloc[-1] >= -1e-6)
    assert out["ht_ytd_low_dist"].iloc[-1] > 0


def test_prior_session_vp_constant_levels():
    prev = pd.DataFrame({
        "ts_event": pd.date_range("2026-06-01 14:00", periods=100, freq="1min", tz="UTC"),
        "price": np.concatenate([np.full(50, 100.0), np.full(50, 101.0)]),
        "size": np.concatenate([np.full(50, 10.0), np.full(50, 2.0)]),
    })
    idx = pd.date_range("2026-06-02 14:00", periods=10, freq="5s", tz="UTC")
    px = pd.Series(102.0, index=idx)
    out = volume_profile.compute_prior_session(prev, px, timestep="5s")
    assert out["vp_yday_poc_dist"].nunique(dropna=True) == 1
    assert out["vp_yday_in_value_area"].iloc[0] in (0.0, 1.0)


def test_build_ext_frame_always_emits_all_cols():
    from src.features.feat_ext import EXT_COLS, build_ext_frame

    idx = pd.date_range("2026-06-02 14:00", periods=12, freq="5s", tz="UTC")
    empty = pd.DataFrame()
    px = pd.Series(dtype="float64", index=idx)
    out = build_ext_frame(
        grid=idx, price=px, candles_1m=empty, tbbo=empty,
        prev_tbbo=empty, prior_hl=empty, vix_1m=empty,
        daily_iv=empty, session_date=date(2026, 6, 2),
    )
    assert list(out.columns) == list(EXT_COLS) or set(EXT_COLS).issubset(out.columns)


def test_iv_rank_365():
    idx = pd.date_range("2026-06-02 14:00", periods=5, freq="5s", tz="UTC")
    iv = pd.Series([0.20, 0.21, 0.22, 0.23, 0.24], index=idx)
    daily = pd.DataFrame({
        "date": pd.date_range("2025-06-01", periods=200, freq="D"),
        "iv": np.linspace(0.10, 0.30, 200),
    })
    out = vix_mod.compute_iv_rank(iv, daily, date(2026, 6, 2))
    assert out["iv_rank_365"].between(0, 100).all()
    assert out["iv_rank_365"].iloc[-1] > out["iv_rank_365"].iloc[0]
