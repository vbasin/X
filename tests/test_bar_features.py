"""Native 1-bar features (15s hive, ADR-075)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.bar_features import (
    BAR_FEATURE_COLS,
    RTH_FLOW_COLS,
    attach_rth_session_features,
    compute_bar_features,
    timestep_delta,
)
from src.features.native_tf import (
    globex_session_date,
    globex_session_window,
    split_by_utc_month,
    write_month_parts,
)
from src.features.order_flow import _compute_ofi
from src.search.custom_test import parse_feature_catalog


def _tbbo(ts, **cols) -> pd.DataFrame:
    n = len(ts)
    base = {
        "ts_event": pd.DatetimeIndex(ts, tz="UTC"),
        "price": cols.get("price", np.full(n, 100.0)),
        "size": cols.get("size", np.ones(n)),
        "side": cols.get("side", ["B"] * n),
        "bid_px": cols.get("bid_px", np.full(n, 99.75)),
        "ask_px": cols.get("ask_px", np.full(n, 100.25)),
        "bid_sz": cols.get("bid_sz", np.full(n, 10.0)),
        "ask_sz": cols.get("ask_sz", np.full(n, 10.0)),
    }
    for k, v in cols.items():
        base[k] = v
    return pd.DataFrame(base)


def test_empty_tbbo_has_schema():
    out = compute_bar_features(pd.DataFrame())
    assert list(out.columns) == ["bar_end_ts", *BAR_FEATURE_COLS]
    assert out.empty


def test_bar_end_ts_is_closed_label():
    ts = pd.date_range("2026-08-25 13:30:00", periods=3, freq="1s", tz="UTC")
    out = compute_bar_features(_tbbo(ts), "15s")
    assert len(out) == 1
    assert out["bar_end_ts"].iloc[0] == pd.Timestamp("2026-08-25 13:30:15", tz="UTC")
    assert out["bar_end_ts"].dt.tz is not None


def test_cs_ohlc_return_range_clv_close():
    # One 15s bar: open 100, high 110, low 90, close 105
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:05",
        "2026-08-25 13:30:10",
        "2026-08-25 13:30:14",
    ], utc=True)
    out = compute_bar_features(_tbbo(
        ts,
        price=[100.0, 110.0, 90.0, 105.0],
        size=[1, 1, 1, 1],
    ), "15s")
    row = out.iloc[0]
    assert row["cs_close"] == 105.0
    assert row["cs_range"] == 20.0
    np.testing.assert_allclose(row["cs_return"], 0.05, rtol=1e-6)
    # CLV = (2*105 - 110 - 90) / 20 = 10/20 = 0.5
    np.testing.assert_allclose(row["cs_clv"], 0.5, rtol=1e-6)
    assert row["sd_tick_count"] == 4
    np.testing.assert_allclose(row["sd_price_impact_per_tick"], 5.0, rtol=1e-6)


def test_clv_nan_when_flat_bar():
    ts = pd.date_range("2026-08-25 13:30:00", periods=2, freq="1s", tz="UTC")
    out = compute_bar_features(_tbbo(ts, price=[100.0, 100.0]), "15s")
    assert np.isnan(out["cs_clv"].iloc[0])
    assert out["cs_range"].iloc[0] == 0.0


def test_ofi_matches_order_flow_module_on_occupied_bars():
    ts = pd.date_range("2026-08-25 13:30:00", periods=4, freq="1s", tz="UTC")
    tbbo = _tbbo(
        ts,
        bid_px=[100.0, 100.0, 100.25, 100.25],
        bid_sz=[10.0, 15.0, 8.0, 8.0],
        ask_px=[100.25, 100.25, 100.50, 100.50],
        ask_sz=[10.0, 10.0, 12.0, 11.0],
        price=[100.25, 100.25, 100.50, 100.50],
        size=[1, 1, 1, 1],
        side=["B", "B", "B", "A"],
    )
    got = compute_bar_features(tbbo, "15s")
    ref = _compute_ofi(tbbo, "15s")
    assert ref is not None
    start = got["bar_end_ts"] - timestep_delta("15s")
    aligned = ref.reindex(start)
    np.testing.assert_allclose(
        got["of_ofi"].to_numpy(dtype="float64"),
        aligned["of_ofi"].to_numpy(dtype="float64"),
        rtol=1e-5, atol=1e-5, equal_nan=True,
    )


def test_same_side_runs_reset_at_bar_boundary():
    # Bar 0: BBBAB → buy max 3, sell max 1
    # Bar 1: BB    → buy max 2 (must not continue the prior buy streak)
    t0 = pd.Timestamp("2026-08-25 13:30:00", tz="UTC")
    ts = [t0 + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3, 4, 15, 16)]
    sides = ["B", "B", "B", "A", "B", "B", "B"]
    out = compute_bar_features(_tbbo(ts, side=sides), "15s")
    assert len(out) == 2
    assert out["ss_buy_run_max"].iloc[0] == 3
    assert out["ss_sell_run_max"].iloc[0] == 1
    assert out["ss_buy_run_max"].iloc[1] == 2
    assert out["ss_sell_run_max"].iloc[1] == 0


def test_trade_location_size_weighted_and_sweep():
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:01",
        "2026-08-25 13:30:02",
    ], utc=True)
    # mid=100, half=0.25 → loc at ask = +1, at bid = -1, through ask = +3
    tbbo = _tbbo(
        ts,
        price=[100.25, 99.75, 100.75],
        size=[2.0, 1.0, 1.0],
        bid_px=[99.75, 99.75, 99.75],
        ask_px=[100.25, 100.25, 100.25],
        side=["B", "A", "B"],
    )
    out = compute_bar_features(tbbo, "15s").iloc[0]
    # loc: +1, -1, +3; weighted (2*1 + 1*(-1) + 1*3) / 4 = 4/4 = 1
    np.testing.assert_allclose(out["tl_trade_location"], 1.0, rtol=1e-5)
    # sweep only the last print (size 1 of 4)
    np.testing.assert_allclose(out["tl_aggressive_pct"], 0.25, rtol=1e-5)


def test_sd_bid_ask_delta_uses_first_last_of_bar():
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:05",
        "2026-08-25 13:30:10",
    ], utc=True)
    out = compute_bar_features(_tbbo(
        ts,
        bid_sz=[10.0, 7.0, 4.0],
        ask_sz=[8.0, 9.0, 12.0],
    ), "15s").iloc[0]
    assert out["sd_bid_sz_delta"] == -6.0
    assert out["sd_ask_sz_delta"] == 4.0


def test_timestep_1m_uses_one_minute_bars():
    ts = pd.date_range("2026-08-25 13:30:00", periods=5, freq="10s", tz="UTC")
    out = compute_bar_features(_tbbo(ts), "1m")
    assert len(out) == 1
    assert out["bar_end_ts"].iloc[0] == pd.Timestamp("2026-08-25 13:31:00", tz="UTC")


def test_sparse_bars_do_not_fill_empty_slots():
    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:31",  # two 15s gaps later
    ], utc=True)
    out = compute_bar_features(_tbbo(ts), "15s")
    assert len(out) == 2
    ends = list(out["bar_end_ts"])
    assert ends == [
        pd.Timestamp("2026-08-25 13:30:15", tz="UTC"),
        pd.Timestamp("2026-08-25 13:30:45", tz="UTC"),
    ]


def test_globex_session_window_summer_and_winter():
    start, end = globex_session_window(date(2026, 8, 26))
    assert start.isoformat() == "2026-08-25T22:00:00+00:00"
    assert end.isoformat() == "2026-08-26T22:00:00+00:00"
    start, end = globex_session_window(date(2026, 1, 15))
    assert start.isoformat() == "2026-01-14T23:00:00+00:00"
    assert end.isoformat() == "2026-01-15T23:00:00+00:00"
    chi = ZoneInfo("America/Chicago")
    open_sun = datetime(2025, 4, 6, 17, 0, tzinfo=chi)
    assert globex_session_date(open_sun) == date(2025, 4, 7)


def test_write_month_parts_splits_on_size_cap(tmp_path):
    ts = pd.date_range("2026-04-07 13:30:00", periods=400, freq="15s", tz="UTC")
    tbbo = _tbbo(
        ts,
        price=np.linspace(100, 120, 400),
        size=np.full(400, 2.0),
        side=["B", "A"] * 200,
    )
    feats = compute_bar_features(tbbo, "15s")
    assert len(feats) == 400
    parts = write_month_parts(feats, tmp_path, timeframe="15s", max_bytes=8_000)
    assert len(parts) >= 2
    assert [p.name for p in parts] == [f"features_15s_part{i}.parquet" for i in range(1, len(parts) + 1)]
    for p in parts:
        assert p.stat().st_size > 0
        assert p.stat().st_size <= 8_000 or pd.read_parquet(p).shape[0] == 1
    restored = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    assert len(restored) == 400
    assert restored["bar_end_ts"].is_monotonic_increasing
    assert "bar_end_ts" in restored.columns
    rebuild = write_month_parts(feats, tmp_path, timeframe="15s", max_bytes=50_000_000)
    assert len(rebuild) == 1
    leftover = list(tmp_path.glob("features_15s_part*.parquet"))
    assert len(leftover) == 1


def test_split_by_utc_month():
    df = pd.DataFrame({
        "bar_end_ts": pd.to_datetime([
            "2025-04-30 23:59:45+00:00",
            "2025-05-01 00:00:00+00:00",
            "2025-05-01 00:00:15+00:00",
        ], utc=True),
        "cs_close": [1.0, 2.0, 3.0],
    })
    months = split_by_utc_month(df)
    assert set(months) == {(2025, 4), (2025, 5)}
    assert len(months[(2025, 4)]) == 1
    assert len(months[(2025, 5)]) == 2


def test_catalog_15s_parses_like_5s_catalog():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs/reference/FEATURE_CATALOG_15s.md").read_text()
    parsed = parse_feature_catalog(text)
    names = {f["name"] for f in parsed["features"]}
    for col in BAR_FEATURE_COLS:
        assert col in names, col
    prefixes = {g["prefix"] for g in parsed["groups"]}
    assert prefixes == {"of_", "cs_", "tl_", "ss_", "sd_", "lg_", "st_"}
    assert sum(g["count"] for g in parsed["groups"]) == 23
    assert "lg_target_v_atr" in names and "st_target_v_close" in names
    for col in (
        "of_delta", "of_cvd", "of_ofi_cum", "of_ofi_cum_slope_5",
        "cs_return_slope_5", "cs_atr_5",
    ):
        assert col in names, col


def test_of_delta_signed_volume_matches_order_flow():
    from src.features.order_flow import compute as of_compute

    ts = pd.to_datetime([
        "2026-08-25 13:30:00",
        "2026-08-25 13:30:01",
        "2026-08-25 13:30:02",
        "2026-08-25 13:30:03",
    ], utc=True)
    tbbo = _tbbo(
        ts,
        size=[2.0, 3.0, 1.0, 4.0],
        side=["B", "B", "A", "N"],
    )
    got = compute_bar_features(tbbo, "15s")
    # B+2 + B+3 + A−1 + N0 = 4
    np.testing.assert_allclose(got["of_delta"].iloc[0], 4.0, rtol=1e-6)
    ref = of_compute(tbbo, "15s")
    start = got["bar_end_ts"] - timestep_delta("15s")
    aligned = ref.reindex(start)
    np.testing.assert_allclose(
        got["of_delta"].to_numpy(dtype="float64"),
        aligned["of_delta"].to_numpy(dtype="float64"),
        rtol=1e-5, atol=1e-5, equal_nan=True,
    )


def _chi_bars(times_ct, **cols) -> pd.DataFrame:
    chi = ZoneInfo("America/Chicago")
    ts = [
        datetime.fromisoformat(t).replace(tzinfo=chi).astimezone(timezone.utc)
        for t in times_ct
    ]
    n = len(ts)
    frame = {
        "bar_end_ts": ts,
        "of_delta": cols.get("of_delta", np.ones(n)),
        "of_ofi": cols.get("of_ofi", np.ones(n)),
        "cs_return": cols.get("cs_return", np.full(n, 0.01)),
        "cs_close": cols.get("cs_close", np.full(n, 100.0)),
        "cs_range": cols.get("cs_range", np.full(n, 1.0)),
        "cs_clv": cols.get("cs_clv", np.zeros(n)),
    }
    for k, v in cols.items():
        frame[k] = v
    return pd.DataFrame(frame)


def test_cvd_and_ofi_cum_rth_window_and_next_day_reset():
    df = _chi_bars(
        [
            "2026-08-26 08:29:45",
            "2026-08-26 08:30:00",
            "2026-08-26 08:30:15",
            "2026-08-26 15:00:00",
            "2026-08-26 15:00:15",
            "2026-08-27 08:30:00",
            "2026-08-27 08:30:15",
        ],
        of_delta=[10.0, 1.0, 2.0, 3.0, 99.0, 4.0, 5.0],
        of_ofi=[10.0, 1.0, 2.0, 3.0, 99.0, 4.0, 5.0],
    )
    out = attach_rth_session_features(df, overwrite=True)
    assert np.isnan(out["of_cvd"].iloc[0])
    np.testing.assert_allclose(out["of_cvd"].iloc[1], 1.0)
    np.testing.assert_allclose(out["of_cvd"].iloc[2], 3.0)
    np.testing.assert_allclose(out["of_cvd"].iloc[3], 6.0)
    assert np.isnan(out["of_cvd"].iloc[4])
    np.testing.assert_allclose(out["of_cvd"].iloc[5], 4.0)
    np.testing.assert_allclose(out["of_cvd"].iloc[6], 9.0)
    np.testing.assert_allclose(out["of_ofi_cum"].iloc[2], 3.0)
    np.testing.assert_allclose(out["of_ofi_cum"].iloc[5], 4.0)
    for c in RTH_FLOW_COLS:
        assert c in out.columns


def test_cvd_window_uses_america_chicago_in_winter():
    df = _chi_bars(
        ["2026-01-15 08:29:45", "2026-01-15 08:30:00"],
        of_delta=[7.0, 2.0],
        of_ofi=[7.0, 2.0],
    )
    out = attach_rth_session_features(df, overwrite=True)
    assert np.isnan(out["of_cvd"].iloc[0])
    np.testing.assert_allclose(out["of_cvd"].iloc[1], 2.0)


def test_ofi_cum_slope_5_relative_change():
    # Six consecutive RTH bars starting 08:30 CT; of_ofi = 1 each → cum 1..6
    times = ["2026-08-26 08:30:00"]
    t0 = datetime.fromisoformat("2026-08-26 08:30:00")
    for i in range(1, 6):
        times.append((t0 + pd.Timedelta(seconds=15 * i)).isoformat())
    df = _chi_bars(times, of_delta=np.ones(6), of_ofi=np.ones(6))
    out = attach_rth_session_features(df, overwrite=True)
    assert np.isnan(out["of_ofi_cum_slope_5"].iloc[4])
    # t=5 (0-based): cum=6, t-5 cum=1 → (6-1)/1 = 5
    np.testing.assert_allclose(out["of_ofi_cum_slope_5"].iloc[5], 5.0)
    np.testing.assert_allclose(out["of_ofi_cum"].iloc[5], 6.0)


def test_cs_return_slope_5_and_zero_denom_nan():
    times = ["2026-08-26 08:30:00"]
    t0 = datetime.fromisoformat("2026-08-26 08:30:00")
    for i in range(1, 6):
        times.append((t0 + pd.Timedelta(seconds=15 * i)).isoformat())
    rets = np.array([0.0, 0.02, 0.02, 0.02, 0.02, 0.04])
    df = _chi_bars(times, cs_return=rets, of_delta=np.ones(6), of_ofi=np.ones(6))
    out = attach_rth_session_features(df, overwrite=True)
    # t-5 return is 0 → denom NaN
    assert np.isnan(out["cs_return_slope_5"].iloc[5])
    rets2 = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.04])
    df2 = _chi_bars(times, cs_return=rets2, of_delta=np.ones(6), of_ofi=np.ones(6))
    out2 = attach_rth_session_features(df2, overwrite=True)
    np.testing.assert_allclose(out2["cs_return_slope_5"].iloc[5], (0.04 - 0.02) / 0.02)


def test_cs_atr_5_is_sma_of_true_range():
    times = ["2026-08-26 08:30:00"]
    t0 = datetime.fromisoformat("2026-08-26 08:30:00")
    for i in range(1, 5):
        times.append((t0 + pd.Timedelta(seconds=15 * i)).isoformat())
    # close 100,101,103,102,104; range 2 → H/L from CLV=0: mid of range at close
    # CLV=0, range=2 → H=C+1, L=C-1
    close = np.array([100.0, 101.0, 103.0, 102.0, 104.0])
    df = _chi_bars(
        times,
        cs_close=close,
        cs_range=np.full(5, 2.0),
        cs_clv=np.zeros(5),
        of_delta=np.ones(5),
        of_ofi=np.ones(5),
    )
    out = attach_rth_session_features(df, overwrite=True)
    assert np.isnan(out["cs_atr_5"].iloc[3])
    high = close + 1.0
    low = close - 1.0
    prev = np.array([np.nan, 100.0, 101.0, 103.0, 102.0])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = 2.0  # no prev close → range
    np.testing.assert_allclose(out["cs_atr_5"].iloc[4], np.mean(tr), rtol=1e-5)
