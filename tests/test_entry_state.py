"""Tests for Layer 1 entry-state features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.cash_session_profile import LevelConfig, SessionLevels
from src.training.auction_levels import ProfileConfig, session_profile
from src.training.entry_state import (
    DIRECTIONAL, PREFIX, EntryStateConfig, build_entry_state,
)


def _level_hist(n: int = 3, lo: float = 19960.0, val: float = 19985.0,
                poc: float = 20000.0, vah: float = 20015.0,
                hi: float = 20040.0) -> list[SessionLevels]:
    """Synthetic multi-session history bracketing the ``_tape()`` price range."""
    return [SessionLevels(session=pd.Timestamp(f"2026-04-{20 - i:02d}"),
                          lo=lo, val=val, poc=poc, vah=vah, hi=hi)
            for i in range(n)]

# Non-order-flow, non-prior-session, non-candle, non-clustered-bound columns
# may never be entirely NaN in a session with prints; order-flow/prior/gap
# need external data the fixtures below omit by default, atr/realized-vol
# need candles_1m, and bound/channel need level_hist.
_GATED = ("cvd", "ofi", "delta", "prior", "gap", "atr_r", "realized_vol",
         "bound", "channel")


def _tape(n_prints: int = 20_000, seed: int = 5, start: str = "2026-05-04 13:30:00",
         span_seconds: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    offsets = np.sort(rng.uniform(0, span_seconds, n_prints))
    ts = pd.DatetimeIndex(
        pd.Timestamp(start, tz="UTC") + pd.to_timedelta(offsets, unit="s"), name="ts")
    price = 20000 + np.cumsum(rng.normal(0, 0.3, n_prints))
    size = rng.integers(1, 30, n_prints).astype("float64")
    side = rng.choice([1.0, -1.0], n_prints)
    return pd.DataFrame({"price": price, "size": size, "side": side}, index=ts)


def _bars(n: int = 4000, seed: int = 5, start: str = "2026-05-04 13:30:00") -> pd.DataFrame:
    """1s OHLCV bars, only used to build a *prior* session's Profile fixture."""
    rng = np.random.default_rng(seed)
    close = 20000 + np.cumsum(rng.normal(0, 1.2, n))
    half = np.abs(rng.normal(0, 0.5, n))
    idx = pd.date_range(start, periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + half, "low": close - half,
                         "close": close, "volume": rng.integers(1, 30, n)}, index=idx)


def _grid(tape: pd.DataFrame, span_seconds: int = 4000) -> pd.DataFrame:
    start = tape.index[0].floor("s")
    idx = pd.date_range(start, periods=span_seconds, freq="1s", tz="UTC")
    return pd.DataFrame(index=idx)


def _entries(grid: pd.DataFrame, step: int = 60) -> np.ndarray:
    return np.arange(step, len(grid) - step, step)


def test_builds_one_row_per_entry_with_prefixed_columns():
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)
    es = build_entry_state(tape, grid, ei, r_value=40.0)
    assert len(es) == len(ei)
    assert (es["ts"].to_numpy() == grid.index[ei].to_numpy()).all()
    feat = [c for c in es.columns if c != "ts"]
    assert feat and all(c.startswith(PREFIX) for c in feat)


def test_no_column_is_entirely_nan_without_order_flow_or_candles():
    """Order-flow, prior-session and candle-derived columns may be NaN."""
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    structural = [c for c in es.columns
                  if c != "ts" and not any(k in c for k in _GATED)]
    assert es[structural].notna().any().all()


def test_order_flow_is_nan_without_hive():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    for c in ("es_cvd_session", "es_ofi_cum", "es_delta_1m"):
        assert es[c].isna().all()


def test_order_flow_populates_from_hive():
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)
    bars = grid.index.floor("15s").unique() + pd.Timedelta("15s")
    rng = np.random.default_rng(1)
    close_15s = tape["price"].reindex(bars, method="ffill").ffill().bfill()
    hive = pd.DataFrame({"of_cvd": np.cumsum(rng.normal(0, 50, len(bars))),
                         "of_delta": rng.normal(0, 50, len(bars)),
                         "of_ofi_cum": np.cumsum(rng.normal(0, 10, len(bars))),
                         "cs_close": close_15s.to_numpy()},
                        index=bars)
    es = build_entry_state(tape, grid, ei, r_value=40.0, hive=hive)
    assert es["es_cvd_session"].notna().any()
    assert es["es_ofi_cum"].notna().any()
    assert es["es_delta_1m"].notna().any()


def test_atr_and_realized_vol_populate_from_candles_1m():
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)
    bucket = pd.date_range(grid.index[0].floor("min"), periods=80, freq="1min", tz="UTC")
    rng = np.random.default_rng(2)
    close = 20000 + np.cumsum(rng.normal(0, 1.0, len(bucket)))
    half = np.abs(rng.normal(0, 0.5, len(bucket)))
    candles_1m = pd.DataFrame({"open": close, "high": close + half, "low": close - half,
                               "close": close, "volume": rng.integers(1, 30, len(bucket))},
                              index=bucket)
    es = build_entry_state(tape, grid, ei, r_value=40.0, candles_1m=candles_1m)
    assert es["es_atr_r"].notna().any()
    assert es["es_realized_vol"].notna().any()


def test_atr_and_realized_vol_nan_without_candles_1m():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    assert es["es_atr_r"].isna().all()
    assert es["es_realized_vol"].isna().all()


def test_features_are_causal():
    """Prints after the last entry's decision instant cannot change any feature."""
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)[:20]
    cutoff_ts = grid.index[ei[-1]]
    full = build_entry_state(tape, grid, ei, r_value=40.0)
    trunc = build_entry_state(tape[tape.index <= cutoff_ts], grid, ei, r_value=40.0)
    cols = [c for c in full.columns if c != "ts"]
    pd.testing.assert_frame_equal(full[cols], trunc[cols], check_exact=False, atol=1e-9)


def test_distances_scale_with_r():
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)
    a = build_entry_state(tape, grid, ei, r_value=20.0)
    b = build_entry_state(tape, grid, ei, r_value=40.0)
    ratio = a["es_vs_session_open"] / b["es_vs_session_open"].replace(0, np.nan)
    assert np.nanmedian(ratio) == pytest.approx(2.0, rel=1e-6)


def test_prior_session_columns_nan_without_prior():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    for c in ("es_vs_prior_poc", "es_open_vs_prior_va", "es_gap_r"):
        assert es[c].isna().all()


def test_prior_session_columns_populate_with_prior():
    prev = _bars(seed=9, start="2026-05-01 13:30:00")
    tape = _tape(seed=5)
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0,
                           prior_profile=session_profile(prev, ProfileConfig()),
                           prior_close=float(prev["close"].iloc[-1]))
    for c in ("es_vs_prior_poc", "es_vs_prior_vah", "es_vs_prior_val",
              "es_open_vs_prior_va", "es_gap_r"):
        assert es[c].notna().all()


def test_bound_columns_nan_without_level_hist():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    for c in ("es_n_bounds", "es_inside_bound", "es_inside_bound_strength",
              "es_inside_bound_tests", "es_inside_bound_breaks",
              "es_inside_bound_rejects", "es_inside_bound_hold_ratio",
              "es_channel_width_r", "es_vs_bound_above_r",
              "es_vs_bound_below_r", "es_pos_in_channel"):
        assert es[c].isna().all()


def test_bound_columns_populate_with_level_hist():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0,
                           level_hist=_level_hist(), lcfg=LevelConfig())
    assert (es["es_n_bounds"] > 0).any()
    assert es["es_inside_bound"].isin([0.0, 1.0]).all()
    # Every row with two flanking bounds gets a signed distance to each.
    flanked = es["es_vs_bound_above_r"].notna() & es["es_vs_bound_below_r"].notna()
    assert flanked.any()
    assert (es.loc[flanked, "es_channel_width_r"] > 0).all()
    assert es["es_pos_in_channel"].dropna().between(-0.5, 0.5).all()
    # tests/breaks/rejects are cumulative counts, never negative.
    for c in ("es_inside_bound_tests", "es_inside_bound_breaks", "es_inside_bound_rejects"):
        assert (es[c].dropna() >= 0).all()


def test_bound_interaction_counts_are_causal():
    """Truncating the tape after the last entry's decision instant cannot
    change any entry's bound-derived feature -- same causal contract the rest
    of the file is held to (:func:`test_features_are_causal`), now exercised
    through the clustered-bound path specifically.
    """
    tape = _tape()
    grid = _grid(tape)
    ei = _entries(grid)[:40]
    hist = _level_hist()
    cutoff_ts = grid.index[ei[-1]]
    full = build_entry_state(tape, grid, ei, r_value=40.0, level_hist=hist)
    trunc = build_entry_state(tape[tape.index <= cutoff_ts], grid, ei,
                              r_value=40.0, level_hist=hist)
    cols = [c for c in full.columns if c != "ts"]
    pd.testing.assert_frame_equal(full[cols], trunc[cols], check_exact=False, atol=1e-9)


def test_value_area_flag_is_consistent_with_distances():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    inside = es["es_in_value_area"] == 1.0
    ok = inside & es["es_vs_dev_vah"].notna()
    assert (es.loc[ok, "es_vs_dev_vah"] <= 1e-9).all()
    assert (es.loc[ok, "es_vs_dev_val"] >= -1e-9).all()


def test_session_extreme_distances_have_correct_sign():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    assert (es["es_vs_session_high"] <= 1e-9).all()
    assert (es["es_vs_session_low"] >= -1e-9).all()


def test_clock_columns_are_chicago_minutes():
    tape = _tape(n_prints=6000, span_seconds=6 * 3600, start="2026-05-04 13:30:00")  # 08:30 CT
    grid = _grid(tape, span_seconds=6 * 3600)
    es = build_entry_state(tape, grid, np.array([0, 3600]), r_value=40.0)
    assert es["es_min_since_open"].iloc[0] == pytest.approx(0.0)
    assert es["es_min_since_open"].iloc[1] == pytest.approx(60.0)
    assert es["es_min_to_close"].iloc[0] == pytest.approx(390.0)


def test_directional_list_matches_emitted_columns():
    """Every name in DIRECTIONAL must exist, or the short mirror silently no-ops."""
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0)
    missing = set(DIRECTIONAL) - set(es.columns)
    assert not missing, f"DIRECTIONAL names not emitted: {sorted(missing)}"


def test_rejects_non_positive_r():
    tape = _tape()
    grid = _grid(tape)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            build_entry_state(tape, grid, _entries(grid), r_value=bad)


def test_no_infinities_survive():
    tape = _tape()
    grid = _grid(tape)
    es = build_entry_state(tape, grid, _entries(grid), r_value=40.0,
                           cfg=EntryStateConfig())
    assert not np.isinf(es.drop(columns=["ts"]).to_numpy(dtype="float64")).any()
