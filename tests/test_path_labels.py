"""Tests for honest 1s-path labels and dense targets (src/training/path_labels.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.path_labels import (
    BIG_DOWN, BIG_UP, CENSORED, NO_SIGNAL, STOP, TARGET,
    PathLabelConfig, iter_session_labels, label_session, session_r, summarise,
)
from src.features.session_path import prepare_path, session_grid as _session_grid

_OPEN = "2026-03-02 14:30:00+00:00"     # 08:30 America/Chicago (CST, UTC-6)


def _grid(closes, *, highs=None, lows=None, start=_OPEN) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="1s", tz="UTC")
    c = np.asarray(closes, dtype="float64")
    return _session_grid(prepare_path(pd.DataFrame(
        {"high": c if highs is None else np.asarray(highs, dtype="float64"),
         "low": c if lows is None else np.asarray(lows, dtype="float64"),
         "close": c}, index=idx)))


def _cfg(**kw) -> PathLabelConfig:
    base = dict(mode="fixed", target_points=10.0, stop_points=5.0, horizon_seconds=60,
                n_buckets=6, entry_start="08:30", entry_end="08:31",
                vol_window_seconds=10, vol_lookback_sessions=2, min_stop_ticks=8.0)
    base.update(kw)
    return PathLabelConfig(**base)


# ── config invariants ─────────────────────────────────────────────────────────

def test_stop_must_be_tighter_than_target_or_labels_overlap():
    with pytest.raises(ValueError, match="stop_points"):
        PathLabelConfig(mode="fixed", target_points=10.0, stop_points=10.0)
    with pytest.raises(ValueError, match="stop_r"):
        PathLabelConfig(mode="vol_normalized", target_r=0.5, stop_r=0.5)


def test_horizon_must_divide_into_buckets():
    with pytest.raises(ValueError, match="divisible"):
        PathLabelConfig(horizon_seconds=600, n_buckets=7)


def test_bucket_seconds_is_derived():
    assert PathLabelConfig(horizon_seconds=600, n_buckets=20).bucket_seconds == 30


# ── first-touch labelling ─────────────────────────────────────────────────────

def test_target_before_stop_is_big_up():
    g = _grid([100.0] * 10 + [110.0] + [100.0] * 60)
    out = label_session(g, np.nan, _cfg())
    assert out["label"].iloc[0] == BIG_UP
    assert out["event_up"].iloc[0] == TARGET
    assert out["offset_up"].iloc[0] == 10


def test_downward_target_is_big_down():
    g = _grid([100.0] * 5 + [90.0] + [100.0] * 60)
    out = label_session(g, np.nan, _cfg())
    assert out["label"].iloc[0] == BIG_DOWN
    assert out["event_down"].iloc[0] == TARGET


def test_stop_first_is_no_signal_even_when_the_target_prints_later():
    # dips 5 (stop) at t=3, then runs to +10 at t=20 — a close-only labeller calls this a win
    g = _grid([100.0] * 3 + [95.0] + [100.0] * 16 + [110.0] + [100.0] * 60)
    out = label_session(g, np.nan, _cfg())
    assert out["label"].iloc[0] == NO_SIGNAL
    assert out["event_up"].iloc[0] == STOP


def test_intrabar_stop_is_seen_when_the_close_hides_it():
    """The bias this module exists to remove: a bar closing +10 that traded −6 first."""
    closes = [100.0, 110.0] + [110.0] * 60
    highs = [100.0, 110.0] + [110.0] * 60
    lows = [100.0, 94.0] + [110.0] * 60
    out = label_session(_grid(closes, highs=highs, lows=lows), np.nan, _cfg())
    assert out["label"].iloc[0] == NO_SIGNAL          # stop wins the straddled bar
    assert bool(out["ambiguous"].iloc[0]) is True
    optimistic = label_session(_grid(closes, highs=highs, lows=lows), np.nan,
                               _cfg(tie_policy="target_first"))
    assert optimistic["label"].iloc[0] == BIG_UP      # the optimistic bound


def test_neither_barrier_is_censored_not_negative():
    g = _grid([100.0] * 80)
    out = label_session(g, np.nan, _cfg())
    assert out["label"].iloc[0] == NO_SIGNAL
    assert out["event_up"].iloc[0] == CENSORED
    assert out["offset_up"].iloc[0] == 60             # the full horizon


def test_up_and_down_labels_are_mutually_exclusive():
    g = _grid(list(100 + np.random.default_rng(0).normal(0, 3, 400)))
    out = label_session(g, np.nan, _cfg(entry_end="08:35"))
    assert not ((out["event_up"] == TARGET) & (out["event_down"] == TARGET)).any()


# ── dense targets ─────────────────────────────────────────────────────────────

def test_hazard_bucket_matches_the_touch_time():
    # horizon 60s / 6 buckets = 10s per bucket; a touch at t=25 lands in bucket 2
    g = _grid([100.0] * 24 + [110.0] + [100.0] * 60)
    out = label_session(g, np.nan, _cfg())
    assert out["offset_up"].iloc[0] == 24
    assert out["bucket_up"].iloc[0] == (24 - 1) // 10


def test_censored_entries_get_the_out_of_range_bucket():
    """A censored row must not teach the hazard head a real bucket."""
    cfg = _cfg()
    out = label_session(_grid([100.0] * 80), np.nan, cfg)
    assert out["event_up"].iloc[0] == CENSORED
    assert out["bucket_up"].iloc[0] == cfg.n_buckets


def test_excursions_are_recorded_on_flat_bars_too():
    """The dense signal: MFE/MAE exist even where the sparse label is NO_SIGNAL."""
    highs = [100.0, 103.0] + [100.0] * 70
    lows = [100.0, 100.0, 98.0] + [100.0] * 69
    out = label_session(_grid([100.0] * 72, highs=highs, lows=lows), np.nan, _cfg())
    assert out["label"].iloc[0] == NO_SIGNAL
    assert out["mfe_up_points"].iloc[0] == pytest.approx(3.0)
    assert out["mae_up_points"].iloc[0] == pytest.approx(2.0)


def test_r_units_scale_the_excursions():
    highs = [100.0, 106.0] + [100.0] * 70
    out = label_session(_grid([100.0] * 72, highs=highs),
                        20.0, _cfg(mode="vol_normalized", target_r=0.5, stop_r=0.25))
    # R=20 → target 10, stop 5, and a 6-point MFE is 0.3R
    assert out["target_points"].iloc[0] == pytest.approx(10.0)
    assert out["mfe_up_r"].iloc[0] == pytest.approx(0.3)


def test_truncated_windows_are_flagged():
    """An entry too close to the grid end cannot be resolved and must be marked."""
    # 100 bars, 60s entry window, 60s horizon: entry 0 resolves, entry 59 does not.
    out = label_session(_grid([100.0] * 100), np.nan, _cfg())
    assert bool(out["truncated"].iloc[0]) is False
    assert bool(out["truncated"].iloc[-1]) is True


# ── vol normalisation and the microstructure floor ────────────────────────────

def test_vol_normalized_barriers_scale_with_r():
    cfg = _cfg(mode="vol_normalized", target_r=0.743, stop_r=0.0891)
    lo = label_session(_grid([100.0] * 80), 10.0, cfg)
    hi = label_session(_grid([100.0] * 80), 60.0, cfg)
    assert lo["target_points"].iloc[0] == pytest.approx(7.43)
    assert hi["target_points"].iloc[0] == pytest.approx(44.58)
    assert hi["stop_points"].iloc[0] / lo["stop_points"].iloc[0] == pytest.approx(6.0)


def test_min_stop_ticks_marks_an_era_unusable():
    """2011's normalised stop is ~2 ticks — inside the spread, so not trainable."""
    cfg = _cfg(mode="vol_normalized", target_r=0.743, stop_r=0.0891, min_stop_ticks=8.0)
    old = label_session(_grid([100.0] * 80), 6.3, cfg)      # 2011: R ~= 6.3 pt
    new = label_session(_grid([100.0] * 80), 67.3, cfg)     # 2026: R ~= 67.3 pt
    assert old["stop_ticks"].iloc[0] < 8.0 and bool(old["usable"].iloc[0]) is False
    assert new["stop_ticks"].iloc[0] > 8.0 and bool(new["usable"].iloc[0]) is True


def test_session_r_is_the_median_block_range():
    highs = [100.0] * 10 + [104.0] * 10 + [102.0] * 10
    lows = [100.0] * 30
    g = _grid([100.0] * 30, highs=highs, lows=lows)
    # three 10s blocks with ranges 0, 4, 2 → median 2
    assert session_r(g, _cfg(vol_window_seconds=10, entry_end="08:31")) == pytest.approx(2.0)


def test_session_r_returns_nan_when_the_window_is_too_short():
    assert np.isnan(session_r(_grid([100.0] * 5), _cfg(vol_window_seconds=600)))


# ── streaming builder ─────────────────────────────────────────────────────────

def _session(day: str, closes) -> tuple[str, pd.DataFrame]:
    return day, _grid(closes, start=f"{day} 14:30:00+00:00")


def _breathing(n: int = 100, amp: float = 1.0) -> list[float]:
    """A session with a small but non-zero range, so R is well-defined and > 0."""
    return list(100.0 + amp * np.sin(np.arange(n) * np.pi / 5))


def test_vol_mode_skips_sessions_until_history_exists():
    """A barrier must be knowable at the open, so R uses prior sessions only."""
    days = ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"]
    sessions = [_session(d, _breathing()) for d in days]
    out = list(iter_session_labels(sessions, _cfg(mode="vol_normalized",
                                                  vol_lookback_sessions=2)))
    assert [k for k, _ in out] == days[2:]          # first two seed the yardstick


def test_fixed_mode_labels_from_the_first_session():
    sessions = [_session("2026-03-02", _breathing())]
    out = list(iter_session_labels(sessions, _cfg(mode="fixed")))
    assert len(out) == 1


def test_r_is_causal_not_contemporaneous():
    """Session 3's barrier must come from sessions 1-2, not from session 3 itself."""
    cfg = _cfg(mode="vol_normalized", vol_lookback_sessions=2, vol_window_seconds=10)
    quiet = _breathing(amp=1.0)          # R ~ 2 points
    wild = list(100 + np.arange(100) * 0.5)   # R ~ 5 points per 10s block
    sessions = [_session("2026-03-02", quiet), _session("2026-03-03", quiet),
                _session("2026-03-04", wild)]
    out = dict(iter_session_labels(sessions, cfg))
    r_used = out["2026-03-04"]["r_value"].iloc[0]
    r_quiet = session_r(_grid(quiet), cfg)
    r_wild = session_r(_grid(wild), cfg)
    assert r_used == pytest.approx(r_quiet)       # priced off the prior quiet sessions
    assert r_used < r_wild                        # not off its own realised range


def test_summarise_reports_per_year_geography():
    frames = [label_session(_grid([100.0] * 10 + [110.0] + [100.0] * 60), np.nan, _cfg())]
    s = summarise(frames)
    assert list(s["year"]) == [2026]
    assert s["dir_rate"].iloc[0] > 0
    assert "stop_ticks" in s.columns


def test_summarise_handles_no_frames():
    assert summarise([]).empty
