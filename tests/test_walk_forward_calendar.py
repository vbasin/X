"""Expanding calendar walk-forward (ADR-042)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import (
    apply_schedule_to_arrays,
    build_expanding_schedule,
    default_walk_forward,
    overlay_from_board,
)


def test_recommended_6_fold_two_month_blocks():
    sched = build_expanding_schedule(
        start="2025-05-01", stop="2026-06-30",
        train_n=2, train_unit="months",
        test_n=2, test_unit="months",
    )
    assert len(sched) == 6
    assert [s["train"]["start"] for s in sched] == ["2025-05-01"] * 6
    assert [s["train"]["end"] for s in sched] == [
        "2025-06-30", "2025-08-31", "2025-10-31",
        "2025-12-31", "2026-02-28", "2026-04-30",
    ]
    assert [s["oof"]["start"] for s in sched] == [
        "2025-07-01", "2025-09-01", "2025-11-01",
        "2026-01-01", "2026-03-01", "2026-05-01",
    ]
    assert [s["oof"]["end"] for s in sched] == [
        "2025-08-31", "2025-10-31", "2025-12-31",
        "2026-02-28", "2026-04-30", "2026-06-30",
    ]
    assert all(s["val"] is None for s in sched)


def test_expanding_identity_train1_plus_test1_equals_train2():
    sched = build_expanding_schedule(
        start="2025-05-01", stop="2026-06-30",
        train_n=2, train_unit="months",
        test_n=2, test_unit="months",
    )
    for i in range(len(sched) - 1):
        assert sched[i + 1]["train"]["end"] == sched[i]["oof"]["end"]


def test_val_from_train_shrinks_fit_window():
    sched = build_expanding_schedule(
        start="2025-05-01", stop="2026-06-30",
        train_n=2, train_unit="months",
        test_n=2, test_unit="months",
        val_enabled=True, val_n=10, val_unit="days", val_cut_from="train",
    )
    f0 = sched[0]
    assert f0["val"]["end"] == "2025-06-30"
    assert f0["val"]["start"] == "2025-06-21"
    assert f0["train"]["end"] == "2025-06-20"
    assert f0["oof"]["start"] == "2025-07-01"


def test_val_from_test_shrinks_oof():
    sched = build_expanding_schedule(
        start="2025-05-01", stop="2026-06-30",
        train_n=2, train_unit="months",
        test_n=2, test_unit="months",
        val_enabled=True, val_n=1, val_unit="weeks", val_cut_from="test",
    )
    f0 = sched[0]
    assert f0["val"]["start"] == "2025-07-01"
    assert f0["val"]["end"] == "2025-07-07"
    assert f0["oof"]["start"] == "2025-07-08"
    assert f0["train"]["end"] == "2025-06-30"


def test_day_increments_until_stop():
    sched = build_expanding_schedule(
        start="2026-06-01", stop="2026-06-10",
        train_n=3, train_unit="days",
        test_n=2, test_unit="days",
    )
    # train Jun 1-3, test 4-5; then train 1-5 test 6-7; train 1-7 test 8-9; train 1-9 test 10
    assert len(sched) == 4
    assert sched[-1]["oof"]["end"] == "2026-06-10"
    assert sched[-1]["clipped"] is True


def test_too_short_window_raises():
    with pytest.raises(ValueError, match="overruns stop"):
        build_expanding_schedule(
            start="2026-06-01", stop="2026-06-02",
            train_n=2, train_unit="months",
            test_n=1, test_unit="months",
        )


def test_overlay_disabled_is_none():
    assert overlay_from_board({"enabled": False, "start": "2025-05-01",
                               "stop": "2026-06-30", "train_n": 2,
                               "train_unit": "months", "test_n": 2,
                               "test_unit": "months"}) is None


def test_overlay_includes_schedule_and_date_range():
    ov = overlay_from_board(default_walk_forward())
    assert ov["date_range"] == {"start": "2025-05-01", "end": "2026-06-30"}
    assert ov["walk_forward_folds"] == 6
    assert ov["walk_forward"]["mode"] == "expanding_calendar"
    assert ov["walk_forward"]["purge_bars"] == 120
    assert len(ov["walk_forward"]["schedule"]) == 6


def test_apply_schedule_masks_and_purge():
    # 40 consecutive calendar days of 2 bars/day starting 2025-05-01.
    days = pd.date_range("2025-05-01 14:00", periods=40, freq="D", tz="America/Chicago")
    ts = np.repeat(days.tz_convert("UTC").to_numpy(), 8)
    n = len(ts)
    X = np.arange(n, dtype="float32").reshape(-1, 1)
    y = np.ones(n, dtype="int32")
    sched = build_expanding_schedule(
        start="2025-05-01", stop="2025-05-20",
        train_n=5, train_unit="days",
        test_n=5, test_unit="days",
    )
    folds = apply_schedule_to_arrays(
        X, y, ts, sched, ["c"], tz_name="America/Chicago", purge_bars=2,
    )
    assert len(folds) >= 2
    # Train for fold 0 is May 1-5 (40 bars) minus 2 purge = 38.
    assert len(folds[0]["y_train"]) == 38
    assert len(folds[0]["y_oof"]) == 40
    assert len(folds[0]["y_val"]) == 0
    # Expanding: later fold train is longer.
    assert len(folds[1]["y_train"]) > len(folds[0]["y_train"])


# ── interval-aware purge (ADR-052) ───────────────────────────────────────────
#
# purge_bars is a raw ROW count, so it only equals the 10-minute TBM label horizon at the
# bar interval it was written for. The default 120 is 10 minutes at 5s and 2 minutes at
# 1s — under-purging by 8 minutes, which leaks train rows whose forward window overlaps
# the OOF block.

def _ts(freq: str, n: int = 2000) -> np.ndarray:
    return pd.date_range("2026-01-05 14:30", periods=n, freq=freq, tz="UTC").values


def test_probe_bar_seconds_detects_resolution():
    from src.validation.walk_forward import probe_bar_seconds
    assert probe_bar_seconds(_ts("5s")) == pytest.approx(5.0)
    assert probe_bar_seconds(_ts("1s")) == pytest.approx(1.0)


def test_probe_bar_seconds_falls_back_on_degenerate_input():
    from src.validation.walk_forward import probe_bar_seconds
    assert probe_bar_seconds(_ts("5s", n=1), default=7.0) == pytest.approx(7.0)
    assert probe_bar_seconds(None, default=7.0) == pytest.approx(7.0)


def test_purge_seconds_converts_with_the_probed_interval():
    from src.validation.walk_forward import resolve_purge_rows
    assert resolve_purge_rows(_ts("5s"), purge_seconds=600)[0] == 120
    assert resolve_purge_rows(_ts("1s"), purge_seconds=600)[0] == 600
    assert resolve_purge_rows(_ts("30s"), purge_seconds=600)[0] == 20


def test_purge_bars_is_silent_at_5s_and_warns_at_1s(caplog):
    from src.validation.walk_forward import resolve_purge_rows
    with caplog.at_level("WARNING"):
        assert resolve_purge_rows(_ts("5s"), purge_bars=120)[0] == 120
    assert not caplog.records                      # 120 x 5s == the 600s horizon
    with caplog.at_level("WARNING"):
        assert resolve_purge_rows(_ts("1s"), purge_bars=120)[0] == 120
    assert any("leak" in r.message for r in caplog.records)


def test_purge_seconds_takes_precedence_over_purge_bars():
    from src.validation.walk_forward import resolve_purge_rows
    rows, _ = resolve_purge_rows(_ts("1s"), purge_bars=120, purge_seconds=600)
    assert rows == 600


def test_apply_schedule_honours_purge_seconds_at_1s():
    n = 4 * 24 * 60 * 60          # four days of 1s bars
    ts = pd.date_range("2026-01-05 14:30", periods=n, freq="1s", tz="UTC").values
    X = np.zeros((n, 2), dtype="float32")
    y = np.ones(n, dtype="int32")
    sched = [{"fold": 0,
              "train": {"start": "2026-01-05", "end": "2026-01-06"},
              "oof": {"start": "2026-01-07", "end": "2026-01-08"}}]
    by_bars = apply_schedule_to_arrays(X, y, ts, sched, ["a", "b"], purge_bars=120)
    by_secs = apply_schedule_to_arrays(X, y, ts, sched, ["a", "b"], purge_seconds=600)
    # 600 rows purged instead of 120 → 480 fewer train rows, same OOF block
    assert len(by_bars[0]["X_train"]) - len(by_secs[0]["X_train"]) == 480
    assert len(by_bars[0]["X_oof"]) == len(by_secs[0]["X_oof"])


def test_compiled_overlay_carries_purge_seconds():
    """The Custom Test designer's overlay is where the purge reaches the controller."""
    from src.validation.walk_forward import DEFAULT_PURGE_SECONDS, overlay_from_board
    wf = overlay_from_board(default_walk_forward())["walk_forward"]
    assert wf["purge_bars"] == 120                       # unchanged, 5s-correct
    assert wf["purge_seconds"] == DEFAULT_PURGE_SECONDS == 600
