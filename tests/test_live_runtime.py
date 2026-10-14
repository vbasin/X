"""Always-on live feature runtime: heartbeat freshness + warm helpers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features import live_runtime as lr


def test_is_runtime_fresh_true(tmp_path, monkeypatch):
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    monkeypatch.setattr(lr, "_HEARTBEAT_FRESH_S", 45.0)
    hb.write_text(json.dumps({
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "last_bar": "2026-08-14T15:00:00+00:00",
    }))
    assert lr.is_runtime_fresh() is True


def test_is_runtime_fresh_stale(tmp_path, monkeypatch):
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    monkeypatch.setattr(lr, "_HEARTBEAT_FRESH_S", 45.0)
    hb.write_text(json.dumps({
        "ok": True,
        "ts": "2020-01-01T00:00:00+00:00",
    }))
    assert lr.is_runtime_fresh() is False


def test_is_runtime_fresh_not_ok(tmp_path, monkeypatch):
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    hb.write_text(json.dumps({
        "ok": False,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))
    assert lr.is_runtime_fresh() is False


def test_resolve_feature_day_uses_disk_when_runtime_fresh(tmp_path, monkeypatch):
    """When heartbeat is fresh, resolve must not call extend_date."""
    import src.webui.live as live_mod
    from datetime import date

    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    hb.write_text(json.dumps({
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))

    pdir = tmp_path / "pq"
    pdir.mkdir()
    idx = pd.date_range("2026-08-14 14:00:00", periods=3, freq="5s", tz="UTC")
    df = pd.DataFrame({"_close": [1.0, 2.0, 3.0], "rv_x": [0.1, 0.2, 0.3]}, index=idx)
    path = pdir / "features_2026-08-14.parquet"
    df.to_parquet(path)

    monkeypatch.setattr(live_mod, "_parquet_path", lambda d: pdir / f"features_{d}.parquet")
    monkeypatch.setattr(live_mod, "session_context", lambda _oid=None: {
        "session_date": date(2026, 8, 14),
        "in_session": True,
        "session_closed": False,
        "timezone": "America/Chicago",
        "trading_start": "08:30",
        "trading_end": "14:45",
    })

    class _Boom:
        def extend_date(self, *_a, **_k):
            raise AssertionError("extend_date must not run when runtime fresh")

        def build_date(self, *_a, **_k):
            raise AssertionError("build_date must not run")

    monkeypatch.setattr(live_mod, "_engine", lambda: _Boom())

    out = live_mod.resolve_feature_day()
    assert out["source"] == "live_runtime"
    assert len(out["df"]) == 3


def _session(monkeypatch, live_mod, day):
    monkeypatch.setattr(live_mod, "session_context", lambda _oid=None: {
        "session_date": day,
        "in_session": True,
        "session_closed": False,
        "timezone": "America/Chicago",
        "trading_start": "08:30",
        "trading_end": "14:45",
    })


def test_resolve_feature_day_extends_when_runtime_stale(tmp_path, monkeypatch):
    """A stale heartbeat must extend the hive day, not rebuild it."""
    import src.webui.live as live_mod
    from datetime import date

    day = date(2026, 8, 14)
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    hb.write_text(json.dumps({"ok": True, "ts": "2020-01-01T00:00:00+00:00"}))

    idx = pd.date_range("2026-08-14 14:00:00", periods=3, freq="5s", tz="UTC")
    frame = pd.DataFrame({"_close": [1.0, 2.0, 3.0]}, index=idx)
    path = tmp_path / "features_2026-08-14.parquet"
    frame.to_parquet(path)
    monkeypatch.setattr(live_mod, "_load_feature_frame", lambda _d: (frame, path))
    _session(monkeypatch, live_mod, day)

    calls: list[str] = []

    class _Eng:
        def extend_date(self, d, **_k):
            calls.append(f"extend:{d}")
            return path

        def build_date(self, d):
            calls.append(f"build:{d}")
            raise AssertionError("build_date must not run when the hive day exists")

    monkeypatch.setattr(live_mod, "_engine", lambda: _Eng())
    out = live_mod.resolve_feature_day()
    assert calls == [f"extend:{day}"]
    assert out["source"] == "live_extend"
    assert len(out["df"]) == 3


def test_resolve_feature_day_force_rebuild_uses_build_date(tmp_path, monkeypatch):
    import src.webui.live as live_mod
    from datetime import date

    day = date(2026, 8, 14)
    hb = tmp_path / "hb.json"
    monkeypatch.setattr(lr, "_HEARTBEAT_PATH", hb)
    hb.write_text(json.dumps({
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))
    idx = pd.date_range("2026-08-14 14:00:00", periods=2, freq="5s", tz="UTC")
    frame = pd.DataFrame({"_close": [1.0, 2.0]}, index=idx)
    path = tmp_path / "features.parquet"
    monkeypatch.setattr(live_mod, "_load_feature_frame", lambda _d: (frame, path))
    _session(monkeypatch, live_mod, day)

    calls: list[str] = []

    class _Eng:
        def extend_date(self, *_a, **_k):
            raise AssertionError("extend_date must not run on force_rebuild")

        def build_date(self, d):
            calls.append(f"build:{d}")
            return path

    monkeypatch.setattr(live_mod, "_engine", lambda: _Eng())
    out = live_mod.resolve_feature_day(force_rebuild=True)
    assert calls == [f"build:{day}"]
    assert out["source"] == "live_build"
