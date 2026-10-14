"""Feature-engine query streaming: compact batches, no client-side fetchall."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.features.engine import FeatureEngine, frame_from_cursor


class _Cursor:
    def __init__(self, batches: list[list[tuple]]) -> None:
        self._batches = list(batches)
        self.fetches: list[int] = []

    def fetchmany(self, n: int):
        self.fetches.append(n)
        if not self._batches:
            return []
        return self._batches.pop(0)

    def fetchall(self):
        raise AssertionError("fetchall buffers the whole result")


def test_frame_from_cursor_compacts_batches():
    ts = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
    batches = [
        [(None, None, None, None)],  # leading nulls must not lock the column as object
        [(ts, "NQZ6", 20000.25, 12), (ts, "ESZ6", 5700.5, None)],
        [(ts, "NQZ6", 20000.5, 4)],
        [],
    ]
    cur = _Cursor(batches)
    df = frame_from_cursor(cur, ["ts", "symbol", "bid_px", "bid_sz"], batch=2)
    assert cur.fetches  # streamed
    assert len(df) == 4
    assert str(df["symbol"].dtype) == "category"
    assert pd.isna(df["symbol"].iloc[0])
    assert list(df["symbol"].iloc[1:].astype(str)) == ["NQZ6", "ESZ6", "NQZ6"]
    assert df["bid_px"].dtype == "float64"
    assert df["bid_px"].iloc[1:].tolist() == [20000.25, 5700.5, 20000.5]
    assert pd.isna(df["bid_sz"].iloc[0]) and pd.isna(df["bid_sz"].iloc[2])
    assert df["bid_sz"].iloc[1] == 12
    assert df["ts"].dtype == "datetime64[ns, UTC]"
    assert pd.isna(df["ts"].iloc[0])
    assert df["ts"].iloc[1] == pd.Timestamp(ts)
    # Repeated text is codes, not one Python str per row.
    assert df["symbol"].cat.categories.tolist() == ["NQZ6", "ESZ6"]


def test_frame_from_cursor_empty():
    df = frame_from_cursor(_Cursor([[]]), ["symbol", "bid_px"])
    assert list(df.columns) == ["symbol", "bid_px"]
    assert df.empty


def test_query_uses_named_cursor_not_fetchall(monkeypatch):
    import psycopg

    seen: dict[str, object] = {}

    class Cur(_Cursor):
        def __init__(self) -> None:
            super().__init__([[("NQZ6", 1.5)], []])
            self.description = [("symbol",), ("bid_px",)]
            self.itersize = None

        def execute(self, sql, params):
            seen["sql"] = sql
            seen["params"] = params

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Conn:
        def cursor(self, name=None):
            assert name, "a named cursor streams; an unnamed one buffers the result"
            seen["name"] = name
            cur = Cur()
            seen["cur"] = cur
            return cur

        def transaction(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(
        "src.features.engine.get_dsn", lambda: "postgresql://example/db",
    )
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: Conn())
    eng = FeatureEngine()
    df = eng._query("SELECT x", ("p",))
    assert seen["sql"] == "SELECT x"
    assert seen["params"] == ("p",)
    assert str(seen["name"]).startswith("vtech_q_")
    assert seen["cur"].itersize == 100_000
    assert seen["closed"] is True
    assert df["symbol"].tolist() == ["NQZ6"]
    assert df["bid_px"].iloc[0] == 1.5
